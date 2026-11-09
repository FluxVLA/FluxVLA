# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TextVQA generation evaluation with official ten-answer scoring."""

import copy
import gc
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from fluxvla.datasets.utils.textvqa import (load_textvqa_validation,
                                            validate_textvqa_images)
from fluxvla.engines.utils import build_vla_from_cfg, initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.root import RUNNERS
from fluxvla.evaluators.textvqa import textvqa_score
from fluxvla.transforms.paligemma_vlm_transforms import \
    ProcessPaliGemmaVLMSFTInputs
from fluxvla.transforms.vlm_transforms import ProcessVLMSFTInputs

overwatch = initialize_overwatch(__name__)


def resolve_vlm_checkpoint(path, checkpoint_dir=None):
    if path == 'base':
        return None
    if path not in (None, 'latest'):
        result = Path(path).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(f'VLM checkpoint not found: {result}')
        return result
    if checkpoint_dir is None:
        raise ValueError('Specify --ckpt-path or eval.checkpoint_dir.')
    # Only prefer aliases published after checkpoint writing has finished.
    for name in ('latest-checkpoint.safetensors', 'latest-checkpoint.pt'):
        published = Path(checkpoint_dir) / name
        if published.is_symlink() and published.is_file():
            return published.resolve()
    candidates = []
    for candidate in Path(checkpoint_dir).glob('*'):
        match = re.match(r'step-(\d+)-', candidate.name)
        if (match and candidate.is_file()
                and candidate.suffix in ('.safetensors', '.pt', '.pth')):
            candidates.append(
                (int(match[1]), candidate.suffix == '.safetensors', candidate))
    if not candidates:
        raise FileNotFoundError(
            f'No saved VLM checkpoints in {checkpoint_dir}. Train until the '
            'first checkpoint, or use --ckpt-path base for the baseline.')
    return max(candidates)[-1].resolve()


@RUNNERS.register_module()
class TextVQAEvalRunner:
    """Generate from images/questions only, then score against annotations.

    Each rank owns an unsharded frozen model and disjoint question indices.
    No distributed forward collectives or padding duplicates are introduced;
    ranks can finish different numbers of questions/tokens safely.
    """

    def __init__(self,
                 cfg,
                 annotation_path,
                 image_root,
                 preprocessing,
                 ckpt_path=None,
                 checkpoint_dir=None,
                 compare_base=True,
                 run_dir='work_dirs/pi05_textvqa_eval',
                 prompt_suffix=('\nAnswer the question using '
                                'a single word or phrase.'),
                 max_samples=None,
                 max_new_tokens=32,
                 enable_mixed_precision_training=True,
                 mixed_precision_dtype='bf16',
                 keep_params_fp32=True,
                 device=None,
                 seed=42):
        self.cfg = cfg
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        preprocessing = dict(preprocessing)
        processor_type = preprocessing.pop('type',
                                           'ProcessPaliGemmaVLMSFTInputs')
        if processor_type == 'ProcessVLMSFTInputs':
            self.processor = ProcessVLMSFTInputs(**preprocessing)
        elif processor_type == 'ProcessPaliGemmaVLMSFTInputs':
            self.processor = ProcessPaliGemmaVLMSFTInputs(**preprocessing)
        else:
            raise ValueError(
                f'Unsupported TextVQA processor: {processor_type}')
        self.native_inputs = processor_type == 'ProcessVLMSFTInputs'
        self.checkpoint = resolve_vlm_checkpoint(ckpt_path, checkpoint_dir)
        self.compare_base = compare_base or self.checkpoint is None
        self.run_dir = Path(run_dir)
        if max_samples is not None and (type(max_samples) is not int
                                        or max_samples < 1):
            raise ValueError('max_samples must be a positive integer or None.')
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('max_new_tokens must be a positive integer.')
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.prompt_suffix = prompt_suffix
        self.autocast = enable_mixed_precision_training
        self.dtype = str_to_dtype(mixed_precision_dtype)
        self.keep_params_fp32 = keep_params_fp32
        self.device = torch.device(
            device or (f'cuda:{os.environ.get("LOCAL_RANK", 0)}'
                       if torch.cuda.is_available() else 'cpu'))
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.seed = seed
        self.vla = None

    def run_setup(self):
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        rows = load_textvqa_validation(self.annotation_path)
        self.dataset_size = len(rows)
        self.rows = rows[:self.max_samples]
        validate_textvqa_images(self.rows, self.image_root)
        self.model_cfg = copy.deepcopy(self.cfg.model)
        self.base_checkpoint = self.model_cfg.get('pretrained_name_or_path')
        backbone = self.model_cfg.get('vlm_backbone', {})
        if (not self.base_checkpoint
                and backbone.get('type') == 'CausalVLMBackbone'):
            # Native HF backbones load pretrained weights in their constructor.
            self.base_checkpoint = backbone.get('model_path')
        if self.model_cfg.get('use_lora', False):
            raise ValueError('TextVQA evaluation expects full checkpoints; '
                             'merge LoRA adapters first.')
        self.vla = build_vla_from_cfg(self.model_cfg)
        if not callable(getattr(self.vla, 'generate_text', None)):
            raise ValueError('TextVQA requires model.generate_text.')
        self.vla.validate_training_task('vlm')
        # Promote before loading: Pi05 creates some norm weights in BF16.
        if self.keep_params_fp32 or not self.autocast:
            self.vla.float()
        if self.compare_base:
            if not self.base_checkpoint:
                raise ValueError(
                    'Base comparison requires pretrained weights.')
            self.vla.from_pretrained()
        if not self.keep_params_fp32 and self.autocast:
            self.vla.to(dtype=self.dtype)
        self.vla.requires_grad_(False).eval().to(self.device)

    def _eos_token_id(self):
        if self.native_inputs:
            return self.processor.eos_token_id
        return self.processor._get_tokenizer().eos_id()

    def _preprocessing_metadata(self):
        if self.native_inputs:
            return self.processor.generation_metadata()
        return dict(
            image_size=self.processor.image_size,
            tokenizer_sha256=hashlib.sha256(
                Path(self.processor.tokenizer_path).read_bytes()).hexdigest())

    def _load_checkpoint(self):
        if self.checkpoint.suffix == '.safetensors':
            state = load_file(str(self.checkpoint), device='cpu')
        else:
            state = torch.load(
                self.checkpoint,
                map_location='cpu',
                weights_only=True,
                mmap=True)
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            # FSDP change_key_name=True groups weights by top-level module.
            flat_state = {}
            for key, value in state.items():
                if isinstance(value, dict):
                    for subkey, tensor in value.items():
                        flat_state[f'{key}.{subkey}'] = tensor
                else:
                    flat_state[key] = value
            state = flat_state
        # Saved FluxVLA keys already match: do not apply OpenPI name mapping.
        self.vla.load_state_dict(state, strict=True)

    @torch.inference_mode()
    def _evaluate(self, name):
        predictions = []
        eos_token_id = self._eos_token_id()
        for completed, index in enumerate(
                range(self.rank, len(self.rows), self.world_size), start=1):
            row = self.rows[index]
            prompt = row['question'] + self.prompt_suffix
            inputs = self.processor.encode_prompt(
                prompt, [self.image_root / f'{row["image_id"]}.jpg'])
            inputs = {
                k:
                (v if self.native_inputs else v.unsqueeze(0)).to(self.device)
                for k, v in inputs.items()
            }
            with torch.autocast(
                    self.device.type, dtype=self.dtype, enabled=self.autocast):
                sequences = self.vla.generate_text(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=eos_token_id,
                    do_sample=False)
            new_ids = sequences[0, inputs['input_ids'].shape[1]:].tolist()
            prediction = self.processor.decode_answer(new_ids)
            predictions.append(
                dict(
                    question_id=row['question_id'],
                    image_id=row['image_id'],
                    question=row['question'],
                    prompt=prompt,
                    prediction=prediction,
                    answers=row['answers'],
                    score=textvqa_score(prediction, row['answers']),
                    generated_tokens=len(new_ids),
                    reached_token_limit=(len(new_ids) == self.max_new_tokens
                                         and new_ids[-1] != eos_token_id)))
            if completed % 50 == 0:
                overwatch.info(f'TextVQA {name}: rank {self.rank} completed '
                               f'{completed} questions.')
        if dist.is_initialized():
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, predictions)
            predictions = [row for part in gathered for row in part]
        by_id = {row['question_id']: row for row in predictions}
        if (len(predictions) != len(self.rows) or len(by_id) != len(self.rows)
                or set(by_id) != {row['question_id']
                                  for row in self.rows}):
            raise RuntimeError('Missing or duplicate TextVQA predictions.')
        predictions = [by_id[row['question_id']] for row in self.rows]
        accuracy = sum(row['score'] for row in predictions) / len(predictions)
        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            path = self.run_dir / f'{name}_predictions.jsonl'
            temporary = path.with_suffix('.jsonl.tmp')
            temporary.write_text(''.join(
                json.dumps(row, ensure_ascii=False) + '\n'
                for row in predictions))
            temporary.replace(path)
        return dict(
            accuracy=accuracy,
            accuracy_percent=100 * accuracy,
            num_questions=len(predictions),
            token_limit_count=sum(row['reached_token_limit']
                                  for row in predictions))

    def run(self):
        report = dict(
            task='textvqa_generation',
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint=str(self.checkpoint) if self.checkpoint else 'base',
            base_checkpoint=self.base_checkpoint,
            dataset=dict(
                name='TextVQA',
                split='val',
                version='0.5.1',
                annotation_path=str(self.annotation_path.resolve()),
                annotation_sha256=hashlib.sha256(
                    self.annotation_path.read_bytes()).hexdigest(),
                image_root=str(self.image_root),
                total_questions=self.dataset_size,
                evaluated_questions=len(self.rows),
                selection='first_n_in_official_order'),
            protocol=dict(
                inputs='image_and_question',
                external_ocr=False,
                prompt_suffix=self.prompt_suffix,
                decoding='greedy',
                max_new_tokens=self.max_new_tokens,
                scoring='EvalAI normalization; leave-one-out VQA accuracy',
                **self._preprocessing_metadata()),
            precision=dict(
                autocast=self.autocast,
                dtype=str(self.dtype),
                keep_params_fp32=self.keep_params_fp32),
            world_size=self.world_size,
            metrics={})
        if self.compare_base:
            overwatch.info('Generating TextVQA answers with the base model.')
            report['metrics']['base'] = self._evaluate('base')
        if self.checkpoint is not None:
            self._load_checkpoint()
            overwatch.info(f'Generating TextVQA answers: {self.checkpoint}')
            report['metrics']['checkpoint'] = self._evaluate('checkpoint')
            if self.compare_base:
                report['delta_accuracy_percentage_points'] = (
                    report['metrics']['checkpoint']['accuracy_percent'] -
                    report['metrics']['base']['accuracy_percent'])
        if self.rank == 0:
            self.cfg.dump(str(self.run_dir / 'config.py'))
            temporary = self.run_dir / 'summary.json.tmp'
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(self.run_dir / 'summary.json')
            self._write_markdown(report)
            overwatch.info(f'TextVQA report: {self.run_dir / "summary.md"}')
        if dist.is_initialized():
            dist.barrier()
        return report

    def _write_markdown(self, report):
        lines = [
            '# TextVQA validation', '',
            f'Checkpoint: `{report["checkpoint"]}`', '',
            f'Questions: {len(self.rows)} / {self.dataset_size}. '
            'Image + question; no external OCR; greedy decoding.', '',
            '| Model | TextVQA accuracy | Token limit reached |',
            '| --- | ---: | ---: |'
        ]
        for name, metrics in report['metrics'].items():
            lines.append(f'| {name} | {metrics["accuracy_percent"]:.2f}% | '
                         f'{metrics["token_limit_count"]} |')
        if 'delta_accuracy_percentage_points' in report:
            delta = report['delta_accuracy_percentage_points']
            lines += ['', f'Checkpoint minus base: {delta:+.3f} pp.']
        if len(self.rows) != self.dataset_size:
            lines += ['', 'Subset smoke check; not a full validation score.']
        lines += [
            '', 'Scores use ten human answers per question and EvalAI '
            'normalization. This is not the OCR-assisted LLaVA protocol.', ''
        ]
        (self.run_dir / 'summary.md').write_text('\n'.join(lines))

    def cleanup(self):
        self.vla = None
        self.rows = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
