# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VLM evaluation runner using VLMEvalKit.

Backend:
  VLMEvalKit (transformers). vLLM acceleration is planned later.

Install:
  git clone https://github.com/open-compass/VLMEvalKit
  cd VLMEvalKit && pip install -e .

Usage:
  python scripts/eval.py --config <config> --ckpt-path <path>
  --cfg-options eval.benchmarks=[gqa,realworldqa]
  --cfg-options eval.use_llm_judge=True

Add a benchmark:
  1. Add mapping in BENCHMARK_TO_VLMEVAL.
  2. Add config in DATASET_CONFIG with {class, dataset}.
  3. For video benchmarks, an empty config {} is allowed.

STVQA:
  If OpenXLab URL is unavailable, place TSV at ~/LMUData/STVQA.tsv
  or set eval.stvqa_dataset_url / STVQA_DATASET_URL.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.root import RUNNERS

# =============================================================================
# Monkey-patches for VLMEvalKit (custom checkpoints, verbose debug)
# =============================================================================


def _patch_batch_encoding_dtype():
    """Patch BatchEncoding.to(dtype) to cast float tensors only."""
    import torch
    from transformers.tokenization_utils_base import BatchEncoding

    _orig_to = BatchEncoding.to

    def _patched_to(self, device=None, dtype=None, *args, **kwargs):
        result = _orig_to(self, device=device, *args, **kwargs)
        if dtype is not None:
            for k in list(self.keys()):
                v = self.get(k)
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    self[k] = v.to(
                        device=v.device if device is None else device,
                        dtype=dtype)
        return result

    BatchEncoding.to = _patched_to


def _patch_vlmeval_processor_tokenizer():
    """Ensure processor.tokenizer exists for custom checkpoints."""
    import vlmeval.vlm.qwen2_vl.model as q2
    import vlmeval.vlm.qwen3_vl.model as q3

    def make_patched_init(orig_init):

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if not hasattr(self.processor, 'tokenizer'):
                self.processor.tokenizer = self.processor

        return patched_init

    q3.Qwen3VLChat.__init__ = make_patched_init(q3.Qwen3VLChat.__init__)
    q2.Qwen2VLChat.__init__ = make_patched_init(q2.Qwen2VLChat.__init__)


def _append_prompt_suffix(struct, suffix: str) -> None:
    """Append suffix to the last text element in a prompt struct."""
    if not suffix:
        return
    for i in range(len(struct) - 1, -1, -1):
        if isinstance(struct[i], dict) and struct[i].get('type') == 'text':
            struct[i]['value'] = struct[i]['value'] + suffix
            return


def _patch_infer_data_extras():
    """Apply prompt suffix and/or print GT/Pred when env flags are enabled."""
    prompt_suffix = os.environ.get('VLM_PROMPT_SUFFIX', '')
    verbose_debug = os.environ.get('VLM_VERBOSE_DEBUG') == '1'
    if not prompt_suffix and not verbose_debug:
        return

    from vlmeval import inference

    _orig = inference.infer_data

    def _patched(model,
                 model_name,
                 work_dir,
                 dataset,
                 out_file,
                 verbose=False,
                 api_nproc=4,
                 use_vllm=False,
                 ignore_failed=False):
        import torch
        from tqdm import tqdm
        from vlmeval.config import supported_VLM
        from vlmeval.inference import FAIL_MSG
        from vlmeval.smp import dump, get_rank_and_world_size, load, osp

        dataset_name = dataset.dataset_name
        prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
        res = load(prev_file) if osp.exists(prev_file) else {}
        if osp.exists(out_file):
            res.update(load(out_file))

        rank, world_size = get_rank_and_world_size()
        sheet_indices = list(range(rank, len(dataset), world_size))
        lt = len(sheet_indices)
        data = dataset.data.iloc[sheet_indices]
        data_indices = list(data['index'])

        all_finished = all(data.iloc[j]['index'] in res for j in range(lt))
        if all_finished:
            res = {k: res[k] for k in data_indices}
            dump(res, out_file)
            return model

        data = data[~data['index'].isin(res)]
        lt = len(data)

        kwargs = {
            'use_vllm': use_vllm
        } if model_name and any(
            x in model_name
            for x in ['Llama-4', 'Qwen2-VL', 'Qwen2.5-VL']) else {}
        ws_bak = os.environ.pop('WORLD_SIZE', None)
        model = supported_VLM[model_name](
            **kwargs) if isinstance(model, str) else model
        if ws_bak:
            os.environ['WORLD_SIZE'] = ws_bak

        if getattr(model, 'is_api', False):
            return _orig(
                model,
                model_name,
                work_dir,
                dataset,
                out_file,
                verbose=verbose,
                api_nproc=api_nproc,
                use_vllm=use_vllm)

        model.set_dump_image(dataset.dump_image)

        for i in tqdm(
                range(lt),
                desc=f'Infer {model_name}/{dataset_name}, '
                f'Rank {rank}/{world_size}'):
            idx = data.iloc[i]['index']
            if idx in res:
                continue

            row = data.iloc[i]
            if hasattr(dataset, 'force_use_dataset_prompt'
                       ) and dataset.force_use_dataset_prompt:
                struct = dataset.build_prompt(row)
            elif hasattr(model, 'use_custom_prompt'
                         ) and model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(row, dataset=dataset_name)
            else:
                struct = dataset.build_prompt(row)

            _append_prompt_suffix(struct, prompt_suffix)

            if os.environ.get('SKIP_ERR', False) == '1':
                try:
                    response = model.generate(
                        message=struct, dataset=dataset_name)
                except RuntimeError as err:
                    torch.cuda.synchronize()
                    import warnings
                    warnings.warn(f'{type(err)} {str(err)}')
                    response = f'{FAIL_MSG}: {type(err)} {str(err)}'
            else:
                response = model.generate(message=struct, dataset=dataset_name)
            torch.cuda.empty_cache()

            if verbose_debug:
                gt = row.get('answer', row.get('answer_key', 'N/A'))
                if isinstance(gt, str) and len(gt) > 200:
                    gt = gt[:200] + '...'
                print(f'[GT] {gt}', flush=True)
                print(f'[Pred] {response}', flush=True)
                print('-' * 60, flush=True)

            res[idx] = response
            if (i + 1) % 10 == 0:
                dump(res, out_file)

        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    inference.infer_data = _patched


def _patch_stvqa_dataset_url():
    """Override STVQA TSV download URL."""
    import vlmeval.dataset.image_vqa as iv
    url = os.environ.get('STVQA_DATASET_URL')
    if url:
        iv.ImageVQADataset.DATASET_URL['STVQA'] = url
    md5v = os.environ.get('STVQA_DATASET_MD5')
    if md5v:
        iv.ImageVQADataset.DATASET_MD5['STVQA'] = md5v


# =============================================================================
# Benchmark mapping: our name -> VLMEvalKit dataset name.
# It must match dataset.dataset_name, otherwise filenames mismatch.
# =============================================================================
BENCHMARK_TO_VLMEVAL: Dict[str, str] = {
    'gqa': 'GQA_TestDev_Balanced',
    'science_qa': 'ScienceQA_TEST',
    'textvqa': 'TextVQA_VAL',
    'vizwiz': 'VizWiz',
    'pope': 'POPE',
    'mme': 'MME',
    'mmbench_en': 'MMBench_DEV_EN_V11',
    'mmbench_cn': 'MMBench_DEV_CN_V11',
    'seed': 'SEEDBench_IMG',
    'mmmu': 'MMMU',
    'mathvista': 'MathVista',
    'ai2d': 'AI2D_TEST',
    'chartqa': 'ChartQA_TEST',
    'docvqa': 'DocVQA_VAL',
    'infovqa': 'InfoVQA_VAL',
    'stvqa': 'STVQA',
    'ocrbench': 'OCRBench',
    'mmstar': 'MMStar',
    'realworldqa': 'RealWorldQA',
    'coco': 'COCO',
    'mathverse': 'MathVerse',
    'mathvision': 'MathVision',
    'videomme': 'Video-MME',
    'mmvet': 'MMVet',
    'llava_in_the_wild': 'LLaVABench',
    'simplevqa': 'SimpleVQA',
    'mmvp': 'MMVP',
}

# =============================================================================
# VLMEvalKit --config data format: {class, dataset}.
# The key must equal dataset.dataset_name, otherwise file names mismatch.
# =============================================================================
DATASET_CONFIG: Dict[str, Dict[str, str]] = {
    'GQA_TestDev_Balanced': {
        'class': 'ImageVQADataset',
        'dataset': 'GQA_TestDev_Balanced'
    },
    'ScienceQA_TEST': {
        'class': 'ImageMCQDataset',
        'dataset': 'ScienceQA_TEST'
    },
    'TextVQA_VAL': {
        'class': 'ImageVQADataset',
        'dataset': 'TextVQA_VAL'
    },
    'VizWiz': {
        'class': 'ImageVQADataset',
        'dataset': 'VizWiz'
    },
    'POPE': {
        'class': 'ImageYORNDataset',
        'dataset': 'POPE'
    },
    'MME': {
        'class': 'MMERealWorld',
        'dataset': 'MME'
    },
    'MMBench_DEV_EN_V11': {
        'class': 'ImageMCQDataset',
        'dataset': 'MMBench_DEV_EN_V11'
    },
    'MMBench_DEV_CN_V11': {
        'class': 'ImageMCQDataset',
        'dataset': 'MMBench_DEV_CN_V11'
    },
    'SEEDBench_IMG': {
        'class': 'ImageMCQDataset',
        'dataset': 'SEEDBench_IMG'
    },
    'MMMU': {
        'class': 'MMMUDataset',
        'dataset': 'MMMU'
    },
    'MathVista': {
        'class': 'MathVista',
        'dataset': 'MathVista'
    },
    'AI2D_TEST': {
        'class': 'ImageMCQDataset',
        'dataset': 'AI2D_TEST'
    },
    'ChartQA_TEST': {
        'class': 'ImageVQADataset',
        'dataset': 'ChartQA_TEST'
    },
    'DocVQA_VAL': {
        'class': 'ImageVQADataset',
        'dataset': 'DocVQA_VAL'
    },
    'InfoVQA_VAL': {
        'class': 'ImageVQADataset',
        'dataset': 'InfoVQA_VAL'
    },
    'STVQA': {
        'class': 'ImageVQADataset',
        'dataset': 'STVQA'
    },
    'OCRBench': {
        'class': 'OCRBench',
        'dataset': 'OCRBench'
    },
    'MMStar': {
        'class': 'ImageMCQDataset',
        'dataset': 'MMStar'
    },
    'RealWorldQA': {
        'class': 'ImageMCQDataset',
        'dataset': 'RealWorldQA'
    },
    'COCO': {
        'class': 'ImageCaptionDataset',
        'dataset': 'COCO'
    },
    'MathVerse': {
        'class': 'MathVerse',
        'dataset': 'MathVerse'
    },
    'MathVision': {
        'class': 'MathVision',
        'dataset': 'MathVision'
    },
    'Video-MME': {},  # video: empty config uses supported_video_datasets
    'MMVet': {
        'class': 'MMVet',
        'dataset': 'MMVet'
    },
    'LLaVABench': {
        'class': 'LLaVABench',
        'dataset': 'LLaVABench'
    },
    'SimpleVQA': {
        'class': 'SimpleVQA',
        'dataset': 'SimpleVQA'
    },
    'MMVP': {
        'class': 'ImageMCQDataset',
        'dataset': 'MMVP'
    },
}


def _ensure_vlmevalkit_available() -> Path:
    """Locate VLMEvalKit root from install or VLMEVALKIT_ROOT."""
    root = os.environ.get('VLMEVALKIT_ROOT')
    if root:
        p = Path(root)
        if (p / 'run.py').exists():
            return p
        raise FileNotFoundError(
            f'VLMEVALKIT_ROOT={root} but run.py not found. '
            'Ensure VLMEvalKit is cloned and run.py exists at the root.')
    try:
        import vlmeval
        repo_root = Path(vlmeval.__file__).resolve().parent.parent
        if (repo_root / 'run.py').exists():
            return repo_root
    except ImportError:
        pass
    raise FileNotFoundError(
        'VLMEvalKit not found. Either:\n'
        '  1) git clone https://github.com/open-compass/VLMEvalKit\n'
        '     && cd VLMEvalKit && pip install -e .\n'
        '  2) Or set VLMEVALKIT_ROOT to the cloned repo path.\n'
        'See https://github.com/open-compass/VLMEvalKit')


def _run_vlmevalkit_inprocess(repo_root: Path, argv: List[str],
                              env: Dict[str, str]) -> None:
    """Run VLMEvalKit run.main() in-process with patches applied."""
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    cwd_bak = os.getcwd()
    path_bak = sys.path.copy()
    env_bak = {
        k: os.environ.get(k)
        for k in (
            'VLM_VERBOSE_DEBUG',
            'VLM_PROMPT_SUFFIX',
            'MMEVAL_ROOT',
            'VLMEVALKIT_ROOT',
            'STVQA_DATASET_URL',
            'STVQA_DATASET_MD5',
        )
    }
    try:
        os.environ.update(env)
        os.chdir(root_str)
        sys.argv = argv

        _patch_batch_encoding_dtype()
        _patch_vlmeval_processor_tokenizer()
        _patch_infer_data_extras()
        _patch_stvqa_dataset_url()

        from run import load_env
        from run import main as run_main
        load_env()
        run_main()
    finally:
        os.chdir(cwd_bak)
        sys.path[:] = path_bak
        for k, v in env_bak.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]


def _detect_model_class(ckpt_path: str) -> str:
    """Detect Qwen2-VL vs Qwen3-VL from HF config. Default Qwen3VLChat."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
        mt = (getattr(cfg, 'model_type', '') or '').lower()
        tmt = ''
        if hasattr(cfg, 'text_config') and cfg.text_config:
            tmt = (getattr(cfg.text_config, 'model_type', '') or '').lower()
        if 'qwen3' in mt or 'qwen3' in tmt or 'qwen3_vl' in mt:
            return 'Qwen3VLChat'
        return 'Qwen2VLChat'
    except Exception:
        return 'Qwen3VLChat'


def _model_name_from_ckpt(ckpt_path: str) -> str:
    """Derive unique model name from checkpoint path."""
    p = Path(ckpt_path.rstrip('/')).resolve()
    name = p.name or 'CustomVLM'
    # Sanitize for use in paths
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
    return safe or 'CustomVLM'


def _build_vlmeval_config(
    ckpt_path: str,
    benchmarks: List[str],
) -> Tuple[Dict, str]:
    """Build VLMEvalKit --config JSON content. Returns (config, model_name)."""
    model_name = _model_name_from_ckpt(ckpt_path)
    model_class = _detect_model_class(ckpt_path)
    data_config = {}
    for b in benchmarks:
        vlm_name = BENCHMARK_TO_VLMEVAL.get(b, b)
        cfg = DATASET_CONFIG.get(vlm_name, {})
        data_config[vlm_name] = cfg

    if not data_config:
        raise ValueError('No valid benchmarks. '
                         'Check BENCHMARK_TO_VLMEVAL and DATASET_CONFIG.')

    model_cfg = {
        'model_path': str(Path(ckpt_path).resolve()),
        # transformers backend only; vLLM support is planned.
        'use_vllm': False,
        'use_custom_prompt': False,
    }
    if model_class == 'Qwen2VLChat':
        model_cfg.update({
            'min_pixels': 1280 * 28 * 28,
            'max_pixels': 16384 * 28 * 28,
        })

    config = {
        'model': {
            model_name: {
                'class': model_class,
                **model_cfg
            }
        },
        'data': data_config,
    }
    return config, model_name


@RUNNERS.register_module()
class VLMEvalRunner:
    """VLM evaluation via VLMEvalKit (transformers backend).

    Args:
        use_llm_judge: Whether to enable LLM API judging.
            False (default): use exact_matching (no API required).
            True: use API-based judges such as ChatGPT.
        ckpt_path: Path to HF checkpoint.
        benchmarks: Benchmark names (see BENCHMARK_TO_VLMEVAL). None = all.
        work_dir: Output dir for VLMEvalKit (default: outputs/vlm_eval).
        reuse: Reuse existing predictions.
            Model name is derived from checkpoint path.
        mode: 'all' (infer+eval) or 'infer' only.
        verbose: Print model output and GT during inference for debugging.
        prompt_suffix: Text suffix appended to each question, e.g.
            " Please answer with only the option letter."
        stvqa_dataset_url: TSV download URL for STVQA.
            If OpenXLab STVQA.tsv is unavailable, set this field or
            place TSV at ~/LMUData/STVQA.tsv.
        stvqa_dataset_md5: Optional MD5 checksum for stvqa_dataset_url.
    """

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        use_llm_judge: bool = False,
        ckpt_path: Optional[str] = None,
        benchmarks: Optional[List[str]] = None,
        work_dir: Optional[str] = None,
        reuse: bool = True,
        mode: str = 'all',
        verbose: bool = False,
        prompt_suffix: Optional[str] = None,
        stvqa_dataset_url: Optional[str] = None,
        stvqa_dataset_md5: Optional[str] = None,
        **kwargs,
    ):
        self.cfg = cfg
        self.verbose = verbose
        self.prompt_suffix = prompt_suffix or ''
        self.stvqa_dataset_url = stvqa_dataset_url
        self.stvqa_dataset_md5 = stvqa_dataset_md5
        self.use_llm_judge = use_llm_judge
        self.ckpt_path = ckpt_path or (getattr(cfg, 'eval', None) and getattr(
            cfg.eval, 'ckpt_path', None))
        self.benchmarks = benchmarks if benchmarks is not None else list(
            BENCHMARK_TO_VLMEVAL.keys())
        self.work_dir = work_dir or 'outputs/vlm_eval'
        self.reuse = reuse
        self.mode = mode

    def run_setup(self):
        """No-op; VLMEvalKit loads model internally."""
        pass

    def run(self):
        """Run evaluation via VLMEvalKit with custom-checkpoint patches."""
        if not self.ckpt_path:
            raise ValueError('ckpt_path is required for VLM evaluation.')

        repo_root = _ensure_vlmevalkit_available()

        config, model_name = _build_vlmeval_config(
            self.ckpt_path,
            self.benchmarks,
        )

        # Use absolute paths to avoid path resolution issues after chdir.
        work_dir_abs = str(Path(self.work_dir).resolve())
        os.makedirs(work_dir_abs, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            config_path = f.name

        try:
            argv = [
                'run.py',
                '--config',
                config_path,
                '--work-dir',
                work_dir_abs,
                '--mode',
                self.mode,
            ]
            if self.reuse:
                argv.append('--reuse')
            if not self.use_llm_judge:
                argv.extend(['--judge', 'exact_matching'])
            if self.verbose:
                argv.append('--verbose')

            env = os.environ.copy()
            if self.verbose:
                env['VLM_VERBOSE_DEBUG'] = '1'  # Also print ground truth.
            if self.prompt_suffix:
                env['VLM_PROMPT_SUFFIX'] = self.prompt_suffix
            if self.stvqa_dataset_url:
                env['STVQA_DATASET_URL'] = self.stvqa_dataset_url
            if self.stvqa_dataset_md5:
                env['STVQA_DATASET_MD5'] = self.stvqa_dataset_md5
            if 'MMEVAL_ROOT' not in env:
                env['MMEVAL_ROOT'] = work_dir_abs
            if 'VLMEVALKIT_ROOT' not in env and repo_root:
                env['VLMEVALKIT_ROOT'] = str(repo_root)

            judge_mode = 'LLM API' if self.use_llm_judge else 'exact_matching'
            print(f'VLMEvalRunner: model={model_name}, judge={judge_mode}, '
                  f'benchmarks={self.benchmarks}')
            print(f'Running: {" ".join(argv)}')

            _run_vlmevalkit_inprocess(repo_root, argv, env)
        finally:
            os.unlink(config_path)
