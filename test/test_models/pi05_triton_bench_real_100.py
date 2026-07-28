import statistics
import time

import torch
import yaml
from mmengine import Config
from safetensors.torch import load_file

from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere
from scripts.test_pi05_dummy_forward import _make_dummy_batch

CFG_PATH = '/mnt/nvme/sober/checkpoints/pi05_aloha_fold_towel/config.yaml'
CKPT_PATH = '/mnt/nvme/sober/checkpoints/pi05_aloha_fold_towel/checkpoints/step-021062-epoch-01-loss=0.0205.safetensors'

set_seed_everywhere(0)
device = torch.device('cuda')

raw = yaml.safe_load(open(CFG_PATH, 'r'))
model_cfg = raw['inference_model']
model_cfg = dict(model_cfg)
model_cfg['type'] = 'PI05FlowMatchingInference'
model_cfg['num_views'] = len(raw['inference']['dataset']['img_keys'])
model_cfg['llm_backbone'] = dict(model_cfg['llm_backbone'])
model_cfg['llm_backbone']['type'] = 'ConditionGemmaInferenceModel'
model_cfg['llm_expert'] = dict(model_cfg['llm_expert'])
model_cfg['llm_expert']['type'] = 'ConditionGemmaInferenceModel'
model_cfg['vision_backbone'] = dict(model_cfg['vision_backbone'])
model_cfg['vision_backbone']['type'] = 'SigLIPViTBackboneInference'
model_cfg['projector'] = dict(model_cfg['projector'])
model_cfg['projector']['type'] = 'LinearProjectorInference'
for k in ['action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out']:
    model_cfg[k] = dict(model_cfg[k])
    model_cfg[k]['type'] = 'LinearProjectorInference'
model_cfg['pretrained_name_or_path'] = None
model_cfg['name_mapping'] = None

print('Building PI05FlowMatchingInference with real checkpoint...')
vla = build_vla_from_cfg(Config(dict(model=model_cfg)).model).to(device)
vla.eval()
print('CUDA:', torch.cuda.get_device_name(0))

print('Loading weights:', CKPT_PATH)
state = load_file(CKPT_PATH, device='cpu')
missing, unexpected = vla.load_state_dict(state, strict=False)
print(
    f'load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}'
)
if missing[:5]:
    print('sample_missing:', missing[:5])
if unexpected[:5]:
    print('sample_unexpected:', unexpected[:5])

vocab_size = int(getattr(vla.llm_backbone.config, 'vocab_size', 257152))
batch = _make_dummy_batch(
    device=device,
    batch_size=1,
    num_views=int(vla.num_views),
    image_size=224,
    lang_len=min(32, int(vla.triton_max_prompt_len)),
    max_action_dim=int(vla.max_action_dim),
    n_action_steps=int(vla.n_action_steps),
    vocab_size=vocab_size,
)


def one():
    return vla.predict_action(
        images=batch['images'],
        lang_tokens=batch['lang_tokens'],
        states=batch['states'],
        img_masks=batch['img_masks'],
        lang_masks=batch['lang_masks'],
        noise=batch['noise'],
    )


with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = one()
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1000.0
        print(f'cold_start_ms: {cold_ms:.3f}')
        print('cold_output_shape:', tuple(out.shape))

        warmup = 3
        for _ in range(warmup):
            _ = one()
        torch.cuda.synchronize()

        times = []
        runs = 100
        for _ in range(runs):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = one()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t1) * 1000.0)

print('timed_runs:', runs)
print('predict_action output shape:', tuple(out.shape))
print(
    f'latency_ms: min={min(times):.3f} max={max(times):.3f} mean={statistics.mean(times):.3f} median={statistics.median(times):.3f}'
)
print(f'total_wall_ms: {sum(times):.3f}')
if len(times) > 1:
    print(f'stdev_ms: {statistics.stdev(times):.3f}')
print('first10_ms:', [round(x, 3) for x in times[:10]])
print('OK')
