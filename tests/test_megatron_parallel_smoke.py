"""Two-rank CPU smoke test for the optional MCore parallel wrappers."""

import torch
import torch.distributed as dist


def main() -> None:
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group('gloo')
    from fluxvla.engines.runners.megatron_parallel import (ShardedOptimizer,
                                                            apply_tensor_parallel)

    device = (torch.device('cuda', dist.get_rank())
              if dist.get_backend() == 'nccl' else torch.device('cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4)).to(device)
    apply_tensor_parallel(model, dist.group.WORLD, minimum_out_features=0)
    assert tuple(model[0].weight.shape) == (2, 4)
    loss = model(torch.ones(2, 4, device=device)).sum()
    loss.backward()
    optimizer = ShardedOptimizer(torch.optim.AdamW(model.parameters(), lr=1e-3),
                                  dist.group.WORLD, bucket_cap_mb=1)
    assert optimizer.bucket_count == 2
    optimizer.step()
    optimizer.zero_grad()
    dist.barrier()
    print(f'rank={dist.get_rank()} mcore_parallel_smoke=passed', flush=True)
    if initialized_here:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
