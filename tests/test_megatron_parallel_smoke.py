"""Two-rank CPU smoke test for the optional MCore parallel wrappers."""

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group('gloo')
    from fluxvla.engines.runners.megatron_parallel import (ShardedOptimizer,
                                                            apply_tensor_parallel)

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    apply_tensor_parallel(model, dist.group.WORLD, minimum_out_features=0)
    assert tuple(model[0].weight.shape) == (2, 4)
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer = ShardedOptimizer(torch.optim.AdamW(model.parameters(), lr=1e-3),
                                  dist.group.WORLD)
    optimizer.step()
    optimizer.zero_grad()
    dist.barrier()
    print(f'rank={dist.get_rank()} mcore_parallel_smoke=passed', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
