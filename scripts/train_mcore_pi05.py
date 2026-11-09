"""Native Megatron-Core PI0.5 training entry for FluxVLA.

Run with Megatron-LM on ``PYTHONPATH``; no Megatron source is modified.  The
entrypoint uses Megatron's model provider, forward/backward schedule and
optimizer initialization, including its native distributed optimizer when
``--use-distributed-optimizer`` is supplied.
"""

from __future__ import annotations

from fluxvla.mcore.launcher import run_mcore
from fluxvla.mcore.pi05.recipe import PI05_SPEC


def main():
    run_mcore(PI05_SPEC)


if __name__ == "__main__":
    main()
