"""Native Megatron-Core LingBot-VLA-v2 training entry."""

from fluxvla.mcore.launcher import run_mcore
from fluxvla.mcore.lingbot_vla_v2.recipe import LINGBOT_VLA_V2_SPEC


def main() -> None:
    run_mcore(LINGBOT_VLA_V2_SPEC)


if __name__ == "__main__":
    main()
