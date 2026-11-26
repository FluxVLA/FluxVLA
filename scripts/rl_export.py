"""Export full RLinf actor weights for native FluxVLA evaluation."""

import argparse

from fluxvla.engines.utils.rl_export import export_native_bundle, export_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'checkpoint', help='global_step_N directory or full_weights.pt')
    parser.add_argument(
        'output', help='Output .pt/.pth file, or native bundle directory')
    parser.add_argument(
        '--native-bundle',
        action='store_true',
        help='Include safetensors, portable eval config, stats and tokenizer')
    parser.add_argument(
        '--rl-config', help='Resolved YAML config from the actual RL run')
    args = parser.parse_args()
    if args.native_bundle:
        if not args.rl_config:
            parser.error('--native-bundle requires --rl-config')
        from omegaconf import OmegaConf
        manifest = export_native_bundle(args.checkpoint, args.output,
                                        OmegaConf.load(args.rl_config))
        retained = manifest['policy_tensor_count']
        removed = manifest['removed_value_head_tensors']
        print(f'Exported native evaluation bundle to {args.output}; '
              f'retained {retained} policy tensors, '
              f'removed {removed} value_head tensors.')
    else:
        if args.rl_config:
            parser.error('--rl-config is used with --native-bundle')
        retained, removed = export_policy(args.checkpoint, args.output)
        print(f'Exported {retained} policy tensors to {args.output}; '
              f'removed {removed} value_head tensors. '
              'Native names are preserved.')


if __name__ == '__main__':
    main()
