"""Export full RLinf actor weights for native FluxVLA evaluation."""

import argparse
from pathlib import Path

import torch


def export_policy(checkpoint, output):
    """Remove the RL critic without renaming or casting policy tensors.

    Accept a global_step_N directory or its actor/model_state_dict/
    full_weights.pt file. Only consolidated weights can be exported.
    """
    source = Path(checkpoint).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if source.is_dir():
        source = source / 'actor/model_state_dict/full_weights.pt'
    if not source.is_file():
        raise FileNotFoundError(f'Full actor checkpoint not found: {source}')
    if destination.suffix not in ('.pt', '.pth'):
        raise ValueError('Export path must end in .pt or .pth')
    if destination.exists():
        raise FileExistsError(f'Export already exists: {destination}')

    state = torch.load(source, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        state = state.get('model', state.get('state_dict', state))
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a non-empty full actor state dict')
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
           for key, value in state.items()):
        raise ValueError('Full actor state dict must map names to tensors')
    if any('_flat_param' in key.split('.') for key in state):
        raise ValueError('Export requires consolidated full_weights.pt')
    policy = {
        key: value
        for key, value in state.items() if not key.startswith('value_head.')
    }
    if not policy:
        raise ValueError('Checkpoint contains no policy weights')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(policy, stream)
    return len(policy), len(state) - len(policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'checkpoint', help='global_step_N directory or full_weights.pt')
    parser.add_argument('output', help='Output policy .pt/.pth file')
    args = parser.parse_args()
    retained, removed = export_policy(args.checkpoint, args.output)
    print(f'Exported {retained} policy tensors to {args.output}; '
          f'removed {removed} value_head tensors. Native names are preserved.')


if __name__ == '__main__':
    main()
