# Migrating hosted VLA configs and Python calls

Hosted models now use a provider entry point plus an explicit robot policy
selection. This is a breaking naming and construction change: saved configs
and external scripts must be migrated along with the code. No aliases for
the removed public classes are registered.

## Model construction

| Removed `type`       | New `type`   | Required `embodiment` |
| -------------------- | ------------ | --------------------- |
| `OpenAILiberoVLA`    | `OpenAIVLA`  | `libero`              |
| `OpenAIRobocasaVLA`  | `OpenAIVLA`  | `robocasa`            |
| `OpenAIOliVLA`       | `OpenAIVLA`  | `oli`                 |
| `MiniMaxLiberoVLA`   | `MiniMaxVLA` | `libero`              |
| `MiniMaxRobocasaVLA` | `MiniMaxVLA` | `robocasa`            |

For example, migrate this saved config:

```python
inference_model = dict(type='OpenAILiberoVLA', model='gpt-6-astra')
```

to:

```python
inference_model = dict(
    type='OpenAIVLA', embodiment='libero', model='gpt-6-astra')
```

Keep the remaining provider and motion settings. Selecting a different
embodiment changes the observation and action contract; it is not an alias
for the old scene. `MiniMaxVLA` currently supports `libero` and `robocasa`.
For the tested MiniMax Responses route with OLI, see
[the OLI instructions](openai_oli_control.md#switching-to-minimax).

Direct Python callers must also update imports and pass provider options by
keyword. A positional model name would now be interpreted as `embodiment`.
For example:

```python
from fluxvla.models.vlas.openai_vla import OpenAIVLA

vla = OpenAIVLA(
    embodiment='oli',
    model='gpt-6-astra',
    base_url='https://api.openai.com/v1',
    api_key_env='OPENAI_API_KEY',
)
actions = vla.predict_action(**batch)
```

The provider wrapper accepts keyword batches. Convert positional
`predict_action` calls to named fields matching the selected policy.
Existing runners already pass keyword batches. The public wrapper is an
`nn.Module` containing the selected `_policy`; code that inspects class
inheritance, class-level scene constants or child modules must account for
this composition. Prefer instance diagnostics and the public
`predict_action` method over assumptions about the old concrete class.

Tests replacing private transport or interpolation methods should patch
`vla._policy` directly. The wrapper forwards attribute access, but its own
`__dict__` is distinct from the policy's, which matters to tools that inspect
and restore instance attributes.

## Dataset construction and imports

| Removed dataset `type`      | New dataset `type`           |
| --------------------------- | ---------------------------- |
| `OpenAILiberoEvalDataset`   | `ToolCallingLiberoDataset`   |
| `OpenAIRobocasaEvalDataset` | `ToolCallingRobocasaDataset` |
| `OpenAIOliInferenceDataset` | `ToolCallingOliDataset`      |

Update both config registration names and direct Python imports. The module
`fluxvla.datasets.openai_eval_dataset` has been replaced by
`fluxvla.datasets.tool_calling_dataset`:

```python
from fluxvla.datasets.tool_calling_dataset import ToolCallingOliDataset

dataset = ToolCallingOliDataset(image_names=['head', 'left_wrist', 'right_wrist'])
batch = dataset(observation)
```

The internal shared policies were renamed from `ToolCallingVLA`,
`ToolCallingRobocasaVLA` and `ToolCallingOliVLA` to `LiberoToolPolicy`,
`RobocasaToolPolicy` and `OliToolPolicy`. These are internal action adapters,
not replacements to put in a public model config's `type` field. Import
constants explicitly if needed; provider modules now export only their
provider VLA through `__all__`.

## Compatibility boundaries

With matching embodiment and keyword settings, the refactor retains the
policy's observation processing, tool schema, motion conversion and action
layout. The HTTP `User-Agent` now identifies the provider entry point.
Old class identity, import paths, positional calls and error text referring
to the concrete class are not preserved. Calling a model's `forward` is
still unsupported; hosted models are inference-only.

Migration does not repair malformed model targets. In particular, OLI
orientation fields must be supplied in pairs or both omitted; an incomplete
pair is rejected before an action chunk is returned, as before the refactor.
API transport compatibility and numerical comparisons do not substitute for
simulator or robot validation.

Update active configs, scripts and tests. Retain old names in migration
examples, immutable experiment records and tests explicitly checking that
removed entries are rejected. Copies outside this repository require their
owners to apply the same migration. The earlier removed `OpenAIResponses*`
names are not restored by this change.
