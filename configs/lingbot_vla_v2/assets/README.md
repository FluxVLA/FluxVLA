# RoboTwin normalization statistics

`robotwin_norm_stats.json` is a configuration asset, not a universal default
for every RoboTwin dataset or robot embodiment. It contains separate arm and
effector q01/q99 values for observation state and actions.

The file was relocated without modification from
`fluxvla/datasets/lingbot_vla_v2/assets/robotwin_norm_stats.json` in the pre-review
code snapshot `7c2e8cd`. Its SHA-256 is:

```text
5fc926443cebbd6484194982a768c4bd07772aa176dfbc64c6c6b59784855d76
```

The recorded producer count is 6,062,592. These statistics were not recomputed
from the smaller online validation dataset. The original producer dataset
revision is not recorded in this asset and still needs upstream provenance;
do not infer that the count identifies a particular dataset or checkpoint.

The current LingBot input contract normalizes in FP32 using
`(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`, without clipping, before casting model
inputs to the configured dtype. Keep the statistics identical to the reference
training run. Override `LINGBOT_NORM_STATS` for a different verified dataset
contract rather than silently substituting or recomputing these values.
