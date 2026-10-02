# Feishu Evaluation Reporting

FluxVLA can upload RoboTwin, RoboCasa, and LIBERO evaluation summaries to a
Feishu spreadsheet. The upload is best-effort: evaluation results are still
written locally even when Feishu reporting is not configured or the Feishu API
rejects the write.

This feature works with:

- `scripts/eval.py`
- `scripts/eval.sh`
- `scripts/train.py --eval-after-train`
- `scripts/train.sh ... --eval-after-train`
- `tools/summarize_robotwin_eval_results.py`
- `tools/summarize_robocasa_eval_results.py`
- `tools/summarize_libero_eval_results.py`
- `scripts/ros_inference_server.sh` with FluxThemis `ReportEvaluation`

## FluxThemis ROS Evaluation Reporting

An integrated ROS evaluation can make FluxVLA the authoritative result owner
without changing the usual two-terminal launch commands. Configure the shared
target model file as follows:

```python
themis = dict(
    transport=dict(
        service_name='/fluxvla/predict_action',
        report_service_name='/fluxvla/report_evaluation',
        # observation and transport fields ...
    ),
    ros_server=dict(
        evaluation_reporting=dict(
            # Optional; defaults to the directory containing checkpoints/.
            result_output_dir=None,
            # Optional direct overrides; otherwise use the environment below.
            feishu=dict(),
        ),
        # inference fields ...
    ),
    # runner ...
)
```

`report_service_name` enables the acknowledged lifecycle channel. Omitting it
keeps the PredictAction-only server compatible with older/local clients.
`result_output_dir` defaults to the directory containing the checkpoint's
`checkpoints/` directory. Relative overrides resolve from that same directory;
absolute paths are also supported.

For every accepted run, the server writes FluxVLA's native layout:

```text
<result_output_dir>/eval_runs/<checkpoint_stem>/
  EVAL-<suite>-<model_family>-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

The directory contains `events.jsonl`, `rank0.txt`, and
`rank_progress/rank0.json` during rollout. At `run_end`, it also contains the
suite's `<suite>/task<task_index>_results.json` and
`task_status/<suite>_task<task_index>.status` files plus `failed_tasks.txt`,
`summary.txt`, `summary.csv`, `summary.json`, and `task_success_rates.csv`.
`[ros-eval]` and `[eval-progress]` messages, the final directory, and Feishu
report/skip reasons appear in the FluxVLA server terminal.

The `evaluation_reporting.feishu` mapping accepts `sheet_url`, `app_id`,
`app_secret`, and `timeout`. Missing values use `FEISHU_SHEET_URL`,
`FEISHU_APP_ID`, and `FEISHU_APP_SECRET` from the server process. `timeout`
defaults to 10 seconds for this ROS path. Prefer the environment variables for
credentials so secrets stay out of configuration snapshots and event journals.

ROS-triggered upload is intentionally full-suite-only. The terminal `run_end`
must be `finished`, no task filter may be present, the numeric task set must
match the authoritative suite manifest, and every task must have exactly the
configured number of completed trials. Smoke, filtered, partial, interrupted,
and failed runs keep their native files but do not append a spreadsheet row.

## What Gets Written

For a summary that passes the caller's reporting checks, the reporter writes
one row. It does not deduplicate rows; uploading the same summary again appends
another row.

RoboTwin uses this header:

```text
id, commit id, config, ckpt_path, Easy, Hard, all
```

RoboCasa uses this header:

```text
id, commit id, config, ckpt_path, Cabinet, Drawer, Microwave, Generalization, all
```

LIBERO uses this header:

```text
id, commit id, config, ckpt_path, libero_10, libero_goal, libero_object, libero_spatial, all
```

Column behavior:

- `id` is an auto-incrementing row id.
- `commit id` is `git rev-parse HEAD` from the current repo.
- `config` is the config path passed to evaluation when available.
- `ckpt_path` is the checkpoint path saved in the evaluation summary.
- Suite/group columns are formatted as percentages, for example `50.00%`.
- `all` is computed from total successes and total trials when those counts
  are present.

If the target worksheet is empty, FluxVLA writes the header and the first
result row. If the worksheet is non-empty, the first row must exactly match the
expected header. A header mismatch is treated as unsafe, and the worksheet is
left unchanged. If you are migrating an older sheet, insert `ckpt_path`
between `config` and the first result column, or use a new empty worksheet.

Rows are appended after the last non-empty row. This avoids writing below a
large block of empty rows if the Feishu API returns padded blank rows.

## Feishu Setup

Create or reuse a Feishu custom app, then configure both API permissions and
document permissions:

1. In the Feishu Open Platform app page, copy the app credentials from
   `Credentials and Basic Info`.
2. Enable spreadsheet read/write permissions for the app. In Feishu this is
   the electronic spreadsheet permission such as viewing, commenting, editing,
   and managing spreadsheets.
3. Publish the app version again after changing permissions. If your tenant
   requires approval, wait until the new permissions are approved.
4. Open the target spreadsheet and add the app as a document app or
   collaborator with edit permission.
5. Make sure the app and the spreadsheet are in the same Feishu tenant.

If the API returns `91403 Forbidden`, the app credentials are usually valid but
the app identity cannot read or edit that specific spreadsheet. Re-check steps
2-5 above.

## Environment Variables

Set these in the same shell that launches training or evaluation:

```bash
export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<spreadsheet_token>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'
export FEISHU_TIMEOUT=10
```

`FEISHU_TIMEOUT` is optional and defaults to `10` seconds.

Do not escape `?` or `=` inside quoted URLs. This is correct:

```bash
export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>?sheet=<sheet_id>'
```

This is not recommended:

```bash
export FEISHU_SHEET_URL="https://example.feishu.cn/sheets/<token>\?sheet\=<sheet_id>"
```

The reporter normalizes this common shell-escaped form, but plain quoted URLs
are clearer and less error-prone.

## Choosing The Target Worksheet

`FEISHU_SHEET_URL` controls which worksheet is written.

If the URL includes `?sheet=<sheet_id>`, FluxVLA writes exactly that worksheet:

```bash
export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>?sheet=<sheet_id>'
```

This is useful when you want to write to an existing tab such as `Sheet4` even
if its title is not `robotwin`, `robocasa`, or `libero`.

If the URL does not include `sheet=`, FluxVLA chooses a worksheet by report
kind:

- RoboTwin writes to a worksheet named `robotwin`.
- RoboCasa writes to a worksheet named `robocasa`.
- LIBERO writes to a worksheet named `libero`.
- If the worksheet does not exist, FluxVLA creates it.

Use the spreadsheet-level URL when you want one document with separate
`robotwin`, `robocasa`, and `libero` tabs:

```bash
export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>'
```

Use a sheet-specific URL when you want to force the write into the currently
selected worksheet:

```bash
export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>?sheet=<sheet_id>'
```

Successful logs include the actual target:

```text
[feishu] report enabled: kind=robocasa, url_sheet_id=MVLmbP
[feishu] wrote result row: Sheet4 (selection=url sheet, sheet_id=MVLmbP, url=https://...)
```

`selection=url sheet` means the `sheet=` query selected the worksheet.
`selection=report kind sheet` means FluxVLA selected `robotwin`, `robocasa`, or
`libero` from the report kind.

## RoboTwin With `scripts/eval_robotwin_manager.sh`

When the manager evaluates both `clean` (Easy) and `random` (Hard), its summary
tool uploads **one row containing both Easy and Hard**. The top-level
`summary.json` contains both difficulties in `group_stats`; the per-condition
summaries remain in `clean/` and `random/` for inspection, but are not uploaded
separately. The Feishu `all` column uses total successes divided by total
trials across both difficulties, not the mean of their percentages. A
single-condition run still fills only its own difficulty column.

To report a completed manager run without rerunning simulation, use the
directory that contains both `clean/` and `random/`:

```bash
export CONFIG='configs/dit4dit/dit4dit_robotwin_all_full_finetune.py'
RUN_DIR='/path/to/existing/robotwin_manager_run'
python tools/summarize_robotwin_eval_results.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR"
```

This reads the existing worker results and uses the `FEISHU_*` environment
variables above. The checkpoint is read from the worker summaries unless
`--ckpt` is supplied. Re-uploading appends a new combined row; it does not
merge or remove rows previously uploaded separately. Partial runs keep missing
difficulty rates blank and return a non-zero exit status for retry.

## RoboTwin With `scripts/eval.sh`

Example:

```bash
cd /path/to/FluxVLA

export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/eval.sh \
  configs/gr00tn15/gr00tn15_eagle_3b_robotwin_all_data_full_finetune.py \
  work_dirs/gr00tn15_eagle_3b_robotwin_all_data_full_finetune/checkpoints/checkpoints/step-142386-epoch-03-loss=0.0027.safetensors \
  --cfg-options \
    eval.task_suite_name=random
```

RoboTwin uses `task_suite_name=clean` (Easy) or `task_suite_name=random`
(Hard). To evaluate both, pass `--cfg-options eval.runner.task_suite_name=[clean,random]`
for namespaced configs, or `eval.task_suite_name=[clean,random]` for flat configs.

After multiple suites finish, `scripts/eval.py` writes
`eval_runs/<checkpoint_stem>/robotwin_eval_summary_<timestamp>.json` beneath the
evaluation output root, including when `feishu_report=False`. If reporting is
enabled, it uploads one row per suite: `clean` fills `Easy`, and `random` fills
`Hard`.

## RoboCasa With `scripts/eval.sh`

Example:

```bash
cd /path/to/FluxVLA

export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts/eval.sh \
  configs/gr00tn15/gr00tn15_eagle_3b_robocasa_30_eps_full_finetune.py \
  work_dirs/gr00tn15_eagle_3b_robocasa_30_eps_full_finetune/checkpoints/latest-checkpoint.safetensors \
  --cfg-options \
    eval.num_trials_per_task=1 \
    eval.max_episode_steps=5
```

For full RoboCasa evaluation, remove the smoke-test overrides or set
`eval.num_trials_per_task` and `eval.max_episode_steps` to the values you need.

## LIBERO With `scripts/eval.sh`

Example:

```bash
cd /path/to/FluxVLA

export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/eval.sh \
  configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors \
  --cfg-options \
    eval.num_trials_per_task=1 \
    eval.max_steps=5 \
    eval.save_rollout_videos=False \
    eval.save_failed_rollout_videos=False
```

After LIBERO finishes multiple suites, `scripts/eval.py` writes
`eval_runs/<checkpoint_stem>/libero_eval_summary_<timestamp>.json` beneath the
evaluation output root, including when `feishu_report=False`. If reporting is
enabled and `task_ids` is unset, it uploads one row containing
`libero_10`, `libero_goal`, `libero_object`, `libero_spatial`, and `all`.

## Eval After Train

The same environment variables work with `--eval-after-train`. The training
process relaunches evaluation in a fresh process after saving the checkpoint,
and the Feishu variables are inherited by that evaluation process.

RoboTwin smoke example:

```bash
cd /path/to/FluxVLA

export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
WANDB_MODE=disabled \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts/train.sh \
  configs/gr00tn15/gr00tn15_eagle_3b_robotwin_all_data_full_finetune.py \
  work_dirs/gr00tn15_eagle_3b_robotwin_all_data_full_finetune \
  --eval-after-train \
  --cfg-options \
    train_dataloader.per_device_batch_size=1 \
    runner.max_epochs=None \
    runner.max_steps=1 \
    runner.save_iter_interval=1 \
    runner.save_epoch_interval=999 \
    runner.max_keep_ckpts=1 \
    eval.task_suite_name=clean \
    eval.num_trials_per_task=1 \
    eval.max_episode_steps=5 \
    eval.save_video=False
```

RoboCasa smoke example:

```bash
cd /path/to/FluxVLA

export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>?sheet=<sheet_id>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
WANDB_MODE=disabled \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts/train.sh \
  configs/gr00tn15/gr00tn15_eagle_3b_robocasa_30_eps_full_finetune.py \
  work_dirs/gr00tn15_eagle_3b_robocasa_30_eps_full_finetune \
  --eval-after-train \
  --cfg-options \
    train_dataloader.per_device_batch_size=1 \
    runner.max_epochs=None \
    runner.max_steps=1 \
    runner.save_iter_interval=1 \
    runner.save_epoch_interval=999 \
    runner.max_keep_ckpts=1 \
    eval.num_trials_per_task=1 \
    eval.max_episode_steps=5
```

LIBERO smoke example:

```bash
cd /path/to/FluxVLA

export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
WANDB_MODE=disabled \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts/train.sh \
  configs/gr00tn15/gr00tn15_eagle_3b_libero_10_full_finetune.py \
  work_dirs/gr00tn15_eagle_3b_libero_10_full_finetune \
  --eval-after-train \
  --cfg-options \
    train_dataloader.per_device_batch_size=1 \
    runner.max_epochs=None \
    runner.max_steps=1 \
    runner.save_iter_interval=1 \
    runner.save_epoch_interval=999 \
    runner.max_keep_ckpts=1 \
    eval.num_trials_per_task=1 \
    eval.max_steps=5 \
    eval.save_rollout_videos=False \
    eval.save_failed_rollout_videos=False
```

## Upload An Existing Summary

You can upload existing RoboTwin summaries without rerunning evaluation.
Before uploading, keep only the latest summary for each suite in the search
directory. If both `clean` and `random` are present, both are uploaded.

```bash
cd /path/to/FluxVLA

SUMMARY=$(find work_dirs/gr00tn15_eagle_3b_robotwin_all_data_full_finetune \
  -path '*/EVAL-robotwin-groot-*/summary.json' \
  -printf '%T@ %p\n' | sort -nr | head -2 | cut -d' ' -f2-)

python - <<PY
from fluxvla.engines.utils.feishu_reporter import maybe_report_summary_to_feishu

for summary_path in """$SUMMARY""".splitlines():
    result = maybe_report_summary_to_feishu(
        summary_path,
        "robotwin",
        config="configs/gr00tn15/gr00tn15_eagle_3b_robotwin_all_data_full_finetune.py",
        logger=print,
        log_unconfigured=True,
    )
    print(result)
PY
```

You can upload an existing RoboCasa summary without rerunning evaluation:

```bash
cd /path/to/FluxVLA

SUMMARY=$(find work_dirs/gr00tn15_eagle_3b_robocasa_30_eps_full_finetune \
  -path '*/EVAL-robocasa-groot-*/summary.json' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

python - <<PY
from fluxvla.engines.utils.feishu_reporter import maybe_report_summary_to_feishu

result = maybe_report_summary_to_feishu(
    "$SUMMARY",
    "robocasa",
    config="configs/gr00tn15/gr00tn15_eagle_3b_robocasa_30_eps_full_finetune.py",
    logger=print,
    log_unconfigured=True,
)
print(result)
PY
```

For LIBERO summaries generated by `tools/summarize_libero_eval_results.py`,
you can pass Feishu arguments directly:

```bash
python tools/summarize_libero_eval_results.py \
  --scan-root work_dirs/libero_eval_runs \
  --output-dir work_dirs/libero_eval_summary \
  --feishu-sheet-url "$FEISHU_SHEET_URL" \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET"
```

## Config Fields

Direct `eval.py` / `eval.sh` reporting and `--eval-after-train` read credentials
from the environment; `eval.feishu_*` credential fields are not forwarded to
the reporter.

```python
eval = dict(
    feishu_sheet_url='https://example.feishu.cn/sheets/<token>',
    feishu_app_id='cli_xxx',
    feishu_app_secret='xxx',
)
```

For direct evaluation, set `eval.runner.feishu_report=False` in a namespaced
config, or `eval.feishu_report=False` in a flat config, to disable uploading.
Local summary generation still runs.
RoboCasa and LIBERO also skip direct upload when `task_ids` is set.
Reducing trial counts or step limits alone does not disable direct uploading.

## Troubleshooting

### `Feishu reporting is not configured`

No credentials reached the reporter. Check the launch environment:

```bash
printf '%s\n' "$FEISHU_SHEET_URL"
test -n "$FEISHU_APP_ID" && echo "FEISHU_APP_ID is set"
test -n "$FEISHU_APP_SECRET" && echo "FEISHU_APP_SECRET is set"
```

Do not print real secrets in shared logs.

### `invalid Feishu Sheets URL`

The link must contain `/sheets/<spreadsheet_token>` and use a Feishu/Lark
domain. Prefer a plain quoted URL:

```bash
export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>?sheet=<sheet_id>'
```

### `91403 Forbidden`

The app token was obtained, but the app identity cannot access the target
spreadsheet. Re-check app spreadsheet permissions, publish/approval status, the
spreadsheet collaborator/document-app permission, and tenant ownership.

### `header mismatch`

The target worksheet is not empty and its first row does not match the expected
header for the report kind. Either clear the worksheet, use another empty
worksheet, or set the first row to the exact header listed above.

### The log says `wrote result row`, but I cannot see it

Open the exact URL printed in the success log. If the input URL contains
`?sheet=<sheet_id>`, the write goes to that specific worksheet, even if another
tab is named `robotwin`, `robocasa`, or `libero`.

Repeated runs append rows. If you ran an older version before the append fix,
one row may have been written far below the visible rows; search for the config
path or commit id in the sheet.
