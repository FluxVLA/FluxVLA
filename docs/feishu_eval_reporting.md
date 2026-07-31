# Feishu Evaluation Reporting

FluxVLA can upload LIBERO, LIBERO-Plus, and RoboCasa evaluation summaries to a Feishu
spreadsheet. The upload is best-effort: evaluation results are still written
locally even when Feishu reporting is not configured or the Feishu API rejects
the write.

This feature works with:

- `scripts/eval.py`
- `scripts/eval.sh`
- `scripts/train.py --eval-after-train`
- `scripts/train.sh ... --eval-after-train`
- `tools/summarize_libero_eval_results.py`
- `scripts/ros_inference_server.sh` with FluxThemis `ReportEvaluation`
- `tools/summarize_libero_plus_results.py`

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
            result_output_dir='work_dirs/fluxthemis',
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
`result_output_dir` defaults to `work_dirs/fluxthemis`, resolves relative to the
FluxVLA repository, and must remain inside its `work_dirs` tree.

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

For an evaluation that passes the applicable reporting gate, the reporter writes
one row per completed summary. It does not deduplicate rows, so running the same
command twice appends two rows.

LIBERO uses this header:

```text
id, commit id, config, ckpt_path, libero_10, libero_goal, libero_object, libero_spatial, all
```

LIBERO-Plus uses this header:

```text
id, commit id, config, ckpt_path, Camera, Robot, Language, Light, Background, Noise, Layout, Total
```

RoboCasa uses this header:

```text
id, commit id, config, ckpt_path, Cabinet, Drawer, Microwave, Generalization, all
```

Column behavior:

- `id` is an auto-incrementing row id.
- `commit id` is `git rev-parse HEAD` from the current repo.
- `config` is the config path passed to evaluation when available.
- `ckpt_path` is the checkpoint path saved in the evaluation summary.
- Suite/group columns are formatted as percentages, for example `50.00%`.
- `all` is computed from total successes and total trials when those counts
  are present.
- LIBERO-Plus `Total` is micro-averaged over all 10,030 task trials, not the
  arithmetic mean of its seven perturbation columns.

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
if its title is not `libero`, `libero_plus`, or `robocasa`.

If the URL does not include `sheet=`, FluxVLA chooses a worksheet by report
kind:

- LIBERO writes to a worksheet named `libero`.
- LIBERO-Plus writes to a worksheet named `libero_plus`.
- RoboCasa writes to a worksheet named `robocasa`.
- If the worksheet does not exist, FluxVLA creates it.

Use the spreadsheet-level URL when you want one document with separate
`libero`, `libero_plus`, and `robocasa` tabs:

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
`selection=report kind sheet` means FluxVLA selected `libero`, `libero_plus`,
or `robocasa` from the report kind.

## RoboCasa With `scripts/eval.sh`

Example:

```bash
cd /path/to/FluxVLA

export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>?sheet=<sheet_id>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts/eval.sh \
  configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  work_dirs/gr00t_eagle_3b_robocasa_finetune/checkpoints/latest-checkpoint.safetensors \
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

When LIBERO evaluates multiple suites in one run, `scripts/eval.py` combines
the per-suite summaries and writes one Feishu row containing
`libero_10`, `libero_goal`, `libero_object`, `libero_spatial`, and `all`.

## LIBERO-Plus With `scripts/eval_libero_plus_manager.sh`

LIBERO-Plus evaluates one suite per manager invocation and uploads only after a
strict, complete four-suite summary is available. Partial summaries are always
kept local and are never uploaded.

Run the four suites with `scripts/eval_libero_plus_manager.sh` as documented in
the main README, then aggregate the shared run root with Feishu variables set:

```bash
cd /path/to/FluxVLA

export FEISHU_SHEET_URL='https://example.feishu.cn/sheets/<token>'
export FEISHU_APP_ID='cli_xxx'
export FEISHU_APP_SECRET='xxx'

python tools/summarize_libero_plus_results.py \
  --run-root work_dirs/libero_plus_pi05 \
  --output-dir work_dirs/libero_plus_pi05 \
  --title PI0.5 \
  --config configs/pi05 \
  --ckpt /path/to/checkpoints
```

Do not pass `--allow-incomplete` for a report intended for comparison. The
summarizer rejects duplicate tasks, missing official tasks, unknown task IDs,
and results containing anything other than one trial per task before upload.

## Eval After Train

The same environment variables work with `--eval-after-train`. The training
process relaunches evaluation in a fresh process after saving the checkpoint,
and the Feishu variables are inherited by that evaluation process.

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
  configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  work_dirs/gr00t_eagle_3b_robocasa_finetune \
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
  configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py \
  work_dirs/gr00t_eagle_3b_libero_10_full_finetune \
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

You can upload an existing RoboCasa summary without rerunning evaluation:

```bash
cd /path/to/FluxVLA

SUMMARY=$(find work_dirs/gr00t_eagle_3b_robocasa_finetune \
  -path '*/EVAL-robocasa-groot-*/summary.json' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

python - <<PY
from fluxvla.engines.utils.feishu_reporter import maybe_report_summary_to_feishu

result = maybe_report_summary_to_feishu(
    "$SUMMARY",
    "robocasa",
    config="configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py",
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

Environment variables are the recommended way to pass secrets. For controlled
internal runs, the same values can also be placed in config fields:

```python
eval = dict(
    feishu_sheet_url='https://example.feishu.cn/sheets/<token>',
    feishu_app_id='cli_xxx',
    feishu_app_secret='xxx',
    feishu_timeout=10.0,
)
```

For namespaced LIBERO manager configs, `eval.manager.feishu_sheet_url`,
`eval.manager.feishu_app_id`, and `eval.manager.feishu_app_secret` are also
accepted. Avoid committing real secrets to git.

## Troubleshooting

### `Feishu reporting is not configured`

No Feishu environment variables or config fields were found. Check:

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
tab is named `libero` or `robocasa`.

Repeated runs append rows. If you ran an older version before the append fix,
one row may have been written far below the visible rows; search for the config
path or commit id in the sheet.
