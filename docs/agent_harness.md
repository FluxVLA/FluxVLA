# AgentHarness inference

`AgentHarnessPolicy` is an opt-in, episode-scoped policy around the existing
literal-subtask planner and VLA. Dataset/tokenization, action normalization,
and simulator rollout loops do not own its state. Plain VLA,
`HierarchicalPolicy`, and candidate-only `per_request` inference are unchanged.

## First-stage capabilities and limits

- Keep a bounded recent event window and per-subtask execution summaries.
- Distinguish issued actions, actual execution acknowledgments, zero-action
  cancellation, and unknown completion. Switching labels is not success.
- Hold an instruction for `min_subtask_chunks` actually executed chunks, then
  confirm changes across `switch_confirmations` executed observations.
  `max_hold_chunks` bypasses switch debouncing, not candidate validation.
- Compare explicitly configured camera thumbnails after real execution.
  `changed` means sensor change, **not semantic task progress**. Missing prior
  evidence or no executed action yields `unknown`.
- Reconsider stalls using only the planner's valid original training labels.
  `stop_on_stall=False` is the default: camera-change thresholds are not a
  calibrated failure detector. Enabling it stops without issuing dummy actions
  after `no_change_patience` observations and `max_stall_replans` attempts.
  With stopping disabled, normal replanning continues after that budget, but
  stall-driven bypasses of holding/switch confirmation do not.
- API planners receive bounded execution memory as data. Existing local
  `LocalVLMSubtaskPlanner` deliberately ignores history/feedback and keeps its
  original current-images + goal SFT prompt. Its weights are not retrained.

This stage supports `completion_mode='disabled'` only. No reward, environment
success predicate, hidden object pose, or future annotation enters harness
memory or planning. The environment may still score/terminate benchmark
episodes, separately from the agent. A memory-aware local planner, semantic
progress model, task graph, or autonomous success decision requires separate
training/validation; these are not supplied by the current checkpoint.

## Configurations

- Local Qwen3.5: `configs/hierarchical/robodojo_pi05_qwen35_agent_harness.py`.
- API planner: `configs/hierarchical/robodojo_pi05_gpt6_agent_harness.py`.

Both retain the existing model checkpoints and full evaluation schedule.
The GPT config retains explicit errors for goals absent from its training
catalog; the Qwen config retains its base's full-training-vocabulary OOD mode.
The harness does not add a semantic candidate retriever or new skills.

The provided launch entries select four tasks × five episodes. Launch
**RoboDojo AgentHarness + Qwen3.5 + PI05: Debug VLA Server** in FluxVLA, and
wait for `model worker ready`, then use the matching **Themis Eval + Video**
entry or run Themis separately below. Starting Themis first also works: it
waits for the server. To use two terminals instead of launch.json, start the
server with:

```bash
cd /root/projects/FluxVLA
CFG=/root/projects/FluxVLA/configs/hierarchical/robodojo_pi05_qwen35_agent_harness.py
OUT=/root/projects/FluxVLA/work_dirs/robodojo_pi05_qwen35_agent_harness
env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u MASTER_ADDR -u MASTER_PORT CUDA_VISIBLE_DEVICES=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/projects/FluxVLA:/root/projects/FluxThemis \
  TMPDIR=/mnt/workspace/fluxthemis-runtime/tmp \
  OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  /root/miniconda3/envs/fluxvla/bin/python -u scripts/distributed_zmq_server.py \
  --config "$CFG" --frontend-bind tcp://127.0.0.1:15565 \
  --backend-bind tcp://127.0.0.1:15566 \
  --request-timeout-s 3600 --startup-timeout-s 3600 --cfg-options \
  themis.server.evaluation_reporting.result_output_dir="$OUT" \
  policy.harness.memory.trace_dir="$OUT/harness" \
  'themis.evaluation.environment.task_name=[stack_blocks,stack_bowls,push_T,insert_tubes]' \
  themis.evaluation.episodes_per_task=5 \
  themis.evaluation.episodes_per_task_overrides._delete_=True \
  themis.evaluation.max_episode_steps=550 \
  eval.num_trials_per_task=5 \
  eval.num_trials_per_task_overrides._delete_=True
```

In the other terminal, run Themis with:

```bash
cd /root/projects/FluxThemis
CFG=/root/projects/FluxVLA/configs/hierarchical/robodojo_pi05_qwen35_agent_harness.py
OUT=/root/projects/FluxVLA/work_dirs/robodojo_pi05_qwen35_agent_harness
env CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/projects/FluxThemis:/root/projects/FluxVLA \
  TMPDIR=/mnt/workspace/fluxthemis-runtime/tmp \
  OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
  /root/miniconda3/envs/RoboDojo/bin/python -u scripts/distributed_eval.py \
  --config "$CFG" --simulator-gpus 1 --workers-per-gpu 1 \
  --server-endpoint tcp://127.0.0.1:15565 \
  --rollout-bind tcp://127.0.0.1:15558 \
  --rollout-endpoint tcp://127.0.0.1:15558 \
  --save-videos --video-dir "$OUT/videos" \
  --startup-timeout-s 3600 --episode-timeout-s 86400 \
  --simulator-restart-limit 3 --cfg-options \
  themis.transport.connect_timeout_s=3600 \
  themis.transport.request_timeout_s=3600 \
  'themis.evaluation.environment.task_name=[stack_blocks,stack_bowls,push_T,insert_tubes]' \
  themis.evaluation.episodes_per_task=5 \
  themis.evaluation.episodes_per_task_overrides._delete_=True \
  themis.evaluation.max_episode_steps=550
```

For full evaluation, remove the task/count/step overrides on **both** sides.
Use a fresh output directory. The local VLA server command is the existing
`scripts/distributed_zmq_server.py` entry with the new config, one visible GPU,
and these launch-time overrides:

```text
themis.server.evaluation_reporting.result_output_dir=<OUT>
policy.harness.memory.trace_dir=<OUT>/harness
```

## Episode ownership and execution safety

Stateful serving requires `themis.server.mode='stateful'`,
`policy_execution='episode'`, **one model worker / one simulator stream**, and
`FluxVLAPolicyClient.send_policy_context=True`. The common client carries the
episode ID, request sequence, and actual executed-step count. Use the same
config on both sides; an unmodified action-only client cannot supply memory.
The current FluxThemis client already supports this; no simulator-specific
patches are needed. Requests within an episode must be ordered and cannot be
replayed. Do not interleave streams or reuse episode IDs. `AgentHarnessPolicy`
rejects `per_request` serving.

The current serving protocol acknowledges a chunk on the **next prediction
request**, using the increase in actual executed steps. If an episode ends
without another request, its final chunk remains pending with unknown
execution in harness memory. Benchmark episode reports do not fabricate that
missing acknowledgment. These counters therefore mean **confirmed execution**,
not necessarily every action the simulator performed.

Native synchronous and RTC/async entry points reuse their existing policy
execution barrier: join/cancel the previous actor, acknowledge actual actions,
capture a fresh observation, then plan. Queued actions are not counted as
executed; a new subtask continues to use the existing model-history reset.
This is safe barrier-based support, not speculative overlapping planners.

## Memory artifacts

With `trace_dir` enabled, each runtime/episode receives its own directory:

```text
<trace_dir>/<runtime-id>/<sha256(episode-id)>/
  events.jsonl       # append-only issued/acknowledged/selection evidence
  memory.json        # atomic latest execution-memory snapshot
  frame-*-camera-*.png
```

RAM event, summary, and keyframe windows are bounded. Saved keyframes are also
bounded per episode and immutable. Summaries always mark completion
`unverified`; truncating old summaries preserves total execution counters.
Optional fsync defaults on. Paths are metadata, not images accessible to GPT.

**Artifacts are diagnostic memory, not automatic physical-task recovery.**
Reset clears memory even for the same goal and records an unresolved pending
chunk as unknown execution. Restart never replays saved actions or assumes
that the physical scene still matches a checkpoint. Continuing a physical
task after a restart requires fresh sensing and explicit recovery logic.

For breakpoints, inspect `AgentHarnessPolicy.step`, `AgentHarness.choose`, and
`AgentHarness.acknowledge`. Serving prediction metadata contains `harness`,
including the active instruction, actual execution counters, switch proposal,
change evidence, recent history, summaries, and trace path.
