## 说明文档草稿​

克隆仓库

```bash
git clone \
    --branch gr00t-n17-native-libero-pr \
    https://github.com/jzzzzzzzzzzzzzzzz/FluxVLA.git \


conda create -n fluxvla-n17 python=3.10 -y
conda activate fluxvla-n17
bash scripts/install_env.sh sim-only --skip-robocasa
```
建议先在新环境里临时设置国内 pip 源，再跑安装脚本。
```bash
  conda activate fluxvla-n17

  python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
  python -m pip config set global.trusted-host mirrors.aliyun.com
  python -m pip config set global.timeout 120

  如果阿里源慢，换清华源：

  python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

  conda 也可以加国内源：

  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
  conda config --set show_channel_urls yes
```
  然后安装：
```bash
  PIP_INDEX_MODE=mirror \
  PIP_INDEX_URLS="https://mirrors.aliyun.com/pypi/simple https://pypi.tuna.tsinghua.edu.cn/simple" \
  GH_PROXY_CANDIDATES="https://ghfast.top https://gh.llkk.cc https://gh-proxy.com" \
  bash scripts/install_env.sh sim-only --skip-robocasa
```
预编译的flash att有问题不对应我们的版本,所以重装
```bash
python -m pip uninstall -y flash-attn

MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE \

python -m pip install --no-build-isolation --no-cache-dir \

"flash-attn==2.8.3.post1"
``` 
装完检查
```bash
import torch
import flash_attn
from fluxvla.processors.groot_n17_processor import GrootN17Processor

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash_attn import ok")
print("groot n17 processor import ok")
``` 
通过的话
```bash
python -m pip install --no-build-isolation -e .
``` 
导出资产环境 （如果没有共享存储，直接下载资产到对应路径）
```bash
mkdir -p checkpoints/GR00T-N1.7-LIBERO checkpoints/nvidia datasets

  ln -sfn /mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-3B \
    checkpoints/GR00T-N1.7-3B

  ln -sfn /mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    checkpoints/GR00T-N1.7-LIBERO/libero_10

  ln -sfn /mnt/data/cpfs/mnt/data/yiming/fluxvla-n17-native-dev/checkpoints/nvidia/Cosmos-Reason2-2B \
    checkpoints/nvidia/Cosmos-Reason2-2B

  ln -sfn /mnt/workspace/mnt/data/liyinhao/datasets/libero_10_no_noops_lerobotv2.1 \
    datasets/libero_10_no_noops_lerobotv2.1

声明导出

  export N17_INIT_CKPT=$PWD/checkpoints/GR00T-N1.7-3B
  export N17_PROCESSOR_META=$PWD/checkpoints/GR00T-N1.7-LIBERO/libero_10
  export N17_BACKBONE_MODEL_PATH=$PWD/checkpoints/nvidia/Cosmos-Reason2-2B
  export LIBERO_DATA_ROOT=$PWD/datasets/libero_10_no_noops_lerobotv2.1
``` 
## 参考命令模板1：
20k微调&&自动评测命令模板,仅做本地测试使用,完整训练请调整参数
```bash
conda activate fluxvla-n17
cd /root/projects/fluxvla-n17-libero-repro

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache
export MPLCONFIGDIR=/tmp/matplotlib
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

export N17_INIT_CKPT=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-3B
export N17_PROCESSOR_META=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-LIBERO/libero_spatial
export N17_BACKBONE_MODEL_PATH=/mnt/data/cpfs/mnt/data/yiming/fluxvla-n17-native-dev/checkpoints/nvidia/Cosmos-Reason2-2B
export LIBERO_DATA_ROOT=/mnt/workspace/mnt/data/liyinhao/datasets/libero_spatial_no_noops_lerobotv2.1

export N17_MAX_STEPS=20000
export N17_PER_DEVICE_BATCH_SIZE=1
export N17_GRAD_ACCUM_STEPS=1
export N17_NUM_WORKERS=2
export N17_SAVE_ITER_INTERVAL=1000
export N17_MAX_KEEP_CKPTS=2
export N17_ENABLE_GRAD_CKPT=1
export N17_SHARDING_STRATEGY=full-shard
export N17_ACTIVE_TRACKERS="('jsonl',)"

export N17_AUTO_EVAL_TRIALS=3
export N17_AUTO_EVAL_SEED=7
export N17_AUTO_EVAL_OUTPUT_DIR=work_dirs/dsw_n17_libero10_2k_auto_eval
```
```bash
python tools/prepare_groot_n17_libero.py --suite libero_10
``` 

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/train.py \
  --config configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py \
  --work-dir work_dirs/dsw_n17_libero10_2k \
  --eval-after-train \
  --cfg-options \
    'eval.task_ids=[0]' \
    eval.num_trials_per_task=3 \
    eval.save_rollout_videos=False \
    eval.save_failed_rollout_videos=False \
    eval.result_output_dir=work_dirs/dsw_n17_libero10_2k_auto_eval

``` 

## 参考命令模板2：
以 libero_goal 为例：
```bash
export N17_PROCESSOR_META=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-LIBERO/libero_goal
export LIBERO_DATA_ROOT=/mnt/workspace/mnt/data/liyinhao/datasets/libero_goal_no_noops_lerobotv2.1

export N17_MAX_STEPS=20000
export N17_PER_DEVICE_BATCH_SIZE=10
export N17_GRAD_ACCUM_STEPS=8
export N17_AUTO_EVAL_TRIALS=50
export N17_AUTO_EVAL_SEED=7
export N17_AUTO_EVAL_OUTPUT_DIR=work_dirs/groot_n17_native_libero_goal_full_auto_eval

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train.py \
  --config configs/gr00t/gr00t_n17_native_libero_goal_full_finetune.py \
  --work-dir work_dirs/groot_n17_native_libero_goal_full \
  --eval-after-train
``` 
  其他 suite 对应关系：
```bash
libero_10:
  config=configs/gr00t/gr00t_n17_native_libero_10_full_finetune.py
  meta=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-LIBERO/libero_10
  data=/mnt/workspace/mnt/data/liyinhao/datasets/libero_10_no_noops_lerobotv2.1
  work_dir=work_dirs/groot_n17_native_libero_10_full

libero_goal:
  config=configs/gr00t/gr00t_n17_native_libero_goal_full_finetune.py
  meta=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-LIBERO/libero_goal
  data=/mnt/workspace/mnt/data/liyinhao/datasets/libero_goal_no_noops_lerobotv2.1
  work_dir=work_dirs/groot_n17_native_libero_goal_full

libero_object:
  config=configs/gr00t/gr00t_n17_native_libero_object_full_finetune.py
  meta=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-LIBERO/libero_object
  data=/mnt/workspace/mnt/data/liyinhao/datasets/libero_object_no_noops_lerobotv2.1
  work_dir=work_dirs/groot_n17_native_libero_object_full

libero_spatial:
  config=configs/gr00t/gr00t_n17_native_libero_spatial_full_finetune.py
  meta=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-LIBERO/libero_spatial
  data=/mnt/workspace/mnt/data/liyinhao/datasets/libero_spatial_no_noops_lerobotv2.1
  work_dir=work_dirs/groot_n17_native_libero_spatial_full
``` 
  N17_INIT_CKPT 和 N17_BACKBONE_MODEL_PATH 不随 suite 改：
```bash
export N17_INIT_CKPT=/mnt/data/cpfs/mnt/data/yiming/fluxvla/checkpoints/GR00T-N1.7-3B
export N17_BACKBONE_MODEL_PATH=/mnt/data/cpfs/mnt/data/yiming/fluxvla-n17-native-dev/checkpoints/nvidia/Cosmos-Reason2-2B
  ``` 