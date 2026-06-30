# FluxVLA Engine：具現知能向け「ワンストップ」の VLA エンジニアリング基盤

<p align="center">
  <img src="assets/fluxvla.png" alt="FluxVLA" width="600">
</p>

<div align="center">
<a href="https://huggingface.co/limxdynamics/FluxVLAEngine"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://fluxvla.limxdynamics.com"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://fluxvla.limxdynamics.com/zh/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://github.com/limxdynamics/FluxVLA/issues/1"><img src="https://img.shields.io/badge/微信-green?logo=wechat"></a>
<a href="https://github.com/limxdynamics/FluxVLA/issues/1"><img src="https://img.shields.io/badge/飛書-3370FF?logo=lark&logoColor=white"></a>
</div>

<div align="center">

[English](README.md) | [簡体中文](README_zh-CN.md) | 日本語

</div>

FluxVLA Engine は、具現知能（Embodied Intelligence）の実運用を見据えた、エンドツーエンドの全チェーン一体型エンジニアリングプラットフォームです。統一設定、標準インターフェース、モジュール分離、デプロイ可能性を中核とした設計思想により、データから実機へのデプロイまでをつなぐ完全なエンジニアリング・クローズドループを構築します。また「標準化された産学研の基盤」を目標として、VLA 研究・開発におけるエンジニアリング上の参入障壁を大幅に引き下げます。

## フレームワーク

<p align="center">
  <img src="assets/framework.png" alt="Framework Architecture" width="800">
</p>

## パフォーマンス

| Codebase                    |                                                     Libero-Spatial                                                      |                                                      Libero-Object                                                      |                                                      Libero-Goal                                                      |                                                     Libero-Long                                                     | Libero-Average |
| --------------------------- | :---------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: | :------------: |
| FluxVLA(SmolVLA)            |      [86.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/smolvla_libero_spatial_full_finetune_bs64)      |      [92.4](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/smolvla_libero_object_full_finetune_bs64)       |      [91.4](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/smolvla_libero_goal_full_finetune_bs64)       |      [68.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/smolvla_libero_10_full_finetune_bs64)       |      84.7      |
| FluxVLA(GR00T)              |  [97.4](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_spatial_full_finetune_bs64)   |   [96.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_object_full_finetune_bs64)   |   [94.6](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_goal_full_finetune_bs64)   | [93.0±1.5](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_libero_10_full_finetune_bs64) |      95.3      |
| FluxVLA(Qwen3VL 0.6B+GR00T) | [96.0](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_qwen3vl_0.6b_libero_object_full_finetune_bs64) | [99.4](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_qwen3vl_0.6b_libero_object_full_finetune_bs64) | [95.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_qwen3vl_0.6b_libero_goal_full_finetune_bs64) | [94.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_qwen3vl_0.6b_libero_10_full_finetune_bs64) |     96.20      |
| FluxVLA(DreamZero)          | [98.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/dreamzero_libero_spatial_full_finetune_w_cache_bs64) | [98.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/dreamzero_libero_object_full_finetune_w_cache_bs64)  | [93.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/dreamzero_libero_goal_full_finetune_w_cache_bs64)  | [94.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/dreamzero_libero_10_full_finetune_w_cache_bs64)  |     96.25      |
| FluxVLA(PI0)                |   [98.6](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_paligemma_libero_spatial_full_finetune_bs64)   |   [98.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_paligemma_libero_object_full_finetune_bs64)    |   [96.8](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_paligemma_libero_goal_full_finetune_bs64)    |   [93.2](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_paligemma_libero_10_full_finetune_bs64)    |     96.85      |
| FluxVLA(PI0.5)              |  [98.6](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_spatial_full_finetune_bs64)   |   [99.6](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_object_full_finetune_bs64)   |   [98.0](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_goal_full_finetune_bs64)   | [95.6±1.0](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_libero_10_full_finetune_bs64) |     97.95      |

*リンク付きのスコアから対応するチェックポイントにアクセスできます。*

#### RoboCasa GR1

| モデル         |     学習データ     | Cabinet | Drawer | Microwave | Generalization |                                                       Average                                                        |
| -------------- | :----------------: | :-----: | :----: | :-------: | :------------: | :------------------------------------------------------------------------------------------------------------------: |
| FluxVLA(GR00T) | 24 タスク、30 デモ |  27.5%  | 37.5%  |   45.0%   |     50.3%      | [46.9%](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_eagle_3b_robocasa_gr1_24x30_finetune_bs64) |

#### 注記

- `Cabinet`：`PnPBottleToCabinetClose` + `PnPWineToCabinetClose`。
- `Drawer`：`PnPCanToDrawerClose` + `PnPCupToDrawerClose`。
- `Microwave`：`PnPMilkToMicrowaveClose` + `PnPPotatoToMicrowaveClose`。
- `Generalization`：残り 18 個のポストトレーニング新規タスク。
- すべての成功率は episode 単位の micro-average です。

## 📢 最新情報

**\[2026/06/22\]** 🔥 Oli ヒューマノイドの全身（移動操作）実機推論の最小パス（operator + runner + サンプル設定）が利用可能になりました。詳細は [docs/oli_whole_body.md](docs/oli_whole_body.md) を参照してください。

**\[2026/06/17\]** 🔥 ARM 報酬モデリングと RA-BC/AW-BC 再重み付けをサポートしました。セットアップと使い方は [docs/arm.md](docs/arm.md) を参照してください。

**\[2026/06/10\]** 🔥 GR00T による RoboCasa GR1 シミュレーションタスクに対応しました。

**\[2026/06/04\]** 🔥 Pi0.5-RTC の Triton バックエンドをサポートしました。詳細は [inference_acceleration](docs/inference_acceleration.md) を参照してください。

**\[2026/05/28\]** 🔥 双腕操作向けのモデル分離型 DAgger パイプライン [FluxDAgger](https://github.com/FluxVLA/FluxDAgger) を公開しました。さまざまな VLA と報酬モデルを容易に接続できます。

**\[2026/05/28\]** 🔥 具身操作シミュレーション Benchmark [FluxBisim](https://github.com/FluxVLA/FluxBisim) を公開しました。

**\[2026/05/09\]** 🔥 SmolVLA をサポートしました。

**\[2026/04/24\]** 🔥 Pi0.5-RTC をサポートしました。

**\[2026/04/22\]** 🔥 ZMQ ベースのリモート推論フレームワークをサポートしました。

**\[2026/04/15\]** 🔥 DreamZero WAM をサポートしました。

**\[2026/04/08\]** 🔥 FluxVLA をオープンソース化しました。

## 🛠️ インストール

以下のいずれかのインストール方法を選択してください：

- **推奨：一括インストールスクリプト**：通常の学習、シミュレーション評価、実機推論環境に使用します。
- **既存の FluxVLA 環境を更新する**：以前の FluxVLA をインストール済みで、変更された package だけ更新したい場合に使用します。
- **最初から手動でインストールする**：各 package のインストール手順を自分で制御したい場合のみ使用します。

### 推奨：一括インストールスクリプト

```bash
conda create -n fluxvla python=3.10 -y
conda activate fluxvla

# いずれかを選択: sim-only, real-only, full
bash scripts/install_env.sh sim-only
# bash scripts/install_env.sh real-only
# bash scripts/install_env.sh full
```

<details>
<summary><b>インストーラで問題が出る場合：mode と CUDA profile を確認する</b></summary>

`sim-only` はシミュレーション / LIBERO / RoboCasa 関連依存関係に加えて、固定バージョンの RoboCasa source checkout を `./src` にインストールします。`real-only` は実機とリモート推論の依存関係を、`full` は両方をインストールします。RoboCasa checkout が不要な場合は `--skip-robocasa` を使ってください。
インストーラが RoboCasa source checkout をインストールする場合
（`sim-only`、`full`、または `real-only --with-robocasa`）、RoboCasa
simulator assets もデフォルトでダウンロードします。インストーラは
`scripts/download_robocasa_assets.py` を呼び出し、
`FLUXVLA_ROBOCASA_ASSET_ENDPOINT`（デフォルトは `HF_ENDPOINT`、次に
`https://hf-mirror.com`）を使用します。asset だけをスキップするには
`--skip-robocasa-assets`、source checkout と asset の両方をスキップするには
`--skip-robocasa` を使ってください。

スクリプトは CUDA PyTorch profile を自動選択し、まず現在の CUDA toolkit / `nvcc` version を優先します。CUDA >= 12.8 では `cu128`、それ以外では `cu124` を選択します。toolkit が見つからない場合は driver-reported CUDA、最後に GPU generation を fallback として使います。`--profile cu128` または `--profile cu124` で明示指定できます。

PyTorch のインストール後、実際の Python tag、PyTorch バージョン、CUDA major version、C++ ABI、CPU architecture から FlashAttention wheel を自動選択します。対応する prebuilt wheel がない場合は `FLASH_ATTN_WHEEL_URL` を明示するか、`--skip-flash-attn` を使ってください。

`av` は conda の依存解決が遅くなるのを避けるため、デフォルトではまず pip wheel からインストールされます。wheel がない場合は conda にフォールバックします。conda-forge 版が必要な場合は `FLUXVLA_AV_INSTALLER=conda` を指定してください。

実機 runner にはシステム側の ROS も必要です。ROS Noetic の環境では、推論を起動する前に ROS を source してください：

```bash
source /opt/ros/noetic/setup.bash
```

</details>

<details>
<summary><b>インストーラで問題が出る場合：キャッシュ済みまたは mirror 上の FlashAttention wheel を使う</b></summary>

FlashAttention wheel は大きいため、ネットワークが遅い環境では GitHub release のダウンロードが初回インストール時間の大部分を占めることがあります。繰り返しインストールする場合は、正確に一致する wheel ファイルを `./wheelhouse/`、`./wheels/`、または `~/.cache/fluxvla/wheels/` に置いてください。インストーラはネットワークへアクセスする前にそれを使用します。ローカルファイルや内部 mirror を明示することもできます：

```bash
FLASH_ATTN_WHEEL_FILE=/path/to/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
bash scripts/install_env.sh sim-only --profile cu128

FLASH_ATTN_WHEEL_BASE_URLS="https://your-mirror.example.com/fluxvla/wheels" \
bash scripts/install_env.sh sim-only --profile cu128
```

</details>

<details>
<summary><b>インストーラで問題が出る場合：pip mirror と timeout を指定する</b></summary>

インストーラは既存の pip 設定を優先します。その index に package がない場合、または pip index が設定されていない場合は、PyPI と複数の一般的な mirror を probe し、このマシンでの応答速度順に再試行します。ネットワークが遅い、または不安定な場合は候補 mirror と timeout を明示できます：

```bash
PIP_INDEX_CANDIDATES="https://mirrors.aliyun.com/pypi/simple https://mirrors.cloud.tencent.com/pypi/simple https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple" \
PIP_INSTALL_TIMEOUT=7200 \
PIP_NETWORK_TIMEOUT=900 \
GH_PROXY=https://ghfast.top \
bash scripts/install_env.sh full
```

</details>

### 既存の FluxVLA 環境を更新する

FluxVLA(v0.1.0) をすでに clone / install している場合、conda 環境を作り直す必要はありません。最新コードを pull し、現在の simulation / model stack で実際に変わった package だけ更新してください：

```bash
bash scripts/update_env.sh
```

すでに checkout を更新済みの場合は `--skip-pull`、FluxVLA を editable mode で再インストールしない場合は `--skip-project` を指定してください。

<details>
<summary><b>等価な手動コマンド</b></summary>

```bash
git pull
python -m pip install --upgrade "transformers==5.3.0" "datasets==4.0.0"
python -m pip install "mujoco==3.2.6" gymnasium lxml bddl==1.0.1 hydra-core==1.2.0 robomimic==0.2.0
python -m pip install --force-reinstall --no-deps "libero @ git+https://github.com/yinchimaoliang/LIBERO.git@058fda1ddebe92918af091cb6816759ca6d003f0"
python -m pip install --force-reinstall --no-deps "robosuite @ git+https://github.com/yinchimaoliang/robosuite.git@e293cc32ff3c48957a4ebcad09952432b0dc9049"
python -m pip install --no-build-isolation -e .
python -c "import transformers; print(transformers.__version__)"
```

</details>

RoboCasa GR00T support は引き続き optional です。現在のインストーラは `sim-only` と `full` で `./src` 配下の Isaac-GR00T と RoboCasa GR1 local checkout を自動管理します。RoboCasa configs を使わない場合は `--skip-robocasa` を指定してください。

更新スクリプトは PyTorch や FlashAttention を再インストールしません。既存の `flash-attn==2.5.5` は、現在の PyTorch/CUDA build に対してまだ import できる場合のみ使い続けてください：

```bash
python - <<'PY'
import torch, flash_attn
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash-attn", flash_attn.__version__)
PY
```

現在のインストーラや以下の手順で PyTorch を更新する場合は、対応する FlashAttention wheel も再インストールしてください。現在のインストーラはデフォルトで `flash-attn==2.8.3.post1` を使います。

### 最初から手動でインストールする

`scripts/install_env.sh` を使わない場合のみ手動インストールを選択してください。先に PyTorch、次に FlashAttention、最後に FluxVLA の残りの依存関係をインストールします。

<details>
<summary><b>1. conda 環境を作成する</b></summary>

```bash
conda create -n fluxvla python=3.10 -y
conda activate fluxvla
```

</details>

<details>
<summary><b>2. PyTorch（CUDA バージョン）をインストールする</b></summary>

> **重要**：`pip install -r requirements.txt` を実行する前に、必ず公式の CUDA インデックスから PyTorch を先にインストールしてください。デフォルトの PyPI インデックスでは CUDA 対応ビルドを取得できません。

```bash
# CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

他の CUDA バージョンの場合は、`cu128` を該当する値（例：`cu118`、`cu121`）に置き換えてください。詳細は [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) および [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/) を参照してください。

</details>

<details>
<summary><b>3. flash-attention をインストールする</b></summary>

一括インストールスクリプトは公式 release assets から prebuilt
FlashAttention wheel をダウンロードします。手動で入れる場合も、ソースビルドではなく Python、PyTorch、C++ ABI に合う wheel を指定してください：

```bash
PYTAG=$(python - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)
ABI=$(python - <<'PY'
import torch
print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())
PY
)

pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
```

PyTorch 2.6 を使う場合は、URL 内の `torch2.8` を `torch2.6` に置き換えてください。

FlashAttention wheel は Python、PyTorch、CUDA、C++ ABI に強く依存します。`flash-attn==2.5.5` 自体は禁止ではありませんが、現在使っている PyTorch/CUDA stack 向けに build されていて、上の import check が通る場合のみ保持してください。PyTorch を更新した後は、対応する FlashAttention wheel を再インストールしてください。

</details>

<details>
<summary><b>4. av をインストールする</b></summary>

```bash
conda install -c conda-forge av=14.4.0
```

</details>

<details>
<summary><b>5. fluxvla とその他の依存関係をインストールする</b></summary>

```bash
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

> **補足**：`requirements.txt` は `requirements-base.txt`、`requirements-sim.txt`、`requirements-real.txt` をまとめるファイルになりました。PyTorch はインストールしないため、先に CUDA 版 PyTorch を入れるか、`scripts/install_env.sh` を使用してください。

</details>

<details>
<summary><b>任意：RoboCasa GR00T source checkout</b></summary>

RoboCasa GR00T 設定（例：`configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py`）には、固定バージョンの Isaac-GR00T と RoboCasa GR1 task checkout が必要です。one-click installer は `sim-only` と `full` でこれらをデフォルトでインストールし、`./src` 配下に配置します：

```bash
bash scripts/install_env.sh sim-only
```

checkout root を変更するには `FLUXVLA_ROBOCASA_SRC_ROOT=/path/to/src` を使います。source install をスキップするには `--skip-robocasa`、`real-only` mode でも強制するには `--with-robocasa` を使ってください。runtime dependencies と patch 済み robosuite は `requirements-sim.txt` からインストールされます。

インストーラを使わない場合の同等の手順は次のとおりです：

```bash
pip install "mujoco==3.2.6" gymnasium lxml
pip install "robosuite @ git+https://github.com/yinchimaoliang/robosuite.git@e293cc32ff3c48957a4ebcad09952432b0dc9049"

git clone https://github.com/NVIDIA/Isaac-GR00T.git ./src/Isaac-GR00T
git -C ./src/Isaac-GR00T checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
pip install --no-deps -e ./src/Isaac-GR00T

git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git \
  ./src/robocasa-gr1-tabletop-tasks
git -C ./src/robocasa-gr1-tabletop-tasks checkout 4840e671596f93ca03651524b9f72ffb1aadfeff
pip install --no-deps -e ./src/robocasa-gr1-tabletop-tasks
```

editable インストールでは `--no-deps` を推奨します。RoboCasa 関連パッケージが FluxVLA のモデルスタックで固定された依存を置き換えないようにするためです。RoboCasa のアセットとデータセットの準備は[データとアセットの準備](#データとアセットの準備)を参照してください。

</details>

<details>
<summary><b>任意：LIBERO / MuJoCo EGL オンライン評価設定</b></summary>

レイトレーシング非対応のデバイス（例：A100）で LIBERO を評価したい場合は、[EGL Device GPU Rendering Configuration](https://github.com/google-deepmind/mujoco/issues/572#issuecomment-2419965230) を参照してください。

`scripts/install_env.sh sim-only` と `scripts/install_env.sh full` は MuJoCo EGL を自動で確認します。EGL デバイスが見えない場合、インストーラは以下のシステムパッケージのインストール、NVIDIA GLVND vendor ファイルの作成、`MUJOCO_GL=egl` 用の conda activation hook の作成を試みます。厳密に失敗させたい場合は `FLUXVLA_EGL_SETUP=always`、スキップする場合は `--skip-egl-setup` を使ってください。

**システム依存関係のインストール**

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
sudo apt-get update
sudo apt-get install -y libegl1 libglvnd0 libopengl0 libegl-dev libgl1-mesa-dev libx11-dev libglew-dev libosmesa6-dev
```

**環境チェック**

`/proc/1/environ` に以下の環境変数が含まれていることを確認してください：

- `NVIDIA_DRIVER_CAPABILITIES=all`
- `NVARCH=x86_64`
- `NVIDIA_REQUIRE_CUDA=cuda>=12.4`
- `brand=tesla` かつ `driver>=470`

**EGL 設定ファイルの作成**

`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` を作成し、内容は以下の通りにしてください：

```json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
```

その後、環境がすでに設定していない場合は `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` を付けて eval を起動してください。

</details>

<details>
<summary><b>pre-commit フックの設定（任意だが推奨）</b></summary>

コードの品質と一貫性を担保するため（特に C++/CUDA コード）、pre-commit フックを導入することを推奨します：

```bash
pip install pre-commit
pre-commit install
```

これにより、コミット前に自動でコードのチェックとフォーマットが行われます。

</details>

<details>
<summary><b>Weights & Biases（wandb）の設定</b></summary>

[Weights & Biases](https://wandb.ai/) は、実験のトラッキングと可視化に使われます。設定手順は次の通りです：

1. wandb をインストール（`requirements.txt` に含まれています）：

```bash
pip install wandb
```

2. wandb アカウントにログイン：

```bash
wandb login
```

3. 環境変数を設定：

```bash
export WANDB_PROJECT=fluxvla        # プロジェクト名（デフォルト：fluxvla）
export WANDB_ENTITY=your-team-name  # チーム名またはユーザー名（デフォルト：None）
export WANDB_MODE=online            # online、offline、または disabled（デフォルト：online）
```

4. 学習時に wandb のログを無効化したい場合は、次を設定：

```bash
export WANDB_MODE=disabled
```

補足：すべての wandb 設定は環境変数から読み取られるため、設定ファイルに追加設定は不要です。

</details>

<details>
<summary><b>TensorBoard の設定（オプション）</b></summary>

[TensorBoard](https://www.tensorflow.org/tensorboard) はオプションのログバックエンドとして、実験メトリクスの可視化に使用できます。設定手順は次の通りです：

1. 設定ファイルの `active_trackers` に `'tensorboard'` を追加：

```python
metric=dict(
    type='VLAMetric',
    active_trackers=('jsonl', 'wandb', 'tensorboard'),
    ...
)
```

設定ファイルを変更せずに、コマンドラインから有効化することも可能です：

```bash
--cfg-options 'runner.metric.active_trackers=[jsonl,wandb,tensorboard]'
```

2. トレーニング後、TensorBoard を起動してメトリクスを確認：

```bash
tensorboard --logdir work_dirs/tensorboard
```

補足：各実験のイベントファイルは `{work_dir}/tensorboard/{run_id}/` に保存され、複数の実験を自動的に比較できます。`TENSORBOARD_LOG_PATH` 環境変数が設定されている場合、そのパスがログディレクトリとして直接使用されます。

</details>

## データとアセットの準備

<details>
<summary><b>用意済みのデータをそのまま使う</b></summary>

必要なデータセットをダウンロードし、`./datasets` ディレクトリに配置してください。設定に応じて、必要なデータセットだけをダウンロードします。

| データセット            | ダウンロードリンク                                                                                                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| libero-object           | [limxdynamics/FluxVLAData/libero_object_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_object_no_noops_lerobotv2.1)         |
| libero-spatial          | [limxdynamics/FluxVLAData/libero_spatial_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_spatial_no_noops_lerobotv2.1)       |
| libero-10               | [limxdynamics/FluxVLAData/libero_10_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_10_no_noops_lerobotv2.1)                 |
| libero-goal             | [limxdynamics/FluxVLAData/libero_goal_no_noops_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/libero_goal_no_noops_lerobotv2.1)             |
| modified_libero_rlds    | [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)                                                                                 |
| RoboCasa GR1 (30 demos) | [limxdynamics/FluxVLAData/robocasa_gr1_24tasks_first30ep](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/robocasa_gr1_24tasks_first30ep)                 |
| RoboCasa GR1            | [limxdynamics/FluxVLAData/robocasa_lerobot_V2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/robocasa_lerobot_V2.1)                                   |
| ARM manual test         | [limxdynamics/FluxVLAData/ARM_manual_test_10Episodes_lerobotv3.0](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/ARM_manual_test_10Episodes_lerobotv3.0) |
| RealRobot_AgileX_aloha  | [limxdynamics/FluxVLAData/RealRobot_AgileX_aloha_lerobot_v2](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/RealRobot_AgileX_aloha_lerobot_v2)           |
| RealRobot_UR3_Chem      | [limxdynamics/FluxVLAData/RealRobot_UR3_Chem_lerobot_v2](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/RealRobot_UR3_Chem_lerobot_v2)                   |

例えば、`libero-10` データセットをダウンロードする場合：

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "libero_10_no_noops_lerobotv2.1/*" --local-dir ./datasets
```

`libero_10_no_noops_lerobotv2.1` を、ダウンロードしたいデータセットに対応するフォルダ名に置き換えてください。

公開済みの 30 デモのサブセットで RoboCasa GR00T を学習する場合は、データセットを `./datasets` にダウンロードします：

```bash
huggingface-cli download limxdynamics/FluxVLAData \
  --repo-type dataset \
  --include "robocasa_gr1_24tasks_first30ep/*" \
  --local-dir ./datasets
```

全量の RoboCasa GR1 データで学習する場合は、include パターンを `robocasa_lerobot_V2.1/*` に置き換えてください。

</details>

<details>
<summary><b>ARM データセット</b></summary>

組み込みの ARM サンプル設定 `configs/arm/arm_clip_aloha_example.py` は、progress ラベル付きの LeRobot v3.x データセットが `./datasets/ARM_manual_test_10Episodes_lerobotv3.0` にあることを前提にしています。

公開済みのサンプルデータセットは、次のコマンドで期待される場所にダウンロードできます：

```bash
huggingface-cli download limxdynamics/FluxVLAData \
  --repo-type dataset \
  --include "ARM_manual_test_10Episodes_lerobotv3.0/*" \
  --local-dir ./datasets
```

ARM の学習は、このデータセットの `progress` 列を直接読み取ります。`progress` を持たない policy / DAgger データセットで RA-BC / AW-BC を使う場合は、まず ARM checkpoint を学習または読み込み、`scripts/compute_arm_awbc_progress.py` で `arm_progress.parquet` を生成してください。詳細は [docs/arm.md](docs/arm.md) と [tools/arm_awbc/README.md](tools/arm_awbc/README.md) を参照してください。

</details>

<details>
<summary><b>アセットの準備</b></summary>

RoboCasa GR1 tabletop タスクでは、以下の FluxVLA asset downloader を
サポートされる手順として使用してください。表はスクリプトが使用する
upstream archive の一覧です。これらの archive を手動でダウンロードして
展開するだけでは不十分です。このスクリプトは directory layout の修正と、
固定された RoboCasa GR1 checkout 向けの Objaverse XML metadata の正規化も
行います。

| アセット archive                                           | ダウンロードリンク                                                                                               | ローカルディレクトリ                                       |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `objaverse.zip`, `textures.zip`, `generative_textures.zip` | [robocasa/robocasa-assets](https://huggingface.co/datasets/robocasa/robocasa-assets)                             | `./src/robocasa-gr1-tabletop-tasks/robocasa/models/assets` |
| `fixtures.zip`                                             | [jianzhang96/robocasa-assets](https://huggingface.co/datasets/jianzhang96/robocasa-assets)                       | `./src/robocasa-gr1-tabletop-tasks/robocasa/models/assets` |
| `sketchfab.zip`, `lightwheel.zip`                          | [nvidia/PhysicalAI-DigitalCousin-Assets](https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets) | `./src/robocasa-gr1-tabletop-tasks/robocasa/models/assets` |

`scripts/install_env.sh` を使う場合、この downloader は RoboCasa source
checkout と一緒にデフォルトで実行されます。ただし `--skip-robocasa` または
`--skip-robocasa-assets` を指定した場合は実行されません。手動インストールや
asset の再取得では、FluxVLA repository root から以下を実行してください。
指定した Hugging Face endpoint 経由で必要な archive をダウンロードし、
RoboCasa の asset directory に展開し、Objaverse XML metadata を正規化します:

```bash
python scripts/download_robocasa_assets.py --endpoint https://hf-mirror.com
```

archive または展開済み asset がすでにローカルにある場合でも、XML 互換処理
を適用するため、このスクリプトは実行してください。asset がすでに
`./src/robocasa-gr1-tabletop-tasks/robocasa/models/assets` に展開済みの場合は、
validation と XML 正規化だけを実行できます:

```bash
python scripts/download_robocasa_assets.py --normalize-only
```

シンボリックリンクは必須ではなく、アセットが別のローカルディスクや共有ストレージに既に存在する場合の利便性のための手段にすぎません。

</details>

<details>
<summary><b>SARM データセット</b></summary>

FluxVLA の SARM ワークフローは、標準的な LeRobot v2.1 / v3.x データセットをサポートします。通常の observation / action フィールドに加えて、episodes メタデータに SARM subtask アノテーション列が必要です。

公開済みの SARM サンプルデータセット:

- LeRobot v3.x 版の学習 / 推論向け手動 sparse+dense アノテーション付きデータ: [limxdynamics/FluxVLAData/SARM_manual_test_10Episodes_lerobotv3.0](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_manual_test_10Episodes_lerobotv3.0)
- LeRobot v3.x 版の手動または VLM アノテーション用未注釈データ: [limxdynamics/FluxVLAData/SARM_vlm_test_10Episodes_lerobotv3.0](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_vlm_test_10Episodes_lerobotv3.0)
- 新しい LeRobot v2.1 manual 変換版。学習 / 推論や旧来ツール互換向け: [limxdynamics/FluxVLAData/SARM_manual_test_10Episodes_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_manual_test_10Episodes_lerobotv2.1)
- 新しい LeRobot v2.1 vlm 変換版。手動 stage 書き込みや VLM 自動アノテーション向け: [limxdynamics/FluxVLAData/SARM_vlm_test_10Episodes_lerobotv2.1](https://huggingface.co/datasets/limxdynamics/FluxVLAData/tree/main/SARM_vlm_test_10Episodes_lerobotv2.1)

`./datasets` へは次のようにダウンロードできます:

```bash
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_manual_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_vlm_test_10Episodes_lerobotv3.0/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_manual_test_10Episodes_lerobotv2.1/*" --local-dir ./datasets
huggingface-cli download limxdynamics/FluxVLAData --repo-type dataset --include "SARM_vlm_test_10Episodes_lerobotv2.1/*" --local-dir ./datasets
```

`manual_*` はそのまま学習 / 推論に使えます。`vlm_*` は手動 stage 書き込みや VLM 自動アノテーションの開始点として使います。`meta/episodes.jsonl` と episode 単位動画を前提とするツールでは v2.1 を、ネイティブな LeRobot v3.x metadata を保ちたい場合は v3.0 を優先してください。

LeRobot v3.x の SARM データセットを使う前に、動画メタデータを確認してください:

- LeRobot v3.x では、複数 episode を 1 本の MP4 にまとめても、1 episode ごとに 1 本の MP4 でも構いません。

- 複数 episode が同じ MP4 を共有する場合は、各 episode の `from_timestamp` / `to_timestamp` がその動画内の区間を正しく表している必要があります。

- 動画がすでに `file-000.mp4`、`file-001.mp4` のように episode ごとに分かれている場合は、各 episode が対応する `file_index` を指し、`from_timestamp` は通常 `0.0` に戻ります。

- ディレクトリ内に複数の MP4 があるのに、すべての episode が `file-000.mp4` を指している場合、その metadata は壊れているため、使用前に修正してください。

- SARM データセット構成、アノテーション列の契約、progress 推論の使い方は [docs/sarm.md](docs/sarm.md) を参照してください。

- 手動 stage 書き込みや VLM ベースの自動アノテーションは [tools/sarm_annotate/README.md](tools/sarm_annotate/README.md) を参照してください。

</details>

<details>
<summary><b>プライベートデータセットのディレクトリ構造</b></summary>

fluxvla をプライベートデータセットで学習する場合、まず生データ（例：ALOHA ロボットで収集した HDF5 ファイル）を LeRobot Dataset v2.1 形式に変換する必要があります。変換手順の詳細は [データ変換ガイド](docs/data_convert.md) をご覧ください。

SARM については、必要な SARM アノテーション列が含まれていれば、FluxVLA は LeRobot v2.1 と v3.x の両方を扱えます。必要なメタデータ形式は [docs/sarm.md](docs/sarm.md) にまとめています。

変換後のデータセットのディレクトリ構造は次のとおりです：

```
├── data
│   └── chunk-000
│   │   └── episode_000000.parquet
│   │   └── episode_000001.parquet
│   │   └── ...（さらに多くの parquet ファイル）
│   │   └── episode_00000N.parquet
│   └── chunk-001
│   └── ...（さらに多くの chunk）
│   └── chunk-00N
├── meta
│   └── episodes.jsonl
│   └── episodes_stats.jsonl
│   └── info.json
│   └── tasks.jsonl
├── videos
│   └── chunk-000
│   │   └── camera name 0
│   │   │   └── episode_000000.mp4
│   │   │   └── episode_000001.mp4
│   │   │   └── ...（さらに多くの mp4 ファイル）
│   │   │   └── episode_00000N.mp4
│   │   └── camera name 1
│   │   └── ...（さらに多くのカメラ）
│   │   └── camera name N
│   └── chunk-001
│   └── ...（さらに多くの chunk）
│   └── chunk-00N
```

</details>

## 🤗 チェックポイント準備

必要な事前学習済みチェックポイントをダウンロードし、`./checkpoints` ディレクトリに配置してください。設定に応じて必要なチェックポイントだけをダウンロードします。

ARM と SARM のワークフローでは、通常は学習 / 推論用の CLIP チェックポイントが必要です。SARM の VLM ベース自動アノテーションでは、公式 SARM で使われている Qwen3-VL チェックポイントも必要です。詳細は [docs/arm.md](docs/arm.md) と [docs/sarm.md](docs/sarm.md) を参照してください。

<details>
<summary><b>VLA モデル</b></summary>

| モデル      | サイズ | ダウンロードリンク                                                                         |
| ----------- | ------ | ------------------------------------------------------------------------------------------ |
| GR00T N1.5  | 3B     | [🤗 Hugging Face](https://huggingface.co/nvidia/GR00T-N1.5-3B/tree/main)                   |
| OpenVLA     | 7B     | [🤗 Hugging Face](https://huggingface.co/openvla/openvla-7b)                               |
| PI0_base    | 3B     | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi0_base)    |
| PI05_base   | 3B     | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_base)   |
| PI05_libero | 3B     | [🤗 Hugging Face](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_libero) |

</details>

<details>
<summary><b>視覚言語モデル（VLM）</b></summary>

| モデル     | サイズ | ダウンロードリンク                                                       |
| ---------- | ------ | ------------------------------------------------------------------------ |
| Qwen2.5-VL | 3B     | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)    |
| Qwen3-VL   | 30B    | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) |

</details>

<details>
<summary><b>大規模言語モデル（LLM）</b></summary>

| モデル   | サイズ | ダウンロードリンク                                                           |
| -------- | ------ | ---------------------------------------------------------------------------- |
| Qwen 2.5 | 3B     | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B)                    |
| Qwen 2.5 | 7B     | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B)                    |
| Llama 2  | 7B     | [🤗 Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) |

</details>

<details>
<summary><b>視覚バックボーンネットワーク</b></summary>

| モデル              | ダウンロードリンク                                                                   |
| ------------------- | ------------------------------------------------------------------------------------ |
| CLIP ViT-B/32       | [🤗 Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)               |
| ViT-Large (DINOv2)  | [🤗 Hugging Face](https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m) |
| ViT-SO400M (SigLIP) | [🤗 Hugging Face](https://huggingface.co/timm/ViT-SO400M-14-SigLIP)                  |
| SigLIP2             | [🤗 Hugging Face](https://huggingface.co/google/siglip2-base-patch16-224)            |
| paligemma           | [🤗 Hugging Face](https://huggingface.co/google/paligemma-3b-pt-224)                 |

> **ヒント**：`huggingface-cli download <model-name> --local-dir ./checkpoints/<model-name>` を使うとダウンロードを高速化できます。

組み込みの ARM と SARM 設定では、CLIP ファイルを `./checkpoints/clip-vit-base-patch32` に配置してください：

```bash
huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./checkpoints/clip-vit-base-patch32
```

VLM ベースの自動アノテーションを使う場合は、公式 SARM VLM を `./checkpoints/Qwen3-VL-30B-A3B-Instruct` に配置してください。

</details>

## 🌟 特徴

<details>
<summary><b>All-in-one：1 つの設定ファイルで全工程を管理</b></summary>

- データ、モデル、学習、評価、推論、デプロイに必要な主要パラメータを 1 つの設定ファイルで統一管理できます（再現性とデプロイ性が向上します）。

</details>

<details>
<summary><b>異なる VLA モデルに対応</b></summary>

- OpenVLA、LlavaVLA、Gr00t、Pi0、Pi0.5 をサポートします。

</details>

<details>
<summary><b>異なるモジュールに対応</b></summary>

- Llama、Gemma、Qwen 系の LLM バックボーンをサポートします。
- DINOv2、SigLIP の視覚バックボーンをサポートします。
- PaliGemma、Qwen-VL の VLM バックボーンをサポートします。

</details>

<details>
<summary><b>報酬モデリングワークフローに対応</b></summary>

- [SARM](https://github.com/xdofai/opensarm) の学習、アノテーション、progress 推論をサポートし、LeRobot v2.1/v3.x データセットに対応しています。詳細は [docs/sarm.md](docs/sarm.md) を参照してください。
- [ARM](https://arxiv.org/abs/2604.03037) の報酬モデリング、progress 再構成、RA-BC / AW-BC サンプル再重み付けをサポートします。詳細は [docs/arm.md](docs/arm.md) を参照してください。

</details>

<details>
<summary><b>異なる学習戦略に対応</b></summary>

- FSDP と DDP の併用に対応し、LoRA 学習モードもサポートします。
- train 後に即 eval（eval-after-train）に対応します。
- checkpoint から学習を再開できます。

</details>

<details>
<summary><b>データと重みのフォーマット</b></summary>

- Parquet データセットをサポートし、LeRobot 形式のデータも読み込み可能です。
- safetensors 形式のモデル重みをサポートします。

</details>

<details>
<summary><b>評価と推論の能力</b></summary>

- マルチ GPU によるレイトレーシング非対応デバイスでの libero 評価をサポートします。
- ZMQ ベースのリモート推論インフラをサポートします。サーバー/クライアントアーキテクチャにより、モデル推論を GPU サーバーにオフロードし、リソースが限られたエッジデバイスへのデプロイを可能にします。詳細は [リモート推論サービス](docs/remote_inference_serving.md) を参照してください。
- [RTC（Real-Time Chunking）](docs/rtc.md) をサポートし、チャンク間の軌跡の連続性を向上させます。
- GR00T と PI0.5 の推論を高速化します。詳細は [Inference Acceleration](docs/inference_acceleration.md) を参照してください。Triton の融合カーネル、CUDA Graph のキャプチャ、CUDA のカスタム演算子が含まれます。
- Oli ヒューマノイドの全身（移動操作）実機推論の最小パスを提供（rospy センサ入力 + WebSocket 制御出力；base/ハンド指令はロボット SDK の統合ポイント）。詳細は [docs/oli_whole_body.md](docs/oli_whole_body.md) を参照。

</details>

<p align="center">
  <img src="assets/VLA_speedup.png" alt="VLA Speedup" width="800">
</p>

## 使い方

<details>
<summary><b>ローカルデバッグ</b></summary>

```
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node [NUM_GPUS] scripts/train.py --config [CONFIG_PATH] --work-dir [WORK_DIR] --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE]
```

例：

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py --work-dir ./checkpoints/pi05_paligemma_libero_10_full_finetune --cfg-options train_dataloader.per_device_batch_size=2
```

RoboCasa GR00T のスモーク学習の例：

```bash
WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --work-dir work_dirs/smoke_groot_robocasa_train \
  --cfg-options \
    runner.type=FSDPTrainRunner \
    runner.sharding_strategy=no-shard \
    train_dataloader.per_device_batch_size=1 \
    runner.enable_gradient_checkpointing=False \
    runner.max_steps=2 \
    runner.save_iter_interval=1 \
    runner.max_keep_ckpts=2 \
    "runner.metric.active_trackers=('jsonl',)"
```

</details>

<details>
<summary><b>ローカル評価</b></summary>

```
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node [NUM_GPUS] scripts/eval.py --config [CONFIG_PATH] --ckpt-path [CKPT_PATH] --cfg-options [CFG_OPTIONS]
```

例：

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/eval.py --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py --ckpt-path checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors
```

RoboCasa GR00T の評価の例：

```bash
MUJOCO_GL=egl WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
  --config configs/gr00t/gr00t_eagle_3b_robocasa_finetune.py \
  --ckpt-path work_dirs/gr00t_eagle_3b_robocasa_gr1_24x30_finetune_bs64/checkpoints/step-010000.safetensors \
  --cfg-options \
    eval.norm_stats_path=work_dirs/official_groot_gr1_dataset_statistics.json \
    eval.output_dir=work_dirs/gr00t_eagle_3b_robocasa_eval \
    eval.num_trials_per_task=20
```

</details>

<details>
<summary><b>クラスター学習</b></summary>

```
export WANDB_MODE=disabled
bash scripts/train.sh [CONFIG] [WORK_DIR] --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE] train_dataloader.batch_size=[GLOBAL_BATCH_SIZE] runner.max_steps=[MAX_STEPS] runner.save_interval=[SAVE_INTERVAL] runner.max_keep_ckpts=[MAX_KEEP_CKPTS] --eval-after-train
```

</details>

<details>
<summary><b>checkpoint から学習を再開する</b></summary>

checkpoint から学習を再開するには、`--resume-from` パラメータで checkpoint ファイルのパスを指定します。学習は保存されている global step、epoch、モデル状態、最適化器状態から継続されます。

**ローカル学習の例：**

```
export WANDB_MODE=disabled
/root/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --work-dir ./work_dirs/pi05_paligemma_libero_10_full_finetune \
  --resume-from ./work_dirs/pi05_paligemma_libero_10_full_finetune/checkpoints/checkpoint_epoch_5.pt \
  --cfg-options train_dataloader.per_device_batch_size=2
```

**クラスター学習の例：**

```
export WANDB_MODE=disabled
bash scripts/train.sh [CONFIG] [WORK_DIR] \
  --resume-from [CHECKPOINT_PATH] \
  --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE] runner.max_steps=[MAX_STEPS]
```

</details>

<details>
<summary><b>クラスター評価</b></summary>

```
export WANDB_MODE=disabled
bash scripts/eval.sh [CONFIG] [CKPT_PATH] --cfg-options [CFG_OPTIONS]
```

</details>

<details>
<summary><b>実ロボットでの推論</b></summary>

実機ロボット上で推論を実行する際は、まずロボット側で環境をセットアップし、その上で次のコマンドを実行してください：

```
python scripts/inference_real_robot.py --config [CONFIG] -- ckpt-path [CKPT_PATH]
```

</details>

## よくある質問（FAQ）

<details>
<summary><b>Q：モデルまたはデータセットのダウンロード時に Hugging Face へ接続できない。</b></summary>

<b>A：</b> Hugging Face の接続問題（ダウンロードが遅い、タイムアウト、接続拒否など）が発生する場合は、コマンド実行前に次の環境変数を設定し、[hf-mirror](https://hf-mirror.com) を利用してください：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

</details>

<details>
<summary><b>Q：<code>conda install av</code> の環境解決が非常に遅い。</b></summary>

<b>A：</b>依存関係の解決を高速化するために `libmamba` ソルバを使えます：

```bash
conda install -c conda-forge av=14.4.0 --solver=libmamba
```

</details>

<details>
<summary><b>Q：LIBERO 上での GR00T の評価結果が不安定。</b></summary>

<b>A：</b>これは想定される挙動です。GR00T の LIBERO 上での性能は、乱数シード、ハードウェア環境、学習 epoch 数に敏感です。これらの要因の小さな変化でも、評価結果が大きく揺れる可能性があります。複数の乱数シードで実験し、評価結果に基づいて最適な checkpoint を選ぶことをおすすめします。

</details>

<details>
<summary><b>Q：<code>pip install -r requirements.txt</code> 実行時に <code>egl_probe</code> のビルドが失敗し、<code>RuntimeError: CMake must be installed</code> と表示される。</b></summary>

<b>A：</b> `egl_probe` はビルドに CMake が必要です。conda（推奨）または apt で CMake をインストールしてください：

```bash
conda install -c conda-forge cmake
# または
sudo apt install cmake
```

> **補足**：`pip install cmake` は使わないでください。pip の `cmake` は Python のラッパーであり、pip がビルド環境を分離するため失敗する可能性があります。

</details>

<details>
<summary><b>Q：<code>egl_probe</code> のビルドが失敗し、<code>Compatibility with CMake < 3.5 has been removed from CMake</code> と表示される。</b></summary>

<b>A：</b> これは通常、あなたの CMake バージョンが `egl_probe` の `CMakeLists.txt` に対して新しすぎることが原因です。インストール前に次の環境変数を設定してください：

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -r requirements.txt
```

</details>

<details>
<summary><b>Q：インストール後に NumPy バージョンのエラーが出る（例：<code>RuntimeError: Numpy is not available</code> またはバージョン互換性警告）。</b></summary>

<b>A：</b> インストール中に一部の依存関係が固定された NumPy バージョンを書き換えることがあります。正しいバージョンを直接入れ直してください：

```bash
pip install numpy==1.26.4
```

</details>

## コントリビューション

貢献の手順とガイドラインは [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) を参照してください。

クイック約束（最小限）：

- **先に相談**：新機能／新モデル／大きな変更は、まず GitHub Issue で目的・設計・範囲を共有してください。
- **upstream からブランチ作成**：`upstream/main` を起点にし、`feat/`、`fix/`、`docs/` などの接頭辞を推奨します（詳細はガイド参照）。
- **PR 前にチェック**：ローカルの pre-commit が通り、CI が green であることを確認してください。
- **コミットメッセージ**：Conventional Commits を推奨します（例はガイド参照）。

## サポート

本リポジトリを利用中に問題が発生した場合は、お気軽にご連絡ください。[mason@limxdynamics.com](mason@limxdynamics.com) と [wayne@limxdynamics.com](wayne@limxdynamics.com) まで直接お問い合わせいただくか、GitHub の issue からヘルプを依頼できます。

## 🙏 引用・謝辞

FluxVLA を研究やプロジェクトで利用した場合は、以下の文献を引用してください：

```bibtex
@software{FluxVLA2026,
  author  = {Li, Yinhao and Mao, Weixin and Lan, Zihan and Rong, Jikun and Zhu, Minzhao and Mao, Yiming and Shen, Bowen and Huang, Xu},
  title   = {{FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence}},
  year    = {2026},
  month   = apr,
  version = {1.0.0},
  doi     = {10.5281/zenodo.20049506},
  url     = {https://github.com/FluxVLA/FluxVLA},
  license = {Apache-2.0},
}

@InProceedings{Mao_2026_CVPR,
    author    = {Mao, Yiming and Yu, Zixi and Mao, Weixin and Li, Yinhao and Hu, Qirui and Lan, Zihan and Zhu, Minzhao and Chen, Hua},
    title     = {ARM: Advantage Reward Modeling for Long-Horizon Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {4468-4477}
}
```

**謝辞:** 本プロジェクトは、以下のオープンソースプロジェクトおよびコミュニティの活動から恩恵を受けています。心より感謝いたします：[LeRobot](https://github.com/huggingface/lerobot)、[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/main)、[DreamZero](https://arxiv.org/abs/2602.15922)（[code](https://github.com/dreamzero0/dreamzero)）、[OpenVLA](https://github.com/openvla/openvla)、[OpenPI (pi0)](https://github.com/Physical-Intelligence/openpi)、[LLaVA](https://github.com/haotian-liu/LLaVA)、[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)、[Qwen](https://github.com/QwenLM)、[Triton](https://github.com/triton-lang/triton)、[RTC](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)、[Training RTC](https://arxiv.org/pdf/2512.05964)、[Realtime-VLA](https://github.com/Dexmal/realtime-vla)。もし謝辞に漏れがありましたら、issue または pull request でお知らせください。適切に謝辞へ反映します。

## ロードマップ

- さらに多くの視覚バックボーンネットワークをサポート。
- さらに多くの VLM バックボーンをサポート。
- さらに多くの VLA 手法をサポート。
- VLM データ、または推論チェーン（CoT）データを用いた学習に対応。
- logger 機能を完全実装。
- Isaac Sim に対応。
