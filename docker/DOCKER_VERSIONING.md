# FluxVLA Docker 版本管理指南

本文档说明统一入口 `build_docker.sh` 的版本管理机制与日常使用方法。

______________________________________________________________________

## 1. 标签策略

| 标签                          | 含义                                                 | 可变性       | 谁来用                         |
| ----------------------------- | ---------------------------------------------------- | ------------ | ------------------------------ |
| `fluxvla:orin-base`           | 基础浮动标签，不带 ROS / 不带 flash-attn             | 每次构建覆盖 | 作为分层构建基座               |
| `fluxvla:orin-fa`             | 带 flash-attn 2.5.5 SM87 的浮动标签，不带 ROS        | 每次构建覆盖 | FlashAttention 推理 / 分层复用 |
| `fluxvla:orin`                | 旧单镜像浮动标签，不带 ROS，含 flash-attn 2.5.5 SM87 | 每次构建覆盖 | 旧部署回滚 / 兼容测试          |
| `fluxvla:orin-ros`            | 带 ROS Noetic 的浮动标签，含 flash-attn 2.5.5 SM87   | 每次构建覆盖 | 真机 ROS                       |
| `fluxvla:orin-ros-fa`         | 带 ROS Noetic + flash-attn 2.5.5 SM87 的浮动标签     | 每次构建覆盖 | 真机 ROS + FlashAttention      |
| `fluxvla:<variant>-<version>` | 不可变发布版（如 `orin-1.0.0` / `orin-ros-1.0.0`）   | **永不覆盖** | 开源发布 / 部署 / 回滚         |

核心心智模型：**浮动标签管"现在"，版本标签管"历史"。**

镜像仍会写入 Git SHA、构建时间等元数据；需要溯源时用 `docker inspect` 查看镜像标签即可，不再额外生成溯源镜像标签。

______________________________________________________________________

## 2. 构建镜像

### 推荐：分层构建

```bash
docker/build_docker.sh
```

默认 target 是 `all`，会把 base、flash-attn wheel、FA 镜像、ROS 镜像和最终组合镜像拆开构建，避免任一层失败时整条链路重编。

单独构建某一层：

```bash
docker/build_docker.sh base
docker/build_docker.sh wheel
docker/build_docker.sh fa
docker/build_docker.sh ros
docker/build_docker.sh ros-fa
```

### 发布构建（带版本号）

确认稳定、达到里程碑后，传入语义版本号：

```bash
docker/build_docker.sh all 1.0.0
```

产出标签：

```
fluxvla:orin-base               # 浮动，最新
fluxvla:orin-base-1.0.0         # 不可变发布版
fluxvla:orin-fa
fluxvla:orin-fa-1.0.0
fluxvla:orin-ros
fluxvla:orin-ros-1.0.0
fluxvla:orin-ros-fa
fluxvla:orin-ros-fa-1.0.0
```

### 版本号怎么递增（SemVer）

| 变更类型                          | 版本号 | 例子              |
| --------------------------------- | ------ | ----------------- |
| 修 bug / 调参数（向后兼容）       | PATCH  | `1.0.0` → `1.0.1` |
| 新增功能（向后兼容）              | MINOR  | `1.0.1` → `1.1.0` |
| 破坏性变更（换 PyTorch 大版本等） | MAJOR  | `1.1.0` → `2.0.0` |

______________________________________________________________________

## 3. Git SHA 元数据

每次构建都会把 Git SHA 写入镜像元数据。如果工作区有**未提交的改动**，SHA 会带 `-dirty` 后缀，提醒该构建包含未提交内容。要发正式版本前，先 commit 让 SHA 干净。

______________________________________________________________________

## 4. 运行容器

`run_docker.sh` 默认跑推荐的完整分层浮动标签。镜像名可被环境变量覆盖：

```bash
IMAGE="${FLUXVLA_IMAGE:-fluxvla:orin-ros-fa}"
```

然后：

```bash
# 日常：跑最新推荐镜像 fluxvla:orin-ros-fa
./run_docker.sh

# 临时跑指定版本（无需改脚本）
FLUXVLA_IMAGE=fluxvla:orin-ros-fa-1.0.0 ./run_docker.sh
```

______________________________________________________________________

## 5. 查看镜像元数据

镜像自带 OCI 标准标签和运行时环境变量，可随时溯源：

```bash
# 查看 OCI 标签（版本、commit、构建时间、基础镜像）
docker inspect fluxvla:orin --format '{{json .Config.Labels}}' | python3 -m json.tool

# 容器内查看版本信息
docker run --rm fluxvla:orin env | grep FLUXVLA
```

输出示例：

```
FLUXVLA_VERSION=1.0.0
FLUXVLA_GIT_SHA=83ce216
FLUXVLA_BUILD_DATE=2026-06-05T10:22:00Z
```

______________________________________________________________________

## 6. 回滚

版本标签不可变，所以回滚就是一行 retag：

```bash
# 把稳定版重新指回浮动标签
docker tag fluxvla:orin-1.0.0 fluxvla:orin
./run_docker.sh   # 立刻回到能跑的状态
```

> 这就是为什么**版本标签绝不能覆盖**——它是你的"后悔药"。

______________________________________________________________________

## 7. 清理（防止磁盘爆掉）

每个镜像 23GB+，需定期清理被覆盖后留下的悬空层。

```bash
# 查看所有 fluxvla 镜像和占用
docker images fluxvla

# 清理悬空镜像（被覆盖后无 tag 的层）
docker image prune -f
```

**保留策略建议**：

| 标签类型                    | 保留数量     |
| --------------------------- | ------------ |
| 版本标签（`orin-1.0.0` 等） | 最近 2-3 个  |
| 浮动标签（`orin`）          | 1 个（最新） |

______________________________________________________________________

## 8. 完整一天工作流示例

```bash
# 早上：拉代码、构建
docker/build_docker.sh                          # → orin-base / orin-fa / orin-ros / orin-ros-fa

# 白天：反复改代码、测试
FLUXVLA_IMAGE=fluxvla:orin-ros-fa ./run_docker.sh
# ... 改 bug ...
docker/build_docker.sh ros-fa                   # 只重建最终组合镜像
FLUXVLA_IMAGE=fluxvla:orin-ros-fa ./run_docker.sh

# 下午：测试通过，冻结版本
docker/build_docker.sh all 1.1.0                # → orin-ros-fa-1.1.0 等版本标签
```

______________________________________________________________________

## 9. 速查表

| 操作           | 命令                                                                    |
| -------------- | ----------------------------------------------------------------------- |
| 推荐分层构建   | `docker/build_docker.sh`                                                |
| 发版构建       | `docker/build_docker.sh all 1.0.0`                                      |
| 运行最新       | `./run_docker.sh`                                                       |
| 运行指定版本   | `FLUXVLA_IMAGE=fluxvla:orin-ros-fa-1.0.0 ./run_docker.sh`               |
| 查看版本元数据 | `docker inspect fluxvla:orin-ros-fa --format '{{json .Config.Labels}}'` |
| 回滚           | `docker tag fluxvla:orin-ros-fa-1.0.0 fluxvla:orin-ros-fa`              |
| 清理悬空镜像   | `docker image prune -f`                                                 |
