# Jetson Orin 上的 FluxVLA Docker 运行环境

FluxVLA Orin 运行环境通过云端 Docker 镜像发布，当前分支不再需要本地构建 Docker 镜像。

推荐镜像：

```bash
docker pull fluxvla/fluxvla:fluxvla-orin-1.0.0
```

## 1. 准备存储

镜像和模型 checkpoint 都比较大，建议把 Docker 数据目录和运行数据放到 NVMe。

```bash
sudo mkdir -p /mnt/nvme/{checkpoints,datasets,work_dirs}
docker info | grep "Docker Root Dir"
```

如果 Docker 仍使用 `/var/lib/docker`，建议先迁移：

```bash
sudo apt update
sudo apt install -y rsync
sudo systemctl stop docker
sudo mkdir -p /mnt/nvme/docker
sudo rsync -aHAX /var/lib/docker/ /mnt/nvme/docker/

sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "data-root": "/mnt/nvme/docker"
}
EOF

sudo systemctl start docker
docker info | grep "Docker Root Dir"
```

## 2. 拉取镜像

```bash
docker pull fluxvla/fluxvla:fluxvla-orin-1.0.0
docker images fluxvla/fluxvla
```

## 3. 启动容器

在仓库根目录执行：

```bash
scripts/run_docker.sh
```

`scripts/run_docker.sh` 默认使用：

```bash
fluxvla/fluxvla:fluxvla-orin-1.0.0
```

默认挂载：

| 宿主机路径     | 容器路径             |
| -------------- | -------------------- |
| 当前仓库根目录 | `/workspace/FluxVLA` |
| `/mnt/nvme`    | `/mnt/nvme`          |

容器内默认工作目录是 `/workspace/FluxVLA`。

如果要临时指定其它镜像：

```bash
FLUXVLA_IMAGE=<registry>/<repo>:<tag> scripts/run_docker.sh
```

## 4. 直接运行单条命令

```bash
scripts/run_docker.sh python3 -c "import torch; print(torch.cuda.is_available())"
```

选择 attention 实现：

```bash
ATTN_IMPLEMENTATION=flash_attention_2 scripts/run_docker.sh
ATTN_IMPLEMENTATION=eager scripts/run_docker.sh
```

## 5. ROS 环境变量

脚本会自动透传宿主机上的这些变量：

```bash
ROS_MASTER_URI
ROS_IP
ROS_HOSTNAME
```

示例：

```bash
ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1 scripts/run_docker.sh
```

## 6. 验证 CUDA

先验证 NVIDIA container runtime：

```bash
sudo docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.4.0 \
  bash -lc "cat /etc/nv_tegra_release"
```

再验证 FluxVLA 镜像内的 PyTorch：

```bash
scripts/run_docker.sh python3 -c "import torch; print(torch.cuda.is_available())"
```

## 7. 查看镜像信息

```bash
docker inspect fluxvla/fluxvla:fluxvla-orin-1.0.0 --format '{{json .Config.Labels}}' | python3 -m json.tool
```
