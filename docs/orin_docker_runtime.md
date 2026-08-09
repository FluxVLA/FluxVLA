# FluxVLA Docker Runtime on Jetson Orin

FluxVLA Orin runtime is distributed as a published Docker image. The current branch does not require local Docker builds.

Recommended image:

```bash
docker pull fluxvla/fluxvla:fluxvla-orin-1.0.0
```

## 1. Prepare Storage

The image and model checkpoints are large, so place Docker data and runtime assets on NVMe when possible.

```bash
sudo mkdir -p /mnt/nvme/{checkpoints,datasets,work_dirs}
docker info | grep "Docker Root Dir"
```

If Docker still uses `/var/lib/docker`, migrate it before pulling large images:

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

## 2. Pull the Image

```bash
docker pull fluxvla/fluxvla:fluxvla-orin-1.0.0
docker images fluxvla/fluxvla
```

## 3. Start the Container

Run from the repository root:

```bash
scripts/run_docker.sh
```

`scripts/run_docker.sh` defaults to:

```bash
fluxvla/fluxvla:fluxvla-orin-1.0.0
```

It mounts:

| Host path       | Container path       |
| --------------- | -------------------- |
| repository root | `/workspace/FluxVLA` |
| `/mnt/nvme`     | `/mnt/nvme`          |

The container working directory is `/workspace/FluxVLA`.

To override the image:

```bash
FLUXVLA_IMAGE=<registry>/<repo>:<tag> scripts/run_docker.sh
```

## 4. Run One Command

```bash
scripts/run_docker.sh python3 -c "import torch; print(torch.cuda.is_available())"
```

Choose an attention implementation:

```bash
ATTN_IMPLEMENTATION=flash_attention_2 scripts/run_docker.sh
ATTN_IMPLEMENTATION=eager scripts/run_docker.sh
```

## 5. ROS Runtime Variables

The launcher forwards these variables when they are set on the host:

```bash
ROS_MASTER_URI
ROS_IP
ROS_HOSTNAME
```

Example:

```bash
ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1 scripts/run_docker.sh
```

## 6. Validate CUDA

First validate the NVIDIA container runtime:

```bash
sudo docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.4.0 \
  bash -lc "cat /etc/nv_tegra_release"
```

Then validate PyTorch inside the FluxVLA image:

```bash
scripts/run_docker.sh python3 -c "import torch; print(torch.cuda.is_available())"
```

## 7. Image Metadata

```bash
docker inspect fluxvla/fluxvla:fluxvla-orin-1.0.0 --format '{{json .Config.Labels}}' | python3 -m json.tool
```
