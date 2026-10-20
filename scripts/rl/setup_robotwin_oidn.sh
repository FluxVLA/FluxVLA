#!/usr/bin/env bash
# OIDN 2.3.3 is the first release with NVIDIA Blackwell support.
set -euo pipefail
RL_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
RL_ROOT=${FLUX_ROBOTWIN_ROOT:-$RL_SOURCE/work_dirs/robotwin}
OIDN_ARCHIVE="$RL_ROOT/assets/oidn-2.3.3.x86_64.linux.tar.gz"
mkdir -p "$RL_ROOT/src" "$RL_ROOT/assets"
if [[ ! -f "$OIDN_ARCHIVE" ]]; then
    aria2c --continue=true --max-connection-per-server=8 --split=8 \
        --min-split-size=1M --console-log-level=warn --summary-interval=30 \
        --dir="$RL_ROOT/assets" --out=oidn-2.3.3.x86_64.linux.tar.gz.aria \
        https://github.com/OpenImageDenoise/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz \
        --max-tries=5 --retry-wait=3
    mv "$OIDN_ARCHIVE.aria" "$OIDN_ARCHIVE"
fi
printf '%s  %s\n' \
    3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f \
    "$OIDN_ARCHIVE" | sha256sum -c -
tar -xzf "$OIDN_ARCHIVE" -C "$RL_ROOT/src"
sha256sum "$OIDN_ARCHIVE" > "$RL_ROOT/assets/oidn-2.3.3.sha256"
printf 'OIDN_READY\n'
