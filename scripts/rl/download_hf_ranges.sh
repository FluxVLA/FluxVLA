#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 URL OUTPUT TOTAL_BYTES EXPECTED_SHA256 [PARALLEL=4] [CHUNK_BYTES=67108864]" >&2
  exit 2
fi

url=$1
output=$2
total_bytes=$3
expected_sha=$4
parallel=${5:-4}
chunk_bytes=${6:-67108864}
chunk_dir="${output}.ranges"

mkdir -p "$chunk_dir"

chunk_count=$(((total_bytes + chunk_bytes - 1) / chunk_bytes))

fetch_chunk() {
  local idx=$1
  local start=$((idx * chunk_bytes))
  local end=$((start + chunk_bytes - 1))
  local expected=$chunk_bytes
  local path tmp got attempt

  if ((end >= total_bytes)); then
    end=$((total_bytes - 1))
    expected=$((end - start + 1))
  fi

  path=$(printf '%s/chunk-%05d' "$chunk_dir" "$idx")
  if [[ -f "$path" ]] && [[ $(stat -c %s "$path") -eq $expected ]]; then
    echo "skip chunk=$idx bytes=$expected"
    return 0
  fi

  if [[ -e "$path" ]]; then
    mv "$path" "${path}.incomplete.$(date +%s)"
  fi

  # curl's built-in retry tries to truncate the same output file.  Some OSS
  # FUSE mounts reject that operation, so each retry gets a fresh sequential
  # file instead.
  for attempt in $(seq 1 20); do
    tmp="${path}.tmp.$$.${attempt}"
    if curl --http1.1 -L --fail \
      --speed-time 120 --speed-limit 1024 --max-time 1800 \
      --range "${start}-${end}" --output "$tmp" --silent --show-error "$url"; then
      got=$(stat -c %s "$tmp")
      if [[ $got -eq $expected ]]; then
        mv "$tmp" "$path"
        echo "done chunk=$idx/$((chunk_count - 1)) bytes=$got attempt=$attempt"
        return 0
      fi
      mv "$tmp" "${path}.bad-size-${got}.attempt-${attempt}"
      echo "retry chunk=$idx expected=$expected got=$got attempt=$attempt" >&2
    elif [[ -e "$tmp" ]]; then
      got=$(stat -c %s "$tmp")
      mv "$tmp" "${path}.curl-failed-${got}.attempt-${attempt}"
      echo "retry chunk=$idx curl_failed bytes=$got attempt=$attempt" >&2
    fi
    sleep 2
  done
  echo "chunk=$idx failed after 20 attempts" >&2
  return 1
}

export url output total_bytes expected_sha parallel chunk_bytes chunk_dir chunk_count
export -f fetch_chunk

seq 0 $((chunk_count - 1)) | xargs -P "$parallel" -n 1 bash -c 'fetch_chunk "$1"' _

for idx in $(seq 0 $((chunk_count - 1))); do
  start=$((idx * chunk_bytes))
  end=$((start + chunk_bytes - 1))
  expected=$chunk_bytes
  if ((end >= total_bytes)); then
    end=$((total_bytes - 1))
    expected=$((end - start + 1))
  fi
  path=$(printf '%s/chunk-%05d' "$chunk_dir" "$idx")
  [[ -f "$path" ]] && [[ $(stat -c %s "$path") -eq $expected ]] || {
    echo "missing or invalid chunk: $path" >&2
    exit 1
  }
done

assembled="${output}.assembling.$(date +%s)"
for idx in $(seq 0 $((chunk_count - 1))); do
  path=$(printf '%s/chunk-%05d' "$chunk_dir" "$idx")
  dd if="$path" of="$assembled" oflag=append conv=notrunc status=none
done

actual_bytes=$(stat -c %s "$assembled")
if [[ $actual_bytes -ne $total_bytes ]]; then
  echo "assembled size mismatch: expected=$total_bytes got=$actual_bytes" >&2
  exit 1
fi

actual_sha=$(sha256sum "$assembled" | awk '{print $1}')
if [[ "$actual_sha" != "$expected_sha" ]]; then
  mv "$assembled" "${assembled}.bad-sha-${actual_sha}"
  echo "sha256 mismatch: expected=$expected_sha got=$actual_sha" >&2
  exit 1
fi

if [[ -e "$output" ]]; then
  mv "$output" "${output}.pre-range-download.$(date +%s)"
fi
mv "$assembled" "$output"
echo "verified output=$output bytes=$actual_bytes sha256=$actual_sha"
