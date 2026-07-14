#!/usr/bin/env python3
"""Prepare or verify GR00T N1.7 LIBERO assets for FluxVLA."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


SUITES = {
    "libero_10": "libero_10_no_noops_lerobotv2.1",
    "libero_goal": "libero_goal_no_noops_lerobotv2.1",
    "libero_object": "libero_object_no_noops_lerobotv2.1",
    "libero_spatial": "libero_spatial_no_noops_lerobotv2.1",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_section(title: str) -> None:
    print(f"\n== {title} ==")


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _find_files(root: Path, patterns: Iterable[str]) -> bool:
    if not root.exists():
        return False
    for pattern in patterns:
        if any(root.glob(pattern)):
            return True
    return False


def _check_path(name: str, path: Path, required: Iterable[str]) -> CheckResult:
    missing = [item for item in required if not (path / item).exists()]
    if missing:
        return CheckResult(name, False, f"{path} missing {missing}")
    return CheckResult(name, True, str(path))


def _check_module(module: str, attr: str | None = None) -> CheckResult:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        if attr is not None:
            getattr(mod, attr)
        return CheckResult(module, True, f"version={version}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(module, False, f"{type(exc).__name__}: {exc}")


def _print_results(results: Iterable[CheckResult]) -> bool:
    all_ok = True
    for result in results:
        status = "OK" if result.ok else "MISSING"
        print(f"[{status}] {result.name}: {result.detail}")
        all_ok = all_ok and result.ok
    return all_ok


def _download_model(repo_id: str, local_dir: Path, dry_run: bool) -> None:
    _run([
        "huggingface-cli",
        "download",
        repo_id,
        "--local-dir",
        str(local_dir),
    ], dry_run=dry_run)


def _download_repo_subdir(repo_id: str,
                          include: str,
                          local_dir: Path,
                          repo_type: str | None,
                          dry_run: bool) -> None:
    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--include",
        include,
        "--local-dir",
        str(local_dir),
    ]
    if repo_type is not None:
        cmd.extend(["--repo-type", repo_type])
    _run(cmd, dry_run=dry_run)


def _download_assets(args, suites: list[str]) -> None:
    _print_section("Download")
    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    _download_model("nvidia/GR00T-N1.7-3B",
                    ckpt_dir / "GR00T-N1.7-3B",
                    args.dry_run)
    _download_model("nvidia/Cosmos-Reason2-2B",
                    ckpt_dir / "nvidia" / "Cosmos-Reason2-2B",
                    args.dry_run)
    for suite in suites:
        _download_repo_subdir("nvidia/GR00T-N1.7-LIBERO",
                              f"{suite}/*",
                              ckpt_dir / "GR00T-N1.7-LIBERO",
                              None,
                              args.dry_run)
        dataset_name = SUITES[suite]
        _download_repo_subdir("limxdynamics/FluxVLAData",
                              f"{dataset_name}/*",
                              data_dir,
                              "dataset",
                              args.dry_run)


def _check_assets(args, suites: list[str]) -> bool:
    _print_section("Environment")
    env_results = [
        _check_module("torch"),
        _check_module("transformers", "Qwen3VLProcessor"),
        _check_module("mmengine"),
        _check_module("datasets"),
        _check_module("safetensors"),
        _check_module("huggingface_hub"),
        _check_module("gymnasium"),
        _check_module("mujoco"),
        _check_module("robosuite"),
        _check_module("libero"),
        _check_module("tensorflow"),
    ]
    env_ok = _print_results(env_results)

    _print_section("Checkpoints")
    ckpt_results = [
        _check_path("GR00T-N1.7-3B",
                    args.ckpt_dir / "GR00T-N1.7-3B",
                    ["config.json", "processor_config.json",
                     "statistics.json"]),
        _check_path("Cosmos-Reason2-2B",
                    args.ckpt_dir / "nvidia" / "Cosmos-Reason2-2B",
                    ["config.json"]),
    ]
    for suite in suites:
        ckpt_results.append(
            _check_path(f"GR00T-N1.7-LIBERO/{suite}",
                        args.ckpt_dir / "GR00T-N1.7-LIBERO" / suite,
                        ["processor_config.json", "statistics.json"]))
    ckpt_ok = _print_results(ckpt_results)

    _print_section("Datasets")
    data_results = []
    for suite in suites:
        dataset_root = args.data_dir / SUITES[suite]
        result = _check_path(
            SUITES[suite],
            dataset_root,
            ["meta/info.json", "meta/episodes_stats.jsonl",
             "meta/tasks.jsonl", "meta/episodes.jsonl", "data"])
        if result.ok and not _find_files(dataset_root / "data", ["*.parquet",
                                                                  "**/*.parquet"]):
            result = CheckResult(result.name, False,
                                 f"{dataset_root}/data has no parquet files")
        data_results.append(result)
    data_ok = _print_results(data_results)

    _print_section("Next Commands")
    for suite in suites:
        config = f"configs/gr00t/gr00t_n17_native_{suite}_full_finetune.py"
        work_dir = f"work_dirs/groot_n17_native_{suite}_full"
        print(f"# Train {suite}")
        print("torchrun --standalone --nnodes=1 --nproc-per-node=8 "
              f"scripts/train.py --config {config} --work-dir {work_dir}")
        print(f"# Eval {suite}")
        print(f"python scripts/eval.py --config {config} "
              f"--ckpt-path {work_dir}/checkpoints/<checkpoint>.safetensors")
    return env_ok and ckpt_ok and data_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare/check GR00T N1.7 LIBERO FluxVLA assets.")
    parser.add_argument("--suite",
                        choices=sorted(SUITES),
                        action="append",
                        help="Suite to check/download. Repeatable.")
    parser.add_argument("--all-suites",
                        action="store_true",
                        help="Use all four LIBERO suites.")
    parser.add_argument("--data-dir",
                        type=Path,
                        default=_repo_root() / "datasets",
                        help="FluxVLA dataset directory.")
    parser.add_argument("--ckpt-dir",
                        type=Path,
                        default=_repo_root() / "checkpoints",
                        help="FluxVLA checkpoint directory.")
    parser.add_argument("--download",
                        action="store_true",
                        help="Download missing public HF assets first.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print download commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (shutil.which("huggingface-cli") is None and args.download
            and not args.dry_run):
        print("[ERROR] huggingface-cli not found. Install huggingface_hub "
              "or run scripts/install_env.sh first.", file=sys.stderr)
        return 2
    suites = sorted(SUITES) if args.all_suites or not args.suite else args.suite
    args.data_dir = args.data_dir.expanduser().resolve()
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
    print(f"[INFO] data_dir={args.data_dir}")
    print(f"[INFO] ckpt_dir={args.ckpt_dir}")
    print(f"[INFO] suites={','.join(suites)}")
    if args.download:
        _download_assets(args, suites)
    ok = _check_assets(args, suites)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
