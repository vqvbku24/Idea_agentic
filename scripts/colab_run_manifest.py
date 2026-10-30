from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(root: Path, args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_files(root: Path) -> list[Path]:
    files = sorted((root / "src/pilot").glob("*.py"))
    files.extend(sorted((root / "config").glob("*.yaml")))
    files.append(root / "data/processed/sources_final.json")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and validate a reproducible Colab run.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ref", help="Expected commit or tag checked out by the notebook.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dense-max-length", type=int, default=1024)
    parser.add_argument("--reference-max-lengths", default="1024,8192")
    parser.add_argument("--translation-chunk-tokens", type=int, default=400)
    parser.add_argument("--dense-model", default="BAAI/bge-m3")
    parser.add_argument("--translation-model", default="Helsinki-NLP/opus-mt-fr-en")
    args = parser.parse_args()
    root = args.root.resolve()
    status = run(root, ["status", "--porcelain"])
    if status:
        raise SystemExit("ABORT: working tree is dirty:\n" + status)
    commit = run(root, ["rev-parse", "HEAD"])
    if args.ref and run(root, ["rev-parse", args.ref]) != commit:
        raise SystemExit(f"ABORT: HEAD {commit} does not match expected ref {args.ref}")

    hashes = {}
    mismatches = []
    for path in tracked_files(root):
        relative = path.relative_to(root).as_posix()
        working_hash = sha256(path)
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root)
        commit_hash = hashlib.sha256(committed).hexdigest()
        hashes[relative] = {"working_tree_sha256": working_hash, "commit_sha256": commit_hash}
        if working_hash != commit_hash:
            mismatches.append(relative)
    if mismatches:
        raise SystemExit("ABORT: file hashes differ from HEAD:\n" + "\n".join(mismatches))

    try:
        import torch
        device = {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        device = {"torch": None, "cuda_available": False, "cuda_device": None}
    output = root / "results/real" / f"run_meta_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"head": commit, "ref": args.ref, "status_porcelain": ""},
        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "platform": {"python": platform.python_version(), "system": platform.platform(), "machine": platform.machine()},
        "device": device,
        "seeds": {"selection": args.seed},
        "model_ids": {"dense": args.dense_model, "translation": args.translation_model},
        "max_lengths": {
            "dense": args.dense_max_length,
            "reference": [int(value) for value in args.reference_max_lengths.split(",")],
            "translation_chunk_tokens": args.translation_chunk_tokens,
        },
        "file_hashes": hashes,
        "validation": {"working_tree_clean": True, "hashes_match_head": True},
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
