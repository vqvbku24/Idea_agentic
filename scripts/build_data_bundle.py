from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def artifact_root() -> Path:
    for candidate in (ROOT / "results/real/real", ROOT / "results/real-colab/real", ROOT / "results/real"):
        if (candidate / "papers_x2.jsonl").exists():
            return candidate
    raise FileNotFoundError("papers_x2.jsonl not found in results/real artefact roots")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path, data_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.glob("*_x2.*")):
        if path.is_file() and path.name not in {"run_meta_x2.json", "data_bundle_x2.zip"}:
            files.append(path)
    for name in (
        "papers_x2.jsonl",
        "development_papers_x2.jsonl",
        "french_distractors_native_x2.jsonl",
        "extraction_details.csv",
    ):
        path = root / name
        if path.exists() and path not in files:
            files.append(path)
    english = data_root / "english_distractors_metadata.jsonl"
    if english.exists():
        files.append(english)
    queries = [ROOT / "results/demo/queries.jsonl", root / "queries.jsonl"]
    files.extend(path for path in queries if path.exists() and "human" not in path.parts)
    return sorted(set(path.resolve() for path in files))


def contains_paper_text(path: Path) -> bool:
    if path.name == "queries.jsonl":
        return False
    return path.suffix.lower() == ".jsonl" or path.name in {
        "extraction_review_x2.md",
        "extraction_defects_x2.md",
        "residual_defects_x2.md",
        "leak_fail_cases_x2.md",
    }


def main() -> None:
    root = artifact_root()
    data_root = ROOT / "data/processed"
    output_root = ROOT / "results/real"
    output_root.mkdir(parents=True, exist_ok=True)
    files = collect_files(root, data_root)
    if not files:
        raise RuntimeError("No bundle files found")
    manifest_files = []
    bundle_path = output_root / "data_bundle_x2.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            if path.parent == root:
                relative = (Path("results/real") / path.name).as_posix()
                canonical = output_root / path.name
                if path.resolve() != canonical.resolve():
                    shutil.copy2(path, canonical)
                path = canonical
            else:
                try:
                    relative = path.relative_to(ROOT).as_posix()
                except ValueError:
                    relative = path.name
            bundle.write(path, relative)
            manifest_files.append({
                "path": relative,
                "sha256": sha256(path),
                "contains_paper_text": contains_paper_text(path),
                "size": path.stat().st_size,
            })
    manifest = {
        "schema_version": 1,
        "bundle": bundle_path.name,
        "files": manifest_files,
        "paper_text_files": [item["path"] for item in manifest_files if item["contains_paper_text"]],
    }
    manifest_path = output_root / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact_root": str(root),
        "bundle": str(bundle_path),
        "files": len(files),
        "paper_text_files": len(manifest["paper_text_files"]),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
