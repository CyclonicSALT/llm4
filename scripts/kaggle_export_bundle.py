#!/usr/bin/env python3
"""
Run after training on Kaggle. Bundles notebook, config, outputs (scores, reports),
and data — EXCLUDES full model dirs to avoid "No space left on device".
Download model folders separately from Output tab if needed.
"""
import os
import zipfile
from pathlib import Path

work = Path("/kaggle/working")
out_zip = work / "kaggle_export_light.zip"

# Paths/dirs to EXCLUDE (they fill disk and cause OSError 28)
EXCLUDE_DIRS = {
    "models",           # full saved models
    "checkpoints",      # training checkpoints
    "input_data",       # we already have this in /kaggle/input
    ".virtual_documents",
    "__pycache__",
    ".git",
}
EXCLUDE_SUFFIXES = (".zip", ".bin", ".safetensors")  # skip large weights inside any included dir

def add_to_archive(zip_path: str, base_dir: Path) -> None:
    base_dir = Path(base_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_dir):
            root = Path(root)
            rel = root.relative_to(base_dir)
            # Don't descend into excluded dirs
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                if f.endswith(".zip") or f.endswith(EXCLUDE_SUFFIXES):
                    continue
                path = root / f
                try:
                    arcname = path.relative_to(base_dir)
                except ValueError:
                    continue
                zf.write(path, arcname)
                print(".", end="", flush=True)

def main():
    # 1. Optional: copy notebook by reading (no extra copy on disk)
    nb_src = None
    for nb in work.rglob("*.ipynb"):
        if "kaggle_export" not in str(nb) and ".virtual_documents" not in str(nb):
            nb_src = nb
            break
    if nb_src and nb_src.exists():
        dest_nb = work / "notebook_export.ipynb"
        try:
            dest_nb.write_bytes(nb_src.read_bytes())
        except OSError as e:
            print("Skip notebook copy (no space):", e)

    # 2. Zip only light artifacts (no models/checkpoints)
    add_to_archive(str(out_zip), work)
    print()
    print("Done. Download:", out_zip.name)
    print("Contains: notebook (if copied), output/*.json, data/*.jsonl, config, code — NOT full models.")
    print("To keep models: download models/ from Output tab before session ends, or use Kaggle 'Save Version'.")

if __name__ == "__main__":
    main()
