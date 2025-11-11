#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


def sha256(path: Path) -> str:
    """Compute SHA256 for a file path."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Record data provenance and checksums.")
    ap.add_argument("--config", required=True, help="configs/paths.yml")
    ap.add_argument("--manifest", required=True, help="out/interim/manifest_data.json")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    cfg_path = Path(args.config)
    manifest_path = Path(args.manifest)

    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())
    raw = Path(cfg["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    Path("out/interim").mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "time": time.time(),
        "git": os.popen("git rev-parse --short HEAD").read().strip(),
        "sources": cfg.get("sources", {}),
        "files": [],  # type: List[Dict[str, str]]
    }

    for sub in ["ukb_rbc_gwas", "smr_curated", "k562_gwps", "hct116_gwps"]:
        d = raw / sub
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*"):
            if p.is_file():
                manifest["files"].append({"path": str(p), "sha256": sha256(p)})

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
