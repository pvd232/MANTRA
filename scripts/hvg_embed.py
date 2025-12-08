#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import yaml
import scanpy as sc

from mantra.embeddings import EmbeddingConfig, compute_embeddings


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute manifold embeddings on a QC’d AnnData with HVGs "
            "and store them in .obsm/.uns. Driven by YAML EmbeddingConfig."
        )
    )
    p.add_argument(
        "--params",
        type=str,
        required=True,
        help="YAML params file (must contain an 'embeddings' section)",
    )
    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Input AnnData file (QC’d, HVGs annotated).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output AnnData file (default: overwrite --ad).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    emb_cfg = EmbeddingConfig(**params.get("embeddings", {}))

    in_path = Path(args.ad)
    out_path = Path(args.out) if args.out is not None else in_path

    print(f"[load] Reading AnnData from {in_path} ...", flush=True)
    ad = sc.read_h5ad(in_path)

    # compute embeddings in-place
    ad = compute_embeddings(ad, emb_cfg)

    print(f"[save] Writing updated AnnData with embeddings to {out_path} ...", flush=True)
    ad.write(out_path)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
