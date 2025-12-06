from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class K562ReplogleGWPS:
    """
    Schema for the Replogle et al. K562 genome-wide Perturb-seq AnnData object.

    This reflects:
      - ad.obs columns
      - ad.var index (gene_id) and columns
      - overall shape and X dtype

    Shape of AnnData object:
        (n_cells, n_genes) = (247_914, 8_749)

    Each row (cell) in the 247_914 rows has:
      - CRISPR perturbation metadata (target gene, sgID_AB, etc.)
      - QC / library metrics (mitopercent, UMI_count, n_counts, ...)

    Each column (gene) in the 8_749 vars has:
      - Ensembl gene_id index
      - gene_name, genomic position, and HVG statistics.
    """

    # ---- Matrix shape / X ----
    n_cells: int = 247_914
    n_genes: int = 8_749

    # Expression matrix X: dense numpy array (e.g. log-normalized counts)
    # Shape: (n_cells, n_genes)
    X: Optional[np.ndarray] = None

    # ---- ad.obs columns (cell metadata) ----

    # 10x GEM group / library batch ID (1..56)
    gem_group: Optional[np.int64] = None

    # Target gene symbol for the perturbation in this cell
    # categorical with ~2.3k levels (e.g. 'AAAS', 'GATA1', ...)
    gene: Optional[str] = None

    # Target gene Ensembl ID for the perturbation
    # categorical with ~2.3k levels (e.g. 'ENSG00000102738', ...)
    gene_id: Optional[str] = None

    # Target transcript ID(s) (Ensembl transcript IDs, sometimes comma-separated)
    # e.g. 'ENST00000248058.1', 'ENST00000263377.2,ENST00000371835.4', ...
    transcript: Optional[str] = None

    # Convenience concatenation: "<idx>_<gene>_P1P2_<gene_id>"
    # e.g. '5261_MRPS31_P1P2_ENSG00000102738'
    gene_transcript: Optional[str] = None

    # sgRNA pair identifier for this perturbation
    # e.g. 'AAAS_-_53715438.23-P1P2|AAAS_+_53715355.23-P1P2'
    sgID_AB: Optional[str] = None

    # Percent of UMIs mapping to mitochondrial genes for this cell
    mitopercent: Optional[np.float32] = None

    # Raw UMI count per cell (pre-scaling)
    UMI_count: Optional[np.float32] = None

    # Z-scored UMI count within gem_group
    z_gemgroup_UMI: Optional[np.float32] = None

    # Per-cell scaling factor used to normalize core-adjusted UMI counts
    core_scale_factor: Optional[np.float32] = None

    # Adjusted UMI count after applying core_scale_factor
    core_adjusted_UMI_count: Optional[np.float32] = None

    # Total counts per cell (Scanpy-style n_counts; may mirror UMI_count)
    n_counts: Optional[np.float32] = None

    # Number of genes detected per cell (non-zero entries in that row)
    n_genes_by_counts: Optional[np.int64] = None

    # ---- ad.var index + columns (gene metadata) ----

    # ad.var_names / index: Ensembl gene IDs
    # e.g. 'ENSG00000188976', ...
    var_gene_id: Optional[str] = None

    # Human-readable gene symbol
    # e.g. 'NOC2L', 'PLEKHN1', 'HES4', ...
    gene_name: Optional[str] = None

    # Chromosome name
    # e.g. 'chr1', 'chr2', ...
    chr: Optional[str] = None

    # Genomic start coordinate (bp)
    start: Optional[np.int64] = None

    # Genomic end coordinate (bp)
    end: Optional[np.int64] = None

    # Ensembl gene class / version string
    # e.g. 'gene_version11', 'gene_version10', ...
    class_: Optional[str] = None  # 'class' is a keyword in Python

    # Strand ('+' or '-')
    strand: Optional[str] = None

    # Genomic length in bp
    length: Optional[np.int64] = None

    # Whether this gene is included in the expression matrix
    in_matrix: Optional[bool] = None

    # Mean expression across cells (raw space used for var stats)
    mean: Optional[np.float32] = None

    # Std dev across cells
    std: Optional[np.float32] = None

    # Coefficient of variation (std / mean)
    cv: Optional[np.float32] = None

    # Fano factor (variance / mean)
    fano: Optional[np.float32] = None

    # Highly variable gene flag from Scanpy HVG selection
    highly_variable: Optional[bool] = None

    # Scanpy HVG stats: means, dispersions, and normalized dispersions
    means: Optional[np.float32] = None
    dispersions: Optional[np.float32] = None
    dispersions_norm: Optional[np.float32] = None

    # ---- ad.uns (unstructured metadata) ----

    # Highly-variable gene params / summary (Scanpy)
    hvg_uns: Optional[Dict] = field(default_factory=dict)

    # Log1p transform parameters (Scanpy)
    log1p_uns: Optional[Dict] = field(default_factory=dict)
