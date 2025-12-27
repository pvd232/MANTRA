import json
import scanpy as sc
import re

def build_class_map():
    # Load raw regulators
    ad = sc.read_h5ad("data/raw/k562_gwps.h5ad", backed="r")
    regs = ad.obs["gene"].unique().tolist()
    
    class_map = {}
    
    # Common family prefixes/patterns
    families = {
        "ZNF": "Zinc Finger",
        "SLC": "Solute Carrier",
        "PDE": "Phosphodiesterase",
        "FOX": "Forkhead Box",
        "HOX": "Homeobox",
        "KIF": "Kinesin",
        "RBM": "RNA Binding Motif",
        "NUP": "Nucleoporin",
        "FAN": "Fanconi Anemia",
        "TMEM": "Transmembrane Protein",
        "TEAD": "TEA Domain",
        "MARS": "Aminoacyl-tRNA Synthetase",
        "UPF": "Up-frameshift Protein",
        "RPS": "Ribosomal Protein Small",
        "RPL": "Ribosomal Protein Large",
        "ATP": "ATPase",
        "GAB": "GABA Receptor",
        "WNT": "Wnt Signaling",
        "KDM": "Lysine Demethylase",
        "TET": "Ten-eleven Translocation",
        "SMAD": "SMAD Family",
        "AKT": "AKT Kinase",
        "MAPK": "MAP Kinase",
        "CYP": "Cytochrome P450",
    }
    
    for r in regs:
        assigned = False
        for prefix, family_name in families.items():
            if r.upper().startswith(prefix):
                class_map[r] = family_name
                assigned = True
                break
        
        if not assigned:
            # Fallback for generic categorization or singletons
            # Look for common numeric suffixes as a proxy for family membership
            match = re.search(r'^([A-Z]+)', r.upper())
            if match:
                root = match.group(1)
                if len(root) >= 3:
                     class_map[r] = root
                else:
                     class_map[r] = "UNKNOWN"
            else:
                class_map[r] = "UNKNOWN"

    output_path = "/home/machina/MANTRA/experiments/nexus/v3_functional_slotting/models/regulator_to_class.json"
    with open(output_path, 'w') as f:
        json.dump(class_map, f, indent=2)
    
    print(f"Mapped {len(class_map)} regulators to {len(set(class_map.values()))} functional classes.")
    print(f"Mapping saved to {output_path}")

if __name__ == "__main__":
    build_class_map()
