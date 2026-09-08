import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate all feat-grid CSV files into unified train/test files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing feat-grid-union_*.csv and feat-grid-intersect_*.csv files"
    )
    return parser.parse_args()

args = parse_args()
input_dir = Path(args.input_dir)

if not input_dir.exists():
    print(f"❌ Erreur: le répertoire '{input_dir}' n'existe pas!")
    exit(1)

print(f"📁 Dossier d'entrée: {input_dir.absolute()}")

# -------------------------
# CONCAT UNION FILES
# -------------------------
union_files = sorted(input_dir.glob("feat-grid-union_*.csv"))

print(f"{len(union_files)} union files found")

dfs_union = []
for f in union_files:
    print("loading", f.name)
    df = pd.read_csv(f)
    dfs_union.append(df)

merged_union = pd.concat(dfs_union, ignore_index=True)

print("union rows:", len(merged_union))

output_union = input_dir / "feat-grid-all-union.csv"
merged_union.to_csv(output_union, index=False)
print(f"✅ Sauvegardé: {output_union}")


# -------------------------
# CONCAT INTERSECT FILES
# -------------------------
intersect_files = sorted(input_dir.glob("feat-grid-intersect_*.csv"))

print(f"{len(intersect_files)} intersect files found")

dfs_intersect = []
for f in intersect_files:
    print("loading", f.name)
    df = pd.read_csv(f)
    dfs_intersect.append(df)

merged_intersect = pd.concat(dfs_intersect, ignore_index=True)

print("intersect rows:", len(merged_intersect))

output_intersect = input_dir / "feat-grid-all-intersect.csv"
merged_intersect.to_csv(output_intersect, index=False)
print(f"✅ Sauvegardé: {output_intersect}")

print("\n✨ Terminé!")