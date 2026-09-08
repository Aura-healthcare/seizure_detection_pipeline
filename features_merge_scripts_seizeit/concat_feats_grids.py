#!/usr/bin/env python3
"""
Concatène tous les feat-grid CSV par run (produits par merge_feat_grid.py)
en deux fichiers uniques pour tout le dataset SeizeIT2.

Cherche dans --input-dir les fichiers feat-grid-union_*.csv et
feat-grid-intersect_*.csv (un par run) et écrit :
  - feat-grid-all-union.csv
  - feat-grid-all-intersect.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    """Déclare et parse les arguments CLI du script."""
    parser = argparse.ArgumentParser(
        description="Concatène tous les fichiers feat-grid en deux fichiers unifiés (union / intersect)."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dossier contenant les fichiers feat-grid-union_*.csv et feat-grid-intersect_*.csv"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie pour les fichiers concaténés (par défaut : identique à --input-dir)"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        metavar="PATIENT_ID",
        help="Liste des patients à concaténer, ex: sub-001 sub-002 (par défaut : tous les patients trouvés)"
    )
    return parser.parse_args()


def matches_patients(run_id: str, patients: list[str] | None) -> bool:
    """Indique si run_id (ex: "sub-001_ses-01_..._run-01") appartient à l'un des
    patients demandés. Le patient_id est le préfixe du run id, avant le premier
    underscore. Si patients vaut None, tous les fichiers sont acceptés.
    """
    if patients is None:
        return True
    return run_id.split("_")[0] in patients


def concat_files(
    input_dir: Path,
    output_dir: Path,
    prefix: str,
    output_name: str,
    label: str,
    patients: list[str] | None,
) -> None:
    """Concatène les fichiers "{prefix}{run_id}.csv" dans un seul CSV output_name.

    Si patients est fourni, seuls les fichiers dont le run id appartient à
    l'un des patients demandés sont concaténés.
    label sert uniquement à rendre les messages affichés plus lisibles
    (ex: "union" ou "intersect").
    """
    files = sorted(
        f for f in input_dir.glob(f"{prefix}*.csv")
        if matches_patients(f.stem[len(prefix):], patients)
    )
    print(f"{len(files)} fichiers {label} trouvés")

    if not files:
        print(f"⚠️  Aucun fichier {label} à concaténer, on passe.")
        return

    dfs = []
    for f in files:
        print("chargement de", f.name)
        dfs.append(pd.read_csv(f))

    merged = pd.concat(dfs, ignore_index=True)
    print(f"lignes {label} :", len(merged))

    output_path = output_dir / output_name
    merged.to_csv(output_path, index=False)
    print(f"✅ Sauvegardé : {output_path}")


def main():
    """Point d'entrée : concatène séparément les fichiers union puis intersect."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"❌ Erreur : le répertoire '{input_dir}' n'existe pas !")
        exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Dossier d'entrée : {input_dir.absolute()}")
    print(f"📁 Dossier de sortie : {output_dir.absolute()}")
    if args.patients:
        print(f"👤 Patients sélectionnés : {', '.join(args.patients)}")
    else:
        print("👤 Patients sélectionnés : tous")

    # Un run peut apparaître dans l'union mais pas dans l'intersection (ex:
    # run sans fichier ACC), donc les deux listes de fichiers sont traitées
    # indépendamment plutôt que supposées identiques.
    concat_files(input_dir, output_dir, "feat-grid-union_", "feat-grid-all-union.csv", "union", args.patients)
    concat_files(input_dir, output_dir, "feat-grid-intersect_", "feat-grid-all-intersect.csv", "intersect", args.patients)

    print("\n✨ Terminé !")


if __name__ == "__main__":
    main()
