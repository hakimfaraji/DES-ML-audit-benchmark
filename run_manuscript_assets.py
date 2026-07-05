#!/usr/bin/env python3
"""Build manuscript-ready main-text assets and optional supplementary assets."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRIPTS = REPO / "scripts"
DEFAULT_DATASET = REPO / "data" / "Unified_DES_dataset_GOLD_descriptor_ready_subset.csv"


def run(cmd: list[str], label: str) -> None:
    print("\n" + "=" * 80)
    print(f"[RUN] {label}")
    print(" ".join(str(c) for c in cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=str(REPO), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate manuscript-ready tables and figures")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to frozen GOLD CSV")
    ap.add_argument("--outdir", default="reference_outputs", help="Output directory")
    ap.add_argument("--include-si", action="store_true", help="Also generate supplementary assets that are inexpensive to reproduce")
    args = ap.parse_args()

    py = sys.executable
    out = Path(args.outdir)
    run([py, str(SCRIPTS / "24_build_final_main_assets.py"), "--dataset", args.dataset, "--outdir", str(out / "main_text_assets")],
        "Main-text Tables 1-4 and Figures 2-4")

    if args.include_si:
        run([py, str(SCRIPTS / "18_build_figure_s1_dataset_imbalance.py"), "--input", args.dataset, "--outdir", str(out / "supplementary_figures")],
            "Supplementary Figure S1")
        run([py, str(SCRIPTS / "22_build_figure_s6_viscosity_diagnostics.py"), "--input", args.dataset, "--outdir", str(out / "supplementary_figures")],
            "Supplementary Figure S6")

    print("\n[DONE] Manuscript asset generation complete.")


if __name__ == "__main__":
    main()
