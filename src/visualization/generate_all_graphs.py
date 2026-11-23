"""
Master Orchestrator: Generate and Validate All Project Graphs (Phases 1-4)
Executes phase-wise plotting modules, verifies outputs at 300 DPI, and exports inventory summary.
"""

import os
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pandas as pd

from src.visualization.phase1_graphs import generate_phase1_graphs
from src.visualization.phase2_graphs import generate_phase2_graphs
from src.visualization.phase3_graphs import generate_phase3_graphs
from src.visualization.phase4_graphs import generate_phase4_graphs
from src.visualization.graph_utils import OUTPUT_BASE

REPORTS_DIR = OUTPUT_BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("STARTING COMPLETE GRAPH GENERATION PIPELINE (PHASES 1-4)")
    print("=" * 80)
    t0 = time.time()

    # Phase 1: 5 graphs
    generate_phase1_graphs()

    # Phase 2: 6 graphs
    generate_phase2_graphs()

    # Phase 3: 9 graphs
    generate_phase3_graphs()

    # Phase 4: 10 graphs
    generate_phase4_graphs()

    total_time = time.time() - t0
    print(f"\nAll graphs successfully generated in {total_time:.2f} seconds!")

    # Verify all generated PNG files
    summary_records = []
    for phase_num in range(1, 5):
        p_dir = OUTPUT_BASE / f"phase{phase_num}"
        png_files = sorted(list(p_dir.glob("*.png")))
        print(f"Phase {phase_num}: {len(png_files)} graphs generated in {p_dir}")
        for p in png_files:
            summary_records.append({
                "phase": f"Phase {phase_num}",
                "graph_file": p.name,
                "file_size_kb": round(p.stat().st_size / 1024, 2),
                "full_path": str(p)
            })

    df_summary = pd.DataFrame(summary_records)
    csv_path = REPORTS_DIR / "generated_graphs_inventory.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Inventory saved to: {csv_path}")
    print(f"Total Graphs Generated: {len(df_summary)}")


if __name__ == "__main__":
    main()
