"""
Academic Graph Utilities and Styling Standards
Ensures clean, publication-ready figures with consistent color palettes, typography, and 300 DPI exports.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
OUTPUT_BASE = WORKSPACE_ROOT / "results" / "graph_analysis"
ARTIFACTS_DIR = Path(r"C:\Users\LIKITHA\.gemini\antigravity\brain\b6f19db9-20d1-43e1-a7a8-2b5fb123f340\graphs")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Academic Color Palette
PALETTE = {
    "primary": "#1f77b4",       # Deep Blue
    "secondary": "#2ca02c",     # Clinical Green
    "accent": "#d62728",        # Crimson Red
    "warning": "#ff7f0e",       # Amber Orange
    "purple": "#9467bd",        # Royal Purple
    "slate": "#7f7f7f",         # Neutral Gray
    "teal": "#17becf",          # Cyan Teal
    "anemia": "#e74c3c",        # Soft Red for Anemia
    "non_anemia": "#2ecc71"     # Soft Green for Non-Anemia
}

def set_academic_style():
    """Sets publication-grade matplotlib style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.autolayout': True
    })

def save_academic_figure(fig, filename: str, phase_dir: Path):
    """Saves figure at 300 DPI PNG and also copies to artifacts directory for inline rendering."""
    phase_dir.mkdir(parents=True, exist_ok=True)
    out_png = phase_dir / f"{filename}.png"
    out_pdf = phase_dir / f"{filename}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    
    # Also save to conversation artifacts directory for inline rendering preview
    artifact_png = ARTIFACTS_DIR / f"{filename}.png"
    fig.savefig(artifact_png, dpi=200, bbox_inches="tight")
    
    plt.close(fig)
    print(f"Saved: {out_png}")
    return out_png
