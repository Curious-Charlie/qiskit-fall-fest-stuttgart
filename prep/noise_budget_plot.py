"""Pitch figure: hardware decay rate vs what each spectral task tolerates.
Run from the repo root:  python prep/noise_budget_plot.py   -> prep/noise_budget.png
Reads Hackathon/nmr-hqs/hardware/summary.json (read-only) if present to add the measured point."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "Hackathon" / "nmr-hqs" / "hardware" / "summary.json"

# palette (validated, light mode): slots 1-3 of the reference categorical palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BRACKET, LOST = "#c9c8c2", "#ecebe7"

GAMMA_EXP = np.pi  # 1 Hz FWHM: the molecule's own linewidth (reference spectra)
XMIN, XMAX = 1, 3e4
rows = [  # (label, bracket lo, bracket hi)  -- thresholds measured in prep/gamma_sweep_check.py etc.
    ("Molecule A\n0.9 Hz fine structure", 3, 5),
    ("Molecules A / B\nmain multiplets", 20, 43),
    ("Molecule C\n4 lines, J = 7.9 Hz", 20, 43),
    ("Identify the molecule\n(matching task)", 1e4, 1e4),
]
Y = {lab: i for i, (lab, _, _) in enumerate(rows)}

plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(12, 5.6), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for lab, lo, hi in rows:
    y = Y[lab]
    ax.fill_betweenx([y - 0.32, y + 0.32], hi, XMAX, color=LOST, lw=0, zorder=1)
    if hi > lo:
        ax.fill_betweenx([y - 0.32, y + 0.32], lo, hi, color=BRACKET, lw=0, zorder=1)
    else:
        ax.plot([lo, lo], [y - 0.32, y + 0.32], color=BRACKET, lw=3, zorder=1)
txt = dict(fontsize=9.5, color=INK2, va="center")
ax.text(5.6, Y[rows[0][0]] + 0.43, "lost between Γ = 3–5", ha="left", **txt)
ax.text(np.sqrt(20 * 43), Y[rows[1][0]] + 0.43, "lost between Γ = 20–43", ha="center", **txt)
ax.text(np.sqrt(20 * 43), Y[rows[2][0]] + 0.43, "lost between Γ = 20–43", ha="center", **txt)
ax.text(8.5e3, Y[rows[3][0]], "still correct at Γ = 10⁴ →", ha="right", **txt)

# molecule's own linewidth
ax.axvline(GAMMA_EXP, color=INK2, ls=(0, (4, 3)), lw=1.4, zorder=2)
ax.text(GAMMA_EXP * 0.92, -0.42, "molecule's own linewidth\n(T₂ relaxation, 1 Hz): Γ ≈ 3.1 s⁻¹",
        ha="right", va="center", fontsize=9.5, color=INK2)

# predicted: ibm_kingston median, A/B (12 CZ per step): range 11.7 (Γ≈n·λ/dt) to 16.8 (Aer fit, 16 CX)
for lab in (rows[0][0], rows[1][0]):
    y = Y[lab] - 0.12
    ax.plot([11.7, 16.8], [y, y], color=ORANGE, lw=2, zorder=3)
    ax.plot([11.7, 16.8], [y, y], ls="none", marker="s", ms=8, color=ORANGE,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.text(18.5, y, "12–17", color=INK, fontsize=9.5, va="center")
# predicted: ibm_marrakesh best pair, C (3 CZ per step)
yC = Y[rows[2][0]] - 0.12
ax.plot([3.7], [yC], ls="none", marker="o", ms=9, color=BLUE, markeredgecolor=SURFACE,
        markeredgewidth=2, zorder=4)
ax.text(3.7 * 1.12, yC - 0.02, "3.7 (predicted)", color=INK, fontsize=9.5, va="center")

handles = [
    plt.Line2D([], [], ls="none", marker="o", ms=9, color=BLUE, label="Predicted · ibm_marrakesh best pair (3 CZ / step)"),
    plt.Line2D([], [], color=ORANGE, marker="s", ms=8, lw=2, label="Predicted · ibm_kingston median (12 CZ / step)"),
]
# measured (if the hardware session has produced summary.json)
if SUMMARY.exists():
    s = json.loads(SUMMARY.read_text())
    if "trotter" in s:
        g = s["trotter"]["effective_decay_rate_per_s"]
        ge = s["trotter"].get("decay_rate_uncertainty_per_s", 0) or 0
        yM = Y[rows[2][0]] + 0.14
        ax.errorbar([g], [yM], xerr=[[min(ge, g * 0.99)], [ge]], ls="none", marker="D", ms=8, color=AQUA,
                    markeredgecolor=SURFACE, markeredgewidth=2, ecolor=AQUA, elinewidth=2, capsize=3, zorder=5)
        ax.text(g * 1.12 + ge, yM + 0.02, f"{g:.1f} ± {ge:.1f} (measured)", color=INK, fontsize=9.5, va="center")
        handles.append(plt.Line2D([], [], ls="none", marker="D", ms=8, color=AQUA,
                                  label="Measured · ibm_marrakesh, Trotter steps on hardware"))
handles += [
    plt.matplotlib.patches.Patch(color=BRACKET, label="threshold where the feature is lost (measured bracket)"),
    plt.matplotlib.patches.Patch(color=LOST, label="feature lost"),
]

ax.set_xscale("log")
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(-0.75, 3.6)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlabel("Effective decay rate of the simulated signal  Γ  (s⁻¹)      [line width FWHM = Γ / π  Hz]")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.tick_params(axis="y", length=0)
ax.set_title("Noise budget for NMR spectroscopy on a quantum computer", loc="left", fontsize=14,
             color=INK, pad=30, fontweight="bold")
ax.text(0.0, 1.035, "Hardware decay must stay left of the grey zone. Left of the dashed line it hides inside the "
         "molecule's own physical linewidth.", fontsize=10.5, color=INK2, transform=ax.transAxes)
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=9.5)
fig.tight_layout()
out = Path(__file__).with_name("noise_budget.png")
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("saved", out, "| measured point included:", SUMMARY.exists())
