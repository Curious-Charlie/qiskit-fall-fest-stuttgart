"""Turn the hardware results into spectra with the notebook's own functions and compare.

Run from Hackathon/nmr-hqs after `python hardware/hardware_run.py fetch`:
    python hardware/analyse.py
"""
import json, pickle, sys, warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from hardware_run import HERE, MOLECULE, notebook_objects  # noqa: E402

warnings.filterwarnings("ignore")
NOISE_STRENGTH = 5.0  # the notebook default
REFERENCE = "spectrum_1"  # molecule C, from the matching task

# Notebook objects, plus its spectrum function and ppm conversion (section 6).
ns = notebook_objects()
for cell in json.load(open("nmr_work_agent.ipynb"))["cells"]:
    src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    if cell["cell_type"] == "code" and src.startswith(("def time_signal_to_spectrum", "# Gyromagnetic ratio")):
        exec(src, ns)
n_steps, dt = ns["number_of_trottersteps"], ns["trotter_timestep"]
times = np.arange(n_steps) * dt

# Noise-free values for the same initial state |++> and the same Trotter step.
from qiskit.quantum_info import Operator, Statevector  # noqa: E402

state, step_op, ideal = Statevector.from_label("++"), Operator(ns["circuit"]), []
for _ in range(n_steps):
    ideal.append([state.expectation_value(ns["obs_x"]).real, state.expectation_value(ns["obs_y"]).real])
    state = state.evolve(step_op)
ideal = np.array(ideal)


def spectrum(exp_x, exp_y, noise=NOISE_STRENGTH):
    """Sections 5.3 and 6 of the notebook for a single initial state."""
    window = np.exp(-noise * times)
    spec, ppm = ns["time_signal_to_spectrum"](
        exp_x[0] * exp_x * window, exp_x[0] * exp_y * window, dt, n_steps,
        Hz_to_ppm_conversion=ns["Hz_to_ppm_conversion"], chem_shift_hz=ns["chemical_shift_hz"],
    )
    return np.asarray(ppm, dtype=float), np.asarray(spec, dtype=float)


def peak_positions(ppm, spec):
    """Peaks that rise at least 20% of the tallest peak above their surroundings.

    The 5% rule of the noise sweep is too low here: with 500 shots the noise floor reaches about 15%.
    """
    return [round(float(v), 3) for v in ppm[find_peaks(spec, prominence=0.2)[0]]]


with open(Path("files") / f"{REFERENCE}.pkl", "rb") as f:
    ref = pickle.load(f)
ref_ppm, ref_spec = np.asarray(ref["frequencies"], float), np.asarray(ref["spectrum"], float)
ref_spec = ref_spec / ref_spec.max()

jobs = json.loads((HERE / "jobs.json").read_text())
variants = [v for v in ("compressed", "trotter") if (HERE / f"{MOLECULE} {v}.npz").exists()]
titles = {
    "compressed": "each time point compiled to one 2-qubit unitary (3 CZ)",
    "trotter": "Trotter steps executed one after another (3 CZ per step, up to 897)",
}
summary = {}
fig, axes = plt.subplots(len(variants), 2, figsize=(14, 4.8 * len(variants)), dpi=100, squeeze=False)
ideal_ppm, ideal_spec = spectrum(ideal[:, 0], ideal[:, 1])
for row, variant in enumerate(variants):
    data = np.load(HERE / f"{MOLECULE} {variant}.npz")
    hw_x, hw_y = data["exp_x"], data["exp_y"]

    # How much of the signal survives, and how well it follows the noise-free one.
    hw, ex = hw_x + 1j * hw_y, ideal[:, 0] + 1j * ideal[:, 1]
    last = slice(2 * n_steps // 3, n_steps)
    retained = float(np.sqrt(np.mean(np.abs(hw[last]) ** 2) / np.mean(np.abs(ex[last]) ** 2)))
    overlap = float(abs(np.vdot(ex, hw)) / (np.linalg.norm(ex) * np.linalg.norm(hw)))

    hw_ppm, hw_spec = spectrum(hw_x, hw_y)
    peaks = peak_positions(hw_ppm, hw_spec)
    summary[variant] = {
        "backend": jobs[variant]["backend"], "physical_qubits": jobs[variant]["pair"], "shots": jobs[variant]["shots"],
        "job_id": jobs[variant]["job_id"],
        "signal_amplitude_in_last_third_relative_to_noise_free": round(retained, 2),
        "overlap_of_time_signal_with_noise_free": round(overlap, 2),
        "peaks_ppm": peaks,
        "noise_free_peaks_ppm": peak_positions(ideal_ppm, ideal_spec),
        "correlation_with_noise_free_spectrum": round(float(np.corrcoef(hw_spec, ideal_spec)[0, 1]), 3),
    }

    ax = axes[row, 0]
    ax.plot(times, ideal[:, 0], color="C1", lw=1, label="Noise-free simulation")
    ax.plot(times, hw_x, color="C2", lw=1, label=jobs[variant]["backend"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\langle X_{tot} \rangle$ for the initial state $|++\rangle$")
    ax.set_title(f"Time signal: {titles[variant]}", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[row, 1]
    ax.plot(ref_ppm, ref_spec, color="C0", label="HQS Spectrum Tools")
    ax.plot(ideal_ppm, ideal_spec, color="C1", ls="--", label="Noise-free simulation")
    ax.plot(hw_ppm, hw_spec, color="C2", label=jobs[variant]["backend"])
    ax.set_xlim(6.2, 6.5)
    ax.set_xlabel("Frequency [ppm]")
    ax.set_ylabel("Spectrum")
    ax.set_title(
        f"Spectrum at noise_strength = {NOISE_STRENGTH:g} s$^{{-1}}$: hardware peaks at "
        + ", ".join(f"{v:.2f}" for v in peaks) + " ppm",
        fontsize=10,
    )
    ax.legend(fontsize=7)

fig.suptitle(f"{MOLECULE} on real hardware against {REFERENCE}", fontsize=12)
fig.tight_layout()
fig.savefig(HERE / f"{MOLECULE} hardware.png", dpi=200, bbox_inches="tight")
(HERE / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
