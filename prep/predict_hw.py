"""READ-ONLY: job status + live calibration of the pair used on ibm_marrakesh, then predict
Gamma_hw and amplitude for the 'trotter' and 'compressed' variants BEFORE the data comes back.
Submits nothing."""
import json
import sys
from pathlib import Path

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

nb_dir = Path(sys.argv[1])
jobs = json.loads((nb_dir / "hardware" / "jobs.json").read_text())

# dt / steps actually used by the agent notebook (parse assignments only, execute nothing)
src = "".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
              for c in json.load(open(nb_dir / "nmr_work_agent.ipynb"))["cells"] if c["cell_type"] == "code")
vals = {}
for line in src.splitlines():
    s = line.strip()
    for key in ("trotter_timestep", "number_of_trottersteps"):
        if s.startswith(key) and "=" in s and "==" not in s:
            try:
                vals[key] = float(eval(s.split("=", 1)[1].split("#")[0], {}, {}))
            except Exception:
                pass
print("notebook settings:", vals)
dt = vals.get("trotter_timestep", 0.0025)
steps = int(vals.get("number_of_trottersteps", 300))

service = QiskitRuntimeService()
for v, info in jobs.items():
    job = service.job(info["job_id"])
    print(f"job {v}: {info['job_id']} status={job.status()}")

be = service.backend(jobs["trotter"]["backend"])
pair = tuple(jobs["trotter"]["pair"])
tg = be.target
props = be.properties()
cz_key = pair if pair in tg["cz"] else pair[::-1]
e_cz = tg["cz"][cz_key].error
d_cz = tg["cz"][cz_key].duration
e_sx = np.mean([tg["sx"][(q,)].error for q in pair])
d_sx = np.mean([tg["sx"][(q,)].duration for q in pair])
ro = [props.readout_error(q) for q in pair]
t1 = [props.t1(q) for q in pair]
t2 = [props.t2(q) for q in pair]
print(f"\n{be.name} pair {pair}: CZ err {e_cz:.2e} ({d_cz*1e9:.0f} ns) | sx err {e_sx:.1e} ({d_sx*1e9:.0f} ns) | "
      f"readout {np.round(ro,4)} | T1 {np.round(np.array(t1)*1e6)} us | T2 {np.round(np.array(t2)*1e6)} us")

# ---- predictions (simple, stated model) ----
N_CZ_STEP = 3          # from the hardware session's own description of the trotter variant
N_SX_STEP = 8          # ~ typical single-qubit sx count per step for a generic 2q block (assumption)
lam_cz = 4 / 3 * e_cz  # depolarising parameter matching the average gate error
lam_sx = 2 * e_sx
tau_step = N_CZ_STEP * d_cz + N_SX_STEP / 2 * d_sx  # rough serial duration of one step
g_gate = (N_CZ_STEP * lam_cz + N_SX_STEP / 2 * lam_sx) / dt
g_idle = tau_step / np.mean(t2) / dt
amp_readout = 1 - 2 * np.mean(ro)
print(f"\nPREDICTION (model: depolarising gates + T2 dephasing during the step; dt={dt*1e3:.2f} ms):")
print(f"  trotter   : Gamma_hw ≈ {g_gate:.1f} (gates) + {g_idle:.1f} (T2 idle) = {g_gate+g_idle:.1f} s^-1 ; "
      f"amplitude A ≈ 1-2*readout = {amp_readout:.3f}")
print(f"  compressed: Gamma_hw ≈ 0 s^-1 (fixed <=3 CZ per circuit) ; amplitude A ≈ "
      f"{amp_readout*(1-lam_cz)**3:.3f} (readout x 3 CZ)")
print(f"  physical reference: Gamma_exp ≈ pi*1 Hz = {np.pi:.2f} s^-1 ; C resolution threshold Gamma_res ≈ 20-43 s^-1")
print(f"  depth of last trotter circuit: {N_CZ_STEP*(steps-1)} CZ, duration ≈ {tau_step*(steps-1)*1e6:.0f} us "
      f"vs T2 {np.mean(t2)*1e6:.0f} us")
