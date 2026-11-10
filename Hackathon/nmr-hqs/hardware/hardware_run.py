"""Molecule C on a real IBM device, using the circuits of nmr_work_agent.ipynb.

Run from Hackathon/nmr-hqs:
    python hardware/hardware_run.py prepare            # build + transpile, check locally, no QPU time
    python hardware/hardware_run.py submit compressed  # each time point = one 2-qubit unitary (<= 3 CZ)
    python hardware/hardware_run.py submit trotter     # k Trotter steps really executed (depth grows with time)
    python hardware/hardware_run.py fetch              # download finished results into hardware/

Only the initial state |++> is run: |--> gives the same contribution (Z on every qubit commutes with H)
and |+->, |-+> have total-X eigenvalue 0, so they carry zero weight in the correlation function.
"""
import contextlib, io, json, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
MOLECULE = "molecule C"
SHOTS = 500
JOBS_FILE = HERE / "jobs.json"


def notebook_objects():
    """Sections 1-4 of the notebook for MOLECULE (Hamiltonian, one Trotter step, settings)."""
    nb = json.load(open("nmr_work_agent.ipynb"))
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in nb["cells"]:
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            if cell["cell_type"] != "code":
                continue
            if src.startswith("obs_x"):  # section 5.1 is the last cell needed
                exec(src, ns)
                break
            exec(src.replace('molecule_name = "molecule A"', f'molecule_name = "{MOLECULE}"'), ns)
    assert ns["molecule_name"] == MOLECULE and ns["number_spins"] == 2
    return ns


def build(backend, pair=None):
    """ISA circuits for both variants on one fixed pair of physical qubits."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import generate_preset_pass_manager

    ns = notebook_objects()
    n_steps, step = ns["number_of_trottersteps"], ns["circuit"]
    step_op = Operator(step)

    def logical(k, compressed=True):
        qc = QuantumCircuit(2)
        qc.h([0, 1])  # |++>
        if k:
            if compressed:
                qc.unitary(step_op.power(k), [0, 1])
            else:
                qc.compose(step, inplace=True)
        return qc

    # One fixed pair of neighbouring qubits for everything: lowest CZ error among pairs whose
    # readout error is below 1% and whose T1 and T2 are above 100 microseconds.
    if pair is None:
        props, candidates = backend.properties(), []
        for a, b in backend.coupling_map.get_edges():
            try:
                cz = props.gate_error("cz", (a, b))
                readout = max(props.readout_error(a), props.readout_error(b))
                coherence = min(props.t1(a), props.t1(b), props.t2(a), props.t2(b))
            except Exception:  # pair listed in the other direction, or a property is missing
                continue
            if readout < 0.01 and coherence > 100e-6 and cz < 1:
                candidates.append((cz, [a, b]))
        pair = min(candidates)[1]
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=pair, seed_transpiler=1)

    compressed = pm.run([logical(k) for k in range(n_steps)])

    init_isa = pm.run(logical(0))
    step_only = QuantumCircuit(2)
    step_only.compose(step, inplace=True)
    step_isa = pm.run(step_only)
    assert step_isa.layout.final_index_layout() == list(pair) == init_isa.layout.final_index_layout()
    trotter, current = [], init_isa
    for _ in range(n_steps):
        trotter.append(current)
        current = current.compose(step_isa)  # one more Trotter step, no optimisation across steps

    obs = [ns["obs_x"].apply_layout(init_isa.layout), ns["obs_y"].apply_layout(init_isa.layout)]
    return ns, pair, {"compressed": compressed, "trotter": trotter}, obs


def cz_count(circ):
    return circ.count_ops().get("cz", 0)


def main():
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

    mode = sys.argv[1]
    service = QiskitRuntimeService()
    jobs = json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else {}

    if mode == "fetch":
        for variant, info in jobs.items():
            job = service.job(info["job_id"])
            status = job.status()
            print(variant, info["job_id"], info["backend"], status)
            if str(status) not in ("DONE", "JobStatus.DONE"):
                continue
            result = job.result()
            evs = np.array([pub.data.evs for pub in result])    # (time point, [X_tot, Y_tot])
            stds = np.array([pub.data.stds for pub in result])
            np.savez(HERE / f"{MOLECULE} {variant}.npz", exp_x=evs[:, 0], exp_y=evs[:, 1],
                     std_x=stds[:, 0], std_y=stds[:, 1], pair=info["pair"])
            usage = job.usage() if hasattr(job, "usage") else None
            print("  saved", evs.shape, "| QPU seconds used:", usage)
        return

    backend_name = jobs.get("compressed", {}).get("backend") or (sys.argv[3] if len(sys.argv) > 3 else None)
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    ns, pair, circuits, obs = build(backend, pair=jobs.get("compressed", {}).get("pair"))
    props = backend.properties() if hasattr(backend, "properties") else None
    print(f"backend {backend.name}, physical qubits {pair}")
    if props is not None:
        try:
            print(f"  CZ error on this pair: {props.gate_error('cz', pair):.2e}; "
                  f"readout error: {[round(props.readout_error(q), 4) for q in pair]}; "
                  f"T1 [us]: {[round(props.t1(q) * 1e6) for q in pair]}; T2 [us]: {[round(props.t2(q) * 1e6) for q in pair]}")
        except Exception as error:  # pair may be stored in the other direction
            print("  (pair properties not available:", str(error)[:80], ")")
    for variant, circs in circuits.items():
        counts = [cz_count(c) for c in circs]
        print(f"  {variant}: {len(circs)} circuits, CZ per circuit min {min(counts)}, max {max(counts)}")

    if mode == "prepare":
        # Noise-free local check of the transpiled circuits against the notebook's exact values.
        from qiskit.quantum_info import Operator, Statevector
        step_op = Operator(ns["circuit"])
        state = Statevector.from_label("++")
        exact = []
        for _ in range(ns["number_of_trottersteps"]):
            exact.append([state.expectation_value(ns["obs_x"]).real, state.expectation_value(ns["obs_y"]).real])
            state = state.evolve(step_op)
        exact = np.array(exact)
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator
        check = [0, 1, 7, 50, 299]
        for variant, circs in circuits.items():
            res = AerEstimator().run([(circs[k], obs) for k in check]).result()
            got = np.array([pub.data.evs for pub in res])
            print(f"  local noise-free check, {variant}: max deviation from exact = {np.abs(got - exact[check]).max():.2e}")
        return

    variant = sys.argv[2]
    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = SHOTS
    estimator.options.resilience_level = 0      # raw results: no error mitigation
    estimator.options.max_execution_time = int(sys.argv[4]) if len(sys.argv) > 4 else 200  # hard cap on QPU seconds
    job = estimator.run([(circ, obs) for circ in circuits[variant]])
    jobs[variant] = {"job_id": job.job_id(), "backend": backend.name, "pair": list(pair), "shots": SHOTS}
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))
    print(f"submitted {variant}: job {job.job_id()} on {backend.name}")


if __name__ == "__main__":
    main()
