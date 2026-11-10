"""Practitioner-level checks for the 2026 HQS NMR prompt.
(1) ibm_kingston (FakeKingston) error rates; (2) 2q gates per Trotter step after transpilation;
(3) Aer depolarising sim: fitted effective decay rate Gamma_eff vs dt; (4) task-based (assignment)
robustness vs Gamma; (5) shot-noise effect."""
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime.fake_provider import FakeKingston
from scipy.signal import find_peaks
from struqture_py.spins import PauliHamiltonian

files = Path(sys.argv[1])
NAMES = ["molecule A", "molecule B", "molecule C"]
GB = 267_522_128.0 * 11.7433
NU0 = GB / (2 * np.pi)


def load(name):
    h = PauliHamiltonian.from_json(json.load(open(files / f"{name}_B11.7433T.json")))
    n = h.current_number_spins()
    zbar = np.mean([float(h.get(f"{i}Z")) for i in range(n)])
    labels, coeffs = [], []
    for key, val in zip(h.keys(), h.values()):
        sites = {int(m.group(1)): m.group(2) for m in re.finditer(r"(\d+)([XYZ])", str(key))}
        p = ["I"] * n
        for s, o in sites.items():
            p[n - 1 - s] = o
        labels.append("".join(p))
        coeffs.append(float(val) - (zbar if len(sites) == 1 else 0.0))
    return SparsePauliOp(labels, coeffs), n, zbar


def step_circuit(H, dt):
    qc = QuantumCircuit(H.num_qubits)
    qc.append(PauliEvolutionGate(H, dt, synthesis=LieTrotter()), range(H.num_qubits))
    return qc


# ---------------------------------------------------------------- (1) hardware numbers
be = FakeKingston()
tg = be.target
twoq = [g for g in tg.operation_names if tg.operation_from_name(g).num_qubits == 2]
e2 = np.array([p.error for p in tg[twoq[0]].values() if p is not None and p.error is not None])
e1 = np.array([p.error for p in tg["sx"].values() if p is not None and p.error is not None])
ro = np.array([p.error for p in tg["measure"].values() if p is not None and p.error is not None])
t2s = np.array([q.t2 for q in tg.qubit_properties if q is not None and q.t2])
print(f"(1) FakeKingston: 2q gate '{twoq[0]}' median err {np.median(e2):.2e} (best 10% {np.percentile(e2,10):.2e}) | "
      f"sx {np.median(e1):.1e} | readout {np.median(ro):.1e} | T2 median {np.median(t2s)*1e6:.0f} us")
E2 = float(np.median(e2))

# ---------------------------------------------------------------- (2) 2q gates per Trotter step
DT = 0.0025
print("\n(2) two-qubit gates per Trotter step (dt=2.5 ms):")
n2q = {}
for name in NAMES:
    H, n, _ = load(name)
    st = step_circuit(H, DT)
    ta = transpile(st, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    tk = transpile(st, be, optimization_level=3, seed_transpiler=7)
    k2 = sum(v for k, v in tk.count_ops().items() if k in twoq)
    n2q[name] = (ta.count_ops().get("cx", 0), k2)
    # whole 300-step evolution for 2-qubit molecule compresses to a single 2q unitary
    extra = ""
    if n == 2:
        full = QuantumCircuit(2)
        for _ in range(300):
            full.compose(st, inplace=True)
        tf = transpile(full, be, optimization_level=3, seed_transpiler=7)
        extra = f" | full 300-step evolution at opt-level 3: {sum(v for k, v in tf.count_ops().items() if k in twoq)} {twoq[0]}"
    print(f"   {name}: all-to-all cx = {ta.count_ops().get('cx',0)}, on ibm_kingston {twoq[0]} = {k2}{extra}")
    print(f"      -> naive Gamma_eff ~ n2q*err/dt = {k2*E2/DT:6.1f} s^-1 ; max depth of notebook default "
          f"(300 steps) = {300*k2} two-qubit gates")

# ---------------------------------------------------------------- (3) Aer: fitted Gamma_eff vs dt
print("\n(3) Aer depolarising (lambda from Kingston median 2q err), molecule A, fitted Gamma_eff:")
H, n, _ = load("molecule A")
lam2, lam1 = 4 / 3 * E2, 2 * float(np.median(e1))
nm = NoiseModel(basis_gates=["rz", "sx", "x", "cx"])
nm.add_all_qubit_quantum_error(depolarizing_error(lam2, 2), ["cx"])
nm.add_all_qubit_quantum_error(depolarizing_error(lam1, 1), ["sx", "x"])
est = AerEstimator(options={"backend_options": {"noise_model": nm, "method": "density_matrix"},
                            "run_options": {"shots": None}, "default_precision": 0})
phi = ParameterVector("phi", n)
init = QuantumCircuit(n)
for q in range(n):
    init.h(q)
    init.rz(phi[q], q)
init_t = transpile(init, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
bits = [b for b in np.ndindex(*(2,) * n) if n - 2 * sum(b) > 0]
params = np.array([[np.pi * x for x in b] for b in bits])
m = np.array([(n - 2 * sum(b)) / 2 for b in bits])
ox = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(n)], num_qubits=n)
oy = SparsePauliOp.from_sparse_list([("Y", [i], 1.0) for i in range(n)], num_qubits=n)
T_TOTAL = 0.30
for dt in [0.0025, 0.00125]:
    steps = int(round(T_TOTAL / dt))
    st = transpile(step_circuit(H, dt), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    ncx = st.count_ops().get("cx", 0)
    circ, pubs = init_t, []
    for _ in range(steps):
        pubs.append((circ, [[ox], [oy]], params))
        circ = circ.compose(st)
    t0 = time.time()
    ev = np.array([r.data.evs for r in est.run(pubs).result()])  # (steps, 2, nstates)
    noisy = np.concatenate([ev[:, 0, :] @ m, ev[:, 1, :] @ m])
    # noiseless reference with the same Trotter step (exact unitary power)
    U = Operator(st).data
    from qiskit.quantum_info import Statevector
    psi = np.array([Statevector(init_t.assign_parameters(p)).data for p in params]).T
    X, Y = ox.to_matrix(), oy.to_matrix()
    cx_, cy_ = [], []
    for _ in range(steps):
        cx_.append(np.real(np.einsum("ij,ik,kj->j", psi.conj(), X, psi)) @ m)
        cy_.append(np.real(np.einsum("ij,ik,kj->j", psi.conj(), Y, psi)) @ m)
        psi = U @ psi
    tt = np.arange(steps) * dt
    clean = np.concatenate([cx_, cy_])
    best = (np.inf, None, None)
    for g in np.linspace(0, 150, 3001):
        f = np.concatenate([np.exp(-g * tt)] * 2) * clean
        A = (f @ noisy) / (f @ f)
        r = np.sum((noisy - A * f) ** 2)
        if r < best[0]:
            best = (r, g, A)
    rel = np.sqrt(best[0] / np.sum(noisy**2))
    print(f"   dt={dt*1e3:.2f} ms: {ncx} cx/step, {steps} steps -> fitted Gamma_eff = {best[1]:.1f} s^-1 "
          f"(amp {best[2]:.3f}, rel. residual {rel:.3f}); naive ncx*lam/dt*(2/3) = {ncx*lam2/dt*2/3:.1f} "
          f"[{time.time()-t0:.0f}s]")


# ---------------------------------------------------------------- (4) task-based robustness
class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {("numpy", "dtype"), ("numpy._core.numeric", "_frombuffer")}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


refs = {}
for k in [1, 2, 3]:
    with open(files / f"spectrum_{k}.pkl", "rb") as fh:
        d = NumpyOnly(fh).load()
    s = np.asarray(d["spectrum"], float)
    refs[k] = (np.asarray(d["frequencies"], float), s / np.abs(s).max())


def sticks(name):
    H, n, zbar = load(name)
    E, V = np.linalg.eigh(H.to_matrix())
    Im = V.conj().T @ SparsePauliOp.from_sparse_list(
        [("X", [i], 0.5) for i in range(n)] + [("Y", [i], -0.5j) for i in range(n)], num_qubits=n).to_matrix() @ V
    dbar = (-2 * zbar / GB - 1) * 1e6
    return [(dbar + (E[a] - E[b]) / GB * 1e6, abs(Im[a, b]) ** 2)
            for a in range(2**n) for b in range(2**n) if abs(Im[a, b]) ** 2 > 1e-9]


ST = {nm_: sticks(nm_) for nm_ in NAMES}
truth = {"molecule A": 3, "molecule B": 2, "molecule C": 1}


def sim_spec(name, ppm, gamma):
    g = (gamma / np.pi) / 2 / NU0 * 1e6  # HWHM in ppm
    return sum(w * g**2 / ((ppm - p) ** 2 + g**2) for p, w in ST[name])


print("\n(4) assignment robustness: margin = corr(correct) - max corr(wrong), worst molecule")
for gamma in [5, 20, 50, 100, 300, 1000, 3000, 10000]:
    margins = []
    for name in NAMES:
        cs = {k: np.corrcoef(sim_spec(name, f, gamma), s)[0, 1] for k, (f, s) in refs.items()}
        margins.append(cs[truth[name]] - max(v for k, v in cs.items() if k != truth[name]))
    print(f"   Gamma={gamma:6d} s^-1 (FWHM {gamma/np.pi:7.1f} Hz): worst margin {min(margins):+.3f} "
          f"{'OK' if min(margins) > 0 else 'FAILS'}")

# ---------------------------------------------------------------- (5) shot noise (molecule A, exact dynamics)
print("\n(5) shot noise on molecule A (all 8 states, notebook dt/steps, exp window Gamma):")
H, n, _ = load("molecule A")
E, V = np.linalg.eigh(H.to_matrix())
X, Y = ox.to_matrix(), oy.to_matrix()
allbits = list(np.ndindex(*(2,) * n))
mm = np.array([(n - 2 * sum(b)) / 2 for b in allbits])
plus, minus = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
psi0 = []
for b in allbits:
    v = np.array([1.0 + 0j])
    for x in reversed(b):  # qubit 0 = rightmost factor
        v = np.kron(v, minus if x else plus)
    psi0.append(v)
psi0 = np.array(psi0).T
N_STEPS = 300
tt = np.arange(N_STEPS) * DT
fx = np.zeros((N_STEPS, len(allbits)))
fy = np.zeros_like(fx)
for k, t in enumerate(tt):
    ps = V @ (np.exp(-1j * E * t)[:, None] * (V.conj().T @ psi0))
    fx[k] = np.real(np.einsum("ij,ik,kj->j", ps.conj(), X, ps))
    fy[k] = np.real(np.einsum("ij,ik,kj->j", ps.conj(), Y, ps))


def fft_causal(f):
    L = len(f)
    fe = np.zeros(2 * L - 1); fe[:L] = f; fe[L:] = f[:-1][::-1]; fe /= 2
    fo = np.zeros(2 * L - 1); fo[:L] = f; fo[L:] = -f[:-1][::-1]; fo /= 2; fo[0] = 0
    return (np.fft.fftshift(np.fft.fft(fe, norm="ortho")) + np.fft.fftshift(np.fft.fft(fo, norm="ortho")))[::2]


def spec(fx_, fy_, gamma):
    w = np.exp(-gamma * tt)
    S = fft_causal((fx_ @ mm) * w).real + fft_causal((fy_ @ mm) * w).imag
    return np.abs(S) / np.abs(S).max()


rng = np.random.default_rng(0)
for gamma in [5, 20]:
    S0 = spec(fx, fy, gamma)
    p0 = len(find_peaks(S0, prominence=0.03)[0])
    for shots in [100, 1000, 10000]:
        sig = np.sqrt(n / shots)  # std of <sum X> estimate, independent single-qubit outcomes
        cs, ps = [], []
        for _ in range(20):
            S = spec(fx + sig * rng.standard_normal(fx.shape), fy + sig * rng.standard_normal(fy.shape), gamma)
            cs.append(np.corrcoef(S, S0)[0, 1])
            ps.append(len(find_peaks(S, prominence=0.03)[0]))
        print(f"   Gamma={gamma:3d}, shots/pt={shots:6d}: corr with exact {np.mean(cs):.3f}, "
              f"peaks {np.mean(ps):5.1f} (exact {p0}) | total shots = {shots*N_STEPS*len(allbits)*2:,}")
