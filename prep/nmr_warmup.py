# %% [markdown]
# # NMR-on-a-quantum-computer warm-up (no HQS packages needed)
#
# Self-contained rehearsal of the HQS hackathon pipeline (same as their 2025 workshop):
#   H(shifts, J)  ->  rotating frame  ->  Trotter circuit  ->  X-basis product states
#   ->  <I_x(t)>, <I_y(t)> via Estimator  ->  C_xx, C_yx  ->  FFT  ->  spectrum
# plus two things the template doesn't give you:
#   * an EXACT reference (dense diagonalisation) pushed through the same FFT,
#     plus a stick spectrum, so you can tell Trotter error from post-processing bugs;
#   * a depolarising-noise sweep (the "how much noise can the spectrum tolerate" part).
#
# Run as a script (`python prep/nmr_warmup.py`) or cell-by-cell in VS Code / Jupyter.

# %%
import itertools
import time

import matplotlib

matplotlib.use("Agg") if __name__ == "__main__" else None
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter

# %% [markdown]
# ## 1. Physics: the liquid-state NMR Hamiltonian
#
# $H = -\sum_l \omega_l I^z_l + 2\pi \sum_{k<l} J_{kl}\, \mathbf I_k\cdot\mathbf I_l$, with
# $\omega_l = \gamma B (1+\delta_l)$.
# The rotating frame at the mean Larmor frequency removes the huge common precession (~MHz).
# Only the *offsets* $\Omega_l = 2\pi\nu_0(\delta_l-\bar\delta)\cdot10^{-6}$ (~100 Hz) are left.
# With $I = \sigma/2$, this gives the Pauli form
# $H_{rot} = -\sum_l \tfrac{\Omega_l}{2} Z_l + \tfrac{2\pi}{4}\sum_{k<l} J_{kl}(X_kX_l+Y_kY_l+Z_kZ_l)$.
#
# A toy 3-spin molecule on a 60 MHz benchtop spectrometer (1 ppm = 60 Hz) keeps the
# Trotter step count small. Swap in your own shifts / J's.

# %%
NU0_HZ = 60e6  # 1H Larmor frequency (60 MHz benchtop; 500 MHz -> 1 ppm = 500 Hz)
SHIFTS_PPM = np.array([7.25, 6.40, 3.90])  # delta_l
J_HZ = {(0, 1): 8.0, (1, 2): 6.0, (0, 2): 1.5}  # scalar couplings
N = len(SHIFTS_PPM)
DELTA_BAR = SHIFTS_PPM.mean()


def hamiltonian_pauli(shifts_ppm, j_hz, nu0_hz):
    """Rotating-frame Hamiltonian as a SparsePauliOp (rad/s). Site l -> Qiskit qubit l."""
    n = len(shifts_ppm)
    omega = 2 * np.pi * nu0_hz * (shifts_ppm - shifts_ppm.mean()) * 1e-6
    terms = [("Z", [l], -omega[l] / 2) for l in range(n)]
    for (k, l), J in j_hz.items():
        for p in "XYZ":
            terms.append((p + p, [k, l], 2 * np.pi * J / 4))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


H = hamiltonian_pauli(SHIFTS_PPM, J_HZ, NU0_HZ)
print(H)
print(f"max |coeff| = {np.max(np.abs(H.coeffs)):.1f} rad/s")

# %% [markdown]
# ## 2. Choosing dt and the number of steps: the most important practical decision
# * Nyquist: every line must satisfy |nu - nu_ref| < 1/(2 dt), otherwise it aliases (folds back).
# * Resolution: the frequency grid spacing is 1/(n_steps * dt). It must resolve J (a few Hz).
# * Trotter accuracy: roughly ||H|| dt << 1 (first order), which is why the rotating frame matters.
# CHECK THE UNITS of whatever Hamiltonian they hand you before choosing dt.

# %%
offsets_hz = NU0_HZ * (SHIFTS_PPM - DELTA_BAR) * 1e-6
span_hz = np.max(np.abs(offsets_hz)) + sum(J_HZ.values())
DT = 0.4 / (2 * span_hz)  # ~2.5x margin under Nyquist
N_STEPS = 512
GAMMA = 2.0  # s^-1 exponential window (line broadening, ~0.6 Hz FWHM)
print(f"dt = {DT*1e3:.3f} ms | Nyquist = ±{1/(2*DT):.0f} Hz | grid = {1/(N_STEPS*DT):.2f} Hz "
      f"| T = {N_STEPS*DT:.2f} s | ||H||dt ~ {np.sum(np.abs(H.coeffs))*DT:.2f}")

# %% [markdown]
# ## 3. Exact reference (ED). C_ab(t) = Tr(I_a(t) I_b), Heisenberg picture.
# The spectrum is $S(\nu)=\mathrm{Re}\,\mathcal F[C_{xx}](\nu)+\mathrm{Im}\,\mathcal F[C_{yx}](\nu)$, using $e^{-i\nu t}$.
# This follows from $\mathrm{Tr}(I^-(t)I^+) = 2(C_{xx}-iC_{yx})$ because $[H, I_z^{tot}]=0$.
# Stick lines sit at $E_a-E_b$ with weight $|\langle a|I^-|b\rangle|^2$.

# %%
def total_op(p, n):
    return SparsePauliOp.from_sparse_list([(p, [i], 0.5) for i in range(n)], num_qubits=n)


IX, IY = total_op("X", N), total_op("Y", N)
times = np.arange(N_STEPS) * DT


def exact_correlations(H, times):
    E, V = np.linalg.eigh(H.to_matrix())
    Ix_e = V.conj().T @ IX.to_matrix() @ V
    Iy_e = V.conj().T @ IY.to_matrix() @ V
    dE = E[:, None] - E[None, :]
    cxx, cyx = [], []
    for t in times:
        ph = np.exp(1j * dE * t)  # (I(t))_ab = e^{i(E_a-E_b)t} I_ab
        cxx.append(np.real(np.sum((ph * Ix_e) * Ix_e.T)))
        cyx.append(np.real(np.sum((ph * Iy_e) * Ix_e.T)))
    return np.array(cxx), np.array(cyx), (E, V)


def spectrum_from_correlations(cxx, cyx, dt, gamma=GAMMA, pad=8):
    """Windowed, zero-padded causal FT -> (ppm axis, normalised spectrum)."""
    t = np.arange(len(cxx)) * dt
    w = np.exp(-gamma * t)
    w[0] *= 0.5  # trapezoid rule for a one-sided (causal) integral
    L = pad * len(cxx)
    fxx = np.fft.fftshift(np.fft.fft(cxx * w, n=L))
    fyx = np.fft.fftshift(np.fft.fft(cyx * w, n=L))
    nu = np.fft.fftshift(np.fft.fftfreq(L, d=dt))  # Hz, rotating frame
    S = fxx.real + fyx.imag
    return DELTA_BAR + nu / NU0_HZ * 1e6, S / np.max(np.abs(S))


cxx_ex, cyx_ex, (E, V) = exact_correlations(H, times)
ppm, S_exact = spectrum_from_correlations(cxx_ex, cyx_ex, DT)

Im = V.conj().T @ (IX - 1j * IY).to_matrix() @ V
sticks = [(DELTA_BAR + (E[a] - E[b]) / (2 * np.pi) / NU0_HZ * 1e6, abs(Im[a, b]) ** 2)
          for a in range(2**N) for b in range(2**N) if abs(Im[a, b]) ** 2 > 1e-6]
print("strongest stick lines (ppm, weight):",
      sorted([(round(p, 3), round(w, 3)) for p, w in sticks], key=lambda x: -x[1])[:6])

# %% [markdown]
# ## 4. Quantum circuit: Trotter step, X-basis initial states, Estimator
# $C_{ax}(t) = \sum_n m_n \langle n|I_a(t)|n\rangle$, where $|n\rangle$ runs over X-basis product
# states and $m_n$ is their $I_x^{tot}$ eigenvalue. This is the identity inserted in $I_x$'s
# eigenbasis. Every term is just "prepare a product state, evolve, measure <I_x>,<I_y>".

# %%
def trotter_step(H, dt, order=1):
    synth = LieTrotter() if order == 1 else SuzukiTrotter(order=order)
    qc = QuantumCircuit(H.num_qubits)
    qc.append(PauliEvolutionGate(H, dt, synthesis=synth), range(H.num_qubits))
    return qc.decompose(reps=2)


def xbasis_template(n):
    phi = ParameterVector("phi", n)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)  # |+>
        qc.rz(phi[q], q)  # phi=pi -> |->
    return qc, phi


def quantum_correlations(H, dt, n_steps, states="all", order=1, estimator=None, pm=None):
    """Returns C_xx, C_yx sampled at k*dt. states='all' or 'positive' (HQS trick)."""
    n = H.num_qubits
    step = trotter_step(H, dt, order)
    init, phi = xbasis_template(n)
    bits = list(itertools.product([0, 1], repeat=n))
    if states == "positive":
        bits = [b for b in bits if n - 2 * sum(b) > 0]
    params = np.array([[np.pi * b for b in bs] for bs in bits])  # shape (n_states, n)
    m = np.array([(n - 2 * sum(b)) / 2 for b in bits])  # I_x^tot eigenvalue of each state
    if estimator is None:
        # Fast noiseless path: apply the exact one-step Trotter unitary to every initial state.
        # Same maths as running circuits of depth k, but O(n_steps) instead of O(n_steps^2).
        U = Operator(step).data
        psi = np.array([Statevector(init.assign_parameters(p)).data for p in params]).T
        Ix, Iy = IX.to_matrix(), IY.to_matrix()
        evs = np.zeros((n_steps, 2, len(bits)))
        for k in range(n_steps):
            evs[k, 0] = np.real(np.einsum("ij,ik,kj->j", psi.conj(), Ix, psi))
            evs[k, 1] = np.real(np.einsum("ij,ik,kj->j", psi.conj(), Iy, psi))
            psi = U @ psi
    else:
        # Estimator path (what you'd do on a noisy simulator / hardware): one PUB per time step,
        # params of shape (n_states, n) broadcast against observables of shape (2, 1).
        # Transpile the pieces ONCE and compose; transpiling every deep circuit is O(n^2).
        init_c, step_c = (pm.run(init), pm.run(step)) if pm is not None else (init, step)
        obs = [[IX], [IY]]
        circ, pubs = init_c, []
        for _ in range(n_steps):
            pubs.append((circ, obs, params))
            circ = circ.compose(step_c)
        res = estimator.run(pubs).result()
        evs = np.array([r.data.evs for r in res])  # (n_steps, 2, n_states)
    cxx = evs[:, 0, :] @ m
    cyx = evs[:, 1, :] @ m
    if states == "positive":  # HQS template only uses m>0 states; rescale by 2
        cxx, cyx = 2 * cxx, 2 * cyx
    return cxx, cyx


t0 = time.time()
cxx_q, cyx_q = quantum_correlations(H, DT, N_STEPS, states="all", order=1)
print(f"Trotter-1 statevector run: {time.time()-t0:.1f}s")
_, S_trot = spectrum_from_correlations(cxx_q, cyx_q, DT)
cxx_p, cyx_p = quantum_correlations(H, DT, N_STEPS, states="positive", order=1)
_, S_pos = spectrum_from_correlations(cxx_p, cyx_p, DT)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


print(f"max|C_xx(Trotter) - C_xx(exact)| = {np.max(np.abs(cxx_q-cxx_ex)):.2e}")
print(f"RMSE spectrum: Trotter-1 all states vs exact     = {rmse(S_trot, S_exact):.4f}")
print(f"RMSE spectrum: Trotter-1 positive-only vs exact  = {rmse(S_pos, S_exact):.4f}")

# %% [markdown]
# ## 5. Noise: how much can the spectrum tolerate?
# This is a crude model: depolarising error on every 1q and 2q gate, using Aer's Estimator
# with exact expectation values (no shots). Circuit depth grows linearly with t, so noise
# acts like extra line broadening plus amplitude decay. Watch the RMSE grow with p2.

# %%
try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    BASIS = ["rz", "sx", "x", "cx"]

    def noisy_estimator(p2):
        nm = NoiseModel(basis_gates=BASIS)
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        return AerEstimator(options={"backend_options": {"noise_model": nm, "method": "density_matrix"},
                                     "run_options": {"shots": None}, "default_precision": 0})

    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=BASIS)
    step_isa = pm.run(trotter_step(H, DT))
    print(f"one Trotter step: {step_isa.count_ops().get('cx', 0)} CX, depth {step_isa.depth()}")

    NOISE_STEPS = 128  # shorter run keeps the noise sweep fast; same dt, coarser resolution
    _, S_ref_short = spectrum_from_correlations(cxx_ex[:NOISE_STEPS], cyx_ex[:NOISE_STEPS], DT)
    noisy = {}
    for p2 in [1e-4, 1e-3, 5e-3]:
        t0 = time.time()
        cx_n, cy_n = quantum_correlations(H, DT, NOISE_STEPS, states="positive",
                                          estimator=noisy_estimator(p2), pm=pm)
        ppm_n, noisy[p2] = spectrum_from_correlations(cx_n, cy_n, DT)
        print(f"p2={p2:.0e}: RMSE vs exact = {rmse(noisy[p2], S_ref_short):.4f} ({time.time()-t0:.1f}s)")
except ImportError as e:
    noisy = {}
    print("qiskit-aer not available, skipping noise sweep:", e)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(ppm, S_exact, lw=2, label="exact (ED)")
ax.plot(ppm, S_trot, "--", label="Trotter-1, all states")
ax.plot(ppm, S_pos, ":", label="Trotter-1, positive-m only (x2)")
for p, w in sticks:
    ax.vlines(p, 0, -0.3 * w / max(w for _, w in sticks), color="k", lw=1)
ax.set(xlabel="δ / ppm", ylabel="normalised S", title="Noiseless")
ax.invert_xaxis()
ax.legend(fontsize=8)
ax = axes[1]
if noisy:
    ax.plot(ppm_n, S_ref_short, lw=2, label="exact (same T)")
    for p2, S in noisy.items():
        ax.plot(ppm_n, S, label=f"depol p2={p2:g}")
    ax.set(xlabel="δ / ppm", title=f"Noise sweep ({NOISE_STEPS} steps)")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
plt.tight_layout()
out = "prep/nmr_warmup.png"
plt.savefig(out, dpi=120)
print("saved", out)
