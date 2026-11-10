"""Verify candidate 'interesting phenomena' on the 2026 molecules (exact numerics, no Trotter circuits)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import schur
from struqture_py.spins import PauliHamiltonian

files = Path(sys.argv[1])
P = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
     "Z": np.array([[1, 0], [0, -1]], complex), "I": np.eye(2)}


def op(n, ops):
    m = np.array([[1.0 + 0j]])
    for s in range(n):
        m = np.kron(m, P[ops.get(s, "I")])
    return m


def terms(name, flip_pairs=()):
    h = PauliHamiltonian.from_json(json.load(open(files / f"{name}_B11.7433T.json")))
    n = h.current_number_spins()
    zbar = np.mean([float(h.get(f"{i}Z")) for i in range(n)])
    out = []
    for key, val in zip(h.keys(), h.values()):
        sites = {int(m.group(1)): m.group(2) for m in re.finditer(r"(\d+)([XYZ])", str(key))}
        c = float(val)
        if len(sites) == 1:
            c -= zbar
        if len(sites) == 2 and tuple(sorted(sites)) in flip_pairs:
            c = -c
        out.append((c, op(n, sites)))
    return out, n


def lines_from_eig(E, V, n, wrap=None):
    Im = V.conj().T @ sum(0.5 * (op(n, {i: "X"}) - 1j * op(n, {i: "Y"})) for i in range(n)) @ V
    L = []
    for a in range(2**n):
        for b in range(2**n):
            w = abs(Im[a, b]) ** 2
            if w > 1e-9:
                f = E[a] - E[b]
                if wrap:
                    f = (f + wrap / 2) % wrap - wrap / 2
                L.append((f / (2 * np.pi), w))
    return np.array(L)


def exact_lines(T, n):
    H = sum(c * M for c, M in T)
    E, V = np.linalg.eigh(H)
    return lines_from_eig(E, V, n), E, V


def trotter_lines(T, n, dt):
    U = np.eye(2**n, dtype=complex)
    for c, M in T:  # first term applied first
        U = (np.cos(c * dt) * np.eye(2**n) - 1j * np.sin(c * dt) * M) @ U
    Tm, Z = schur(U, output="complex")
    eps = -np.angle(np.diag(Tm)) / dt  # Floquet quasi-energies (rad/s)
    return lines_from_eig(eps, Z, n, wrap=2 * np.pi / dt)


def cluster(L, tol):
    L = L[np.argsort(L[:, 0])]
    out = []
    for f, w in L:
        if out and abs(f - out[-1][0]) < tol:
            F, W = out[-1]
            out[-1] = ((F * W + f * w) / (W + w), W + w)
        else:
            out.append((f, w))
    return np.array(out)


def lorentz(L, nu, fwhm):
    g = fwhm / 2
    return sum(w * g**2 / ((nu - f) ** 2 + g**2) for f, w in L)


print("=== (a) molecule C: roof effect <-> eigenstate entanglement ===")
T, n = terms("molecule C")
L, E, V = exact_lines(T, n)
Lc = cluster(L, 1e-6)
print("lines (Hz, weight):", [(round(f, 2), round(w, 3)) for f, w in Lc])
conc = [2 * abs(V[0, k] * V[3, k] - V[1, k] * V[2, k]) for k in range(4)]
J, dnu = 7.92, 36.4
D = np.hypot(dnu, J)
print("eigenstate concurrences:", np.round(conc, 4), "| J/D =", round(J / D, 4),
      "| roof ratio (1+J/D)/(1-J/D) =", round((1 + J / D) / (1 - J / D), 3))

print("\n=== (b) Trotter = Floquet: line-position and intensity errors vs dt ===")
for name in ["molecule C", "molecule A", "molecule B"]:
    T, n = terms(name)
    Lex = cluster(exact_lines(T, n)[0], 1e-6)
    strong = Lex[Lex[:, 1] > 0.05 * Lex[:, 1].max()]
    print(f"{name}: {len(strong)} strong lines")
    for dt in [0.005, 0.0025, 0.00125, 0.000625]:
        Lt = cluster(trotter_lines(T, n, dt), 1e-6)
        pos_err, int_err = [], []
        for f, w in strong:
            k = np.argmin(np.abs(Lt[:, 0] - f) - 1e-3 * Lt[:, 1])  # nearest line
            pos_err.append(abs(Lt[k, 0] - f))
            int_err.append(abs(Lt[k, 1] - w) / w)
        print(f"   dt={dt*1e3:6.3f} ms  Nyquist ±{1/(2*dt):5.0f} Hz | max pos err {max(pos_err):8.4f} Hz "
              f"| max rel intensity err {max(int_err):7.4f}")

print("\n=== (c) does the sign of J matter? (molecule B; 1 Hz Lorentzians, normalised) ===")
nu = np.linspace(-250, 250, 50001)
T0, n = terms("molecule B")
S0 = lorentz(cluster(exact_lines(T0, n)[0], 1e-6), nu, 1.0)
S0 /= S0.max()
for label, pairs in [("flip J02 (-0.5 -> +0.5)", ((0, 2),)), ("flip J01", ((0, 1),)),
                     ("flip ALL J", ((0, 1), (0, 2), (1, 2)))]:
    T1, _ = terms("molecule B", flip_pairs=pairs)
    S1 = lorentz(cluster(exact_lines(T1, n)[0], 1e-6), nu, 1.0)
    S1 /= S1.max()
    print(f"   {label:26s}: max |ΔS| = {np.max(np.abs(S1 - S0)):.2e}")

print("\n=== (d) positive-m trick: f_nbar(t) = -f_n(t) exactly? (molecule A) ===")
T, n = terms("molecule A")
H = sum(c * M for c, M in T)
E, V = np.linalg.eigh(H)
plus, minus = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
SX = sum(op(n, {i: "X"}) for i in range(n))
SY = sum(op(n, {i: "Y"}) for i in range(n))
worst = 0
for bits in range(2**n):
    b = [(bits >> (n - 1 - i)) & 1 for i in range(n)]
    psi = np.array([1.0 + 0j])
    psib = np.array([1.0 + 0j])
    for bi in b:
        psi = np.kron(psi, minus if bi else plus)
        psib = np.kron(psib, plus if bi else minus)
    for t in np.linspace(0, 0.75, 7):
        U = V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T
        for O in (SX, SY):
            a = np.vdot(U @ psi, O @ (U @ psi)).real
            c = np.vdot(U @ psib, O @ (U @ psib)).real
            worst = max(worst, abs(a + c))
print(f"   max |f_n + f_nbar| over all states, t, (X,Y) = {worst:.2e}")
