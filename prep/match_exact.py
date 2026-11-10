"""Exact-diagonalisation matching of molecules A/B/C to spectrum_1/2/3 (2026 HQS NMR prompt)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from struqture_py.spins import PauliHamiltonian

files = Path(sys.argv[1])
GAMMA_H = 267_522_128.0
FIELD = 11.7433
GB = GAMMA_H * FIELD  # rad/s, reference Larmor


class NumpyOnly(pickle.Unpickler):
    ALLOWED = {("numpy", "dtype"), ("numpy._core.numeric", "_frombuffer")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_pkl(p):
    with open(p, "rb") as f:
        return NumpyOnly(f).load()


P = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
     "Z": np.array([[1, 0], [0, -1]], complex), "I": np.eye(2)}


def op(n, ops):  # ops: {site: 'X'}; site 0 = first tensor factor (convention-free for spectra)
    m = np.array([[1.0 + 0j]])
    for s in range(n):
        m = np.kron(m, P[ops.get(s, "I")])
    return m


def stick_spectrum(h):
    n = h.current_number_spins()
    z = np.array([float(h.get(f"{i}Z")) for i in range(n)])
    zbar = z.mean()
    H = np.zeros((2**n, 2**n), complex)
    for key, val in zip(h.keys(), h.values()):
        s = str(key)
        sites = {}
        import re
        for mm in re.finditer(r"(\d+)([XYZ])", s):
            sites[int(mm.group(1))] = mm.group(2)
        c = float(val) - (zbar if (len(sites) == 1 and list(sites.values())[0] == "Z") else 0.0)
        H += c * op(n, sites)
    E, V = np.linalg.eigh(H)
    Iminus = sum(0.5 * (op(n, {i: "X"}) - 1j * op(n, {i: "Y"})) for i in range(n))
    Im = V.conj().T @ Iminus @ V
    delta_bar_ppm = (-2 * zbar / GB - 1) * 1e6  # Pauli coeff = -gamma(1+delta)B/2
    lines = []
    for a in range(2**n):
        for b in range(2**n):
            w = abs(Im[a, b]) ** 2
            if w > 1e-8:
                lines.append((delta_bar_ppm + (E[a] - E[b]) / GB * 1e6, w))
    shifts = (-2 * z / GB - 1) * 1e6
    return np.array(lines), shifts


def broaden(lines, ppm_axis, fwhm_ppm):
    g = fwhm_ppm / 2
    S = np.zeros_like(ppm_axis, dtype=float)
    for p, w in lines:
        S += w * g**2 / ((ppm_axis - p) ** 2 + g**2)
    return S / S.max()


refs = {}
for k in [1, 2, 3]:
    d = load_pkl(files / f"spectrum_{k}.pkl")
    refs[k] = (np.asarray(d["frequencies"], float), np.asarray(d["spectrum"], float).real)
hqsC = load_pkl(files / "hqs_spectrum_tools_molecule C_B11.7433T.pkl")
print("pkl keys:", list(hqsC.keys()) if isinstance(hqsC, dict) else type(hqsC))

for k, (f, s) in refs.items():
    s = s / np.max(np.abs(s))
    pk = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > 0.1]
    print(f"spectrum_{k}: ppm range [{f.min():.3f}, {f.max():.3f}], {len(f)} pts, "
          f"peaks@ {np.round(np.sort(f[pk]), 3).tolist()}")

mols = {}
for name in ["molecule A", "molecule B", "molecule C"]:
    h = PauliHamiltonian.from_json(json.load(open(files / f"{name}_B11.7433T.json")))
    lines, shifts = stick_spectrum(h)
    mols[name] = lines
    strong = lines[lines[:, 1] > 0.05 * lines[:, 1].max()]
    print(f"{name}: chemical shifts (ppm) = {np.round(np.sort(shifts), 3).tolist()} | "
          f"{len(strong)} lines in [{strong[:,0].min():.3f}, {strong[:,0].max():.3f}] ppm")

# validate convention with the labelled HQS spectrum of molecule C
fC, sC = np.asarray(hqsC["frequencies"], float), np.asarray(hqsC["spectrum"], float).real
sC = sC / np.max(np.abs(sC))
for fwhm in [0.001, 0.002, 0.004]:
    mine = broaden(mols["molecule C"], fC, fwhm)
    print(f"validation vs HQS molecule C (fwhm {fwhm} ppm): corr = {np.corrcoef(mine, sC)[0,1]:.3f}")

print("\nSimilarity matrix (Pearson corr; best over linewidths 0.001–0.004 ppm):")
print(" " * 12 + "".join(f"spectrum_{k:<4}" for k in refs))
for name, lines in mols.items():
    row = []
    for k, (f, s) in refs.items():
        s = s / np.max(np.abs(s))
        row.append(max(np.corrcoef(broaden(lines, f, w), s)[0, 1] for w in [0.001, 0.002, 0.004]))
    print(f"{name:12s}" + "".join(f"{r:<14.3f}" for r in row))
