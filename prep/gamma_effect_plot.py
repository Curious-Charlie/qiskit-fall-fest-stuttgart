"""What does increasing Gamma do? Molecule C, exact dynamics, long T (no truncation effects).
Usage: python prep/gamma_effect_plot.py Hackathon/nmr-hqs/files  -> prep/gamma_effect.png"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from struqture_py.spins import PauliHamiltonian

files = Path(sys.argv[1])
P = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
     "Z": np.array([[1, 0], [0, -1]], complex), "I": np.eye(2)}


def op(n, ops):
    m = np.array([[1.0 + 0j]])
    for s in range(n):
        m = np.kron(m, P[ops.get(s, "I")])
    return m


h = PauliHamiltonian.from_json(json.load(open(files / "molecule C_B11.7433T.json")))
n = h.current_number_spins()
zbar = np.mean([float(h.get(f"{i}Z")) for i in range(n)])
H = np.zeros((2**n, 2**n), complex)
for key, val in zip(h.keys(), h.values()):
    sites = {int(m.group(1)): m.group(2) for m in re.finditer(r"(\d+)([XYZ])", str(key))}
    H += (float(val) - (zbar if len(sites) == 1 else 0.0)) * op(n, sites)

E, V = np.linalg.eigh(H)
SX = V.conj().T @ sum(op(n, {i: "X"}) for i in range(n)) @ V
SY = V.conj().T @ sum(op(n, {i: "Y"}) for i in range(n)) @ V
dE = E[:, None] - E[None, :]
dt, N = 0.0025, 2000  # T = 5 s: long enough that truncation plays no role
t = np.arange(N) * dt
cxx = np.array([np.real(np.sum(np.exp(1j * dE * tk) * SX * SX.T)) for tk in t])
cyx = np.array([np.real(np.sum(np.exp(1j * dE * tk) * SY * SX.T)) for tk in t])


def spectrum(gamma, pad=8):
    w = np.exp(-gamma * t)
    w[0] *= 0.5
    L = pad * N
    S = np.fft.fftshift(np.fft.fft(cxx * w, n=L)).real + np.fft.fftshift(np.fft.fft(cyx * w, n=L)).imag
    nu = np.fft.fftshift(np.fft.fftfreq(L, d=dt))
    return nu, S * dt


gammas = [3, 10, 25, 60]
colors = ["C0", "C1", "C2", "C3"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for g, c in zip(gammas, colors):
    ax[0].plot(t, cxx * np.exp(-g * t), color=c, lw=1, label=f"Γ={g} s⁻¹")
    nu, S = spectrum(g)
    ax[1].plot(nu, S, color=c, lw=1.5, label=f"Γ={g}  (FWHM {g/np.pi:.1f} Hz)")
    ax[2].plot(nu, S / S.max(), color=c, lw=1.5, label=f"Γ={g}")
ax[0].set(xlim=(0, 1.0), xlabel="t / s", ylabel="C_xx(t)·e^{-Γt}", title="Time domain: faster decay")
ax[1].set(xlim=(-40, 40), xlabel="ν − ν_ref / Hz", ylabel="S(ν) (not normalised)",
          title="Frequency domain: wider + lower (area conserved)")
ax[2].set(xlim=(-40, 40), xlabel="ν − ν_ref / Hz", ylabel="S(ν) / max",
          title="What the notebook shows (normalised): lines merge")
for a in ax[1:]:
    for x in [-22.63, -14.71, 14.71, 22.63]:
        a.axvline(x, color="k", lw=0.5, ls=":")
for a in ax:
    a.legend(fontsize=8)
plt.tight_layout()
out = Path(__file__).with_name("gamma_effect.png")
plt.savefig(out, dpi=110)
nu, S3 = spectrum(3)
_, S60 = spectrum(60)
print("saved", out, "| area ratio Γ=60/Γ=3:", round(np.trapezoid(S60, nu) / np.trapezoid(S3, nu), 3),
      "| peak height ratio:", round(S60.max() / S3.max(), 3))
