"""Check pitfalls of the notebook's noise_strength sweep using exact correlations (no Trotter).
Re-implements the notebook's post-processing (window -> fft_causal -> Re/Im combo -> np.abs)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from struqture_py.spins import PauliHamiltonian

files = Path(sys.argv[1])
P = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
     "Z": np.array([[1, 0], [0, -1]], complex), "I": np.eye(2)}


def op(n, ops):
    m = np.array([[1.0 + 0j]])
    for s in range(n):
        m = np.kron(m, P[ops.get(s, "I")])
    return m


def rot_hamiltonian(name):
    h = PauliHamiltonian.from_json(json.load(open(files / f"{name}_B11.7433T.json")))
    n = h.current_number_spins()
    zbar = np.mean([float(h.get(f"{i}Z")) for i in range(n)])
    H = np.zeros((2**n, 2**n), complex)
    for key, val in zip(h.keys(), h.values()):
        sites = {int(m.group(1)): m.group(2) for m in re.finditer(r"(\d+)([XYZ])", str(key))}
        single_z = len(sites) == 1 and list(sites.values())[0] == "Z"
        H += (float(val) - (zbar if single_z else 0.0)) * op(n, sites)
    return H, n


def correlations(H, n, dt, N):
    E, V = np.linalg.eigh(H)
    SX = V.conj().T @ sum(op(n, {i: "X"}) for i in range(n)) @ V
    SY = V.conj().T @ sum(op(n, {i: "Y"}) for i in range(n)) @ V
    dE = E[:, None] - E[None, :]
    t = np.arange(N) * dt
    cxx = np.array([np.real(np.sum(np.exp(1j * dE * tk) * SX * SX.T)) for tk in t])
    cyx = np.array([np.real(np.sum(np.exp(1j * dE * tk) * SY * SX.T)) for tk in t])
    return t, cxx, cyx


def fft_causal(f):  # same as notebook
    L = len(f)
    fe = np.zeros(2 * L - 1, dtype=f.dtype); fe[:L] = f; fe[L:] = f[:-1][::-1]; fe /= 2
    fo = np.zeros(2 * L - 1, dtype=f.dtype); fo[:L] = f; fo[L:] = -f[:-1][::-1]; fo /= 2; fo[0] = 0
    return (np.fft.fftshift(np.fft.fft(fe, norm="ortho")) + np.fft.fftshift(np.fft.fft(fo, norm="ortho")))[::2]


def spectrum(cxx, cyx, t, dt, gamma, pad=1):
    w = np.exp(-gamma * t)
    x, y = cxx * w, cyx * w
    if pad > 1:  # zero-padding AFTER windowing (pure interpolation)
        x = np.concatenate([x, np.zeros((pad - 1) * len(x))])
        y = np.concatenate([y, np.zeros((pad - 1) * len(y))])
    S = fft_causal(x).real + fft_causal(y).imag
    S /= np.max(np.abs(S))
    nu = np.fft.fftshift(np.fft.fftfreq(len(x), d=dt))
    return nu, S


def npeaks(S, prom=0.03):
    return len(find_peaks(S, prominence=prom)[0])


DT = 0.0025
for name, true_lines in [("molecule C", 4), ("molecule A", 12)]:
    H, n = rot_hamiltonian(name)
    print(f"\n===== {name} (true resolved lines at ~1 Hz linewidth: {true_lines}) =====")
    for N in [300, 1200]:
        t, cxx, cyx = correlations(H, n, DT, N)
        T = N * DT
        print(f"-- N={N}, T={T:.2f}s, grid={1/T:.2f} Hz")
        print(f"{'Gamma':>7} {'FWHM Hz':>8} {'e^-GT':>7} | {'peaks abs':>9} {'peaks real':>10} "
              f"{'min(real)':>9} | {'peaks abs, 8x pad':>17}")
        for g in [0.1, 1, 3, 5, 10, 20, 43, 60, 90]:
            nu, S = spectrum(cxx, cyx, t, DT, g)
            _, Sp = spectrum(cxx, cyx, t, DT, g, pad=8)
            print(f"{g:7.1f} {g/np.pi:8.2f} {np.exp(-g*T):7.3f} | {npeaks(np.abs(S)):9d} {npeaks(S):10d} "
                  f"{S.min():9.2f} | {npeaks(np.abs(Sp)):17d}")
