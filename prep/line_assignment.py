"""Brute-force (exact diagonalisation) line list for molecules A/B/C, with a physical label per line.
Usage: python prep/line_assignment.py Hackathon/nmr-hqs/files"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from struqture_py.spins import PauliHamiltonian

files = Path(sys.argv[1])
GB = 267_522_128.0 * 11.7433
NU0 = GB / (2 * np.pi)
P = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
     "Z": np.array([[1, 0], [0, -1]], complex), "I": np.eye(2)}


def op(n, ops):  # site 0 = first tensor factor = most significant bit
    m = np.array([[1.0 + 0j]])
    for s in range(n):
        m = np.kron(m, P[ops.get(s, "I")])
    return m


def bits(idx, n):
    return [(idx >> (n - 1 - s)) & 1 for s in range(n)]  # 0 = up, 1 = down


for name in ["molecule C", "molecule A", "molecule B"]:
    h = PauliHamiltonian.from_json(json.load(open(files / f"{name}_B11.7433T.json")))
    n = h.current_number_spins()
    z = np.array([float(h.get(f"{i}Z")) for i in range(n)])
    zbar = z.mean()
    H = np.zeros((2**n, 2**n), complex)
    J = {}
    for key, val in zip(h.keys(), h.values()):
        sites = {int(m.group(1)): m.group(2) for m in re.finditer(r"(\d+)([XYZ])", str(key))}
        H += (float(val) - (zbar if len(sites) == 1 else 0.0)) * op(n, sites)
        if len(sites) == 2:
            J[tuple(sorted(sites))] = 4 * float(val) / (2 * np.pi)
    off_hz = -2 * (z - zbar) / (2 * np.pi)  # Larmor offsets in the rotating frame (Hz)
    E, V = np.linalg.eigh(H)
    dom = [int(np.argmax(np.abs(V[:, k]) ** 2)) for k in range(2**n)]
    purity = [float(np.max(np.abs(V[:, k]) ** 2)) for k in range(2**n)]
    Im = V.conj().T @ sum(0.5 * (op(n, {i: "X"}) - 1j * op(n, {i: "Y"})) for i in range(n)) @ V
    lines = []
    for a in range(2**n):
        for b in range(2**n):
            w = abs(Im[a, b]) ** 2
            if w < 1e-12:
                continue
            ba, bb = bits(dom[a], n), bits(dom[b], n)
            flipped = [s for s in range(n) if ba[s] != bb[s]]
            if len(flipped) == 1:
                k = flipped[0]
                env = ",".join(f"{'↑' if ba[s]==0 else '↓'}{s}" for s in range(n) if s != k)
                label = f"spin {k} flips | neighbours {env}"
            else:
                label = f"COMBINATION ({len(flipped)} spins flip)"
            lines.append(((E[a] - E[b]) / (2 * np.pi), w, label))
    lines.sort()
    strong = [l for l in lines if l[1] > 0.02]
    print(f"\n===== {name}: N={n}, single-quantum transitions with nonzero intensity: {len(lines)} "
          f"(theory C(2N,N-1) = {len([1 for _ in combinations(range(2*n), n-1)])}) | strong: {len(strong)}")
    print("   offsets (Hz):", np.round(off_hz, 1), "| J (Hz):", {k: round(v, 2) for k, v in J.items()})
    print("   eigenstate 'purity' (weight of dominant product state):", np.round(sorted(purity), 3))
    for f, w, lab in lines:
        ppm = (-2 * zbar / GB - 1) * 1e6 + f / NU0 * 1e6
        print(f"   {f:9.2f} Hz  ({ppm:6.3f} ppm)  intensity {w:6.3f}   {lab}")
    # how many peaks survive at ~1 Hz resolution? merge strong lines closer than 1 Hz
    fs = sorted(f for f, w, _ in strong)
    groups = 1 + sum(1 for x, y in zip(fs, fs[1:]) if y - x > 1.0)
    print(f"   -> strong lines merged if closer than 1 Hz: {groups} peaks")
