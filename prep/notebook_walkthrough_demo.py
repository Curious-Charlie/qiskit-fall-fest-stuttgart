"""Walk through the 2025 HQS nmr_simulator.ipynb cells on a 2-spin toy (self-contained)."""
import re

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from struqture_py.spins import PauliHamiltonian


def struqture_to_qiskit(h, n):  # same logic as HQS utils: site i -> qubit n-1-i in the label
    labels, coeffs = [], []
    for key, val in zip(h.keys(), h.values()):
        p = ["I"] * n
        for m in re.finditer(r"(\d+)([XYZ])", str(key)):
            p[n - 1 - int(m.group(1))] = m.group(2)
        labels.append("".join(p))
        coeffs.append(complex(val))
    return SparsePauliOp(labels, coeffs)


print("=== [Cell 5] struqture PauliHamiltonian (2 spins, toy numbers) ===")
H = PauliHamiltonian()
H.set("0Z", -30.0)
H.set("1Z", -20.0)
for p in "XYZ":
    H.set(f"0{p}1{p}", 3.0)
print(H)

print("\n=== [Cell 7] rotating frame: subtract mean Z ===")
mean_z = np.mean([H.get("0Z"), H.get("1Z")])
corr = PauliHamiltonian()
for i in range(2):
    corr.set(f"{i}Z", mean_z)
H_rot = H - corr
print("mean_z =", mean_z)
print(H_rot)

print("\n=== [Cell 9] -> Qiskit SparsePauliOp ===")
Hq = struqture_to_qiskit(H_rot, 2)
print(Hq)

print("\n=== [Cell 12] one Trotter step ===")
dt = 0.01
step = QuantumCircuit(2)
step.append(PauliEvolutionGate(Hq, dt, synthesis=LieTrotter()), [0, 1])
print("before decompose:")
print(step.draw(output="text"))
step_d = step.decompose(reps=1)
print("after decompose(reps=1):")
print(step_d.draw(output="text", fold=140))
print("elementary ops:", dict(step_d.decompose(reps=2).count_ops()))

print("\n=== [Cell 14] X-basis initial-state template ===")
phi = ParameterVector("phi", 2)
init = QuantumCircuit(2)
for q in range(2):
    init.ry(np.pi / 2, q)
    init.rz(phi[q], q)
print(init.draw(output="text"))

print("\n=== [Cell 16] observables ===")
obs_x = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(2)], num_qubits=2)
obs_y = SparsePauliOp.from_sparse_list([("Y", [i], 1.0) for i in range(2)], num_qubits=2)
print("obs_x =", obs_x.to_list(), " obs_y =", obs_y.to_list())

print("\n=== [Cell 18] Estimator ===")
circ = init.compose(step_d).compose(step_d)  # init + 2 Trotter steps -> t = 2 dt
res = StatevectorEstimator().run([(circ, [obs_x, obs_y], {phi[0]: 0.0, phi[1]: 0.0})]).result()
print("len(result) =", len(res), "| res[0].data.evs =", res[0].data.evs)
res2 = StatevectorEstimator().run([(init, [obs_x, obs_y], {phi[0]: 0.0, phi[1]: np.pi})]).result()
print("t=0, state |+->: evs =", res2[0].data.evs)
