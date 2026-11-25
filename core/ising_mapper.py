"""
Maps QUBO representation to an Ising Cost Hamiltonian (SparsePauliOp in Qiskit).
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .qubo_formulation import QUBOFormulation


class IsingMapper:
    """
    Converts a QUBO matrix into an Ising Hamiltonian:
        H_C = sum_i h_i Z_i + sum_{i < j} J_{ij} Z_i Z_j + offset * I
    using the algebraic transformation:
        x_i = (I - Z_i) / 2
    """

    def __init__(self, qubo: QUBOFormulation):
        self.qubo = qubo
        self.n = qubo.n
        self.h_vector, self.J_matrix, self.ising_offset = self._qubo_to_ising()
        self.cost_hamiltonian = self._build_sparse_pauli_op()

    def _qubo_to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Derives Ising linear fields h_i, coupling constants J_ij, and total offset.
        """
        n = self.n
        Q = self.qubo.Q_matrix
        qubo_offset = self.qubo.offset

        h = np.zeros(n, dtype=float)
        J = np.zeros((n, n), dtype=float)
        ising_offset = float(qubo_offset)

        # 1. Process diagonal linear terms: Q_ii * (I - Z_i) / 2
        for i in range(n):
            q_ii = Q[i, i]
            ising_offset += q_ii / 2.0
            h[i] -= q_ii / 2.0

        # 2. Process off-diagonal coupling terms: Q_ij * (I - Z_i - Z_j + Z_i Z_j) / 4
        for i in range(n):
            for j in range(i + 1, n):
                q_ij = Q[i, j]
                ising_offset += q_ij / 4.0
                h[i] -= q_ij / 4.0
                h[j] -= q_ij / 4.0
                J[i, j] = q_ij / 4.0

        return h, J, ising_offset

    def _build_sparse_pauli_op(self) -> SparsePauliOp:
        """
        Constructs Qiskit SparsePauliOp from Ising coefficients.
        Note: Qiskit little-endian notation: qubit k corresponds to index (n - 1 - k).
        """
        n = self.n
        pauli_list: List[Tuple[str, float]] = []

        # Single-qubit Z terms: sum_i h_i Z_i
        for i in range(n):
            coeff = float(self.h_vector[i])
            if abs(coeff) > 1e-9:
                pauli_str = ["I"] * n
                pauli_str[n - 1 - i] = "Z"
                pauli_list.append(("".join(pauli_str), coeff))

        # Two-qubit ZZ coupling terms: sum_{i < j} J_ij Z_i Z_j
        for i in range(n):
            for j in range(i + 1, n):
                coeff = float(self.J_matrix[i, j])
                if abs(coeff) > 1e-9:
                    pauli_str = ["I"] * n
                    pauli_str[n - 1 - i] = "Z"
                    pauli_str[n - 1 - j] = "Z"
                    pauli_list.append(("".join(pauli_str), coeff))

        # Identity constant offset: offset * I
        if abs(self.ising_offset) > 1e-9:
            pauli_list.append(("I" * n, float(self.ising_offset)))

        return SparsePauliOp.from_list(pauli_list)

    def evaluate_spin_energy(self, z_spins: np.ndarray) -> float:
        """
        Evaluates energy for z_i in {-1, +1}.
        z_i = +1 corresponds to x_i = 0 (unselected).
        z_i = -1 corresponds to x_i = 1 (selected).
        """
        energy = self.ising_offset
        for i in range(self.n):
            energy += self.h_vector[i] * z_spins[i]
            for j in range(i + 1, self.n):
                energy += self.J_matrix[i, j] * z_spins[i] * z_spins[j]
        return float(energy)

    def get_summary(self) -> Dict[str, Any]:
        """Returns details of the Cost Hamiltonian."""
        return {
            "num_qubits": self.n,
            "num_pauli_terms": len(self.cost_hamiltonian),
            "ising_offset": self.ising_offset,
            "h_vector": self.h_vector,
            "J_matrix": self.J_matrix,
            "hamiltonian": self.cost_hamiltonian
        }
