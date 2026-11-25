"""
QUBO (Quadratic Unconstrained Binary Optimization) formulation.
"""

from typing import Tuple, Dict, Any
import numpy as np
from .portfolio_problem import PortfolioProblem


class QUBOFormulation:
    """
    Transforms the constrained Portfolio Optimization problem into QUBO format:
        f(x) = x^T Q x + c_offset
    where x in {0, 1}^n.
    """

    def __init__(self, problem: PortfolioProblem):
        self.problem = problem
        self.n = problem.num_assets
        self.Q_matrix, self.offset = self._build_qubo()

    def _build_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Derives QUBO matrix Q and constant scalar offset.

        Problem formulation:
            Min f(x) = q * x^T Sigma x - mu^T x + lambda * (sum(x_i) - K)^2

        Expanding the penalty term:
            (sum(x_i) - K)^2 = sum_i x_i^2 + 2 sum_{i < j} x_i x_j - 2K sum_i x_i + K^2
            Since x_i in {0, 1}, x_i^2 = x_i:
            = sum_i (1 - 2K) x_i + 2 sum_{i < j} x_i x_j + K^2

        Expanding Markowitz term:
            q * sum_{i,j} Sigma_ij x_i x_j - sum_i mu_i x_i
            = sum_i (q * Sigma_ii - mu_i) x_i + 2 sum_{i < j} (q * Sigma_ij) x_i x_j

        Combining into upper-triangular Q:
            Diagonal elements (linear terms):
                Q[i, i] = q * Sigma_ii - mu_i + lambda * (1 - 2 * K)
            Off-diagonal elements (i < j):
                Q[i, j] = 2 * q * Sigma_ij + 2 * lambda
            Constant offset:
                offset = lambda * (K^2)
        """
        n = self.n
        q = self.problem.risk_factor
        mu = self.problem.expected_returns
        sigma = self.problem.covariance_matrix
        lam = self.problem.penalty_lambda
        k = self.problem.num_to_select

        Q = np.zeros((n, n), dtype=float)

        # Diagonal elements (linear terms)
        for i in range(n):
            Q[i, i] = q * sigma[i, i] - mu[i] + lam * (1.0 - 2.0 * k)

        # Off-diagonal elements (quadratic coupling terms, upper triangular)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = 2.0 * q * sigma[i, j] + 2.0 * lam

        offset = lam * (k ** 2)
        return Q, offset

    def evaluate_qubo_energy(self, x: np.ndarray) -> float:
        """Computes x^T Q x + offset."""
        x_vec = np.asarray(x, dtype=float)
        return float(np.dot(x_vec.T, np.dot(self.Q_matrix, x_vec)) + self.offset)

    def get_summary(self) -> Dict[str, Any]:
        """Returns details of the QUBO formulation."""
        return {
            "num_variables": self.n,
            "Q_matrix": self.Q_matrix,
            "offset": self.offset,
            "sparsity": float(np.count_nonzero(self.Q_matrix) / (self.n ** 2))
        }
