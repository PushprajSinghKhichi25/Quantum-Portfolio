"""
QAOA (Quantum Approximate Optimization Algorithm) Solver for Portfolio Optimization.
Constructs parameterized quantum circuits and optimizes variational angles.
"""

from typing import Dict, Any, List, Tuple
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from core.portfolio_problem import PortfolioProblem
from core.ising_mapper import IsingMapper


class QAOAPortfolioSolver:
    """
    Implements QAOA using Qiskit circuits and classical-quantum hybrid loop.
    Supports arbitrary depth p, parameter optimization via COBYLA,
    and statevector/shot-based sampling.
    """

    def __init__(self, problem: PortfolioProblem, ising: IsingMapper):
        self.problem = problem
        self.ising = ising
        self.n = ising.n
        self.hamiltonian = ising.cost_hamiltonian
        self.h_vector = ising.h_vector
        self.J_matrix = ising.J_matrix

    def build_qaoa_circuit(self, p: int) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        r"""
        Builds the parameterized QAOA quantum circuit of depth p:
            |psi(gamma, beta)> = prod_{l=1}^p [ e^{-i beta_l H_M} e^{-i gamma_l H_C} ] H^{\otimes n} |0>

        Returns:
            qc: The parameterized QuantumCircuit
            gamma_params: ParameterVector for cost unitary phases
            beta_params: ParameterVector for mixer unitary phases
        """
        qc = QuantumCircuit(self.n)

        gamma = ParameterVector("gamma", p)
        beta = ParameterVector("beta", p)

        # 1. State preparation: Uniform superposition |+>^n
        qc.h(range(self.n))

        # 2. Alternating layers of Cost Unitary and Mixer Unitary
        for l in range(p):
            # --- Cost Hamiltonian Unitary U(H_C, gamma_l) ---
            # Single-qubit Z rotations: exp(-i * gamma * h_i * Z_i)
            for i in range(self.n):
                coeff = float(self.h_vector[i])
                if abs(coeff) > 1e-9:
                    qc.rz(2.0 * gamma[l] * coeff, i)

            # Two-qubit ZZ coupling rotations: exp(-i * gamma * J_ij * Z_i Z_j)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    coeff = float(self.J_matrix[i, j])
                    if abs(coeff) > 1e-9:
                        qc.rzz(2.0 * gamma[l] * coeff, i, j)

            # --- Mixer Hamiltonian Unitary U(H_M, beta_l) ---
            # Single-qubit X rotations: exp(-i * beta * X_i)
            for i in range(self.n):
                qc.rx(2.0 * beta[l], i)

        return qc, gamma, beta

    def solve(
        self,
        p: int = 1,
        shots: int = 4096,
        max_iter: int = 80,
        initial_params: np.ndarray = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs the classical-quantum optimization loop to find optimal gamma and beta.
        """
        start_time = time.perf_counter()
        qc, gamma_params, beta_params = self.build_qaoa_circuit(p)

        # Initial random or heuristic parameter guess
        if initial_params is None:
            np.random.seed(seed)
            # Gammas typically in [0, 2pi], Betas in [0, pi]
            init_gamma = np.random.uniform(0.1, np.pi, p)
            init_beta = np.random.uniform(0.1, np.pi / 2, p)
            params_0 = np.concatenate([init_gamma, init_beta])
        else:
            params_0 = np.asarray(initial_params, dtype=float)

        optimization_history: List[Dict[str, Any]] = []

        # Objective function for classical optimizer: compute <psi(gamma, beta)| H_C |psi(gamma, beta)>
        def cost_function(param_values: np.ndarray) -> float:
            gammas = param_values[:p]
            betas = param_values[p:]

            param_dict = {}
            for l in range(p):
                param_dict[gamma_params[l]] = gammas[l]
                param_dict[beta_params[l]] = betas[l]

            bound_circuit = qc.assign_parameters(param_dict)
            state = Statevector(bound_circuit)
            energy = float(state.expectation_value(self.hamiltonian).real)

            optimization_history.append({
                "iteration": len(optimization_history) + 1,
                "energy": energy,
                "gammas": list(gammas),
                "betas": list(betas)
            })
            return energy

        # Run classical optimization
        opt_result = minimize(
            cost_function,
            params_0,
            method="COBYLA",
            options={"maxiter": max_iter, "disp": False}
        )

        optimal_params = opt_result.x
        optimal_gammas = optimal_params[:p]
        optimal_betas = optimal_params[p:]
        final_energy = float(opt_result.fun)

        # Bind optimal parameters to generate final quantum state
        final_param_dict = {}
        for l in range(p):
            final_param_dict[gamma_params[l]] = optimal_gammas[l]
            final_param_dict[beta_params[l]] = optimal_betas[l]

        optimal_circuit = qc.assign_parameters(final_param_dict)
        final_state = Statevector(optimal_circuit)

        # Sample measurement counts (Qiskit string convention: rightmost bit is qubit 0)
        np.random.seed(seed)
        raw_counts = final_state.sample_counts(shots=shots)

        # Process measurement samples
        sampled_results: List[Dict[str, Any]] = []
        valid_samples_count = 0
        best_sampled_valid = None
        min_valid_cost = float("inf")

        for qiskit_bitstr, count in raw_counts.items():
            # Convert Qiskit little-endian string to forward asset order [x_0, x_1, ..., x_{n-1}]
            # In Qiskit, index k of the asset is at qiskit_bitstr[n - 1 - k]
            forward_bitstr = "".join(qiskit_bitstr[self.n - 1 - k] for k in range(self.n))
            eval_dict = self.problem.evaluate_allocation(forward_bitstr)
            probability = count / shots

            entry = {
                **eval_dict,
                "raw_qiskit_bitstring": qiskit_bitstr,
                "shots_count": count,
                "probability": probability
            }
            sampled_results.append(entry)

            if eval_dict["is_valid"]:
                valid_samples_count += count
                if eval_dict["total_cost"] < min_valid_cost:
                    min_valid_cost = eval_dict["total_cost"]
                    best_sampled_valid = entry

        # Feasibility rate: percentage of sampled shots satisfying K = 4
        feasibility_rate = (valid_samples_count / shots) * 100.0

        # Sort sampled results by sampling probability (most frequent first)
        sampled_results.sort(key=lambda x: x["shots_count"], reverse=True)

        elapsed_time = time.perf_counter() - start_time

        return {
            "p": p,
            "elapsed_time_sec": elapsed_time,
            "optimal_gammas": optimal_gammas,
            "optimal_betas": optimal_betas,
            "final_expected_energy": final_energy,
            "shots": shots,
            "feasibility_rate_pct": feasibility_rate,
            "most_frequent_state": sampled_results[0],
            "best_sampled_valid_portfolio": best_sampled_valid,
            "sampled_results": sampled_results,
            "optimization_history": optimization_history,
            "circuit_depth": optimal_circuit.depth(),
            "circuit_gate_count": dict(optimal_circuit.count_ops())
        }

    def get_summary_dataframe(self, qaoa_result: Dict[str, Any], top_n: int = 5) -> pd.DataFrame:
        """Formats the top sampled configurations into a clean readable table."""
        rows = []
        for rank, item in enumerate(qaoa_result["sampled_results"][:top_n], 1):
            stocks_str = ", ".join(item["selected_tickers"]) if item["selected_tickers"] else "None"
            rows.append({
                "Rank": rank,
                "Bitstring": item["bitstring"],
                "Valid?": "YES" if item["is_valid"] else f"NO ({item['num_selected']} stocks)",
                "Probability": f"{item['probability'] * 100:.2f}%",
                "Counts": item["shots_count"],
                "Return (%)": f"{item['portfolio_return_pct']:.2f}%" if item["is_valid"] else "N/A",
                "Risk (%)": f"{item['portfolio_risk_pct']:.2f}%" if item["is_valid"] else "N/A",
                "Selected Stocks": stocks_str
            })
        return pd.DataFrame(rows)
