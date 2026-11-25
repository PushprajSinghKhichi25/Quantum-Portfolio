"""
Classical Solvers for Portfolio Optimization:
Exhaustive Brute Force Ground Truth and Quadratic Optimization.
"""

from typing import Dict, Any, List
import itertools
import time
import numpy as np
import pandas as pd
from core.portfolio_problem import PortfolioProblem
from core.qubo_formulation import QUBOFormulation


class ClassicalSolver:
    """
    Computes exact ground-truth solutions using classical brute-force search
    and unconstrained/constrained quadratic analysis.
    """

    def __init__(self, problem: PortfolioProblem, qubo: QUBOFormulation):
        self.problem = problem
        self.qubo = qubo
        self.n = problem.num_assets

    def solve_exact_brute_force(self) -> Dict[str, Any]:
        """
        Evaluates all 2^N possible bitstrings (256 states for N=8).
        Identifies both:
          1. The global minimum of the QUBO energy landscape (all states).
          2. The optimal valid portfolio (states strictly with sum(x_i) == K).
        """
        start_time = time.perf_counter()

        all_results: List[Dict[str, Any]] = []
        valid_results: List[Dict[str, Any]] = []

        # Generate all 2^n bit combinations
        for bits in itertools.product([0, 1], repeat=self.n):
            bit_arr = np.array(bits, dtype=int)
            eval_dict = self.problem.evaluate_allocation(bit_arr)
            qubo_energy = self.qubo.evaluate_qubo_energy(bit_arr)
            eval_dict["qubo_energy"] = qubo_energy

            all_results.append(eval_dict)
            if eval_dict["is_valid"]:
                valid_results.append(eval_dict)

        elapsed_time = time.perf_counter() - start_time

        # Sort all states by QUBO cost (lower is better)
        all_sorted = sorted(all_results, key=lambda x: x["total_cost"])
        # Sort valid states by total cost (or equivalently Markowitz cost since penalty is 0)
        valid_sorted = sorted(valid_results, key=lambda x: x["total_cost"])

        best_global = all_sorted[0]
        best_valid = valid_sorted[0]

        # Top 5 valid portfolios
        top_5_valid = valid_sorted[:5]

        return {
            "elapsed_time_sec": elapsed_time,
            "total_states_evaluated": len(all_results),
            "valid_states_count": len(valid_results),
            "best_global_state": best_global,
            "best_valid_portfolio": best_valid,
            "all_states": all_results,
            "valid_states": valid_results,
            "top_5_valid": top_5_valid
        }

    def get_summary_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Formats the top valid portfolios into a clean readable table."""
        rows = []
        for rank, item in enumerate(results["top_5_valid"], 1):
            rows.append({
                "Rank": rank,
                "Selected Stocks": ", ".join(item["selected_tickers"]),
                "Expected Return (%)": f"{item['portfolio_return_pct']:.2f}%",
                "Portfolio Risk (%)": f"{item['portfolio_risk_pct']:.2f}%",
                "Sharpe Ratio": f"{item['sharpe_ratio']:.3f}",
                "QUBO Energy": f"{item['total_cost']:.4f}",
                "Bitstring": item["bitstring"]
            })
        return pd.DataFrame(rows)
