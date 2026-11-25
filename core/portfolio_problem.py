"""
Mathematical representation of the Portfolio Optimization problem.
"""

from typing import Dict, Any, List, Union
import numpy as np


class PortfolioProblem:
    """
    Defines the constrained Markowitz portfolio optimization problem:
    Maximize Expected Returns while Minimizing Portfolio Variance (Risk)
    Subject to Budget/Cardinality constraint: Select exactly K stocks.
    """

    def __init__(
        self,
        tickers: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        num_to_select: int = 4,
        risk_factor: float = 0.5,
        penalty_lambda: float = 3.0,
        total_budget: float = 1_000_000.0
    ):
        self.tickers = list(tickers)
        self.num_assets = len(tickers)
        self.expected_returns = np.asarray(expected_returns, dtype=float)
        self.covariance_matrix = np.asarray(covariance_matrix, dtype=float)
        self.num_to_select = int(num_to_select)
        self.risk_factor = float(risk_factor)
        self.penalty_lambda = float(penalty_lambda)
        self.total_budget = float(total_budget)

    def evaluate_allocation(self, x: Union[np.ndarray, List[int], str]) -> Dict[str, Any]:
        """
        Evaluates a binary decision vector x.
        x can be a numpy array, list of 0/1s, or a bitstring (e.g., '11001001').
        """
        if isinstance(x, str):
            x_arr = np.array([int(b) for b in x], dtype=float)
        else:
            x_arr = np.asarray(x, dtype=float)

        num_selected = int(np.sum(x_arr))
        is_valid = (num_selected == self.num_to_select)

        # Selected tickers
        selected_indices = np.where(x_arr > 0.5)[0]
        selected_tickers = [self.tickers[i] for i in selected_indices]

        # Financial metrics (assuming equal capital allocation across selected stocks)
        if num_selected > 0:
            weights = x_arr / num_selected
            portfolio_return = float(np.dot(weights, self.expected_returns))
            portfolio_variance = float(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            portfolio_risk = float(np.sqrt(max(portfolio_variance, 0.0)))
            sharpe_proxy = portfolio_return / portfolio_risk if portfolio_risk > 1e-6 else 0.0
            per_stock_budget = self.total_budget / num_selected
        else:
            portfolio_return = 0.0
            portfolio_risk = 0.0
            sharpe_proxy = 0.0
            per_stock_budget = 0.0

        # Unconstrained Markowitz Objective: q * x^T Sigma x - mu^T x
        markowitz_cost = (
            self.risk_factor * float(np.dot(x_arr.T, np.dot(self.covariance_matrix, x_arr)))
            - float(np.dot(self.expected_returns, x_arr))
        )

        # Constraint Penalty: lambda * (sum(x_i) - K)^2
        constraint_violation = num_selected - self.num_to_select
        penalty_cost = self.penalty_lambda * (constraint_violation ** 2)

        # Total QUBO Cost: lower is better
        total_cost = markowitz_cost + penalty_cost

        return {
            "bitstring": "".join(str(int(b)) for b in x_arr),
            "binary_vector": x_arr.astype(int),
            "num_selected": num_selected,
            "is_valid": is_valid,
            "selected_tickers": selected_tickers,
            "portfolio_return_pct": portfolio_return * 100,
            "portfolio_risk_pct": portfolio_risk * 100,
            "sharpe_ratio": sharpe_proxy,
            "markowitz_cost": markowitz_cost,
            "penalty_cost": penalty_cost,
            "total_cost": total_cost,
            "per_stock_budget": per_stock_budget
        }
