"""
Runs benchmarking experiments comparing Classical Brute Force vs QAOA (p=1 and p=2),
and generates comparative visualization plots saved in the results/ folder.
"""

import os
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from data.portfolio_data import (
    STOCK_TICKERS,
    EXPECTED_RETURNS,
    COVARIANCE_MATRIX
)
from core.portfolio_problem import PortfolioProblem
from core.qubo_formulation import QUBOFormulation
from core.ising_mapper import IsingMapper
from solvers.classical_solver import ClassicalSolver
from solvers.qaoa_solver import QAOAPortfolioSolver
import config


def run_benchmarks(save_plot: bool = True) -> Dict[str, Any]:
    """Runs classical and quantum benchmarks and produces comparative figures."""
    print("=" * 75)
    print("     QUANTUM PORTFOLIO OPTIMIZATION: BENCHMARK SUITE")
    print("=" * 75)

    # 1. Initialize Problem, QUBO, and Ising Mapper
    problem = PortfolioProblem(
        tickers=STOCK_TICKERS,
        expected_returns=EXPECTED_RETURNS,
        covariance_matrix=COVARIANCE_MATRIX,
        num_to_select=config.NUM_STOCKS_TO_SELECT,
        risk_factor=config.RISK_FACTOR_Q,
        penalty_lambda=config.PENALTY_LAMBDA,
        total_budget=config.TOTAL_BUDGET_INR
    )
    qubo = QUBOFormulation(problem)
    ising = IsingMapper(qubo)

    # 2. Run Classical Solver
    print("\n[1/3] Running Classical Exact Solver (Exhaustive Brute Force)...")
    classical_solver = ClassicalSolver(problem, qubo)
    classical_res = classical_solver.solve_exact_brute_force()
    best_class = classical_res["best_valid_portfolio"]
    print(f"  -> Found optimal portfolio in {classical_res['elapsed_time_sec'] * 1000:.2f} ms")
    print(f"  -> Selected: {', '.join(best_class['selected_tickers'])}")
    print(f"  -> Return: {best_class['portfolio_return_pct']:.2f}%, Risk: {best_class['portfolio_risk_pct']:.2f}%")

    # 3. Run QAOA (p=1)
    print("\n[2/3] Running Quantum QAOA Solver (Depth p=1)...")
    qaoa_solver = QAOAPortfolioSolver(problem, ising)
    qaoa_p1 = qaoa_solver.solve(p=1, shots=config.QAOA_SHOTS, max_iter=config.MAX_ITER, seed=config.SEED)
    best_qaoa_p1 = qaoa_p1["best_sampled_valid_portfolio"]
    print(f"  -> Optimization finished in {qaoa_p1['elapsed_time_sec']:.2f} s")
    print(f"  -> Feasibility Rate: {qaoa_p1['feasibility_rate_pct']:.2f}%")
    if best_qaoa_p1:
        print(f"  -> Top valid: {', '.join(best_qaoa_p1['selected_tickers'])} (Prob: {best_qaoa_p1['probability'] * 100:.2f}%)")

    # 4. Run QAOA (p=2)
    print("\n[3/3] Running Quantum QAOA Solver (Depth p=2)...")
    qaoa_p2 = qaoa_solver.solve(p=2, shots=config.QAOA_SHOTS, max_iter=config.MAX_ITER, seed=config.SEED)
    best_qaoa_p2 = qaoa_p2["best_sampled_valid_portfolio"]
    print(f"  -> Optimization finished in {qaoa_p2['elapsed_time_sec']:.2f} s")
    print(f"  -> Feasibility Rate: {qaoa_p2['feasibility_rate_pct']:.2f}%")
    if best_qaoa_p2:
        print(f"  -> Top valid: {', '.join(best_qaoa_p2['selected_tickers'])} (Prob: {best_qaoa_p2['probability'] * 100:.2f}%)")

    # 5. Visualizations
    if save_plot:
        plot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            "portfolio_optimization_comparison.png"
        )
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        generate_plots(problem, classical_res, qaoa_p1, qaoa_p2, plot_path)
        print(f"\n[+] Visualizations successfully saved to:\n    {plot_path}")

    return {
        "problem": problem,
        "classical": classical_res,
        "qaoa_p1": qaoa_p1,
        "qaoa_p2": qaoa_p2
    }


def generate_plots(
    problem: PortfolioProblem,
    classical_res: Dict[str, Any],
    qaoa_p1: Dict[str, Any],
    qaoa_p2: Dict[str, Any],
    save_path: str
):
    """Generates a 4-panel dashboard comparing Classical and Quantum QAOA results."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Quantum vs Classical Portfolio Optimization Dashboard", fontsize=18, fontweight="bold", y=0.98)

    # -------------------------------------------------------------------------
    # Panel 1: Markowitz Efficient Frontier & Portfolio Space
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    valid_states = classical_res["valid_states"]
    risks = [item["portfolio_risk_pct"] for item in valid_states]
    returns = [item["portfolio_return_pct"] for item in valid_states]
    sharpes = [item["sharpe_ratio"] for item in valid_states]

    sc = ax1.scatter(risks, returns, c=sharpes, cmap="viridis", s=60, alpha=0.6, edgecolors="none", label="All 70 Valid Portfolios")
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label("Sharpe Ratio Proxy", fontsize=10)

    # Highlight Classical Optimum
    best_c = classical_res["best_valid_portfolio"]
    ax1.scatter(
        best_c["portfolio_risk_pct"],
        best_c["portfolio_return_pct"],
        color="crimson",
        marker="*",
        s=320,
        edgecolors="black",
        linewidth=1.5,
        label=f"Classical Optimal ({', '.join(best_c['selected_tickers'][:2])}...)",
        zorder=5
    )

    # Highlight QAOA p=2 Best
    best_q2 = qaoa_p2["best_sampled_valid_portfolio"]
    if best_q2:
        ax1.scatter(
            best_q2["portfolio_risk_pct"],
            best_q2["portfolio_return_pct"],
            color="dodgerblue",
            marker="o",
            s=140,
            facecolors="none",
            edgecolors="blue",
            linewidth=2.5,
            label=f"QAOA (p=2) Selected",
            zorder=6
        )

    ax1.set_title("1. Markowitz Risk vs Return Space (All 4-Stock Subsets)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Portfolio Risk / Volatility (%)", fontsize=11)
    ax1.set_ylabel("Expected Return (%)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # Panel 2: QAOA Optimization Convergence (Energy vs Iteration)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    hist_p1 = qaoa_p1["optimization_history"]
    hist_p2 = qaoa_p2["optimization_history"]

    iters_p1 = [h["iteration"] for h in hist_p1]
    energies_p1 = [h["energy"] for h in hist_p1]
    iters_p2 = [h["iteration"] for h in hist_p2]
    energies_p2 = [h["energy"] for h in hist_p2]

    ax2.plot(iters_p1, energies_p1, label="QAOA Depth p=1", color="darkorange", linewidth=2.0)
    ax2.plot(iters_p2, energies_p2, label="QAOA Depth p=2", color="royalblue", linewidth=2.0)

    # Ground state energy line
    min_energy = classical_res["best_global_state"]["total_cost"]
    ax2.axhline(min_energy, color="forestgreen", linestyle="--", linewidth=1.8, label=f"Ground State Energy ({min_energy:.2f})")

    ax2.set_title("2. Classical-Quantum Hybrid Optimization Convergence", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Classical Optimizer Iteration (COBYLA)", fontsize=11)
    ax2.set_ylabel(r"Expectation Value $\langle H_C \rangle$", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)

    # -------------------------------------------------------------------------
    # Panel 3: QAOA (p=2) Measurement Probability Distribution (Top 10 Bitstrings)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    top_samples = qaoa_p2["sampled_results"][:10]
    bitstrings = [s["bitstring"] for s in top_samples]
    probs = [s["probability"] * 100 for s in top_samples]
    is_valids = [s["is_valid"] for s in top_samples]

    colors = ["#2ecc71" if v else "#e74c3c" for v in is_valids]
    bars = ax3.bar(range(len(bitstrings)), probs, color=colors, edgecolor="black", alpha=0.85)
    ax3.set_xticks(range(len(bitstrings)))
    ax3.set_xticklabels(bitstrings, rotation=45, ha="right", fontsize=9, family="monospace")

    # Add legend manually for valid vs invalid
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Valid Portfolio (4 Stocks)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Invalid (Violates Constraint)")
    ]
    ax3.legend(handles=legend_elements, loc="upper right", frameon=True)

    ax3.set_title("3. Sampled State Probability Distribution (QAOA p=2)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Sampled Bitstring [REL, TCS, INF, HDFC, ICICI, ITC, WIP, TATAM]", fontsize=10)
    ax3.set_ylabel("Sampling Probability (%)", fontsize=11)

    # -------------------------------------------------------------------------
    # Panel 4: Comparative Performance Metrics Summary
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    categories = ["Expected Return (%)", "Risk (%)", "Sharpe Ratio", "Feasibility Rate (%)"]
    
    classical_vals = [
        best_c["portfolio_return_pct"],
        best_c["portfolio_risk_pct"],
        best_c["sharpe_ratio"] * 10, # scaled for visual balance
        100.0 # Classical always 100% feasible
    ]

    p1_best = qaoa_p1["best_sampled_valid_portfolio"]
    qaoa_p1_vals = [
        p1_best["portfolio_return_pct"] if p1_best else 0.0,
        p1_best["portfolio_risk_pct"] if p1_best else 0.0,
        (p1_best["sharpe_ratio"] * 10) if p1_best else 0.0,
        qaoa_p1["feasibility_rate_pct"]
    ]

    p2_best = qaoa_p2["best_sampled_valid_portfolio"]
    qaoa_p2_vals = [
        p2_best["portfolio_return_pct"] if p2_best else 0.0,
        p2_best["portfolio_risk_pct"] if p2_best else 0.0,
        (p2_best["sharpe_ratio"] * 10) if p2_best else 0.0,
        qaoa_p2["feasibility_rate_pct"]
    ]

    x = np.arange(len(categories))
    width = 0.25

    ax4.bar(x - width, classical_vals, width, label="Classical Exact", color="#34495e")
    ax4.bar(x, qaoa_p1_vals, width, label="QAOA (p=1)", color="#e67e22")
    ax4.bar(x + width, qaoa_p2_vals, width, label="QAOA (p=2)", color="#3498db")

    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.set_ylabel("Metric Value / Scale", fontsize=11)
    ax4.set_title("4. Classical vs Quantum Metric Comparison", fontsize=13, fontweight="bold")
    ax4.legend(loc="upper right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_benchmarks(save_plot=True)
