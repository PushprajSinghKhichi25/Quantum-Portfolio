"""
Quantum Portfolio Optimization Project
Main executable script demonstrating Classical vs QAOA Quantum approaches.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.portfolio_data import (
    STOCK_TICKERS,
    STOCK_NAMES,
    EXPECTED_RETURNS,
    COVARIANCE_MATRIX,
    get_portfolio_dataframe
)
from core.portfolio_problem import PortfolioProblem
from core.qubo_formulation import QUBOFormulation
from core.ising_mapper import IsingMapper
from solvers.classical_solver import ClassicalSolver
from solvers.qaoa_solver import QAOAPortfolioSolver
from experiments.compare_experiments import run_benchmarks
import config


def print_banner():
    banner = r"""
===============================================================================
       QUANTUM PORTFOLIO OPTIMIZATION: QAOA VS CLASSICAL BENCHMARK
===============================================================================
  A hybrid quantum-classical algorithmic framework for selecting an optimal
  stock portfolio under Markowitz Mean-Variance and Cardinality Constraints.
===============================================================================
"""
    print(banner)


def main():
    print_banner()

    # -------------------------------------------------------------------------
    # 1. Display Problem Setup and Universe
    # -------------------------------------------------------------------------
    print("STEP 1: STOCK UNIVERSE & INVESTMENT PARAMETERS")
    print("-" * 75)
    print(f"Total Investment Budget   : \u20b9{config.TOTAL_BUDGET_INR:,.2f} (\u20b910 Lakhs)")
    print(f"Candidate Universe Size   : {len(STOCK_TICKERS)} Stocks")
    print(f"Stocks to Pick (K)        : {config.NUM_STOCKS_TO_SELECT} Stocks")
    print(f"Capital per Selected Stock: \u20b9{config.TOTAL_BUDGET_INR / config.NUM_STOCKS_TO_SELECT:,.2f} (\u20b92.5 Lakhs)")
    print(f"Risk Aversion Factor (q)  : {config.RISK_FACTOR_Q}")
    print(f"Penalty Multiplier (\u03bb)    : {config.PENALTY_LAMBDA}")
    print("\nShortlisted Indian Blue-Chip Companies:")
    stock_df = get_portfolio_dataframe()
    print(stock_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. Formulate Problem and Map to Quantum Operators
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 2: MATHEMATICAL FORMULATION & QUANTUM ISING MAPPING")
    print("-" * 75)
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

    print(f"[+] QUBO Dimension           : {qubo.n}x{qubo.n} matrix")
    print(f"[+] QUBO Constant Offset     : {qubo.offset:.4f}")
    print(f"[+] Qiskit SparsePauliOp     : {ising.cost_hamiltonian.num_qubits} qubits, {len(ising.cost_hamiltonian)} Pauli terms")
    print(f"[+] Ising Constant Offset    : {ising.ising_offset:.4f}")

    # -------------------------------------------------------------------------
    # 3. Classical Ground Truth (Brute Force)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3: CLASSICAL BENCHMARK (EXACT BRUTE FORCE)")
    print("-" * 75)
    classical = ClassicalSolver(problem, qubo)
    classical_res = classical.solve_exact_brute_force()

    print(f"[+] Evaluated all {classical_res['total_states_evaluated']} states ({classical_res['valid_states_count']} valid 4-stock combinations)")
    print(f"[+] Classical Execution Time : {classical_res['elapsed_time_sec'] * 1000:.3f} ms")
    
    best_c = classical_res["best_valid_portfolio"]
    print("\n>>> CLASSICAL OPTIMAL PORTFOLIO:")
    print(f"    Stocks Selected : {', '.join(best_c['selected_tickers'])}")
    print(f"    Expected Return : {best_c['portfolio_return_pct']:.2f}% per annum")
    print(f"    Portfolio Risk  : {best_c['portfolio_risk_pct']:.2f}% volatility")
    print(f"    Sharpe Ratio    : {best_c['sharpe_ratio']:.3f}")
    print(f"    Binary Bitstring: {best_c['bitstring']}")
    print(f"    Total Cost (H_C): {best_c['total_cost']:.4f}")

    print("\nTop 3 Valid Classical Portfolios:")
    summary_c_df = classical.get_summary_dataframe(classical_res).head(3)
    print(summary_c_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. Quantum QAOA Optimization (p=1 and p=2)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 4: QUANTUM QAOA ALGORITHM (CIRCUITS & HYBRID OPTIMIZATION)")
    print("-" * 75)
    qaoa = QAOAPortfolioSolver(problem, ising)

    # QAOA Depth p=1
    print(f"[*] Running QAOA Depth p=1 (Shots: {config.QAOA_SHOTS}, Optimizer: {config.CLASSICAL_OPTIMIZER})...")
    qaoa_p1 = qaoa.solve(p=1, shots=config.QAOA_SHOTS, max_iter=config.MAX_ITER, seed=config.SEED)
    best_p1 = qaoa_p1["best_sampled_valid_portfolio"]
    print(f"    Execution Time   : {qaoa_p1['elapsed_time_sec']:.2f} s")
    print(f"    Optimal \u03b3 (Gamma) : {[round(x, 4) for x in qaoa_p1['optimal_gammas']]}")
    print(f"    Optimal \u03b2 (Beta)  : {[round(x, 4) for x in qaoa_p1['optimal_betas']]}")
    print(f"    Feasibility Rate : {qaoa_p1['feasibility_rate_pct']:.2f}% of sampled shots satisfied K=4")
    if best_p1:
        print(f"    Best Valid Pick  : {', '.join(best_p1['selected_tickers'])} (Return: {best_p1['portfolio_return_pct']:.2f}%, Risk: {best_p1['portfolio_risk_pct']:.2f}%)")

    # QAOA Depth p=2
    print(f"\n[*] Running QAOA Depth p=2 (Shots: {config.QAOA_SHOTS}, Optimizer: {config.CLASSICAL_OPTIMIZER})...")
    qaoa_p2 = qaoa.solve(p=2, shots=config.QAOA_SHOTS, max_iter=config.MAX_ITER, seed=config.SEED)
    best_p2 = qaoa_p2["best_sampled_valid_portfolio"]
    print(f"    Execution Time   : {qaoa_p2['elapsed_time_sec']:.2f} s")
    print(f"    Optimal \u03b3 (Gamma) : {[round(x, 4) for x in qaoa_p2['optimal_gammas']]}")
    print(f"    Optimal \u03b2 (Beta)  : {[round(x, 4) for x in qaoa_p2['optimal_betas']]}")
    print(f"    Feasibility Rate : {qaoa_p2['feasibility_rate_pct']:.2f}% of sampled shots satisfied K=4")
    if best_p2:
        print(f"    Best Valid Pick  : {', '.join(best_p2['selected_tickers'])} (Return: {best_p2['portfolio_return_pct']:.2f}%, Risk: {best_p2['portfolio_risk_pct']:.2f}%)")

    print("\nTop 5 Quantum Sampled Configurations (QAOA p=2):")
    q_summary_df = qaoa.get_summary_dataframe(qaoa_p2, top_n=5)
    print(q_summary_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Side-by-Side Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 5: COMPREHENSIVE SIDE-BY-SIDE BENCHMARK COMPARISON")
    print("-" * 75)

    comp_data = [
        {
            "Metric / Property": "Selected Stocks",
            "Classical Exact": ", ".join(best_c["selected_tickers"]),
            "QAOA (p=1)": ", ".join(best_p1["selected_tickers"]) if best_p1 else "None",
            "QAOA (p=2)": ", ".join(best_p2["selected_tickers"]) if best_p2 else "None"
        },
        {
            "Metric / Property": "Bitstring",
            "Classical Exact": best_c["bitstring"],
            "QAOA (p=1)": best_p1["bitstring"] if best_p1 else "N/A",
            "QAOA (p=2)": best_p2["bitstring"] if best_p2 else "N/A"
        },
        {
            "Metric / Property": "Expected Return",
            "Classical Exact": f"{best_c['portfolio_return_pct']:.2f}%",
            "QAOA (p=1)": f"{best_p1['portfolio_return_pct']:.2f}%" if best_p1 else "N/A",
            "QAOA (p=2)": f"{best_p2['portfolio_return_pct']:.2f}%" if best_p2 else "N/A"
        },
        {
            "Metric / Property": "Portfolio Volatility",
            "Classical Exact": f"{best_c['portfolio_risk_pct']:.2f}%",
            "QAOA (p=1)": f"{best_p1['portfolio_risk_pct']:.2f}%" if best_p1 else "N/A",
            "QAOA (p=2)": f"{best_p2['portfolio_risk_pct']:.2f}%" if best_p2 else "N/A"
        },
        {
            "Metric / Property": "Sharpe Proxy",
            "Classical Exact": f"{best_c['sharpe_ratio']:.3f}",
            "QAOA (p=1)": f"{best_p1['sharpe_ratio']:.3f}" if best_p1 else "N/A",
            "QAOA (p=2)": f"{best_p2['sharpe_ratio']:.3f}" if best_p2 else "N/A"
        },
        {
            "Metric / Property": "Hamiltonian Cost",
            "Classical Exact": f"{best_c['total_cost']:.4f}",
            "QAOA (p=1)": f"{best_p1['total_cost']:.4f}" if best_p1 else "N/A",
            "QAOA (p=2)": f"{best_p2['total_cost']:.4f}" if best_p2 else "N/A"
        },
        {
            "Metric / Property": "Valid Budget Feasibility",
            "Classical Exact": "100% (Guaranteed)",
            "QAOA (p=1)": f"{qaoa_p1['feasibility_rate_pct']:.1f}%",
            "QAOA (p=2)": f"{qaoa_p2['feasibility_rate_pct']:.1f}%"
        },
        {
            "Metric / Property": "Execution Time",
            "Classical Exact": f"{classical_res['elapsed_time_sec'] * 1000:.2f} ms",
            "QAOA (p=1)": f"{qaoa_p1['elapsed_time_sec']:.2f} s",
            "QAOA (p=2)": f"{qaoa_p2['elapsed_time_sec']:.2f} s"
        }
    ]

    comp_df = pd.DataFrame(comp_data)
    print(comp_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 6. Generate Comparative Plots
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 6: GENERATING BENCHMARK VISUALIZATIONS")
    print("-" * 75)
    plot_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "portfolio_optimization_comparison.png")
    from experiments.compare_experiments import generate_plots
    generate_plots(problem, classical_res, qaoa_p1, qaoa_p2, plot_file)
    print(f"[+] Comparison dashboard generated at:\n    {plot_file}")

    print("\n" + "=" * 75)
    print("                 PROJECT EXECUTION COMPLETE")
    print("===============================================================================\n")


if __name__ == "__main__":
    main()
