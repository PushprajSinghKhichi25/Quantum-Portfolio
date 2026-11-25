"""
Configuration settings for the Quantum Portfolio Optimization project.
"""

TOTAL_BUDGET_INR = 1_000_000  # ₹10 Lakhs
NUM_STOCKS_TO_SELECT = 4      # K = 4 stocks out of 8
RISK_FACTOR_Q = 0.5           # Risk aversion parameter (q)
PENALTY_LAMBDA = 3.0          # Penalty multiplier for budget constraint violation

# QAOA Settings
QAOA_DEFAULT_DEPTH = 2        # p = circuit layers
QAOA_SHOTS = 4096             # Measurement shots for simulator
CLASSICAL_OPTIMIZER = "COBYLA" # Classical optimizer for variational parameters
MAX_ITER = 80                 # Maximum iterations for classical optimizer
SEED = 42
