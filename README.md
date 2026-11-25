# ⚛️ Quantum Portfolio Optimization: QAOA vs Classical Benchmark

A complete, production-grade implementation of **Constrained Markowitz Portfolio Optimization** using both **Classical Solvers (Exact Brute-Force)** and **Quantum Algorithms (QAOA with Qiskit 2.x)**.

---

## 🎯 1. Problem Statement

Given **₹10,00,000 (₹10 Lakhs)** to invest across **8 shortlisted Indian blue-chip companies**, select an **optimal subset of exactly 4 stocks** ($K = 4$, ₹2.5 Lakhs per stock) that **maximizes expected annual returns** while **minimizing portfolio volatility (risk)**.

### Candidate Universe (8 Companies)

| Ticker | Company | Sector | Expected Return ($\mu_i$) | Volatility ($\sigma_i$) |
| :--- | :--- | :--- | :---: | :---: |
| **RELIANCE** | Reliance Industries | Energy / Conglomerate | 12.0% | 20.00% |
| **TCS** | Tata Consultancy Services | IT Services | 15.0% | 21.91% |
| **INFY** | Infosys | IT Services | 14.0% | 22.36% |
| **HDFCBANK** | HDFC Bank | Banking | 11.0% | 20.49% |
| **ICICIBANK** | ICICI Bank | Banking | 16.0% | 22.80% |
| **ITC** | ITC Limited | FMCG | 9.0% | 15.81% |
| **WIPRO** | Wipro | IT Services | 13.0% | 21.45% |
| **TATAMOTORS** | Tata Motors | Automobile | 18.0% | 27.39% |

---

## 📐 2. Mathematical Formulation

### A. The Core Tension: Return vs. Risk

If an investor simply greedily picked the 4 highest-return stocks (**Tata Motors 18%, ICICI Bank 16%, TCS 15%, Infosys 14%**), two of the four stocks would be in IT services. Because tech stocks share high covariance, a tech downturn would severely damage the portfolio.

We use the **Markowitz Mean-Variance framework**:
$$\min_{x} \quad q \cdot x^T \Sigma x - \mu^T x$$
where:
- $x = [x_0, x_1, \dots, x_{n-1}]^T \in \{0, 1\}^n$ is the binary decision vector.
- $\Sigma$ is the $n \times n$ covariance matrix.
- $\mu$ is the expected returns vector.
- $q \in [0, 1]$ is the risk-aversion parameter ($q = 0.5$).

### B. Cardinality / Budget Constraint

The investor must select **exactly $K = 4$ stocks**:
$$\sum_{i=1}^n x_i = K$$

In unconstrained optimization frameworks (like QUBO and Quantum Annealing/QAOA), hard constraints are transformed into quadratic soft penalties:
$$P(x) = \lambda \left(\sum_{i=1}^n x_i - K\right)^2$$
where $\lambda > 0$ is the penalty weight. If $\sum x_i \neq K$, $P(x)$ adds a steep energy cost.

### C. Quadratic Unconstrained Binary Optimization (QUBO)

Expanding the penalty and grouping linear and quadratic terms:
$$\left(\sum_{i=1}^n x_i - K\right)^2 = \sum_{i=1}^n (1 - 2K) x_i + 2 \sum_{i < j} x_i x_j + K^2$$

This yields the complete QUBO objective function:
$$\min_{x} f(x) = \sum_{i=1}^n Q_{ii} x_i + \sum_{i < j} Q_{ij} x_i x_j + \text{offset}$$
where:
- **Diagonal (linear):** $Q_{ii} = q \Sigma_{ii} - \mu_i + \lambda (1 - 2K)$
- **Off-diagonal (coupling):** $Q_{ij} = 2 q \Sigma_{ij} + 2\lambda$
- **Constant scalar:** $\text{offset} = \lambda K^2$

---

## 🔬 3. Mapping QUBO to Quantum Ising Hamiltonian

To run this on a quantum computer, binary decision variables $x_i \in \{0, 1\}$ are mapped to quantum spin operators (Pauli-$Z$ matrices):
$$x_i \mapsto \frac{I - Z_i}{2}$$

Substituting this transformation into the QUBO function yields the **Cost Hamiltonian** $H_C$:
$$H_C = \sum_{i=0}^{n-1} h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + C_{\text{Ising}} \cdot I$$

The ground state (state with the minimum eigenvalue) of $H_C$ encodes the optimal portfolio allocation.

---

## ⚡ 4. The QAOA Quantum Algorithm

The **Quantum Approximate Optimization Algorithm (QAOA)** is a hybrid quantum-classical algorithm:

1. **State Initialization:**
   Initialize all $n$ qubits into an equal superposition:
   $$|\psi_0\rangle = H^{\otimes n} |0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0, 1\}^n} |x\rangle$$

2. **Parameterized Unitary Evolution (Depth $p$):**
   Alternately apply the Problem Unitary $U(H_C, \gamma)$ and Mixer Unitary $U(H_M, \beta)$:
   $$|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{l=1}^p \left( e^{-i \beta_l H_M} e^{-i \gamma_l H_C} \right) |\psi_0\rangle$$
   - **Cost Phase Separator:** $e^{-i \gamma_l H_C}$ applies single-qubit $R_Z(2 \gamma_l h_i)$ and two-qubit $R_{ZZ}(2 \gamma_l J_{ij})$ gates.
   - **Mixer Unitary:** $H_M = \sum_{i=0}^{n-1} X_i$ applies single-qubit transverse rotations $R_X(2 \beta_l)$.

3. **Classical Optimization Loop:**
   Evaluate the expectation value $\langle H_C \rangle = \langle \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) | H_C | \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) \rangle$ and use a classical optimizer (**COBYLA**) to tune $(\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*)$.

4. **Measurement & Sampling:**
   Measure the optimized quantum state in the computational basis to obtain the highest-probability portfolio bitstrings.

---

## 📁 5. Project Directory Structure

```
Quantum Portfolio/
├── README.md               <- Comprehensive documentation and guide
├── requirements.txt        <- Python dependencies
├── config.py               <- Budget, stock count K, lambda, and QAOA parameters
├── main.py                 <- Main executable driver script
├── data/
│   ├── __init__.py
│   └── portfolio_data.py   <- 8 shortlisted Indian stocks and covariance matrix
├── core/
│   ├── __init__.py
│   ├── portfolio_problem.py <- Financial metrics, Sharpe proxy, cost evaluator
│   ├── qubo_formulation.py  <- Derives Q matrix and offset
│   └── ising_mapper.py      <- Converts QUBO to Qiskit SparsePauliOp
├── solvers/
│   ├── __init__.py
│   ├── classical_solver.py  <- Exhaustive brute-force (256 states, 70 valid)
│   └── qaoa_solver.py       <- Parameterized Qiskit QAOA circuit & optimizer
├── experiments/
│   ├── __init__.py
│   └── compare_experiments.py <- Benchmarking suite and Matplotlib plotter
└── results/
    └── portfolio_optimization_comparison.png <- Generated 4-panel dashboard
```

---

## 🚀 6. How to Run the Project

### A. Activate the Virtual Environment
Open your terminal and navigate to the project directory:
```bash
cd "$HOME/Downloads/Quantum Portfolio"
source venv/bin/activate
```

### B. Run the Full Comparison
```bash
python main.py
```

### C. Run the Benchmark Experiment Suite
```bash
python experiments/compare_experiments.py
```

---

## 📊 7. Benchmark Results & Key Takeaways

### Classical vs Quantum Comparison Table

| Metric / Property | Classical Exact | QAOA ($p=1$) | QAOA ($p=2$) |
| :--- | :---: | :---: | :---: |
| **Selected Stocks** | **TCS, ICICI Bank, ITC, Tata Motors** | **TCS, ICICI Bank, ITC, Tata Motors** | **TCS, ICICI Bank, ITC, Tata Motors** |
| **Optimal Bitstring** | `01001101` | `01001101` | `01001101` |
| **Expected Return** | **14.50%** | **14.50%** | **14.50%** |
| **Portfolio Risk (Volatility)** | **14.40%** | **14.40%** | **14.40%** |
| **Sharpe Ratio Proxy** | **1.007** | **1.007** | **1.007** |
| **Ground Energy** | **-0.4140** | **-0.4140** | **-0.4140** |
| **Budget Feasibility Rate** | 100% (Guaranteed) | 60.5% | 51.3% |
| **Execution Time** | ~1.8 ms | ~0.04 s | ~0.12 s |

### Key Financial Insights
1. **The Power of Diversification:** Notice that **Infosys** (14% return) was dropped in favor of **ITC** (9% return). Why? Because holding TCS, ICICI Bank, and Tata Motors along with ITC gives a portfolio risk of only **14.40%**, whereas adding Infosys would duplicate tech sector risk and reduce the Sharpe ratio.
2. **Quantum Accuracy:** The QAOA quantum circuit with depth $p=1$ and $p=2$ **correctly identifies the exact ground-truth global optimum** (`01001101`).
3. **The Soft Constraint Tradeoff:** Unlike classical brute-force which hard-filters for $\sum x_i = 4$, QAOA explores the entire Hilbert space ($2^8 = 256$ states). The penalty term $\lambda = 3.0$ concentrates $>50–60\%$ of total measurement shots strictly into valid 4-stock configurations.
