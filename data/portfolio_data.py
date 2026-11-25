"""
Stock data and covariance matrix definition for the 8 shortlisted Indian companies.
"""

import numpy as np
import pandas as pd

# Shortlisted 8 companies
STOCK_TICKERS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "ITC",
    "WIPRO",
    "TATAMOTORS"
]

STOCK_NAMES = [
    "Reliance Industries",
    "Tata Consultancy Services",
    "Infosys",
    "HDFC Bank",
    "ICICI Bank",
    "ITC Limited",
    "Wipro",
    "Tata Motors"
]

SECTORS = [
    "Energy / Conglomerate",
    "IT Services",
    "IT Services",
    "Banking",
    "Banking",
    "FMCG",
    "IT Services",
    "Automobile"
]

# Expected annual returns from problem description
EXPECTED_RETURNS = np.array([
    0.12,  # Reliance: 12%
    0.15,  # TCS: 15%
    0.14,  # Infosys: 14%
    0.11,  # HDFC Bank: 11%
    0.16,  # ICICI Bank: 16%
    0.09,  # ITC: 9%
    0.13,  # Wipro: 13%
    0.18   # Tata Motors: 18%
], dtype=float)

# Annualized Covariance Matrix reflecting sector correlations
# IT stocks (TCS, INFY, WIPRO) have high mutual covariance.
# Banking stocks (HDFCBANK, ICICIBANK) have high mutual covariance.
# ITC (FMCG) has defensive low covariance.
# Tata Motors has higher volatility.
COVARIANCE_MATRIX = np.array([
    # REL    TCS    INFY   HDFC   ICICI  ITC    WIPRO  TATAM
    [0.040, 0.012, 0.011, 0.015, 0.016, 0.006, 0.010, 0.018], # RELIANCE
    [0.012, 0.048, 0.042, 0.010, 0.012, 0.005, 0.038, 0.014], # TCS
    [0.011, 0.042, 0.050, 0.009, 0.011, 0.004, 0.036, 0.015], # INFY
    [0.015, 0.010, 0.009, 0.042, 0.038, 0.007, 0.008, 0.016], # HDFCBANK
    [0.016, 0.012, 0.011, 0.038, 0.052, 0.008, 0.010, 0.020], # ICICIBANK
    [0.006, 0.005, 0.004, 0.007, 0.008, 0.025, 0.004, 0.007], # ITC
    [0.010, 0.038, 0.036, 0.008, 0.010, 0.004, 0.046, 0.013], # WIPRO
    [0.018, 0.014, 0.015, 0.016, 0.020, 0.007, 0.013, 0.075]  # TATAMOTORS
], dtype=float)


def get_portfolio_dataframe() -> pd.DataFrame:
    """Returns a clean overview DataFrame of all available assets."""
    volatilities = np.sqrt(np.diag(COVARIANCE_MATRIX))
    df = pd.DataFrame({
        "Ticker": STOCK_TICKERS,
        "Company": STOCK_NAMES,
        "Sector": SECTORS,
        "Expected Return (%)": EXPECTED_RETURNS * 100,
        "Volatility/Risk (%)": np.round(volatilities * 100, 2)
    })
    return df


def get_covariance_dataframe() -> pd.DataFrame:
    """Returns the covariance matrix as a labeled DataFrame."""
    return pd.DataFrame(COVARIANCE_MATRIX, index=STOCK_TICKERS, columns=STOCK_TICKERS)
