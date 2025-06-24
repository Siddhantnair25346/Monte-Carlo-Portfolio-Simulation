# Monte Carlo Simulation for a Multi-Asset Portfolio

This project models the future value of a multi-asset investment portfolio using Monte Carlo methods based on Geometric Brownian Motion (GBM). It simulates thousands of potential return paths under uncertainty, enabling robust portfolio-level risk analysis and performance evaluation.

---

## Project Overview

The simulation supports portfolios of any size and incorporates key risk-return inputs:
- Asset-level **expected returns** and **volatilities**
- A **correlation matrix** to reflect relationships between assets
- **Custom weightings** for each asset in the portfolio

It then performs 10,000+ simulations over a user-defined time horizon and visualizes potential outcomes through charts and risk metrics.

---

## Features

- **Multi-Asset Monte Carlo Engine**
  - Simulates price paths using GBM for any number of assets
  - Accounts for correlations between assets using Cholesky decomposition

- **Statistical Outputs**
  - Mean, median, 5th and 95th percentile final values
  - Standard deviation of portfolio outcomes

- **Risk Analysis**
  - Sharpe Ratio: excess return per unit of volatility
  - 95% Value at Risk (VaR): downside risk in worst-case scenarios

- **Interactive Visualizations**
  - Line chart of sample simulations
  - Histogram of final simulated portfolio values

---

## How to Use

1. Clone this repository or download the script
2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. Configure your portfolio:
   - Input expected returns, volatilities, weights, and a correlation matrix manually
4. Run the script:
   ```bash
   python monte_carlo_portfolio.py
   ```

---

## Example Portfolio Setup

```python
expected_returns = np.array([0.09, 0.13, 0.06])
volatilities = np.array([0.16, 0.25, 0.05])
weights = np.array([0.5, 0.3, 0.2])

correlation_matrix = np.array([
    [1.0, 0.3, 0.1],
    [0.3, 1.0, 0.2],
    [0.1, 0.2, 1.0]
])
```

---

## Applications

- Evaluating portfolio-level risk and return profiles
- Understanding diversification benefits via correlations
- Backtesting asset allocations under probabilistic assumptions
- Showcasing technical skills in quantitative finance and Python

---

## Future Improvements

- Scenario-based stress testing
- Rolling-window rebalancing
- Portfolio optimization under constraints
- Integration with Excel dashboards

---

## License

This project is for educational and non-commercial use.

---

## Acknowledgements

Built using Python, `numpy`, `pandas`, and `matplotlib`. Designed to support students and analysts exploring quantitative methods in investment management.
