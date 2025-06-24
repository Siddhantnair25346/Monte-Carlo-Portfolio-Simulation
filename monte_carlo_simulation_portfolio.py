import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def monte_carlo_portfolio_simulation(
    initial_investment: float,
    expected_returns: float,
    volatilities: float,
    weights : np.ndarray,
    correlation_matrix: np.ndarray,
    time_horizon_years: float,
    num_simulations: int = 10000,
    time_steps_per_year: int = 252
) -> pd.DataFrame:
    """
    Monte Carlo simulation for a portfolio of assets.

    - expected_returns: Array of annual expected returns for each asset
    - volatilities: Array of annual volatilities for each asset
    - weights: Array of portfolio weights (should sum to 1)
    - correlation_matrix: Covariance or correlation matrix between assets
    """

    # Time configuration
    total_steps = int(time_horizon_years * time_steps_per_year)
    dt = 1 / time_steps_per_year
    num_assets = len(weights)


    # Covariance matrix for asset returns
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

    # Cholesky decomposition for generating correlated shocks
    chol_matrix = np.linalg.cholesky(cov_matrix)

    # Initialize portfolio paths
    portfolio_paths = np.zeros((total_steps + 1, num_simulations))
    portfolio_paths[0] = initial_investment

    for i in range(num_simulations):
        # Generate correlated random shocks
        random_normals = np.random.normal(0, 1, (total_steps, num_assets))
        correlated_shocks = random_normals @ chol_matrix.T * np.sqrt(dt)

        # Initialize log price paths
        log_prices = np.zeros((total_steps + 1, num_assets))

        for a in range(num_assets):
            drift = (expected_returns[a] - 0.5 * volatilities[a]**2) * dt
            for t in range(1, total_steps + 1):
                log_prices[t, a] = log_prices[t - 1, a] + drift + correlated_shocks[t - 1, a]

        # Convert log returns to actual price.
        prices = np.exp(log_prices)  # Convert log returns to price paths
        weighted_prices = prices @ weights
        portfolio_paths[:, i] = initial_investment * weighted_prices / weighted_prices[0]

    return pd.DataFrame(portfolio_paths)

def plot_simulations(simulations_df: pd.DataFrame):
    """
    Plot a subset of simulated paths and histogram of final values.

    Parameters:
    - simulations_df: DataFrame containing simulated paths
    """

    plt.figure(figsize=(12, 6))
    plt.plot(simulations_df.iloc[:, :100], linewidth=0.8, alpha=0.6)
    plt.title("Monte Carlo Simulation of Portfolio Value")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(simulations_df.iloc[-1], bins=50, edgecolor='black', alpha=0.75)
    plt.title("Distribution of Final Investment Values")
    plt.xlabel("Final Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def summary_statistics(simulations_df: pd.DataFrame):
    """
    Print and return summary statistics of the simulation.

    Parameters:
    - simulations_df: DataFrame containing simulated paths
    """
    final_values = simulations_df.iloc[-1]
    stats = {
        "Mean Final Value": np.mean(final_values),
        "Median Final Value": np.median(final_values),
        "5th Percentile": np.percentile(final_values, 5),
        "95th Percentile": np.percentile(final_values, 95),
        "Standard Deviation": np.std(final_values)
    }

    print("\nSummary Statistics:")
    for k, v in stats.items():
        print(f"{k}: £{v:,.2f}")
    
    return stats

def risk_metrics(simulations_df: pd.DataFrame, risk_free_rate: float = 0.02):
    """
    Calculate and print Sharpe Ratio and 95% Value at Risk (VaR).
    """
    final_values = simulations_df.iloc[-1]
    mean = np.mean(final_values)
    std_dev = np.std(final_values)
    initial = simulations_df.iloc[0, 0]

    total_return = (mean - initial) / initial
    sharpe_ratio = (total_return - risk_free_rate) / std_dev
    var_95 = np.percentile(final_values, 5)

    print("\nRisk Metrics:")
    print(f"Annualized Return: {total_return:.2%}")
    print(f"Portfolio Std Dev: £{std_dev:,.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"95% Value at Risk: £{initial - var_95:,.2f}")

    return {
        "Annualized Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "VaR 95%": initial - var_95
    }


# === Example Usage ===
if __name__ == "__main__":
    # Example: 3-asset portfolio (Large Cap, REIT, Small Cap)
    expected_returns = np.array([0.08, 0.06, 0.12])
    volatilities = np.array([0.15, 0.10, 0.25])
    weights = np.array([0.5, 0.3, 0.2])  # Must sum to 1

    correlation_matrix = np.array([
        [1.0, 0.3, 0.4],
        [0.3, 1.0, 0.2],
        [0.4, 0.2, 1.0]
    ])

    simulations = monte_carlo_portfolio_simulation(
        initial_investment=10000,
        expected_returns=expected_returns,
        volatilities=volatilities,
        weights=weights,
        correlation_matrix=correlation_matrix,
        time_horizon_years=1,
        num_simulations=10000
    )

    plot_simulations(simulations)
    summary_statistics(simulations)
    risk_metrics(simulations, risk_free_rate=0.03)