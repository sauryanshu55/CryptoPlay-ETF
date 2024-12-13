import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Define constants
equities_ticker = "ESPO"
crypto_tickers = ["BTC-USD", "ETH-USD"]
start_date = "2020-01-01"
end_date = "2023-01-01"

# Fetch historical data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data

def calculate_annualized_metrics(returns):
    mean_returns = returns.mean() * 252  # Annualize returns
    cov_matrix = returns.cov() * 252    # Annualize covariance
    return mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    _, _, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe_ratio

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]
    
    result = minimize(neg_sharpe_ratio, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def main():
    # Fetch data
    equities_data = fetch_data([equities_ticker], start_date, end_date)
    crypto_data = fetch_data(crypto_tickers, start_date, end_date)

    # Calculate returns
    equities_returns = equities_data.pct_change().dropna()
    crypto_returns = crypto_data.pct_change().dropna()

    # Calculate annualized metrics
    equities_mean_returns, equities_cov_matrix = calculate_annualized_metrics(equities_returns)
    crypto_mean_returns, crypto_cov_matrix = calculate_annualized_metrics(crypto_returns)

    # Optimize individual portfolios
    equities_weights = optimize_portfolio(equities_mean_returns, equities_cov_matrix)
    crypto_weights = optimize_portfolio(crypto_mean_returns, crypto_cov_matrix)

    # Combine into asset class weights
    equities_portfolio_return = np.dot(equities_weights, equities_mean_returns)
    equities_portfolio_volatility = np.sqrt(np.dot(equities_weights.T, np.dot(equities_cov_matrix, equities_weights)))

    crypto_portfolio_return = np.dot(crypto_weights, crypto_mean_returns)
    crypto_portfolio_volatility = np.sqrt(np.dot(crypto_weights.T, np.dot(crypto_cov_matrix, crypto_weights)))

    # Calculate Sharpe ratio-based allocation
    total_volatility = equities_portfolio_volatility + crypto_portfolio_volatility
    equities_allocation = equities_portfolio_volatility / total_volatility
    crypto_allocation = crypto_portfolio_volatility / total_volatility

    print("Equities Allocation (Asset Class):", equities_allocation)
    print("Crypto Allocation (Asset Class):", crypto_allocation)
    print("Equities Weights:", equities_weights)
    print("Crypto Weights:", crypto_weights)

if __name__ == "__main__":
    main()
