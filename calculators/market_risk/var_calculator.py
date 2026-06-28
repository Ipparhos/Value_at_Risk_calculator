import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from data.data_loader import fetch_stock_data


def calculate_var_historical_method(ticker_symbol: str = "^GSPC", period: str = "252d", confidence_level: float = 0.95):
    # Calculate daily returns
    stock_data = fetch_stock_data(ticker_symbol, period)
    stock_data['returns'] = stock_data['Close'].pct_change()

    # drop Nan value
    stock_data = stock_data.dropna()
    # print(stock_data.head())

    # Calculate VaR using historical method
    var_hist = stock_data['returns'].quantile(1 - confidence_level)

    return var_hist, stock_data['returns']


#variance-covariance method
def calculate_var_parametric_method(ticker_symbol: str = "^GSPC", period: str = "252d", confidence_level: float = 0.95):
    # Calculate daily returns
    stock_data = fetch_stock_data(ticker_symbol, period)
    stock_data['returns'] = stock_data['Close'].pct_change()

    # drop Nan value
    stock_data = stock_data.dropna()
    # print(stock_data.head())

    # Calculate VaR using parametric method
    mu = stock_data['returns'].mean()
    sigma = stock_data['returns'].std()
    z_score = norm.ppf(1 - confidence_level)
    var_parametric = mu + z_score * sigma
    
    return var_parametric, stock_data['returns']


def calculate_var_monte_carlo_method(ticker_symbol: str = "^GSPC", period: str = "252d", confidence_level: float = 0.95, num_simulations: int = 10000):
    # Calculate daily returns
    stock_data = fetch_stock_data(ticker_symbol, period)
    stock_data['returns'] = stock_data['Close'].pct_change()

    # Simulate future returns using Monte Carlo
    num_simulations = 10000
    simulation_horizon = 252  # Number of trading days in a year
    simulated_returns = np.random.normal(stock_data['returns'].mean(), stock_data['returns'].std(), (simulation_horizon, num_simulations))
    
    # Calculate the simulated portfolio values
    initial_investment = 1000000
    simulated_portfolio_values = initial_investment * np.exp(np.cumsum(simulated_returns, axis=0))

    # Calculate the portfolio returns
    portfolio_returns = simulated_portfolio_values[-1] / simulated_portfolio_values[0] - 1

    # Calculate VaR using Monte Carlo method
    var_monte_carlo = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    return var_monte_carlo, portfolio_returns


def plot_var(returns:pd.Series, var_hist:float, method: str, ticker_symbol: str = "^GSPC", confidence_level: float = 0.95):
    # Plot the returns and VaR threshold
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(var_hist, color='red', linestyle='--', label=f'VaR ({confidence_level*100}%): {var_hist:.2%}')
    plt.title(f'{method.upper()} Returns Distribution of {ticker_symbol}')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # var_hist, returns = calculate_var_historical_method()
    # plot_var(returns, var_hist, 'Historical')

    # var_param, returns = calculate_var_parametric_method()
    # plot_var(returns, var_param, 'Parametric')


    var_monte_carlo, returns = calculate_var_monte_carlo_method(ticker_symbol="AAPL")
    plot_var(returns, var_monte_carlo, 'Monte Carlo')