
# List of seven assets (example: Apple, Microsoft, Google, Amazon, Facebook, Tesla, Netflix)
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NFLX']

pip install yfinance

import yfinance as yf
import datetime
import pandas as pd

# List of seven assets (example: Apple, Microsoft, Google, Amazon, Facebook, Tesla, Netflix)
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NFLX']

# Set start and end dates for the past 2 years
start_date = datetime.datetime.now() - datetime.timedelta(days=2*365)
end_date = datetime.datetime.now()

# Fetch data using yfinance
data = yf.download(assets, start=start_date, end=end_date)['Adj Close']

# Ensure fetched data is not empty
if data.empty:
    print("No data fetched for the selected assets. Please check the asset symbols and try again.")
else:
    print("Fetched data successfully.")
    print(data.head())  # Print the first few rows of the fetched data

# Calculate mean returns
mean_returns = data.pct_change().mean()

# Calculate variance-covariance matrix
cov_matrix = data.pct_change().cov()

# Calculate correlation matrix
correlation_matrix = data.pct_change().corr()

import numpy as np
from scipy.optimize import minimize

# Objective function for portfolio optimization (minimizing negative Sharpe ratio)
def objective(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Constraints for optimization (weights sum up to 1)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Bounds for weights (each weight between 0 and 1)
bounds = tuple((0, 1) for asset in range(len(assets)))

# Initial guess (equal weights for all assets)
initial_weights = [1./len(assets) for asset in assets]

# Risk-free rate (consider a reasonable value)
risk_free_rate = 0.02

# Optimize the portfolio
optimal_weights = minimize(objective, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                           method='SLSQP', bounds=bounds, constraints=constraints)

# Output optimal weights
print("Optimal Weights for Portfolio:", optimal_weights.x)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
# Calculate efficient frontier using Monte Carlo simulation
ef_points = pd.DataFrame(columns=['Return', 'Volatility', 'Sharpe Ratio'])


# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(ef_points['Volatility'], ef_points['Return'], c=ef_points['Sharpe Ratio'], cmap='YlGnBu', marker='o')
#plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], marker='x', color='r', s=200, label='Optimal Portfolio')
# Optimize the portfolio
optimal_weights = minimize(objective, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                           method='SLSQP', bounds=bounds, constraints=constraints)

# Calculate portfolio statistics for the optimal weights
portfolio_return = np.dot(optimal_weights.x, mean_returns)
portfolio_volatility = np.sqrt(np.dot(optimal_weights.x.T, np.dot(cov_matrix, optimal_weights.x)))
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# Store the optimal portfolio data
max_sharpe_portfolio = {
    'Return': portfolio_return,
    'Volatility': portfolio_volatility,
    'Sharpe Ratio': sharpe_ratio,
    'Weights': optimal_weights.x
}
plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.legend()
plt.show()

# Output optimal weights to a file
with open('optimal_weights.txt', 'w') as file:
    file.write('Optimal Weights for Portfolio:\n')
    for asset, weight in zip(assets, optimal_weights.x):
        file.write(f'{asset}: {weight:.4f}\n')