import pandas as pd
import numpy as np
import yfinance as yf

#  stock tickers and ETFs
stock_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ"]
etf_tickers = ["SPY", "IWM", "DIA"]


# calculate annualized volatility
def calculate_annualized_volatility(data):
    return data.pct_change().rolling(window=63).std() * np.sqrt(252)


# calculate beta
def calculate_beta(stock_returns, market_returns):
    beta = np.cov(stock_returns, market_returns, ddof=0)[0, 1] / np.var(
        market_returns, ddof=0
    )
    return beta


# rawdown metrics based on 52-week low and high
def calculate_drawdown(data):
    rolling_high = data.rolling(window=252).max()
    rolling_low = data.rolling(window=252).min()
    avg_drawdown = (rolling_low - rolling_high) / rolling_high
    max_drawdown = avg_drawdown.min()
    return avg_drawdown, max_drawdown


# total return and annualized total return
def calculate_total_return(data):
    total_return = (data[-1] / data[0]) - 1
    data_index = pd.to_datetime(data.index)  # Convert the index to DateTimeIndex
    annualized_return = (1 + total_return) ** (
        252 / len(np.unique(data_index.date))
    ) - 1
    return total_return, annualized_return


# stock data
stock_data = yf.download(stock_tickers, period="3mo")  # Adjust period as needed

#  stock risk analysis
stock_risk_df = pd.DataFrame({"Ticker": stock_tickers})
stock_risk_df["Portfolio Weight"] = 1 / len(stock_tickers)

# mean annualized volatility for all stocks
stock_risk_df["Annualized Volatility"] = calculate_annualized_volatility(
    stock_data["Adj Close"].mean(axis=1)
).mean()

# Cbeta values
market_data = yf.download(etf_tickers, period="1y")  # Adjust period as needed
for etf_ticker in etf_tickers:
    stock_risk_df[f"Beta against {etf_ticker}"] = np.nan

for etf_ticker in etf_tickers:
    market_returns = market_data["Adj Close"][etf_ticker].pct_change().dropna()
    for stock_ticker in stock_tickers:
        stock_returns = stock_data["Adj Close"][stock_ticker].pct_change().dropna()

        # Align data
        common_dates = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns[common_dates]
        market_returns = market_returns[common_dates]

        beta = calculate_beta(stock_returns, market_returns)
        stock_risk_df.loc[
            stock_risk_df["Ticker"] == stock_ticker, f"Beta against {etf_ticker}"
        ] = beta

# Calculate drawdown metrics
avg_drawdown, max_drawdown = calculate_drawdown(stock_data["Adj Close"].mean(axis=1))
stock_risk_df["Average Weekly Drawdown"] = avg_drawdown
stock_risk_df["Maximum Weekly Drawdown"] = max_drawdown


# C total return
total_returns, annualized_returns = calculate_total_return(
    stock_data["Adj Close"].mean(axis=1)
)
stock_risk_df["Total Return"] = total_returns
stock_risk_df["Annualized Total Return"] = annualized_returns

#  portfolio risk
portfolio_risk_df = pd.DataFrame({"ETF Ticker": etf_tickers})
portfolio_risk_df["Correlation against ETF"] = (
    stock_data["Adj Close"].mean(axis=1).corr(market_returns)
)

# e covariance matrix
covariances = np.cov(
    stock_data["Adj Close"].mean(axis=1).pct_change().dropna(), market_returns, ddof=0
)
portfolio_risk_df["Covariance of Portfolio against ETF"] = covariances[0, 1]

risk_free_rate = 0.5
portfolio_risk_df["Sharpe Ratio"] = (annualized_returns - risk_free_rate) / np.sqrt(
    covariances[0, 0]
)
portfolio_volatility = calculate_annualized_volatility(
    stock_data["Adj Close"].mean(axis=1)
).mean()
portfolio_risk_df[
    "Annualized Volatility Spread"
] = portfolio_volatility - market_returns.std() * np.sqrt(252)

# Create a correlation matrix
correlation_matrix = stock_data["Adj Close"].mean(axis=1).corr(market_returns)

# Display the results
print("Stock Risk Analysis:")
print(stock_risk_df)

print("\nPortfolio Risk against ETFs:")
print(portfolio_risk_df)

print("\nCorrelation Matrix:")
print(correlation_matrix)
