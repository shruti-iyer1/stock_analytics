import numpy as np

# Portfolio simulation function
def simulate_portfolio(data, weights):
    """
    Simulate portfolio performance based on historical data and weights.
    """
    # Normalize weights
    weights = np.array(weights)
    weights /= weights.sum()

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    # Portfolio daily returns
    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)

    # Portfolio statistics
    portfolio_cumulative_return = (1 + portfolio_daily_returns).cumprod()
    portfolio_annualized_return = portfolio_daily_returns.mean() * 252
    portfolio_annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252)
    sharpe_ratio = portfolio_annualized_return / portfolio_annualized_volatility

    stats = {
        "Cumulative Return": portfolio_cumulative_return[-1] - 1,
        "Annualized Return": portfolio_annualized_return,
        "Annualized Volatility": portfolio_annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
    }

    return portfolio_cumulative_return, stats