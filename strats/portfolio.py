"""
Portfolio return calculation.

This is intentionally small: it aligns positions and returns by ticker, then
multiplies same-date weights by same-date monthly returns. The signal builders
are responsible for shifting signals far enough back to avoid look-ahead bias.
"""

def calculate_portfolio_returns(positions, monthly_returns):
    """
    Calculate portfolio returns from position weights and monthly asset returns.

    Positive weights are long positions.
    Negative weights are short positions.

    Portfolio return is the weighted average of asset returns.
    """
    common_tickers = positions.columns.intersection(monthly_returns.columns)

    aligned_positions = positions[common_tickers]
    aligned_returns = monthly_returns[common_tickers]

    active_exposure = aligned_positions.abs().sum(axis=1)
    portfolio_returns = (aligned_positions * aligned_returns).sum(axis=1, min_count=1)
    portfolio_returns = portfolio_returns.where(active_exposure > 0)

    return portfolio_returns
