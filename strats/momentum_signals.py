"""
Signal construction and cross-sectional selection rules.

This file owns the ranking inputs used by every strategy. Each signal is
shifted by skip_months before it is used for portfolio construction so the
backtest does not rank assets using the same month being traded.
"""

import numpy as np


def calculate_momentum_scores(monthly_prices, lookback_months=12, skip_months=1):
    """
    Calculate momentum scores for each ticker using monthly prices.

    Default is 12-1 momentum:
    - look back 12 months
    - skip the most recent 1 month
    - measure return from 12 months ago to 1 month ago

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        Monthly adjusted close prices. Columns are tickers, rows are dates.
    lookback_months : int
        How far back to start the momentum calculation.
    skip_months : int
        How many recent months to skip.

    Returns
    -------
    pd.DataFrame
        Momentum score for each ticker at each date.
    """
    if lookback_months <= skip_months:
        raise ValueError("lookback_months must be greater than skip_months.")

    start_prices = monthly_prices.shift(lookback_months)
    end_prices = monthly_prices.shift(skip_months)

    momentum_scores = (end_prices / start_prices) - 1

    return momentum_scores


def select_top_momentum_assets(momentum_scores, top_n=10):
    """
    Select the top momentum assets each month.

    This is used for long-only momentum and for the long side
    of the classic long-short momentum strategy.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    ranks = momentum_scores.rank(axis=1, ascending=False, method="first")

    selection = ranks <= top_n

    return selection


def select_bottom_momentum_assets(momentum_scores, bottom_n=10):
    """
    Select the bottom momentum assets each month.

    This is used for the short side of the classic long-short
    momentum strategy.
    """
    if bottom_n < 1:
        raise ValueError("bottom_n must be at least 1.")

    ranks = momentum_scores.rank(axis=1, ascending=True, method="first")

    selection = ranks <= bottom_n

    return selection


def select_strongest_absolute_momentum_assets(momentum_scores, top_n=10):
    """
    Select assets with the strongest momentum signals by absolute value.

    This means both very positive and very negative momentum scores
    can be selected.

    Positive selected scores become long positions.
    Negative selected scores become short positions.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    absolute_scores = momentum_scores.abs()

    ranks = absolute_scores.rank(axis=1, ascending=False, method="first")

    selection = ranks <= top_n

    return selection


def calculate_volatility_adjusted_momentum_scores(
    monthly_prices,
    lookback_months=6,
    skip_months=1,
    volatility_lookback_months=6,
):
    """
    Calculate volatility-adjusted momentum scores.

    Regular momentum:
        momentum = price 1 month ago / price N months ago - 1

    Volatility-adjusted momentum:
        adjusted score = momentum / trailing volatility

    This rewards stocks with strong momentum, but penalizes stocks whose
    returns are extremely volatile.
    """
    momentum_scores = calculate_momentum_scores(
        monthly_prices=monthly_prices,
        lookback_months=lookback_months,
        skip_months=skip_months,
    )

    monthly_returns = monthly_prices.pct_change(fill_method=None)

    trailing_volatility = (
        monthly_returns
        .shift(skip_months)
        .rolling(volatility_lookback_months)
        .std()
    )

    volatility_adjusted_scores = momentum_scores / trailing_volatility

    volatility_adjusted_scores = volatility_adjusted_scores.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return volatility_adjusted_scores


def calculate_sma_crossover_scores(
    monthly_prices,
    fast_months=3,
    slow_months=12,
    skip_months=1,
):
    """
    Score assets by a simple moving average crossover.

    Positive scores mean the fast trend is above the slow trend.
    Prices are shifted by skip_months so the signal only uses data that
    would have been known before the traded month.
    """
    if fast_months < 1 or slow_months < 1:
        raise ValueError("fast_months and slow_months must be positive.")

    if fast_months >= slow_months:
        raise ValueError("fast_months must be less than slow_months.")

    shifted_prices = monthly_prices.shift(skip_months)
    fast_average = shifted_prices.rolling(fast_months).mean()
    slow_average = shifted_prices.rolling(slow_months).mean()

    return (fast_average / slow_average) - 1


def calculate_ema_crossover_scores(
    monthly_prices,
    fast_months=3,
    slow_months=12,
    skip_months=1,
):
    """
    Score assets by an exponential moving average crossover.

    EMA gives more weight to recent prices than SMA while preserving the
    same fast-vs-slow trend interpretation.
    """
    if fast_months < 1 or slow_months < 1:
        raise ValueError("fast_months and slow_months must be positive.")

    if fast_months >= slow_months:
        raise ValueError("fast_months must be less than slow_months.")

    shifted_prices = monthly_prices.shift(skip_months)
    fast_average = shifted_prices.ewm(span=fast_months, adjust=False).mean()
    slow_average = shifted_prices.ewm(span=slow_months, adjust=False).mean()

    return (fast_average / slow_average) - 1


def calculate_wma_crossover_scores(
    monthly_prices,
    fast_months=3,
    slow_months=12,
    skip_months=1,
):
    """
    Score assets by a linearly weighted moving average crossover.

    WMA weights the most recent price in the window highest, the oldest price
    lowest, and sits between SMA and EMA in how aggressively it reacts.
    """
    if fast_months < 1 or slow_months < 1:
        raise ValueError("fast_months and slow_months must be positive.")

    if fast_months >= slow_months:
        raise ValueError("fast_months must be less than slow_months.")

    shifted_prices = monthly_prices.shift(skip_months)

    def weighted_mean(values):
        weights = np.arange(1, len(values) + 1)
        return np.dot(values, weights) / weights.sum()

    fast_average = shifted_prices.rolling(fast_months).apply(
        weighted_mean,
        raw=True,
    )
    slow_average = shifted_prices.rolling(slow_months).apply(
        weighted_mean,
        raw=True,
    )

    return (fast_average / slow_average) - 1
