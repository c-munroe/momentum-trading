"""
Momentum signal construction and portfolio position building
for monthly cross-sectional backtests.
"""

from strats.momentum_signals import (
    calculate_ema_crossover_scores,
    calculate_momentum_scores,
    calculate_sma_crossover_scores,
    calculate_volatility_adjusted_momentum_scores,
    calculate_wma_crossover_scores,
    select_bottom_momentum_assets,
    select_strongest_absolute_momentum_assets,
    select_top_momentum_assets,
)

from strats.long_only import (
    build_equal_weight_positions,
    build_inverse_volatility_weight_positions,
    build_scaled_weight_positions,
)

from strats.long_short import (
    build_equal_weight_classic_long_short_positions,
    build_scaled_classic_long_short_positions,
)

from strats.long_short_abs import (
    build_equal_weight_directional_abs_positions,
    build_scaled_directional_abs_positions,
)

from strats.portfolio import calculate_portfolio_returns


def apply_eligibility_table(signal_scores, eligibility_table):
    """
    Remove ineligible names from current-month selection.

    Signals are still calculated from the full price history. Eligibility only
    controls which names may enter the portfolio at each formation date.
    """
    if eligibility_table is None:
        return signal_scores

    aligned_table = eligibility_table.reindex(
        index=signal_scores.index,
        columns=signal_scores.columns,
        fill_value=False,
    )

    return signal_scores.where(aligned_table)


def build_momentum_strategy(
    monthly_prices,
    monthly_returns,
    lookback_months=12,
    skip_months=1,
    signal_type="momentum",
    volatility_lookback_months=6,
    fast_months=3,
    slow_months=12,
    top_n=10,
    weighting="equal",
    max_weight=None,
    weight_volatility_lookback_months=6,
    strategy_type="long_only",
    gross_exposure=1.0,
    eligibility_table=None,
):
    """
    Build a monthly momentum strategy.

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        Monthly adjusted close prices.
    monthly_returns : pd.DataFrame
        Monthly returns.
    lookback_months : int
        Number of months to look back for the momentum signal.
    skip_months : int
        Number of recent months to skip.
    signal_type : str
        "momentum", "volatility_adjusted", "sma_crossover",
        "ema_crossover", or "wma_crossover".
    volatility_lookback_months : int
        Rolling volatility window used when signal_type is "volatility_adjusted".
    fast_months : int
        Fast moving average window for crossover signals.
    slow_months : int
        Slow moving average window for crossover signals.
    top_n : int
        Number of assets to select.
    weighting : str
        "equal", "scaled", or "inverse_vol".
    max_weight : float, optional
        Maximum absolute weight for one asset.
    strategy_type : str
        "long_only":
            Long the top momentum names.

        "classic_long_short":
            Long top winners and short bottom losers.

        "directional_abs":
            Select strongest absolute momentum signals.
            Long positive signals and short negative signals.

    gross_exposure : float
        Total absolute exposure of the portfolio.
        For example, gross_exposure=1.0 means 100% total gross exposure.
    eligibility_table : pd.DataFrame, optional
        Boolean matrix aligned to monthly formation dates and tickers. True
        means a security may be selected that month.

    Returns
    -------
    tuple
        portfolio_returns, positions, momentum_scores
    """
    if signal_type == "momentum":
        momentum_scores = calculate_momentum_scores(
            monthly_prices=monthly_prices,
            lookback_months=lookback_months,
            skip_months=skip_months,
        )
    elif signal_type == "volatility_adjusted":
        momentum_scores = calculate_volatility_adjusted_momentum_scores(
            monthly_prices=monthly_prices,
            lookback_months=lookback_months,
            skip_months=skip_months,
            volatility_lookback_months=volatility_lookback_months,
        )
    elif signal_type == "sma_crossover":
        momentum_scores = calculate_sma_crossover_scores(
            monthly_prices=monthly_prices,
            fast_months=fast_months,
            slow_months=slow_months,
            skip_months=skip_months,
        )
    elif signal_type == "ema_crossover":
        momentum_scores = calculate_ema_crossover_scores(
            monthly_prices=monthly_prices,
            fast_months=fast_months,
            slow_months=slow_months,
            skip_months=skip_months,
        )
    elif signal_type == "wma_crossover":
        momentum_scores = calculate_wma_crossover_scores(
            monthly_prices=monthly_prices,
            fast_months=fast_months,
            slow_months=slow_months,
            skip_months=skip_months,
        )
    else:
        raise ValueError(
            "signal_type must be 'momentum', 'volatility_adjusted', "
            "'sma_crossover', 'ema_crossover', or 'wma_crossover'."
        )

    momentum_scores = apply_eligibility_table(
        signal_scores=momentum_scores,
        eligibility_table=eligibility_table,
    )

    if strategy_type == "long_only":
        selection = select_top_momentum_assets(
            momentum_scores=momentum_scores,
            top_n=top_n,
        )

        if weighting == "equal":
            positions = build_equal_weight_positions(selection)

        elif weighting == "scaled":
            positions = build_scaled_weight_positions(
                signal_scores=momentum_scores,
                selection=selection,
                max_weight=max_weight,
            )

        elif weighting == "inverse_vol":
            positions = build_inverse_volatility_weight_positions(
                monthly_returns=monthly_returns,
                selection=selection,
                volatility_lookback_months=weight_volatility_lookback_months,
                gross_exposure=gross_exposure,
                max_weight=max_weight,
            )

        else:
            raise ValueError(
                "long_only weighting must be 'equal', 'scaled', or 'inverse_vol'."
            )

    elif strategy_type == "classic_long_short":
        long_selection = select_top_momentum_assets(
            momentum_scores=momentum_scores,
            top_n=top_n,
        )

        short_selection = select_bottom_momentum_assets(
            momentum_scores=momentum_scores,
            bottom_n=top_n,
        )

        if weighting == "equal":
            positions = build_equal_weight_classic_long_short_positions(
                long_selection=long_selection,
                short_selection=short_selection,
                gross_exposure=gross_exposure,
            )

        elif weighting == "scaled":
            positions = build_scaled_classic_long_short_positions(
                signal_scores=momentum_scores,
                long_selection=long_selection,
                short_selection=short_selection,
                max_weight=max_weight,
                gross_exposure=gross_exposure,
            )

        else:
            raise ValueError("weighting must be either 'equal' or 'scaled'.")

    elif strategy_type == "directional_abs":
        selection = select_strongest_absolute_momentum_assets(
            momentum_scores=momentum_scores,
            top_n=top_n,
        )

        if weighting == "equal":
            positions = build_equal_weight_directional_abs_positions(
                signal_scores=momentum_scores,
                selection=selection,
                gross_exposure=gross_exposure,
            )

        elif weighting == "scaled":
            positions = build_scaled_directional_abs_positions(
                signal_scores=momentum_scores,
                selection=selection,
                max_weight=max_weight,
                gross_exposure=gross_exposure,
            )

        else:
            raise ValueError("weighting must be either 'equal' or 'scaled'.")

    else:
        raise ValueError(
            "strategy_type must be 'long_only', 'classic_long_short', or 'directional_abs'."
        )

    portfolio_returns = calculate_portfolio_returns(
        positions=positions,
        monthly_returns=monthly_returns,
    )

    return portfolio_returns, positions, momentum_scores
