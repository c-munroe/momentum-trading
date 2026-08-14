"""
Threshold momentum portfolio construction.

Top-N strategies always hold a fixed number of ranked stocks. Instead, threshold
strategies hold a variable number of stocks whose raw momentum clears the
cutoff, and hold cash when no stocks qualify.
"""

from strats.portfolio import calculate_portfolio_returns


def select_threshold_momentum_assets(raw_momentum_scores, threshold):
    """
    Select stocks with raw momentum strictly greater than the threshold.
    """
    return raw_momentum_scores.gt(threshold).fillna(False)


def build_equal_weight_threshold_positions(
    selection,
    gross_exposure=1.0,
):
    """
    Equal-weight every selected stock each month.
    """
    selected_count = selection.sum(axis=1)

    positions = selection.astype(float).div(
        selected_count.where(selected_count != 0),
        axis=0,
    ) * gross_exposure

    return positions.fillna(0.0)


def build_threshold_momentum_strategy(
    raw_momentum_scores,
    monthly_returns,
    threshold,
    gross_exposure=1.0,
):
    """
    Build long-only threshold momentum returns.

    raw_momentum_scores should already include the project's lookback and
    skip-month timing. Months with no positions are cash months with 0% return.
    """
    selection = select_threshold_momentum_assets(
        raw_momentum_scores=raw_momentum_scores,
        threshold=threshold,
    )
    positions = build_equal_weight_threshold_positions(
        selection=selection,
        gross_exposure=gross_exposure,
    )
    portfolio_returns = calculate_portfolio_returns(
        positions=positions,
        monthly_returns=monthly_returns,
    ).reindex(positions.index)

    active_exposure = positions.abs().sum(axis=1)
    portfolio_returns = portfolio_returns.where(active_exposure > 0, 0.0)

    return portfolio_returns, positions, selection
