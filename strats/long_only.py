"""
Long-only portfolio construction.

The functions in this file take a boolean asset selection matrix and convert
it into monthly portfolio weights. They do not decide which names are selected;
that stays in momentum_signals.py so signal research and weighting research
can be changed independently.
"""

from strats.weights import (
    EPSILON,
    _cap_and_normalize_nonnegative_weights,
    calculate_inverse_volatility_weights,
)


def build_equal_weight_positions(selection):
    """
    Assign equal weights to all selected assets each month.

    This is for long-only portfolios.
    """
    selected_count = selection.sum(axis=1)

    positions = selection.astype(float).div(
        selected_count.where(selected_count != 0),
        axis=0,
    )

    return positions.fillna(0.0)


def build_scaled_weight_positions(signal_scores, selection, max_weight=None):
    """
    Assign larger long-only weights to assets with stronger momentum signals.

    Scaling logic:
    1. Keep only the momentum scores for selected assets.
    2. Shift the selected scores so the weakest selected asset is close to zero.
    3. Stronger momentum scores become larger positive values.
    4. Divide each score by the total selected score for that month.
    5. This makes all portfolio weights sum to 1, with larger weights going
       to assets that have stronger relative momentum signals.
    """
    selected_scores = signal_scores.where(selection)

    row_min = selected_scores.min(axis=1)

    shifted_scores = selected_scores.sub(row_min, axis=0) + EPSILON

    raw_weights = shifted_scores.where(selection, 0.0).fillna(0.0)

    positions = raw_weights.apply(
        lambda row: _cap_and_normalize_nonnegative_weights(
            weights=row,
            max_weight=max_weight,
            target_sum=1.0,
        ),
        axis=1,
    )

    return positions.fillna(0.0)


def build_inverse_volatility_weight_positions(
    monthly_returns,
    selection,
    volatility_lookback_months=6,
    gross_exposure=1.0,
    max_weight=None,
):
    """
    Assign larger weights to selected assets with lower trailing volatility.
    """
    return calculate_inverse_volatility_weights(
        monthly_returns=monthly_returns,
        selection=selection,
        volatility_lookback_months=volatility_lookback_months,
        gross_exposure=gross_exposure,
        max_weight=max_weight,
    )
