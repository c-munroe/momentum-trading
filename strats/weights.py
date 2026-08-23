"""
Shared weight normalization and position-cap helpers.

These helpers keep portfolio construction honest after signal scaling. They
renormalize active positions to the requested exposure and enforce a true max
weight by redistributing leftover exposure across uncapped names.
"""

import pandas as pd


EPSILON = 1e-8


def _cap_and_normalize_nonnegative_weights(weights, max_weight=None, target_sum=1.0):
    """
    Normalize nonnegative weights to a target sum while optionally applying
    a true maximum position cap.

    This is used for long-only weights and for each side of long-short weights.
    """
    weights = weights.fillna(0.0).clip(lower=0.0)

    active_weights = weights[weights > 0].copy()

    if active_weights.empty:
        return weights * 0.0

    if max_weight is None:
        normalized_weights = active_weights / active_weights.sum() * target_sum

        result = pd.Series(0.0, index=weights.index)
        result.loc[normalized_weights.index] = normalized_weights

        return result

    if len(active_weights) * max_weight < target_sum:
        raise ValueError(
            "max_weight is too small for the number of active positions."
        )

    capped_weights = pd.Series(0.0, index=weights.index)

    remaining_weights = active_weights.copy()
    remaining_target = target_sum

    while not remaining_weights.empty:
        normalized_weights = (
            remaining_weights / remaining_weights.sum() * remaining_target
        )

        over_cap = normalized_weights > max_weight

        if not over_cap.any():
            capped_weights.loc[normalized_weights.index] = normalized_weights
            break

        capped_names = normalized_weights[over_cap].index

        capped_weights.loc[capped_names] = max_weight

        remaining_target -= max_weight * len(capped_names)

        remaining_weights = remaining_weights.drop(capped_names)

    return capped_weights


def _cap_and_normalize_signed_weights(positions, max_weight=None, gross_exposure=1.0):
    """
    Normalize signed positions so total absolute exposure equals gross_exposure,
    while optionally applying a true max absolute position cap.

    Positive weights are long positions.
    Negative weights are short positions.
    """
    positions = positions.fillna(0.0)

    signs = (positions > 0).astype(float) - (positions < 0).astype(float)
    absolute_weights = positions.abs()

    capped_absolute_weights = _cap_and_normalize_nonnegative_weights(
        weights=absolute_weights,
        max_weight=max_weight,
        target_sum=gross_exposure,
    )

    signed_positions = capped_absolute_weights * signs

    return signed_positions


def calculate_inverse_volatility_weights(
    monthly_returns,
    selection,
    volatility_lookback_months=6,
    gross_exposure=1.0,
    max_weight=None,
):
    """
    Size selected long positions by inverse trailing realized volatility.

    The trailing volatility is shifted one month so weights for month t only
    use returns available before month t.
    """
    trailing_volatility = monthly_returns.shift(1).rolling(
        volatility_lookback_months,
    ).std()
    valid_volatility = trailing_volatility.where(trailing_volatility > 0)
    raw_weights = (1.0 / valid_volatility).where(selection)

    positions = raw_weights.apply(
        lambda row: _cap_and_normalize_nonnegative_weights(
            weights=row,
            max_weight=max_weight,
            target_sum=gross_exposure,
        ),
        axis=1,
    )

    return positions.fillna(0.0)
