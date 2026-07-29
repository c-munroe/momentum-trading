
"""
Directional absolute-momentum portfolio construction.

Unlike classic long-short, this approach ranks assets by the size of the
signal and lets the sign of the selected signal decide direction. A strong
positive score becomes long; a strong negative score becomes short.
"""

from strats.weights import _cap_and_normalize_signed_weights


def build_equal_weight_directional_abs_positions(
    signal_scores,
    selection,
    gross_exposure=1.0,
):
    """
    Build equal-weight directional absolute momentum positions.

    Directional absolute momentum:
    - Pick the strongest momentum scores by absolute value.
    - If the selected score is positive, go long.
    - If the selected score is negative, go short.

    Every selected asset gets the same absolute weight.
    """
    selected_scores = signal_scores.where(selection, 0.0).fillna(0.0)

    directions = (selected_scores > 0).astype(float) - (
        selected_scores < 0
    ).astype(float)

    gross_exposure_by_month = directions.abs().sum(axis=1)

    positions = directions.div(
        gross_exposure_by_month.where(gross_exposure_by_month != 0),
        axis=0,
    ) * gross_exposure

    return positions.fillna(0.0)


def build_scaled_directional_abs_positions(
    signal_scores,
    selection,
    max_weight=None,
    gross_exposure=1.0,
):
    """
    Build scaled directional absolute momentum positions.

    Directional absolute momentum:
    - Pick the strongest momentum scores by absolute value.
    - If the selected score is positive, go long.
    - If the selected score is negative, go short.
    - Larger absolute momentum scores get larger absolute position sizes.
    """
    raw_positions = signal_scores.where(selection, 0.0).fillna(0.0)

    gross_exposure_by_month = raw_positions.abs().sum(axis=1)

    positions = raw_positions.div(
        gross_exposure_by_month.where(gross_exposure_by_month != 0),
        axis=0,
    ) * gross_exposure

    positions = positions.fillna(0.0)

    if max_weight is not None:
        positions = positions.apply(
            lambda row: _cap_and_normalize_signed_weights(
                positions=row,
                max_weight=max_weight,
                gross_exposure=gross_exposure,
            ),
            axis=1,
        )

    return positions.fillna(0.0)
