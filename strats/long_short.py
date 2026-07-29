
"""
Classic long-short portfolio construction.

This style always owns the strongest momentum names and shorts the weakest
momentum names. Gross exposure is split evenly between the long and short side,
so gross_exposure=1.0 means 50% long and 50% short before any cap logic.
"""

from strats.weights import EPSILON, _cap_and_normalize_nonnegative_weights


def build_equal_weight_classic_long_short_positions(
    long_selection,
    short_selection,
    gross_exposure=1.0,
):
    """
    Build equal-weight classic long-short momentum positions.

    Classic long-short:
    - Long the top momentum assets.
    - Short the bottom momentum assets.

    With gross_exposure=1.0:
    - Long side gets +50% exposure.
    - Short side gets -50% exposure.
    - Total absolute exposure is 100%.
    """
    long_target = gross_exposure / 2
    short_target = gross_exposure / 2

    # Prevent any ticker from being both long and short.
    short_selection = short_selection & ~long_selection

    long_count = long_selection.sum(axis=1)
    short_count = short_selection.sum(axis=1)

    long_positions = long_selection.astype(float).div(
        long_count.where(long_count != 0),
        axis=0,
    ) * long_target

    short_positions = short_selection.astype(float).div(
        short_count.where(short_count != 0),
        axis=0,
    ) * -short_target

    positions = long_positions.fillna(0.0) + short_positions.fillna(0.0)

    return positions.fillna(0.0)


def build_scaled_classic_long_short_positions(
    signal_scores,
    long_selection,
    short_selection,
    max_weight=None,
    gross_exposure=1.0,
):
    """
    Build scaled classic long-short momentum positions.

    Long side:
    - Higher positive momentum receives larger long weights.

    Short side:
    - Lower / weaker momentum receives larger short weights.

    With gross_exposure=1.0:
    - Long side gets +50% exposure.
    - Short side gets -50% exposure.
    - Total absolute exposure is 100%.
    """
    long_target = gross_exposure / 2
    short_target = gross_exposure / 2

    # Prevent any ticker from being both long and short.
    short_selection = short_selection & ~long_selection

    long_scores = signal_scores.where(long_selection)
    short_scores = signal_scores.where(short_selection)

    # For longs, stronger momentum gets more weight.
    long_row_min = long_scores.min(axis=1)
    long_raw_weights = long_scores.sub(long_row_min, axis=0) + EPSILON
    long_raw_weights = long_raw_weights.where(long_selection, 0.0).fillna(0.0)

    # For shorts, weaker/lower momentum gets more short weight.
    short_row_max = short_scores.max(axis=1)
    short_raw_weights = short_scores.rsub(short_row_max, axis=0) + EPSILON
    short_raw_weights = short_raw_weights.where(short_selection, 0.0).fillna(0.0)

    long_positions = long_raw_weights.apply(
        lambda row: _cap_and_normalize_nonnegative_weights(
            weights=row,
            max_weight=max_weight,
            target_sum=long_target,
        ),
        axis=1,
    )

    short_positions = short_raw_weights.apply(
        lambda row: _cap_and_normalize_nonnegative_weights(
            weights=row,
            max_weight=max_weight,
            target_sum=short_target,
        ),
        axis=1,
    )

    positions = long_positions - short_positions

    return positions.fillna(0.0)
