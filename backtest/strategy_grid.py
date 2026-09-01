"""
Strategy grid construction for the natural-resource momentum backtest.
"""

import pandas as pd

from strats.builder import build_momentum_strategy
from strats.momentum_signals import calculate_momentum_scores
from strats.threshold_momentum import build_threshold_momentum_strategy


# total absolute portfolio exposure:
# - long-only: 100% long
# - market-neutral long/short: 50% long and 50% short
GROSS_EXPOSURE = 1.0

STRATEGY_TYPES = ["long_only", "classic_long_short", "directional_abs"]
TOP_N_VALUES = [10, 20, 30]
SCALED_CAPS = [0.25, 0.20, 0.15]
INVERSE_VOL_EXPERIMENT_SIGNAL = "voladj6_12-1"
INVERSE_VOL_LOOKBACK_MONTHS = 6

MOMENTUM_WINDOWS = [
    {"label": "3-1", "lookback_months": 3, "skip_months": 1},
    {"label": "6-1", "lookback_months": 6, "skip_months": 1},
    {"label": "12-1", "lookback_months": 12, "skip_months": 1},
    {"label": "18-1", "lookback_months": 18, "skip_months": 1},
]

THRESHOLD_MOMENTUM_LEVELS = [
    {"label": "gt0pct", "threshold": 0.00},
    {"label": "gt20pct", "threshold": 0.20},
    {"label": "gt50pct", "threshold": 0.50},
]

MOVING_AVERAGE_WINDOWS = [
    {"label": "3-12", "fast_months": 3, "slow_months": 12, "skip_months": 1},
    {"label": "6-12", "fast_months": 6, "slow_months": 12, "skip_months": 1},
    {"label": "6-18", "fast_months": 6, "slow_months": 18, "skip_months": 1},
]


def build_signal_settings():
    """
    Build signal definitions used by the grid.

    Raw and volatility-adjusted momentum use lookback/skip windows;
    SMA, EMA, and WMA compare fast moving averages against slow moving averages.
    """
    settings = []

    for window in MOMENTUM_WINDOWS:
        settings.append(
            {
                "name": f"raw_{window['label']}",
                "signal_type": "momentum",
                "lookback_months": window["lookback_months"],
                "skip_months": window["skip_months"],
                "signal_family": "raw_momentum",
            }
        )
        settings.append(
            {
                "name": f"voladj6_{window['label']}",
                "signal_type": "volatility_adjusted",
                "lookback_months": window["lookback_months"],
                "skip_months": window["skip_months"],
                "volatility_lookback_months": 6,
                "signal_family": "vol_adjusted_momentum",
            }
        )

    for window in MOVING_AVERAGE_WINDOWS:
        for signal_type, label in [
            ("sma_crossover", "sma"),
            ("ema_crossover", "ema"),
            ("wma_crossover", "wma"),
        ]:
            settings.append(
                {
                    "name": f"{label}_{window['label']}",
                    "signal_type": signal_type,
                    "fast_months": window["fast_months"],
                    "slow_months": window["slow_months"],
                    "skip_months": window["skip_months"],
                    "signal_family": f"{label}_crossover",
                }
            )

    return settings


SIGNAL_SETTINGS = build_signal_settings()


def make_strategy_name(signal_name, strategy_type, weighting, top_n, max_weight=None):
    if weighting == "equal":
        return f"{signal_name} | {strategy_type} | equal | top{top_n}"

    if weighting == "inverse_vol":
        return f"{signal_name} | {strategy_type} | inverse_vol | top{top_n}"

    return (
        f"{signal_name} | {strategy_type} | scaled cap{int(max_weight * 100)} "
        f"| top{top_n}"
    )


def iter_portfolio_variants():
    for strategy_type in STRATEGY_TYPES:
        for top_n in TOP_N_VALUES:
            yield {
                "strategy_type": strategy_type,
                "top_n": top_n,
                "weighting": "equal",
                "max_weight": None,
            }
            for max_weight in SCALED_CAPS:
                yield {
                    "strategy_type": strategy_type,
                    "top_n": top_n,
                    "weighting": "scaled",
                    "max_weight": max_weight,
                }


def iter_inverse_volatility_experiment_variants(signal_setting):
    if signal_setting["name"] != INVERSE_VOL_EXPERIMENT_SIGNAL:
        return

    for top_n in TOP_N_VALUES:
        yield {
            "strategy_type": "long_only",
            "top_n": top_n,
            "weighting": "inverse_vol",
            "max_weight": None,
            "weight_volatility_lookback_months": INVERSE_VOL_LOOKBACK_MONTHS,
        }


def run_strategy_grid(
    resource_monthly_prices,
    resource_monthly_returns,
    eligibility_table=None,
):
    strategy_returns = {}
    strategy_metadata = []
    latest_signal_scores = None

    for signal_setting in SIGNAL_SETTINGS:
        variants = list(iter_portfolio_variants())
        variants.extend(iter_inverse_volatility_experiment_variants(signal_setting))

        for variant in variants:
            strategy_name = make_strategy_name(
                signal_name=signal_setting["name"],
                strategy_type=variant["strategy_type"],
                weighting=variant["weighting"],
                top_n=variant["top_n"],
                max_weight=variant["max_weight"],
            )

            try:
                returns, _, latest_signal_scores = build_momentum_strategy(
                    monthly_prices=resource_monthly_prices,
                    monthly_returns=resource_monthly_returns,
                    lookback_months=signal_setting.get("lookback_months", 12),
                    skip_months=signal_setting.get("skip_months", 1),
                    signal_type=signal_setting["signal_type"],
                    volatility_lookback_months=signal_setting.get(
                        "volatility_lookback_months",
                        6,
                    ),
                    fast_months=signal_setting.get("fast_months", 3),
                    slow_months=signal_setting.get("slow_months", 12),
                    top_n=variant["top_n"],
                    weighting=variant["weighting"],
                    max_weight=variant["max_weight"],
                    weight_volatility_lookback_months=variant.get(
                        "weight_volatility_lookback_months",
                        INVERSE_VOL_LOOKBACK_MONTHS,
                    ),
                    strategy_type=variant["strategy_type"],
                    gross_exposure=GROSS_EXPOSURE,
                    eligibility_table=eligibility_table,
                )
            except Exception:
                continue

            strategy_returns[strategy_name] = returns
            strategy_metadata.append(
                {
                    "strategy_name": strategy_name,
                    **signal_setting,
                    **variant,
                    "gross_exposure": GROSS_EXPOSURE,
                }
            )

    threshold_returns, threshold_metadata = run_threshold_strategy_grid(
        resource_monthly_prices=resource_monthly_prices,
        resource_monthly_returns=resource_monthly_returns,
        eligibility_table=eligibility_table,
    )
    strategy_returns.update(threshold_returns)
    strategy_metadata.extend(threshold_metadata)

    metadata = pd.DataFrame(strategy_metadata).set_index("strategy_name")

    return strategy_returns, metadata, latest_signal_scores


def make_threshold_strategy_name(signal_name, threshold_label):
    return f"{signal_name} | threshold_{threshold_label} | long_only | equal"


def run_threshold_strategy_grid(
    resource_monthly_prices,
    resource_monthly_returns,
    eligibility_table=None,
):
    """
    Run raw momentum thresholds separately from the top-N strategy grid.
    """
    strategy_returns = {}
    strategy_metadata = []

    for window in MOMENTUM_WINDOWS:
        signal_name = f"raw_{window['label']}"
        raw_momentum_scores = calculate_momentum_scores(
            monthly_prices=resource_monthly_prices,
            lookback_months=window["lookback_months"],
            skip_months=window["skip_months"],
        )

        for threshold_setting in THRESHOLD_MOMENTUM_LEVELS:
            strategy_name = make_threshold_strategy_name(
                signal_name=signal_name,
                threshold_label=threshold_setting["label"],
            )
            returns, _, _ = build_threshold_momentum_strategy(
                raw_momentum_scores=raw_momentum_scores,
                monthly_returns=resource_monthly_returns,
                threshold=threshold_setting["threshold"],
                gross_exposure=GROSS_EXPOSURE,
                eligibility_table=eligibility_table,
            )

            strategy_returns[strategy_name] = returns
            strategy_metadata.append(
                {
                    "strategy_name": strategy_name,
                    "name": signal_name,
                    "signal_family": "raw_momentum",
                    "signal_type": "momentum",
                    "lookback_months": window["lookback_months"],
                    "skip_months": window["skip_months"],
                    "selection_method": "threshold",
                    "threshold_label": threshold_setting["label"],
                    "threshold": threshold_setting["threshold"],
                    "strategy_type": "long_only",
                    "weighting": "equal",
                    "top_n": None,
                    "max_weight": None,
                    "gross_exposure": GROSS_EXPOSURE,
                }
            )

    return strategy_returns, strategy_metadata
