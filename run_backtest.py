"""
Run a compact natural-resource momentum research grid.
- build raw, volatility-adjusted, SMA, EMA, and WMA crossover signals
- test long-only and long/short portfolio construction
- calculate absolute and market-relative metrics
- rank strategies out of sample with train/test and walk-forward checks
- display plots
"""

import matplotlib.pyplot as plt
import pandas as pd

from backtest_metrics import (
    build_equity_curves,
    calculate_equal_weight_universe_returns,
    calculate_performance_metrics,
    calculate_results_table,
)
from backtest_reporting import (
    plot_equity_curves,
    plot_sharpe_bars,
    plot_walk_forward_comparison,
    print_summary_table,
    print_table,
)
from backtest_validation import (
    build_train_test_validation_summary,
    calculate_period_results,
    calculate_walk_forward_validation,
    rank_results,
    split_train_test_index,
)
from data_download import (
    ALL_TICKERS,
    END_DATE,
    START_DATE,
    build_dataset,
    download_price_data,
    report_download_summary,
)
from dynamic_universe import (
    DEFAULT_START_YEAR,
    LIQUIDITY_LOOKBACK_DAYS,
    MAX_UNIVERSE_SIZE,
    MIN_AVG_DAILY_DOLLAR_VOLUME,
    MIN_LIQUIDITY_OBSERVATIONS,
    build_annual_universes,
    download_price_volume_data,
    get_universe_for_date,
    print_annual_diagnostics,
)
from equities_list import NATURAL_RESOURCE_TICKERS
from momentum_strat import build_momentum_strategy
from strats.momentum_signals import calculate_momentum_scores
from strats.threshold_momentum import build_threshold_momentum_strategy


# strategy on monthly returns
PERIODS_PER_YEAR = 12

# total absolute portfolio exposure:
# - long-only: 100% long
# - market-neutral long/short: 50% long and 50% short
GROSS_EXPOSURE = 1.0

SHOW_PLOTS = True

# static preserves the original hand-curated universe behavior
# dynamic adds annual liquidity eligibility from dynamic_universe.py
UNIVERSE_MODE = "static"

# train/test validation:
# first 70% of usable history is treated as in-sample data for ranking
# strategies; the final 30% is held out for evaluation
TRAIN_SAMPLE_FRACTION = 0.70

# A strategy must have at least 36 valid monthly returns before it is eligible
# to be ranked to avoid favoring a strategy based on a very short history
MIN_RANKING_MONTHS = 36

# Walk-forward validation:
# at each decision date, evaluates strategies using only the previous 120 months
# (10 years), selects the best one, and then holds that strategy for the next
# 12 months (1 year). Then moves forward 12 months and repeats
#
# Creates a rolling, historically realistic sequence of decisions in which
# no test-period return is used to select the strategy for that same test period.
WALK_FORWARD_TRAIN_MONTHS = 120
WALK_FORWARD_TEST_MONTHS = 12


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

BENCHMARKS = ["SPY", "XLE", "XLB", "XME", "GDX", "GLD", "USO", "DBC"]
RELATIVE_BENCHMARKS = ["SPY", "XLE", "XLB", "GLD", "DBC"]

STRATEGY_TYPES = ["long_only", "classic_long_short", "directional_abs"]
TOP_N_VALUES = [10, 20, 30]
SCALED_CAPS = [0.25, 0.20, 0.15]

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
    Build signal definitions used by the grid

    Raw and volatility-adjusted momentum use lookback/skip windows;
    SMA, EMA, and WMA compare fast moving averages against slow moving averages
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


def get_last_complete_month_end(today=None):
    """
    Return the most recent completed month-end.

    Yahoo data for the current partial month is resampled to that month-end label.
    Historical validation should not treat an unfinished month as complete.
    """
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    today = today.normalize()
    current_month_end = today.to_period("M").to_timestamp("M")

    if today <= current_month_end:
        return (today.to_period("M") - 1).to_timestamp("M")

    return current_month_end


def keep_complete_months(monthly_frame, last_complete_month_end):
    return monthly_frame.loc[monthly_frame.index <= last_complete_month_end]


def validate_universe_mode(universe_mode):
    if universe_mode not in {"static", "dynamic"}:
        raise ValueError("UNIVERSE_MODE must be either 'static' or 'dynamic'.")


def make_strategy_name(signal_name, strategy_type, weighting, top_n, max_weight=None):
    if weighting == "equal":
        return f"{signal_name} | {strategy_type} | equal | top{top_n}"

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


def run_strategy_grid(
    resource_monthly_prices,
    resource_monthly_returns,
    eligibility_mask=None,
):
    strategy_returns = {}
    strategy_metadata = []
    latest_signal_scores = None

    for signal_setting in SIGNAL_SETTINGS:
        for variant in iter_portfolio_variants():
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
                    strategy_type=variant["strategy_type"],
                    gross_exposure=GROSS_EXPOSURE,
                    eligibility_mask=eligibility_mask,
                )
            except Exception as error:
                print(f"Skipped {strategy_name}: {error}")
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
        eligibility_mask=eligibility_mask,
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
    eligibility_mask=None,
):
    """
    run raw momentum thresholds separately from the top-N strategy grid
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
                eligibility_mask=eligibility_mask,
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


def build_monthly_eligibility_mask(monthly_index, tickers, annual_universes):
    """
    Convert annual universes into a month-end boolean eligibility matrix.
    """
    mask = pd.DataFrame(False, index=monthly_index, columns=tickers)

    for date in monthly_index:
        eligible_tickers = get_universe_for_date(
            date=date,
            annual_universes=annual_universes,
        )
        available_tickers = [ticker for ticker in eligible_tickers if ticker in mask]
        mask.loc[date, available_tickers] = True

    return mask


def print_static_universe_diagnostics(resource_monthly_returns):
    print("\nUniverse mode: static")
    print(f"Resource tickers: {resource_monthly_returns.shape[1]}")


def print_dynamic_universe_setup():
    print("\nUniverse mode: dynamic")
    print("Candidate pool: NATURAL_RESOURCE_TICKERS")
    print(f"Candidate names: {len(NATURAL_RESOURCE_TICKERS)}")
    print(f"First reconstitution year: {DEFAULT_START_YEAR}")
    print(f"Liquidity threshold: ${MIN_AVG_DAILY_DOLLAR_VOLUME:,.0f} ADDV")
    print(f"Liquidity lookback: {LIQUIDITY_LOOKBACK_DAYS} trading days")
    print(f"Minimum observations: {MIN_LIQUIDITY_OBSERVATIONS}")
    print(f"Maximum annual universe: {MAX_UNIVERSE_SIZE}")
    print("\nMarket-cap filter: NOT ACTIVE")
    print("Resource classification filter: NOT ACTIVE")
    print("Candidate survivorship bias: STILL PRESENT")


def build_dynamic_universe(
    monthly_index,
    resource_tickers,
):
    """
    Build annual liquidity-filtered universes for dynamic development mode.

    This still starts from NATURAL_RESOURCE_TICKERS. It is a dynamic liquidity
    filter, not a survivorship-bias-free historical security master.
    """
    print_dynamic_universe_setup()
    unadjusted_prices, volumes = download_price_volume_data(
        tickers=NATURAL_RESOURCE_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    result = build_annual_universes(
        candidate_tickers=NATURAL_RESOURCE_TICKERS,
        prices=unadjusted_prices,
        volumes=volumes,
        market_cap_data=None,
        subsector_data=None,
        require_resource_classification=False,
        start_year=DEFAULT_START_YEAR,
        end_year=monthly_index.max().year,
        allow_missing_market_cap=True,
    )
    eligibility_mask = build_monthly_eligibility_mask(
        monthly_index=monthly_index,
        tickers=resource_tickers,
        annual_universes=result.annual_universes,
    )

    print_annual_diagnostics(result.diagnostics)

    return result, eligibility_mask


def prepare_universe_mode(
    universe_mode,
    resource_monthly_prices,
    resource_monthly_returns,
):
    validate_universe_mode(universe_mode)

    if universe_mode == "static":
        print_static_universe_diagnostics(resource_monthly_returns)
        return None, None

    return build_dynamic_universe(
        monthly_index=resource_monthly_returns.index,
        resource_tickers=resource_monthly_returns.columns,
    )


def main():
    prices = download_price_data(ALL_TICKERS)
    report_download_summary(prices)

    dataset = build_dataset(prices)
    last_complete_month_end = get_last_complete_month_end()
    resource_monthly_prices = keep_complete_months(
        dataset["resource"]["monthly_prices"],
        last_complete_month_end,
    )
    resource_monthly_returns = keep_complete_months(
        dataset["resource"]["monthly_returns"],
        last_complete_month_end,
    )
    benchmark_monthly_returns = keep_complete_months(
        dataset["benchmark"]["monthly_returns"],
        last_complete_month_end,
    )

    print(f"\nUsing completed monthly data through {last_complete_month_end.date()}.")

    dynamic_universe_result, eligibility_mask = prepare_universe_mode(
        universe_mode=UNIVERSE_MODE,
        resource_monthly_prices=resource_monthly_prices,
        resource_monthly_returns=resource_monthly_returns,
    )

    strategy_returns, strategy_metadata, _ = run_strategy_grid(
        resource_monthly_prices=resource_monthly_prices,
        resource_monthly_returns=resource_monthly_returns,
        eligibility_mask=eligibility_mask,
    )

    if not strategy_returns:
        raise RuntimeError("No strategies successfully ran.")

    print(f"Strategies run: {len(strategy_returns)}")

    strategy_returns_frame = pd.DataFrame(strategy_returns)
    train_index, test_index = split_train_test_index(
        strategy_returns_frame,
        train_fraction=TRAIN_SAMPLE_FRACTION,
        min_ranking_months=MIN_RANKING_MONTHS,
    )

    in_sample_results = calculate_period_results(
        strategy_returns=strategy_returns,
        strategy_metadata=strategy_metadata,
        benchmark_returns=benchmark_monthly_returns,
        period_index=train_index,
        relative_benchmarks=RELATIVE_BENCHMARKS,
    )
    out_sample_results = calculate_period_results(
        strategy_returns=strategy_returns,
        strategy_metadata=strategy_metadata,
        benchmark_returns=benchmark_monthly_returns,
        period_index=test_index,
        relative_benchmarks=RELATIVE_BENCHMARKS,
    )
    full_sample_results = calculate_results_table(
        strategy_returns=strategy_returns,
        strategy_metadata=strategy_metadata,
        benchmark_returns=benchmark_monthly_returns,
        relative_benchmarks=RELATIVE_BENCHMARKS,
    )

    top_in_sample = rank_results(
        in_sample_results,
        by="sharpe_ratio",
        min_periods=MIN_RANKING_MONTHS,
    ).head(20)
    top_full_sample = rank_results(
        full_sample_results,
        by="sharpe_ratio",
        min_periods=MIN_RANKING_MONTHS,
    ).head(20)
    validation_summary = build_train_test_validation_summary(
        in_sample_results=in_sample_results,
        out_sample_results=out_sample_results,
        min_ranking_months=MIN_RANKING_MONTHS,
    )

    walk_forward_returns, walk_forward_decisions = calculate_walk_forward_validation(
        returns_frame=strategy_returns_frame,
        train_months=WALK_FORWARD_TRAIN_MONTHS,
        test_months=WALK_FORWARD_TEST_MONTHS,
        min_ranking_months=MIN_RANKING_MONTHS,
    )
    walk_forward_metrics = calculate_performance_metrics(walk_forward_returns)
    resource_universe_returns = calculate_equal_weight_universe_returns(
        monthly_returns=resource_monthly_returns,
        eligibility_mask=eligibility_mask,
    )

    comparison_returns = strategy_returns_frame.copy()

    available_benchmarks = [
        benchmark for benchmark in BENCHMARKS if benchmark in benchmark_monthly_returns
    ]
    comparison_returns = comparison_returns.join(
        benchmark_monthly_returns[available_benchmarks],
        how="outer",
    )

    equity_curves = build_equity_curves(comparison_returns)
    chart_columns = list(
        dict.fromkeys(
            top_in_sample.head(5).index.tolist()
            + ["SPY", "XLE", "GLD"]
        )
    )
    plot_equity_curves(
        equity_curves=equity_curves,
        columns=chart_columns,
        show_plots=SHOW_PLOTS,
    )

    if not walk_forward_returns.empty:
        plot_walk_forward_comparison(
            walk_forward_returns=walk_forward_returns,
            resource_universe_returns=resource_universe_returns,
            benchmark_returns=benchmark_monthly_returns,
            show_plots=SHOW_PLOTS,
        )

    plot_sharpe_bars(
        in_sample_results,
        min_ranking_months=MIN_RANKING_MONTHS,
        show_plots=SHOW_PLOTS,
    )

    if SHOW_PLOTS:
        plt.show()
        plt.close("all")

    print(
        "\nTrain/test split: "
        f"{train_index.min().date()} to {train_index.max().date()} in-sample, "
        f"{test_index.min().date()} to {test_index.max().date()} out-of-sample."
    )
    print_summary_table("Top strategies by in-sample Sharpe", top_in_sample.head(10))
    print_summary_table("Top strategies by full-sample Sharpe", top_full_sample.head(10))
    print_table("In-sample winners checked out of sample", validation_summary.head(15))

    walk_forward_summary = pd.DataFrame([walk_forward_metrics], index=["Walk-forward"])
    print_table("Walk-forward selected-strategy performance", walk_forward_summary)

    if not walk_forward_decisions.empty:
        print_table(
            "Recent walk-forward selections",
            walk_forward_decisions.tail(10).set_index("test_start"),
        )

    return {
        "strategy_returns": strategy_returns,
        "strategy_metadata": strategy_metadata,
        "in_sample_results": in_sample_results,
        "out_sample_results": out_sample_results,
        "full_sample_results": full_sample_results,
        "walk_forward_returns": walk_forward_returns,
        "walk_forward_decisions": walk_forward_decisions,
        "resource_universe_returns": resource_universe_returns,
        "dynamic_universe_result": dynamic_universe_result,
        "eligibility_mask": eligibility_mask,
    }


if __name__ == "__main__":
    main()
