"""
Run a compact natural-resource momentum research grid.

The runner keeps the project focused:
- build raw, volatility-adjusted, SMA, EMA, and WMA crossover signals
- test long-only and long/short portfolio construction
- calculate absolute and market-relative metrics
- rank strategies out of sample with train/test and walk-forward checks
- display plots
"""

import matplotlib.pyplot as plt
import pandas as pd

from data_download import (
    ALL_TICKERS,
    build_dataset,
    download_price_data,
    report_download_summary,
)
from momentum_strat import build_momentum_strategy

PERIODS_PER_YEAR = 12
GROSS_EXPOSURE = 1.0

SHOW_PLOTS = True

TRAIN_SAMPLE_FRACTION = 0.70
MIN_RANKING_MONTHS = 36
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

MOVING_AVERAGE_WINDOWS = [
    {"label": "3-12", "fast_months": 3, "slow_months": 12, "skip_months": 1},
    {"label": "6-12", "fast_months": 6, "slow_months": 12, "skip_months": 1},
    {"label": "6-18", "fast_months": 6, "slow_months": 18, "skip_months": 1},
]

REGIMES = [
    {"name": "Pre-GFC", "start": "2000-01-31", "end": "2007-10-31"},
    {"name": "GFC", "start": "2007-11-30", "end": "2009-02-28"},
    {"name": "Post-GFC expansion", "start": "2009-03-31", "end": "2019-12-31"},
    {"name": "COVID and recovery", "start": "2020-01-31", "end": "2021-12-31"},
    {"name": "Inflation and rate shock", "start": "2022-01-31", "end": "2023-12-31"},
    {"name": "Recent", "start": "2024-01-31", "end": None},
]


def build_signal_settings():
    """
    Build the signal definitions used by the grid.

    Raw and volatility-adjusted momentum use lookback/skip windows.
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


def calculate_equal_weight_universe_returns(monthly_returns):
    return monthly_returns.mean(axis=1, skipna=True)


def returns_to_equity_curve(returns):
    return (1 + returns.dropna()).cumprod()


def build_equity_curves(returns_df):
    return returns_df.apply(returns_to_equity_curve)


def calculate_performance_metrics(returns):
    """
    Calculate the core metrics needed to rank and compare strategies.
    """
    returns = returns.dropna()

    if returns.empty:
        return {
            "periods": 0,
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "win_rate": None,
            "final_value": None,
            "valid_for_chart": False,
        }

    equity_curve = returns_to_equity_curve(returns)
    final_value = equity_curve.iloc[-1]
    cumulative_return = final_value - 1
    annualized_return = (
        None
        if final_value <= 0
        else final_value ** (PERIODS_PER_YEAR / len(returns)) - 1
    )
    annualized_volatility = returns.std() * (PERIODS_PER_YEAR ** 0.5)
    sharpe_ratio = (
        None
        if annualized_return is None or annualized_volatility == 0
        else annualized_return / annualized_volatility
    )
    drawdowns = equity_curve / equity_curve.cummax() - 1

    return {
        "periods": len(returns),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": drawdowns.min(),
        "win_rate": (returns > 0).mean(),
        "final_value": final_value,
        "valid_for_chart": final_value > 0 and drawdowns.min() > -1,
    }


def calculate_relative_metrics(strategy_returns, benchmark_returns, benchmark_name):
    """
    Calculate beta, alpha, correlation, and information ratio vs one benchmark.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["strategy", "benchmark"]

    if len(aligned) < 2 or aligned["benchmark"].var() == 0:
        return {
            f"beta_{benchmark_name}": None,
            f"alpha_{benchmark_name}": None,
            f"corr_{benchmark_name}": None,
            f"info_ratio_{benchmark_name}": None,
        }

    beta = aligned["strategy"].cov(aligned["benchmark"]) / aligned["benchmark"].var()
    monthly_alpha = aligned["strategy"].mean() - beta * aligned["benchmark"].mean()
    annualized_alpha = monthly_alpha * PERIODS_PER_YEAR
    active_returns = aligned["strategy"] - aligned["benchmark"]
    tracking_error = active_returns.std() * (PERIODS_PER_YEAR ** 0.5)
    information_ratio = (
        None
        if tracking_error == 0
        else active_returns.mean() * PERIODS_PER_YEAR / tracking_error
    )

    return {
        f"beta_{benchmark_name}": beta,
        f"alpha_{benchmark_name}": annualized_alpha,
        f"corr_{benchmark_name}": aligned["strategy"].corr(aligned["benchmark"]),
        f"info_ratio_{benchmark_name}": information_ratio,
    }


def calculate_results_table(strategy_returns, strategy_metadata, benchmark_returns):
    rows = []

    for strategy_name, returns in strategy_returns.items():
        row = {"strategy_name": strategy_name}
        row.update(strategy_metadata.loc[strategy_name].to_dict())
        row.update(calculate_performance_metrics(returns))

        for benchmark in RELATIVE_BENCHMARKS:
            if benchmark in benchmark_returns:
                row.update(
                    calculate_relative_metrics(
                        strategy_returns=returns,
                        benchmark_returns=benchmark_returns[benchmark],
                        benchmark_name=benchmark,
                    )
                )

        rows.append(row)

    return pd.DataFrame(rows).set_index("strategy_name")


def calculate_results_from_frame(returns_frame):
    rows = []

    for name in returns_frame.columns:
        rows.append(
            {
                "strategy_name": name,
                **calculate_performance_metrics(returns_frame[name]),
            }
        )

    return pd.DataFrame(rows).set_index("strategy_name")


def format_results_table(results):
    formatted = results.copy()

    percent_keywords = [
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "win_rate",
        "alpha_",
    ]
    decimal_keywords = [
        "sharpe_ratio",
        "final_value",
        "beta_",
        "corr_",
        "info_ratio_",
    ]

    for column in formatted.columns:
        if any(keyword in column for keyword in percent_keywords):
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )
        elif any(keyword in column for keyword in decimal_keywords):
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else ""
            )
        elif pd.api.types.is_datetime64_any_dtype(formatted[column]):
            formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")

    return formatted


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


def run_strategy_grid(resource_monthly_prices, resource_monthly_returns):
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

    metadata = pd.DataFrame(strategy_metadata).set_index("strategy_name")

    return strategy_returns, metadata, latest_signal_scores


def rank_results(results, by="sharpe_ratio", min_periods=MIN_RANKING_MONTHS):
    ranked = results[
        (results["valid_for_chart"] == True)
        & results[by].notna()
        & (results["periods"] >= min_periods)
    ].copy()

    return ranked.sort_values(by, ascending=False)


def slice_returns_dict(strategy_returns, index):
    return {
        name: returns.reindex(index)
        for name, returns in strategy_returns.items()
    }


def split_train_test_index(returns_frame, train_fraction=TRAIN_SAMPLE_FRACTION):
    usable_index = returns_frame.dropna(how="all").index

    if len(usable_index) < MIN_RANKING_MONTHS + 1:
        raise ValueError("Not enough monthly return history for a train/test split.")

    split_position = int(len(usable_index) * train_fraction)
    split_position = max(MIN_RANKING_MONTHS, split_position)
    split_position = min(split_position, len(usable_index) - 1)

    return usable_index[:split_position], usable_index[split_position:]


def calculate_period_results(
    strategy_returns,
    strategy_metadata,
    benchmark_returns,
    period_index,
):
    return calculate_results_table(
        strategy_returns=slice_returns_dict(strategy_returns, period_index),
        strategy_metadata=strategy_metadata,
        benchmark_returns=benchmark_returns.reindex(period_index),
    )


def build_train_test_validation_summary(in_sample_results, out_sample_results, top_n=15):
    top_in_sample = rank_results(in_sample_results, by="sharpe_ratio").head(top_n)
    rows = []

    for strategy_name in top_in_sample.index:
        out_row = out_sample_results.loc[strategy_name]
        in_row = in_sample_results.loc[strategy_name]
        rows.append(
            {
                "strategy_name": strategy_name,
                "in_sample_periods": in_row["periods"],
                "in_sample_annualized_return": in_row["annualized_return"],
                "in_sample_sharpe_ratio": in_row["sharpe_ratio"],
                "in_sample_max_drawdown": in_row["max_drawdown"],
                "out_sample_periods": out_row["periods"],
                "out_sample_annualized_return": out_row["annualized_return"],
                "out_sample_sharpe_ratio": out_row["sharpe_ratio"],
                "out_sample_max_drawdown": out_row["max_drawdown"],
                "out_sample_final_value": out_row["final_value"],
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("strategy_name")


def calculate_walk_forward_validation(
    returns_frame,
    train_months=WALK_FORWARD_TRAIN_MONTHS,
    test_months=WALK_FORWARD_TEST_MONTHS,
    ranking_metric="sharpe_ratio",
):
    returns_frame = returns_frame.dropna(how="all")

    if len(returns_frame) <= train_months:
        return pd.Series(dtype=float), pd.DataFrame()

    walk_forward_returns = []
    decisions = []
    index = returns_frame.index

    for test_start in range(train_months, len(index), test_months):
        test_end = min(test_start + test_months, len(index))
        train_slice = returns_frame.iloc[test_start - train_months:test_start]
        test_slice = returns_frame.iloc[test_start:test_end]

        candidate_results = calculate_results_from_frame(train_slice)
        ranked_candidates = rank_results(
            candidate_results,
            by=ranking_metric,
            min_periods=MIN_RANKING_MONTHS,
        )

        if ranked_candidates.empty or test_slice.empty:
            continue

        selected_strategy = ranked_candidates.index[0]
        selected_test_returns = test_slice[selected_strategy].dropna()

        if selected_test_returns.empty:
            continue

        train_metrics = ranked_candidates.loc[selected_strategy]
        test_metrics = calculate_performance_metrics(selected_test_returns)
        walk_forward_returns.append(selected_test_returns)
        decisions.append(
            {
                "train_start": train_slice.index.min(),
                "train_end": train_slice.index.max(),
                "test_start": selected_test_returns.index.min(),
                "test_end": selected_test_returns.index.max(),
                "selected_strategy": selected_strategy,
                "train_sharpe_ratio": train_metrics["sharpe_ratio"],
                "test_sharpe_ratio": test_metrics["sharpe_ratio"],
                "test_annualized_return": test_metrics["annualized_return"],
                "test_max_drawdown": test_metrics["max_drawdown"],
                "test_final_value": test_metrics["final_value"],
            }
        )

    if not walk_forward_returns:
        return pd.Series(dtype=float), pd.DataFrame(decisions)

    combined_returns = pd.concat(walk_forward_returns).sort_index()
    combined_returns = combined_returns[~combined_returns.index.duplicated(keep="first")]

    return combined_returns, pd.DataFrame(decisions)


def build_regime_summary(strategy_returns_frame, benchmark_returns, strategy_names):
    benchmark_names = [name for name in ["SPY", "XLE", "GLD"] if name in benchmark_returns]
    combined_returns = strategy_returns_frame.reindex(columns=strategy_names).join(
        benchmark_returns[benchmark_names],
        how="outer",
    )
    rows = []

    for regime in REGIMES:
        start = pd.Timestamp(regime["start"])
        end = None if regime["end"] is None else pd.Timestamp(regime["end"])
        regime_returns = combined_returns.loc[start:end]

        if regime_returns.dropna(how="all").empty:
            continue

        for asset_name in regime_returns.columns:
            metrics = calculate_performance_metrics(regime_returns[asset_name])

            if metrics["periods"] == 0:
                continue

            rows.append(
                {
                    "regime": regime["name"],
                    "asset": asset_name,
                    "periods": metrics["periods"],
                    "annualized_return": metrics["annualized_return"],
                    "annualized_volatility": metrics["annualized_volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "final_value": metrics["final_value"],
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index(["regime", "asset"])


def plot_equity_curves(equity_curves, columns, title="Top Strategies vs Benchmarks"):
    available_columns = [column for column in columns if column in equity_curves]

    if not available_columns:
        return

    plt.figure(figsize=(13, 7))

    for column in available_columns:
        plt.plot(equity_curves.index, equity_curves[column], label=column)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if not SHOW_PLOTS:
        plt.close()


def plot_sharpe_bars(results, top_n=15, title="Top In-Sample Strategies by Sharpe"):
    sharpe = (
        rank_results(results, by="sharpe_ratio")
        .head(top_n)["sharpe_ratio"]
        .sort_values()
    )

    if sharpe.empty:
        return

    plt.figure(figsize=(12, max(6, 0.35 * len(sharpe))))
    plt.barh(sharpe.index, sharpe)
    plt.title(title)
    plt.xlabel("Sharpe Ratio")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if not SHOW_PLOTS:
        plt.close()


def print_summary_table(title, table):
    columns = [
        "periods",
        "signal_family",
        "strategy_type",
        "weighting",
        "top_n",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "beta_SPY",
        "corr_SPY",
        "final_value",
    ]
    available_columns = [column for column in columns if column in table]

    print(f"\n{title}")
    print(format_results_table(table[available_columns]).to_string())


def print_table(title, table):
    print(f"\n{title}")

    if table.empty:
        print("No rows available.")
        return

    print(format_results_table(table).to_string())


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

    strategy_returns, strategy_metadata, _ = run_strategy_grid(
        resource_monthly_prices=resource_monthly_prices,
        resource_monthly_returns=resource_monthly_returns,
    )

    if not strategy_returns:
        raise RuntimeError("No strategies successfully ran.")

    strategy_returns_frame = pd.DataFrame(strategy_returns)
    train_index, test_index = split_train_test_index(strategy_returns_frame)

    in_sample_results = calculate_period_results(
        strategy_returns=strategy_returns,
        strategy_metadata=strategy_metadata,
        benchmark_returns=benchmark_monthly_returns,
        period_index=train_index,
    )
    out_sample_results = calculate_period_results(
        strategy_returns=strategy_returns,
        strategy_metadata=strategy_metadata,
        benchmark_returns=benchmark_monthly_returns,
        period_index=test_index,
    )
    full_sample_results = calculate_results_table(
        strategy_returns=strategy_returns,
        strategy_metadata=strategy_metadata,
        benchmark_returns=benchmark_monthly_returns,
    )

    top_in_sample = rank_results(in_sample_results, by="sharpe_ratio").head(20)
    top_full_sample = rank_results(full_sample_results, by="sharpe_ratio").head(20)
    validation_summary = build_train_test_validation_summary(
        in_sample_results=in_sample_results,
        out_sample_results=out_sample_results,
    )

    walk_forward_returns, walk_forward_decisions = calculate_walk_forward_validation(
        strategy_returns_frame,
    )
    walk_forward_metrics = calculate_performance_metrics(walk_forward_returns)

    comparison_returns = strategy_returns_frame.copy()
    comparison_returns["Resource universe equal-weight"] = (
        calculate_equal_weight_universe_returns(resource_monthly_returns)
    )

    if not walk_forward_returns.empty:
        comparison_returns["Walk-forward selected strategy"] = walk_forward_returns

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
            ["Walk-forward selected strategy"]
            + top_in_sample.head(5).index.tolist()
            + ["SPY", "XLE", "GLD"]
        )
    )
    plot_equity_curves(equity_curves, chart_columns)
    plot_sharpe_bars(in_sample_results)

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

    regime_strategy_names = list(
        dict.fromkeys(
            top_in_sample.head(3).index.tolist()
            + (["Walk-forward selected strategy"] if not walk_forward_returns.empty else [])
        )
    )
    regime_frame = strategy_returns_frame.copy()

    if not walk_forward_returns.empty:
        regime_frame["Walk-forward selected strategy"] = walk_forward_returns

    regime_summary = build_regime_summary(
        strategy_returns_frame=regime_frame,
        benchmark_returns=benchmark_monthly_returns,
        strategy_names=regime_strategy_names,
    )
    print_table("Regime summary", regime_summary)


if __name__ == "__main__":
    main()
