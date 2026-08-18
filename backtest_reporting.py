"""
Console and plot reporting for the momentum backtest.
"""

import matplotlib.pyplot as plt
import pandas as pd

from backtest_metrics import build_equity_curves
from backtest_validation import rank_results


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


def plot_equity_curves(
    equity_curves,
    columns,
    title="Top Strategies vs Benchmarks",
    show_plots=True,
):
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

    if not show_plots:
        plt.close()


def build_walk_forward_comparison_returns(
    walk_forward_returns,
    resource_universe_returns,
    benchmark_returns,
):
    """
    fair comparison uses the same dates as the walk-forward returns
    """
    walk_forward_returns = walk_forward_returns.dropna()

    if walk_forward_returns.empty:
        return pd.DataFrame()

    comparison_returns = pd.DataFrame(
        {"Walk-forward selected strategy": walk_forward_returns}
    )
    comparison_returns["Resource universe equal-weight"] = (
        resource_universe_returns.reindex(walk_forward_returns.index)
    )

    for benchmark in ["SPY", "XLE", "GLD"]:
        if benchmark in benchmark_returns:
            comparison_returns[benchmark] = benchmark_returns[benchmark].reindex(
                walk_forward_returns.index
            )

    return comparison_returns.dropna(axis=1, how="any")


def plot_walk_forward_comparison(
    walk_forward_returns,
    resource_universe_returns,
    benchmark_returns,
    show_plots=True,
):
    """
    benchmarks are rebased with no earlier head start
    """
    comparison_returns = build_walk_forward_comparison_returns(
        walk_forward_returns=walk_forward_returns,
        resource_universe_returns=resource_universe_returns,
        benchmark_returns=benchmark_returns,
    )

    if comparison_returns.empty:
        return

    equity_curves = build_equity_curves(
        comparison_returns,
        include_start=True,
    )
    plot_equity_curves(
        equity_curves=equity_curves,
        columns=comparison_returns.columns,
        title="Walk-Forward Strategy vs Benchmarks",
        show_plots=show_plots,
    )


def plot_sharpe_bars(
    results,
    top_n=15,
    title="Top In-Sample Strategies by Sharpe",
    min_ranking_months=36,
    show_plots=True,
):
    sharpe = (
        rank_results(results, by="sharpe_ratio", min_periods=min_ranking_months)
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

    if not show_plots:
        plt.close()


def print_summary_table(title, table):
    columns = [
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

    visible_table = table.drop(
        columns=[
            column
            for column in table.columns
            if column == "periods" or column.endswith("_periods")
        ],
        errors="ignore",
    )

    print(format_results_table(visible_table).to_string())
