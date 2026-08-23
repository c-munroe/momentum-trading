"""
Console and plot reporting for the momentum backtest.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button

from backtest.metrics import (
    build_equity_curves,
    calculate_performance_metrics,
    calculate_results_from_frame,
)
from backtest.validation import rank_results


PLOT_FIGSIZE = (11, 6.5)
BAR_FIGSIZE = (11, 6.8)


def apply_research_plot_style(ax):
    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def get_strategy_display_name(strategy_name):
    """
    Convert internal strategy identifiers to deterministic plot labels.
    """
    if " | " not in str(strategy_name):
        return strategy_name

    parts = str(strategy_name).split(" | ")

    if len(parts) < 4:
        return strategy_name

    signal_name = parts[0]
    strategy_type = parts[1]
    weighting = parts[2]
    top_n = parts[3].replace("top", "Top ")

    signal_label = signal_name
    if signal_name.startswith("voladj6_"):
        signal_label = (
            signal_name.replace("voladj6_", "")
            + " Vol-Adjusted"
        )
    elif signal_name.startswith("raw_"):
        signal_label = signal_name.replace("raw_", "") + " Raw Momentum"
    elif signal_name.startswith("sma_"):
        signal_label = signal_name.replace("sma_", "") + " SMA"
    elif signal_name.startswith("ema_"):
        signal_label = signal_name.replace("ema_", "") + " EMA"
    elif signal_name.startswith("wma_"):
        signal_label = signal_name.replace("wma_", "") + " WMA"

    if strategy_type.startswith("threshold_"):
        threshold_label = (
            strategy_type
            .replace("threshold_gt", ">")
            .replace("pct", "% Threshold")
        )
        strategy_type = parts[2] if len(parts) > 2 else "long_only"
        weighting = parts[3] if len(parts) > 3 else "equal"
        top_n = threshold_label

    strategy_label = {
        "long_only": "Long Only",
        "classic_long_short": "Classic Long-Short",
        "directional_abs": "Directional Absolute",
    }.get(strategy_type, strategy_type.replace("_", " ").title())

    weighting_label = weighting
    if weighting == "inverse_vol":
        weighting_label = "Inverse Vol"
    elif weighting == "equal":
        weighting_label = "Equal Weight"
    elif weighting.startswith("scaled cap"):
        weighting_label = weighting.replace("scaled cap", "Scaled Cap ")

    return f"{signal_label} | {strategy_label} | {weighting_label} | {top_n}"


def get_plot_display_label(label):
    known_labels = {
        "Resource universe equal-weight": "Resource Equal-Weight",
        "Single-winner walk-forward": "Single-Winner Selector",
        "Top-3 ensemble": "Top-3 Ensemble",
        "Top-5 ensemble": "Top-5 Ensemble",
        "Diversified Top-5 ensemble": "Diversified Top-5",
    }

    return known_labels.get(label, get_strategy_display_name(label))


def format_results_table(results):
    formatted = results.copy()

    percent_keywords = [
        "cumulative_return",
        "return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "win_rate",
        "alpha_",
        "share",
    ]
    decimal_keywords = [
        "sharpe_ratio",
        "calmar_ratio",
        "final_value",
        "beta_",
        "corr_",
        "info_ratio_",
    ]

    for column in formatted.columns:
        if pd.api.types.is_bool_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: str(value) if pd.notna(value) else ""
            )
        elif pd.api.types.is_datetime64_any_dtype(formatted[column]):
            formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")
        elif any(keyword in column for keyword in percent_keywords):
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )
        elif any(keyword in column for keyword in decimal_keywords):
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else ""
            )

    return formatted


def plot_equity_curves(
    equity_curves,
    columns,
    title="Top Strategies vs Benchmarks",
    subtitle=None,
    ylabel="Growth of $1 (log scale)",
    primary_column=None,
    show_plots=True,
):
    available_columns = [column for column in columns if column in equity_curves]

    if not available_columns:
        return

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    for column in available_columns:
        is_primary = column == primary_column
        ax.plot(
            equity_curves.index,
            equity_curves[column],
            label=get_plot_display_label(column),
            linewidth=2.7 if is_primary else 1.7,
            alpha=1.0 if is_primary else 0.88,
        )

    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_yscale("log")
    apply_research_plot_style(ax)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92, loc="best")
    fig.tight_layout()

    if not show_plots:
        plt.close(fig)


def build_walk_forward_comparison_returns(
    walk_forward_returns,
    resource_universe_returns,
    benchmark_returns,
    fixed_strategy_returns=None,
    fixed_strategy_label=None,
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

    if "SPY" in benchmark_returns:
        comparison_returns["SPY"] = benchmark_returns["SPY"].reindex(
            walk_forward_returns.index
        )

    if fixed_strategy_returns is not None and fixed_strategy_label is not None:
        comparison_returns[fixed_strategy_label] = fixed_strategy_returns.reindex(
            walk_forward_returns.index
        )

    return comparison_returns.dropna(axis=1, how="any")


def format_period_dates(index):
    index = pd.Index(index).dropna()

    if index.empty:
        return ""

    return f"{index.min().date()} - {index.max().date()}"


def format_period_years(index):
    index = pd.Index(index).dropna()

    if index.empty:
        return ""

    return f"{index.min().year}-{index.max().year}"


def plot_walk_forward_comparison(
    walk_forward_returns,
    resource_universe_returns,
    benchmark_returns,
    fixed_strategy_returns=None,
    fixed_strategy_label=None,
    show_plots=True,
):
    """
    benchmarks are rebased with no earlier head start
    """
    comparison_returns = build_walk_forward_comparison_returns(
        walk_forward_returns=walk_forward_returns,
        resource_universe_returns=resource_universe_returns,
        benchmark_returns=benchmark_returns,
        fixed_strategy_returns=fixed_strategy_returns,
        fixed_strategy_label=fixed_strategy_label,
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
        title=(
            "Walk-Forward Model-Selection Robustness vs Benchmarks "
            f"({format_period_years(comparison_returns.index)})"
        ),
        show_plots=show_plots,
    )


def plot_fixed_strategy_oos_growth(
    selected_strategy_name,
    strategy_returns_frame,
    resource_universe_returns,
    benchmark_returns,
    test_index,
    show_plots=True,
):
    comparison_returns = pd.DataFrame(
        {
            selected_strategy_name: strategy_returns_frame[
                selected_strategy_name
            ].reindex(test_index),
            "Resource universe equal-weight": resource_universe_returns.reindex(
                test_index
            ),
        }
    )

    for benchmark in ["SPY", "XLE", "GLD"]:
        if benchmark in benchmark_returns:
            comparison_returns[benchmark] = benchmark_returns[benchmark].reindex(
                test_index
            )

    comparison_returns = comparison_returns.dropna(how="any")

    if comparison_returns.empty:
        return

    equity_curves = build_equity_curves(
        comparison_returns,
        include_start=True,
    )
    plot_equity_curves(
        equity_curves=equity_curves,
        columns=comparison_returns.columns,
        title="Fixed Strategy — Held-Out OOS Growth of $1",
        subtitle=(
            "Selected using in-sample Sharpe only; "
            f"held-out period {format_period_dates(comparison_returns.index)}"
        ),
        primary_column=selected_strategy_name,
        show_plots=show_plots,
    )


def plot_walk_forward_robustness_growth(
    walk_forward_common_returns,
    show_plots=True,
):
    columns = [
        "Single-winner walk-forward",
        "Top-3 ensemble",
        "Top-5 ensemble",
        "Diversified Top-5 ensemble",
        "Resource universe equal-weight",
        "SPY",
    ]
    available_columns = [
        column
        for column in columns
        if column in walk_forward_common_returns
    ]

    if not available_columns:
        return

    comparison_returns = walk_forward_common_returns[available_columns].dropna(
        how="any"
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
        title="Walk-Forward Robustness Analysis — Growth of $1",
        subtitle=(
            "Annual model-selection robustness; "
            f"evaluation period {format_period_dates(comparison_returns.index)}"
        ),
        primary_column="Single-winner walk-forward",
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

    display_index = [get_strategy_display_name(name) for name in sharpe.index]

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    ax.barh(display_index, sharpe, alpha=0.88)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Sharpe Ratio", fontsize=10)
    ax.set_ylabel("")
    apply_research_plot_style(ax)
    fig.tight_layout()

    if not show_plots:
        plt.close(fig)


def plot_oos_sharpe_for_in_sample_winners(
    top_in_sample,
    out_sample_results,
    period_index,
    top_n=15,
    show_plots=True,
):
    selected_strategies = top_in_sample.head(top_n).index
    sharpe = out_sample_results.reindex(selected_strategies)["sharpe_ratio"].dropna()

    if sharpe.empty:
        return

    sharpe = sharpe.iloc[::-1]

    plt.figure(figsize=(12, max(6, 0.35 * len(sharpe))))
    plt.barh(sharpe.index, sharpe)
    plt.title(
        "OOS Sharpe of Top In-Sample Strategies "
        f"({format_period_dates(period_index)})"
    )
    plt.xlabel("Sharpe Ratio")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if not show_plots:
        plt.close()


def plot_in_sample_vs_oos_sharpe(
    top_in_sample,
    out_sample_results,
    train_index,
    test_index,
    top_n=15,
    show_plots=True,
):
    selected_strategies = top_in_sample.head(top_n).index
    comparison = pd.DataFrame(
        {
            "In-sample": top_in_sample.reindex(selected_strategies)["sharpe_ratio"],
            "OOS": out_sample_results.reindex(selected_strategies)[
                "sharpe_ratio"
            ],
        }
    ).dropna(how="all")

    if comparison.empty:
        return

    comparison.index = [
        get_strategy_display_name(strategy_name)
        for strategy_name in comparison.index
    ]
    comparison = comparison.iloc[::-1]

    ax = comparison.plot.barh(
        figsize=BAR_FIGSIZE,
        width=0.72,
    )
    ax.set_title(
        "In-Sample vs OOS Sharpe of Top In-Sample Strategies "
        f"({format_period_dates(train_index)} vs {format_period_dates(test_index)})"
        "\nOrdered by in-sample Sharpe; OOS values are held-out results",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Sharpe Ratio", fontsize=10)
    ax.set_ylabel("")
    ax.legend(fontsize=9, frameon=True, framealpha=0.92)
    apply_research_plot_style(ax)
    plt.tight_layout()

    if not show_plots:
        plt.close()


def plot_annual_universe_table(annual_universes, show_plots=True):
    """
    Show one annual dynamic universe at a time.
    """
    if not annual_universes:
        return

    years = sorted(annual_universes)
    current_index = 0
    fig, ax = plt.subplots(figsize=(18, 12))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.12)
    ax.axis("off")

    previous_ax = fig.add_axes([0.38, 0.035, 0.10, 0.045])
    next_ax = fig.add_axes([0.52, 0.035, 0.10, 0.045])
    previous_button = Button(previous_ax, "Previous")
    next_button = Button(next_ax, "Next")

    def render_year(index):
        year = years[index]
        securities = [str(security) for security in annual_universes[year]]
        column_count = min(5, max(1, math.ceil(len(securities) / 45)))
        rows_per_column = max(1, math.ceil(len(securities) / column_count))
        font_size = 8 if rows_per_column <= 45 else 7

        ax.clear()
        ax.axis("off")
        ax.set_title(
            f"{year} - {len(securities)} names",
            fontsize=16,
            fontweight="bold",
            pad=18,
        )

        for column in range(column_count):
            start = column * rows_per_column
            end = start + rows_per_column
            column_items = securities[start:end]
            x_position = column / column_count + 0.01

            for row, security in enumerate(column_items):
                y_position = 0.98 - (row / rows_per_column) * 0.94
                ax.text(
                    x_position,
                    y_position,
                    security,
                    transform=ax.transAxes,
                    fontsize=font_size,
                    family="monospace",
                    va="top",
                )

        fig.suptitle(
            f"Annual Dynamic Universe Membership ({index + 1} of {len(years)})",
            y=0.985,
            fontsize=10,
        )
        fig.canvas.draw_idle()

    def show_previous(_event):
        nonlocal current_index
        current_index = (current_index - 1) % len(years)
        render_year(current_index)

    def show_next(_event):
        nonlocal current_index
        current_index = (current_index + 1) % len(years)
        render_year(current_index)

    previous_button.on_clicked(show_previous)
    next_button.on_clicked(show_next)
    fig._annual_universe_buttons = (previous_button, next_button)

    render_year(current_index)

    if not show_plots:
        plt.close(fig)


def build_walk_forward_comparison_summary(
    strategy_returns_frame,
    walk_forward_returns,
    resource_universe_returns,
    benchmark_monthly_returns,
    top_full_sample,
    train_months,
    min_ranking_months,
):
    """
    Compare walk-forward selection with fixed strategy references.
    """
    walk_forward_returns = walk_forward_returns.dropna()

    if walk_forward_returns.empty:
        return pd.DataFrame()

    evaluation_index = walk_forward_returns.index
    usable_returns = strategy_returns_frame.dropna(how="all")
    initial_train_slice = usable_returns.iloc[:train_months]
    initial_train_results = rank_results(
        results=calculate_results_from_frame(initial_train_slice),
        by="sharpe_ratio",
        min_periods=min_ranking_months,
    )

    rows = []

    def add_row(label, strategy_name, returns, valid_oos_comparison):
        metrics = calculate_performance_metrics(returns.reindex(evaluation_index))
        rows.append(
            {
                "comparison": label,
                "selected_strategy": strategy_name,
                "evaluation_start": evaluation_index.min(),
                "evaluation_end": evaluation_index.max(),
                "valid_oos_comparison": valid_oos_comparison,
                "annualized_return": metrics["annualized_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar_ratio": metrics["calmar_ratio"],
                "final_value": metrics["final_value"],
            }
        )

    add_row(
        label="A. Walk-forward selected strategy",
        strategy_name="rolling annual selection",
        returns=walk_forward_returns,
        valid_oos_comparison=True,
    )

    if not initial_train_results.empty:
        fixed_in_sample_strategy = initial_train_results.index[0]
        add_row(
            label="B. Best fixed initial in-sample strategy",
            strategy_name=fixed_in_sample_strategy,
            returns=strategy_returns_frame[fixed_in_sample_strategy],
            valid_oos_comparison=True,
        )

    if not top_full_sample.empty:
        full_sample_strategy = top_full_sample.index[0]
        add_row(
            label="C. Best fixed full-sample strategy (reference only)",
            strategy_name=full_sample_strategy,
            returns=strategy_returns_frame[full_sample_strategy],
            valid_oos_comparison=False,
        )

    add_row(
        label="D. Resource universe equal-weight",
        strategy_name="resource universe",
        returns=resource_universe_returns,
        valid_oos_comparison=True,
    )

    if "SPY" in benchmark_monthly_returns:
        add_row(
            label="E. SPY",
            strategy_name="SPY",
            returns=benchmark_monthly_returns["SPY"],
            valid_oos_comparison=True,
        )

    return pd.DataFrame(rows).set_index("comparison")


def build_oos_benchmark_comparison(
    strategy_returns_frame,
    top_in_sample,
    resource_universe_returns,
    benchmark_monthly_returns,
    test_index,
):
    """
    Compare the best initial in-sample strategy over the exact held-out window.
    """
    if top_in_sample.empty:
        return pd.DataFrame(), pd.DataFrame()

    test_index = pd.Index(test_index)
    selected_strategy = top_in_sample.index[0]
    rows = []

    comparison_inputs = [
        (
            "A. Best fixed strategy selected in-sample only",
            selected_strategy,
            strategy_returns_frame[selected_strategy],
        ),
        (
            "B. Resource universe equal-weight",
            "resource universe",
            resource_universe_returns,
        ),
    ]

    for benchmark in ["SPY", "XLE", "GLD"]:
        if benchmark in benchmark_monthly_returns:
            comparison_inputs.append(
                (
                    f"{chr(ord('A') + len(comparison_inputs))}. {benchmark}",
                    benchmark,
                    benchmark_monthly_returns[benchmark],
                )
            )

    for label, name, returns in comparison_inputs:
        metrics = calculate_performance_metrics(returns.reindex(test_index))
        rows.append(
            {
                "comparison": label,
                "selected_strategy": name,
                "evaluation_start": test_index.min(),
                "evaluation_end": test_index.max(),
                "annualized_return": metrics["annualized_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar_ratio": metrics["calmar_ratio"],
                "final_value": metrics["final_value"],
            }
        )

    comparison = pd.DataFrame(rows).set_index("comparison")
    strategy_row = comparison.iloc[0]
    beat_rows = []

    for label, row in comparison.iloc[1:].iterrows():
        beat_rows.append(
            {
                "benchmark": row["selected_strategy"],
                "beat_annualized_return": (
                    strategy_row["annualized_return"] > row["annualized_return"]
                ),
                "beat_sharpe": strategy_row["sharpe_ratio"] > row["sharpe_ratio"],
            }
        )

    beat_table = pd.DataFrame(beat_rows).set_index("benchmark")

    return comparison, beat_table


def build_fixed_strategy_oos_summary(
    top_in_sample,
    out_sample_results,
    train_index,
    test_index,
):
    if top_in_sample.empty:
        return pd.DataFrame()

    selected_strategy = top_in_sample.index[0]
    out_row = out_sample_results.loc[selected_strategy]

    return pd.DataFrame(
        [
            {
                "selected_using": "in-sample Sharpe only",
                "in_sample_start": pd.Index(train_index).min(),
                "in_sample_end": pd.Index(train_index).max(),
                "oos_start": pd.Index(test_index).min(),
                "oos_end": pd.Index(test_index).max(),
                "selected_strategy": selected_strategy,
                "oos_annualized_return": out_row["annualized_return"],
                "oos_annualized_volatility": out_row["annualized_volatility"],
                "oos_sharpe_ratio": out_row["sharpe_ratio"],
                "oos_max_drawdown": out_row["max_drawdown"],
                "oos_calmar_ratio": out_row["calmar_ratio"],
                "oos_final_value": out_row["final_value"],
            }
        ],
        index=["Fixed 70/30 OOS"],
    )


def build_inverse_volatility_experiment_report(
    in_sample_results,
    out_sample_results,
    benchmark_monthly_returns,
    resource_universe_returns,
    test_index,
):
    """
    Compare the controlled inverse-volatility variants with equal weights.
    """
    rows = []
    signal_name = "voladj6_12-1"

    for top_n in [10, 20, 30]:
        for weighting in ["equal", "inverse_vol"]:
            strategy_name = (
                f"{signal_name} | long_only | {weighting} | top{top_n}"
            )

            if (
                strategy_name not in in_sample_results.index
                or strategy_name not in out_sample_results.index
            ):
                continue

            in_row = in_sample_results.loc[strategy_name]
            out_row = out_sample_results.loc[strategy_name]
            rows.append(
                {
                    "comparison": f"top{top_n} {weighting}",
                    "strategy_name": strategy_name,
                    "top_n": top_n,
                    "weighting": weighting,
                    "in_sample_annualized_return": in_row["annualized_return"],
                    "in_sample_annualized_volatility": (
                        in_row["annualized_volatility"]
                    ),
                    "in_sample_sharpe_ratio": in_row["sharpe_ratio"],
                    "in_sample_max_drawdown": in_row["max_drawdown"],
                    "in_sample_calmar_ratio": in_row["calmar_ratio"],
                    "out_sample_annualized_return": out_row["annualized_return"],
                    "out_sample_annualized_volatility": (
                        out_row["annualized_volatility"]
                    ),
                    "out_sample_sharpe_ratio": out_row["sharpe_ratio"],
                    "out_sample_max_drawdown": out_row["max_drawdown"],
                    "out_sample_calmar_ratio": out_row["calmar_ratio"],
                    "out_sample_final_value": out_row["final_value"],
                }
            )

    benchmark_inputs = [
        ("resource universe", resource_universe_returns),
    ]

    for benchmark in ["SPY", "XLE", "GLD"]:
        if benchmark in benchmark_monthly_returns:
            benchmark_inputs.append((benchmark, benchmark_monthly_returns[benchmark]))

    for name, returns in benchmark_inputs:
        metrics = calculate_performance_metrics(returns.reindex(test_index))
        rows.append(
            {
                "comparison": name,
                "strategy_name": name,
                "top_n": None,
                "weighting": "benchmark",
                "in_sample_annualized_return": None,
                "in_sample_annualized_volatility": None,
                "in_sample_sharpe_ratio": None,
                "in_sample_max_drawdown": None,
                "out_sample_annualized_return": metrics["annualized_return"],
                "out_sample_annualized_volatility": metrics[
                    "annualized_volatility"
                ],
                "out_sample_sharpe_ratio": metrics["sharpe_ratio"],
                "out_sample_max_drawdown": metrics["max_drawdown"],
                "out_sample_calmar_ratio": metrics["calmar_ratio"],
                "out_sample_final_value": metrics["final_value"],
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("comparison")


def build_walk_forward_ensemble_comparison(
    walk_forward_returns,
    ensemble_returns,
    resource_universe_returns,
    benchmark_monthly_returns,
):
    rows = []
    comparison_returns = pd.DataFrame(
        {"Single-winner walk-forward": walk_forward_returns}
    )

    for method_name in [
        "Top-3 ensemble",
        "Top-5 ensemble",
        "Diversified Top-5 ensemble",
    ]:
        if method_name in ensemble_returns:
            comparison_returns[method_name] = ensemble_returns[method_name]

    comparison_returns["Resource universe equal-weight"] = resource_universe_returns

    for benchmark in ["SPY", "XLE", "GLD"]:
        if benchmark in benchmark_monthly_returns:
            comparison_returns[benchmark] = benchmark_monthly_returns[benchmark]

    comparison_returns = comparison_returns.dropna(how="any")

    if comparison_returns.empty:
        return pd.DataFrame(), comparison_returns

    for column in comparison_returns.columns:
        metrics = calculate_performance_metrics(comparison_returns[column])
        rows.append(
            {
                "method": column,
                "evaluation_start": comparison_returns.index.min(),
                "evaluation_end": comparison_returns.index.max(),
                "cumulative_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar_ratio": metrics["calmar_ratio"],
                "final_value": metrics["final_value"],
            }
        )

    if not rows:
        return pd.DataFrame(), comparison_returns

    return pd.DataFrame(rows).set_index("method"), comparison_returns


def build_walk_forward_annual_performance_table(
    walk_forward_comparison_returns,
):
    method_columns = [
        "Single-winner walk-forward",
        "Top-3 ensemble",
        "Top-5 ensemble",
        "Diversified Top-5 ensemble",
    ]
    available_columns = [
        column
        for column in method_columns
        if column in walk_forward_comparison_returns
    ]
    rows = []

    for year, year_returns in walk_forward_comparison_returns[
        available_columns
    ].groupby(walk_forward_comparison_returns.index.year):
        row = {"year": year}

        for column in available_columns:
            metrics = calculate_performance_metrics(year_returns[column])
            output_column = {
                "Single-winner walk-forward": "single_winner_return",
                "Top-3 ensemble": "top3_return",
                "Top-5 ensemble": "top5_return",
                "Diversified Top-5 ensemble": "diversified_top5_return",
            }[column]
            row[output_column] = metrics["cumulative_return"]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("year")


def classify_selected_signal_family(strategy_name):
    if strategy_name.startswith("voladj6_12-1"):
        return "voladj6_12-1"

    if strategy_name.startswith("voladj6_"):
        return "other volatility-adjusted momentum"

    if strategy_name.startswith("raw_"):
        return "raw momentum"

    if (
        strategy_name.startswith("sma_")
        or strategy_name.startswith("ema_")
        or strategy_name.startswith("wma_")
    ):
        return "moving-average signals"

    return "other"


def classify_selected_strategy_type(strategy_name):
    if "classic_long_short" in strategy_name:
        return "classic long-short"

    if "directional_abs" in strategy_name:
        return "directional absolute"

    return "long-only"


def build_selection_distribution_table(
    ensemble_decisions,
    classifier,
    categories,
    index_name,
):
    counts = {category: 0 for category in categories}
    total_selected_members = 0

    if ensemble_decisions.empty:
        return pd.DataFrame()

    for selected_names in ensemble_decisions["selected_strategy_names"]:
        for strategy_name in str(selected_names).split(" ; "):
            if not strategy_name:
                continue

            total_selected_members += 1
            category = classifier(strategy_name)
            counts[category] = counts.get(category, 0) + 1

    rows = []

    for category in categories:
        rows.append(
            {
                index_name: category,
                "selected_count": counts[category],
                "selection_share": (
                    counts[category] / total_selected_members
                    if total_selected_members
                    else None
                ),
            }
        )

    return pd.DataFrame(rows).set_index(index_name)


def build_walk_forward_signal_family_diagnostics(ensemble_decisions):
    categories = [
        "voladj6_12-1",
        "other volatility-adjusted momentum",
        "raw momentum",
        "moving-average signals",
        "other",
    ]

    return build_selection_distribution_table(
        ensemble_decisions=ensemble_decisions,
        classifier=classify_selected_signal_family,
        categories=categories,
        index_name="signal_family",
    )


def build_walk_forward_strategy_type_diagnostics(ensemble_decisions):
    categories = [
        "long-only",
        "classic long-short",
        "directional absolute",
    ]

    return build_selection_distribution_table(
        ensemble_decisions=ensemble_decisions,
        classifier=classify_selected_strategy_type,
        categories=categories,
        index_name="strategy_type",
    )


def build_walk_forward_final_summary(
    walk_forward_ensemble_comparison,
):
    if walk_forward_ensemble_comparison.empty:
        return pd.DataFrame()

    summary_rows = [
        "Single-winner walk-forward",
        "Top-3 ensemble",
        "Top-5 ensemble",
        "Diversified Top-5 ensemble",
        "Resource universe equal-weight",
        "SPY",
    ]
    columns = [
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
    ]
    available_rows = [
        row
        for row in summary_rows
        if row in walk_forward_ensemble_comparison.index
    ]

    return walk_forward_ensemble_comparison.loc[available_rows, columns]


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
        "calmar_ratio",
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
