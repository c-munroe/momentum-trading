"""
Run the natural-resource momentum research grid.

This is the project-level integration point. It assembles universe
construction, data preparation, strategy generation, portfolio construction,
backtesting, validation, benchmarking, reporting, and plotting.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from backtest.data_download import (
    ALL_TICKERS,
    BENCHMARK_TICKERS,
    COMMODITY_FUTURES_TICKERS,
    build_dataset,
    download_price_data,
    get_last_complete_month_end,
    keep_complete_months,
    report_download_summary,
)
from backtest.metrics import (
    calculate_equal_weight_universe_returns,
    calculate_performance_metrics,
    calculate_results_table,
)
from backtest.reporting import (
    build_fixed_strategy_oos_summary,
    build_inverse_volatility_experiment_report,
    build_oos_benchmark_comparison,
    build_walk_forward_annual_performance_table,
    build_walk_forward_comparison_summary,
    build_walk_forward_ensemble_comparison,
    build_walk_forward_final_summary,
    build_walk_forward_signal_family_diagnostics,
    build_walk_forward_strategy_type_diagnostics,
    format_period_dates,
    plot_annual_universe_table,
    plot_fixed_strategy_oos_growth,
    plot_in_sample_vs_oos_sharpe,
    plot_sharpe_bars,
    plot_walk_forward_robustness_growth,
    print_summary_table,
    print_table,
)
from backtest.universe_setup import prepare_universe_mode
from backtest.validation import (
    build_train_test_validation_summary,
    calculate_period_results,
    calculate_walk_forward_ensemble_validation,
    calculate_walk_forward_validation,
    rank_results,
    split_train_test_index,
)
from strats.grid import run_strategy_grid


SHOW_PLOTS = os.getenv("SHOW_PLOTS", "1").lower() not in {"0", "false", "no"}

# static preserves the original hand-curated universe behavior
# dynamic uses CRSP annual PERMNO universes for resource stocks
UNIVERSE_MODE = os.getenv("UNIVERSE_MODE", "dynamic").lower()

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

RELATIVE_BENCHMARKS = ["SPY", "XLE", "XLB", "GLD", "DBC"]


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)


def main():
    download_tickers = ALL_TICKERS

    if UNIVERSE_MODE == "dynamic":
        download_tickers = BENCHMARK_TICKERS + COMMODITY_FUTURES_TICKERS

    prices = download_price_data(download_tickers)
    report_download_summary(
        prices,
        expected_tickers=download_tickers,
        show_missing_tickers=UNIVERSE_MODE == "static",
    )

    dataset = build_dataset(
        prices,
        resource_tickers=[] if UNIVERSE_MODE == "dynamic" else None,
    )
    last_complete_month_end = get_last_complete_month_end()
    benchmark_monthly_returns = keep_complete_months(
        dataset["benchmark"]["monthly_returns"],
        last_complete_month_end,
    )

    if UNIVERSE_MODE == "static":
        resource_monthly_prices = keep_complete_months(
            dataset["resource"]["monthly_prices"],
            last_complete_month_end,
        )
        resource_monthly_returns = keep_complete_months(
            dataset["resource"]["monthly_returns"],
            last_complete_month_end,
        )
    else:
        resource_monthly_prices = None
        resource_monthly_returns = None

    (
        dynamic_universe_result,
        eligibility_table,
        resource_monthly_prices,
        resource_monthly_returns,
    ) = prepare_universe_mode(
        universe_mode=UNIVERSE_MODE,
        resource_monthly_prices=resource_monthly_prices,
        resource_monthly_returns=resource_monthly_returns,
    )

    stock_month_end = resource_monthly_returns.index.max()
    benchmark_monthly_returns = keep_complete_months(
        benchmark_monthly_returns,
        stock_month_end,
    )

    if UNIVERSE_MODE == "dynamic":
        print(f"\nUsing CRSP stock monthly data through {stock_month_end.date()}.")
    else:
        print(f"\nUsing completed monthly data through {stock_month_end.date()}.")

    strategy_returns, strategy_metadata, _ = run_strategy_grid(
        resource_monthly_prices=resource_monthly_prices,
        resource_monthly_returns=resource_monthly_returns,
        eligibility_table=eligibility_table,
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
    ensemble_returns, ensemble_decisions = calculate_walk_forward_ensemble_validation(
        returns_frame=strategy_returns_frame,
        strategy_metadata=strategy_metadata,
        train_months=WALK_FORWARD_TRAIN_MONTHS,
        test_months=WALK_FORWARD_TEST_MONTHS,
        min_ranking_months=MIN_RANKING_MONTHS,
    )
    walk_forward_metrics = calculate_performance_metrics(walk_forward_returns)
    resource_universe_returns = calculate_equal_weight_universe_returns(
        monthly_returns=resource_monthly_returns,
        eligibility_table=eligibility_table,
    )
    (
        walk_forward_ensemble_comparison,
        walk_forward_common_returns,
    ) = build_walk_forward_ensemble_comparison(
        walk_forward_returns=walk_forward_returns,
        ensemble_returns=ensemble_returns,
        resource_universe_returns=resource_universe_returns,
        benchmark_monthly_returns=benchmark_monthly_returns,
    )
    walk_forward_annual_performance = build_walk_forward_annual_performance_table(
        walk_forward_common_returns,
    )
    walk_forward_signal_family_diagnostics = build_walk_forward_signal_family_diagnostics(
        ensemble_decisions,
    )
    walk_forward_strategy_type_diagnostics = build_walk_forward_strategy_type_diagnostics(
        ensemble_decisions,
    )
    walk_forward_final_summary = build_walk_forward_final_summary(
        walk_forward_ensemble_comparison,
    )
    oos_comparison, oos_beat_table = build_oos_benchmark_comparison(
        strategy_returns_frame=strategy_returns_frame,
        top_in_sample=top_in_sample,
        resource_universe_returns=resource_universe_returns,
        benchmark_monthly_returns=benchmark_monthly_returns,
        test_index=test_index,
    )
    fixed_strategy_oos_summary = build_fixed_strategy_oos_summary(
        top_in_sample=top_in_sample,
        out_sample_results=out_sample_results,
        train_index=train_index,
        test_index=test_index,
    )
    inverse_volatility_experiment = build_inverse_volatility_experiment_report(
        in_sample_results=in_sample_results,
        out_sample_results=out_sample_results,
        benchmark_monthly_returns=benchmark_monthly_returns,
        resource_universe_returns=resource_universe_returns,
        test_index=test_index,
    )
    dynamic_return_coverage = pd.DataFrame()

    if dynamic_universe_result is not None:
        universe_available = eligibility_table.any(axis=1)
        first_universe_month = universe_available[universe_available].index.min()
        pre_universe_returns = strategy_returns_frame.loc[
            strategy_returns_frame.index < first_universe_month
        ]
        valid_strategy_index = strategy_returns_frame.dropna(how="all").index
        first_walk_forward_training_start = (
            walk_forward_decisions["train_start"].min()
            if not walk_forward_decisions.empty
            else pd.NaT
        )
        dynamic_return_coverage = pd.DataFrame(
            [
                {
                    "first_valid_universe_month": first_universe_month,
                    "pre_universe_non_null_values": int(
                        pre_universe_returns.notna().sum().sum()
                    ),
                    "first_valid_strategy_return_month": valid_strategy_index.min(),
                    "first_valid_train_test_month": train_index.min(),
                    "first_valid_walk_forward_training_start": (
                        first_walk_forward_training_start
                    ),
                }
            ],
            index=["Dynamic CRSP"],
        )

    if not top_in_sample.empty:
        plot_fixed_strategy_oos_growth(
            selected_strategy_name=top_in_sample.index[0],
            strategy_returns_frame=strategy_returns_frame,
            resource_universe_returns=resource_universe_returns,
            benchmark_returns=benchmark_monthly_returns,
            test_index=test_index,
            show_plots=SHOW_PLOTS,
        )

    plot_sharpe_bars(
        in_sample_results,
        title=(
            "Top Strategies by In-Sample Sharpe "
            f"({format_period_dates(train_index)})"
        ),
        min_ranking_months=MIN_RANKING_MONTHS,
        show_plots=SHOW_PLOTS,
    )
    plot_in_sample_vs_oos_sharpe(
        top_in_sample=top_in_sample,
        out_sample_results=out_sample_results,
        train_index=train_index,
        test_index=test_index,
        show_plots=SHOW_PLOTS,
    )
    plot_sharpe_bars(
        full_sample_results,
        title=(
            "Top Strategies by Full-Sample Sharpe "
            f"({format_period_dates(strategy_returns_frame.dropna(how='all').index)})"
            "\nDescriptive full-sample performance — not OOS evidence"
        ),
        min_ranking_months=MIN_RANKING_MONTHS,
        show_plots=SHOW_PLOTS,
    )
    plot_walk_forward_robustness_growth(
        walk_forward_common_returns=walk_forward_common_returns,
        show_plots=SHOW_PLOTS,
    )

    if SHOW_PLOTS:
        plt.show()
        plt.close("all")

        if dynamic_universe_result is not None:
            plot_annual_universe_table(
                annual_universes=(
                    dynamic_universe_result.display_annual_universes
                    or dynamic_universe_result.annual_universes
                ),
                show_plots=SHOW_PLOTS,
            )
            plt.show()
            plt.close("all")

    print(
        "\nTrain/test split: "
        f"{train_index.min().date()} to {train_index.max().date()} in-sample, "
        f"{test_index.min().date()} to {test_index.max().date()} out-of-sample."
    )
    if not dynamic_return_coverage.empty:
        print_table("Dynamic return coverage", dynamic_return_coverage)

    print_table(
        "PRIMARY VALIDATION: Fixed Strategy - 70/30 Out-of-Sample Validation",
        fixed_strategy_oos_summary,
    )
    print_table(
        "Fixed Strategy - Exact 70/30 OOS Benchmark Comparison",
        oos_comparison,
    )
    print_table(
        "Fixed Strategy - OOS Benchmark Beat Checks",
        oos_beat_table,
    )

    print_summary_table(
        "Supporting table: Top strategies by in-sample Sharpe",
        top_in_sample.head(10),
    )
    print_summary_table(
        "Supporting table: Top strategies by full-sample Sharpe",
        top_full_sample.head(10),
    )
    print_table(
        "Supporting table: Top in-sample strategies checked out of sample",
        validation_summary.head(10),
    )
    print_table(
        "Inverse-volatility sizing experiment",
        inverse_volatility_experiment,
    )

    walk_forward_summary = pd.DataFrame([walk_forward_metrics], index=["Walk-forward"])
    print_table(
        "SECONDARY ROBUSTNESS ANALYSIS: Walk-forward robustness - single-winner selector",
        walk_forward_summary,
    )

    walk_forward_comparison_summary = build_walk_forward_comparison_summary(
        strategy_returns_frame=strategy_returns_frame,
        walk_forward_returns=walk_forward_returns,
        resource_universe_returns=resource_universe_returns,
        benchmark_monthly_returns=benchmark_monthly_returns,
        top_full_sample=top_full_sample,
        train_months=WALK_FORWARD_TRAIN_MONTHS,
        min_ranking_months=MIN_RANKING_MONTHS,
    )
    print_table(
        "Walk-forward robustness: fixed-strategy diagnostic comparison",
        walk_forward_comparison_summary,
    )
    print_table(
        "Walk-forward robustness: model-selection ensembles",
        walk_forward_ensemble_comparison,
    )
    print_table(
        "Walk-forward robustness: annual method returns",
        walk_forward_annual_performance,
    )
    if not ensemble_decisions.empty:
        print_table(
            "Walk-forward robustness: ensemble selection-history diagnostics",
            ensemble_decisions.set_index(["method", "test_year"]),
        )
    print_table(
        "Walk-forward robustness: signal-family distribution",
        walk_forward_signal_family_diagnostics,
    )
    print_table(
        "Walk-forward robustness: strategy-type distribution",
        walk_forward_strategy_type_diagnostics,
    )
    print_table(
        "Walk-forward robustness: final summary",
        walk_forward_final_summary,
    )

    if not walk_forward_decisions.empty:
        print_table(
            "Walk-forward robustness: single-winner selection-history diagnostics",
            walk_forward_decisions.set_index("test_start"),
        )

    return {
        "strategy_returns": strategy_returns,
        "strategy_metadata": strategy_metadata,
        "in_sample_results": in_sample_results,
        "out_sample_results": out_sample_results,
        "full_sample_results": full_sample_results,
        "walk_forward_returns": walk_forward_returns,
        "walk_forward_decisions": walk_forward_decisions,
        "ensemble_returns": ensemble_returns,
        "ensemble_decisions": ensemble_decisions,
        "walk_forward_comparison_summary": walk_forward_comparison_summary,
        "walk_forward_ensemble_comparison": walk_forward_ensemble_comparison,
        "walk_forward_annual_performance": walk_forward_annual_performance,
        "walk_forward_signal_family_diagnostics": (
            walk_forward_signal_family_diagnostics
        ),
        "walk_forward_strategy_type_diagnostics": (
            walk_forward_strategy_type_diagnostics
        ),
        "walk_forward_final_summary": walk_forward_final_summary,
        "fixed_strategy_oos_summary": fixed_strategy_oos_summary,
        "oos_comparison": oos_comparison,
        "oos_beat_table": oos_beat_table,
        "inverse_volatility_experiment": inverse_volatility_experiment,
        "dynamic_return_coverage": dynamic_return_coverage,
        "resource_universe_returns": resource_universe_returns,
        "dynamic_universe_result": dynamic_universe_result,
        "eligibility_table": eligibility_table,
    }


if __name__ == "__main__":
    main()
