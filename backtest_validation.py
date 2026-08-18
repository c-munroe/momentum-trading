"""
Train/test and walk-forward validation helpers.
"""

import pandas as pd

from backtest_metrics import (
    calculate_performance_metrics,
    calculate_results_from_frame,
    calculate_results_table,
)


def rank_results(results, by="sharpe_ratio", min_periods=36):
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


def split_train_test_index(returns_frame, train_fraction=0.70, min_ranking_months=36):
    usable_index = returns_frame.dropna(how="all").index

    if len(usable_index) < min_ranking_months + 1:
        raise ValueError("Not enough monthly return history for a train/test split.")

    split_position = int(len(usable_index) * train_fraction)
    split_position = max(min_ranking_months, split_position)
    split_position = min(split_position, len(usable_index) - 1)

    return usable_index[:split_position], usable_index[split_position:]


def calculate_period_results(
    strategy_returns,
    strategy_metadata,
    benchmark_returns,
    period_index,
    relative_benchmarks,
):
    return calculate_results_table(
        strategy_returns=slice_returns_dict(strategy_returns, period_index),
        strategy_metadata=strategy_metadata,
        benchmark_returns=benchmark_returns.reindex(period_index),
        relative_benchmarks=relative_benchmarks,
    )


def build_train_test_validation_summary(
    in_sample_results,
    out_sample_results,
    top_n=15,
    min_ranking_months=36,
):
    top_in_sample = rank_results(
        in_sample_results,
        by="sharpe_ratio",
        min_periods=min_ranking_months,
    ).head(top_n)
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
    train_months=120,
    test_months=12,
    ranking_metric="sharpe_ratio",
    min_ranking_months=36,
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
            min_periods=min_ranking_months,
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
