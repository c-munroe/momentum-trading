"""
Performance metrics and equity-curve helpers for the momentum backtest.
"""

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 12


def calculate_equal_weight_universe_returns(monthly_returns, eligibility_table=None):
    """
    Calculate an equal-weight resource-universe return

    When eligibility_table is supplied, only currently eligible names are included
    in that month's resource-universe benchmark.
    """
    if eligibility_table is None:
        return monthly_returns.mean(axis=1, skipna=True)

    aligned_table = eligibility_table.reindex(
        index=monthly_returns.index,
        columns=monthly_returns.columns,
        fill_value=False,
    )

    return monthly_returns.where(aligned_table).mean(axis=1, skipna=True)


def get_curve_start_index(first_return_index):
    if isinstance(first_return_index, pd.Timestamp):
        return first_return_index - pd.Timedelta(nanoseconds=1)

    return first_return_index


def returns_to_equity_curve(returns, include_start=False):
    returns = returns.dropna()
    equity_curve = (1 + returns).cumprod()

    if include_start and not returns.empty:
        start_index = get_curve_start_index(returns.index[0])
        start_value = pd.Series([1.0], index=[start_index])

        return pd.concat([start_value, equity_curve])

    return equity_curve


def build_equity_curves(returns_df, include_start=False):
    return returns_df.apply(
        lambda returns: returns_to_equity_curve(
            returns,
            include_start=include_start,
        )
    )


def calculate_sharpe_ratio(
    returns,
    periods_per_year=PERIODS_PER_YEAR,
):
    """
    annualized sharpe, zero risk-free rate

    uses mean monthly return / sample monthly volatility * sqrt(periods per year)
    """
    if periods_per_year <= 0:
        return None

    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 2:
        return None

    mean_return = returns.mean()
    return_volatility = returns.std(ddof=1)

    if (
        not np.isfinite(mean_return)
        or not np.isfinite(return_volatility)
        or np.isclose(return_volatility, 0.0)
    ):
        return None

    sharpe_ratio = mean_return / return_volatility * np.sqrt(periods_per_year)

    return sharpe_ratio if np.isfinite(sharpe_ratio) else None


def calculate_calmar_ratio(annualized_return, max_drawdown):
    """
    Calculate Calmar as annualized return divided by absolute max drawdown.
    """
    if annualized_return is None or max_drawdown is None:
        return None

    if not np.isfinite(annualized_return) or not np.isfinite(max_drawdown):
        return None

    if np.isclose(max_drawdown, 0.0):
        return np.nan

    calmar_ratio = annualized_return / abs(max_drawdown)

    return calmar_ratio if np.isfinite(calmar_ratio) else None


def calculate_performance_metrics(returns):
    """
    Calculate the core metrics needed to rank and compare strategies

    annualized_return is compounded annual growth. Sharpe uses zero rf
    """
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        return {
            "periods": 0,
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "calmar_ratio": None,
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
    periodic_volatility = returns.std(ddof=1)
    annualized_volatility = (
        periodic_volatility * np.sqrt(PERIODS_PER_YEAR)
        if np.isfinite(periodic_volatility)
        else None
    )
    sharpe_ratio = calculate_sharpe_ratio(
        returns=returns,
        periods_per_year=PERIODS_PER_YEAR,
    )
    drawdowns = equity_curve / equity_curve.cummax() - 1
    max_drawdown = drawdowns.min()

    return {
        "periods": len(returns),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calculate_calmar_ratio(
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
        ),
        "win_rate": (returns > 0).mean(),
        "final_value": final_value,
        "valid_for_chart": final_value > 0 and max_drawdown > -1,
    }


def calculate_relative_metrics(strategy_returns, benchmark_returns, benchmark_name):
    """
    Calculate beta, alpha, correlation, and information ratio vs one benchmark
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


def calculate_results_table(
    strategy_returns,
    strategy_metadata,
    benchmark_returns,
    relative_benchmarks,
):
    rows = []

    for strategy_name, returns in strategy_returns.items():
        row = {"strategy_name": strategy_name}
        row.update(strategy_metadata.loc[strategy_name].to_dict())
        row.update(calculate_performance_metrics(returns))

        for benchmark in relative_benchmarks:
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
