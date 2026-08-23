"""
Universe setup helpers for static and dynamic backtest modes.
"""

import pandas as pd

from universe import (
    LIQUIDITY_LOOKBACK_DAYS,
    MIN_AVG_DAILY_DOLLAR_VOLUME,
    MIN_UNIVERSE_SIZE,
    build_crsp_dynamic_universe_data,
    get_universe_for_date,
    print_annual_diagnostics,
)


def validate_universe_mode(universe_mode):
    if universe_mode not in {"static", "dynamic"}:
        raise ValueError("UNIVERSE_MODE must be either 'static' or 'dynamic'.")


def build_monthly_eligibility_table(monthly_index, tickers, annual_universes):
    """
    Convert annual universes into a month-end boolean eligibility matrix.
    """
    table = pd.DataFrame(False, index=monthly_index, columns=tickers)

    for date in monthly_index:
        eligible_tickers = get_universe_for_date(
            date=date,
            annual_universes=annual_universes,
        )
        available_tickers = [ticker for ticker in eligible_tickers if ticker in table]
        table.loc[date, available_tickers] = True

    return table


def print_static_universe_diagnostics(resource_monthly_returns):
    print("\nUniverse mode: static")
    print(f"Resource tickers: {resource_monthly_returns.shape[1]}")


def print_dynamic_universe_setup(result):
    print("\nUniverse mode: dynamic (CRSP)")
    print(
        f"Data source: CRSP {result.data_start_date.year}-{result.data_end_date.year}"
    )
    print("Resource classification: point-in-time SIC/NAICS")
    print("Market cap: >= $1B")
    print(
        "Liquidity: "
        f">= ${MIN_AVG_DAILY_DOLLAR_VOLUME / 1_000_000:.0f}M "
        f"{LIQUIDITY_LOOKBACK_DAYS}-day ADDV"
    )
    print(f"Minimum universe target: {MIN_UNIVERSE_SIZE}")
    print("Maximum universe: none")


def build_dynamic_universe():
    """
    Build CRSP-backed annual universes and resource return matrices.
    """
    crsp_dynamic_data = build_crsp_dynamic_universe_data()
    result = crsp_dynamic_data.result

    print_dynamic_universe_setup(result)
    print_annual_diagnostics(result.diagnostics)

    return (
        result,
        crsp_dynamic_data.eligibility_table,
        crsp_dynamic_data.monthly_prices,
        crsp_dynamic_data.monthly_returns,
    )


def prepare_universe_mode(
    universe_mode,
    resource_monthly_prices=None,
    resource_monthly_returns=None,
):
    validate_universe_mode(universe_mode)

    if universe_mode == "static":
        print_static_universe_diagnostics(resource_monthly_returns)
        return None, None, resource_monthly_prices, resource_monthly_returns

    return build_dynamic_universe()
