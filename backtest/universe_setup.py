"""
Universe setup helpers for static and dynamic backtest modes.
"""

import pandas as pd

from data_download import END_DATE, START_DATE
from universe import (
    DEFAULT_START_YEAR,
    LIQUIDITY_LOOKBACK_DAYS,
    MIN_MARKET_CAP,
    MIN_AVG_DAILY_DOLLAR_VOLUME,
    MIN_LIQUIDITY_OBSERVATIONS,
    MIN_UNIVERSE_SIZE,
    YAHOO_LISTED_EXCHANGES,
    YAHOO_RESOURCE_SECTORS,
    build_annual_universes,
    build_yahoo_resource_candidate_pool,
    calculate_historical_market_caps,
    download_historical_shares_outstanding,
    download_price_volume_data,
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


def print_dynamic_universe_setup(candidate_count):
    print("\nUniverse mode: dynamic")
    print("Candidate source: Yahoo current screener")
    print(f"Sectors: {', '.join(YAHOO_RESOURCE_SECTORS)}")
    print(f"Exchanges: {', '.join(YAHOO_LISTED_EXCHANGES)}")
    print(f"Candidates found: {candidate_count}")
    print("Survivorship bias: still present")
    print("\nActive filters:")
    print(f"Historical market cap: >= ${MIN_MARKET_CAP:,.0f}")
    print(f"Liquidity threshold: ${MIN_AVG_DAILY_DOLLAR_VOLUME:,.0f} ADDV")
    print(f"Liquidity lookback: {LIQUIDITY_LOOKBACK_DAYS} trading days")
    print(f"Minimum observations: {MIN_LIQUIDITY_OBSERVATIONS}")
    print(f"Minimum universe target: {MIN_UNIVERSE_SIZE}")
    print("Maximum annual universe: none")


def build_dynamic_candidate_pool():
    return build_yahoo_resource_candidate_pool()


def build_dynamic_universe(
    monthly_index,
    resource_tickers,
    candidate_tickers,
):
    """
    Build annual liquidity-filtered universes for dynamic development mode.

    The candidate list comes from today's Yahoo screener. That is broader than
    the static list, but still not a survivorship-bias-free security master
    """
    if candidate_tickers is None:
        candidate_tickers = build_dynamic_candidate_pool()

    print_dynamic_universe_setup(candidate_count=len(candidate_tickers))
    unadjusted_prices, volumes = download_price_volume_data(
        tickers=candidate_tickers,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    historical_shares = download_historical_shares_outstanding(
        tickers=unadjusted_prices.columns,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    historical_market_caps = calculate_historical_market_caps(
        prices=unadjusted_prices,
        shares_outstanding=historical_shares,
    )
    result = build_annual_universes(
        candidate_tickers=candidate_tickers,
        prices=unadjusted_prices,
        volumes=volumes,
        market_cap_data=historical_market_caps,
        subsector_data=None,
        require_resource_classification=False,
        start_year=DEFAULT_START_YEAR,
        end_year=monthly_index.max().year,
        allow_missing_market_cap=False,
    )
    result.candidate_source = "Yahoo current screener"
    result.candidate_count = len(candidate_tickers)
    result.shares_coverage_count = len(historical_shares.columns)
    eligibility_table = build_monthly_eligibility_table(
        monthly_index=monthly_index,
        tickers=resource_tickers,
        annual_universes=result.annual_universes,
    )

    print_annual_diagnostics(result.diagnostics)

    return result, eligibility_table


def prepare_universe_mode(
    universe_mode,
    resource_monthly_prices,
    resource_monthly_returns,
    candidate_tickers=None,
):
    validate_universe_mode(universe_mode)

    if universe_mode == "static":
        print_static_universe_diagnostics(resource_monthly_returns)
        return None, None

    return build_dynamic_universe(
        monthly_index=resource_monthly_returns.index,
        resource_tickers=resource_monthly_returns.columns,
        candidate_tickers=candidate_tickers,
    )
