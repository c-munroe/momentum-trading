"""
Data download and calendar aggregation.

Yahoo Finance adjusted closes are the raw input for this project. This module
keeps data access separate from strategy logic and produces consistent daily
and month-end price/return tables for resources, benchmarks, and futures.
"""

import pandas as pd
import yfinance as yf

from equities_list import (
    NATURAL_RESOURCE_TICKERS,
    BENCHMARK_TICKERS,
    COMMODITY_FUTURES_TICKERS,
)


START_DATE = "2000-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

ALL_TICKERS = NATURAL_RESOURCE_TICKERS + BENCHMARK_TICKERS + COMMODITY_FUTURES_TICKERS

DATASETS = {
    "resource": NATURAL_RESOURCE_TICKERS,
    "benchmark": BENCHMARK_TICKERS,
    "commodity_futures": COMMODITY_FUTURES_TICKERS,
}


def download_price_data(tickers, start_date=START_DATE, end_date=END_DATE):
    """
    Download adjusted close prices from Yahoo Finance.

    auto_adjust=True adjusts prices for dividends and splits, which gives
    a cleaner return series for backtesting.
    """
    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True,
        group_by="column",
        threads=True,
    )

    if raw_data.empty:
        raise ValueError("Yahoo Finance returned no data for the requested tickers.")

    close_prices = raw_data["Close"].copy()

    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame(name=tickers[0])

    # Remove tickers where Yahoo returned no price history.
    close_prices = close_prices.dropna(axis=1, how="all")

    return close_prices


def report_download_summary(prices):
    """
    Print which tickers downloaded successfully and which were missing.
    """
    downloaded_tickers = list(prices.columns)
    missing_tickers = sorted(set(ALL_TICKERS) - set(downloaded_tickers))

    print(f"Downloaded tickers: {len(downloaded_tickers)}")
    print(f"Missing tickers: {missing_tickers}")


def split_prices_by_universe(prices):
    """
    Split the full price DataFrame into resource stocks, benchmark ETFs,
    and commodity futures.
    """
    available_resource_tickers = [
        ticker for ticker in NATURAL_RESOURCE_TICKERS if ticker in prices.columns
    ]

    available_benchmark_tickers = [
        ticker for ticker in BENCHMARK_TICKERS if ticker in prices.columns
    ]

    available_commodity_futures_tickers = [
        ticker for ticker in COMMODITY_FUTURES_TICKERS if ticker in prices.columns
    ]

    resource_prices = prices[available_resource_tickers]
    benchmark_prices = prices[available_benchmark_tickers]
    commodity_futures_prices = prices[available_commodity_futures_tickers]

    return resource_prices, benchmark_prices, commodity_futures_prices


def calculate_returns(prices):
    """
    Create daily returns, monthly prices, and monthly returns.
    """
    daily_returns = prices.pct_change(fill_method=None)

    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change(fill_method=None)

    return daily_returns, monthly_prices, monthly_returns


def build_dataset(prices):
    """
    Build daily/monthly price and return datasets for each ticker universe.
    """
    dataset = {}

    for name, tickers in DATASETS.items():
        available_tickers = [ticker for ticker in tickers if ticker in prices.columns]
        universe_prices = prices[available_tickers]
        daily_returns, monthly_prices, monthly_returns = calculate_returns(universe_prices)

        dataset[name] = {
            "daily_prices": universe_prices,
            "daily_returns": daily_returns,
            "monthly_prices": monthly_prices,
            "monthly_returns": monthly_returns,
        }

    return dataset


def main():
    prices = download_price_data(ALL_TICKERS)

    report_download_summary(prices)

    resource_prices, benchmark_prices, commodity_futures_prices = split_prices_by_universe(
        prices
    )

    resource_daily_returns, resource_monthly_prices, resource_monthly_returns = (
        calculate_returns(resource_prices)
    )

    benchmark_daily_returns, benchmark_monthly_prices, benchmark_monthly_returns = (
        calculate_returns(benchmark_prices)
    )

    (
        commodity_futures_daily_returns,
        commodity_futures_monthly_prices,
        commodity_futures_monthly_returns,
    ) = calculate_returns(commodity_futures_prices)

    print("\nResource daily prices:")
    print(resource_prices.head())

    print("\nResource monthly returns:")
    print(resource_monthly_returns.head())

    print("\nBenchmark monthly returns:")
    print(benchmark_monthly_returns.head())

    print("\nCommodity futures monthly returns:")
    print(commodity_futures_monthly_returns.head())


if __name__ == "__main__":
    main()
