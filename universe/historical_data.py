"""
Point-in-time data helpers for universe construction.
"""

import pandas as pd
import yfinance as yf


YAHOO_RESOURCE_SECTORS = ("Energy", "Basic Materials")
YAHOO_LISTED_EXCHANGES = ("NMS", "NYQ", "ASE")
YAHOO_SCREENER_PAGE_SIZE = 250


def build_yahoo_resource_candidate_pool(
    sectors=YAHOO_RESOURCE_SECTORS,
    exchanges=YAHOO_LISTED_EXCHANGES,
    page_size=YAHOO_SCREENER_PAGE_SIZE,
):
    """
    Pull today's Yahoo-listed resource-sector stock candidates

    This is broader than NATURAL_RESOURCE_TICKERS, but it is still today's
    listed universe. However, this means delisted, acquired, bankrupt, and renamed
    historical names are still missing
    """
    candidates = []

    for sector in sectors:
        offset = 0

        while True:
            query = yf.EquityQuery(
                "and",
                [
                    yf.EquityQuery("eq", ["sector", sector]),
                    yf.EquityQuery("is-in", ["exchange", *exchanges]),
                ],
            )
            response = yf.screen(
                query,
                offset=offset,
                size=page_size,
                sortField="ticker",
                sortAsc=True,
            )
            quotes = response.get("quotes", [])

            for quote in quotes:
                symbol = quote.get("symbol")

                if symbol and quote.get("quoteType") == "EQUITY":
                    candidates.append(symbol)

            total = response.get("total", 0)
            offset += len(quotes)

            if not quotes or offset >= total:
                break

    return list(dict.fromkeys(candidates))


def download_price_volume_data(tickers, start_date, end_date):
    """
    Download unadjusted close prices and volume from Yahoo Finance

    Use raw close prices for the liquidity calculation since we want the price
    that was actually trading at the time. Adjusted prices are still used
    elsewhere for returns and momentum
    """
    raw_data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw_data.empty:
        raise ValueError("Yahoo Finance returned no data for the requested tickers.")

    unadjusted_close = _extract_yfinance_field(raw_data, "Close", tickers)
    volumes = _extract_yfinance_field(raw_data, "Volume", tickers)

    unadjusted_close = unadjusted_close.dropna(axis=1, how="all")
    volumes = volumes.reindex(columns=unadjusted_close.columns).dropna(axis=1, how="all")

    return unadjusted_close, volumes


def _extract_yfinance_field(raw_data, field, tickers):
    """
    Normalize yfinance output to a DataFrame for one or many tickers.
    """
    field_data = raw_data[field].copy()

    if isinstance(field_data, pd.Series):
        field_data = field_data.to_frame(name=list(tickers)[0])

    return field_data


def download_historical_shares_outstanding(tickers, start_date, end_date):
    """
    Download sparse historical shares outstanding from Yahoo Finance

    No current shares value is filled backward. If Yahoo has no historical
    shares value before a reconstitution date, that ticker has no market-cap
    observation for that annual screen
    """
    shares_by_ticker = {}

    for ticker in tickers:
        try:
            shares = yf.Ticker(ticker).get_shares_full(
                start=start_date,
                end=end_date,
            )
        except Exception:
            continue

        if shares is None or shares.empty:
            continue

        shares = shares.dropna()
        shares = shares[shares > 0]

        if shares.empty:
            continue

        shares.index = pd.to_datetime(shares.index).tz_localize(None).normalize()
        shares = shares[~shares.index.duplicated(keep="last")].sort_index()
        shares_by_ticker[ticker] = shares

    if not shares_by_ticker:
        return pd.DataFrame()

    return pd.DataFrame(shares_by_ticker).sort_index()


def calculate_historical_market_caps(prices, shares_outstanding):
    """
    Calculate historical market cap from price and historical shares

    Shares are forward-filled only after Yahoo reports a historical shares
    observation. Missing older shares stay missing and therefore fail the
    market-cap filter
    """
    if prices.empty or shares_outstanding is None or shares_outstanding.empty:
        return pd.DataFrame()

    common_columns = [
        ticker for ticker in prices.columns if ticker in shares_outstanding.columns
    ]
    prices = prices[common_columns].sort_index()
    shares_outstanding = shares_outstanding[common_columns].sort_index()

    if prices.empty or shares_outstanding.empty:
        return pd.DataFrame()

    shares_daily = shares_outstanding.reindex(prices.index, method="ffill")

    return prices * shares_daily


def get_last_available_date_before(index, date):
    """
    Return the last index value strictly before date.

    Strictly before matters: if reconstitution is labeled January 1, the screen
    must not use any data from January 1 or later
    """
    eligible_dates = index[index < pd.Timestamp(date)]

    if eligible_dates.empty:
        return None

    return eligible_dates.max()


def get_point_in_time_market_cap(market_cap_data, as_of_date):
    """
    Return market caps known before as_of_date.

    Supported input shapes:
    1. Long table with columns: date, ticker, market_cap
    2. Wide table indexed by date with tickers as columns and market caps as values

    For sparse data, each ticker uses its own latest valid observation strictly
    before as_of_date. One ticker's missing value on the final available date
    should not discard another ticker's valid market-cap observation from a few
    days earlier.

    If no table is supplied, this function returns an empty Series. It does not
    query current market cap and does not forward-fill today's values backward.
    """
    if market_cap_data is None or market_cap_data.empty:
        return pd.Series(dtype=float)

    if {"date", "ticker", "market_cap"}.issubset(market_cap_data.columns):
        data = market_cap_data[["date", "ticker", "market_cap"]].copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] < pd.Timestamp(as_of_date)]

        if data.empty:
            return pd.Series(dtype=float)

        data = data.dropna(subset=["market_cap"])
        data = data.sort_values(["ticker", "date"])

        return data.groupby("ticker", sort=False).tail(1).set_index("ticker")[
            "market_cap"
        ]

    data = market_cap_data.copy().sort_index()
    data = data[data.index < pd.Timestamp(as_of_date)]

    if data.empty:
        return pd.Series(dtype=float)

    # forward-fill only within the historical slice that ends before as_of_date
    # the final row then represents the latest known value for each ticker
    return data.ffill().iloc[-1].dropna()


def get_point_in_time_subsector(subsector_data, as_of_date):
    """
    Return subsector labels known before as_of_date.

    Supported input shapes:
    1. Long table with columns: date, ticker, subsector
    2. Wide table indexed by date with tickers as columns and subsectors as values

    This function only uses rows strictly before as_of_date. It does not
    backward-fill modern classifications into older reconstitution years.
    Missing labels are allowed here. They only fail eligibility when
    require_resource_classification=True is set later in filter_eligible_stocks.
    """
    if subsector_data is None or subsector_data.empty:
        return pd.Series(dtype=object)

    if {"date", "ticker", "subsector"}.issubset(subsector_data.columns):
        data = subsector_data[["date", "ticker", "subsector"]].copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["date"] < pd.Timestamp(as_of_date)]

        if data.empty:
            return pd.Series(dtype=object)

        data = data.sort_values(["ticker", "date"])

        return data.groupby("ticker", sort=False).tail(1).set_index("ticker")[
            "subsector"
        ]

    data = subsector_data.copy().sort_index()
    data = data[data.index < pd.Timestamp(as_of_date)]

    if data.empty:
        return pd.Series(dtype=object)

    # forward-fill only inside the historical slice
    # this keeps each ticker's own latest label without using future labels
    return data.ffill().iloc[-1].dropna()
