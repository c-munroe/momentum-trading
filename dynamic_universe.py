"""
Annual point-in-time universe construction for the momentum backtest.

The workflow is:
1. Start from a candidate ticker list.
2. On each annual reconstitution date, use only data available before that date.
3. Optionally require a point-in-time resource subsector classification.
4. Filter by trailing average daily dollar volume using unadjusted close * volume.
5. Filter by point-in-time market capitalization if such data is supplied.
6. Rank eligible names and keep up to MAX_UNIVERSE_SIZE.

Important limitation
--------------------
The project currently uses Yahoo Finance -- that is enough to calculate a
historical liquidity screen from contemporaneous unadjusted close * reported
volume, but it is not enough to build defensible historical market-cap screens.
This file therefore does not use today's market cap as a substitute for
historical market cap.

The default example still starts from NATURAL_RESOURCE_TICKERS. That is useful
for development, but it still has hand-selection and survivorship bias. A stock
that traded historically but is missing from today's supplied candidate list
cannot be selected by this engine, even if it would have passed every screen.

There are two separate data problems this module does not pretend to solve:
- Candidate survivorship bias: the supplied security master must include stocks
  that later delisted, failed, merged, or were acquired.
- Classification bias: historical industry labels must be point-in-time. Using
  today's resource classification backward would introduce look-ahead bias.

For development, allow_missing_market_cap=True lets you build liquidity-only
universes while explicitly marking market cap as not checked. That mode is
useful for plumbing and inspection, but it is not institutionally defensible.
"""

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from equities_list import NATURAL_RESOURCE_TICKERS


MIN_MARKET_CAP = 1_000_000_000
MIN_AVG_DAILY_DOLLAR_VOLUME = 5_000_000
MAX_UNIVERSE_SIZE = 100
LIQUIDITY_LOOKBACK_DAYS = 60
MIN_LIQUIDITY_OBSERVATIONS = 40
DEFAULT_START_YEAR = 2001
DEFAULT_END_YEAR = pd.Timestamp.today().year
DEFAULT_RESOURCE_SUBSECTORS = {
    "integrated_energy",
    "oil_gas_producer",
    "natural_gas_producer",
    "midstream",
    "lng",
    "refining",
    "oilfield_services",
    "gold_mining",
    "silver_mining",
    "precious_metals_royalty",
    "copper_base_metals",
    "diversified_mining",
    "iron_ore",
    "aluminum",
    "steel",
    "rare_earths",
    "chemicals",
    "fertilizers",
    "agriculture_inputs",
    "forestry",
    "construction_materials",
    "uranium",
    "coal",
    "lithium_battery_materials",
}


@dataclass
class DynamicUniverseResult:
    """
    Container for annual universe output and diagnostics.

    annual_universes maps each reconstitution year to the final selected tickers.
    details has one row per year/ticker with pass/fail diagnostics.
    diagnostics has one summary row per year.
    """

    annual_universes: dict[int, list[str]]
    details: pd.DataFrame
    diagnostics: pd.DataFrame


def download_price_volume_data(tickers, start_date, end_date):
    """
    Download unadjusted close and volume data from Yahoo Finance.

    Liquidity should be based on the trading price investors saw at the time,
    so this function explicitly uses auto_adjust=False and extracts the raw
    historical Close field. The rest of the project can continue using adjusted
    closes for return calculations; this standalone file only uses unadjusted
    close for average daily dollar volume.
    """
    raw_data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=True,
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


def get_reconstitution_date(year):
    """
    Use January 1 as the labeled annual reconstitution date.

    Markets are usually closed on January 1. The actual eligibility data is
    therefore taken from the last trading data available before this date.
    """
    return pd.Timestamp(year=year, month=1, day=1)


def get_last_available_date_before(index, date):
    """
    Return the last index value strictly before date.

    Strictly before matters: if reconstitution is labeled January 1, the screen
    must not use any data from January 1 or later.
    """
    eligible_dates = index[index < pd.Timestamp(date)]

    if eligible_dates.empty:
        return None

    return eligible_dates.max()


def calculate_trailing_dollar_volume(
    prices,
    volumes,
    as_of_date,
    lookback_days=LIQUIDITY_LOOKBACK_DAYS,
):
    """
    Calculate trailing average daily dollar volume before as_of_date.

    dollar_volume = unadjusted_close * reported_volume

    The window ends on the last available trading day strictly before
    as_of_date. This avoids accidentally using same-day or future information.

    Returns
    -------
    tuple
        avg_dollar_volume, valid_observation_counts, last_data_date
    """
    prices, volumes = prices.align(volumes, join="inner", axis=0)
    prices, volumes = prices.align(volumes, join="inner", axis=1)

    _validate_liquidity_inputs(prices=prices, volumes=volumes)

    last_data_date = get_last_available_date_before(prices.index, as_of_date)

    if last_data_date is None:
        return pd.Series(dtype=float), pd.Series(dtype="int64"), None

    dollar_volume = prices * volumes
    trailing_window = dollar_volume.loc[:last_data_date].tail(lookback_days)
    valid_observations = trailing_window.count()
    average_dollar_volume = trailing_window.mean(skipna=True)

    # Keep the average visible for diagnostics, but eligibility will separately
    # require at least min_observations valid data points.
    average_dollar_volume = average_dollar_volume.where(
        valid_observations >= 1,
        pd.NA,
    )

    return average_dollar_volume, valid_observations, last_data_date


def _validate_liquidity_inputs(prices, volumes):
    """
    Basic sanity checks for liquidity data.

    Missing data is allowed and remains missing. Negative volume is not a
    defensible trading observation, so it is rejected instead of being clipped
    or silently converted to zero.
    """
    if prices.empty or volumes.empty:
        return

    if (volumes.dropna() < 0).any().any():
        raise ValueError("Volume data contains negative values.")

    if (prices.dropna() <= 0).any().any():
        raise ValueError("Price data contains non-positive values.")


def get_point_in_time_market_cap(market_cap_data, as_of_date):
    """
    Return market caps known before as_of_date.

    Supported input shapes:
    1. Long table with columns: date, ticker, market_cap.
    2. Wide table indexed by date with tickers as columns and market caps as values.

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

    # Forward-fill only within the historical slice that ends before as_of_date.
    # The final row then represents the latest known value for each ticker.
    return data.ffill().iloc[-1].dropna()


def get_point_in_time_subsector(subsector_data, as_of_date):
    """
    Return subsector labels known before as_of_date.

    Supported input shapes:
    1. Long table with columns: date, ticker, subsector.
    2. Wide table indexed by date with tickers as columns and subsectors as values.

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

    # Forward-fill only inside the historical slice. This lets sparse wide
    # tables keep each ticker's own latest label without using future labels.
    return data.ffill().iloc[-1].dropna()


def filter_eligible_stocks(
    candidate_tickers,
    trailing_dollar_volume,
    market_caps,
    trailing_liquidity_observations=None,
    subsectors=None,
    allowed_resource_subsectors=DEFAULT_RESOURCE_SUBSECTORS,
    require_resource_classification=False,
    min_market_cap=MIN_MARKET_CAP,
    min_avg_daily_dollar_volume=MIN_AVG_DAILY_DOLLAR_VOLUME,
    min_liquidity_observations=MIN_LIQUIDITY_OBSERVATIONS,
    allow_missing_market_cap=False,
):
    """
    Build one diagnostic row per ticker for an annual screen.

    allow_missing_market_cap exists only for development. When True, the market
    cap test is marked as passed even if no point-in-time market cap is present.
    The row still records market_cap_available=False so the limitation is visible.

    require_resource_classification controls whether point-in-time subsector
    metadata is an eligibility gate. Keep it False when candidate_tickers is
    already a hand-curated resource list. Set it True only when passing a broad
    candidate universe and a real point-in-time classification table.
    """
    rows = []
    allowed_resource_subsectors = set(allowed_resource_subsectors)
    trailing_liquidity_observations = (
        pd.Series(dtype="int64")
        if trailing_liquidity_observations is None
        else trailing_liquidity_observations
    )
    subsectors = pd.Series(dtype=object) if subsectors is None else subsectors

    for ticker in candidate_tickers:
        avg_dollar_volume = trailing_dollar_volume.get(ticker, pd.NA)
        liquidity_observations = trailing_liquidity_observations.get(ticker, 0)
        market_cap = market_caps.get(ticker, pd.NA)
        subsector = subsectors.get(ticker, pd.NA)

        has_liquidity = pd.notna(avg_dollar_volume)
        has_sufficient_liquidity_history = (
            pd.notna(liquidity_observations)
            and liquidity_observations >= min_liquidity_observations
        )
        has_market_cap = pd.notna(market_cap)
        has_subsector = pd.notna(subsector)
        is_resource_subsector = (
            has_subsector and subsector in allowed_resource_subsectors
        )

        passed_liquidity = (
            has_liquidity
            and has_sufficient_liquidity_history
            and avg_dollar_volume >= min_avg_daily_dollar_volume
        )
        passed_market_cap = (
            market_cap >= min_market_cap
            if has_market_cap
            else allow_missing_market_cap
        )
        passed_resource_classification = (
            is_resource_subsector if require_resource_classification else True
        )
        eligible = bool(
            passed_resource_classification and passed_liquidity and passed_market_cap
        )

        exclusion_reasons = []

        if require_resource_classification:
            if not has_subsector:
                exclusion_reasons.append("missing_subsector_classification")
            elif not is_resource_subsector:
                exclusion_reasons.append("non_resource_subsector")

        if not has_liquidity:
            exclusion_reasons.append("missing_liquidity")
        elif not has_sufficient_liquidity_history:
            exclusion_reasons.append("insufficient_liquidity_history")
        elif not passed_liquidity:
            exclusion_reasons.append("below_liquidity_threshold")

        if not has_market_cap:
            if allow_missing_market_cap:
                exclusion_reasons.append("market_cap_not_checked")
            else:
                exclusion_reasons.append("missing_market_cap")
        elif not passed_market_cap:
            exclusion_reasons.append("below_market_cap_threshold")

        if eligible:
            reason = "eligible"

            if not has_market_cap and allow_missing_market_cap:
                reason = "eligible;market_cap_not_checked"
        else:
            reason = ";".join(exclusion_reasons)

        rows.append(
            {
                "ticker": ticker,
                "avg_daily_dollar_volume": avg_dollar_volume,
                "liquidity_observations": liquidity_observations,
                "market_cap": market_cap,
                "subsector": subsector,
                "liquidity_available": has_liquidity,
                "sufficient_liquidity_history": bool(
                    has_sufficient_liquidity_history
                ),
                "market_cap_available": has_market_cap,
                "subsector_available": has_subsector,
                "resource_classification_required": bool(
                    require_resource_classification
                ),
                "passed_resource_classification": bool(
                    passed_resource_classification
                ),
                "passed_liquidity": bool(passed_liquidity),
                "passed_market_cap": bool(passed_market_cap),
                "eligible": eligible,
                "exclusion_reason": reason,
            }
        )

    return pd.DataFrame(rows).set_index("ticker")


def rank_universe(
    eligibility_table,
    max_names=MAX_UNIVERSE_SIZE,
    ranking_column="avg_daily_dollar_volume",
):
    """
    Rank eligible names and keep up to max_names.

    Liquidity is the default ranking column because it is genuinely historical
    in this first pass. If point-in-time market cap is supplied, this can be
    changed to "market_cap" or a composite rank later.
    """
    eligible = eligibility_table[eligibility_table["eligible"]].copy()

    if eligible.empty:
        return []

    sort_columns = [ranking_column]

    if ranking_column != "avg_daily_dollar_volume":
        sort_columns.append("avg_daily_dollar_volume")

    eligible = eligible.sort_values(
        by=sort_columns,
        ascending=False,
        na_position="last",
    )

    return eligible.head(max_names).index.tolist()


def build_annual_universes(
    candidate_tickers,
    prices,
    volumes,
    market_cap_data=None,
    subsector_data=None,
    allowed_resource_subsectors=DEFAULT_RESOURCE_SUBSECTORS,
    require_resource_classification=False,
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
    min_market_cap=MIN_MARKET_CAP,
    min_avg_daily_dollar_volume=MIN_AVG_DAILY_DOLLAR_VOLUME,
    max_names=MAX_UNIVERSE_SIZE,
    liquidity_lookback_days=LIQUIDITY_LOOKBACK_DAYS,
    min_liquidity_observations=MIN_LIQUIDITY_OBSERVATIONS,
    allow_missing_market_cap=False,
):
    """
    Build annually reconstituted universes and diagnostics.

    Reconstitution convention:
    - The universe for year Y is labeled January 1 of year Y.
    - Eligibility uses prices, volumes, and market cap data strictly before
      January 1 of year Y.
    - The selected universe is held constant until the next annual screen.

    If require_resource_classification=True, subsector_data becomes an
    eligibility filter. A ticker must have a latest valid subsector strictly
    before the reconstitution date, and that label must be in
    allowed_resource_subsectors.

    If require_resource_classification=False, subsector_data is diagnostic only.
    That is the right mode when candidate_tickers is already a manually
    resource-restricted list such as NATURAL_RESOURCE_TICKERS.

    Passing NATURAL_RESOURCE_TICKERS as candidate_tickers is useful for an
    initial mechanics test, but it does not eliminate hand-selection or
    survivorship bias. The engine can only select names present in the supplied
    candidate list.
    """
    candidate_tickers = list(dict.fromkeys(candidate_tickers))
    prices = prices.reindex(columns=[c for c in candidate_tickers if c in prices])
    volumes = volumes.reindex(columns=[c for c in candidate_tickers if c in volumes])

    annual_universes = {}
    detail_frames = []
    diagnostic_rows = []

    for year in range(start_year, end_year + 1):
        reconstitution_date = get_reconstitution_date(year)
        (
            trailing_dollar_volume,
            liquidity_observations,
            last_liquidity_date,
        ) = calculate_trailing_dollar_volume(
            prices=prices,
            volumes=volumes,
            as_of_date=reconstitution_date,
            lookback_days=liquidity_lookback_days,
        )
        market_caps = get_point_in_time_market_cap(
            market_cap_data=market_cap_data,
            as_of_date=reconstitution_date,
        )
        subsectors = get_point_in_time_subsector(
            subsector_data=subsector_data,
            as_of_date=reconstitution_date,
        )
        eligibility_table = filter_eligible_stocks(
            candidate_tickers=candidate_tickers,
            trailing_dollar_volume=trailing_dollar_volume,
            trailing_liquidity_observations=liquidity_observations,
            market_caps=market_caps,
            subsectors=subsectors,
            allowed_resource_subsectors=allowed_resource_subsectors,
            require_resource_classification=require_resource_classification,
            min_market_cap=min_market_cap,
            min_avg_daily_dollar_volume=min_avg_daily_dollar_volume,
            min_liquidity_observations=min_liquidity_observations,
            allow_missing_market_cap=allow_missing_market_cap,
        )
        selected_tickers = rank_universe(
            eligibility_table=eligibility_table,
            max_names=max_names,
        )

        annual_universes[year] = selected_tickers

        detail = eligibility_table.copy()
        detail.insert(0, "year", year)
        detail.insert(1, "reconstitution_date", reconstitution_date)
        detail.insert(2, "last_liquidity_date", last_liquidity_date)
        detail["selected"] = detail.index.isin(selected_tickers)
        detail_frames.append(detail.reset_index())

        market_cap_available = eligibility_table["market_cap_available"]
        real_market_cap_passes = (
            market_cap_available & eligibility_table["passed_market_cap"]
        )
        market_cap_not_checked = (
            ~market_cap_available & eligibility_table["passed_market_cap"]
        )
        subsector_available = eligibility_table["subsector_available"]
        non_resource_names = (
            subsector_available
            & ~eligibility_table["subsector"].isin(allowed_resource_subsectors)
        )

        diagnostic_rows.append(
            {
                "year": year,
                "candidate_names": len(candidate_tickers),
                "subsector_available": int(subsector_available.sum()),
                "subsector_missing": int((~subsector_available).sum()),
                "passed_resource_classification": int(
                    eligibility_table["passed_resource_classification"].sum()
                ),
                "non_resource_names": int(non_resource_names.sum()),
                "passed_liquidity": int(eligibility_table["passed_liquidity"].sum()),
                "insufficient_liquidity_history": int(
                    (
                        eligibility_table["liquidity_available"]
                        & ~eligibility_table["sufficient_liquidity_history"]
                    ).sum()
                ),
                "market_cap_available": int(market_cap_available.sum()),
                "passed_market_cap": int(real_market_cap_passes.sum()),
                "market_cap_not_checked": int(market_cap_not_checked.sum()),
                "eligible": int(eligibility_table["eligible"].sum()),
                "final_universe": len(selected_tickers),
                "last_liquidity_date": last_liquidity_date,
                "resource_classification_required": require_resource_classification,
                "market_cap_required": not allow_missing_market_cap,
                "market_cap_data_available": market_cap_data is not None,
                "subsector_data_available": subsector_data is not None,
            }
        )

    details = pd.concat(detail_frames, ignore_index=True)
    details = details.set_index(["year", "ticker"]).sort_index()
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("year")

    return DynamicUniverseResult(
        annual_universes=annual_universes,
        details=details,
        diagnostics=diagnostics,
    )


def get_universe_for_date(date, annual_universes):
    """
    Return the applicable annual universe for any date.

    If a DynamicUniverseResult is passed, its annual_universes dict is used.
    Dates before the first available reconstitution return an empty list.
    """
    if isinstance(annual_universes, DynamicUniverseResult):
        annual_universes = annual_universes.annual_universes

    date = pd.Timestamp(date)
    available_years = sorted(year for year in annual_universes if year <= date.year)

    if not available_years:
        return []

    return annual_universes[available_years[-1]]


def get_exclusion_reason(result, year, ticker):
    """
    Inspect why one ticker was excluded or selected in a given year.
    """
    if not isinstance(result, DynamicUniverseResult):
        raise TypeError("result must be a DynamicUniverseResult.")

    return result.details.loc[(year, ticker)]


def print_annual_diagnostics(diagnostics):
    """
    Print compact annual counts for quick inspection.
    """
    for year, row in diagnostics.iterrows():
        print(f"\n{year}:")
        print(f"Candidate names: {row['candidate_names']}")
        print(f"Passed liquidity: {row['passed_liquidity']}")

        if row["insufficient_liquidity_history"] > 0:
            print(
                "Insufficient liquidity history: "
                f"{row['insufficient_liquidity_history']}"
            )

        if row["market_cap_not_checked"] > 0:
            print(
                "Passed market cap: "
                f"{row['passed_market_cap']} checked, "
                f"{row['market_cap_not_checked']} not checked"
            )
        else:
            print(f"Passed market cap: {row['passed_market_cap']}")

        if row.get("subsector_data_available", False):
            print(
                "Subsector available: "
                f"{row['subsector_available']} available, "
                f"{row['subsector_missing']} missing"
            )

        if row.get("resource_classification_required", False):
            print(
                "Passed resource classification: "
                f"{row['passed_resource_classification']}"
            )

            if row["non_resource_names"] > 0:
                print(f"Non-resource names: {row['non_resource_names']}")

        print(f"Final universe: {row['final_universe']}")


def _build_example_subsector_data():
    """
    Tiny illustrative point-in-time subsector table for the example only.

    This is not a full classification history. It simply demonstrates the
    accepted long-table shape and how labels appear in result.details.
    """
    rows = []

    for date in ["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31"]:
        rows.extend(
            [
                {
                    "date": date,
                    "ticker": "XOM",
                    "subsector": "integrated_energy",
                },
                {
                    "date": date,
                    "ticker": "NEM",
                    "subsector": "gold_mining",
                },
                {
                    "date": date,
                    "ticker": "FCX",
                    "subsector": "copper_base_metals",
                },
            ]
        )

    return pd.DataFrame(rows)


def _example():
    """
    Small manual example for local inspection.

    This example uses the existing static resource list as the candidate pool
    and liquidity-only mode because the project does not yet have historical
    point-in-time market cap data.

    This validates the mechanics only. It is not a survivorship-bias-free
    historical backtest.
    """
    candidate_tickers = NATURAL_RESOURCE_TICKERS
    start_year = 2020
    end_year = 2023
    start_date = f"{start_year - 1}-01-01"
    end_date = f"{end_year + 1}-01-10"

    try:
        prices, volumes = download_price_volume_data(
            tickers=candidate_tickers,
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as exc:
        print(f"Skipping NATURAL_RESOURCE_TICKERS example: {exc}")
        _toy_broad_universe_example()
        return

    result = build_annual_universes(
        candidate_tickers=candidate_tickers,
        prices=prices,
        volumes=volumes,
        market_cap_data=None,
        subsector_data=_build_example_subsector_data(),
        require_resource_classification=False,
        start_year=start_year,
        end_year=end_year,
        allow_missing_market_cap=True,
    )

    print_annual_diagnostics(result.diagnostics)
    print("\nUniverse for 2021-06-30:")
    print(get_universe_for_date("2021-06-30", result)[:20])
    print("\nExample ticker diagnostics:")
    print(get_exclusion_reason(result, 2021, candidate_tickers[0]))

    _toy_broad_universe_example()


def _toy_broad_universe_example():
    """
    Tiny broad-universe mechanics example using illustrative local data only.

    This is not historical research data. It exists to show how the optional
    resource-classification gate behaves when the candidate pool contains both
    resource and non-resource companies.
    """
    candidate_tickers = ["XOM", "NEM", "AAPL", "JPM", "MISSING"]
    dates = pd.bdate_range("2019-09-01", "2020-12-31")

    prices = pd.DataFrame(
        {
            "XOM": 60.0,
            "NEM": 40.0,
            "AAPL": 100.0,
            "JPM": 110.0,
            "MISSING": 50.0,
        },
        index=dates,
    )
    volumes = pd.DataFrame(
        {
            "XOM": 500_000,
            "NEM": 500_000,
            "AAPL": 500_000,
            "JPM": 500_000,
            "MISSING": 500_000,
        },
        index=dates,
    )
    market_cap_data = pd.DataFrame(
        [
            {"date": "2019-12-31", "ticker": ticker, "market_cap": 10_000_000_000}
            for ticker in candidate_tickers
        ]
    )
    subsector_data = pd.DataFrame(
        [
            {"date": "2019-12-31", "ticker": "XOM", "subsector": "integrated_energy"},
            {"date": "2019-12-31", "ticker": "NEM", "subsector": "gold_mining"},
            {"date": "2019-12-31", "ticker": "AAPL", "subsector": "technology"},
            {"date": "2019-12-31", "ticker": "JPM", "subsector": "financials"},
            # NEM's 2021 universe will see this later point-in-time update.
            {"date": "2020-06-01", "ticker": "NEM", "subsector": "diversified_mining"},
        ]
    )

    result = build_annual_universes(
        candidate_tickers=candidate_tickers,
        prices=prices,
        volumes=volumes,
        market_cap_data=market_cap_data,
        subsector_data=subsector_data,
        require_resource_classification=True,
        start_year=2020,
        end_year=2021,
        max_names=100,
    )

    print("\nToy broad-universe classification example:")
    print_annual_diagnostics(result.diagnostics)
    print("\nSelected in 2020:")
    print(result.annual_universes[2020])
    print("\nAAPL exclusion:")
    print(get_exclusion_reason(result, 2020, "AAPL"))
    print("\nMISSING exclusion:")
    print(get_exclusion_reason(result, 2020, "MISSING"))
    print("\nNEM 2021 point-in-time subsector:")
    print(get_exclusion_reason(result, 2021, "NEM")[["subsector", "eligible"]])


if __name__ == "__main__":
    _example()
