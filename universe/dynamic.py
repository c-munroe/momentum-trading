"""
Build the annual point-in-time universe for the momentum backtest

The process is:
1. Start with a candidate ticker list
2. Rebuild the universe each year using only data available at the time
3. Optionally filter by resource subsector
4. Check trailing average daily dollar volume
5. Filter by historical market cap when that data is available
6. Rank valid names by liquidity
7. Keep all names above the normal liquidity threshold
8. If needed, add below-threshold valid names until MIN_UNIVERSE_SIZE is reached

Yahoo Finance gives us historical price, volume, and sparse historical shares
outstanding. Market cap is calculated as historical price times historical
shares outstanding. We do not use today's market cap or today's shares as a
substitute for missing historical values

The current dynamic backtest starts from today's Yahoo screener results, so
survivorship bias is still present. Stocks that are missing from today's Yahoo
candidate list cannot appear in earlier years

Historical subsector classifications also need to be point-in-time to avoid
look-ahead bias
"""

from dataclasses import dataclass

import pandas as pd

from universe.historical_data import (
    get_last_available_date_before,
    get_point_in_time_market_cap,
    get_point_in_time_subsector,
)


MIN_MARKET_CAP = 1_000_000_000
MIN_AVG_DAILY_DOLLAR_VOLUME = 5_000_000
MIN_UNIVERSE_SIZE = 60
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
    diagnostics has one summary row per year.
    """

    annual_universes: dict[int, list[str]]
    diagnostics: pd.DataFrame
    candidate_source: str = ""
    candidate_count: int = 0
    shares_coverage_count: int = 0


def get_reconstitution_date(year):
    """
    Use January 1 as the labeled annual reconstitution date.

    Markets are usually closed on January 1. The actual eligibility data is
    therefore taken from the last trading data available before this date
    """
    return pd.Timestamp(year=year, month=1, day=1)


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

    # keep the average visible for diagnostics, but eligibility separately
    # requires at least min_observations valid data points
    average_dollar_volume = average_dollar_volume.where(
        valid_observations >= 1,
        pd.NA,
    )

    return average_dollar_volume, valid_observations, last_data_date


def _validate_liquidity_inputs(prices, volumes):
    """
    Basic sanity checks for liquidity data

    Missing data is allowed and remains missing. Negative volume is not a
    valid trading observation, so it is rejected (instead of being clipped
    or silently converted to zero)
    """
    if prices.empty or volumes.empty:
        return

    if (volumes.dropna() < 0).any().any():
        raise ValueError("Volume data contains negative values.")

    if (prices.dropna() <= 0).any().any():
        raise ValueError("Price data contains non-positive values.")


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
        valid_liquidity_candidate = (
            has_liquidity and has_sufficient_liquidity_history
        )
        valid_universe_candidate = bool(
            valid_liquidity_candidate
            and passed_resource_classification
            and passed_market_cap
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
                "valid_liquidity_candidate": bool(valid_liquidity_candidate),
                "valid_universe_candidate": bool(valid_universe_candidate),
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
    min_universe_size=MIN_UNIVERSE_SIZE,
    ranking_column="avg_daily_dollar_volume",
):
    """
    Rank valid names and apply the minimum-universe fallback

    All names that pass the normal liquidity threshold are included. If fewer
    than min_universe_size pass, the most liquid remaining valid names are added
    below the normal liquidity threshold. Missing or insufficient liquidity data,
    failed market-cap checks, and failed resource-classification checks are
    never bypassed
    """
    valid = eligibility_table[eligibility_table["valid_universe_candidate"]].copy()

    if valid.empty:
        return []

    sort_columns = [ranking_column]

    if ranking_column != "avg_daily_dollar_volume":
        sort_columns.append("avg_daily_dollar_volume")

    valid = valid.sort_values(
        by=sort_columns,
        ascending=False,
        na_position="last",
    )

    normal_liquidity_passes = valid[valid["passed_liquidity"]]

    if len(normal_liquidity_passes) >= min_universe_size:
        return normal_liquidity_passes.index.tolist()

    additions_needed = min_universe_size - len(normal_liquidity_passes)
    fallback_candidates = valid[~valid["passed_liquidity"]]
    fallback_additions = fallback_candidates.head(additions_needed)

    return normal_liquidity_passes.index.tolist() + fallback_additions.index.tolist()


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
    min_universe_size=MIN_UNIVERSE_SIZE,
    liquidity_lookback_days=LIQUIDITY_LOOKBACK_DAYS,
    min_liquidity_observations=MIN_LIQUIDITY_OBSERVATIONS,
    allow_missing_market_cap=False,
):
    """
    Build the annual universe and its diagnostics

    For each year:
    - label the universe as January 1 of that year
    - only use price, volume, market cap, and classification data from before that date
    - keep that universe in place until the next annual rebalance

    If require_resource_classification=True, a stock must have a valid historical
    subsector and that subsector must be included in allowed_resource_subsectors

    If require_resource_classification=False, subsector data is only used for
    diagnostics. This is the current setup when candidate discovery already
    uses Yahoo's current sector screener

    Because the current candidate list comes from today's Yahoo screener,
    survivorship bias is still present. The model can only choose from stocks
    included in today's candidate list
    """
    candidate_tickers = list(dict.fromkeys(candidate_tickers))
    prices = prices.reindex(columns=[c for c in candidate_tickers if c in prices])
    volumes = volumes.reindex(columns=[c for c in candidate_tickers if c in volumes])

    annual_universes = {}
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
            min_universe_size=min_universe_size,
        )

        annual_universes[year] = selected_tickers

        detail = eligibility_table.copy()
        detail["selected"] = detail.index.isin(selected_tickers)
        detail["selected_below_liquidity_threshold"] = (
            detail["selected"] & ~detail["passed_liquidity"]
        )

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
        valid_universe_candidates = eligibility_table["valid_universe_candidate"]
        normal_liquidity_passes = eligibility_table["eligible"]
        selected_below_liquidity_threshold = (
            detail["selected_below_liquidity_threshold"]
        )

        diagnostic_rows.append(
            {
                "year": year,
                "candidate_names": len(candidate_tickers),
                "valid_universe_candidates": int(valid_universe_candidates.sum()),
                "normal_liquidity_passes": int(normal_liquidity_passes.sum()),
                "minimum_universe_additions": int(
                    selected_below_liquidity_threshold.sum()
                ),
                "minimum_universe_reached": len(selected_tickers)
                >= min_universe_size,
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
                "min_universe_size": min_universe_size,
                "resource_classification_required": require_resource_classification,
                "market_cap_required": not allow_missing_market_cap,
                "market_cap_data_available": market_cap_data is not None,
                "subsector_data_available": subsector_data is not None,
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows).set_index("year")

    return DynamicUniverseResult(
        annual_universes=annual_universes,
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
