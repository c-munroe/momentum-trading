"""
CRSP point-in-time annual universe construction.
"""

import pandas as pd

from universe.dynamic import (
    DEFAULT_START_YEAR,
    LIQUIDITY_LOOKBACK_DAYS,
    MIN_AVG_DAILY_DOLLAR_VOLUME,
    MIN_LIQUIDITY_OBSERVATIONS,
    MIN_MARKET_CAP,
    MIN_UNIVERSE_SIZE,
    DynamicUniverseResult,
    get_reconstitution_date,
    rank_universe,
)
from universe.crsp.data import _metadata_quality_score
from universe.resource_classification import classify_resource_industry


CRSP_MIN_MARKET_CAP_THOUSANDS = MIN_MARKET_CAP // 1_000
ALLOWED_PRIMARY_EXCHANGES = {"N", "Q", "A"}
ALLOWED_SHARE_TYPES = {"NS", "AD", "SB"}


def _valid_security_mask(frame):
    return (
        frame["security_type"].eq("EQTY").fillna(False)
        & frame["security_subtype"].eq("COM").fillna(False)
        & frame["share_type"].isin(ALLOWED_SHARE_TYPES).fillna(False)
        & frame["primary_exch"].isin(ALLOWED_PRIMARY_EXCHANGES).fillna(False)
        & frame["trading_status_flg"].eq("A").fillna(False)
        & frame["security_active_flg"].eq("Y").fillna(False)
    )


def _choose_latest_name_rows(names):
    names = names.copy()
    names["_metadata_quality"] = _metadata_quality_score(
        names,
        allowed_share_types=ALLOWED_SHARE_TYPES,
    )
    names["_row_order"] = range(len(names))

    latest = (
        names.sort_values(
            ["permno", "sec_info_start_dt", "_metadata_quality", "_row_order"]
        )
        .groupby("permno", sort=False)
        .tail(1)
        .drop(columns=["_metadata_quality", "_row_order"])
    )

    return latest.set_index("permno")


def _latest_monthly_before(monthly, as_of_date, candidate_permnos):
    monthly = monthly[
        (monthly["date"] < as_of_date) & monthly["permno"].isin(candidate_permnos)
    ].copy()

    if monthly.empty:
        return pd.DataFrame(index=pd.Index([], name="permno"))

    columns = [
        "permno",
        "date",
        "month_end",
        "ticker",
        "primary_exch",
        "security_type",
        "security_subtype",
        "share_type",
        "trading_status_flg",
        "siccd",
        "mth_cap",
        "mth_prc",
        "mth_ret",
        "mth_vol",
        "shrout",
    ]
    latest = monthly.sort_values(["permno", "date"]).groupby("permno", sort=False).tail(
        1
    )
    latest = latest[columns].rename(
        columns={
            "date": "monthly_date",
            "month_end": "monthly_month_end",
            "ticker": "monthly_ticker",
            "primary_exch": "monthly_primary_exch",
            "security_type": "monthly_security_type",
            "security_subtype": "monthly_security_subtype",
            "share_type": "monthly_share_type",
            "trading_status_flg": "monthly_trading_status_flg",
            "siccd": "monthly_siccd",
        }
    )

    return latest.set_index("permno")


def _display_ticker(row):
    for column in ["trade_symbol", "ticker", "monthly_ticker"]:
        value = row.get(column)

        if pd.notna(value):
            return str(value)

    return None


def _classify_security_row(row):
    for sic_column, naics_column in [
        ("siccd", "naics"),
        ("hdr_siccd", None),
        ("monthly_siccd", None),
    ]:
        label = classify_resource_industry(
            siccd=row.get(sic_column),
            naics=row.get(naics_column) if naics_column else None,
        )

        if label is not None:
            return label

    return None


def get_crsp_security_info_as_of(names, monthly, as_of_date, candidate_permnos):
    as_of_date = pd.Timestamp(as_of_date)
    candidate_permnos = pd.Index(candidate_permnos, name="permno")

    effective_names = names[
        (names["permno"].isin(candidate_permnos))
        & (names["sec_info_start_dt"] < as_of_date)
        & (names["sec_info_end_dt"] >= as_of_date)
        & (names["security_beg_dt"] < as_of_date)
        & (names["security_end_dt"] >= as_of_date)
    ].copy()

    if effective_names.empty:
        return pd.DataFrame(index=candidate_permnos)

    latest_names = _choose_latest_name_rows(effective_names)
    latest_monthly = _latest_monthly_before(
        monthly=monthly,
        as_of_date=as_of_date,
        candidate_permnos=candidate_permnos,
    )
    info = latest_names.join(latest_monthly, how="left")
    info["valid_security"] = _valid_security_mask(info)
    info["display_ticker"] = info.apply(_display_ticker, axis=1)
    info["resource_subsector"] = info.apply(_classify_security_row, axis=1)

    return info.reindex(candidate_permnos)


def calculate_crsp_trailing_addv(
    daily,
    as_of_date,
    candidate_permnos=None,
    lookback_days=LIQUIDITY_LOOKBACK_DAYS,
):
    as_of_date = pd.Timestamp(as_of_date)
    eligible_daily = daily[daily["date"] < as_of_date].copy()

    if candidate_permnos is not None:
        eligible_daily = eligible_daily[eligible_daily["permno"].isin(candidate_permnos)]

    if eligible_daily.empty:
        return pd.DataFrame(
            columns=[
                "avg_daily_dollar_volume",
                "liquidity_observations",
                "last_liquidity_date",
            ]
        )

    clean_daily = eligible_daily.dropna(subset=["dly_prc", "dly_vol"]).copy()
    clean_daily = clean_daily[
        (clean_daily["dly_prc"].abs() > 0) & (clean_daily["dly_vol"] >= 0)
    ]
    clean_daily["dollar_volume"] = clean_daily["dly_prc"].abs() * clean_daily["dly_vol"]

    trailing = (
        clean_daily.sort_values(["permno", "date"])
        .groupby("permno", sort=False)
        .tail(lookback_days)
    )

    return trailing.groupby("permno").agg(
        avg_daily_dollar_volume=("dollar_volume", "mean"),
        liquidity_observations=("dollar_volume", "count"),
        last_liquidity_date=("date", "max"),
    )


def build_crsp_annual_eligibility_table(security_info, liquidity):
    table = security_info.join(liquidity, how="left")
    table["valid_security"] = table["valid_security"].fillna(False).astype(bool)
    table["passed_resource_classification"] = table["resource_subsector"].notna()
    table["market_cap_available"] = table["mth_cap"].notna()
    table["passed_market_cap"] = (
        table["market_cap_available"]
        & (table["mth_cap"] >= CRSP_MIN_MARKET_CAP_THOUSANDS)
    )
    table["liquidity_available"] = table["avg_daily_dollar_volume"].notna()
    table["liquidity_observations"] = table["liquidity_observations"].fillna(0).astype(
        "int64"
    )
    table["sufficient_liquidity_history"] = (
        table["liquidity_observations"] >= MIN_LIQUIDITY_OBSERVATIONS
    )
    table["valid_liquidity_candidate"] = (
        table["liquidity_available"] & table["sufficient_liquidity_history"]
    )
    table["passed_liquidity"] = (
        table["valid_liquidity_candidate"]
        & (table["avg_daily_dollar_volume"] >= MIN_AVG_DAILY_DOLLAR_VOLUME)
    )
    table["valid_universe_candidate"] = (
        table["valid_security"]
        & table["passed_resource_classification"]
        & table["passed_market_cap"]
        & table["valid_liquidity_candidate"]
    )
    table["eligible"] = table["valid_universe_candidate"] & table["passed_liquidity"]

    return table


def format_crsp_security_display(security_info, permno):
    if permno not in security_info.index:
        return str(permno)

    ticker = security_info.at[permno, "display_ticker"]

    if pd.isna(ticker):
        return str(permno)

    return f"{ticker} ({permno})"


def build_crsp_annual_universes(
    crsp_data,
    start_year=DEFAULT_START_YEAR,
    end_year=None,
    candidate_permnos=None,
):
    monthly = crsp_data.monthly_stock
    daily = crsp_data.daily_stock
    names = crsp_data.names
    last_crsp_month = monthly["month_end"].max()

    if end_year is None:
        end_year = last_crsp_month.year
    else:
        end_year = min(end_year, last_crsp_month.year)

    if candidate_permnos is None:
        candidate_permnos = sorted(set(daily["permno"]) & set(monthly["permno"]))
    else:
        candidate_permnos = sorted(set(candidate_permnos) & set(monthly["permno"]))

    annual_universes = {}
    display_annual_universes = {}
    diagnostic_rows = []

    for year in range(start_year, end_year + 1):
        reconstitution_date = get_reconstitution_date(year)
        security_info = get_crsp_security_info_as_of(
            names=names,
            monthly=monthly,
            as_of_date=reconstitution_date,
            candidate_permnos=candidate_permnos,
        )
        liquidity = calculate_crsp_trailing_addv(
            daily=daily,
            as_of_date=reconstitution_date,
            candidate_permnos=candidate_permnos,
        )
        eligibility_table = build_crsp_annual_eligibility_table(
            security_info=security_info,
            liquidity=liquidity,
        )
        selected_permnos = [
            int(permno)
            for permno in rank_universe(
                eligibility_table=eligibility_table,
                min_universe_size=MIN_UNIVERSE_SIZE,
            )
        ]

        annual_universes[year] = selected_permnos
        display_annual_universes[year] = [
            format_crsp_security_display(security_info, permno)
            for permno in selected_permnos
        ]

        selected = eligibility_table.index.isin(selected_permnos)
        selected_below_liquidity = selected & ~eligibility_table["passed_liquidity"]
        valid_resource = (
            eligibility_table["valid_security"]
            & eligibility_table["passed_resource_classification"]
        )
        market_cap_passes = valid_resource & eligibility_table["passed_market_cap"]

        diagnostic_rows.append(
            {
                "year": year,
                "candidate_names": len(candidate_permnos),
                "valid_securities": int(eligibility_table["valid_security"].sum()),
                "passed_resource_classification": int(valid_resource.sum()),
                "passed_market_cap": int(market_cap_passes.sum()),
                "sufficient_liquidity_history": int(
                    (
                        valid_resource
                        & eligibility_table["passed_market_cap"]
                        & eligibility_table["sufficient_liquidity_history"]
                    ).sum()
                ),
                "normal_liquidity_passes": int(eligibility_table["eligible"].sum()),
                "minimum_universe_additions": int(selected_below_liquidity.sum()),
                "minimum_universe_reached": len(selected_permnos) >= MIN_UNIVERSE_SIZE,
                "final_universe": len(selected_permnos),
                "last_market_cap_date": eligibility_table.loc[
                    market_cap_passes, "monthly_date"
                ].max(),
                "last_liquidity_date": eligibility_table.loc[
                    eligibility_table["valid_liquidity_candidate"],
                    "last_liquidity_date",
                ].max(),
                "min_universe_size": MIN_UNIVERSE_SIZE,
                "resource_classification_required": True,
                "market_cap_required": True,
                "market_cap_data_available": True,
                "subsector_data_available": True,
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows).set_index("year")
    result = DynamicUniverseResult(
        annual_universes=annual_universes,
        diagnostics=diagnostics,
        candidate_source="CRSP daily resource candidate pool",
        candidate_count=len(candidate_permnos),
        display_annual_universes=display_annual_universes,
        data_source="CRSP",
        data_start_date=monthly["date"].min(),
        data_end_date=monthly["date"].max(),
        duplicate_report=crsp_data.duplicate_report,
    )

    return result
