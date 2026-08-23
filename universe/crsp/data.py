"""
CRSP data loading, cleaning, validation, and summaries.
"""

from dataclasses import dataclass

import pandas as pd

from backtest.config import CRSP_DATA_DIR


CRSP_MONTHLY_FILE = "monthly_stock.csv.gz"
CRSP_NAMES_FILE = "names.csv.gz"
CRSP_DELISTINGS_FILE = "delistings.csv.gz"
CRSP_DAILY_FILE = "daily_stock.csv.gz"


@dataclass
class CrspDataBundle:
    monthly_stock: pd.DataFrame
    names: pd.DataFrame
    delistings: pd.DataFrame
    daily_stock: pd.DataFrame
    duplicate_report: pd.DataFrame


def _clean_string_columns(frame, columns):
    for column in columns:
        if column not in frame:
            continue

        values = frame[column].astype("string").str.strip().str.upper()
        frame[column] = values.mask(values == "")

    return frame


def _coerce_numeric_columns(frame, columns):
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _metadata_quality_score(frame, allowed_share_types=None):
    score = pd.Series(0, index=frame.index, dtype="int64")

    if "trading_status_flg" in frame:
        score += frame["trading_status_flg"].eq("A").fillna(False).astype("int64") * 16
    if "security_active_flg" in frame:
        score += frame["security_active_flg"].eq("Y").fillna(False).astype("int64") * 8
    if "security_type" in frame:
        score += frame["security_type"].eq("EQTY").fillna(False).astype("int64") * 4
    if "security_subtype" in frame:
        score += frame["security_subtype"].eq("COM").fillna(False).astype("int64") * 2
    if "share_type" in frame:
        if allowed_share_types is None:
            score += frame["share_type"].notna().astype("int64")
        else:
            score += frame["share_type"].isin(allowed_share_types).fillna(False).astype(
                "int64"
            )
    if "ticker" in frame:
        score += frame["ticker"].notna().astype("int64")
    if "trade_symbol" in frame:
        score += frame["trade_symbol"].notna().astype("int64")
    if "primary_exch" in frame:
        score -= frame["primary_exch"].eq("X").fillna(False).astype("int64")

    return score


def _clean_duplicate_market_rows(frame, key_columns, core_columns, table_name):
    raw_rows = len(frame)
    exact_duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()

    duplicate_mask = frame.duplicated(key_columns, keep=False)
    duplicate_rows = frame.loc[duplicate_mask]
    duplicate_groups = 0
    conflicting_core_groups = 0

    if not duplicate_rows.empty:
        duplicate_groups = duplicate_rows.groupby(key_columns).ngroups
        core_nunique = duplicate_rows.groupby(key_columns)[core_columns].nunique(
            dropna=False
        )
        core_conflicts = core_nunique.gt(1).any(axis=1)
        conflicting_core_groups = int(core_conflicts.sum())

        if conflicting_core_groups:
            raise ValueError(
                f"{table_name} has {conflicting_core_groups:,} duplicate "
                "PERMNO/date groups with conflicting core market data."
            )

        frame["_metadata_quality"] = _metadata_quality_score(frame)
        frame["_row_order"] = range(len(frame))
        frame = (
            frame.sort_values(
                key_columns + ["_metadata_quality", "_row_order"],
                ascending=[True] * len(key_columns) + [False, True],
            )
            .drop_duplicates(key_columns, keep="first")
            .drop(columns=["_metadata_quality", "_row_order"])
        )

    remaining_duplicate_rows = int(frame.duplicated(key_columns, keep=False).sum())

    if remaining_duplicate_rows:
        raise ValueError(
            f"{table_name} still has duplicate PERMNO/date rows after cleanup."
        )

    report = {
        "table": table_name,
        "raw_rows": raw_rows,
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "metadata_duplicate_groups_reduced": duplicate_groups,
        "conflicting_core_duplicate_groups": conflicting_core_groups,
        "clean_rows": len(frame),
    }

    return frame, report


def load_crsp_monthly_stock(data_dir=CRSP_DATA_DIR):
    columns = [
        "PERMNO",
        "HdrCUSIP",
        "PrimaryExch",
        "ConditionalType",
        "TradingStatusFlg",
        "USIncFlg",
        "IssuerType",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "Ticker",
        "PERMCO",
        "SICCD",
        "MthCalDt",
        "MthPrc",
        "MthCap",
        "MthRet",
        "MthVol",
        "ShrOut",
    ]
    monthly = pd.read_csv(
        data_dir / CRSP_MONTHLY_FILE,
        usecols=columns,
        parse_dates=["MthCalDt"],
    ).rename(
        columns={
            "PERMNO": "permno",
            "HdrCUSIP": "hdr_cusip",
            "PrimaryExch": "primary_exch",
            "ConditionalType": "conditional_type",
            "TradingStatusFlg": "trading_status_flg",
            "USIncFlg": "us_inc_flg",
            "IssuerType": "issuer_type",
            "SecurityType": "security_type",
            "SecuritySubType": "security_subtype",
            "ShareType": "share_type",
            "Ticker": "ticker",
            "PERMCO": "permco",
            "SICCD": "siccd",
            "MthCalDt": "date",
            "MthPrc": "mth_prc",
            "MthCap": "mth_cap",
            "MthRet": "mth_ret",
            "MthVol": "mth_vol",
            "ShrOut": "shrout",
        }
    )

    monthly["date"] = monthly["date"].dt.normalize()
    monthly["month_end"] = monthly["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = _coerce_numeric_columns(
        monthly,
        [
            "permno",
            "permco",
            "siccd",
            "mth_prc",
            "mth_cap",
            "mth_ret",
            "mth_vol",
            "shrout",
        ],
    )
    monthly["permno"] = monthly["permno"].astype("int64")
    monthly = _clean_string_columns(
        monthly,
        [
            "hdr_cusip",
            "primary_exch",
            "conditional_type",
            "trading_status_flg",
            "us_inc_flg",
            "issuer_type",
            "security_type",
            "security_subtype",
            "share_type",
            "ticker",
        ],
    )

    return _clean_duplicate_market_rows(
        frame=monthly,
        key_columns=["permno", "date"],
        core_columns=["mth_prc", "mth_cap", "mth_ret", "mth_vol", "shrout"],
        table_name="monthly_stock",
    )


def load_crsp_daily_stock(data_dir=CRSP_DATA_DIR):
    columns = [
        "PERMNO",
        "HdrCUSIP",
        "PrimaryExch",
        "ConditionalType",
        "TradingStatusFlg",
        "Ticker",
        "PERMCO",
        "SICCD",
        "DlyCalDt",
        "DlyPrc",
        "DlyVol",
    ]
    daily = pd.read_csv(
        data_dir / CRSP_DAILY_FILE,
        usecols=columns,
        parse_dates=["DlyCalDt"],
    ).rename(
        columns={
            "PERMNO": "permno",
            "HdrCUSIP": "hdr_cusip",
            "PrimaryExch": "primary_exch",
            "ConditionalType": "conditional_type",
            "TradingStatusFlg": "trading_status_flg",
            "Ticker": "ticker",
            "PERMCO": "permco",
            "SICCD": "siccd",
            "DlyCalDt": "date",
            "DlyPrc": "dly_prc",
            "DlyVol": "dly_vol",
        }
    )

    daily["date"] = daily["date"].dt.normalize()
    daily = _coerce_numeric_columns(
        daily,
        ["permno", "permco", "siccd", "dly_prc", "dly_vol"],
    )
    daily["permno"] = daily["permno"].astype("int64")
    daily = _clean_string_columns(
        daily,
        [
            "hdr_cusip",
            "primary_exch",
            "conditional_type",
            "trading_status_flg",
            "ticker",
        ],
    )

    return _clean_duplicate_market_rows(
        frame=daily,
        key_columns=["permno", "date"],
        core_columns=["dly_prc", "dly_vol"],
        table_name="daily_stock",
    )


def load_crsp_names(data_dir=CRSP_DATA_DIR):
    names = pd.read_csv(
        data_dir / CRSP_NAMES_FILE,
        parse_dates=[
            "secinfostartdt",
            "secinfoenddt",
            "securitybegdt",
            "securityenddt",
        ],
    ).rename(
        columns={
            "hdrprimaryexch": "hdr_primary_exch",
            "nasdissuno": "nasd_issuno",
            "hdrsiccd": "hdr_siccd",
            "secinfostartdt": "sec_info_start_dt",
            "secinfoenddt": "sec_info_end_dt",
            "securitybegdt": "security_beg_dt",
            "securityenddt": "security_end_dt",
            "issuernm": "issuer_name",
            "usincflg": "us_inc_flg",
            "issuertype": "issuer_type",
            "securitytype": "security_type",
            "securitysubtype": "security_subtype",
            "sharetype": "share_type",
            "securityactiveflg": "security_active_flg",
            "primaryexch": "primary_exch",
            "tradingsymbol": "trade_symbol",
            "tradingstatusflg": "trading_status_flg",
        }
    )

    date_columns = [
        "sec_info_start_dt",
        "sec_info_end_dt",
        "security_beg_dt",
        "security_end_dt",
    ]
    for column in date_columns:
        names[column] = names[column].dt.normalize()

    names = _coerce_numeric_columns(
        names,
        ["permno", "permco", "nasd_issuno", "hdr_siccd", "siccd", "naics"],
    )
    names["permno"] = names["permno"].astype("int64")
    names = _clean_string_columns(
        names,
        [
            "hdr_primary_exch",
            "cusip",
            "ticker",
            "issuer_name",
            "shareclass",
            "us_inc_flg",
            "issuer_type",
            "security_type",
            "security_subtype",
            "share_type",
            "security_active_flg",
            "primary_exch",
            "trade_symbol",
            "trading_status_flg",
        ],
    )

    return names.drop_duplicates().sort_values(["permno", "sec_info_start_dt"])


def load_crsp_delistings(data_dir=CRSP_DATA_DIR):
    delistings = pd.read_csv(
        data_dir / CRSP_DELISTINGS_FILE,
        parse_dates=["DelistingDt", "DelNextDt", "DelAmtDt", "DelDlyDt"],
    ).rename(
        columns={
            "PERMNO": "permno",
            "DelistingDt": "delisting_date",
            "DelRet": "delisting_return",
            "DelActionType": "del_action_type",
            "DelStatusType": "del_status_type",
            "DelReasonType": "del_reason_type",
            "DelPaymentType": "del_payment_type",
            "DelNextDt": "del_next_date",
            "DelAmtDt": "del_amount_date",
            "DelDlyDt": "del_daily_date",
            "PrimaryExch": "primary_exch",
            "SICCD": "siccd",
        }
    )

    for column in [
        "delisting_date",
        "del_next_date",
        "del_amount_date",
        "del_daily_date",
    ]:
        delistings[column] = delistings[column].dt.normalize()

    delistings = _coerce_numeric_columns(
        delistings,
        ["permno", "delisting_return", "siccd"],
    )
    delistings["permno"] = delistings["permno"].astype("int64")
    delistings = _clean_string_columns(
        delistings,
        [
            "del_action_type",
            "del_status_type",
            "del_reason_type",
            "del_payment_type",
            "primary_exch",
        ],
    )

    return delistings.drop_duplicates()


def load_crsp_data(data_dir=CRSP_DATA_DIR):
    monthly, monthly_report = load_crsp_monthly_stock(data_dir=data_dir)
    names = load_crsp_names(data_dir=data_dir)
    delistings = load_crsp_delistings(data_dir=data_dir)
    daily, daily_report = load_crsp_daily_stock(data_dir=data_dir)

    duplicate_report = pd.DataFrame([monthly_report, daily_report]).set_index("table")

    return CrspDataBundle(
        monthly_stock=monthly,
        names=names,
        delistings=delistings,
        daily_stock=daily,
        duplicate_report=duplicate_report,
    )


def summarize_crsp_data(crsp_data):
    rows = []

    specs = [
        ("monthly_stock", crsp_data.monthly_stock, "date"),
        ("names", crsp_data.names, "sec_info_start_dt"),
        ("delistings", crsp_data.delistings, "delisting_date"),
        ("daily_stock", crsp_data.daily_stock, "date"),
    ]

    for table, frame, date_column in specs:
        duplicate_info = (
            crsp_data.duplicate_report.loc[table].to_dict()
            if table in crsp_data.duplicate_report.index
            else {}
        )
        rows.append(
            {
                "table": table,
                "rows": len(frame),
                "date_min": frame[date_column].min(),
                "date_max": frame[date_column].max(),
                "permnos": frame["permno"].nunique() if "permno" in frame else pd.NA,
                **duplicate_info,
            }
        )

    return pd.DataFrame(rows).set_index("table")
