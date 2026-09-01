"""
Integration layer from CRSP data to backtest-ready monthly matrices.
"""

from dataclasses import dataclass

import pandas as pd

from backtest.config import CRSP_DATA_DIR
from universe.crsp.construction import build_crsp_annual_universes
from universe.crsp.data import CrspDataBundle, load_crsp_data
from universe.dynamic import DynamicUniverseResult, get_universe_for_date


@dataclass
class CrspDynamicUniverseData:
    result: DynamicUniverseResult
    monthly_prices: pd.DataFrame
    monthly_returns: pd.DataFrame
    eligibility_table: pd.DataFrame
    crsp_data: CrspDataBundle


def _assert_unique_monthly_matrix_rows(monthly):
    duplicate_rows = int(monthly.duplicated(["permno", "month_end"], keep=False).sum())

    if duplicate_rows:
        raise ValueError(
            "monthly_stock has duplicate PERMNO/month rows after month-end normalization."
        )


def build_crsp_monthly_matrices(monthly, permnos):
    monthly = monthly[monthly["permno"].isin(permnos)].copy()
    _assert_unique_monthly_matrix_rows(monthly)

    monthly_returns = (
        monthly.pivot(index="month_end", columns="permno", values="mth_ret")
        .sort_index()
        .sort_index(axis=1)
    )
    gross_returns = (1 + monthly_returns).where((1 + monthly_returns) > 0)
    monthly_prices = gross_returns.cumprod()

    monthly_returns.columns.name = None
    monthly_prices.columns.name = None

    return monthly_prices, monthly_returns


def build_crsp_monthly_eligibility_table(monthly_returns, annual_universes):
    permnos = monthly_returns.columns
    table = pd.DataFrame(False, index=monthly_returns.index, columns=permnos)

    for date in table.index:
        annual_permnos = get_universe_for_date(
            date=date,
            annual_universes=annual_universes,
        )
        available_permnos = [permno for permno in annual_permnos if permno in table]
        table.loc[date, available_permnos] = True

    return table


def build_crsp_dynamic_universe_data(data_dir=CRSP_DATA_DIR):
    crsp_data = load_crsp_data(data_dir=data_dir)
    result = build_crsp_annual_universes(crsp_data=crsp_data)
    selected_permnos = sorted(
        {
            permno
            for annual_permnos in result.annual_universes.values()
            for permno in annual_permnos
        }
    )

    monthly_prices, monthly_returns = build_crsp_monthly_matrices(
        monthly=crsp_data.monthly_stock,
        permnos=selected_permnos,
    )
    eligibility_table = build_crsp_monthly_eligibility_table(
        monthly_returns=monthly_returns,
        annual_universes=result.annual_universes,
    )

    return CrspDynamicUniverseData(
        result=result,
        monthly_prices=monthly_prices,
        monthly_returns=monthly_returns,
        eligibility_table=eligibility_table,
        crsp_data=crsp_data,
    )
