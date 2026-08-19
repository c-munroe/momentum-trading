# Natural Resource Momentum Backtest

## Overview

This repository tests momentum strategies across natural-resource equities using monthly Yahoo Finance data. It runs a research grid across signal definitions, portfolio construction rules, benchmark comparisons, and validation methods. The equity universe is resource-linked: energy, metals, mining, chemicals, fertilizers, forestry, construction materials, and uranium.

## Research Question

The core question is whether momentum signals can identify persistent relative strength within natural-resource equities, and whether that result survives different portfolio constructions and validation checks.

## Initial Hypothesis

I believed momentum would be especially strong in natural-resource equities because industries like oil, gas, and mining are heavily influenced by news, geopolitical events, commodity-price moves, and investor sentiment. I expected those forces to create large waves and persistent trends rather than isolated price movements. Because momentum strategies try to capture those trends, I thought natural-resource equities would be a particularly strong place to test whether momentum actually works.

## Strategy Framework

The backtest builds monthly cross-sectional strategies. Each month, stocks receive a signal score, eligible stocks are ranked, and portfolios are formed from selected names. The grid varies signal family, window, strategy type, selection count, weighting, and optional maximum position caps.

Top-N variants test `top10`, `top20`, and `top30`. Scaled variants test caps of 25%, 20%, and 15%. Total gross exposure is 1.0: 100% long for long-only portfolios, or 50% long and 50% short for classic long/short portfolios.

## Momentum Signals

Signal code lives in `strats/momentum_signals.py` and is called through `strats/builder.py`. Raw momentum is calculated from monthly adjusted close prices as the return from a prior starting price to a more recent ending price, with a skip month. Implemented raw windows are `3-1`, `6-1`, `12-1`, and `18-1`.

Volatility-adjusted momentum divides raw momentum by trailing monthly return volatility. The implemented volatility lookback is 6 months.

The project also implements simple, exponential, and linearly weighted moving-average crossover signals. Prices are shifted before calculating moving averages. Crossover windows are `3-12`, `6-12`, and `6-18`; each score is fast average divided by slow average minus one.

## Portfolio Construction

Portfolio construction is split across `strats/long_only.py`, `strats/long_short.py`, `strats/long_short_abs.py`, `strats/threshold_momentum.py`, and `strats/weights.py`. Implemented types:

- Long-only: long the top-ranked stocks by signal score.
- Classic long-short: long the strongest momentum names and short the weakest momentum names.
- Directional absolute momentum: rank stocks by absolute signal value, then use signal sign to decide long or short.
- Threshold momentum: hold stocks whose raw momentum is strictly greater than a threshold.

Threshold strategies are long-only and equal-weighted. They use raw momentum windows with thresholds of 0%, 20%, and 50%. Unlike Top-N strategies, they may hold a variable number of stocks. If no stocks pass, the strategy records a 0% return for that month.

Equal-weight portfolios assign the same weight to each selected position. Scaled portfolios allocate more weight to stronger signals. Long-only scaling shifts selected scores and normalizes them; classic long-short scaling normalizes each side separately; directional absolute scaling weights larger absolute signals more heavily. When `max_weight` is supplied, individual weights are capped and remaining exposure is redistributed.

## Universe Construction

The project supports static and dynamic universe modes.

### Static Universe

Static mode uses the hand-curated `NATURAL_RESOURCE_TICKERS` list in `universe/static.py`. It does not attempt annual reconstitution and should be interpreted as survivorship-biased because the list is manually defined using currently known tickers.

### Dynamic Universe

Dynamic mode builds an annual eligibility table from a broader Yahoo Finance candidate pool. The pool comes from Yahoo's current `Energy` and `Basic Materials` screener results on the `NMS`, `NYQ`, and `ASE` exchanges.

The dynamic process currently does the following:

- discovers current Yahoo-listed Energy and Basic Materials equity candidates
- downloads adjusted close prices for returns and raw close/volume for liquidity checks
- reconstitutes the eligible universe once per year, labeled on January 1
- uses data strictly before the reconstitution date
- calculates trailing average daily dollar volume over a 60-trading-day lookback
- requires at least 40 valid liquidity observations
- applies a $5 million average daily dollar volume threshold
- attempts a historical $1 billion market-cap screen using Yahoo historical shares outstanding multiplied by historical price
- targets a minimum annual universe size of 60 names
- includes all names above the normal liquidity threshold
- adds the most liquid below-threshold valid names only if needed to reach the minimum universe target
- applies no arbitrary maximum annual universe size

The current Yahoo-based dynamic universe is still survivorship-biased. It starts from companies listed in Yahoo's screener today, so missing historical companies cannot enter earlier universes.

Historical market-cap coverage is incomplete. The code does not backfill current shares outstanding into the past. If Yahoo has no historical shares observation before a reconstitution date, the name fails the market-cap screen. Yahoo historical shares and market-cap coverage is sparse before roughly 2016, limiting the reliability of the historical $1 billion filter in earlier years.

## Backtest Methodology

The backtest uses Yahoo Finance adjusted close data starting from `2000-01-01`. Prices are resampled to month-end, and monthly returns are calculated from month-end adjusted prices. The run script excludes the current partial month.

Signals are calculated on monthly prices. Momentum and moving-average inputs are shifted by the configured skip month before portfolio formation, so the same month being traded is not used for ranking. In dynamic mode, annual universe membership controls eligibility; signals may still be calculated from full price history, but ineligible names are masked out before selection.

The strategy grid is evaluated over the full available sample and compared with benchmark ETF returns. The code also calculates an equal-weight resource-universe return.

## Validation

Validation logic is implemented in `backtest/validation.py`. The train/test split uses the first 70% of usable monthly strategy-return history as in-sample data and the final 30% as out-of-sample data. Strategies need at least 36 valid monthly returns before ranking.

The code ranks in-sample strategies by Sharpe ratio and checks them out of sample. Walk-forward validation uses a rolling 120-month training window and a 12-month test window. At each decision date, the best trailing-window strategy by Sharpe ratio is applied over the next test window, and those out-of-sample segments are concatenated.

## Benchmarks

The benchmark ETF tickers used by the project are `SPY`, `XLE`, `XLB`, `XME`, `GDX`, `GLD`, `USO`, and `DBC`.

Relative metrics are currently calculated against `SPY`, `XLE`, `XLB`, `GLD`, and `DBC` when available.

The code also downloads commodity futures in static mode: `GC=F`, `SI=F`, `HG=F`, `CL=F`, `BZ=F`, `NG=F`, `ZC=F`, `ZW=F`, and `ZS=F`.

## Performance Metrics

The current metrics include:

- number of monthly periods
- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- maximum drawdown
- win rate
- final portfolio value
- beta to selected benchmarks
- alpha to selected benchmarks
- correlation to selected benchmarks
- information ratio to selected benchmarks

Annualized return is calculated from compounded monthly returns. Sharpe uses a zero risk-free rate.

## Project Structure

```text
.
├── backtest/
│   ├── data_download.py
│   ├── metrics.py
│   ├── reporting.py
│   ├── run_backtest.py
│   ├── universe_setup.py
│   └── validation.py
├── strats/
│   ├── builder.py
│   ├── long_only.py
│   ├── long_short.py
│   ├── long_short_abs.py
│   ├── momentum_signals.py
│   ├── portfolio.py
│   ├── threshold_momentum.py
│   └── weights.py
└── universe/
    ├── diagnostics.py
    ├── dynamic.py
    ├── historical_data.py
    └── static.py
```

`backtest/` contains data download, metrics, reporting, validation, and the main run script. `strats/` contains signals, selection, weighting, and return calculation. `universe/` contains the static ticker list and dynamic annual universe logic.

## Running the Backtest

Run the main research grid with:

```bash
python -m backtest.run_backtest
```

Universe mode is controlled by the `UNIVERSE_MODE` variable in `backtest/run_backtest.py`:

```python
UNIVERSE_MODE = "static"
# UNIVERSE_MODE = "dynamic"
```

Set it to `"static"` for the hand-curated ticker list or `"dynamic"` to build annual Yahoo-based eligibility tables. With `SHOW_PLOTS = True`, the script displays equity-curve, Sharpe, walk-forward, and dynamic-universe plots.

## Data Sources

Yahoo Finance is the current data source. The project uses Yahoo adjusted close prices for strategy returns, benchmark ETF returns, and commodity futures. In dynamic mode, it also uses Yahoo raw close prices and volume for liquidity screening.

For the historical market-cap attempt, the code calls Yahoo historical shares outstanding through `yfinance`, then multiplies shares by historical prices. It intentionally does not substitute today's shares outstanding for missing historical observations.

## Current Limitations

The static universe is hand-curated and survivorship-biased. The dynamic universe is broader, but still survivorship-biased because its candidate pool comes from companies listed in Yahoo's current Energy and Basic Materials screener rather than a point-in-time security master.

The historical market-cap screen is limited by sparse Yahoo historical shares data, especially before roughly 2016. Earlier annual universes may exclude names because Yahoo lacks historical shares data, not necessarily because those companies were below $1 billion in market cap.

The dynamic universe does not currently use point-in-time historical sector or subsector classifications. In the current run path, resource classification is handled by the current Yahoo sector screener, and `require_resource_classification` is set to `False`.

The backtest does not model transaction costs, slippage, bid/ask spreads, market impact, borrow costs, financing costs, taxes, or turnover constraints. Long/short returns are calculated directly from signed monthly weights and asset returns. The code reports research results, but it does not save a formal results artifact or parameter manifest.

## Future Improvements

Likely next steps include replacing Yahoo universe construction with point-in-time CRSP, WRDS, or another survivorship-bias-aware security master; improving historical market-cap coverage; adding point-in-time classifications; and adding transaction costs, slippage, short borrow costs, financing rates, and turnover reporting.
