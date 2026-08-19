"""
Ticker universes used by the backtest.

The resource equity universe is hand-curated, so 
inteprate results with that limitation in mind. 
"""

# Natural resources / commodities-linked equity universe
# Includes energy, mining, metals, chemicals, fertilizers, forestry,
# construction materials, and uranium names

NATURAL_RESOURCE_TICKERS = [
    # -----------------------------
    # Energy producers / integrated oil & gas
    # These companies are directly tied to oil, natural gas, and global energy prices.
    # -----------------------------
    "XOM",   # Exxon Mobil - integrated oil & gas giant
    "CVX",   # Chevron - integrated oil & gas giant
    "COP",   # ConocoPhillips - large oil & gas producer
    "EOG",   # EOG Resources - U.S. shale oil/gas producer
    "OXY",   # Occidental Petroleum - oil, gas, and chemicals
    "FANG",  # Diamondback Energy - Permian Basin oil producer
    "DVN",   # Devon Energy - U.S. oil & gas producer
    "CTRA",  # Coterra Energy - natural gas and oil producer
    "APA",   # APA Corp. - oil & gas exploration
    "PR",    # Permian Resources - Permian-focused oil producer
    "MTDR",  # Matador Resources - oil & gas producer
    "CIVI",  # Civitas Resources - U.S. oil & gas producer
    "CHRD",  # Chord Energy - Bakken oil producer
    "OVV",   # Ovintiv - North American oil/gas producer
    "CNQ",   # Canadian Natural Resources - Canadian oil sands/energy
    "SU",    # Suncor Energy - Canadian oil sands/integrated energy
    "CVE",   # Cenovus Energy - Canadian oil sands/refining
    "IMO",   # Imperial Oil - Canadian integrated oil
    "PBR",   # Petrobras - Brazilian oil major
    "YPF",   # YPF - Argentine oil & gas
    "EC",    # Ecopetrol - Colombian oil & gas
    "SHEL",  # Shell - global integrated energy
    "BP",    # BP - global integrated energy
    "TTE",   # TotalEnergies - global integrated energy
    "EQNR",  # Equinor - Norwegian oil/gas and offshore energy

    # -----------------------------
    # Midstream, LNG (natural gas), and refining
    # These firms transport, store, export, or refine energy products.
    # They may be less directly exposed to oil prices than producers but still move with energy cycles.
    # -----------------------------
    "WMB",   # Williams Companies - natural gas pipelines
    "KMI",   # Kinder Morgan - pipelines and energy infrastructure
    "OKE",   # ONEOK - natural gas liquids/midstream
    "TRP",   # TC Energy - North American pipelines
    "ENB",   # Enbridge - pipelines and energy infrastructure
    "LNG",   # Cheniere Energy - LNG export company
    "ET",    # Energy Transfer - midstream pipelines/terminals
    "EPD",   # Enterprise Products Partners - large midstream MLP
    "MPLX",  # MPLX - midstream energy logistics
    "PAA",   # Plains All American - oil pipelines/storage
    "VLO",   # Valero Energy - oil refiner
    "MPC",   # Marathon Petroleum - oil refiner
    "PSX",   # Phillips 66 - refining and chemicals
    "DINO",  # HF Sinclair - oil refining
    "SUN",   # Sunoco - fuel distribution

    # -----------------------------
    # Oilfield services / equipment
    # These companies provide drilling, fracking, equipment, and services to energy producers.
    # They are often cyclical and sensitive to energy capital spending.
    # -----------------------------
    "SLB",   # Schlumberger - oilfield services leader
    "HAL",   # Halliburton - oilfield services
    "BKR",   # Baker Hughes - oilfield services/equipment
    "NOV",   # NOV Inc. - drilling equipment
    "FTI",   # TechnipFMC - offshore/subsea energy equipment
    "HP",    # Helmerich & Payne - drilling rigs
    "PTEN",  # Patterson-UTI - contract drilling/services
    "LBRT",  # Liberty Energy - fracking/oilfield services
    "RES",   # RPC Inc. - oilfield services
    "NBR",   # Nabors Industries - drilling contractor

    # -----------------------------
    # Gold, silver, and precious metals
    # These companies are tied to gold, silver, platinum, palladium, and precious metals prices.
    # They may behave differently from energy because gold can act as an inflation/safe-haven asset.
    # -----------------------------
    "NEM",   # Newmont - major gold miner
    "GOLD",  # Barrick Gold - major gold miner
    "AEM",   # Agnico Eagle Mines - gold miner
    "WPM",   # Wheaton Precious Metals - precious metals streaming
    "FNV",   # Franco-Nevada - gold royalty/streaming
    "RGLD",  # Royal Gold - precious metals royalties
    "KGC",   # Kinross Gold - gold miner
    "AU",    # AngloGold Ashanti - gold miner
    "HMY",   # Harmony Gold - gold miner
    "GFI",   # Gold Fields - gold miner
    "AGI",   # Alamos Gold - gold miner
    "OR",    # Osisko Gold Royalties - gold royalty company
    "PAAS",  # Pan American Silver - silver/gold miner
    "HL",    # Hecla Mining - silver/gold miner
    "CDE",   # Coeur Mining - silver/gold miner
    "AG",    # First Majestic Silver - silver miner
    "FSM",   # Fortuna Mining - silver/gold miner
    "SBSW",  # Sibanye Stillwater - platinum, palladium, and gold

    # -----------------------------
    # Base metals, steel, and diversified mining
    # These companies are tied to copper, iron ore, aluminum, steel, rare earths, and industrial metals.
    # They are often sensitive to global growth, China demand, and infrastructure cycles.
    # -----------------------------
    "FCX",   # Freeport-McMoRan - copper/gold miner
    "SCCO",  # Southern Copper - copper miner
    "TECK",  # Teck Resources - copper, zinc, coal/metals
    "RIO",   # Rio Tinto - diversified global mining
    "BHP",   # BHP Group - diversified global mining
    "VALE",  # Vale - iron ore/nickel miner
    "AA",    # Alcoa - aluminum producer
    "CENX",  # Century Aluminum - aluminum producer
    "CLF",   # Cleveland-Cliffs - steel/iron ore
    "NUE",   # Nucor - steel producer
    "STLD",  # Steel Dynamics - steel producer
    "RS",    # Reliance Inc. - metals service center
    "CRS",   # Carpenter Technology - specialty metals/alloys
    "ATI",   # ATI Inc. - specialty metals/materials
    "MT",    # ArcelorMittal - global steel producer
    "MP",    # MP Materials - rare earth materials

    # -----------------------------
    # Chemicals, fertilizers, and agriculture inputs
    # These firms are linked to industrial production, farming cycles, crop prices, and input costs.
    # Fertilizer names can be especially commodity-sensitive.
    # -----------------------------
    "LIN",   # Linde - industrial gases
    "APD",   # Air Products - industrial gases
    "DD",    # DuPont - specialty materials/chemicals
    "DOW",   # Dow Inc. - commodity chemicals
    "LYB",   # LyondellBasell - chemicals/plastics
    "CE",    # Celanese - specialty chemicals
    "FMC",   # FMC Corp. - agricultural chemicals
    "MOS",   # Mosaic - fertilizer/potash/phosphate
    "NTR",   # Nutrien - fertilizer/agriculture inputs
    "CF",    # CF Industries - nitrogen fertilizer

    # -----------------------------
    # Forestry, construction materials, and uranium
    # These are resource-linked but not purely energy/metals.
    # Useful for testing whether momentum works across different resource subsectors.
    # -----------------------------
    "WY",    # Weyerhaeuser - timber/forestry REIT
    "PCH",   # PotlatchDeltic - timberland/wood products
    "VMC",   # Vulcan Materials - aggregates/construction materials
    "MLM",   # Martin Marietta Materials - aggregates/construction materials
    "CCJ",   # Cameco - uranium producer
    "UEC",   # Uranium Energy Corp. - uranium miner/developer
]


# Benchmark ETFs used to compare the strategy against broad market,
# sector, metals, gold, oil, and commodity exposure.

BENCHMARK_TICKERS = [
    "SPY",  # Broad U.S. equity market benchmark
    "XLE",  # Energy sector ETF
    "XLB",  # Materials sector ETF
    "XME",  # Metals and mining ETF
    "GDX",  # Gold miners ETF
    "GLD",  # Gold price proxy
    "USO",  # Oil price proxy
    "DBC",  # Broad commodities ETF
]

# Futures to test strategy against commodities 

COMMODITY_FUTURES_TICKERS = [
    "GC=F",  # Gold futures
    "SI=F",  # Silver futures
    "HG=F",  # Copper futures
    "CL=F",  # WTI crude oil futures
    "BZ=F",  # Brent crude oil futures
    "NG=F",  # Natural gas futures
    "ZC=F",  # Corn futures
    "ZW=F",  # Wheat futures
    "ZS=F",  # Soybean futures
]
