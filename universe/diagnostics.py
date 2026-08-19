"""
Diagnostics for annual dynamic universe construction.
"""


def print_annual_diagnostics(diagnostics):
    """
    Print one compact annual diagnostics table
    """
    columns = [
        "passed_market_cap",
        "normal_liquidity_passes",
        "minimum_universe_additions",
        "final_universe",
    ]
    table = diagnostics[columns].rename(
        columns={
            "passed_market_cap": "market_cap_passes",
            "normal_liquidity_passes": "normal_liquidity_passes",
            "minimum_universe_additions": "fallback_additions",
            "final_universe": "final_universe",
        }
    )

    print("\nAnnual dynamic universe")
    print(table.to_string())
