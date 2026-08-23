"""
Point-in-time resource-industry classification helpers.
"""

import math


def _clean_digits(value):
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""

    try:
        text = str(int(float(text)))
    except ValueError:
        pass

    digits = "".join(character for character in text if character.isdigit())

    if not digits or int(digits) == 0:
        return ""

    return digits


def _clean_int(value):
    digits = _clean_digits(value)

    if not digits:
        return None

    return int(digits)


def _classify_sic(siccd):
    siccd = _clean_int(siccd)

    if siccd is None:
        return None

    if siccd == 1094:
        return "uranium"

    sic_ranges = [
        (800, 899, "forestry"),
        (1000, 1099, "metals_mining"),
        (1200, 1299, "coal"),
        (1300, 1399, "oil_gas"),
        (1400, 1499, "nonmetallic_mining"),
        (2400, 2499, "lumber_wood_products"),
        (2600, 2699, "paper_forest_products"),
        (2800, 2899, "chemicals"),
        (2910, 2911, "refining"),
        (2990, 2999, "petroleum_coal_products"),
        (3200, 3299, "construction_materials"),
        (3300, 3399, "primary_metals"),
        (4610, 4619, "pipelines"),
        (4920, 4925, "natural_gas_distribution"),
        (5050, 5052, "metals_service_centers"),
    ]

    for lower, upper, label in sic_ranges:
        if lower <= siccd <= upper:
            return label

    return None


def _classify_naics(naics):
    naics = _clean_digits(naics)

    if not naics:
        return None

    if naics.startswith("212291"):
        return "uranium"

    naics_prefixes = [
        ("113", "forestry"),
        ("1153", "forestry_support"),
        ("211", "oil_gas"),
        ("2121", "coal"),
        ("2122", "metals_mining"),
        ("2123", "nonmetallic_mining"),
        ("213", "mining_support"),
        ("2212", "natural_gas_distribution"),
        ("321", "lumber_wood_products"),
        ("322", "paper_forest_products"),
        ("324", "petroleum_coal_products"),
        ("325", "chemicals"),
        ("327", "construction_materials"),
        ("331", "primary_metals"),
        ("4235", "metals_service_centers"),
        ("42491", "agriculture_inputs"),
        ("486", "pipelines"),
    ]

    for prefix, label in naics_prefixes:
        if naics.startswith(prefix):
            return label

    return None


def classify_resource_industry(siccd=None, naics=None):
    """
    Return a resource-industry label for historical SIC/NAICS codes.

    Ticker and issuer names are intentionally ignored here. Callers should pass
    only point-in-time classification codes valid for the date being screened.
    """
    return _classify_naics(naics) or _classify_sic(siccd)


def is_resource_industry(siccd=None, naics=None):
    return classify_resource_industry(siccd=siccd, naics=naics) is not None
