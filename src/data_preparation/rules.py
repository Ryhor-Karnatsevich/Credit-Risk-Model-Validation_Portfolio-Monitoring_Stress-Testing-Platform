"""Explicit, reviewable rules; no imputation, clipping or automatic row deletion."""

MISSING_CODES = {
    "CLASSIC FICO": ("9999",), "VANTAGESCORE 4.0": ("9999",),
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": ("999",), "NUMBER OF UNITS": ("99",),
    "OCCUPANCY STATUS": ("9",), "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)": ("999",),
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": ("999",),
    "ORIGINAL LOAN-TO-VALUE (LTV)": ("999",), "NUMBER OF BORROWERS": ("99",),
    "FIRST TIME HOMEBUYER INDICATOR": ("9",), "CHANNEL": ("9",),
    "PROPERTY TYPE": ("99",), "LOAN PURPOSE": ("9",),
    "PROPERTY VALUATION METHOD": ("7",), "ESTIMATED LOAN-TO-VALUE (ELTV)": ("999",),
    "CURRENT LOAN DELINQUENCY STATUS": ("XX",),
}

# Only obvious, source-documented ranges; unusual values are flagged, not removed.
RANGES = {
    "CLASSIC FICO": (300, 850), "VANTAGESCORE 4.0": (300, 850),
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": (0, 65), "NUMBER OF UNITS": (1, 4),
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": (0, 55),
    "ORIGINAL UPB": (0, None), "ORIGINAL INTEREST RATE": (0, None),
    "ORIGINAL LOAN TERM": (1, None), "CURRENT ACTUAL UPB": (0, None),
    "CURRENT INTEREST RATE": (0, None),
}

DATE_FIELDS = {"FIRST PAYMENT DATE", "MATURITY DATE", "PERIOD", "ZERO BALANCE EFFECTIVE DATE"}
CREDIT_EXIT_CODES = ("02", "03", "09")
KNOWN_EXIT_CODES = ("01", "02", "03", "09", "15", "16", "96")

# Review table inputs are origination fields only. Future performance fields
# supply outcomes/quality flags and must not be used as predictive features.
FEATURES = {
    "CLASSIC FICO": ("fico", "DOUBLE"),
    "ORIGINAL UPB": ("original_balance", "DOUBLE"),
    "ORIGINAL INTEREST RATE": ("original_rate", "DOUBLE"),
    "ORIGINAL LOAN-TO-VALUE (LTV)": ("ltv", "DOUBLE"),
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)": ("cltv", "DOUBLE"),
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": ("dti", "DOUBLE"),
    "ORIGINAL LOAN TERM": ("term_months", "INTEGER"),
    "NUMBER OF BORROWERS": ("borrowers_code", "INTEGER"),
    "PROPERTY STATE": ("property_state", "VARCHAR"),
    "PROPERTY TYPE": ("property_type", "VARCHAR"),
    "OCCUPANCY STATUS": ("occupancy", "VARCHAR"),
    "LOAN PURPOSE": ("purpose", "VARCHAR"),
    "CHANNEL": ("channel", "VARCHAR"),
}

USER_GUIDE = "https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf"


def identifier(name):
    return '"' + name.replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def month_date(expression):
    return (
        f"CASE WHEN regexp_full_match({expression}, '[0-9]{{6}}') "
        f"AND substr({expression}, 1, 4) BETWEEN '1990' AND '2100' "
        f"THEN CAST(try_strptime({expression}, '%Y%m') AS DATE) END"
    )
