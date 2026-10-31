"""The trained universe and its sector assignment.

Lifted verbatim from Cell 2 of the training notebook. The models were fitted
against percentile ranks computed across exactly these 231 names, so this list
is part of the model, not a configuration choice. Adding or removing a ticker
shifts every other ticker's rank and invalidates the saved artefacts.

Sectors are declared here rather than pulled from yfinance `.info`, which would
require 231 sequential metadata calls and get rate-limited.
"""

SECTOR_MAP = {
    "IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTM", "PERSISTENT",
        "COFORGE", "MPHASIS", "LTTS", "OFSS", "TATAELXSI", "CYIENT", "SONATSOFTW",
    ],
    "Banks": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
        "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IDFCFIRSTB", "FEDERALBNK",
        "AUBANK", "BANDHANBNK", "RBLBANK", "INDIANB", "KARURVYSYA", "CUB",
    ],
    "Financials": [
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULI",
        "ICICIGI", "SBICARD", "CHOLAFIN", "MUTHOOTFIN", "LICHSGFIN", "PFC",
        "RECLTD", "SHRIRAMFIN", "MFSL", "ABCAPITAL", "HDFCAMC", "M&MFIN",
        "BAJAJHLDNG", "IIFL", "MANAPPURAM",
    ],
    "Auto": [
        "MARUTI", "TMPV", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
        "TVSMOTOR", "ASHOKLEY", "BHARATFORG", "MOTHERSON", "BOSCHLTD", "MRF",
        "BALKRISIND", "EXIDEIND", "APOLLOTYRE", "CEATLTD", "ENDURANCE",
    ],
    "Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "AUROPHARMA",
        "TORNTPHARM", "ALKEM", "ZYDUSLIFE", "BIOCON", "GLENMARK", "IPCALAB",
        "LAURUSLABS", "ABBOTINDIA", "SYNGENE", "PFIZER", "NATCOPHARM",
        "AJANTPHARM", "JBCHEPHARM", "GRANULES",
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO",
        "GODREJCP", "COLPAL", "TATACONSUM", "UBL", "RADICO", "EMAMILTD",
        "JYOTHYLAB", "PGHH", "GILLETTE", "BAJAJCON",
    ],
    "Metals": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "SAIL",
        "NMDC", "NATIONALUM", "HINDZINC", "APLAPOLLO", "RATNAMANI", "WELCORP",
        "JSL", "HINDCOPPER",
    ],
    "Energy": [
        "RELIANCE", "ONGC", "IOC", "BPCL", "HINDPETRO", "GAIL", "PETRONET",
        "OIL", "IGL", "GUJENERGY", "CASTROLIND",
    ],
    "Power": [
        "NTPC", "POWERGRID", "TATAPOWER", "JSWENERGY", "NHPC", "TORNTPOWER",
        "CESC", "THERMAX",
    ],
    "Cement": [
        "ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC", "DALBHARAT", "JKCEMENT",
        "RAMCOCEM", "BIRLACORPN", "HEIDELBERG",
    ],
    "CapitalGoods": [
        "LT", "SIEMENS", "ABB", "HAVELLS", "CUMMINSIND", "BHEL", "BEL",
        "CGPOWER", "KEI", "VOLTAS", "BLUESTARCO", "AIAENG", "GRINDWELL",
        "SKFINDIA", "TIMKEN", "SCHAEFFLER", "KSB", "ELGIEQUIP", "TRITURBINE",
    ],
    "Chemicals": [
        "PIDILITIND", "ASIANPAINT", "BERGEPAINT", "KANSAINER", "SRF",
        "DEEPAKNTR", "AARTIIND", "ATUL", "VINATIORGA", "NAVINFLUOR",
        "TATACHEM", "GNFC", "COROMANDEL", "CHAMBLFERT", "UPL", "PIIND",
        "SUMICHEM", "BASF", "LINDEINDIA", "SOLARINDS", "RALLIS", "GSFC",
    ],
    "Consumer": [
        "TITAN", "TRENT", "PAGEIND", "BATAINDIA", "RELAXO", "VGUARD",
        "WHIRLPOOL", "SYMPHONY", "CENTURYPLY", "KAJARIACER", "SUPREMEIND",
        "ASTRAL", "FINPIPE", "TTKPRESTIG", "HONAUT", "3MINDIA",
    ],
    "Telecom": [
        "BHARTIARTL", "IDEA", "TATACOMM", "SUNTV", "ZEEL", "NETWORK18",
    ],
    "Realty": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE", "PHOENIXLTD",
        "SOBHA", "MAHLIFE",
    ],
    "Transport": [
        "CONCOR", "ADANIPORTS", "GESHIP", "TCI", "BLUEDART", "MAHLOG",
    ],
    "Diversified": [
        "ADANIENT", "GRASIM", "TRIDENT", "WELSPUNLIV", "KPRMILL", "CARBORUNIV",
        "KIRLOSENG",
    ],
}

# Flatten to yfinance tickers and build the reverse sector lookup
TICKERS, TICKER_SECTOR = [], {}

for sector, names in SECTOR_MAP.items():
    for n in names:
        t = f"{n}.NS"
        TICKERS.append(t)
        TICKER_SECTOR[t] = sector

MACRO_VIX   = "^INDIAVIX"
MACRO_INDEX = "^NSEI"


print_summary = (
    f"Universe: {len(TICKERS)} tickers across {len(SECTOR_MAP)} sectors"
)
