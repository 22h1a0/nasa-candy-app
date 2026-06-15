"""
Data Preparation & Feature Engineering
Nassau Candy Distributor - Factory Reallocation System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── Factory metadata ──────────────────────────────────────────────────────────
FACTORIES = {
    "Lot's O' Nuts":     {"lat": 32.881893, "lon": -111.768036, "state": "AZ"},
    "Wicked Choccy's":   {"lat": 32.076176, "lon":  -81.088371, "state": "GA"},
    "Sugar Shack":       {"lat": 48.119140, "lon":  -96.181150, "state": "MN"},
    "Secret Factory":    {"lat": 41.446333, "lon":  -90.565487, "state": "IL"},
    "The Other Factory": {"lat": 35.117500, "lon":  -89.971107, "state": "TN"},
}

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":    "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":            "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":       "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":           "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":    "Wicked Choccy's",
    "Laffy Taffy":                          "Sugar Shack",
    "SweeTARTS":                            "Sugar Shack",
    "Nerds":                                "Sugar Shack",
    "Fun Dip":                              "Sugar Shack",
    "Fizzy Lifting Drinks":                 "Sugar Shack",
    "Everlasting Gobstopper":               "Secret Factory",
    "Hair Toffee":                          "The Other Factory",
    "Lickable Wallpaper":                   "Secret Factory",
    "Wonka Gum":                            "Secret Factory",
    "Kazookles":                            "The Other Factory",
}

REGION_COORDS = {
    "Atlantic":  {"lat": 38.9, "lon": -77.0},
    "Gulf":      {"lat": 29.7, "lon": -90.1},
    "Interior":  {"lat": 41.8, "lon": -87.6},
    "Pacific":   {"lat": 37.8, "lon": -122.4},
}

SHIP_MODE_ORDER = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 3}
SHIP_MODE_BASE  = {"Same Day": 0, "First Class": 1, "Second Class": 3, "Standard Class": 5}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)
    raw_lt = (df["Ship Date"] - df["Order Date"]).dt.days - 904
    df["Lead Time"] = raw_lt % 365          # actual days-to-ship (0–11)
    df["Factory"]   = df["Product Name"].map(PRODUCT_FACTORY)

    def _dist(row):
        f = FACTORIES[row["Factory"]]
        r = REGION_COORDS[row["Region"]]
        return haversine_km(f["lat"], f["lon"], r["lat"], r["lon"])

    df["Distance_km"]     = df.apply(_dist, axis=1)
    df["Order_Month"]     = df["Order Date"].dt.month
    df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek
    df["Ship_Mode_Num"]   = df["Ship Mode"].map(SHIP_MODE_ORDER)
    df["Profit_Margin"]   = (df["Gross Profit"] / df["Sales"] * 100).round(2)
    return df


def build_feature_matrix(df: pd.DataFrame):
    le_product = LabelEncoder().fit(df["Product Name"])
    le_region  = LabelEncoder().fit(df["Region"])
    le_factory = LabelEncoder().fit(df["Factory"])
    le_mode    = LabelEncoder().fit(df["Ship Mode"])

    X = pd.DataFrame({
        "product_enc":   le_product.transform(df["Product Name"]),
        "region_enc":    le_region.transform(df["Region"]),
        "factory_enc":   le_factory.transform(df["Factory"]),
        "ship_mode_enc": le_mode.transform(df["Ship Mode"]),
        "distance_km":   df["Distance_km"],
        "units":         df["Units"],
        "cost":          df["Cost"],
        "order_month":   df["Order_Month"],
        "order_dow":     df["Order_DayOfWeek"],
    })

    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = df["Lead Time"].values

    encoders = {
        "product": le_product, "region": le_region,
        "factory": le_factory, "mode":   le_mode,
        "scaler":  scaler,     "feature_cols": list(X.columns),
    }
    return X_scaled, y, encoders
