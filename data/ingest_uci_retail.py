import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).parent / "raw" / "online_retail_II.csv"
OUTPUT_PATH = Path(__file__).parent / "uci_backtest_data.csv"

TOP_N_PRODUCTS = 20
MAX_QUANTITY_PER_TRANSACTION = 1000
TRIM_TRAILING_DAYS = 2
RESTOCK_INTERVAL_DAYS = 14
RESTOCK_BUFFER_DAYS = 21
RANDOM_SEED = 42

NON_PRODUCT_STOCK_CODES = {
    "POST", "DOT", "M", "C2", "BANK CHARGES", "PADS", "D", "AMAZONFEE", "S", "CRUK",
}


def load_raw_transactions():
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise FileNotFoundError(
            f"No raw file at {RAW_PATH} and ucimlrepo is not installed. "
            f"Either place online_retail_II.csv there or `pip install ucimlrepo`."
        ) from exc

    online_retail = fetch_ucirepo(id=502)
    return online_retail.data.features


def clean_transactions(df):
    df = df.copy()
    df["StockCode"] = df["StockCode"].astype(str).str.upper().str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()

    is_sale = ~df["Invoice"].astype(str).str.startswith("C")
    has_valid_qty_and_price = (
        (df["Quantity"] > 0) & (df["Quantity"] <= MAX_QUANTITY_PER_TRANSACTION) & (df["Price"] > 0)
    )
    has_description = df["Description"].notna() & (df["Description"] != "") & (df["Description"] != "nan")
    is_real_product = ~df["StockCode"].isin(NON_PRODUCT_STOCK_CODES)

    cleaned = df[is_sale & has_valid_qty_and_price & has_description & is_real_product].copy()
    cleaned["date"] = pd.to_datetime(cleaned["InvoiceDate"]).dt.normalize()
    cleaned["product"] = cleaned["Description"]
    return cleaned[["date", "product", "Quantity", "Price"]]


def select_top_products(df, top_n=TOP_N_PRODUCTS):
    revenue_by_product = (
        (df["Quantity"] * df["Price"]).groupby(df["product"]).sum().sort_values(ascending=False)
    )
    return revenue_by_product.head(top_n).index.tolist()


def aggregate_daily(df):
    daily = df.groupby(["product", "date"]).apply(
        lambda g: pd.Series({
            "units_sold": g["Quantity"].sum(),
            "price": (g["Quantity"] * g["Price"]).sum() / g["Quantity"].sum(),
        }),
        include_groups=False,
    )
    return daily.reset_index()


def fill_date_gaps(product_df, full_date_range):
    product_df = product_df.set_index("date").reindex(full_date_range)
    product_df["units_sold"] = product_df["units_sold"].fillna(0)
    product_df["price"] = product_df["price"].ffill().bfill()
    product_df.index.name = "date"
    return product_df.reset_index()


def simulate_stock_levels(units_sold, rng):
    typical_daily_demand = max(units_sold.mean(), 1.0)
    initial_stock = int(typical_daily_demand * RESTOCK_BUFFER_DAYS)

    stock_levels = np.empty(len(units_sold), dtype=int)
    stock = initial_stock
    for day_index, sold in enumerate(units_sold):
        if day_index % RESTOCK_INTERVAL_DAYS == 0:
            restock_amount = int(typical_daily_demand * RESTOCK_INTERVAL_DAYS * rng.uniform(0.9, 1.2))
            stock = min(initial_stock, stock + restock_amount)
        stock = max(stock - int(sold), 0)
        stock_levels[day_index] = stock

    return stock_levels


def build_backtest_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    raw = load_raw_transactions()
    cleaned = clean_transactions(raw)

    top_products = select_top_products(cleaned)
    cleaned = cleaned[cleaned["product"].isin(top_products)]

    daily = aggregate_daily(cleaned)
    full_date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    full_date_range = full_date_range[:-TRIM_TRAILING_DAYS]

    product_frames = []
    for product in top_products:
        product_df = daily[daily["product"] == product].drop(columns="product")
        product_df = fill_date_gaps(product_df, full_date_range)
        product_df["product"] = product
        product_df["stock_level"] = simulate_stock_levels(product_df["units_sold"].to_numpy(), rng)
        product_frames.append(product_df)

    result = pd.concat(product_frames, ignore_index=True)
    result = result[["date", "product", "units_sold", "price", "stock_level"]]
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["units_sold"] = result["units_sold"].round().astype(int)
    result["price"] = result["price"].round(2)
    return result.sort_values(["product", "date"]).reset_index(drop=True)


def generate():
    dataset = build_backtest_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(dataset)} rows for {dataset['product'].nunique()} products to {OUTPUT_PATH}")
    return dataset


if __name__ == "__main__":
    generate()
