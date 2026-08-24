import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "synthetic_sales_data.csv"
NUM_DAYS = 730
RANDOM_SEED = 42

PRODUCTS = [
    {"name": "Wireless Earbuds", "base_price": 1499, "base_demand": 40, "elasticity": 1.8},
    {"name": "Smartwatch", "base_price": 2999, "base_demand": 25, "elasticity": 1.5},
    {"name": "Bluetooth Speaker", "base_price": 999, "base_demand": 35, "elasticity": 1.6},
    {"name": "Laptop Backpack", "base_price": 799, "base_demand": 50, "elasticity": 1.2},
    {"name": "Yoga Mat", "base_price": 499, "base_demand": 60, "elasticity": 1.0},
    {"name": "Air Fryer", "base_price": 3499, "base_demand": 15, "elasticity": 2.0},
    {"name": "Office Chair", "base_price": 5999, "base_demand": 8, "elasticity": 1.3},
    {"name": "Running Shoes", "base_price": 2499, "base_demand": 30, "elasticity": 1.7},
]

WEEKEND_MULTIPLIER = 1.15
FESTIVE_MONTHS = {10: 1.2, 11: 1.35, 12: 1.45}
PROMO_PROBABILITY = 0.08
PROMO_DISCOUNT_RANGE = (0.10, 0.25)
DAILY_PRICE_NOISE_STD = 0.02
RESTOCK_INTERVAL_DAYS = 14


def seasonal_multiplier(date):
    day_of_week_mult = WEEKEND_MULTIPLIER if date.dayofweek >= 5 else 1.0
    month_mult = FESTIVE_MONTHS.get(date.month, 1.0)
    yearly_wave = 1.0 + 0.05 * np.sin(2 * np.pi * date.dayofyear / 365)
    return day_of_week_mult * month_mult * yearly_wave


def simulate_price(base_price, rng, in_promo):
    noise = rng.normal(0, DAILY_PRICE_NOISE_STD)
    price = base_price * (1 + noise)
    if in_promo:
        discount = rng.uniform(*PROMO_DISCOUNT_RANGE)
        price *= (1 - discount)
    return round(max(price, base_price * 0.5), 2)


def simulate_product(product, dates, rng):
    base_price = product["base_price"]
    base_demand = product["base_demand"]
    elasticity = product["elasticity"]

    stock_level = base_demand * 30
    restock_target = base_demand * 25
    in_promo = False
    promo_days_left = 0

    rows = []
    for day_index, date in enumerate(dates):
        if promo_days_left > 0:
            promo_days_left -= 1
            in_promo = promo_days_left > 0
        elif rng.random() < PROMO_PROBABILITY:
            in_promo = True
            promo_days_left = rng.integers(2, 6)

        price = simulate_price(base_price, rng, in_promo)

        price_ratio = price / base_price
        expected_demand = base_demand * (price_ratio ** (-elasticity)) * seasonal_multiplier(date)
        noisy_demand = expected_demand + rng.normal(0, base_demand * 0.15)
        desired_units = max(0, round(noisy_demand))

        units_sold = int(min(desired_units, stock_level))
        stock_level -= units_sold

        if day_index % RESTOCK_INTERVAL_DAYS == 0 and day_index > 0:
            stock_level += restock_target
        if stock_level <= base_demand * 2:
            stock_level += restock_target

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "product": product["name"],
            "units_sold": units_sold,
            "price": price,
            "stock_level": stock_level,
        })

    return rows


def generate():
    rng = np.random.default_rng(RANDOM_SEED)
    end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=NUM_DAYS - 1)
    dates = pd.date_range(start_date, end_date, freq="D")

    all_rows = []
    for product in PRODUCTS:
        all_rows.extend(simulate_product(product, dates, rng))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows for {len(PRODUCTS)} products to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    generate()
