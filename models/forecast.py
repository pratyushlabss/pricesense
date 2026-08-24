import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_PATH = DATA_DIR / "synthetic_sales_data.csv"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
FEATURE_COLUMNS = ["day_of_week", "month", "price", "lag_7", "lag_30"]
LAG_WARMUP_DAYS = 30
DEMAND_CURVE_DEFAULT_STEPS = 20


def resolve_data_path(data_path):
    path = Path(data_path)
    if path.parent == Path("."):
        return DATA_DIR / path.name
    return path


def load_sales_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path, parse_dates=["date"])
    return df.sort_values(["product", "date"]).reset_index(drop=True)


def build_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["lag_7"] = (
        df.groupby("product")["units_sold"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
    )
    df["lag_30"] = (
        df.groupby("product")["units_sold"]
        .transform(lambda s: s.shift(1).rolling(window=30, min_periods=1).mean())
    )
    return df


class DemandForecaster:
    def __init__(self, data_path=DATA_PATH, artifacts_dir=ARTIFACTS_DIR):
        self.data_path = resolve_data_path(data_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._models = {}
        self._fallback_averages = {}
        self._latest_snapshot = {}
        self._raw_df = None

    def _ensure_data_loaded(self):
        if self._raw_df is None:
            raw_df = load_sales_data(self.data_path)
            self._raw_df = build_features(raw_df)

    def list_products(self):
        self._ensure_data_loaded()
        return sorted(self._raw_df["product"].unique().tolist())

    def _model_path(self, product):
        safe_name = product.replace("/", "_").replace(" ", "_")
        return self.artifacts_dir / f"{safe_name}.joblib"

    def _snapshot_from_last_row(self, product_df):
        last_row = product_df.iloc[-1]
        lag_7 = last_row["lag_7"]
        lag_30 = last_row["lag_30"]
        return {
            "day_of_week": int((last_row["day_of_week"] + 1) % 7),
            "month": int((last_row["date"] + pd.Timedelta(days=1)).month),
            "lag_7": float(lag_7) if pd.notna(lag_7) else 0.0,
            "lag_30": float(lag_30) if pd.notna(lag_30) else 0.0,
            "last_date": last_row["date"],
        }

    def _train_product(self, product):
        self._ensure_data_loaded()
        product_df = self._raw_df[self._raw_df["product"] == product]
        trainable_df = product_df.dropna(subset=FEATURE_COLUMNS)

        if len(trainable_df) < LAG_WARMUP_DAYS:
            fallback_average = float(product_df["units_sold"].mean()) if len(product_df) else 0.0
            snapshot = self._snapshot_from_last_row(product_df) if len(product_df) else None
            joblib.dump(
                {"model": None, "average": fallback_average, "snapshot": snapshot},
                self._model_path(product),
            )
            self._models[product] = None
            self._fallback_averages[product] = fallback_average
            self._latest_snapshot[product] = snapshot
            return None

        X = trainable_df[FEATURE_COLUMNS]
        y = trainable_df["units_sold"]

        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.08,
            random_state=42,
        )
        model.fit(X, y)

        snapshot = self._snapshot_from_last_row(trainable_df)
        joblib.dump({"model": model, "average": None, "snapshot": snapshot}, self._model_path(product))
        self._models[product] = model
        self._latest_snapshot[product] = snapshot
        return model

    def _get_model(self, product):
        if product in self._models:
            return self._models[product]

        model_path = self._model_path(product)
        if model_path.exists():
            cached = joblib.load(model_path)
            self._models[product] = cached["model"]
            self._latest_snapshot[product] = cached["snapshot"]
            if cached["model"] is None:
                self._fallback_averages[product] = cached["average"]
            return cached["model"]

        return self._train_product(product)

    def fit(self, data_path=None):
        if data_path is not None:
            self.data_path = resolve_data_path(data_path)
            self._raw_df = None
        self._ensure_data_loaded()
        for product in self.list_products():
            self._train_product(product)
        return self

    train_all = fit

    def predict_demand(self, product, price):
        model = self._get_model(product)
        if model is None:
            return max(0.0, self._fallback_averages.get(product, 0.0))

        snapshot = self._latest_snapshot[product]
        features = pd.DataFrame([{
            "day_of_week": snapshot["day_of_week"],
            "month": snapshot["month"],
            "price": price,
            "lag_7": snapshot["lag_7"],
            "lag_30": snapshot["lag_30"],
        }])
        prediction = model.predict(features[FEATURE_COLUMNS])[0]
        return max(0.0, float(prediction))

    def demand_curve(self, product, price_range, steps=DEMAND_CURVE_DEFAULT_STEPS):
        if isinstance(price_range, tuple) and len(price_range) == 2:
            low, high = price_range
            prices = np.linspace(low, high, steps)
        else:
            prices = price_range
        return [(round(float(price), 2), round(self.predict_demand(product, price), 2)) for price in prices]

    def current_price_and_stock(self, product):
        self._ensure_data_loaded()
        last_row = self._raw_df[self._raw_df["product"] == product].iloc[-1]
        return float(last_row["price"]), int(last_row["stock_level"])


_default_forecaster = None


def get_forecaster():
    global _default_forecaster
    if _default_forecaster is None:
        _default_forecaster = DemandForecaster()
    return _default_forecaster


def predict_demand(product, price):
    return get_forecaster().predict_demand(product, price)


def demand_curve(product, price_range, steps=DEMAND_CURVE_DEFAULT_STEPS):
    return get_forecaster().demand_curve(product, price_range, steps=steps)


if __name__ == "__main__":
    import time

    forecaster = DemandForecaster()

    fit_start = time.time()
    forecaster.fit("synthetic_sales_data.csv")
    fit_elapsed = time.time() - fit_start
    products = forecaster.list_products()
    print(f"Trained {len(products)} product models in {fit_elapsed:.3f}s "
          f"({fit_elapsed / len(products):.3f}s/product avg)")

    product = products[0]
    current_price, current_stock = forecaster.current_price_and_stock(product)
    price_range = (current_price * 0.7, current_price * 1.3)

    curve_start = time.time()
    curve = forecaster.demand_curve(product, price_range, steps=20)
    curve_elapsed = time.time() - curve_start

    print(f"\nDemand curve for '{product}' (current_price={current_price:.2f}, "
          f"stock={current_stock}) computed in {curve_elapsed:.3f}s")
    for price, units in curve:
        print(f"  price={price:8.2f} -> predicted_units={units:6.2f}")
