import time
from pathlib import Path

import pandas as pd

from models.forecast import DemandForecaster
import models.optimizer as optimizer_module
from models.optimizer import optimize_price

SCRATCH_DIR = Path("/private/tmp/claude-501/-Users-pratyush-pricesense/c27989f1-2176-440e-a2b0-ee2f378744db/scratchpad")


def with_forecaster(forecaster, fn):
    original = optimizer_module.get_forecaster
    optimizer_module.get_forecaster = lambda: forecaster
    try:
        return fn()
    finally:
        optimizer_module.get_forecaster = original


def test_insufficient_history_holds_price():
    df = pd.read_csv("data/synthetic_sales_data.csv", parse_dates=["date"])
    sparse = df[df["product"] == "Yoga Mat"].tail(10)
    tiny_path = SCRATCH_DIR / "tiny_sales_optimizer_test.csv"
    sparse.to_csv(tiny_path, index=False)

    forecaster = DemandForecaster(
        data_path=tiny_path,
        artifacts_dir=SCRATCH_DIR / "artifacts_tiny_optimizer_test",
    )
    result = with_forecaster(forecaster, lambda: optimize_price("Yoga Mat"))

    assert result["constraint_hit"] == "insufficient_price_sensitivity_data", result
    assert result["suggested_price"] == result["current_price"], result
    assert result["projected_revenue_change_pct"] == 0.0, result
    print("[PASS] insufficient-history product holds current price:", result["explanation"])


def test_stockout_gives_restock_explanation():
    forecaster = DemandForecaster()
    result = with_forecaster(
        forecaster,
        lambda: optimize_price("Wireless Earbuds", current_stock=0),
    )
    assert result["constraint_hit"] == "no_feasible_price_found", result
    assert "restock" in result["explanation"].lower(), result
    print("[PASS] stockout product returns restock explanation:", result["explanation"])


def test_search_range_and_constraints_applied():
    forecaster = DemandForecaster()
    current_price, current_stock = forecaster.current_price_and_stock("Smartwatch")
    result = with_forecaster(forecaster, lambda: optimize_price("Smartwatch"))

    low = current_price * 0.7
    high = current_price * 1.3
    assert low - 0.01 <= result["suggested_price"] <= high + 0.01, result
    assert result["suggested_price"] >= result["floor_price"] - 0.01, result
    if result["days_of_stock_at_suggested_price"] is not None:
        assert result["days_of_stock_at_suggested_price"] >= optimizer_module.DEFAULT_MIN_DAYS_OF_STOCK - 0.5, result
    print(f"[PASS] Smartwatch suggested_price={result['suggested_price']} within ±30% grid, "
          f"floor={result['floor_price']}, days_of_stock={result['days_of_stock_at_suggested_price']}")


def test_white_hanging_heart_reproduction():
    forecaster = DemandForecaster(data_path="uci_backtest_data.csv")
    result = with_forecaster(
        forecaster,
        lambda: optimize_price("WHITE HANGING HEART T-LIGHT HOLDER"),
    )
    print(f"[INFO] WHITE HANGING HEART T-LIGHT HOLDER: "
          f"current={result['current_price']} suggested={result['suggested_price']} "
          f"revenue_change={result['projected_revenue_change_pct']}% constraint={result['constraint_hit']}")
    print(f"       {result['explanation']}")
    assert result["suggested_price"] > result["current_price"], (
        "expected the stock-runway constraint to push suggested price above current price"
    )


def test_timing_under_5s_both_datasets():
    for label, data_path in [("synthetic", None), ("uci", "uci_backtest_data.csv")]:
        forecaster = DemandForecaster(data_path=data_path) if data_path else DemandForecaster()
        for product in forecaster.list_products():
            start = time.time()
            with_forecaster(forecaster, lambda p=product: optimize_price(p))
            elapsed = time.time() - start
            status = "OK" if elapsed < 5 else "SLOW"
            print(f"[{status}] {label}: {product} took {elapsed:.3f}s")
            assert elapsed < 5, f"{label}/{product} took {elapsed:.3f}s"


if __name__ == "__main__":
    test_insufficient_history_holds_price()
    test_stockout_gives_restock_explanation()
    test_search_range_and_constraints_applied()
    test_white_hanging_heart_reproduction()
    test_timing_under_5s_both_datasets()
    print("\nAll optimizer checks passed.")
