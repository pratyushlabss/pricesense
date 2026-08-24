import argparse
import time

from models.forecast import get_forecaster
from models.optimizer import optimize_price


def run_for_product(product, price=None, stock=None):
    forecaster = get_forecaster()

    current_price, current_stock = forecaster.current_price_and_stock(product)

    start = time.time()
    forecaster.predict_demand(product, current_price)
    train_elapsed = time.time() - start

    price = price if price is not None else current_price
    stock = stock if stock is not None else current_stock

    sample_prices = [round(price * m, 2) for m in (0.7, 0.85, 1.0, 1.15, 1.3)]
    curve_start = time.time()
    curve = forecaster.demand_curve(product, sample_prices)
    curve_elapsed = time.time() - curve_start

    optimize_start = time.time()
    result = optimize_price(product, current_price=price, current_stock=stock)
    optimize_elapsed = time.time() - optimize_start

    print(f"\n=== {product} ===")
    print(f"model ready in {train_elapsed:.3f}s (includes lazy train/load)")
    print(f"current_price={price} current_stock={stock}")
    print("demand curve sample:")
    for p, units in curve:
        print(f"  price={p:8.2f} -> predicted_units={units:6.2f}")
    print(f"demand_curve() took {curve_elapsed:.3f}s")
    print(f"optimize_price() took {optimize_elapsed:.3f}s")
    print(f"suggested_price={result['suggested_price']} "
          f"revenue_change={result['projected_revenue_change_pct']}%")
    print(f"explanation: {result['explanation']}")

    total_elapsed = train_elapsed + curve_elapsed + optimize_elapsed
    status = "OK" if total_elapsed < 5 else "SLOW"
    print(f"[{status}] total time for product: {total_elapsed:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Quick terminal test for forecast + optimizer")
    parser.add_argument("--product", help="Run for a single product only")
    parser.add_argument("--price", type=float, default=None)
    parser.add_argument("--stock", type=int, default=None)
    args = parser.parse_args()

    forecaster = get_forecaster()
    products = [args.product] if args.product else forecaster.list_products()

    for product in products:
        run_for_product(product, price=args.price, stock=args.stock)


if __name__ == "__main__":
    main()
