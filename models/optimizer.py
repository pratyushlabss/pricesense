import numpy as np
from models.forecast import get_forecaster

DEFAULT_MIN_DAYS_OF_STOCK = 7
DEFAULT_MARGIN_FLOOR_PCT = 0.4
PRICE_SEARCH_RANGE_PCT = 0.3
PRICE_SEARCH_STEPS = 61
FLATNESS_RELATIVE_THRESHOLD = 0.01
FLATNESS_ABSOLUTE_EPSILON = 1e-6


def price_search_grid(current_price):
    low = current_price * (1 - PRICE_SEARCH_RANGE_PCT)
    high = current_price * (1 + PRICE_SEARCH_RANGE_PCT)
    return np.linspace(low, high, PRICE_SEARCH_STEPS)


def resolve_floor_price(current_price, cost_price=None, margin_floor_pct=DEFAULT_MARGIN_FLOOR_PCT):
    if cost_price is not None:
        return cost_price
    return current_price * (1 - margin_floor_pct)


def has_price_sensitivity(curve_units, mean_units):
    spread = max(curve_units) - min(curve_units)
    if mean_units <= 0:
        return spread > FLATNESS_ABSOLUTE_EPSILON
    return spread > max(FLATNESS_ABSOLUTE_EPSILON, FLATNESS_RELATIVE_THRESHOLD * mean_units)


def build_explanation(
    product, current_price, current_units, suggested_price, suggested_units,
    revenue_change_pct, current_stock, min_days_of_stock, constraint_hit,
):
    demand_change_pct = (
        ((suggested_units - current_units) / current_units) * 100 if current_units > 0 else 0.0
    )
    direction = "rise" if demand_change_pct >= 0 else "fall"

    if constraint_hit == "insufficient_price_sensitivity_data":
        return (
            f"Not enough price-variation history for {product} to model how demand responds to price; "
            f"holding current price at ₹{current_price:.0f} rather than optimizing against a flat estimate."
        )

    if constraint_hit == "no_feasible_price_found":
        return (
            f"Even at current price ₹{current_price:.0f}, projected demand for {product} would deplete "
            f"the {current_stock}-unit stock in under {min_days_of_stock} days; no price in the ±30% "
            f"search range keeps it in stock longer without falling below the cost floor. Consider "
            f"restocking soon."
        )

    if abs(suggested_price - current_price) < 0.01:
        return (
            f"Current price of ₹{current_price:.0f} for {product} is already close to optimal; "
            f"no meaningful revenue gain from changing it."
        )

    revenue_direction = "increasing" if revenue_change_pct >= 0 else "decreasing"
    return (
        f"Demand is projected to {direction} {abs(demand_change_pct):.0f}% at ₹{suggested_price:.0f} "
        f"vs current ₹{current_price:.0f}, {revenue_direction} total revenue by {abs(revenue_change_pct):.1f}%."
    )


def build_result(
    product, current_price, current_stock, floor_price, current_units,
    suggested_price, suggested_units, revenue_change_pct, min_days_of_stock, constraint_hit,
):
    explanation = build_explanation(
        product, current_price, current_units, suggested_price, suggested_units,
        revenue_change_pct, current_stock, min_days_of_stock, constraint_hit,
    )
    return {
        "product": product,
        "current_price": round(current_price, 2),
        "current_stock": current_stock,
        "floor_price": round(floor_price, 2),
        "current_predicted_units": round(current_units, 2),
        "suggested_price": round(suggested_price, 2),
        "suggested_predicted_units": round(suggested_units, 2),
        "projected_revenue_change_pct": round(revenue_change_pct, 2),
        "days_of_stock_at_suggested_price": (
            round(current_stock / suggested_units, 1) if suggested_units > 0 else None
        ),
        "constraint_hit": constraint_hit,
        "explanation": explanation,
    }


def optimize_price(
    product,
    current_price=None,
    current_stock=None,
    cost_price=None,
    min_days_of_stock=DEFAULT_MIN_DAYS_OF_STOCK,
    margin_floor_pct=DEFAULT_MARGIN_FLOOR_PCT,
):
    forecaster = get_forecaster()

    if current_price is None or current_stock is None:
        inferred_price, inferred_stock = forecaster.current_price_and_stock(product)
        current_price = current_price if current_price is not None else inferred_price
        current_stock = current_stock if current_stock is not None else inferred_stock

    floor_price = resolve_floor_price(current_price, cost_price, margin_floor_pct)
    current_units = forecaster.predict_demand(product, current_price)
    current_revenue = current_price * min(current_units, current_stock)

    grid = price_search_grid(current_price)
    curve = forecaster.demand_curve(product, grid)
    curve_units = [units for _, units in curve]
    mean_units = sum(curve_units) / len(curve_units) if curve_units else 0.0

    if not has_price_sensitivity(curve_units, mean_units):
        return build_result(
            product, current_price, current_stock, floor_price, current_units,
            current_price, current_units, 0.0, min_days_of_stock,
            "insufficient_price_sensitivity_data",
        )

    candidates = []
    for price, predicted_units in curve:
        if price < floor_price:
            continue

        days_of_stock = current_stock / predicted_units if predicted_units > 0 else float("inf")
        if days_of_stock < min_days_of_stock:
            continue

        revenue = price * min(predicted_units, current_stock)
        candidates.append({
            "price": price,
            "predicted_units": predicted_units,
            "revenue": revenue,
            "days_of_stock": days_of_stock,
        })

    if not candidates:
        suggested_price = current_price
        suggested_units = current_units
        suggested_revenue = current_revenue
        constraint_hit = "no_feasible_price_found"
    else:
        best = max(candidates, key=lambda c: c["revenue"])
        suggested_price = best["price"]
        suggested_units = best["predicted_units"]
        suggested_revenue = best["revenue"]
        constraint_hit = None

    revenue_change_pct = (
        ((suggested_revenue - current_revenue) / current_revenue) * 100 if current_revenue > 0 else 0.0
    )

    return build_result(
        product, current_price, current_stock, floor_price, current_units,
        suggested_price, suggested_units, revenue_change_pct, min_days_of_stock, constraint_hit,
    )


if __name__ == "__main__":
    import time

    forecaster = get_forecaster()
    for product in forecaster.list_products():
        start = time.time()
        result = optimize_price(product)
        elapsed = time.time() - start
        print(f"\n{product} ({elapsed:.3f}s)")
        for key, value in result.items():
            print(f"  {key}: {value}")
