CONSTRAINT_MESSAGES = {
    "insufficient_data": "ℹ️ Not enough price-variation history to model demand sensitivity for this product — holding current price.",
    "out_of_stock": "🔴 Out of stock — no price suggestion possible until restocked.",
    "stock_constrained": "⚠️ Stock-runway constraint: no price in the ±30% search range keeps stock lasting the minimum days without breaching the cost floor.",
    "already_optimal": "✅ No constraint applied — current price is already the revenue-maximizing price in the ±30% range.",
    "price_change_suggested": "✅ No constraint applied — this is the unconstrained revenue-maximizing price within ±30%.",
}


def classify_panel_state(optimize_data):
    constraint = optimize_data["constraint_hit"]
    if constraint == "insufficient_price_sensitivity_data":
        return "insufficient_data"
    if constraint == "no_feasible_price_found":
        return "out_of_stock" if optimize_data["current_stock"] == 0 else "stock_constrained"
    if abs(optimize_data["suggested_price"] - optimize_data["current_price"]) < 0.01:
        return "already_optimal"
    return "price_change_suggested"


def build_headline(optimize_data, panel_state):
    current_price = optimize_data["current_price"]
    if panel_state != "price_change_suggested":
        return f"Hold current price at ₹{current_price:.2f}"

    suggested_price = optimize_data["suggested_price"]
    current_units = optimize_data["current_predicted_units"]
    suggested_units = optimize_data["suggested_predicted_units"]
    revenue_change_pct = optimize_data["projected_revenue_change_pct"]
    demand_change_pct = (
        (suggested_units - current_units) / current_units * 100 if current_units > 0 else 0.0
    )
    return (
        f"₹{current_price:.2f} → ₹{suggested_price:.2f}: "
        f"{demand_change_pct:+.0f}% demand, {revenue_change_pct:+.1f}% revenue"
    )


def classify_price_sensitivity(curve_df, panel_state):
    if panel_state == "insufficient_data":
        return "No signal (insufficient data)", 0.0

    price_mid = curve_df["price"].mean()
    units_mid = curve_df["predicted_units"].mean()
    if price_mid <= 0 or units_mid <= 0:
        return "Unknown", 0.0

    price_spread_pct = (curve_df["price"].max() - curve_df["price"].min()) / price_mid
    units_spread_pct = (curve_df["predicted_units"].max() - curve_df["predicted_units"].min()) / units_mid
    if price_spread_pct <= 0:
        return "Unknown", 0.0

    elasticity_index = units_spread_pct / price_spread_pct
    if elasticity_index >= 1.2:
        label = "High price sensitivity"
    elif elasticity_index >= 0.4:
        label = "Medium price sensitivity"
    else:
        label = "Low price sensitivity"
    return label, elasticity_index
