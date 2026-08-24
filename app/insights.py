# Pure derived-insight logic — ranking, scoring, opportunity detection.
# Operates only on already-fetched /optimize response dicts (or the summary
# rows built from them). No HTTP calls, no Streamlit calls: this is the
# "business logic" layer components.py explicitly stays free of.

from models.explainability import classify_panel_state
from models.optimizer import DEFAULT_MARGIN_FLOOR_PCT, DEFAULT_MIN_DAYS_OF_STOCK, PRICE_SEARCH_RANGE_PCT

CONSTRAINT_LABELS = {
    "price_change_suggested": "None",
    "already_optimal": "Already optimal",
    "insufficient_data": "Insufficient data",
    "out_of_stock": "Out of stock",
    "stock_constrained": "Stock constrained",
}
STOCK_RISK_STATES = {"out_of_stock", "stock_constrained"}


def build_summary_row(product, optimize_data):
    panel_state = classify_panel_state(optimize_data)
    current_price = optimize_data["current_price"]
    suggested_price = optimize_data["suggested_price"]
    current_units = optimize_data["current_predicted_units"]
    suggested_units = optimize_data["suggested_predicted_units"]
    revenue_impact_pct = optimize_data["projected_revenue_change_pct"]

    price_change_pct = (
        (suggested_price - current_price) / current_price * 100 if current_price > 0 else 0.0
    )
    demand_change_pct = (
        (suggested_units - current_units) / current_units * 100 if current_units > 0 else 0.0
    )
    demand_sensitivity = (
        abs(demand_change_pct / price_change_pct) if abs(price_change_pct) > 0.01 else None
    )

    return {
        "Product": product,
        "Current Price": current_price,
        "Suggested Price": suggested_price,
        "Forecasted Demand": current_units,
        "Revenue Impact (%)": revenue_impact_pct,
        "Demand Change (%)": demand_change_pct,
        "Stock": optimize_data["current_stock"],
        "Constraint": CONSTRAINT_LABELS[panel_state],
        "Priority": priority_level(panel_state, revenue_impact_pct),
        "_panel_state": panel_state,
        "_demand_sensitivity": demand_sensitivity,
    }


def priority_level(panel_state, revenue_impact_pct):
    """Categorical priority — a transparent function of panel_state (the
    optimizer's own constraint classification) and revenue-impact magnitude.
    Not a fabricated score, not a confidence measure: it says how much
    attention a product's *opportunity* deserves, nothing about how certain
    the model is.

    - Constrained: stock is currently blocking a price recommendation
      (out of stock or stock-runway limited).
    - Already Optimized: the optimizer found the current price is already
      the best price in its search range.
    - High/Medium Priority: an unconstrained price change is recommended,
      bucketed by the size of its projected revenue impact (>=10% / >=3%).
    - Monitor: everything else — either the opportunity is small (<3%
      revenue impact) or there isn't enough price-variation history to
      recommend a change with any confidence.
    """
    if panel_state in STOCK_RISK_STATES:
        return "Constrained"
    if panel_state == "already_optimal":
        return "Already Optimized"
    if panel_state == "price_change_suggested" and abs(revenue_impact_pct) >= 10:
        return "High Priority"
    if panel_state == "price_change_suggested" and abs(revenue_impact_pct) >= 3:
        return "Medium Priority"
    return "Monitor"


def portfolio_stats(summary_rows):
    total = len(summary_rows)
    opportunities = [r for r in summary_rows if r["_panel_state"] == "price_change_suggested" and r["Revenue Impact (%)"] > 0]
    stock_risk = [r for r in summary_rows if r["_panel_state"] in STOCK_RISK_STATES]
    avg_opportunity_impact = (
        sum(r["Revenue Impact (%)"] for r in opportunities) / len(opportunities) if opportunities else 0.0
    )
    return {
        "total_products": total,
        "opportunity_count": len(opportunities),
        "avg_opportunity_impact": avg_opportunity_impact,
        "stock_risk_count": len(stock_risk),
    }


def top_revenue_opportunity(summary_rows):
    candidates = [r for r in summary_rows if r["_panel_state"] == "price_change_suggested"]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["Revenue Impact (%)"])


def top_demand_sensitivity(summary_rows):
    candidates = [r for r in summary_rows if r["_demand_sensitivity"] is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["_demand_sensitivity"])


def top_stock_risk(summary_rows):
    out_of_stock = [r for r in summary_rows if r["_panel_state"] == "out_of_stock"]
    if out_of_stock:
        return out_of_stock[0]
    constrained = [r for r in summary_rows if r["_panel_state"] == "stock_constrained"]
    if constrained:
        return min(constrained, key=lambda r: r["Stock"])
    return None


def explain_opportunity(row):
    """A one-sentence, data-derived explanation of why a recommended price
    change is (or isn't) a net opportunity — built from the same demand/price
    change figures already shown elsewhere, not a separate fabricated claim."""
    revenue_impact = row["Revenue Impact (%)"]
    demand_change = row["Demand Change (%)"]
    price_change = (
        (row["Suggested Price"] - row["Current Price"]) / row["Current Price"] * 100
        if row["Current Price"] > 0 else 0.0
    )

    if row["_panel_state"] != "price_change_suggested":
        return None

    direction = "increase" if price_change >= 0 else "decrease"
    demand_direction = "reduction" if demand_change < 0 else "increase"

    if revenue_impact >= 0:
        return (
            f"PriceSense estimates the expected demand {demand_direction} ({demand_change:+.1f}%) is smaller "
            f"than the effect of the {abs(price_change):.1f}% price {direction}, creating a net positive "
            f"projected revenue opportunity of {revenue_impact:+.1f}%."
        )
    return (
        f"PriceSense estimates the expected demand {demand_direction} ({demand_change:+.1f}%) outweighs the "
        f"effect of the {abs(price_change):.1f}% price {direction}, projecting a {revenue_impact:.1f}% revenue "
        f"change — included here for completeness, not as a recommended action."
    )


def pricing_health_score(row):
    """A transparent 0-100 composite of three independently real signals —
    not a fabricated confidence score, and deliberately not labeled as one.

    - Constraint health (0-40): is a price recommendation even actionable
      right now? Unconstrained/optimal = 40, stock-runway constrained = 20,
      out of stock or insufficient data = 0 (no legitimate basis to score
      higher — there either isn't a usable recommendation or there isn't
      enough history to trust one).
    - Signal quality (0-30): does this product have a defined price-
      sensitivity reading at all (i.e. enough price variation in its
      history for the model to relate price to demand)? Binary: 30 or 0.
    - Opportunity magnitude (0-30): the size of the projected revenue
      impact, capped at 30 points (a 30%+ opportunity maxes this out).

    This says how actionable and well-supported a recommendation is — it
    says nothing about the model's statistical confidence, because no such
    calibrated metric exists in this system.
    """
    panel_state = row["_panel_state"]
    if panel_state in ("price_change_suggested", "already_optimal"):
        constraint_health = 40
    elif panel_state == "stock_constrained":
        constraint_health = 20
    else:
        constraint_health = 0

    signal_quality = 30 if row["_demand_sensitivity"] is not None else 0
    opportunity_magnitude = min(abs(row["Revenue Impact (%)"]), 30)

    return round(constraint_health + signal_quality + opportunity_magnitude)


def guardrails_config():
    """The pricing engine's actual enforced constraints, read directly from
    models/optimizer.py's own constants — not a separate, possibly-stale
    copy. Read-only: this dashboard section displays what IS enforced, it
    does not claim to let the user edit it, because doing so safely would
    require new API surface this pass didn't add."""
    return [
        {
            "label": "Maximum price change (either direction)",
            "value": f"±{PRICE_SEARCH_RANGE_PCT * 100:.0f}%",
            "description": "The optimizer only searches prices within this range of the current price.",
        },
        {
            "label": "Minimum stock runway",
            "value": f"{DEFAULT_MIN_DAYS_OF_STOCK} days",
            "description": "A price is only recommended if projected stock lasts at least this long.",
        },
        {
            "label": "Minimum margin floor",
            "value": f"{DEFAULT_MARGIN_FLOOR_PCT * 100:.0f}% of current price",
            "description": "Without an explicit cost price, the floor is set to this fraction below current price.",
        },
    ]


def bubble_candidates(summary_rows):
    """Products with a defined price-sensitivity signal — the only ones that
    can be plotted on the Opportunity Map's sensitivity axis honestly."""
    return [r for r in summary_rows if r["_demand_sensitivity"] is not None]


def generate_insights(summary_rows, max_insights=5):
    """Rule-based observations over already-computed summary rows. Every
    insight is a direct readout of real model output — no generation, no
    fabricated reasoning. Returns at most `max_insights`, ordered by how
    actionable they are (stock risk and large opportunities first)."""
    insights = []

    top_opportunity = top_revenue_opportunity(summary_rows)
    if top_opportunity and top_opportunity["Revenue Impact (%)"] >= 5:
        insights.append({
            "text": f"{top_opportunity['Product']} shows a {top_opportunity['Revenue Impact (%)']:+.0f}% "
                    f"projected revenue opportunity at the recommended price.",
            "product": top_opportunity["Product"],
        })

    out_of_stock = [r for r in summary_rows if r["_panel_state"] == "out_of_stock"]
    for row in out_of_stock[:2]:
        insights.append({
            "text": f"{row['Product']} is out of stock — no price recommendation is possible until restocked.",
            "product": row["Product"],
        })

    constrained = [r for r in summary_rows if r["_panel_state"] == "stock_constrained"]
    for row in constrained[:2]:
        insights.append({
            "text": f"Current inventory may constrain additional demand for {row['Product']} — its price "
                    f"search was narrowed to protect stock runway.",
            "product": row["Product"],
        })

    sensitive = top_demand_sensitivity(summary_rows)
    if sensitive and sensitive["_demand_sensitivity"] >= 1.5:
        insights.append({
            "text": f"{sensitive['Product']} shows strong demand sensitivity to price — small price changes "
                    f"move demand substantially.",
            "product": sensitive["Product"],
        })

    insensitive_candidates = [r for r in bubble_candidates(summary_rows) if r["_demand_sensitivity"] < 0.3]
    if insensitive_candidates:
        row = min(insensitive_candidates, key=lambda r: r["_demand_sensitivity"])
        insights.append({
            "text": f"Demand for {row['Product']} appears relatively insensitive to price changes across "
                    f"the evaluated range.",
            "product": row["Product"],
        })

    already_optimal_count = sum(1 for r in summary_rows if r["_panel_state"] == "already_optimal")
    if already_optimal_count:
        insights.append({
            "text": f"{already_optimal_count} of {len(summary_rows)} products are already at their "
                    f"revenue-maximizing price — no action needed there.",
            "product": None,
        })

    return insights[:max_insights]
