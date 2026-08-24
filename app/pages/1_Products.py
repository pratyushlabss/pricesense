import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
while not (_project_root / "requirements.txt").exists() and _project_root != _project_root.parent:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import streamlit as st

from app.api_client import fetch_forecast, fetch_optimize, fetch_what_if_point
from app.common import bootstrap, init_page
from app.components import (
    render_demand_chart,
    render_empty_state,
    render_guardrails_card,
    render_pricing_health,
    render_product_comparison,
    render_product_header,
    render_product_selector,
    render_recommendation_card,
    render_section_header,
    render_what_if_controls,
    render_what_if_result,
    render_why_this_price,
)
from app.insights import build_summary_row, guardrails_config, pricing_health_score
from models.explainability import classify_panel_state, classify_price_sensitivity

init_page("Products")
products = bootstrap()

if products is not None:
    st.caption(f"Active dataset: **{st.session_state.dataset_label}**")

    selected_product = render_product_selector(products, key="selected_product")

    forecast_data, forecast_error, _ = fetch_forecast(selected_product, st.session_state.dataset_id)
    optimize_data, optimize_error, _ = fetch_optimize(selected_product, st.session_state.dataset_id)

    if forecast_error or optimize_error:
        render_empty_state("Could not load this product", forecast_error or optimize_error)
    else:
        curve_df = pd.DataFrame(forecast_data["curve"])
        current_price = optimize_data["current_price"]
        panel_state = classify_panel_state(optimize_data)

        render_product_header(selected_product, optimize_data)

        render_section_header("Recommendation")
        rec_col, health_col = st.columns([3, 1])
        with rec_col:
            render_recommendation_card(optimize_data, panel_state)
        with health_col:
            current_row = build_summary_row(selected_product, optimize_data)
            render_pricing_health(pricing_health_score(current_row))

        sensitivity_label, elasticity_index = classify_price_sensitivity(curve_df, panel_state)
        render_why_this_price(optimize_data, panel_state, sensitivity_label, elasticity_index)

        render_section_header("Demand curve")
        what_if_price = render_what_if_controls(
            current_price, optimize_data["suggested_price"],
            key_prefix=f"whatif_{st.session_state.dataset_id}_{selected_product}",
        )

        curve_prices = curve_df["price"].to_numpy()
        curve_units = curve_df["predicted_units"].to_numpy()
        what_if_error = None
        if abs(what_if_price - current_price) < 0.01:
            what_if_units = optimize_data["current_predicted_units"]
        elif curve_prices.min() <= what_if_price <= curve_prices.max():
            what_if_units = float(np.interp(what_if_price, curve_prices, curve_units))
        else:
            what_if_data, what_if_error, _ = fetch_what_if_point(
                selected_product, st.session_state.dataset_id, what_if_price,
            )
            what_if_units = what_if_data["what_if"]["predicted_units"] if what_if_data else None

        what_if_marker = (what_if_price, what_if_units) if what_if_units is not None else None
        render_demand_chart(selected_product, curve_df, optimize_data, panel_state, what_if=what_if_marker)
        render_what_if_result(current_price, what_if_price, what_if_units, what_if_error, optimize_data)

        render_section_header("Pricing guardrails")
        render_guardrails_card(guardrails_config())

        render_section_header("Compare products", "Select two or more products to compare side by side.")
        compare_products = st.multiselect(
            "Products to compare", products, default=[selected_product], key="compare_products",
        )
        compare_rows = []
        for compare_product in compare_products:
            compare_optimize_data, compare_error, _ = fetch_optimize(compare_product, st.session_state.dataset_id)
            if compare_error:
                continue
            compare_rows.append(build_summary_row(compare_product, compare_optimize_data))
        render_product_comparison(compare_rows)
