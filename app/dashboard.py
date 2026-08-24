# Run: streamlit run app/dashboard.py
# Requires the FastAPI backend running first: uvicorn api.main:app --reload --port 8000
# Other pages live in app/pages/ (Streamlit's classic multi-page convention).

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
while not (_project_root / "requirements.txt").exists() and _project_root != _project_root.parent:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.api_client import check_health, fetch_optimize
from app.common import bootstrap, init_page
from app.components import (
    render_empty_state,
    render_hero_header,
    render_insight_feed,
    render_kpi_cards,
    render_opportunity_map,
    render_opportunity_summary,
    render_pipeline_walkthrough,
    render_portfolio_health_chart,
    render_priority_table,
    render_revenue_chart,
    render_section_header,
    render_stock_vs_opportunity_chart,
    render_top_opportunity_hero,
)
from app.insights import (
    build_summary_row,
    generate_insights,
    portfolio_stats,
    top_demand_sensitivity,
    top_revenue_opportunity,
    top_stock_risk,
)

import streamlit as st

init_page("Dashboard", show_title=False)
render_hero_header(st.session_state.dataset_label, check_health())


def _go_to_product(product):
    st.session_state.selected_product = product
    st.switch_page("pages/1_Products.py")


products = bootstrap()

if products is not None:
    summary_rows = []
    for product in products:
        optimize_data, optimize_error, _ = fetch_optimize(product, st.session_state.dataset_id)
        if optimize_error:
            st.warning(f"Could not load optimization for '{product}': {optimize_error}")
            continue
        summary_rows.append(build_summary_row(product, optimize_data))

    if not summary_rows:
        render_empty_state(
            "No product data available",
            "None of the products in this dataset could be analyzed.",
        )
    else:
        stats = portfolio_stats(summary_rows)
        top_opportunity = top_revenue_opportunity(summary_rows)

        render_pipeline_walkthrough(st.session_state.dataset_label, len(products), stats, top_opportunity)

        render_kpi_cards(stats)

        render_section_header("Biggest opportunity", "The single most impactful, unconstrained recommendation.")
        render_top_opportunity_hero(top_opportunity, on_select=_go_to_product)

        render_section_header("Opportunities", "Where the biggest revenue, sensitivity, and stock signals are.")
        render_opportunity_summary(
            top_opportunity,
            top_demand_sensitivity(summary_rows),
            top_stock_risk(summary_rows),
            on_select=_go_to_product,
        )

        render_section_header("Insights", "Automatically generated from the current portfolio's model output.")
        render_insight_feed(generate_insights(summary_rows), on_select=_go_to_product)

        render_section_header("Portfolio analytics")
        chart_col, donut_col = st.columns([2, 1])
        with chart_col:
            render_revenue_chart(summary_rows)
        with donut_col:
            render_portfolio_health_chart(summary_rows)
        map_col, scatter_col = st.columns(2)
        with map_col:
            render_opportunity_map(summary_rows, on_select=_go_to_product)
        with scatter_col:
            render_stock_vs_opportunity_chart(summary_rows)

        render_section_header("Priority products", "Search, filter, and sort every product in the dataset.")
        render_priority_table(summary_rows)
