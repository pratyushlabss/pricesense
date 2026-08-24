import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
while not (_project_root / "requirements.txt").exists() and _project_root != _project_root.parent:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.api_client import fetch_optimize
from app.common import bootstrap, init_page
from app.components import render_empty_state, render_insight_feed, render_section_header
from app.insights import build_summary_row, generate_insights, portfolio_stats, top_revenue_opportunity, top_stock_risk

init_page("AI Insights")
st.caption("A retrieval-style summary of the current portfolio's model output — every line below is a direct "
           "readout of real data, not generated text.")

products = bootstrap()


def _go_to_product(product):
    st.session_state.selected_product = product
    st.switch_page("pages/1_Products.py")


if products is not None:
    summary_rows = []
    for product in products:
        optimize_data, optimize_error, _ = fetch_optimize(product, st.session_state.dataset_id)
        if optimize_error:
            continue
        summary_rows.append(build_summary_row(product, optimize_data))

    if not summary_rows:
        render_empty_state("No product data available", "None of the products in this dataset could be analyzed.")
    else:
        stats = portfolio_stats(summary_rows)
        top_opportunity = top_revenue_opportunity(summary_rows)
        top_risk = top_stock_risk(summary_rows)

        render_section_header("Business brief")
        with st.container(border=True):
            st.markdown("**Biggest opportunity**")
            if top_opportunity:
                st.caption(f"{top_opportunity['Product']} — {top_opportunity['Revenue Impact (%)']:+.1f}% "
                           "projected revenue opportunity.")
            else:
                st.caption("No unconstrained revenue opportunity in this dataset right now.")

            st.markdown("**Inventory**")
            if top_risk:
                st.caption(f"{stats['stock_risk_count']} of {stats['total_products']} products are stock-flagged "
                           f"— most urgent: {top_risk['Product']} ({top_risk['Constraint']}).")
            else:
                st.caption("No immediate stock risk detected across the portfolio.")

            st.markdown("**Pricing**")
            if stats["opportunity_count"]:
                st.caption(f"{stats['opportunity_count']} of {stats['total_products']} products appear suitable "
                           f"for a price change, averaging {stats['avg_opportunity_impact']:+.1f}% projected "
                           "revenue impact.")
            else:
                st.caption("No products currently show an unconstrained pricing opportunity.")

            low_signal = [r for r in summary_rows if r["_panel_state"] == "insufficient_data"]
            st.markdown("**Risk**")
            if low_signal:
                names = ", ".join(r["Product"] for r in low_signal[:5])
                st.caption(f"{len(low_signal)} product(s) have insufficient price-variation history to "
                           f"recommend a change with confidence: {names}.")
            else:
                st.caption("No products are currently held back by insufficient historical signal.")

            st.markdown("**Suggested action**")
            if top_opportunity:
                st.caption(f"Test the recommended price for {top_opportunity['Product']} and monitor demand "
                           "response before rolling further changes out across the portfolio.")
            else:
                st.caption("Review the priority table on the Dashboard for the next-best action.")

        render_section_header("All insights", "Every rule-based observation generated from this dataset.")
        render_insight_feed(generate_insights(summary_rows, max_insights=10), on_select=_go_to_product)
