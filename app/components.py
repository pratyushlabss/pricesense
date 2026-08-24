# Pure rendering functions — every component takes already-fetched data
# (API response dicts, dataframes, plain values, or app/insights.py output)
# and renders Streamlit UI. No HTTP calls in here; that's app/api_client.py's
# job. No ranking/scoring logic in here; that's app/insights.py's job.

import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from models.explainability import build_headline

BADGE_STYLES = {
    "neutral": ("#F3F4F6", "#374151"),   # gray — fallback states, not errors
    "warning": ("#FEF3C7", "#92400E"),   # amber — real, actionable constraint
    "positive": ("#D1FAE5", "#065F46"),  # green — unconstrained / optimal
    "negative": ("#FEE2E2", "#991B1B"),  # red — reserved for genuine negative impact
}
PANEL_BADGE_STYLE = {
    "insufficient_data": "neutral",
    "out_of_stock": "neutral",
    "stock_constrained": "warning",
    "already_optimal": "positive",
    "price_change_suggested": "positive",
}
PANEL_BADGE_TEXT = {
    "insufficient_data": "Insufficient signal",
    "out_of_stock": "Out of stock",
    "stock_constrained": "Stock-runway constrained",
    "already_optimal": "Already optimal",
    "price_change_suggested": "Unconstrained optimum",
}


def inject_global_css():
    st.markdown(
        """
        <style>
        @keyframes pricesenseFadeInUp {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
            transition: box-shadow 0.15s ease, transform 0.15s ease;
            animation: pricesenseFadeInUp 0.25s ease-out;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 10px rgba(0,0,0,0.10), 0 2px 4px rgba(0,0,0,0.06);
        }
        [data-testid="stMetricValue"] {
            font-size: 1.75rem;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] {
            color: #6B7280;
            font-size: 0.85rem;
        }
        [data-testid="stButton"] button {
            border-radius: 8px;
            transition: transform 0.1s ease, box-shadow 0.15s ease;
        }
        [data-testid="stButton"] button:hover {
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        [data-testid="stButton"] button:active {
            transform: scale(0.98);
        }
        [data-testid="stDataFrame"] {
            animation: pricesenseFadeInUp 0.25s ease-out;
        }
        .pricesense-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .pricesense-eyebrow {
            color: #6B7280;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }
        .pricesense-subtitle {
            color: #6B7280;
            font-size: 0.95rem;
            margin-top: -0.5rem;
        }
        .pricesense-insight {
            padding: 0.6rem 0;
            border-bottom: 1px solid #F3F4F6;
            animation: pricesenseFadeInUp 0.3s ease-out;
        }
        .pricesense-insight:last-child {
            border-bottom: none;
        }
        @media (prefers-reduced-motion: reduce) {
            div[data-testid="stVerticalBlockBorderWrapper"],
            [data-testid="stDataFrame"],
            .pricesense-insight {
                animation: none;
            }
            [data-testid="stButton"] button {
                transition: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div class="pricesense-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_badge(text, style="neutral"):
    background, color = BADGE_STYLES[style]
    st.markdown(
        f'<span class="pricesense-badge" style="background:{background};color:{color};">{text}</span>',
        unsafe_allow_html=True,
    )


def render_panel_badge(panel_state):
    render_badge(PANEL_BADGE_TEXT[panel_state], PANEL_BADGE_STYLE[panel_state])


def render_animated_metric(label, target_value, decimals=0, prefix="", suffix="", show_sign=False, delta_text=None, delta_style="neutral"):
    """A st.metric look-alike whose headline number counts up once on load,
    via st.components.v1.html (a first-party Streamlit extension point, not
    a fragile DOM hack). Runs once per render — reruns re-trigger it, since
    Streamlit remounts the iframe each time, same as any other element."""
    element_id = f"metric-{uuid.uuid4().hex}"
    sign_expr = 'value >= 0 ? "+" : ""' if show_sign else '""'
    delta_color = BADGE_STYLES[delta_style][1] if delta_text else "transparent"
    delta_html = f'<div style="font-size:0.8rem;font-weight:600;color:{delta_color};margin-top:2px;">{delta_text or ""}</div>'

    components.html(
        f"""
        <div style="font-family:'Source Sans Pro',sans-serif;background:transparent;">
            <div style="color:#6B7280;font-size:0.85rem;">{label}</div>
            <div id="{element_id}" style="font-size:1.75rem;font-weight:700;color:#111827;">
                {prefix}0{suffix}
            </div>
            {delta_html}
        </div>
        <script>
        (function() {{
            const el = document.getElementById("{element_id}");
            const target = {target_value};
            const duration = 500;
            const start = performance.now();
            function frame(now) {{
                const progress = Math.min((now - start) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const value = target * eased;
                const sign = {sign_expr};
                el.textContent = "{prefix}" + sign + value.toFixed({decimals}) + "{suffix}";
                if (progress < 1) {{
                    requestAnimationFrame(frame);
                }} else {{
                    const finalSign = target >= 0 ? ({str(show_sign).lower()} ? "+" : "") : "";
                    el.textContent = "{prefix}" + finalSign + target.toFixed({decimals}) + "{suffix}";
                }}
            }}
            requestAnimationFrame(frame);
        }})();
        </script>
        """,
        height=70,
    )


def render_health_banner(is_healthy):
    if not is_healthy:
        st.error(
            "Cannot reach the PriceSense API. Some sections below may not load. "
            "Start it with: `uvicorn api.main:app --reload --port 8000`"
        )


def render_hero_header(dataset_label, is_healthy):
    st.markdown("# PriceSense")
    st.markdown('<div class="pricesense-subtitle">Your pricing intelligence command center</div>', unsafe_allow_html=True)
    st.markdown(
        "Turn historical sales data into smarter pricing decisions — demand modeling, revenue "
        "simulation, and explainable recommendations, without a data science team."
    )
    status_color, status_text = ("#059669", "Pricing engine online") if is_healthy else ("#991B1B", "Pricing engine offline")
    st.markdown(
        f'<div style="font-size:0.9rem;color:#374151;">{dataset_label}'
        f'&nbsp;&nbsp;&middot;&nbsp;&nbsp;'
        f'<span style="color:{status_color};">&#9679;</span> {status_text}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(title, message, action_label=None):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(message)
        if action_label:
            st.caption(action_label)


def render_upload_widget():
    return st.file_uploader("Upload your own sales CSV", type=["csv"])


def render_upload_confirmation(upload_data):
    with st.container(border=True):
        render_badge("Validation passed", "positive")
        st.markdown("**Dataset loaded**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{upload_data['row_count']:,}")
        col2.metric("Products", len(upload_data["products"]))
        col3.metric(
            "Date range",
            f"{upload_data['date_range']['start']} to {upload_data['date_range']['end']}",
        )


def render_upload_errors(error_message, error_fields):
    if error_fields:
        st.error("Upload rejected. Fix the following and try again:")
        for field_error in error_fields:
            st.markdown(f"- {field_error}")
    else:
        st.error(f"Upload rejected: {error_message}")


def render_dataset_info_card(dataset_id, products, dataset_label):
    with st.container(border=True):
        st.markdown("**Current dataset**")
        col1, col2 = st.columns(2)
        col1.metric("Name", dataset_label)
        col2.metric("Products", len(products))
        st.caption(f"Dataset ID: {dataset_id}")


def render_kpi_cards(stats):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            render_animated_metric("Products analyzed", stats["total_products"], decimals=0)
    with col2:
        with st.container(border=True):
            render_animated_metric(
                "Price opportunities", stats["opportunity_count"], decimals=0,
                delta_text=f"of {stats['total_products']}" if stats["opportunity_count"] else "none found",
                delta_style="neutral",
            )
    with col3:
        with st.container(border=True):
            if stats["opportunity_count"]:
                render_animated_metric(
                    "Avg. opportunity impact", stats["avg_opportunity_impact"],
                    decimals=1, suffix="%", show_sign=True,
                    delta_text=f"{stats['avg_opportunity_impact']:+.1f}%",
                    delta_style="positive" if stats["avg_opportunity_impact"] >= 0 else "negative",
                )
            else:
                st.metric("Avg. opportunity impact", "N/A")
    with col4:
        with st.container(border=True):
            render_animated_metric(
                "Stock risk", stats["stock_risk_count"], decimals=0,
                delta_text=f"of {stats['total_products']}" if stats["stock_risk_count"] else "none",
                delta_style="warning" if stats["stock_risk_count"] else "neutral",
            )


def render_pipeline_walkthrough(dataset_label, product_count, stats, top_opportunity):
    """A short, honest narration of the real pipeline stages using numbers
    already computed this run — no artificial delays, no fake 'thinking'.
    Presentation layer only; every figure shown is real."""
    with st.expander("See how PriceSense works", expanded=False):
        st.markdown(f"**1. Load sales data** — {product_count} products loaded from {dataset_label}.")
        st.markdown("**2. Analyze historical demand** — a demand model is fit per product from its own "
                     "price and sales history.")
        st.markdown("**3. Estimate price sensitivity** — how much predicted demand moves per 1% price change, "
                     "per product.")
        st.markdown("**4. Simulate price scenarios** — the optimizer searches prices within ±30% of current, "
                     "subject to stock-runway and margin constraints.")
        st.markdown(f"**5. Identify revenue opportunities** — {stats['opportunity_count']} of "
                     f"{stats['total_products']} products show a positive, unconstrained opportunity.")
        if top_opportunity:
            st.markdown(f"**6. Recommendation ready** — biggest opportunity: **{top_opportunity['Product']}** "
                         f"at ₹{top_opportunity['Suggested Price']:.2f} "
                         f"({top_opportunity['Revenue Impact (%)']:+.1f}% projected revenue).")
        else:
            st.markdown("**6. Recommendation ready** — no unconstrained opportunity in this dataset right now.")


def render_top_opportunity_hero(top_opportunity, on_select):
    from app.insights import explain_opportunity

    if top_opportunity is None:
        return

    with st.container(border=True):
        st.markdown('<div class="pricesense-eyebrow">Biggest revenue opportunity</div>', unsafe_allow_html=True)
        st.markdown(f"## {top_opportunity['Product']}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_animated_metric("Current price", top_opportunity["Current Price"], decimals=2, prefix="₹")
        with col2:
            price_change_pct = (
                (top_opportunity["Suggested Price"] - top_opportunity["Current Price"])
                / top_opportunity["Current Price"] * 100 if top_opportunity["Current Price"] > 0 else 0.0
            )
            render_animated_metric(
                "Recommended price", top_opportunity["Suggested Price"], decimals=2, prefix="₹",
                delta_text=f"{price_change_pct:+.1f}% price", delta_style="neutral",
            )
        with col3:
            render_animated_metric(
                "Expected demand change", top_opportunity["Demand Change (%)"], decimals=1, suffix="%", show_sign=True,
            )
        with col4:
            render_animated_metric(
                "Projected revenue impact", top_opportunity["Revenue Impact (%)"], decimals=1, suffix="%", show_sign=True,
                delta_text="projected revenue", delta_style="positive",
            )

        explanation = explain_opportunity(top_opportunity)
        if explanation:
            st.markdown("**Why this matters**")
            st.caption(explanation)

        if st.button("Explore recommendation →", key="hero_explore", width="stretch"):
            on_select(top_opportunity["Product"])


def render_opportunity_summary(top_opportunity, top_sensitivity, top_risk, on_select):
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown('<div class="pricesense-eyebrow">Top revenue opportunity</div>', unsafe_allow_html=True)
            if top_opportunity is None:
                st.caption("No unconstrained opportunities in the current dataset.")
            else:
                st.markdown(f"**{top_opportunity['Product']}**")
                st.caption(f"{top_opportunity['Revenue Impact (%)']:+.1f}% projected revenue impact")
                if st.button("View product", key="opp_view_revenue", width="stretch"):
                    on_select(top_opportunity["Product"])

    with col2:
        with st.container(border=True):
            st.markdown('<div class="pricesense-eyebrow">Highest demand sensitivity</div>', unsafe_allow_html=True)
            if top_sensitivity is None:
                st.caption("No products with a clear price-sensitivity signal.")
            else:
                st.markdown(f"**{top_sensitivity['Product']}**")
                st.caption(f"~{top_sensitivity['_demand_sensitivity']:.2f} demand response per 1% price move")
                if st.button("View product", key="opp_view_sensitivity", width="stretch"):
                    on_select(top_sensitivity["Product"])

    with col3:
        with st.container(border=True):
            st.markdown('<div class="pricesense-eyebrow">Highest stock risk</div>', unsafe_allow_html=True)
            if top_risk is None:
                st.caption("No products currently flagged for stock risk.")
            else:
                st.markdown(f"**{top_risk['Product']}**")
                st.caption(f"{top_risk['Constraint']} — {top_risk['Stock']} units on hand")
                if st.button("View product", key="opp_view_risk", width="stretch"):
                    on_select(top_risk["Product"])


def render_revenue_chart(summary_rows):
    df = pd.DataFrame(summary_rows).sort_values("Revenue Impact (%)")
    positive_color, _ = BADGE_STYLES["positive"]
    negative_color, _ = BADGE_STYLES["negative"]
    colors = ["#991B1B" if value < 0 else "#4F46E5" for value in df["Revenue Impact (%)"]]

    fig = go.Figure(go.Bar(
        x=df["Revenue Impact (%)"], y=df["Product"], orientation="h",
        marker_color=colors,
        text=[f"{value:+.1f}%" for value in df["Revenue Impact (%)"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Projected revenue impact by product",
        xaxis_title="Revenue change (%)",
        height=max(300, 40 * len(df)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    with st.container(border=True):
        st.plotly_chart(fig)


CONSTRAINT_LABEL_STYLE = {
    "None": "positive",
    "Already optimal": "positive",
    "Stock constrained": "warning",
    "Insufficient data": "neutral",
    "Out of stock": "neutral",
}


def render_portfolio_health_chart(summary_rows):
    counts = pd.Series([row["Constraint"] for row in summary_rows]).value_counts()
    colors = [BADGE_STYLES[CONSTRAINT_LABEL_STYLE.get(label, "neutral")][0] for label in counts.index]

    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.6, marker=dict(colors=colors)))
    fig.update_layout(
        title="Portfolio health",
        height=max(300, 40 * len(summary_rows) // 2),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
    )
    with st.container(border=True):
        st.plotly_chart(fig)


def render_stock_vs_opportunity_chart(summary_rows):
    df = pd.DataFrame(summary_rows)
    colors = ["#991B1B" if value < 0 else "#4F46E5" for value in df["Revenue Impact (%)"]]

    fig = go.Figure(go.Scatter(
        x=df["Stock"], y=df["Revenue Impact (%)"], mode="markers",
        marker=dict(size=12, color=colors),
        text=df["Product"],
        hovertemplate="%{text}<br>Stock: %{x} units<br>Revenue impact: %{y:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Stock level vs. revenue opportunity",
        xaxis_title="Current stock (units)",
        yaxis_title="Revenue impact (%)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
    )
    with st.container(border=True):
        st.plotly_chart(fig)


def render_opportunity_map(summary_rows, on_select=None):
    from app.insights import bubble_candidates

    candidates = bubble_candidates(summary_rows)
    if not candidates:
        render_empty_state(
            "Not enough signal for an opportunity map",
            "No products currently have a defined price-sensitivity reading.",
        )
        return

    df = pd.DataFrame(candidates)
    colors = ["#991B1B" if value < 0 else "#4F46E5" for value in df["Revenue Impact (%)"]]

    fig = go.Figure(go.Scatter(
        x=df["_demand_sensitivity"], y=df["Revenue Impact (%)"], mode="markers",
        marker=dict(
            size=df["Forecasted Demand"], sizemode="area",
            sizeref=2.0 * max(df["Forecasted Demand"].max(), 1) / (40.0 ** 2),
            sizemin=6, color=colors,
        ),
        text=df["Product"],
        hovertemplate="%{text}<br>Sensitivity: %{x:.2f}<br>Revenue impact: %{y:+.1f}%"
                       "<br>Forecasted demand: %{marker.size:.0f} units<extra></extra>",
    ))
    fig.update_layout(
        title="Opportunity map",
        xaxis_title="Price sensitivity (demand response per 1% price move)",
        yaxis_title="Revenue impact (%)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
    )
    with st.container(border=True):
        st.plotly_chart(fig)
        st.caption("Bubble size reflects forecasted demand. Products without a defined price-sensitivity "
                   f"reading ({len(summary_rows) - len(candidates)} of {len(summary_rows)}) are excluded rather "
                   "than plotted at a misleading position.")
        if on_select:
            pick = st.selectbox(
                "Jump to a product from the map", [row["Product"] for row in candidates],
                key="opportunity_map_pick", label_visibility="collapsed",
                placeholder="Jump to a product from the map",
                index=None,
            )
            if pick:
                on_select(pick)


def render_insight_feed(insights, on_select=None):
    if not insights:
        render_empty_state("No notable insights yet", "Insights appear once products have been analyzed.")
        return

    with st.container(border=True):
        for i, insight in enumerate(insights):
            cols = st.columns([5, 1]) if (on_select and insight["product"]) else [st.container()]
            with cols[0]:
                st.markdown(f'<div class="pricesense-insight">{insight["text"]}</div>', unsafe_allow_html=True)
            if on_select and insight["product"]:
                with cols[1]:
                    if st.button("View", key=f"insight_{i}", width="stretch"):
                        on_select(insight["product"])


PRIORITY_SORT_ORDER = {"High Priority": 0, "Constrained": 1, "Medium Priority": 2, "Monitor": 3, "Already Optimized": 4}


def render_priority_table(summary_rows):
    df = pd.DataFrame(summary_rows).drop(columns=["_panel_state", "_demand_sensitivity"])
    df["_sort_key"] = df["Priority"].map(PRIORITY_SORT_ORDER)
    df = df.sort_values(["_sort_key", "Revenue Impact (%)"], ascending=[True, False]).drop(columns=["_sort_key"])

    with st.container(border=True):
        search_col, priority_col, constraint_col = st.columns([2, 1, 1])
        with search_col:
            search_term = st.text_input("Search products", placeholder="Type a product name...", key="priority_search")
        with priority_col:
            priority_filter = st.selectbox(
                "Filter by priority", ["All"] + list(PRIORITY_SORT_ORDER.keys()), key="priority_priority_filter",
            )
        with constraint_col:
            constraint_filter = st.selectbox(
                "Filter by status", ["All"] + sorted(df["Constraint"].unique().tolist()), key="priority_filter",
            )

        filtered = df
        if search_term:
            filtered = filtered[filtered["Product"].str.contains(search_term, case=False, na=False)]
        if priority_filter != "All":
            filtered = filtered[filtered["Priority"] == priority_filter]
        if constraint_filter != "All":
            filtered = filtered[filtered["Constraint"] == constraint_filter]

        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
            column_config={
                "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
                "Suggested Price": st.column_config.NumberColumn(format="₹%.2f"),
                "Revenue Impact (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Demand Change (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption(f"Showing {len(filtered)} of {len(df)} products, sorted by priority. "
                   "Click a column header to re-sort.")


def render_product_selector(products, key):
    if st.session_state.get(key) not in products:
        st.session_state[key] = products[0]
    return st.selectbox("Choose a product", products, key=key)


COMPARISON_FORMATTERS = {
    "Current Price": lambda v: f"₹{v:,.2f}",
    "Suggested Price": lambda v: f"₹{v:,.2f}",
    "Forecasted Demand": lambda v: f"{v:,.1f} units",
    "Revenue Impact (%)": lambda v: f"{v:+.1f}%",
    "Demand Change (%)": lambda v: f"{v:+.1f}%",
    "Stock": lambda v: f"{v:,}",
    "Constraint": str,
    "Priority": str,
}


def render_product_comparison(selected_rows):
    if len(selected_rows) < 2:
        st.caption("Select two or more products above to compare them side by side.")
        return

    df = pd.DataFrame(selected_rows).drop(columns=["_panel_state", "_demand_sensitivity"])
    for column, formatter in COMPARISON_FORMATTERS.items():
        df[column] = df[column].apply(formatter)
    df = df.set_index("Product").T

    with st.container(border=True):
        st.dataframe(df, width="stretch")


def render_product_header(product, optimize_data):
    current_price = optimize_data["current_price"]
    suggested_price = optimize_data["suggested_price"]
    current_units = optimize_data["current_predicted_units"]
    suggested_units = optimize_data["suggested_predicted_units"]
    current_stock = optimize_data["current_stock"]
    days_of_stock = optimize_data.get("days_of_stock_at_suggested_price")

    price_change_pct = (suggested_price - current_price) / current_price * 100 if current_price > 0 else 0.0
    demand_change_pct = (suggested_units - current_units) / current_units * 100 if current_units > 0 else 0.0

    with st.container(border=True):
        st.markdown(f"## {product}")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_animated_metric("Current price", current_price, decimals=2, prefix="₹")
        with col2:
            render_animated_metric(
                "Recommended price", suggested_price, decimals=2, prefix="₹",
                delta_text=f"{price_change_pct:+.1f}%", delta_style="neutral",
            )
        with col3:
            render_animated_metric("Demand change", demand_change_pct, decimals=1, suffix="%", show_sign=True)
        with col4:
            revenue_impact = optimize_data["projected_revenue_change_pct"]
            render_animated_metric(
                "Revenue impact", revenue_impact, decimals=1, suffix="%", show_sign=True,
                delta_text=f"{revenue_impact:+.1f}%",
                delta_style="positive" if revenue_impact >= 0 else "negative",
            )
        with col5:
            render_animated_metric(
                "Stock on hand", current_stock, decimals=0,
                delta_text=f"{days_of_stock:.0f} days runway" if days_of_stock else None, delta_style="neutral",
            )


def build_demand_curve_figure(product, curve_df, optimize_data, panel_state, what_if=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["price"], y=curve_df["predicted_units"],
        mode="lines", name="Predicted demand", line=dict(color="#4F46E5", width=3),
        hovertemplate="Price: ₹%{x:.2f}<br>Predicted units: %{y:.1f}<extra></extra>",
    ))

    current_price = optimize_data["current_price"]
    current_units = optimize_data["current_predicted_units"]
    fig.add_trace(go.Scatter(
        x=[current_price], y=[current_units], mode="markers",
        marker=dict(size=14, color="#6B7280", symbol="diamond"),
        name="Current price",
        hovertemplate=f"Current price<br>₹{current_price:.2f}, {current_units:.1f} units<extra></extra>",
    ))

    if panel_state == "price_change_suggested":
        suggested_price = optimize_data["suggested_price"]
        suggested_units = optimize_data["suggested_predicted_units"]
        fig.add_trace(go.Scatter(
            x=[suggested_price], y=[suggested_units], mode="markers",
            marker=dict(size=14, color="#059669", symbol="star"),
            name="Recommended price",
            hovertemplate=f"Recommended price<br>₹{suggested_price:.2f}, {suggested_units:.1f} units<extra></extra>",
        ))

    if what_if is not None:
        what_if_price, what_if_units = what_if
        fig.add_trace(go.Scatter(
            x=[what_if_price], y=[what_if_units], mode="markers",
            marker=dict(size=14, color="#D97706", symbol="circle"),
            name="What-if price",
            hovertemplate=f"What-if price<br>₹{what_if_price:.2f}, {what_if_units:.1f} units<extra></extra>",
        ))

    fig.update_layout(
        title=f"Demand curve — {product}",
        xaxis_title="Price (₹)",
        yaxis_title="Predicted units sold",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


def render_demand_chart(product, curve_df, optimize_data, panel_state, what_if=None):
    fig = build_demand_curve_figure(product, curve_df, optimize_data, panel_state, what_if=what_if)
    with st.container(border=True):
        st.plotly_chart(fig)


def is_dead_product(optimize_data, panel_state):
    return panel_state == "insufficient_data" and optimize_data["current_predicted_units"] < 0.01


def render_pricing_health(score):
    if score >= 70:
        style = "positive"
    elif score >= 40:
        style = "warning"
    else:
        style = "neutral"
    background, color = BADGE_STYLES[style]

    with st.container(border=True):
        st.markdown('<div class="pricesense-eyebrow">Pricing health</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:1.75rem;font-weight:700;color:{color};">{score} / 100</div>',
            unsafe_allow_html=True,
        )
        st.caption("A transparent composite of constraint health, price-sensitivity signal quality, and "
                   "opportunity magnitude — not a model confidence score.")


def render_guardrails_card(guardrails):
    with st.container(border=True):
        st.markdown('<div class="pricesense-eyebrow">Pricing guardrails</div>', unsafe_allow_html=True)
        st.caption("Currently enforced by the pricing engine for every recommendation. Read-only in this view.")
        for rule in guardrails:
            st.markdown(f"- **{rule['label']}:** {rule['value']} — {rule['description']}")


def render_recommendation_card(optimize_data, panel_state):
    headline = build_headline(optimize_data, panel_state)

    with st.container(border=True):
        st.markdown('<div class="pricesense-eyebrow">Recommended action</div>', unsafe_allow_html=True)
        st.markdown(f"#### {headline}")
        render_panel_badge(panel_state)

        if is_dead_product(optimize_data, panel_state):
            st.caption("Zero predicted demand at every price tested. This more likely reflects a delisted "
                       "or out-of-season product than a data-quality gap.")
        else:
            st.caption(optimize_data["explanation"])


def render_why_this_price(optimize_data, panel_state, sensitivity_label, elasticity_index):
    current_units = optimize_data["current_predicted_units"]
    suggested_units = optimize_data["suggested_predicted_units"]
    demand_change_pct = (
        (suggested_units - current_units) / current_units * 100 if current_units > 0 else 0.0
    )

    with st.expander("Explain this recommendation", expanded=False):
        st.caption("PriceSense considered the following factors, all derived from this product's own "
                   "historical price and demand data:")

        st.markdown(f"- **Demand response:** predicted demand shifts {demand_change_pct:+.1f}% at the "
                     f"recommended price versus current.")
        st.markdown(f"- **Revenue impact:** projected total revenue changes by "
                     f"{optimize_data['projected_revenue_change_pct']:+.1f}%.")

        if panel_state == "out_of_stock":
            constraint_line = "Out of stock — no price can be recommended until inventory is replenished."
        elif panel_state == "stock_constrained":
            constraint_line = ("Stock-runway constraint — the ±30% search range was narrowed to keep stock "
                                "lasting the required minimum number of days.")
        elif panel_state == "insufficient_data":
            constraint_line = "No binding constraint — held back by limited historical signal instead."
        else:
            constraint_line = "No stock or cost-floor constraint is currently binding."
        st.markdown(f"- **Stock constraint:** {constraint_line}")

        st.markdown(f"- **Price sensitivity:** {sensitivity_label} (elasticity index {elasticity_index:.2f}).")

        if panel_state == "insufficient_data":
            confidence_line = ("Based on limited historical price variation for this product — not enough "
                                "signal to model demand response with confidence.")
        else:
            confidence_line = "Based on the available historical price and demand patterns for this product."
        st.markdown(f"- **Basis:** {confidence_line}")


def render_what_if_controls(current_price, suggested_price, key_prefix):
    slider_key = f"{key_prefix}_slider"
    what_if_min = max(0.01, round(current_price * 0.1, 2))
    what_if_max = round(current_price * 2.0, 2)
    what_if_step = max(0.01, round(current_price * 0.01, 2))

    if slider_key not in st.session_state:
        st.session_state[slider_key] = current_price

    def _reset_to(value):
        st.session_state[slider_key] = min(max(value, what_if_min), what_if_max)

    reset_col1, reset_col2, _ = st.columns([1, 1, 3])
    with reset_col1:
        st.button("Reset to current", key=f"{key_prefix}_reset_current",
                   on_click=_reset_to, args=(current_price,), width="stretch")
    with reset_col2:
        st.button("Reset to recommended", key=f"{key_prefix}_reset_suggested",
                   on_click=_reset_to, args=(suggested_price,), width="stretch")

    return st.slider(
        "What-if price (₹) — drag to test a manual price, release to see the impact",
        min_value=what_if_min, max_value=what_if_max, step=what_if_step, key=slider_key,
    )


def render_what_if_result(current_price, what_if_price, what_if_units, what_if_error, optimize_data):
    """Three-way comparison: Current strategy / Recommended strategy / your
    What-if price, side by side, plus an honest better-or-worse-than-the-
    recommendation readout. All numbers derive from optimize_data (already
    fetched) and the already-computed what_if_units — no new API calls."""
    with st.container(border=True):
        st.markdown("**Current vs. recommended vs. what-if**")

        if what_if_error:
            st.warning(f"Could not compute demand at ₹{what_if_price:.2f}: {what_if_error}")
            return
        if what_if_units is None:
            st.warning(f"Could not compute demand at ₹{what_if_price:.2f}.")
            return

        current_units = optimize_data["current_predicted_units"]
        suggested_price = optimize_data["suggested_price"]
        suggested_units = optimize_data["suggested_predicted_units"]
        current_stock = optimize_data["current_stock"]
        floor_price = optimize_data["floor_price"]
        min_days_of_stock = optimize_data.get("min_days_of_stock", 7)

        current_revenue = current_price * min(current_units, current_stock)
        suggested_revenue = suggested_price * min(suggested_units, current_stock)
        what_if_revenue = what_if_price * min(what_if_units, current_stock)

        def pct_vs_current(revenue):
            return (revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="pricesense-eyebrow">Current</div>', unsafe_allow_html=True)
            render_animated_metric("Price", current_price, decimals=2, prefix="₹")
            st.caption(f"{current_units:.1f} predicted units")
        with col2:
            st.markdown('<div class="pricesense-eyebrow">Recommended</div>', unsafe_allow_html=True)
            render_animated_metric(
                "Price", suggested_price, decimals=2, prefix="₹",
                delta_text=f"{pct_vs_current(suggested_revenue):+.1f}% revenue",
                delta_style="positive" if pct_vs_current(suggested_revenue) >= 0 else "negative",
            )
            st.caption(f"{suggested_units:.1f} predicted units")
        with col3:
            st.markdown('<div class="pricesense-eyebrow">Your what-if</div>', unsafe_allow_html=True)
            render_animated_metric(
                "Price", what_if_price, decimals=2, prefix="₹",
                delta_text=f"{pct_vs_current(what_if_revenue):+.1f}% revenue",
                delta_style="positive" if pct_vs_current(what_if_revenue) >= 0 else "negative",
            )
            st.caption(f"{what_if_units:.1f} predicted units")

        if suggested_revenue > 0:
            vs_recommended_pct = (what_if_revenue - suggested_revenue) / suggested_revenue * 100
            if abs(vs_recommended_pct) < 0.5:
                st.caption("Your what-if price performs about the same as the recommendation.")
            elif vs_recommended_pct > 0:
                st.success(f"Your what-if price outperforms the recommendation by {vs_recommended_pct:.1f}% "
                           f"projected revenue.")
            else:
                st.info(f"The recommended price outperforms your what-if by {abs(vs_recommended_pct):.1f}% "
                        f"projected revenue.")

        days_of_stock = current_stock / what_if_units if what_if_units > 0 else float("inf")
        violations = []
        if what_if_price < floor_price:
            violations.append(f"is below the cost floor of ₹{floor_price:.2f}")
        if what_if_units > 0 and days_of_stock < min_days_of_stock:
            violations.append(f"would deplete stock in {days_of_stock:.1f} days (minimum {min_days_of_stock})")

        if violations:
            st.warning("Your what-if price " + "; ".join(violations) + ".")
        else:
            st.caption("Your what-if price violates no constraints.")


def render_chat_history(chat_history):
    if not chat_history:
        st.caption("No messages yet. Ask a question below or use one of the suggestions.")
        return
    for role, content in chat_history:
        with st.chat_message(role):
            st.markdown(content)


def render_suggested_question_chips(questions):
    columns = st.columns(len(questions))
    clicked = None
    for column, question in zip(columns, questions):
        with column:
            if st.button(question, key=f"chip_{question}", width="stretch"):
                clicked = question
    return clicked
