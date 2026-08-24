# Shared session-state, bootstrapping, and cross-page helpers.
# Business logic that multiple pages need (dataset resolution, sending a chat
# message) lives here rather than duplicated in each page file.

import streamlit as st

from app.api_client import api_post_chat, check_health, fetch_products
from app.components import inject_global_css, render_empty_state, render_health_banner

DEFAULT_DATASET_LABEL = "Demo dataset (synthetic_sales_data.csv)"
SUGGESTED_CHIP_TEMPLATES = [
    "Why did {product}'s price change?",
    "Which product has the best revenue uplift?",
    "Any stock risk?",
    "What can you help with?",
]


def init_page(title, show_title=True):
    st.set_page_config(page_title=f"PriceSense — {title}", layout="wide")
    _init_session_state()
    inject_global_css()
    with st.sidebar:
        st.markdown("### PriceSense")
        st.caption("Dynamic pricing simulator")
        st.divider()
    if show_title:
        st.title(title)


def _init_session_state():
    defaults = {
        "dataset_id": None,
        "dataset_label": DEFAULT_DATASET_LABEL,
        "last_uploaded_file_id": None,
        "chat_history": [],
        "selected_product": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def bootstrap():
    render_health_banner(check_health())

    products_data, products_error, _ = fetch_products(st.session_state.dataset_id)
    if products_error:
        render_empty_state(
            "Data unavailable",
            products_error,
            "Check that the API is running, or try reloading this page.",
        )
        return None

    if st.session_state.dataset_id is None:
        st.session_state.dataset_id = products_data["dataset_id"]

    products = products_data["products"]
    if not products:
        render_empty_state(
            "No products in this dataset",
            "The active dataset has no products to analyze.",
            "Upload a dataset with at least one product on the Settings page.",
        )
        return None

    return products


def send_chat_message(user_message):
    st.session_state.chat_history.append(("user", user_message))
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving answer..."):
            chat_data, chat_error = api_post_chat(user_message, st.session_state.dataset_id)
        assistant_reply = chat_error if chat_error else chat_data["answer"]
        st.markdown(assistant_reply)

    st.session_state.chat_history.append(("assistant", assistant_reply))
