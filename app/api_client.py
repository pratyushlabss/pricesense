# Pure API client — every HTTP call to the FastAPI backend lives here.
# No rendering, no st.* calls except @st.cache_data (caching is data-layer
# concern, not presentation). Never import forecast.py/optimizer.py directly.

import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
API_UNREACHABLE_MESSAGE = (
    f"Cannot reach the PriceSense API at {API_BASE_URL}. "
    "Start it with: uvicorn api.main:app --reload --port 8000"
)


def extract_error_message(response):
    try:
        detail = response.json().get("detail")
    except ValueError:
        return f"API error (HTTP {response.status_code})"
    if isinstance(detail, dict) and "errors" in detail:
        return "; ".join(detail["errors"])
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def get_error_fields(response):
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    if isinstance(detail, dict) and "errors" in detail:
        return detail["errors"]
    return None


def api_get(path, params=None):
    try:
        response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
    except requests.exceptions.RequestException:
        return None, API_UNREACHABLE_MESSAGE, None
    if response.status_code >= 400:
        return None, extract_error_message(response), get_error_fields(response)
    return response.json(), None, None


def api_post_upload(uploaded_file):
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
            timeout=30,
        )
    except requests.exceptions.RequestException:
        return None, API_UNREACHABLE_MESSAGE, None
    if response.status_code >= 400:
        return None, extract_error_message(response), get_error_fields(response)
    return response.json(), None, None


def api_post_chat(message, dataset_id):
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message},
            params={"dataset_id": dataset_id},
            timeout=10,
        )
    except requests.exceptions.RequestException:
        return None, API_UNREACHABLE_MESSAGE
    if response.status_code >= 400:
        return None, extract_error_message(response)
    return response.json(), None


def check_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@st.cache_data(show_spinner=False)
def fetch_products(dataset_id):
    return api_get("/products", params={"dataset_id": dataset_id})


@st.cache_data(show_spinner=False)
def fetch_forecast(product, dataset_id):
    return api_get(f"/forecast/{product}", params={"dataset_id": dataset_id})


@st.cache_data(show_spinner=False)
def fetch_optimize(product, dataset_id):
    return api_get(f"/optimize/{product}", params={"dataset_id": dataset_id})


@st.cache_data(show_spinner=False)
def fetch_what_if_point(product, dataset_id, price):
    return api_get(f"/forecast/{product}", params={"dataset_id": dataset_id, "price": price, "steps": 1})
