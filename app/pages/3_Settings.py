import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
while not (_project_root / "requirements.txt").exists() and _project_root != _project_root.parent:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.api_client import api_post_upload, check_health, fetch_products
from app.common import init_page
from app.components import (
    render_dataset_info_card,
    render_health_banner,
    render_section_header,
    render_upload_confirmation,
    render_upload_errors,
    render_upload_widget,
)

init_page("Settings")
render_health_banner(check_health())

render_section_header("Current dataset")
products_data, products_error, _ = fetch_products(st.session_state.dataset_id)
if products_error:
    st.error(products_error)
else:
    if st.session_state.dataset_id is None:
        st.session_state.dataset_id = products_data["dataset_id"]
    render_dataset_info_card(products_data["dataset_id"], products_data["products"], st.session_state.dataset_label)

render_section_header("Upload a new dataset", "Replaces the active dataset for every page, including chat history.")
uploaded_file = render_upload_widget()
if uploaded_file is not None and uploaded_file.file_id != st.session_state.last_uploaded_file_id:
    with st.spinner("Validating and loading upload..."):
        upload_data, upload_error, error_fields = api_post_upload(uploaded_file)
    st.session_state.last_uploaded_file_id = uploaded_file.file_id
    if upload_error:
        render_upload_errors(upload_error, error_fields)
    else:
        st.session_state.dataset_id = upload_data["dataset_id"]
        st.session_state.dataset_label = uploaded_file.name
        st.session_state.chat_history = []
        st.session_state.selected_product = None
        render_upload_confirmation(upload_data)
