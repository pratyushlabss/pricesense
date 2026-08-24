import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
while not (_project_root / "requirements.txt").exists() and _project_root != _project_root.parent:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.common import SUGGESTED_CHIP_TEMPLATES, bootstrap, init_page, send_chat_message
from app.components import render_chat_history, render_suggested_question_chips

init_page("Chat")
products = bootstrap()

if products is not None:
    st.caption(f"Active dataset: **{st.session_state.dataset_label}**")
    st.caption("Retrieval over your pricing data, not a general chatbot — ask about price changes, "
               "revenue opportunity, or stock risk.")

    reference_product = st.session_state.selected_product or products[0]
    chips = [template.format(product=reference_product) if "{product}" in template else template
             for template in SUGGESTED_CHIP_TEMPLATES]

    render_chat_history(st.session_state.chat_history)
    clicked_chip = render_suggested_question_chips(chips)
    typed_message = st.chat_input("Ask about pricing, revenue, or stock risk...")

    user_message = clicked_chip or typed_message
    if user_message:
        send_chat_message(user_message)
