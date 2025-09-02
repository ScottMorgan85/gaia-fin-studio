# landing.py
import os, re, datetime, pandas as pd, streamlit as st

LOG = os.environ.get("VISITOR_LOG_PATH", "data/visitor_log.csv")

def _append_csv(row: dict) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        LOG, mode="a", index=False, header=not os.path.isfile(LOG)
    )

def _is_valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .,'-]{1,98}", name.strip()))

def _is_valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", email.strip()))

def render_form() -> None:
    st.title("GAIA — Access")
    st.markdown("Enter your details to continue. We’ll log them for collaboration and support.")

    with st.form("request_form", clear_on_submit=True):
        name  = st.text_input("Your name",  max_chars=100, placeholder="e.g., Jordan Lee")
        email = st.text_input("Work email", max_chars=120, placeholder="e.g., jordan@firm.com")

        # gentle pre-submit hints
        if name and not _is_valid_name(name):
            st.caption("⚠️ Use letters, spaces, or - , . '")
        if email and not _is_valid_email(email):
            st.caption("⚠️ Enter a valid email like name@company.com")

        submitted = st.form_submit_button("Continue →")

    if not submitted:
        return

    # hard validation
    if not name or not _is_valid_name(name):
        st.warning("Please provide a valid name.")
        return
    if not email or not _is_valid_email(email):
        st.warning("Please provide a valid email.")
        return

    # log + pass the gate
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": name.strip(),
        "email": email.strip(),
    }
    _append_csv(row)

    # mark session and proceed immediately
    st.session_state["user_name"] = row["name"]
    st.session_state["user_email"] = row["email"]
    st.session_state["gate_passed"] = True

    # no confetti, just go
    st.rerun
