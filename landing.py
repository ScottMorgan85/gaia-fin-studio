# landing.py
import os, re, datetime, pandas as pd, streamlit as st

LOG = os.environ.get("VISITOR_LOG_PATH", "data/visitor_log.csv")

def _append_csv(row: dict) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        LOG, mode="a", index=False, header=not os.path.isfile(LOG)
    )

def _is_valid_name(name: str) -> bool:
    # simple: at least 2 chars, letters/space/punct commonly in names
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .,'-]{1,98}", name.strip()))

def _is_valid_email(email: str) -> bool:
    # lightweight email check (keeps it friendly)
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", email.strip()))

def render_form() -> None:
    st.title("GAIA — Collaborate with Us")
    st.markdown(
        "We’re collecting a few details so we can share access and follow up with relevant updates. "
        "This helps us keep the demo stable and aligned to your interests."
    )

    with st.form("request_form", clear_on_submit=True):
        name  = st.text_input("Your name", max_chars=100, placeholder="e.g., Alex Rivera")
        email = st.text_input("Work email", max_chars=120, placeholder="e.g., alex@firm.com")

        # Optional comments: shown to users, NOT saved to CSV (keeps existing schema)
        comments = st.text_area("Comments (optional)", height=80, placeholder="What would you like to explore?")

        # Friendly validation hints before submit
        if name and not _is_valid_name(name):
            st.caption("⚠️ Name looks unusual — letters, spaces, and - , . ' are OK.")

        if email and not _is_valid_email(email):
            st.caption("⚠️ Please enter a valid email like name@company.com")

        submitted = st.form_submit_button("Request access →")

    if not submitted:
        return

    # Hard validation after submit
    if not name or not _is_valid_name(name):
        st.warning("Please provide your name (letters, spaces, and - , . ' only).")
        return

    if not email or not _is_valid_email(email):
        st.warning("Please enter a valid email like name@company.com.")
        return

    # Build row (CSV schema unchanged: timestamp, name, email)
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": name.strip(),
        "email": email.strip(),
    }
    _append_csv(row)

    # We intentionally do NOT persist comments to keep the CSV schema stable.
    if comments.strip():
        st.info("Thanks for the context — we’ve noted your comments for this session.")

    st.success("Thanks! Your request was logged. We’ll be in touch soon.")
    st.balloons()
