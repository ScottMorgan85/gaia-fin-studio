import os
import json
import streamlit as st
import boto3
from botocore.exceptions import ClientError
import streamlit as st
import pandas as pd
from datetime import datetime
import csv

# ── Visitor log helper ───────────────────────────────
def log_visitor(payload: dict):
    """
    Append visitor request to a CSV log.
    """
    with open("visitor_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), payload["name"], payload["email"]])


# ── Request Access Form ──────────────────────────────
def render_request_form():
    st.title("🚪 Request Access — GAIA Dashboard")
    st.write("Please enter your details. We'll review and approve you if eligible!")

    name = st.text_input("Your name", max_chars=50)
    email = st.text_input("Email address", max_chars=100)
    submitted = st.button("Request Access →")

    if submitted:
        if not name or not email or "@" not in email:
            st.warning("Please enter a valid name and email.")
            return

        log_visitor({"name": name, "email": email})
        st.success("✅ Thanks! We'll review your request and email you once approved.")
        st.balloons()


# ── Sign-In Form ─────────────────────────────────────
def check_access(name: str, email: str) -> bool:
    try:
        df = pd.read_csv("visitor_log.csv")
        match = df[
            (df[1].str.lower() == name.lower()) &
            (df[2].str.lower() == email.lower())
        ]
        return not match.empty
    except FileNotFoundError:
        return False


def render_sign_in():
    st.title("🔑 Sign In — GAIA Dashboard")

    name = st.text_input("Your name", key="sign_in_name")
    email = st.text_input("Email address", key="sign_in_email")

    if st.button("Sign In →"):
        if check_access(name, email):
            st.session_state["signed_in"] = True
            st.session_state["user_name"] = name
            st.session_state["user_email"] = email
            st.success(f"✅ Welcome back, {name}!")
        else:
            st.error("❌ Not found. Please request access first.")
