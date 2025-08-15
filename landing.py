import re
import streamlit as st
import pandas as pd
from datetime import datetime
import utils  # use the centralized logger -> data/visitor_log.csv

# Common consumer email domains (extend if you want)
ALLOWED_DOMAINS = {
    "gmail.com", "googlemail.com",
    "outlook.com", "hotmail.com", "live.com", "msn.com",
    "yahoo.com", "ymail.com",
    "icloud.com", "me.com",
    "aol.com",
    "proton.me", "protonmail.com"
}

EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)

def _has_enough_vowels(s: str) -> bool:
    s2 = re.sub(r"[^a-z]", "", s.lower())
    if not s2:
        return False
    vowels = sum(1 for c in s2 if c in "aeiou")
    return (vowels / max(1, len(s2))) >= 0.20  # 20%+ vowels

def _no_long_consonant_runs(s: str, max_run: int = 4) -> bool:
    return re.search(rf"[bcdfghjklmnpqrstvwxyz]{{{max_run+1},}}", s.lower()) is None

def _looks_nonrandom_wordlike(s: str) -> bool:
    # Allow spaces, hyphens, apostrophes in names
    s2 = re.sub(r"[^a-z'\- ]", "", s.lower()).strip()
    return _has_enough_vowels(s2) and _no_long_consonant_runs(s2)

def _valid_email(email: str) -> tuple[bool, str]:
    email = email.strip()
    if "@" not in email:
        return False, "Email must contain @."
    if not EMAIL_RE.match(email):
        return False, "Email format looks invalid."
    local, domain = email.rsplit("@", 1)
    if not _no_long_consonant_runs(local):
        return False, "Email local-part looks random (too many consonants in a row)."
    # Accept if domain is common OR looks like a real domain (has dot + 2+ TLD chars)
    if domain.lower() in ALLOWED_DOMAINS:
        return True, ""
    if re.search(r"\.[a-z]{2,}$", domain.lower()):
        return True, ""   # treat normal company domains as valid
    return False, "Email domain looks invalid."

def _valid_name(name: str) -> tuple[bool, str]:
    name = name.strip()
    if len(name) < 2:
        return False, "Name is too short."
    # Allow letters, spaces, hyphens, apostrophes
    if re.search(r"[^a-zA-Z '\-]", name):
        return False, "Name may only contain letters, spaces, apostrophes or hyphens."
    if not _looks_nonrandom_wordlike(name):
        return False, "Name looks random. Please enter a real name."
    return True, ""

def render_gate():
    st.title("🔐 GAIA — Quick Sign-In")
    st.caption("Enter your name and email to continue.")

    with st.form("gaia_gate_form", clear_on_submit=False):
        name = st.text_input("Your name", max_chars=60, placeholder="Ada Lovelace")
        email = st.text_input("Email address", max_chars=120, placeholder="ada.lovelace@gmail.com")
        submitted = st.form_submit_button("Continue →", use_container_width=True)

    if not submitted:
        return

    # Validation
    ok_name, name_msg = _valid_name(name)
    ok_email, email_msg = _valid_email(email)

    problems = []
    if not ok_name:  problems.append(f"• {name_msg}")
    if not ok_email: problems.append(f"• {email_msg}")

    if problems:
        st.error("Please fix the issues below:\n\n" + "\n".join(problems))
        return

    # Log to CSV (data/visitor_log.csv) with timestamp via utils
    try:
        utils.log_visitor({"name": name, "email": email})
    except Exception as e:
        st.warning(f"Logged-in but could not write visitor log: {e}")

    # Mark session as signed in and proceed to default page
    st.session_state["signed_in"] = True
    st.session_state["user_name"] = name
    st.session_state["user_email"] = email
    st.success(f"✅ Welcome, {name}! Loading your dashboard...")
    st.experimental_rerun()
