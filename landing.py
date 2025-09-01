import re
import streamlit as st
import utils  # centralized logger -> data/visitor_log.csv

# Hardcoded contact for questions
CONTACT_EMAIL = "scott@scottmmorgan.com"

# Common consumer email domains (allowed, but we'll gently nudge)
PERSONAL_DOMAINS = {
    "gmail.com", "googlemail.com",
    "outlook.com", "hotmail.com", "live.com", "msn.com",
    "yahoo.com", "ymail.com",
    "icloud.com", "me.com",
    "aol.com",
    "proton.me", "protonmail.com",
}

EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)

def _has_enough_vowels(s: str) -> bool:
    s2 = re.sub(r"[^a-z]", "", (s or "").lower())
    if not s2:
        return False
    vowels = sum(1 for c in s2 if c in "aeiou")
    return (vowels / max(1, len(s2))) >= 0.20  # 20%+ vowels

def _no_long_consonant_runs(s: str, max_run: int = 4) -> bool:
    return re.search(rf"[bcdfghjklmnpqrstvwxyz]{{{max_run+1},}}", (s or "").lower()) is None

def _looks_nonrandom_wordlike(s: str) -> bool:
    # Allow spaces, hyphens, apostrophes in names
    s2 = re.sub(r"[^a-z'\- ]", "", (s or "").lower()).strip()
    return _has_enough_vowels(s2) and _no_long_consonant_runs(s2)

def _valid_email(email: str) -> tuple[bool, str]:
    email = (email or "").strip()
    if "@" not in email:
        return False, "Email must contain @."
    if not EMAIL_RE.match(email):
        return False, "Email format looks invalid."
    local, domain = email.rsplit("@", 1)
    if not _no_long_consonant_runs(local):
        return False, "Email local-part looks random (too many consonants in a row)."
    if re.search(r"\.[a-z]{2,}$", domain.lower()):
        return True, ""
    return False, "Email domain looks invalid."

def _valid_name(name: str) -> tuple[bool, str]:
    name = (name or "").strip()
    if len(name) < 2:
        return False, "Name is too short."
    # Allow letters, spaces, hyphens, apostrophes
    if re.search(r"[^a-zA-Z '\-]", name):
        return False, "Name may only contain letters, spaces, apostrophes or hyphens."
    if not _looks_nonrandom_wordlike(name):
        return False, "Name looks random. Please enter a real name."
    return True, ""

def render_gate():
    st.title("🔒 GAIA — Request Access (Not a login)")
    st.caption(
        "We’re just collecting contact info for **networking and potential collaboration**. "
        "No passwords or SSO. After you submit, the app continues immediately."
    )

    with st.form("gaia_access_request", clear_on_submit=False):
        name = st.text_input("Your name", max_chars=60, placeholder="Jane Doe")
        email = st.text_input("Work email", max_chars=120, placeholder="name@company.com")
        note = st.text_area("Context (optional)", placeholder="What would you like to explore or collaborate on?")
        submitted = st.form_submit_button("Send contact details →", use_container_width=True)

    # Footer contact under the form (always visible)
    st.caption(f"Questions? Email **[{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL})**.")

    if not submitted:
        return

    # Validate, but still keep this lightweight
    ok_name, name_msg = _valid_name(name)
    ok_email, email_msg = _valid_email(email)

    problems = []
    if not ok_name:  problems.append(f"• {name_msg}")
    if not ok_email: problems.append(f"• {email_msg}")

    if problems:
        st.error("Please fix the issues below:\n\n" + "\n".join(problems))
        return

    # Gentle nudge toward company emails (do not block)
    domain = email.split("@")[-1].lower()
    if domain in PERSONAL_DOMAINS:
        st.info("Tip: company emails help us prioritize collaboration follow-ups.")

    # Log to CSV (data/visitor_log.csv). If logging fails, still let them through.
    try:
        utils.log_visitor({"name": name, "email": email, "note": note})
    except Exception as e:
        st.warning(f"Thanks — we recorded your request, but logging failed: {e}")

    # Mark gate as passed and proceed directly to the app
    st.session_state["gate_passed"] = True
    st.session_state["user_name"] = name
    st.session_state["user_email"] = email
    st.success(f"Thanks, {name}! Enjoy the app — we’ll reach out at **{email}** about networking/collab.")
    st.caption("This is not an authentication form and does not create an account.")
    st.rerun()  # <- replace experimental_rerun


# If your app imports landing.render_form(), keep a thin alias
def render_form():
    return render_gate()
