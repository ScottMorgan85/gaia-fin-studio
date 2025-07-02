import os
import json
import streamlit as st
import boto3
from botocore.exceptions import ClientError

def _publish_to_sns(topic_arn: str, payload: dict) -> bool:
    """Send the payload to an SNS topic. Return True if published."""
    try:
        client = boto3.client("sns", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        client.publish(TopicArn=topic_arn, Message=json.dumps(payload))
        return True
    except ClientError as e:
        st.error(f"Could not notify admin: {e.response['Error']['Message']}")
        return False


def render_form() -> None:
    """Render the access‑request form. Call this at top of app.py when GAIA_GATE_ON=true."""
    st.title("GAIA Dashboard — Request Access")
    st.markdown("""
    Please leave your details. We’ll review and send you an access link.
    """)

    with st.form("request_form", clear_on_submit=True):
        name  = st.text_input("Your name", max_chars=50)
        email = st.text_input("Email address", max_chars=100)
        submitted = st.form_submit_button("Request access →")

    if submitted:
        if not name or not email or "@" not in email:
            st.warning("Please enter a valid name and email.")
            return

        topic_arn = os.environ.get("SNS_TOPIC_ARN")
        if not topic_arn:
            st.error("SNS_TOPIC_ARN not configured in environment variables.")
            return

        ok = _publish_to_sns(topic_arn, {"name": name, "email": email})
    if ok:
        import utils
        utils.log_visitor({"name": name, "email": email})
        st.success("Thanks! We will email you an access link once approved.")
        st.balloons()

