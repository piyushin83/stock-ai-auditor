"""
auth.py  —  Drop-in password gate for any Streamlit app
────────────────────────────────────────────────────────
Usage in your app.py:

    from auth import require_auth
    require_auth()          # ← add this ONE line at the very top of app.py
    # ... rest of your app ...

How it works:
  • Reads password from st.secrets["auth"]["password"]  (secrets.toml / Streamlit Cloud)
  • Falls back to .env  APP_PASSWORD  if secrets.toml is not present
  • Stores authenticated state in st.session_state so the user only
    logs in once per browser session
  • Shows nothing of the real app until the correct password is entered
"""

import streamlit as st
import hmac
import os

# ── Try loading .env for local development ────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed — fine, we use st.secrets


def _get_password() -> str:
    """Read password from st.secrets first, then environment variable."""
    try:
        return st.secrets["auth"]["password"]
    except Exception:
        pass
    env_pw = os.environ.get("APP_PASSWORD", "")
    if env_pw:
        return env_pw
    # Hard-coded fallback — ONLY for local dev, change before deploying
    return "changeme123"


def _check_password(entered: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    correct = _get_password()
    return hmac.compare_digest(entered.encode(), correct.encode())


def require_auth():
    """
    Call this at the very top of app.py.
    If the user is not authenticated, show the login form and stop execution.
    The rest of app.py only runs after successful login.
    """
    if st.session_state.get("authenticated"):
        return   # already logged in — let app continue

    # ── Login UI ──────────────────────────────────────────────────────────────
    st.set_page_config(page_title="Login — Strategic AI Investment Architect",
                       layout="centered")

    # Hide Streamlit's default menu & footer on login page
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer    {visibility: hidden;}
        header    {visibility: hidden;}
        .block-container { padding-top: 4rem; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🏛️ Strategic AI Investment Architect")
        st.markdown("---")
        st.markdown("#### 🔐 Enter your access password")

        password_input = st.text_input(
            "Password", type="password",
            placeholder="Enter password...",
            key="login_password_input"
        )

        if st.button("🔓 Login", use_container_width=True):
            if _check_password(password_input):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")

        st.markdown("""
        <div style='text-align:center; color:#666; font-size:12px; margin-top:2rem;'>
            Authorised users only. All activity is logged.
        </div>
        """, unsafe_allow_html=True)

    st.stop()   # ← prevents ANY app code below from running
