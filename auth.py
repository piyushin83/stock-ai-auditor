"""
auth.py  —  Multi-user authentication (Cloud Edition)
═══════════════════════════════════════════════════════
Database : Supabase (PostgreSQL) — cloud, permanent, free
Email    : Resend.com — cloud email API, free 3000/month
Security : bcrypt password hashing, 6-digit OTP, 30min expiry

Usage in app.py (one line at the very top):
    from auth import require_auth, logout_button
    user = require_auth()
"""

import streamlit as st
import bcrypt
import secrets
import string
import os
import re
import datetime
import psycopg2
import psycopg2.extras
import resend

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — reads from Railway environment variables
# ══════════════════════════════════════════════════════════════════════════════

def _db_url():
    return os.environ.get("DATABASE_URL", "")

def _resend_key():
    return os.environ.get("RESEND_API_KEY", "")

def _from_email():
    return os.environ.get("FROM_EMAIL", "onboarding@resend.dev")

def _app_name():
    return os.environ.get("APP_NAME", "Strategic AI Investment Architect")

# ── Hide Streamlit chrome on auth pages ───────────────────────────────────────
_HIDE_CSS = """
<style>
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    header {visibility:hidden;} [data-testid="stToolbar"] {visibility:hidden;}
    .block-container {padding-top:1.5rem;}
    div[data-testid="stForm"] {background:transparent !important;}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def _get_db():
    url = _db_url()
    if not url:
        st.error("DATABASE_URL environment variable not set. Please configure Railway variables.")
        st.stop()
    conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _init_db():
    """Create tables if they don't exist — safe to call on every startup."""
    try:
        conn = _get_db()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                username      TEXT UNIQUE NOT NULL,
                name          TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                verified      INTEGER DEFAULT 0,
                role          TEXT DEFAULT 'user',
                created_at    TIMESTAMP DEFAULT NOW(),
                last_login    TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS otp_tokens (
                id         SERIAL PRIMARY KEY,
                email      TEXT NOT NULL,
                token      TEXT NOT NULL,
                purpose    TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used       INTEGER DEFAULT 0
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        # Tables may already exist — not a fatal error
        pass

_init_db()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _valid_email(email):
    return bool(re.match(r'^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$', email.strip()))

def _valid_username(username):
    return bool(re.match(r'^[a-zA-Z0-9_]{3,20}$', username.strip()))

def _hash_pw(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def _check_pw(pw, hashed):
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False

def _make_otp():
    return ''.join(secrets.choice(string.digits) for _ in range(6))

def _store_otp(email, purpose):
    token   = _make_otp()
    expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE otp_tokens SET used=1 WHERE email=%s AND purpose=%s",
            (email, purpose)
        )
        cur.execute(
            "INSERT INTO otp_tokens (email, token, purpose, expires_at) VALUES (%s,%s,%s,%s)",
            (email, token, purpose, expires)
        )
        conn.commit()
    finally:
        cur.close(); conn.close()
    return token

def _verify_otp(email, token, purpose):
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM otp_tokens
               WHERE email=%s AND token=%s AND purpose=%s AND used=0
               ORDER BY id DESC LIMIT 1""",
            (email, token, purpose)
        )
        row = cur.fetchone()
        if not row:
            return False
        if datetime.datetime.utcnow() > row['expires_at'].replace(tzinfo=None):
            return False
        cur.execute("UPDATE otp_tokens SET used=1 WHERE id=%s", (row['id'],))
        conn.commit()
        return True
    finally:
        cur.close(); conn.close()

def _get_user_by_email(email):
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email.strip().lower(),))
        return cur.fetchone()
    finally:
        cur.close(); conn.close()

def _get_user_by_username(username):
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE LOWER(username)=%s", (username.strip().lower(),))
        return cur.fetchone()
    finally:
        cur.close(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL via Resend
# ══════════════════════════════════════════════════════════════════════════════

def _otp_html(title, subtitle, grad, otp, note):
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;">
      <div style="background:linear-gradient(135deg,{grad});
                  border-radius:12px 12px 0 0;padding:28px;color:#fff;text-align:center;">
        <h2 style="margin:0 0 4px;">🏛️ {_app_name()}</h2>
        <p style="margin:0;opacity:.8;font-size:14px;">{subtitle}</p>
      </div>
      <div style="padding:28px;background:#f8f9fa;border-radius:0 0 12px 12px;border:1px solid #e0e0e0;">
        <p style="font-size:15px;">{title}</p>
        <div style="font-size:40px;font-weight:900;letter-spacing:12px;text-align:center;
                    color:#1565c0;padding:22px;background:#fff;border-radius:10px;
                    border:2px dashed #90caf9;margin:20px 0;">
          {otp}
        </div>
        <p style="color:#777;font-size:13px;">{note}</p>
      </div>
    </div>"""

def _send_email(to, subject, html):
    key = _resend_key()
    if not key:
        # Dev fallback — print to console
        print(f"\n{'='*55}\nEMAIL TO: {to}\nSUBJECT: {subject}\nOTP is in html above\n{'='*55}")
        return True, "dev_mode"
    try:
        resend.api_key = key
        resend.Emails.send({
            "from":    f"{_app_name()} <{_from_email()}>",
            "to":      [to],
            "subject": subject,
            "html":    html,
        })
        return True, "sent"
    except Exception as e:
        return False, str(e)

def _send_verify_email(email, name, otp):
    html = _otp_html(
        f"Hi <b>{name}</b>, your email verification code is:",
        "Verify your email address",
        "#1a237e, #1565c0", otp,
        "This code expires in <b>30 minutes</b>. If you didn't register, ignore this.")
    return _send_email(email, f"Verify your email — {_app_name()}", html)

def _send_reset_email(email, otp):
    html = _otp_html(
        "Your password reset code is:",
        "Password Reset Request",
        "#b71c1c, #c62828", otp,
        "Expires in <b>30 minutes</b>. If you didn't request this, ignore it.")
    return _send_email(email, f"Reset your password — {_app_name()}", html)


# ══════════════════════════════════════════════════════════════════════════════
# USER OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _register_user(email, name, username, password):
    email    = email.strip().lower()
    username = username.strip()

    if not _valid_email(email):
        return False, "Please enter a valid email address."
    if not name.strip():
        return False, "Please enter your full name."
    if not _valid_username(username):
        return False, "Username: 3–20 characters, letters/numbers/underscores only."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    # Check username taken
    if _get_user_by_username(username):
        return False, f"Username '{username}' is already taken — please choose another."
    if _get_user_by_email(email):
        return False, "An account with this email already exists. Try logging in."

    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, username, name, password_hash) VALUES (%s,%s,%s,%s)",
            (email, username, name.strip(), _hash_pw(password))
        )
        conn.commit()
        return True, "ok"
    except Exception as e:
        conn.rollback()
        return False, "Email or username already exists."
    finally:
        cur.close(); conn.close()


def _verify_email_otp(email, otp):
    if not _verify_otp(email.strip().lower(), otp.strip(), "verify"):
        return False, "Invalid or expired code. Request a new one below."
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET verified=1 WHERE email=%s", (email.strip().lower(),))
        conn.commit()
        return True, "ok"
    finally:
        cur.close(); conn.close()


def _login(login_id, password):
    """login_id can be username or email."""
    login_id = login_id.strip()
    user = (_get_user_by_email(login_id.lower())
            if "@" in login_id
            else _get_user_by_username(login_id))
    if not user:
        return False, "No account found. Check your username/email or register.", {}
    if not _check_pw(password, user['password_hash']):
        return False, "Incorrect password.", {}
    if not user['verified']:
        return False, "EMAIL_NOT_VERIFIED", dict(user)
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET last_login=NOW() WHERE email=%s", (user['email'],))
        conn.commit()
    finally:
        cur.close(); conn.close()
    return True, "ok", {
        "email":    user['email'],
        "name":     user['name'],
        "username": user['username'],
        "role":     user['role']
    }


def _reset_password(email, otp, new_pw):
    email = email.strip().lower()
    if len(new_pw) < 8:
        return False, "Password must be at least 8 characters."
    if not _verify_otp(email, otp.strip(), "reset"):
        return False, "Invalid or expired code."
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET password_hash=%s WHERE email=%s", (_hash_pw(new_pw), email))
        conn.commit()
        return True, "ok"
    finally:
        cur.close(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def admin_list_users():
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id,username,name,email,role,verified,created_at,last_login FROM users ORDER BY id")
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close(); conn.close()

def admin_delete_user(email):
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE email=%s", (email.strip().lower(),))
        conn.commit()
        return True, f"{email} deleted."
    finally:
        cur.close(); conn.close()

def admin_set_role(email, role):
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET role=%s WHERE email=%s", (role, email.strip().lower()))
        conn.commit()
        return True, f"{email} role → {role}"
    finally:
        cur.close(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH UI — require_auth()
# ══════════════════════════════════════════════════════════════════════════════

def require_auth():
    """
    Call at the very top of app.py — before any other st.* calls.
    Returns user dict {"email","name","username","role"} when logged in.
    Shows login/register UI and calls st.stop() until authenticated.
    """
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    st.markdown(_HIDE_CSS, unsafe_allow_html=True)
    screen = st.session_state.get("auth_screen", "login")

    _, col, _ = st.columns([1, 2, 1])
    with col:
        # ── Branding ──────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style='text-align:center;margin-bottom:20px;'>
          <div style='font-size:44px;'>🏛️</div>
          <div style='font-size:20px;font-weight:800;color:#1565c0;margin:4px 0;'>
            {_app_name()}
          </div>
          <div style='font-size:13px;color:#888;'>Your personal investment intelligence platform</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Login / Register tabs ─────────────────────────────────────────────
        if screen in ("login", "register"):
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔐  Login", use_container_width=True,
                             type="primary" if screen=="login" else "secondary"):
                    st.session_state["auth_screen"] = "login"; st.rerun()
            with c2:
                if st.button("📝  Register", use_container_width=True,
                             type="primary" if screen=="register" else "secondary"):
                    st.session_state["auth_screen"] = "register"; st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)

        # ── LOGIN ─────────────────────────────────────────────────────────────
        if screen == "login":
            with st.form("login_form", clear_on_submit=False):
                login_id = st.text_input("👤 Username or Email",
                                         placeholder="yourname  or  you@example.com")
                password = st.text_input("🔒 Password", type="password")
                submitted = st.form_submit_button("Login →", use_container_width=True)

            if submitted:
                if not login_id or not password:
                    st.error("Please fill in both fields.")
                else:
                    ok, msg, user = _login(login_id, password)
                    if ok:
                        st.session_state["auth_user"]   = user
                        st.session_state["auth_screen"] = "login"
                        st.rerun()
                    elif msg == "EMAIL_NOT_VERIFIED":
                        st.warning("⚠️ Email not verified. Sending a new code now...")
                        otp = _store_otp(user['email'], "verify")
                        _send_verify_email(user['email'], user['name'], otp)
                        st.session_state["verify_email"] = user['email']
                        st.session_state["auth_screen"]  = "verify"
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔑 Forgot password?", use_container_width=True):
                st.session_state["auth_screen"] = "forgot"; st.rerun()

        # ── REGISTER ──────────────────────────────────────────────────────────
        elif screen == "register":
            st.markdown(
                "<p style='color:#555;font-size:13px;margin-bottom:12px;'>"
                "✅ Free to join — no invite needed.</p>",
                unsafe_allow_html=True)
            with st.form("register_form", clear_on_submit=False):
                name     = st.text_input("👤 Full Name",         placeholder="Jane Smith")
                username = st.text_input("🪪 Choose a Username",
                                         placeholder="janeinvests  (3–20 chars, letters/numbers/_)")
                email    = st.text_input("📧 Email",             placeholder="you@example.com")
                password = st.text_input("🔒 Password",          type="password",
                                         placeholder="Min 8 characters")
                confirm  = st.text_input("🔒 Confirm Password",  type="password")
                submitted = st.form_submit_button("Create My Account →", use_container_width=True)

            if submitted:
                if password != confirm:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, msg = _register_user(email, name, username, password)
                    if ok:
                        otp = _store_otp(email.strip().lower(), "verify")
                        sent, smsg = _send_verify_email(email, name, otp)
                        st.session_state["verify_email"] = email.strip().lower()
                        st.session_state["auth_screen"]  = "verify"
                        if smsg == "dev_mode":
                            st.info("🛠️ Dev mode: OTP printed to server console.")
                        else:
                            st.success(f"🎉 Account created! Check **{email}** for your 6-digit code.")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        # ── VERIFY EMAIL ──────────────────────────────────────────────────────
        elif screen == "verify":
            ve = st.session_state.get("verify_email", "")
            st.markdown(f"""
            <div style='background:#e3f2fd;border-left:4px solid #1565c0;
                        padding:14px;border-radius:6px;margin-bottom:16px;font-size:14px;'>
              📬 A 6-digit code was sent to <b>{ve}</b><br>
              <span style='color:#555;'>Check your inbox and spam folder.</span>
            </div>""", unsafe_allow_html=True)

            with st.form("verify_form", clear_on_submit=False):
                otp = st.text_input("🔢 Enter 6-digit code", placeholder="e.g. 483921", max_chars=6)
                submitted = st.form_submit_button("Verify & Activate →", use_container_width=True)

            if submitted:
                ok, msg = _verify_email_otp(ve, otp)
                if ok:
                    st.success("✅ Email verified! You can now log in.")
                    st.balloons()
                    st.session_state["auth_screen"] = "login"; st.rerun()
                else:
                    st.error(f"❌ {msg}")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📨 Resend code", use_container_width=True):
                row = _get_user_by_email(ve)
                if row:
                    otp  = _store_otp(ve, "verify")
                    sent, smsg = _send_verify_email(ve, row['name'], otp)
                    st.success("✅ New code sent!" if smsg != "dev_mode" else "Dev: check console.")

        # ── FORGOT PASSWORD ───────────────────────────────────────────────────
        elif screen == "forgot":
            st.markdown("#### 🔑 Reset your password")
            with st.form("forgot_form", clear_on_submit=False):
                email     = st.text_input("📧 Your registered email", placeholder="you@example.com")
                submitted = st.form_submit_button("Send Reset Code →", use_container_width=True)

            if submitted and email:
                row = _get_user_by_email(email.strip().lower())
                if row:
                    otp  = _store_otp(email.strip().lower(), "reset")
                    sent, smsg = _send_reset_email(email.strip().lower(), otp)
                    st.session_state["reset_email"] = email.strip().lower()
                    st.session_state["auth_screen"] = "reset"
                    if smsg == "dev_mode": st.info("Dev: check console.")
                    else: st.success(f"✅ Reset code sent to {email}")
                    st.rerun()
                else:
                    st.success("If that email is registered, a code has been sent.")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back to Login", use_container_width=True):
                st.session_state["auth_screen"] = "login"; st.rerun()

        # ── RESET PASSWORD ────────────────────────────────────────────────────
        elif screen == "reset":
            re_email = st.session_state.get("reset_email", "")
            st.markdown(f"""
            <div style='background:#fff3e0;border-left:4px solid #f57c00;
                        padding:14px;border-radius:6px;margin-bottom:16px;font-size:14px;'>
              📬 Reset code sent to <b>{re_email}</b>
            </div>""", unsafe_allow_html=True)

            with st.form("reset_form", clear_on_submit=False):
                otp     = st.text_input("🔢 Reset Code", placeholder="6-digit code", max_chars=6)
                new_pw  = st.text_input("🔒 New Password", type="password", placeholder="Min 8 chars")
                confirm = st.text_input("🔒 Confirm New Password", type="password")
                submitted = st.form_submit_button("Set New Password →", use_container_width=True)

            if submitted:
                if new_pw != confirm:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, msg = _reset_password(re_email, otp, new_pw)
                    if ok:
                        st.success("✅ Password updated! Please log in.")
                        st.session_state["auth_screen"] = "login"; st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back", use_container_width=True):
                st.session_state["auth_screen"] = "login"; st.rerun()

    st.stop()  # ← nothing below runs until authenticated


# ══════════════════════════════════════════════════════════════════════════════
# LOGOUT BUTTON — call in st.sidebar
# ══════════════════════════════════════════════════════════════════════════════

def logout_button():
    user = st.session_state.get("auth_user", {})
    if user:
        st.sidebar.markdown(
            f"<div style='font-size:13px;padding:6px 0;'>"
            f"👤 <b>{user.get('name','')}</b><br>"
            f"<span style='color:#888;font-size:11px;'>@{user.get('username','')} · {user.get('email','')}</span>"
            f"</div>",
            unsafe_allow_html=True)
    if st.sidebar.button("🔒 Logout"):
        st.session_state.pop("auth_user", None)
        st.session_state["auth_screen"] = "login"
        st.rerun()
    st.sidebar.markdown("---")
