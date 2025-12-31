import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from curl_cffi import requests
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import scipy.special as sp  # Added for the 'erf' function fix
import random
import time

# 1. THE BYPASS ENGINE
@st.cache_resource
def load_essentials():
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    return sia

sia = load_essentials()

def get_secure_session():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ]
    session = requests.Session(impersonate="chrome")
    session.headers.update({
        "User-Agent": random.choice(user_agents),
        "Accept": "*/*",
        "Referer": "https://finance.yahoo.com/"
    })
    return session

# 2. UI CONFIGURATION
st.set_page_config(page_title="Strategic AI Architect", layout="wide")
st.title("🏛️ Strategic AI Investment Architect")

# DISCLAIMER
st.markdown("""
<div style="background-color: #fff4f4; padding: 10px; border-radius: 5px; border: 1px solid #ffcccc;">
    ⚠️ <b>AI ADVISORY:</b> Forecasts are mathematical probabilities. <b>Human judgment is required</b> before investing.
</div>
""", unsafe_allow_html=True)

# 3. SIDEBAR
st.sidebar.header("📍 Configuration")
stock_symbol = st.sidebar.text_input("Enter Ticker", value="NVDA").upper()
total_budget = st.sidebar.number_input("Total Budget ($)", value=1000)
target_days = st.sidebar.slider("ROI Target Window (Days)", 30, 90, 90)

# 4. MAIN AUDIT
if st.sidebar.button("🚀 Run Full Strategic Audit"):
    success = False
    with st.spinner(f'Establishing encrypted tunnel for {stock_symbol}...'):
        for attempt in range(3):
            try:
                sess = get_secure_session()
                # Bypass handshake
                sess.get("https://fc.yahoo.com", timeout=5) 
                
                t = yf.Ticker(stock_symbol, session=sess)
                hist = t.history(period="2y", interval="1d", timeout=10)
                
                if not hist.empty:
                    success = True
                    break
            except:
                time.sleep(1)
                continue

        if success:
            # --- CALCULATIONS ---
            info = t.info
            roe = info.get('returnOnEquity', 0.12)
            
            df_p = hist.reset_index()[['Date', 'Close']]
            df_p.columns = ['ds', 'y']
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            
            m = Prophet(daily_seasonality=False, yearly_seasonality=True, interval_width=0.8)
            m.fit(df_p)
            
            future_180 = m.make_future_dataframe(periods=180)
            forecast_180 = m.predict(future_180)
            
            price_now = hist['Close'].iloc[-1]
            target_idx = -(180 - target_days)
            price_at_target = forecast_180.iloc[target_idx]['yhat']
            pred_roi = ((price_at_target - price_now) / price_now) * 100

            # FIXED PROBABILITY MATH using scipy.special.erf
            final_lower = forecast_180.iloc[-1]['yhat_lower']
            final_upper = forecast_180.iloc[-1]['yhat_upper']
            final_mean = forecast_180.iloc[-1]['yhat']
            std_dev = (final_upper - final_lower) / 2.56
            
            # Using sp.erf instead of np.erf
            prob_success = (1 - (0.5 * (1 + sp.erf((price_now - final_mean) / (std_dev * np.sqrt(2)))))) * 100

            # ALLOCATION
            score = 0
            if pred_roi > 5: score += 40 
            if roe > 0.15: score += 60
            imm_cash = total_budget * (score / 100)
            rem_cash = total_budget - imm_cash

            # --- DISPLAY ---
            st.markdown(f"### 📊 Deep Audit: {stock_symbol}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Conviction Score", f"{score}/100")
            c2.metric("Prob. of Profit (180d)", f"{prob_success:.1f}%")
            c3.metric(f"{target_days}-Day ROI", f"{pred_roi:+.1f}%")
            c4.metric("Current Price", f"${price_now:.2f}")

            st.markdown("---")
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("🚀 PHASE 1: IMMEDIATE")
                st.success(f"**BUY:** Invest **${imm_cash:.2f}** today.")
            with col_r:
                st.subheader("⏳ PHASE 2: STAGING")
                st.info(f"**HOLD:** Park **${rem_cash:.2f}** in SGOV ETF.")

            st.markdown("---")
            st.subheader("🤖 180-DAY AI PRICE PROJECTION")
            fig1 = m.plot(forecast_180)
            plt.axvline(forecast_180.iloc[target_idx]['ds'], color='red', linestyle='--')
            st.pyplot(fig1)
        else:
            st.error("⚠️ Connection Error: Yahoo is blocking requests. Please try again in 1 minute.")
