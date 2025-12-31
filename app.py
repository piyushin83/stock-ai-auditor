import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from curl_cffi import requests
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import random

# 1. THE BYPASS ENGINE
@st.cache_resource
def load_essentials():
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    return sia

sia = load_essentials()

def get_secure_session():
    """Creates a fresh browser-mimicking session for every request"""
    # Randomize User-Agents to prevent IP-based fingerprinting
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ]
    
    session = requests.Session(impersonate="chrome")
    session.headers.update({
        "User-Agent": random.choice(user_agents),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://finance.yahoo.com",
        "Referer": "https://finance.yahoo.com/",
        "Connection": "keep-alive"
    })
    return session

# 2. UI CONFIGURATION
st.set_page_config(page_title="AI Strategic Architect", layout="wide")
st.title("🏛️ Strategic AI Investment Architect")
st.warning("⚠️ **DISCLAIMER:** AI forecasts are probabilities, not certainties. Use human judgment before investing.")

# 3. SIDEBAR
st.sidebar.header("📍 Configuration")
stock_symbol = st.sidebar.text_input("Enter Ticker", value="NVDA").upper()
total_budget = st.sidebar.number_input("Total Budget ($)", value=1000)
target_days = st.sidebar.slider("ROI Target Window (Days)", 30, 90, 90)

# 4. MAIN AUDIT WITH RETRY LOGIC
if st.sidebar.button("🚀 Run Full Strategic Audit"):
    success = False
    with st.spinner(f'Establishing encrypted tunnel for {stock_symbol}...'):
        # Try up to 3 times automatically
        for attempt in range(3):
            try:
                session = get_secure_session()
                # Warm up the session
                session.get("https://fc.yahoo.com", timeout=5) 
                
                t = yf.Ticker(stock_symbol, session=session)
                # Shorter history (1y) is less likely to trigger blocks
                hist = t.history(period="1y", interval="1d", timeout=10)
                
                if not hist.empty:
                    success = True
                    break
                else:
                    time.sleep(1) # Wait before retry
            except Exception:
                continue

        if success:
            # --- FINANCIAL CALCULATIONS ---
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

            # PROBABILITY MATH
            final_lower = forecast_180.iloc[-1]['yhat_lower']
            final_upper = forecast_180.iloc[-1]['yhat_upper']
            final_mean = forecast_180.iloc[-1]['yhat']
            std_dev = (final_upper - final_lower) / 2.56
            prob_success = (1 - (0.5 * (1 + np.erf((price_now - final_mean) / (std_dev * np.sqrt(2)))))) * 100

            # ALLOCATION ENGINE
            score = 0
            if pred_roi > 5: score += 40 
            if roe > 0.15: score += 60 # Boosted for fundamental strength
            
            imm_cash = total_budget * (score / 100)
            rem_cash = total_budget - imm_cash

            # --- DISPLAY
