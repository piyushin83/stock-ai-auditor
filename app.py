import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from curl_cffi import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import time

# 1. ROBUST SYSTEM SETUP (Anti-Blocking Logic)
@st.cache_resource
def load_essentials():
    nltk.download('vader_lexicon', quiet=True)
    # We create a session that mimics a real person browsing Yahoo Finance
    session = requests.Session(impersonate="chrome")
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Referer": "https://finance.yahoo.com/",
        "Accept-Language": "en-US,en;q=0.9"
    })
    sia = SentimentIntensityAnalyzer()
    return session, sia

session, sia = load_essentials()

# 2. UI CONFIGURATION
st.set_page_config(page_title="Strategic AI Stock Auditor", layout="wide")
st.title("🏛️ Strategic AI Investment Architect")

st.warning("⚠️ **HUMAN INTELLIGENCE REQUIRED:** This is an AI-powered experiment. Algorithm-generated forecasts can be wrong. Use your brain before investing.")
st.markdown("---")

# 3. SIDEBAR
st.sidebar.header("📍 Configuration")
stock_symbol = st.sidebar.text_input("Enter Ticker", value="NVDA").upper()
total_budget = st.sidebar.number_input("Total Budget ($)", value=1000)
target_days = st.sidebar.slider("ROI Target Window (Days)", 30, 90, 90)

# 4. MAIN AUDIT
if st.sidebar.button("🚀 Run Full Strategic Audit"):
    with st.spinner(f'Establishing Secure Connection for {stock_symbol}...'):
        try:
            # --- THE FIX: PRE-FETCH COOKIES ---
            # We visit Yahoo's home page first to get the necessary cookies
            session.get("https://finance.yahoo.com", timeout=10)
            
            t = yf.Ticker(stock_symbol, session=session)
            # Fetching 2 years instead of 5 to stay under the radar
            hist = t.history(period="2y", interval="1d")
            
            if hist.empty or len(hist) < 20:
                # If hist is empty, it might be a rate limit or bad ticker
                st.error("❌ Data Fetch Failed. Yahoo is temporarily blocking this request. Please wait 10 seconds and try again.")
            else:
                # B. FINANCIAL HEALTH
                info = t.info
                roe = info.get('returnOnEquity', 0.12)
                d_e = info.get('debtToEquity', 100) / 100
                
                # C. AI PROPHET FORECASTING (180 Days)
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

                # D. STRATEGIC ALLOCATION
                score = 0
                if pred_roi > 5: score += 40 
                if roe > 0.15: score += 30
                if d_e < 1.5: score += 30
                
                imm_cash = total_budget * (score / 100)
                rem_cash = total_budget - imm_cash
                
                # --- E. DASHBOARD UI ---
                st.markdown(f"### 📊 Strategic Audit: {stock_symbol}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Conviction Score", f"{score}/100")
                c2.metric("Prob. of Profit (180d)", f"{prob_success:.1f}%")
                c3.metric(f"{target_days}-Day ROI", f"{pred_roi:+.1f}%")
                c4.metric("Current Price", f"${price_now:.2f}")

                st.markdown("---")
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("🚀 PHASE 1: IMMEDIATE ACTION")
                    st.success(f"**BUY NOW:** Invest **${imm_cash:.2f}** today.")
                    
                with col_right:
                    st.subheader(f"⏳ PHASE 2: STRATEGIC STAGING (${rem_cash:.2f})")
                    if score < 50:
                        st.warning("**STRATEGY: Defensive Staging**")
                        st.write(f"- Park **${rem_cash * 0.8:.2f}** in SGOV ETF.")
                    else:
                        st.info("**STRATEGY: Aggressive Accumulation**")
                        st.write(f"- Deploy **${rem_cash:.2f}** over the next 8 weeks.")
                
                st.markdown("---")
                st.subheader("🤖 180-DAY AI PRICE PROJECTION")
                fig1 = m.plot(forecast_180)
                plt.axvline(forecast_180.iloc[target_idx]['ds'], color='red', linestyle='--', label='Target')
                st.pyplot(fig1)

        except Exception as e:
            st.error(f"⚠️ Connection Error: {str(e)}")
            st.info("Try refreshing the page. This is a common temporary block from Yahoo Finance.")

else:
    st.info("👈 Enter a ticker and click 'Run Strategic Audit' to begin.")
