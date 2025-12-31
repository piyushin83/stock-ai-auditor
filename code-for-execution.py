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

# 1. PERMANENT SYSTEM SETUP
@st.cache_resource
def load_essentials():
    nltk.download('vader_lexicon', quiet=True)
    session = requests.Session(impersonate="chrome")
    sia = SentimentIntensityAnalyzer()
    return session, sia

session, sia = load_essentials()

# 2. UI CONFIGURATION
st.set_page_config(page_title="Strategic AI Stock Auditor", layout="wide")
st.title("🏛️ Strategic AI Investment Architect")
st.markdown("---")

# 3. SIDEBAR CONTROLS
st.sidebar.header("📍 Configuration")
stock_symbol = st.sidebar.text_input("Enter Ticker (e.g., NVDA, TSLA)", value="NVDA").upper()
total_budget = st.sidebar.number_input("Total Investment Budget ($)", value=1000)
target_days = st.sidebar.slider("ROI Target Window (Days)", 30, 90, 90)

if st.sidebar.button("🚀 Run Full Strategic Audit"):
    with st.spinner(f'Analyzing {stock_symbol}... auditing financials and 180-day AI trajectory...'):
        
        # A. FETCH DATA
        t = yf.Ticker(stock_symbol, session=session)
        hist = t.history(period="5y")
        
        if hist.empty:
            st.error("⚠️ Connection Error: Ticker is invalid or Yahoo is blocking.")
        else:
            # B. FINANCIAL AUDIT
            info = t.info
            roe = info.get('returnOnEquity', 0.10) 
            d_e = info.get('debtToEquity', 100) / 100
            
            # C. AI PROPHET FORECASTING (180 Days)
            df_p = hist.reset_index()[['Date', 'Close']]
            df_p.columns = ['ds', 'y']
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=True, interval_width=0.8) # 80% confidence interval
            m.fit(df_p)
            
            future_180 = m.make_future_dataframe(periods=180)
            forecast_180 = m.predict(future_180)
            
            price_now = hist['Close'].iloc[-1]
            
            # ROI for the User's Target Window (e.g., 90 days)
            target_idx = -(180 - target_days)
            price_at_target = forecast_180.iloc[target_idx]['yhat']
            pred_roi = ((price_at_target - price_now) / price_now) * 100

            # PROBABILITY OF SUCCESS CALCULATION
            # We look at the 180-day upper and lower bounds
            final_lower = forecast_180.iloc[-1]['yhat_lower']
            final_upper = forecast_180.iloc[-1]['yhat_upper']
            final_mean = forecast_180.iloc[-1]['yhat']
            
            # Estimate probability that price > price_now using normal distribution of the interval
            std_dev = (final_upper - final_lower) / 2.56 # derived from 80% interval
            prob_success = (1 - (0.5 * (1 + np.erf((price_now - final_mean) / (std_dev * np.sqrt(2)))))) * 100

            # D. STRATEGIC ALLOCATION ENGINE
            score = 0
            if pred_roi > 5: score += 40 
            if roe > 0.15: score += 30
            if d_e < 1.5: score += 30
            
            immediate_cash = total_budget * (score / 100)
            remaining_cash = total_budget - immediate_cash
            
            # --- E. DASHBOARD UI RENDERING ---
            st.markdown(f"### 📊 Strategic Audit for {stock_symbol}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Conviction Score", f"{score}/100")
            c2.metric("Prob. of Profit (180d)", f"{prob_success:.1f}%")
            c3.metric(f"{target_days}-Day ROI", f"{pred_roi:+.1f}%")
            c4.metric("Current Price", f"${price_now:.2f}")

            # DOWNLOADABLE PLAN
            rpt = f"STRATEGIC INVESTMENT PLAN: {stock_symbol}\\n"
            rpt += f"Conviction: {score}/100 | Success Prob: {prob_success:.1f}%\\n"
            rpt += f"-----------------------------------\\n"
            rpt += f"Immediate Purchase: ${immediate_cash:.2f}\\n"
            rpt += f"Staging/Phase 2: ${remaining_cash:.2f}"
            st.download_button("📥 Download Plan (.txt)", rpt, file_name=f"{stock_symbol}_Audit.txt")

            st.markdown("---")
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("🚀 PHASE 1: IMMEDIATE ACTION")
                st.info(f"**BUY NOW:** Invest **${immediate_cash:.2f}** today.")
                st.write(f"This represents {score}% of your $1,000 budget.")
                
            with col_right:
                st.subheader(f"⏳ PHASE 2: STRATEGIC STAGING (${remaining_cash:.2f})")
                if score < 50:
                    st.warning("**PLAN: Defensive Staging**")
                    st.write(f"- Park **${remaining_cash * 0.8:.2f}** in a high-yield cash fund (SGOV).")
                    st.write("- Wait for better entry signals before deploying more.")
                else:
                    st.success("**PLAN: Aggressive Accumulation**")
                    st.write(f"- Deploy **${remaining_cash:.2f}** over the next 3 months.")
            
            st.markdown("---")
            st.subheader("🤖 180-DAY AI TRAJECTORY & TARGET LINE")
            fig1 = m.plot(forecast_180)
            plt.axvline(forecast_180.iloc[target_idx]['ds'], color='red', linestyle='--', label='Your Target Window')
            st.pyplot(fig1)
else:
    st.info("👈 Enter a ticker and click 'Run Strategic Audit' to begin.")
