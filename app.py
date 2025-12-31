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

# 1. PERMANENT SYSTEM SETUP (Bypasses Yahoo Blocks)
@st.cache_resource
def load_essentials():
    nltk.download('vader_lexicon', quiet=True)
    # Mimics a real Chrome browser to avoid YFRateLimitError
    session = requests.Session(impersonate="chrome")
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Referer": "https://finance.yahoo.com/"
    })
    sia = SentimentIntensityAnalyzer()
    return session, sia

session, sia = load_essentials()

# 2. UI CONFIGURATION
st.set_page_config(page_title="Strategic AI Stock Auditor", layout="wide")
st.title("🏛️ Strategic AI Investment Architect")

# --- TOP LEVEL DISCLAIMER ---
st.warning("⚠️ **HUMAN INTELLIGENCE REQUIRED:** This is an AI-powered experiment. Algorithm-generated forecasts can be wrong. You must use your own brain and due diligence before investing real capital.")

st.markdown("---")

# 3. SIDEBAR CONTROLS
st.sidebar.header("📍 Configuration")
stock_symbol = st.sidebar.text_input("Enter Ticker (e.g., NVDA, TSLA)", value="NVDA").upper()
total_budget = st.sidebar.number_input("Total Investment Budget ($)", value=1000)
target_days = st.sidebar.slider("ROI Target Window (Days)", 30, 90, 90)

st.sidebar.markdown("---")
st.sidebar.info("🤖 **AI Horizon:** Fixed at 180 Days")
st.sidebar.caption("The graph shows 180 days, but your Conviction Score is based on the slider above.")

# 4. MAIN AUDIT LOGIC
if st.sidebar.button("🚀 Run Full Strategic Audit"):
    with st.spinner(f'Analyzing {stock_symbol}... auditing 180-day trajectory...'):
        try:
            # A. SECURE DATA FETCH
            t = yf.Ticker(stock_symbol, session=session)
            hist = t.history(period="2y") # 2 years is optimal for Prophet stability
            
            if hist.empty:
                st.error("❌ Yahoo Finance refused the connection. Please wait 30 seconds and try again.")
            else:
                # B. FINANCIAL HEALTH METRICS
                info = t.info
                roe = info.get('returnOnEquity', 0.12) 
                d_e = info.get('debtToEquity', 100) / 100
                
                # C. AI PROPHET FORECASTING (180 Days)
                df_p = hist.reset_index()[['Date', 'Close']]
                df_p.columns = ['ds', 'y']
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                
                m = Prophet(daily_seasonality=False, yearly_seasonality=True, interval_width=0.8)
                m.fit(df_p)
                
                # We ALWAYS project 180 days as requested
                future_180 = m.make_future_dataframe(periods=180)
                forecast_180 = m.predict(future_180)
                
                # Calculations
                price_now = hist['Close'].iloc[-1]
                target_idx = -(180 - target_days)
                price_at_target = forecast_180.iloc[target_idx]['yhat']
                pred_roi = ((price_at_target - price_now) / price_now) * 100

                # PROBABILITY MATH (Based on 180-day volatility)
                final_lower = forecast_180.iloc[-1]['yhat_lower']
                final_upper = forecast_180.iloc[-1]['yhat_upper']
                final_mean = forecast_180.iloc[-1]['yhat']
                std_dev = (final_upper - final_lower) / 2.56
                prob_success = (1 - (0.5 * (1 + np.erf((price_now - final_mean) / (std_dev * np.sqrt(2)))))) * 100

                # D. STRATEGIC ALLOCATION ENGINE
                score = 0
                if pred_roi > 5: score += 40 
                if roe > 0.15: score += 30
                if d_e < 1.5: score += 30
                
                immediate_cash = total_budget * (score / 100)
                remaining_cash = total_budget - immediate_cash
                
                # --- E. DASHBOARD UI RENDERING ---
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
                    st.success(f"**BUY NOW:** Invest **${immediate_cash:.2f}** today.")
                    st.write(f"This uses {score}% of your total ${total_budget} budget.")
                    
                with col_right:
                    st.subheader(f"⏳ PHASE 2: STRATEGIC STAGING (${remaining_cash:.2f})")
                    if score < 50:
                        st.warning("**STRATEGY: Defensive Staging**")
                        st.write(f"- Park **${remaining_cash * 0.8:.2f}** in a cash fund (SGOV).")
                        st.write("- Only deploy more if AI Conviction rises next month.")
                    else:
                        st.info("**STRATEGY: Aggressive Accumulation**")
                        st.write(f"- Deploy **${remaining_cash:.2f}** over the next 8 weeks.")
                
                st.markdown("---")
                st.subheader("🤖 180-DAY AI PRICE PROJECTION")
                fig1 = m.plot(forecast_180)
                # Adds a red line for your target window
                plt.axvline(forecast_180.iloc[target_idx]['ds'], color='red', linestyle='--', label='Target')
                plt.title(f"{stock_symbol} 180-Day Trend")
                st.pyplot(fig1)

                # F. FINAL FOOTER DISCLAIMER
                st.markdown("---")
                st.markdown(
                    """
                    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #eeeeee;">
                        <p style="font-size: 0.8em; color: #666; text-align: center;">
                            <b>LEGAL DISCLOSURE:</b> This report is for educational purposes only. Investing in stocks carries significant risk. 
                            AI forecasts are mathematical probabilities, not guarantees. <b>Always use your brain before investing.</b>
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error("⚠️ The financial data server is currently congested. Please wait a moment and try again.")
else:
    st.info("👈 Enter a ticker and click 'Run Strategic Audit' to begin.")
