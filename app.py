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
import scipy.special as sp 
import random
import time

# 1. INITIALIZE ENGINES
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

# 2. UI SETUP
st.set_page_config(page_title="Master AI Terminal", layout="wide")
st.title("🏛️ Master AI Investment Terminal")
st.caption("Point-Based Signal Engine + Phased Investing")

# 3. SIDEBAR
st.sidebar.header("⚙️ System Parameters")
stock_symbol = st.sidebar.text_input("Stock Symbol", value="NVDA").upper()
total_capital = st.sidebar.number_input("Total Capital ($)", value=1000)
target_days = st.sidebar.slider("ROI Target Window (Days)", 30, 90, 90)

# 4. EXECUTION
if st.sidebar.button("🔍 Run Deep Audit"):
    with st.spinner(f"⚙️ Auditing {stock_symbol}..."):
        success = False
        for attempt in range(3):
            try:
                sess = get_secure_session()
                sess.get("https://fc.yahoo.com", timeout=5) 
                t = yf.Ticker(stock_symbol, session=sess)
                hist = t.history(period="5y")
                if not hist.empty:
                    success = True
                    break
            except:
                time.sleep(1)
                continue

        if success:
            # --- INDICATOR 1: FINANCIALS (1 POINT) ---
            info = t.info
            roe = info.get('returnOnEquity', 0)
            debt_to_equity = info.get('debtToEquity', 0) / 100
            fcf = info.get('freeCashflow', 0)
            f_score = (1 if (roe > 0.15 and debt_to_equity < 1.5 and fcf > 0) else 0)

            # --- INDICATOR 2: SENTIMENT (1 POINT) ---
            try:
                news_url = f"https://www.google.com/search?q={stock_symbol}+stock+news&tbm=nws"
                res = sess.get(news_url, timeout=10)
                soup = BeautifulSoup(res.text, 'html.parser')
                headlines = [g.text for g in soup.find_all('div', dict(role='heading'))]
                scores = [sia.polarity_scores(h)['compound'] for h in headlines]
                sentiment_score = (1 if (sum(scores)/len(scores) if scores else 0) > 0.05 else 0)
            except:
                sentiment_score = 0

            # --- INDICATOR 3: AI ROI (1 POINT) ---
            df_p = hist.reset_index()[['Date', 'Close']]
            df_p.columns = ['ds', 'y']
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=True).fit(df_p)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            cur_p = hist['Close'].iloc[-1]
            target_idx = -(180 - target_days)
            roi = ((forecast.iloc[target_idx]['yhat'] - cur_p) / cur_p) * 100
            ai_score = (1 if roi > 10 else 0)

            # --- MASTER SIGNAL CALCULATION ---
            points = f_score + sentiment_score + ai_score
            
            if points == 3:
                action_label = "🌟 ACTION: HIGH CONVICTION BUY"
                confidence_msg = "Confidence: Strong. All three indicators (AI, Financials, Sentiment) are positive."
                immediate_buy = total_capital * 0.15 # Your original 15% logic
            elif points >= 1:
                action_label = "🟡 ACTION: ACCUMULATE / HOLD"
                confidence_msg = "Confidence: Moderate. One or more indicators suggest caution."
                immediate_buy = total_capital * 0.05 # Your original 5% logic
            else:
                action_label = "🛑 ACTION: AVOID"
                confidence_msg = "Confidence: Low. Key indicators are negative or stagnant."
                immediate_buy = 0.00

            parked_cash = total_capital - immediate_buy

            # --- UI RENDERING ---
            st.markdown(f"### 📊 Strategic Portfolio Report: {stock_symbol}")
            
            # Highlight the Signal
            if points == 3: st.success(f"### {action_label}")
            elif points >= 1: st.warning(f"### {action_label}")
            else: st.error(f"### {action_label}")
            
            st.info(confidence_msg)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Points Earned", f"{points}/3")
            c2.metric(f"{target_days}-Day ROI", f"{roi:+.1f}%")
            c3.metric("Financial Health", "STRONG" if f_score == 1 else "CAUTION")
            c4.metric("Market Sentiment", "BULLISH" if sentiment_score == 1 else "NEUTRAL")

            st.markdown("---")
            
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("🚀 PHASE 1: IMMEDIATE")
                st.write(f"**Plan:** Invest **${immediate_buy:.2f}** today.")
                if immediate_buy > 0:
                    st.write(f"Approx **{immediate_buy/cur_p:.2f} shares**.")
                st.error(f"🛡️ **Safety Stop-Loss:** ${cur_p * 0.88:.2f}")
            
            with col_r:
                st.subheader("⏳ PHASE 2: STAGING")
                st.write(f"**Plan:** Reserve **${parked_cash:.2f}**.")
                
                if points == 3:
                    st.success("**💡 Strategy: 'Aggressive Accumulation'**")
                    st.write(f"- Phase in remaining cash over 2 months.")
                    st.write("- **DIP TRIGGER:** If price drops 5%, double the monthly buy.")
                else:
                    st.warning("**💡 Strategy: 'Defensive Staging'**")
                    st.write("- Park cash in SGOV ETF to earn yield.")
                    st.write(f"- Phase in remaining cash over 4 months.")

            st.markdown("---")
            st.subheader("🤖 180-DAY AI PRICE PROJECTION")
            fig = m.plot(forecast)
            plt.axvline(forecast.iloc[target_idx]['ds'], color='red', linestyle='--')
            st.pyplot(fig)
            
        else:
            st.error("⚠️ Connection Error: Yahoo is blocking requests. Please try again.")
