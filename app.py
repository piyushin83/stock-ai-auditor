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
st.markdown("""
<div style="background-color: #fff4f4; padding: 10px; border-radius: 5px; border: 1px solid #ffcccc;">
    ⚠️ <b>AI ADVISORY:</b> Forecasts are mathematical probabilities. <b>Human judgment is required</b> before investing.
</div>
""", unsafe_allow_html=True)

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
            # --- FINANCIAL AUDIT ---
            info = t.info
            roe = info.get('returnOnEquity', 0)
            debt_to_equity = info.get('debtToEquity', 0) / 100
            fcf = info.get('freeCashflow', 0)
            
            f_score = 0
            if roe > 0.15: f_score += 1
            if debt_to_equity < 1.5: f_score += 1
            if fcf > 0: f_score += 1

            # --- SENTIMENT AUDIT ---
            try:
                news_url = f"https://www.google.com/search?q={stock_symbol}+stock+news&tbm=nws"
                res = sess.get(news_url, timeout=10)
                soup = BeautifulSoup(res.text, 'html.parser')
                headlines = [g.text for g in soup.find_all('div', dict(role='heading'))]
                scores = [sia.polarity_scores(h)['compound'] for h in headlines]
                sentiment = sum(scores)/len(scores) if scores else 0
            except:
                sentiment = 0

            # --- AI PREDICTION (180 Days) ---
            df_p = hist.reset_index()[['Date', 'Close']]
            df_p.columns = ['ds', 'y']
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=True).fit(df_p)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            cur_p = hist['Close'].iloc[-1]
            target_idx = -(180 - target_days)
            roi = ((forecast.iloc[target_idx]['yhat'] - cur_p) / cur_p) * 100
            
            # Probability Math
            final_lower = forecast.iloc[-1]['yhat_lower']
            final_upper = forecast.iloc[-1]['yhat_upper']
            final_mean = forecast.iloc[-1]['yhat']
            std_dev = (final_upper - final_lower) / 2.56
            prob_success = (1 - (0.5 * (1 + sp.erf((cur_p - final_mean) / (std_dev * np.sqrt(2)))))) * 100

            # --- SMART ALLOCATION ENGINE ---
            conviction_score = 0
            if roi > 10: conviction_score += 40
            if f_score >= 2: conviction_score += 40
            if sentiment > 0: conviction_score += 20
            
            immediate_buy = (conviction_score / 100) * total_capital
            parked_cash = total_capital - immediate_buy

            # --- UI RENDERING ---
            st.markdown(f"### 📊 Strategic Portfolio Report: {stock_symbol}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Conviction", f"{conviction_score}/100")
            c2.metric("Prob. of Profit (180d)", f"{prob_success:.1f}%")
            c3.metric(f"{target_days}-Day ROI", f"{roi:+.1f}%")
            c4.metric("Current Price", f"${cur_p:.2f}")

            st.markdown("---")
            
            # STRATEGY BOXES
            col_l, col_r = st.columns(2)
            
            with col_l:
                st.subheader("🚀 PHASE 1: IMMEDIATE")
                st.success(f"**Action 1:** Invest **${immediate_buy:.2f}** today.")
                st.write(f"Buy approx **{immediate_buy/cur_p:.2f} shares**.")
                st.error(f"🛡️ **Safety Stop-Loss:** ${cur_p * 0.88:.2f}")
            
            with col_r:
                st.subheader("⏳ PHASE 2: STAGING")
                st.info(f"**Action 2:** Move **${parked_cash:.2f}** to reserve.")
                
                if conviction_score < 50:
                    st.warning("💡 **Strategy: 'Defensive Staging'**")
                    st.write("- Park cash in **SGOV** (Short-term Treasuries) to earn ~5%.")
                    st.write(f"- Phase in **${parked_cash/4:.2f}/mo** over 4 months.")
                else:
                    st.success("💡 **Strategy: 'Aggressive Accumulation'**")
                    st.write(f"- Phase in **${parked_cash/2:.2f}/mo** over 2 months.")
                    st.write("- **DIP TRIGGER:** If price drops **5%**, **double** the monthly buy amount.")

            st.markdown("---")
            st.subheader("🤖 180-DAY AI PRICE PROJECTION")
            fig = m.plot(forecast)
            plt.axvline(forecast.iloc[target_idx]['ds'], color='red', linestyle='--')
            st.pyplot(fig)
            
        else:
            st.error("⚠️ Connection Error: Yahoo is blocking requests. Please try again in 30 seconds.")
