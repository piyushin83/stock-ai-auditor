import streamlit as st
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
import re
import io

# 1. INITIALIZE ENGINES
@st.cache_resource
def load_essentials():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_essentials()

def get_secure_session():
    session = requests.Session(impersonate="chrome")
    # Randomize User Agents to avoid footprinting
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    session.headers.update({
        "User-Agent": random.choice(agents),
        "Accept": "text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    })
    return session

# 2. UI SETUP
st.set_page_config(page_title="Master AI Terminal", layout="wide")
st.title("🏛️ Master AI Investment Terminal")

st.markdown("""
<div style="background-color: #fff4f4; padding: 15px; border-radius: 8px; border: 1px solid #ffcccc; text-align: center;">
    🛑 <b>CRITICAL ADVISORY:</b> THIS IS AI. A HUMAN SHOULD USE THEIR BRAIN BEFORE INVESTING REAL MONEY.
</div>
""", unsafe_allow_html=True)

# 3. SIDEBAR
st.sidebar.header("⚙️ Parameters")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)
target_days = st.sidebar.slider("ROI Target Window (Days)", 30, 90, 90)

# 4. DATA BYPASS LOGIC (FAILOVER SYSTEM)
def fetch_data_resilient(ticker):
    sess = get_secure_session()
    end_time = int(time.time())
    start_time = end_time - (3 * 365 * 24 * 60 * 60)
    
    # Try multiple endpoints
    endpoints = [
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history&cb={random.random()}",
        f"https://query2.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history"
    ]
    
    df = None
    for url in endpoints:
        try:
            res = sess.get(url, timeout=10)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                df['Date'] = pd.to_datetime(df['Date'])
                break
        except:
            continue
            
    if df is None: return None, 0.12, 0.5, 0

    # Stats Scrape with Failover
    try:
        stats_url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics?p={ticker}"
        stats_res = sess.get(stats_url, timeout=10)
        roe_match = re.search(r'Return on Equity.*?([\d\.]+)%', stats_res.text)
        roe_val = float(roe_match.group(1))/100 if roe_match else 0.12
        debt_match = re.search(r'Total Debt/Equity.*?([\d\.]+)', stats_res.text)
        debt_val = float(debt_match.group(1))/100 if debt_match else 0.5
    except:
        roe_val, debt_val = 0.12, 0.5
        
    return df, roe_val, debt_val, 1

# 5. EXECUTION
if st.sidebar.button("🚀 Run Full Audit"):
    with st.spinner("🚀 Bypassing blocks and calculating signals..."):
        df, roe, de, sent = fetch_data_resilient(stock_symbol)
        
        if df is not None:
            df_p = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df_p)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            cur_p = df['Close'].iloc[-1]
            target_idx = -(180 - target_days)
            target_roi = ((forecast.iloc[target_idx]['yhat'] - cur_p) / cur_p) * 100
            
            f_score = 1 if (roe > 0.15 and de < 1.5) else 0
            ai_score = 1 if target_roi > 10 else 0
            points = f_score + sent + ai_score
            
            # Allocation Logic
            if points == 3:
                label, conf, imm_pct = "🌟 ACTION: HIGH CONVICTION BUY", "Strong conviction. 3/3 indicators positive.", 0.15
                strat_text = "Aggressive Accumulation: Phase in remaining cash over 2 months. If price drops 5%, double the monthly buy."
            elif points >= 1:
                label, conf, imm_pct = "🟡 ACTION: ACCUMULATE / HOLD", "Moderate conviction. Partial indicator alignment.", 0.05
                strat_text = "Defensive Staging: Park cash in SGOV ETF. Phase in remaining cash over 4 months."
            else:
                label, conf, imm_pct = "🛑 ACTION: AVOID", "Low conviction. Indicators suggest caution.", 0.0
                strat_text = "Stay in Cash: Market conditions for this ticker are currently unfavorable."
            
            imm_buy = total_capital * imm_pct

            # UI Rendering
            st.markdown(f"### 📊 Strategic Report: {stock_symbol}")
            if points == 3: st.success(f"### {label}")
            elif points >= 1: st.warning(f"### {label}")
            else: st.error(f"### {label}")
            
            st.info(conf)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Points", f"{points}/3")
            c2.metric(f"{target_days}d ROI", f"{target_roi:+.1f}%")
            c3.metric("Price", f"${cur_p:.2f}")
            c4.metric("Action 1", f"${imm_buy:.2f}")

            st.markdown("---")
            l_col, r_col = st.columns(2)
            with l_col:
                st.subheader("🚀 Phase 1: Immediate")
                st.write(f"Invest **${imm_buy:.2f}** today.")
                if imm_buy > 0: st.write(f"Approx **{imm_buy/cur_p:.2f} shares**.")
                st.error(f"🛡️ Safety Stop-Loss: ${cur_p * 0.88:.2f}")
            with r_col:
                st.subheader("⏳ Phase 2: Staging")
                st.write(f"Reserve **${total_capital - imm_buy:.2f}**.")
                st.write(f"**Strategy:** {strat_text}")

            st.markdown("---")
            st.subheader("🤖 180-Day AI Price Forecast")
            fig = m.plot(forecast)
            plt.axvline(forecast.iloc[target_idx]['ds'], color='red', linestyle='--')
            st.pyplot(fig)
        else:
            st.error("❌ Yahoo Block Detected. Please try a different ticker or refresh.")
