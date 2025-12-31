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
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://finance.yahoo.com/"
    })
    return session

# 2. DATA BYPASS LOGIC
def fetch_stock_data_direct(ticker):
    """Bypasses yfinance library entirely to avoid RateLimitError"""
    sess = get_secure_session()
    
    # Get Price History via Download URL (More resilient)
    end_time = int(time.time())
    start_time = end_time - (5 * 365 * 24 * 60 * 60) # 5 years
    download_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history"
    
    res = sess.get(download_url, timeout=15)
    if res.status_code != 200:
        return None, 0.12, 0.5, 0 # Error
    
    df = pd.read_csv(io.StringIO(res.text))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Get Stats via Scraping
    stats_url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    stats_res = sess.get(stats_url, timeout=15)
    stats_text = stats_res.text
    
    # Simple extraction for ROE and Debt
    roe = re.search(r'Return on Equity.*?([\d\.]+)%', stats_text)
    roe_val = float(roe.group(1))/100 if roe else 0.12
    
    d_e = re.search(r'Total Debt/Equity.*?([\d\.]+)', stats_text)
    d_e_val = float(d_e.group(1))/100 if d_e else 0.5
    
    # Get Sentiment via News Scraping
    try:
        news_url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
        news_res = sess.get(news_url, timeout=10)
        soup = BeautifulSoup(news_res.text, 'html.parser')
        headlines = [g.text for g in soup.find_all('div', dict(role='heading'))]
        sent_scores = [sia.polarity_scores(h)['compound'] for h in headlines]
        sentiment = 1 if (sum(sent_scores)/len(sent_scores) if sent_scores else 0) > 0.05 else 0
    except:
        sentiment = 0

    return df, roe_val, d_e_val, sentiment

# 3. UI SETUP
st.set_page_config(page_title="Master AI Terminal", layout="wide")
st.title("🏛️ Master AI Investment Terminal")

# SIDEBAR
st.sidebar.header("⚙️ System Parameters")
stock_symbol = st.sidebar.text_input("Stock Symbol", value="NVDA").upper()
total_capital = st.sidebar.number_input("Total Capital ($)", value=1000)

if st.sidebar.button("🔍 Run Deep Audit"):
    with st.spinner(f"🚀 Executing Raw Scrape for {stock_symbol}..."):
        df, roe, de, sent = fetch_stock_data_direct(stock_symbol)
        
        if df is not None and not df.empty:
            # AI Forecast
            df_p = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df_p)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            cur_p = df['Close'].iloc[-1]
            roi = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # SCORING
            f_score = 1 if (roe > 0.15 and de < 1.5) else 0
            ai_score = 1 if roi > 10 else 0
            points = f_score + sent + ai_score
            
            # LOGIC
            if points == 3:
                action = "🌟 HIGH CONVICTION BUY"; imm = 150
            elif points >= 1:
                action = "🟡 ACCUMULATE / HOLD"; imm = 50
            else:
                action = "🛑 AVOID"; imm = 0
            
            # UI
            if points == 3: st.success(f"### {action}")
            elif points >= 1: st.warning(f"### {action}")
            else: st.error(f"### {action}")
            
            st.info(f"Financials: {'Strong' if f_score else 'Caution'} | Sentiment: {'Bullish' if sent else 'Neutral'} | ROI: {roi:+.1f}%")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Points", f"{points}/3")
            c2.metric("Immediate Buy", f"${imm:.2f}")
            c3.metric("Current Price", f"${cur_p:.2f}")

            st.markdown("---")
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("🚀 Phase 1")
                st.write(f"Invest **${imm:.2f}** today.")
                st.error(f"Stop-Loss: ${cur_p * 0.88:.2f}")
            with col_r:
                st.subheader("⏳ Phase 2")
                st.write(f"Reserve **${total_capital - imm:.2f}**.")
                if points == 3:
                    st.write("Aggressive: Phase in over 2 months. Double if price drops 5%.")
                else:
                    st.write("Defensive: Phase in over 4 months.")

            st.markdown("---")
            st.pyplot(m.plot(forecast))
        else:
            st.error("⚠️ Yahoo is blocking the direct scrape. Please try again in 1 minute.")
