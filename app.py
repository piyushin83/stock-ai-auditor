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
    # Use a rotating set of identifiers
    session = requests.Session(impersonate="chrome")
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://finance.yahoo.com/quote/NVDA/history"
    })
    return session

# 2. THE NUCLEAR DATA BYPASS
def fetch_data_resilient(ticker):
    sess = get_secure_session()
    
    # FETCH PRICES (Direct CSV Download Bypass)
    try:
        end_time = int(time.time())
        start_time = end_time - (3 * 365 * 24 * 60 * 60) # 3 years
        # This endpoint is much harder for Yahoo to block than the API
        csv_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history&includeAdjustedClose=true"
        
        csv_res = sess.get(csv_url, timeout=15)
        if csv_res.status_code != 200:
            return None, 0.12, 0.5, 0
            
        df = pd.read_csv(io.StringIO(csv_res.text))
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        return None, 0.12, 0.5, 0

    # FETCH STATS (Raw HTML Regex Extraction)
    try:
        stats_url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
        stats_res = sess.get(stats_url, timeout=10)
        # We look for the numbers directly in the HTML string to avoid JSON parsing errors
        roe_match = re.search(r'Return on Equity.*?([\d\.]+)%', stats_res.text)
        roe_val = float(roe_match.group(1))/100 if roe_match else 0.12
        
        debt_match = re.search(r'Total Debt/Equity.*?([\d\.]+)', stats_res.text)
        debt_val = float(debt_match.group(1))/100 if debt_match else 0.5
    except:
        roe_val, debt_val = 0.12, 0.5

    # FETCH SENTIMENT
    try:
        news_url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
        news_res = sess.get(news_url, timeout=10)
        soup = BeautifulSoup(news_res.text, 'html.parser')
        headlines = [g.text for g in soup.find_all('div', dict(role='heading'))]
        scores = [sia.polarity_scores(h)['compound'] for h in headlines]
        sentiment_score = 1 if (sum(scores)/len(scores) if scores else 0) > 0.05 else 0
    except:
        sentiment_score = 0

    return df, roe_val, debt_val, sentiment_score

# 3. UI LAYOUT
st.set_page_config(page_title="Master AI Terminal", layout="wide")
st.title("🏛️ Master AI Investment Terminal")

# SIDEBAR
st.sidebar.header("⚙️ Parameters")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)

if st.sidebar.button("🔍 Run Full Audit"):
    with st.spinner("🚀 Pulling raw market data..."):
        df, roe, de, sent = fetch_data_resilient(stock_symbol)
        
        if df is not None:
            # AI Forecast
            df_p = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df_p)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            cur_p = df['Close'].iloc[-1]
            target_roi = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # POINT LOGIC
            f_score = 1 if (roe > 0.15 and de < 1.5) else 0
            ai_score = 1 if target_roi > 10 else 0
            points = f_score + sent + ai_score
            
            # SIGNAL DETERMINATION
            if points == 3:
                label = "🌟 HIGH CONVICTION BUY"; imm = 0.15
                strat = "Aggressive: Phase in over 2 months. If price drops 5%, double the monthly buy amount."
            elif points >= 1:
                label = "🟡 ACCUMULATE / HOLD"; imm = 0.05
                strat = "
