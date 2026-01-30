import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import yfinance as yf

# --- UI SETUP ---
st.set_page_config(page_title="Strategic AI Architect V7.5", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
    .news-card { background-color: #fff; padding: 12px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .fib-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Strategic AI Investment Architect (V7.5)")

# --- CORE ENGINES ---

def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 1.0
    except: return 1.0

def resolve_ticker(user_input):
    user_input = user_input.strip()
    try:
        s = yf.Search(user_input, max_results=1)
        if s.tickers:
            res = s.tickers[0]
            ticker = res['symbol'].split('.')[0]
            name = res.get('longname', ticker)
            exch = res.get('exchange', 'NYQ')
            t_obj = yf.Ticker(res['symbol'])
            native_curr = t_obj.fast_info.get('currency', 'USD')
            suffix_map = {'GER': '.DE', 'FRA': '.DE', 'LSE': '.UK', 'TSE': '.JP'}
            suffix = suffix_map.get(exch, ".US")
            return ticker, name, suffix, native_curr
    except: pass
    return user_input.upper(), user_input.upper(), ".US", "USD"

def get_sentiment(ticker):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0, []
        
        parsed_news = []
        sentiment_score = 0
        rows = news_table.find_all('tr') # FIXED: find_all instead of findAll
        for row in rows[:6]:
            text = row.a.text
            polarity = TextBlob(text).sentiment.polarity
            sentiment_score += polarity
            icon = "🟢" if polarity > 0.05 else "🔴" if polarity < -0.05 else "⚪"
            parsed_news.append(f"{icon} {text}")
        return (sentiment_score / 6), parsed_news
    except: return 0, ["⚠️ News Feed Unavailable"]

def calculate_fibs(df):
    curr_p = df['y'].iloc[-1]
    low_6m = df['y'].tail(126).min()
    diff = max(curr_p - low_6m, curr_p * 0.1)
    return {
        '0.382': curr_p - (diff * 0.382),
        '0.500': curr_p - (diff * 0.500),
        '0.618': curr_p - (diff * 0.618)
    }

# --- MAIN APP ---

with st.sidebar:
    st.header("⚙️ Configuration")
    user_query = st.text_input("Ticker/Company Name", value="SAP")
    currency = st.selectbox("Currency", ["USD", "EUR"])
    capital = st.number_input("Capital Allocation", value=10000)
    run_btn = st.button("🚀 EXECUTE STRATEGIC AUDIT")

if run_btn:
    with st.spinner("Analyzing Market Data..."):
        ticker, name, suffix, native_curr = resolve_ticker(user_query)
        
        # STOOQ DATA FETCHING
        try:
            start = datetime.datetime.now() - datetime.timedelta(days=1825)
            df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, datetime.datetime.now())
            df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        except: df = None
        
        if df is not None and not df.empty:
            fx = get_exchange_rate(native_curr, currency)
            sym = "€" if currency == "EUR" else "$"
            
            # MAs & Technicals
            df['MA50'] = df['y'].rolling(50).mean()
            df['MA200'] = df['y'].rolling(200).mean()
            cur_p = df['y'].iloc[-1] * fx
            ma200_cur = df['MA200'].iloc[-1] * fx
            
            # AI Forecast
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            target_180 = forecast['yhat'].iloc[-1] * fx
            roi_180 = ((target_180 - cur_p) / cur_p) * 100
            
            # --- UNBIASED VERDICT LOGIC ---
            score = 0
            # Weight 1: Long term trend (30pts)
            if cur_p > ma200_cur: score += 30
            # Weight 2: AI Projection (40pts)
            if roi_180 > 10: score += 40
            elif roi_180 > 0: score += 15
            # Weight 3: Sentiment (15pts)
            sent_val, news_items = get_sentiment(ticker)
            if sent_val > 0: score += 15
            
            # Final Verdict
            if roi_180 < -2 or (cur_p < ma200_cur and roi_180 < 2):
                v_label, v_col, action = "AVOID", "v-red", "STAY AWAY - BEARISH TREND"
            elif score >= 70:
                v_label, v_col, action = "STRONG BUY", "v-green", "BULLISH CONVICTION"
            else:
                v_label, v_col, action = "ACCUMULATE", "v-orange", "BUY THE DIPS"

            # --- DISPLAY ---
            st.subheader(f"📊 {name} ({ticker})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"{sym}{cur_p:,.2f}")
            c2.metric("AI 180d Target", f"{sym}{target_180:,.2f}", f"{roi_180:.1f}%")
            c3.metric("Conviction", f"{score}/100")
            c4.metric("Market Sentiment", f"{sent_val:.2f}")

            st.markdown(f'<div class="verdict-box {v_col}">Verdict: {v_label} | {action}</div>', unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("🤖 AI Forecast & Moving Averages")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['ds'], df['y']*fx, color='black', label="Price", alpha=0.5)
                ax.plot(df['ds'], df['MA200']*fx, color='red', label="200-Day MA (Health)")
                ax.plot(forecast['ds'].tail(180), forecast['yhat'].tail(180)*fx, color='blue', label="AI Prediction")
                ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=180), datetime.datetime.now() + datetime.timedelta(days=180)])
                plt.legend()
                st.pyplot(fig)

            with col_r:
                st.subheader("📰 Market Intelligence")
                for item in news_items:
                    st.markdown(f'<div class="news-card">{item}</div>', unsafe_allow_html=True)
                
                st.subheader("📐 Fibonacci Entry Points")
                fib_vals = calculate_fibs(df)
                for k, v in fib_vals.items():
                    st.markdown(f'<div class="fib-box">Level {k}: {sym}{v*fx:,.2f}</div>', unsafe_allow_html=True)

        else:
            st.error("Stooq Data Engine currently unavailable. Please check the ticker suffix or try again later.")
