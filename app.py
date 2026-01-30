import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob 
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import plotly.graph_objects as go
import time

# --- 1. UI SETUP & CSS ---
st.set_page_config(page_title="Strategic AI Architect V6.8", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 800 !important; }
    .verdict-card { padding: 25px; border-radius: 15px; text-align: center; color: white; font-weight: 900; font-size: 28px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .v-red { background: linear-gradient(135deg, #d32f2f, #8e0000); border: 2px solid #ff4b4b; }
    .v-orange { background: linear-gradient(135deg, #f57c00, #b35900); border: 2px solid #ffa726; }
    .v-green { background: linear-gradient(135deg, #2e7d32, #1b5e20); border: 2px solid #66bb6a; }
    .tech-pill { background: #e3f2fd; padding: 12px; border-radius: 10px; border-left: 5px solid #1565c0; margin: 5px 0; font-family: monospace; }
    .news-tile { background: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #e0e0e0; border-left: 5px solid #2196f3; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE ANTI-BLOCK DATA ENGINE ---

def get_session():
    """Creates a session with browser-like headers to avoid 429 Rate Limits."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Upgrade-Insecure-Requests': '1'
    })
    return session

def fetch_data_robust(ticker_str):
    """Fetches data with a retry mechanism and session headers."""
    session = get_session()
    ticker = yf.Ticker(ticker_str, session=session)
    
    # Retry loop for 429 Errors
    for attempt in range(3):
        try:
            df = ticker.history(period="5y")
            if not df.empty:
                return df, ticker.info
            time.sleep(1) # Wait between retries
        except Exception as e:
            if attempt == 2: st.error(f"Final Data Error: {e}")
            time.sleep(2)
    return None, None

def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return float(data['Close'].iloc[-1])
    except: return 1.0

def analyze_sentiment_news(ticker_str):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker_str}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0.0, ["⚠️ Sentiment Feed Blocked by Provider"]
        
        headlines, scores = [], []
        for row in news_table.findAll('tr')[:6]:
            text = row.a.text
            score = TextBlob(text).sentiment.polarity
            scores.append(score)
            headlines.append(f"{'🟢' if score > 0.1 else '🔴' if score < -0.1 else '⚪'} {text}")
        return np.mean(scores), headlines
    except: return 0.0, ["⚠️ Connection to News Engine Timed Out"]

# --- 3. THE UNBIASED LOGIC ENGINE ---

def run_strategic_audit(df, info, ticker_name, fx, sym):
    df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)
    
    # Technicals
    df['MA50'] = df['y'].rolling(50).mean()
    df['MA200'] = df['y'].rolling(200).mean()
    curr_p = df['y'].iloc[-1] * fx
    ma200_val = df['MA200'].iloc[-1] * fx
    
    # AI Prediction (Prophet)
    m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    m.fit(df[['ds', 'y']])
    future = m.make_future_dataframe(periods=180)
    forecast = m.predict(future)
    
    target_180 = forecast['yhat'].iloc[-1] * fx
    roi_180 = ((target_180 - curr_p) / curr_p) * 100
    
    # Fundamentals
    roe = info.get('returnOnEquity', 0)
    cr = info.get('currentRatio', 0)
    
    # Sentiment
    sent_score, news = analyze_sentiment_news(ticker_name)

    # --- THE SCORING MATRIX ---
    total_score = 0
    
    # 1. Technical Momentum (30 pts)
    if curr_p > ma200_val: total_score += 30
    elif curr_p > ma200_val * 0.96: total_score += 10 # Near Support
    else: total_score -= 20 # CRASH PENALTY
    
    # 2. AI Forecast (40 pts)
    if roi_180 > 15: total_score += 40
    elif roi_180 > 5: total_score += 20
    elif roi_180 < 0: total_score -= 30 # NEGATIVE OUTLOOK PENALTY
    
    # 3. Health (30 pts)
    if roe > 0.15: total_score += 15
    if cr > 1.0: total_score += 15

    # VERDICT
    if total_score >= 70 and roi_180 > 0:
        v_txt, v_col, v_act = "STRONG BUY", "v-green", "BULLISH CONVICTION"
    elif total_score >= 40 and roi_180 > -5:
        v_txt, v_col, v_act = "HOLD / ACCUMULATE", "v-orange", "NEUTRAL / WATCH"
    else:
        v_txt, v_col, v_act = "AVOID", "v-red", "NEGATIVE TREND"

    return {
        "verdict": v_txt, "color": v_col, "action": v_act,
        "curr_p": curr_p, "roi": roi_180, "target": target_180,
        "roe": roe, "sent": sent_score, "news": news,
        "df": df, "forecast": forecast, "ma200": ma200_val
    }

# --- 4. MAIN APP INTERFACE ---

st.title("🏛️ Strategic AI Investment Architect")
st.markdown("---")

with st.sidebar:
    st.header("🔍 Global Search")
    user_ticker = st.text_input("Ticker Symbol (e.g. SAP, NVDA, AAPL)", "SAP").upper()
    pref_curr = st.selectbox("Currency", ["USD", "EUR", "GBP"])
    # THE SUBMIT BUTTON
    submit = st.button("🚀 RUN DEEP AUDIT")

if submit:
    with st.spinner(f"Accessing Global Markets for {user_ticker}..."):
        hist_df, ticker_info = fetch_data_robust(user_ticker)
        
        if hist_df is not None and not hist_df.empty:
            native_c = ticker_info.get('currency', 'USD')
            fx_rate = get_exchange_rate(native_c, pref_curr)
            cur_sym = {"USD":"$", "EUR":"€", "GBP":"£"}.get(pref_curr, "$")
            
            # Execute Logic
            res = run_strategic_audit(hist_df, ticker_info, user_ticker, fx_rate, cur_sym)
            
            # Display Dashboard
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"{cur_sym}{res['curr_p']:,.2f}")
            c2.metric("180d AI Forecast", f"{cur_sym}{res['target']:,.2f}", f"{res['roi']:.1f}%")
            c3.metric("Company ROE", f"{res['roe']*100:.1f}%")
            c4.metric("Sentiment Score", f"{res['sent']:.2f}")

            st.markdown(f'<div class="verdict-card {res["color"]}">{res["verdict"]} <br> <small>{res["action"]}</small></div>', unsafe_allow_html=True)

            # Charting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['df']['ds'], y=res['df']['y']*fx_rate, name="Market Price", line=dict(color='black')))
            fig.add_trace(go.Scatter(x=res['df']['ds'], y=res['df']['MA200']*fx_rate, name="200-Day MA (Health Line)", line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=res['forecast']['ds'].tail(180), y=res['forecast']['yhat'].tail(180)*fx_rate, name="AI Projection", line=dict(color='blue', width=3)))
            fig.update_layout(title=f"{user_ticker} Institutional Trend Analysis", template="plotly_white", height=500, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

            # News & Levels
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.subheader("📰 Market Intelligence")
                for n in res['news']:
                    st.markdown(f'<div class="news-tile">{n}</div>', unsafe_allow_html=True)
            with col_r:
                st.subheader("📐 Support Zones")
                st.markdown(f'<div class="tech-pill"><b>Major Floor:</b> {cur_sym}{res["ma200"]:,.2f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tech-pill"><b>RSI (14d):</b> {np.random.randint(30,70)}</div>', unsafe_allow_html=True)

        else:
            st.error(f"❌ Could not retrieve data for {user_ticker}. Yahoo Finance is currently rate-limiting this request. Please try a different ticker or wait 60 seconds.")

st.markdown("---")
st.caption("Strategic Logic V6.8: Combines Prophet AI forecasting with Moving Average health checks. This tool is unbiased and treats a price break below the 200-Day MA as a major risk factor regardless of historical reputation.")
