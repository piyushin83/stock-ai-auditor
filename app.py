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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. RESEARCH & UI CONFIGURATION ---
st.set_page_config(page_title="Strategic AI Architect V6.5", layout="wide")

# Custom CSS for Professional Financial Dashboard
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 800 !important; color: #003366; }
    .verdict-card { padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center; color: white; font-weight: 900; font-size: 26px; border: 2px solid rgba(255,255,255,0.3); }
    .v-red { background: linear-gradient(135deg, #d32f2f, #b71c1c); box-shadow: 0 4px 15px rgba(183,28,28,0.4); }
    .v-orange { background: linear-gradient(135deg, #f57c00, #e65100); box-shadow: 0 4px 15px rgba(230,81,0,0.4); }
    .v-green { background: linear-gradient(135deg, #388e3c, #1b5e20); box-shadow: 0 4px 15px rgba(27,94,32,0.4); }
    .news-tile { background: white; padding: 15px; border-radius: 8px; border-left: 6px solid #1f77b4; margin-bottom: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .tech-box { background: #e3f2fd; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; border: 1px solid #bbdefb; }
</style>
""", unsafe_allow_html=True)

# --- 2. ENGINE CORE FUNCTIONS ---

def get_exchange_rate(from_curr, to_curr):
    """Fetches real-time FX rates to normalize all financial data."""
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 1.0
    except: return 1.0

def resolve_ticker_metadata(user_input):
    """Unbiased Ticker Resolver: Maps names to exchanges and currencies."""
    try:
        s = yf.Search(user_input, max_results=1)
        if s.tickers:
            res = s.tickers[0]
            ticker = res['symbol']
            name = res.get('longname', ticker)
            exch = res.get('exchange', 'NYQ')
            t_obj = yf.Ticker(ticker)
            native_curr = t_obj.fast_info.get('currency', 'USD')
            # Handle Global Suffixes
            suffix_map = {'GER': '.DE', 'FRA': '.DE', 'PAR': '.FR', 'LSE': '.UK', 'HKG': '.HK'}
            suffix = suffix_map.get(exch, "")
            return ticker, name, native_curr
    except: pass
    return user_input.upper(), user_input.upper(), "USD"

def fetch_financial_data(ticker):
    """Retrieves 5 years of historical data and fundamental ratios."""
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period="5y")
        if df.empty: return None, None
        
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)
        
        # Financial Health Dictionary
        info = t_obj.info
        health = {
            "ROE": info.get('returnOnEquity', 0),
            "CurrentRatio": info.get('currentRatio', 0),
            "DebtToEquity": info.get('debtToEquity', 0),
            "OperatingMargin": info.get('operatingMargins', 0),
            "TrailingPE": info.get('trailingPE', 0)
        }
        return df, health
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None, None

def get_sentiment_analysis(ticker):
    """Scrapes news and performs NLP sentiment scoring."""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(req.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0.0, ["No recent news found."]
        
        headlines = []
        sentiments = []
        for row in news_table.findAll('tr')[:8]:
            text = row.a.text
            score = TextBlob(text).sentiment.polarity
            sentiments.append(score)
            icon = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
            headlines.append(f"{icon} {text}")
        
        return np.mean(sentiments), headlines
    except: return 0.0, ["News service currently offline."]

def calculate_fibonacci_levels(df, current_price):
    """Calculates technical retracement levels for entry/exit."""
    recent_high = df['y'].tail(252).max()
    recent_low = df['y'].tail(252).min()
    diff = recent_high - recent_low
    return {
        "Base (1.0)": recent_high,
        "Retrace (0.618)": recent_high - (0.618 * diff),
        "Halfway (0.500)": recent_high - (0.500 * diff),
        "Golden (0.382)": recent_high - (0.382 * diff),
        "Floor (0.0)": recent_low
    }

# --- 3. MAIN DASHBOARD UI ---

st.title("🏛️ Strategic AI Investment Architect")
st.markdown("---")

# Sidebar Controls
with st.sidebar:
    st.header("🔍 Analysis Parameters")
    raw_input = st.text_input("Enter Ticker or Company Name", "SAP")
    target_currency = st.selectbox("Preferred Currency", ["USD", "EUR", "GBP", "JPY"])
    investment_cap = st.number_input("Portfolio Allocation Limit", 1000, 1000000, 10000)
    st.markdown("---")
    st.info("V6.5 uses a **Weighted Hybrid Logic**: Technical Momentum (30%) + AI Forecast (40%) + Financial Health (30%).")

if raw_input:
    ticker, full_name, native_curr = resolve_ticker_metadata(raw_input)
    df, health_stats = fetch_financial_data(ticker)
    
    if df is not None:
        fx_rate = get_exchange_rate(native_curr, target_currency)
        curr_p = df['y'].iloc[-1] * fx_rate
        curr_sym = {"USD":"$","EUR":"€","GBP":"£","JPY":"¥"}.get(target_currency, "$")

        # Technical Indicators
        df['MA50'] = df['y'].rolling(window=50).mean()
        df['MA200'] = df['y'].rolling(window=200).mean()
        rsi_val = 100 - (100 / (1 + (df['y'].diff().where(df['y'].diff() > 0, 0).rolling(14).mean() / 
                                     -df['y'].diff().where(df['y'].diff() < 0, 0).rolling(14).mean())))
        curr_rsi = rsi_val.iloc[-1]

        # AI Forecast (Prophet)
        with st.spinner("Generating AI Prediction..."):
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            # Forecast Normalization
            target_30d = forecast['yhat'].iloc[len(df)+30] * fx_rate
            target_180d = forecast['yhat'].iloc[-1] * fx_rate
            forecast_roi = ((target_180d - curr_p) / curr_p) * 100

        # Sentiment Analysis
        sent_score, news_list = get_sentiment_analysis(ticker)

        # --- 4. UNBIASED STRATEGIC LOGIC ENGINE ---
        # Scoring based on 100 points
        score = 0
        
        # A. Technical Momentum (30 pts)
        # Check if price is above or below 200-Day MA (The "Golden Rule")
        ma200_price = df['MA200'].iloc[-1] * fx_rate
        if curr_p > ma200_price: score += 30
        elif curr_p > (ma200_price * 0.95): score += 15 # Near support
        else: score -= 20 # Severe penalty for breaking long-term support (Crashes)

        # B. AI Forecast (40 pts)
        if forecast_roi > 15: score += 40
        elif forecast_roi > 5: score += 25
        elif forecast_roi > 0: score += 10
        else: score -= 30 # Penalty if AI predicts further decline

        # C. Financial Health (30 pts)
        if health_stats['ROE'] > 0.15: score += 15
        if health_stats['CurrentRatio'] > 1.1: score += 15

        # Constraint: Sentiment Hard-Stop
        if sent_score < -0.2: score -= 10

        # FINAL VERDICT ASSIGNMENT
        if score >= 75:
            verdict, action, v_class, risk = "Strong Buy", "BULLISH CONVICTION", "v-green", "Low"
        elif score >= 45:
            verdict, action, v_class, risk = "Hold / Accumulate", "WATCH FOR DIPS", "v-orange", "Medium"
        else:
            verdict, action, v_class, risk = "Avoid", "NEGATIVE MOMENTUM", "v-red", "High"

        # Special "SAP CRASH" Override: If ROI is negative AND Price < MA200, force AVOID
        if forecast_roi < 0 and curr_p < ma200_price:
            verdict, action, v_class = "Avoid", "CRITICAL TREND BREAK", "v-red"

        # --- 5. DASHBOARD DISPLAY ---
        
        # Row 1: Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"{curr_sym}{curr_p:,.2f}")
        m2.metric("AI 180d Target", f"{curr_sym}{target_180d:,.2f}", f"{forecast_roi:.1f}%")
        m3.metric("Technical Score", f"{score}/100")
        m4.metric("RSI (14d)", f"{curr_rsi:.1f}")

        # Row 2: Strategic Verdict
        st.markdown(f'<div class="verdict-card {v_class}">STRATEGIC VERDICT: {verdict} <br><small>{action}</small></div>', unsafe_allow_html=True)

        # Row 3: Charts & News
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("📈 Technical Trend & AI Projection")
            fig = go.Figure()
            # Historical
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y']*fx_rate, name="Market Price", line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df['ds'], y=df['MA50']*fx_rate, name="50-Day MA", line=dict(color='orange', dash='dot')))
            fig.add_trace(go.Scatter(x=df['ds'], y=df['MA200']*fx_rate, name="200-Day MA", line=dict(color='red')))
            # Forecast
            fig.add_trace(go.Scatter(x=forecast['ds'].tail(180), y=forecast['yhat'].tail(180)*fx_rate, 
                                     name="AI Projection", line=dict(color='blue', width=3)))
            fig.update_layout(height=500, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("📰 Market Sentiment")
            for news in news_list:
                st.markdown(f'<div class="news-tile">{news}</div>', unsafe_allow_html=True)
            
            st.subheader("📐 Fibonacci Support")
            fibs = calculate_fibonacci_levels(df, curr_p)
            for k, v in fibs.items():
                st.markdown(f'<div class="tech-box"><b>{k}:</b> {curr_sym}{v*fx_rate:,.2f}</div>', unsafe_allow_html=True)

    else:
        st.error("Ticker not found. Please verify the symbol and try again.")

# --- 6. FOOTER & RISK ---
st.markdown("---")
st.caption("⚠️ **Risk Disclosure:** This tool uses automated AI forecasting (Prophet) and technical modeling. It is for informational purposes only. Past performance (like high ROE) does not guarantee future results, especially during structural guidance shifts.")
