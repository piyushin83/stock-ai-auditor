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

# 1. UI SETUP
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    .phase-card { background-color: #f4f6f9; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 400px; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
    .stop-loss-box { background-color: #fff1f1; border-left: 8px solid #ff4b4b; padding: 15px; color: #b71c1c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Strategic AI Investment Architect (V5.7)")

# 2. ENGINES
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        data = yf.download(f"{from_curr}{to_curr}=X", period="1d", progress=False)
        return float(data['Close'].iloc[-1])
    except: return 1.0

def resolve_ticker(user_input):
    s = yf.Search(user_input, max_results=1)
    if s.tickers:
        res = s.tickers[0]
        t_obj = yf.Ticker(res['symbol'])
        return res['symbol'], res.get('longname', res['symbol']), t_obj.fast_info.get('currency', 'USD')
    return user_input.upper(), user_input.upper(), "USD"

def get_data(ticker):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=1825) # 5 Years for calculation
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty: return None
    df = df.reset_index()
    # Flatten columns if multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    return df

# 3. SIDEBAR
st.sidebar.header("⚙️ Config")
user_query = st.sidebar.text_input("Ticker", value="SAP")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])

if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner("Analyzing Market Cycles..."):
        symbol, name, native_curr = resolve_ticker(user_query)
        df = get_data(symbol)
        
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            cur_p = float(df['y'].iloc[-1]) * fx
            
            # --- CALCULATIONS ---
            # Moving Averages
            df['MA50'] = df['y'].rolling(window=50).mean()
            df['MA200'] = df['y'].rolling(window=200).mean()
            
            # AI Forecast
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            ai_target_180 = float(forecast['yhat'].iloc[-1]) * fx
            ai_roi = ((ai_target_180 - cur_p) / cur_p) * 100
            
            # --- SCORING LOGIC ---
            score = 50 # Base
            if ai_roi > 10: score += 30
            elif ai_roi < 0: score -= 40
            
            ma50_curr = float(df['MA50'].iloc[-1])
            ma200_curr = float(df['MA200'].iloc[-1])
            if ma50_curr > ma200_curr: score += 20 # Golden Cross
            
            # --- UI OUTPUT ---
            if ai_roi < -2: verdict, v_col = "AVOID", "v-red"
            elif score >= 75: verdict, v_col = "STRONG BUY", "v-green"
            elif score >= 50: verdict, v_col = "HOLD / ACCUMULATE", "v-orange"
            else: verdict, v_col = "AVOID", "v-red"
            
            st.subheader(f"📊 {name} ({symbol})")
            st.markdown(f'<div class="verdict-box {v_col}">Verdict: {verdict}</div>', unsafe_allow_html=True)
            
            # --- CHARTING ---
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot AI Prediction
            forecast_plot = forecast.copy()
            forecast_plot[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
            m.plot(forecast_plot, ax=ax)
            
            # Plot Moving Averages
            ax.plot(df['ds'], df['MA50'] * fx, label='50-Day MA', color='orange', alpha=0.8)
            ax.plot(df['ds'], df['MA200'] * fx, label='200-Day MA', color='red', alpha=0.8)
            
            # Zoom to 12-Month Window (6m Past, 6m Future)
            hist_limit = datetime.datetime.now() - datetime.timedelta(days=180)
            fut_limit = datetime.datetime.now() + datetime.timedelta(days=180)
            ax.set_xlim([hist_limit, fut_limit])
            
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
            
        else: st.error("Ticker not found.")
