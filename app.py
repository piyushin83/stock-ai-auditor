import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

# 1. UI SETUP
st.set_page_config(page_title="Universal Stock Logic V8.4", layout="wide")

# (CSS remains the same for that sleek look)
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    .phase-card { background-color: #f4f6f9; color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; }
    .news-card { background-color: #ffffff; color: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; }
    .v-green { background-color: #2e7d32; } .v-orange { background-color: #f57c00; } .v-red { background-color: #c62828; }
</style>
""", unsafe_allow_html=True)

# 2. THE ROBUST DATA ENGINE
def get_clean_data(user_input):
    """Attempt to get data using direct ticker or keyword search."""
    ticker_str = user_input.strip().upper()
    
    # Try direct fetch first
    t_obj = yf.Ticker(ticker_str)
    df = t_obj.history(period="5y")
    
    # If direct fetch fails, try a quick search to find the ticker
    if df.empty:
        search = yf.Search(ticker_str, max_results=1)
        if search.tickers:
            ticker_str = search.tickers[0]['symbol']
            t_obj = yf.Ticker(ticker_str)
            df = t_obj.history(period="5y")
    
    if df.empty:
        return None, None, None
        
    df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'vol'})
    df['ds'] = df['ds'].dt.tz_localize(None) # Prophet hates timezones
    
    # Get Fundamental Info
    info = t_obj.info
    health = {
        "Name": info.get('longName', ticker_str),
        "ROE": info.get('returnOnEquity', 0),
        "PB": info.get('priceToBook', 0),
        "Debt": info.get('debtToEquity', 0),
        "Currency": info.get('currency', 'USD')
    }
    return df, health, ticker_str

def get_news_sentiment(ticker):
    """Fetches and scores recent news headlines."""
    try:
        t_obj = yf.Ticker(ticker)
        news = t_obj.news[:5] # yfinance's built-in news is more stable
        if not news: return 0, ["No recent news found."]
        
        parsed_news = []
        total_sentiment = 0
        for n in news:
            title = n['title']
            score = TextBlob(title).sentiment.polarity
            total_sentiment += score
            icon = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
            parsed_news.append(f"{icon} {title}")
        return (total_sentiment / len(news)), parsed_news
    except:
        return 0, ["⚠️ News Feed Temporarily Unavailable"]

# 3. SIDEBAR & INPUT
st.sidebar.header("🏛️ Analysis Suite")
raw_input = st.sidebar.text_input("Enter Ticker (e.g. NVDA, AAPL, MSFT)", value="AAPL")
capital = st.sidebar.number_input("Available Capital", value=10000)

if st.sidebar.button("🚀 Analyze Stock"):
    with st.spinner(f"Auditing {raw_input}..."):
        df, health, ticker = get_clean_data(raw_input)
        
        if df is not None:
            # TECHNICALS
            df['MA50'] = df['y'].rolling(window=50).mean()
            df['MA200'] = df['y'].rolling(window=200).mean()
            is_death_cross = df['MA50'].iloc[-1] < df['MA200'].iloc[-1]
            
            # AI SYNCED PROJECTION
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            # Correction Logic: If Death Cross, apply a 7% 'Risk Penalty' to the AI target
            ai_raw_target = forecast['yhat'].iloc[-1]
            ai_final_target = ai_raw_target * 0.93 if is_death_cross else ai_raw_target
            current_p = df['y'].iloc[-1]
            roi = ((ai_final_target - current_p) / current_p) * 100
            
            sentiment_score, headlines = get_news_sentiment(ticker)
            
            # SCORING (Logic-Based)
            score = 50 # Base
            if not is_death_cross: score += 20
            if health['ROE'] > 0.15: score += 15
            if sentiment_score > 0: score += 15
            score = max(0, min(100, score))
            
            # UI OUTPUT
            st.title(f"{health['Name']} ({ticker})")
            
            if is_death_cross:
                st.warning("⚠️ TECHNICAL ALERT: Death Cross detected. AI targets have been adjusted downward for risk.")
            else:
                st.success("✅ BULLISH STRUCTURE: Short-term momentum is above long-term averages.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Logic Score", f"{score}/100")
            col2.metric("Current Price", f"{current_p:.2f} {health['Currency']}")
            col3.metric("AI Target (90d)", f"{ai_final_target:.2f}", f"{roi:.1f}%")

            # Verdict
            if score > 70: st.markdown('<div class="verdict-box v-green">VERDICT: STRONG BUY</div>', unsafe_allow_html=True)
            elif score > 40: st.markdown('<div class="verdict-box v-orange">VERDICT: HOLD / NEUTRAL</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="verdict-box v-red">VERDICT: AVOID / SELL</div>', unsafe_allow_html=True)

            # News & Stats
            l, r = st.columns(2)
            with l:
                st.subheader("🏥 Fundamental Health")
                st.write(f"**ROE:** {health['ROE']:.2%}")
                st.write(f"**P/B Ratio:** {health['PB']:.2f}")
                st.write(f"**Debt/Equity:** {health['Debt']:.2f}")
            with r:
                st.subheader("📰 Recent Sentiments")
                for h in headlines: st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)

            # Chart
            st.subheader("🤖 AI Projection & Trend Lines")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['ds'], df['y'], label="Actual Price", color="black", alpha=0.6)
            ax.plot(df['ds'], df['MA50'], label="50-day MA", color="orange")
            ax.plot(df['ds'], df['MA200'], label="200-day MA", color="red")
            ax.plot(forecast['ds'].tail(90), forecast['yhat'].tail(90), label="AI Forecast", linestyle="--", color="#0288d1")
            plt.legend()
            st.pyplot(fig)
            
        else:
            st.error("Could not find data for that ticker. Please check the symbol (e.g., TSLA, BTC-USD, or NVDA).")
