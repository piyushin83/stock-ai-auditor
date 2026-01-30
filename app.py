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
    .phase-card { background-color: #f4f6f9; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .news-card { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; }
    .fib-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; font-weight: bold; }
    .stop-loss-box { background-color: #fff1f1; border-left: 8px solid #ff4b4b; padding: 15px; margin-bottom: 20px; color: #b71c1c; font-weight: bold; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Strategic AI Investment Architect (V6.0)")

# 2. HELPER ENGINES
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 1.0
    except: return 1.0

def resolve_smart_ticker(user_input):
    user_input = user_input.strip()
    try:
        s = yf.Search(user_input, max_results=1)
        if s.tickers:
            res = s.tickers[0]
            ticker = res['symbol']
            name = res.get('longname', ticker)
            exch = res.get('exchange', 'NYQ')
            t_obj = yf.Ticker(ticker)
            native_curr = t_obj.fast_info.get('currency', 'USD')
            suffix_map = {'LSE': '.UK', 'GER': '.DE', 'FRA': '.DE', 'PAR': '.FR', 'AMS': '.NL', 'TSE': '.JP', 'HKG': '.HK'}
            suffix = suffix_map.get(exch, ".US")
            return ticker.split('.')[0], name, suffix, native_curr
    except: pass
    return user_input.upper(), user_input.upper(), ".US", "USD"

def get_news_sentiment(ticker):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(req.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0, []
        parsed_news = []
        sentiment_score = 0
        rows = news_table.findAll('tr')
        for index, row in enumerate(rows[:5]):
            text = row.a.text
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            sentiment_score += polarity
            parsed_news.append(f"{'🟢' if polarity > 0 else '🔴' if polarity < 0 else '⚪'} {text}")
        return (sentiment_score / 5), parsed_news
    except: return 0, ["⚠️ News Feed Unavailable"]

def calculate_technicals(df):
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    curr_p = df['y'].iloc[-1]
    recent_low = df['y'].tail(126).min() 
    diff = curr_p - recent_low
    if diff <= 0: diff = curr_p * 0.10
    fib_levels = {'0.382': curr_p - (diff * 0.382), '0.500': curr_p - (diff * 0.500), '0.618': curr_p - (diff * 0.618)}
    return df['rsi'].iloc[-1], fib_levels

def get_fundamental_health(ticker, suffix):
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1825)
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        # Placeholder for fundamental fetching (improve as needed per your API)
        health = {"ROE": 0.16, "Debt": 0.5, "Margin": "25%", "CurrentRatio": "1.2"}
        return df, health
    except: return None, None

# 3. SIDEBAR CONFIG
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker", value="SAP")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

# 4. MAIN EXECUTION
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner("Processing..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)
        
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            
            # MA Calculations
            df['MA50'] = df['y'].rolling(window=50).mean()
            df['MA200'] = df['y'].rolling(window=200).mean()
            
            # AI Forecast (Prophet)
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            # Logic Correction: Targets based on Current Market Price
            target_p_30 = forecast['yhat'].iloc[len(df) + 29] * fx
            target_p_180 = forecast['yhat'].iloc[-1] * fx
            ai_roi_180 = ((target_p_180 - cur_p) / cur_p) * 100
            
            rsi, fibs = calculate_technicals(df)
            news_score, headlines = get_news_sentiment(ticker)
            
            # --- UNBIASED WEIGHTED SCORING LOGIC ---
            total_score = 0
            
            # A. AI Forecast Weight (40%)
            if ai_roi_180 > 15: total_score += 40
            elif ai_roi_180 > 5: total_score += 25
            elif ai_roi_180 > 0: total_score += 10
            else: total_score -= 20 # Penalty for negative forecast
            
            # B. Technical Health Weight (30%) - Price vs 200-Day MA
            ma200_val = df['MA200'].iloc[-1] * fx
            if cur_p > ma200_val: total_score += 30
            else: total_score -= 15 # Penalty for Bearish Regime
            
            # C. Fundamentals Weight (30%)
            if health['ROE'] > 0.15: total_score += 15
            if float(health['CurrentRatio']) > 1.0: total_score += 15
            
            # Apply Constraints
            total_score = max(0, min(100, total_score))
            
            # FINAL VERDICT
            if total_score < 40 or ai_roi_180 < -1:
                verdict, action, v_col, risk, pct = "Avoid", "NEGATIVE TREND", "v-red", "High", 0
            elif total_score < 70:
                verdict, action, v_col, risk, pct = "Accumulate", "BUY DIPS", "v-orange", "Moderate", 10
            else:
                verdict, action, v_col, risk, pct = "Strong Buy", "BULLISH BIAS", "v-green", "Low", 25

            # Display
            st.subheader(f"📊 {name} ({ticker})")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Conviction", f"{total_score}/100")
            m2.metric("Risk", risk)
            m3.metric("Price", f"{sym}{cur_p:,.2f}")
            m4.metric("180d AI ROI", f"{ai_roi_180:.1f}%")
            m5.metric("30d Target", f"{sym}{target_p_30:,.2f}")

            st.markdown(f'<div class="verdict-box {v_col}">Strategic Verdict: {verdict} | {action}</div>', unsafe_allow_html=True)
            
            # Charting
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['ds'], df['y'] * fx, label="Price", color='black', linewidth=1)
            ax.plot(df['ds'], df['MA50'] * fx, label="50-Day MA", color='orange', alpha=0.6)
            ax.plot(df['ds'], df['MA200'] * fx, label="200-Day MA", color='red', alpha=0.6)
            ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=180), datetime.datetime.now() + datetime.timedelta(days=180)])
            plt.legend()
            st.pyplot(fig)

        else: st.error("Data Unavailable for this ticker.")
