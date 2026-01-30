import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob 
import datetime
import yfinance as yf
import plotly.graph_objects as go
import time

# --- 1. UI & STYLING ---
st.set_page_config(page_title="Strategic AI Architect V7.0", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 800 !important; color: #002b36; }
    .verdict-card { padding: 30px; border-radius: 15px; text-align: center; color: white; font-weight: 900; font-size: 30px; margin: 20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.2); transition: 0.3s; }
    .v-red { background: linear-gradient(145deg, #e53935, #b71c1c); border-bottom: 8px solid #7f0000; }
    .v-orange { background: linear-gradient(145deg, #fb8c00, #ef6c00); border-bottom: 8px solid #b53d00; }
    .v-green { background: linear-gradient(145deg, #43a047, #1b5e20); border-bottom: 8px solid #003300; }
    .news-card { background: white; padding: 15px; border-radius: 10px; border-left: 6px solid #2196f3; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .tech-pill { background: #e3f2fd; padding: 10px; border-radius: 8px; font-family: monospace; font-size: 14px; border: 1px solid #bbdefb; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA ENGINES (ANTI-RATE LIMIT) ---

def get_safe_session():
    """Creates a session with headers to prevent Yahoo Finance blocking."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    })
    return session

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_str):
    """Fetches stock data with a retry mechanism for Streamlit Cloud stability."""
    session = get_safe_session()
    ticker = yf.Ticker(ticker_str, session=session)
    
    for _ in range(3): # Retry 3 times
        try:
            df = ticker.history(period="5y")
            if not df.empty:
                return df, ticker.info
            time.sleep(1.5)
        except:
            time.sleep(2)
    return None, None

def get_sentiment_score(ticker_str):
    """Scrapes news and calculates sentiment polarity."""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker_str}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0.0, ["⚠️ Sentiment Engine Unavailable"]
        
        headlines, scores = [], []
        for row in news_table.findAll('tr')[:6]:
            text = row.a.text
            score = TextBlob(text).sentiment.polarity
            scores.append(score)
            icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
            headlines.append(f"{icon} {text}")
        return np.mean(scores), headlines
    except:
        return 0.0, ["⚠️ Feed restricted by provider."]

# --- 3. CORE LOGIC ENGINE ---

def process_analysis(df, info, ticker, fx, sym):
    # Data Cleaning for Prophet
    df_p = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    df_p['ds'] = df_p['ds'].dt.tz_localize(None)
    
    # Technical Indicators
    df_p['MA50'] = df_p['y'].rolling(50).mean()
    df_p['MA200'] = df_p['y'].rolling(200).mean()
    
    current_price = df_p['y'].iloc[-1] * fx
    ma200_price = df_p['MA200'].iloc[-1] * fx
    
    # AI Forecasting
    with st.spinner("Training AI Prediction Model..."):
        m = Prophet(yearly_seasonality=True, daily_seasonality=False)
        m.fit(df_p[['ds', 'y']])
        future = m.make_future_dataframe(periods=180)
        forecast = m.predict(future)
        
        target_180 = forecast['yhat'].iloc[-1] * fx
        target_roi = ((target_180 - current_price) / current_price) * 100

    # Fundamentals & Sentiment
    roe = info.get('returnOnEquity', 0)
    sentiment, news = get_sentiment_score(ticker)

    # --- UNBIASED VERDICT LOGIC ---
    # We use a 100-point system to avoid name-based bias
    score = 0
    
    # Technical Weight (35 pts): Is the long-term trend healthy?
    if current_price > ma200_price: score += 35
    elif current_price > (ma200_price * 0.95): score += 15 # Near support
    else: score -= 25 # Major Trend Break (AVOID Signal)

    # Forecast Weight (40 pts): What does the AI see?
    if target_roi > 15: score += 40
    elif target_roi > 5: score += 20
    elif target_roi < 0: score -= 35 # Price Cratering Signal

    # Health Weight (25 pts): ROE and Fundamentals
    if roe > 0.15: score += 25
    elif roe > 0.08: score += 10

    # FINAL VERDICT ASSEMBLY
    if score >= 75 and target_roi > 2:
        v_label, v_color, v_desc = "STRONG BUY", "v-green", "BULLISH MOMENTUM"
    elif score >= 40 and target_roi > -5:
        v_label, v_color, v_desc = "HOLD / ACCUMULATE", "v-orange", "NEUTRAL / RECOVERY"
    else:
        v_label, v_color, v_desc = "AVOID", "v-red", "HIGH RISK / BEARISH"

    return {
        "label": v_label, "color": v_color, "desc": v_desc,
        "price": current_price, "roi": target_roi, "target": target_180,
        "roe": roe, "sent": sentiment, "news": news,
        "raw": df_p, "fcst": forecast, "ma200": ma200_price
    }

# --- 4. DASHBOARD INTERFACE ---

st.title("🏛️ Strategic AI Investment Architect")
st.markdown(f"**Current Date:** {datetime.date.today().strftime('%B %d, %Y')}")

with st.sidebar:
    st.header("⚙️ Control Panel")
    ticker_input = st.text_input("Ticker Symbol", value="SAP").upper()
    currency_choice = st.selectbox("Base Currency", ["USD", "EUR", "GBP"])
    st.markdown("---")
    # THE SUBMIT BUTTON
    run_audit = st.button("🚀 EXECUTE DEEP AUDIT", use_container_width=True)

if run_audit:
    raw_df, stock_info = fetch_stock_data(ticker_input)
    
    if raw_df is not None:
        native_curr = stock_info.get('currency', 'USD')
        fx_val = 1.0 if native_curr == currency_choice else 0.92 # Simple fall-back FX
        sym = {"USD":"$", "EUR":"€", "GBP":"£"}.get(currency_choice, "$")
        
        # Run the Engine
        res = process_analysis(raw_df, stock_info, ticker_input, fx_val, sym)
        
        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"{sym}{res['price']:,.2f}")
        m2.metric("180d AI Forecast", f"{sym}{res['target']:,.2f}", f"{res['roi']:.1f}%")
        m3.metric("ROE", f"{res['roe']*100:.1f}%")
        m4.metric("Market Sentiment", f"{res['sent']:.2f}")

        # Big Verdict Card
        st.markdown(f'<div class="verdict-card {res["color"]}">{res["label"]} <br><small>{res["desc"]}</small></div>', unsafe_allow_html=True)

        # Plotly Charts
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['raw']['ds'], y=res['raw']['y']*fx_val, name="Price", line=dict(color='#002b36', width=2)))
        fig.add_trace(go.Scatter(x=res['raw']['ds'], y=res['raw']['MA200']*fx_val, name="200-Day MA", line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=res['fcst']['ds'].tail(180), y=res['fcst']['yhat'].tail(180)*fx_val, name="AI Path", line=dict(color='#2196f3', width=4)))
        fig.update_layout(title=f"{ticker_input} Advanced Trend Projection", height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Intelligence Section
        col_news, col_tech = st.columns([2, 1])
        with col_news:
            st.subheader("📰 Market Intelligence")
            for h in res['news']:
                st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)
        with col_tech:
            st.subheader("📐 Technical Levels")
            st.markdown(f'<div class="tech-pill"><b>200D MA Support:</b> {sym}{res["ma200"]:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="tech-pill"><b>Price Status:</b> {"OVER" if res["price"] > res["ma200"] else "UNDER"} MA200</div>', unsafe_allow_html=True)

    else:
        st.error(f"⚠️ Error: Yahoo Finance rate-limited the request. Please wait 30 seconds and click 'Execute Deep Audit' again.")

st.markdown("---")
st.caption("Strategic Audit V7.0 | Logic Correction: Price < MA200 & Negative AI ROI = Automatic AVOID.")
