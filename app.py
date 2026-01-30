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

# 1. UI SETUP & CSS
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    .phase-card { background-color: #f4f6f9; color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .news-card { background-color: #ffffff; color: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; }
    
    @media (prefers-color-scheme: dark) {
        .phase-card { background-color: #1e2129; color: #ffffff; border: 1px solid #3d414b; }
        .news-card { background-color: #262730; color: #ffffff; border-left: 5px solid #00b0ff; }
    }

    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
    .impact-announcement { background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; border-left: 5px solid #ffeeba; margin-bottom: 15px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 3. HELPER ENGINES
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
            suffix_map = {'LSE': '.UK', 'GER': '.DE', 'FRA': '.DE', 'PAR': '.FR', 'AMS': '.NL', 'TSE': '.JP'}
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
        parsed_news, sentiment_score = [], 0
        rows = news_table.find_all('tr')
        for row in rows[:5]:
            text = row.a.text
            score = TextBlob(text).sentiment.polarity
            sentiment_score += score
            parsed_news.append(f"{'🟢' if score > 0 else '🔴' if score < 0 else '⚪'} {text}")
        return (sentiment_score / 5), parsed_news
    except: return 0, ["⚠️ News Unavailable"]

def get_fundamental_health(ticker, suffix):
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1825)
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        
        health = {"ROE": 0, "Debt": 0}
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
            def fvz(label):
                td = soup.find('td', string=label)
                return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "-"
            health = {"ROE": float(fvz("ROE"))/100 if fvz("ROE")!="-" else 0, "Debt": float(fvz("Debt/Eq")) if fvz("Debt/Eq")!="-" else 0}
        except: pass
        return df, health
    except: return None, None

# 4. SIDEBAR
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker", value="SAP")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

# 5. MAIN EXECUTION
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner("Executing Strategic AI Audit..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)
        
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            
            # --- FULL FORM MA CALCULATION ---
            df['MA50_Days'] = df['y'].rolling(window=50).mean()
            df['MA200_Days'] = df['y'].rolling(window=200).mean()
            
            # --- AI MODEL (SYNC FIX) ---
            # Increased changepoint_prior_scale to 0.05 so blue area covers black dots better
            m = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            target_180 = forecast['yhat'].iloc[-1] * fx
            ai_roi = ((target_180 - cur_p) / cur_p) * 100
            
            # --- CROSSOVER IMPACT ENGINE ---
            crossover_msg = "No major crossover recently."
            cross_date, cross_val, cross_type = None, None, None
            
            # Check last 30 days for a crossover
            for i in range(len(df)-30, len(df)):
                prev = i - 1
                if df['MA50_Days'].iloc[prev] < df['MA200_Days'].iloc[prev] and df['MA50_Days'].iloc[i] > df['MA200_Days'].iloc[i]:
                    cross_date, cross_val, cross_type = df['ds'].iloc[i], df['MA50_Days'].iloc[i], "GOLDEN"
                    crossover_msg = "🚀 GOLDEN CROSS DETECTED: 50-Day crossed ABOVE 200-Day. This is a MAJOR BULLISH signal indicating a long-term uptrend."
                elif df['MA50_Days'].iloc[prev] > df['MA200_Days'].iloc[prev] and df['MA50_Days'].iloc[i] < df['MA200_Days'].iloc[i]:
                    cross_date, cross_val, cross_type = df['ds'].iloc[i], df['MA50_Days'].iloc[i], "DEATH"
                    crossover_msg = "⚠️ DEATH CROSS DETECTED: 50-Day crossed BELOW 200-Day. This is a MAJOR BEARISH signal indicating a potential market crash."

            # Verdict Logic
            score = 0
            if cur_p > (df['MA200_Days'].iloc[-1] * fx): score += 30
            if ai_roi > 5: score += 40
            elif ai_roi < -2: score -= 40
            
            v_label, v_col = ("STRONG BUY", "v-green") if score >= 70 else ("AVOID", "v-red") if score < 40 else ("HOLD", "v-orange")

            # --- DISPLAY ---
            st.subheader(f"📊 {name} ({ticker})")
            st.markdown(f'<div class="impact-announcement">{crossover_msg}</div>', unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Price", f"{sym}{cur_p:,.2f}")
            m2.metric("AI Forecast (180d)", f"{ai_roi:.1f}%")
            m3.metric("Conviction Score", f"{score}/100")
            
            st.markdown(f'<div class="verdict-box {v_col}">Verdict: {v_label}</div>', unsafe_allow_html=True)

            # --- ENHANCED GRAPH ---
            st.subheader("🤖 AI Prediction & Moving Average Analytics")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            forecast_plot = forecast.copy()
            forecast_plot[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
            m.plot(forecast_plot, ax=ax)
            
            ax.plot(df['ds'], df['MA50_Days'] * fx, label='50-Day Moving Average', color='orange', alpha=0.8)
            ax.plot(df['ds'], df['MA200_Days'] * fx, label='200-Day Moving Average', color='red', alpha=0.8)
            
            # Mark the Impact Point if crossover exists
            if cross_date:
                ax.scatter(cross_date, cross_val * fx, color='gold', s=200, marker='*', zorder=5, label=f"{cross_type} CROSS")
                ax.annotate("IMPACT POINT", (cross_date, cross_val * fx), xytext=(20, 20), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))

            ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=180), datetime.datetime.now() + datetime.timedelta(days=180)])
            plt.legend(loc='upper left')
            st.pyplot(fig)

        else: st.error("Data Unavailable.")
