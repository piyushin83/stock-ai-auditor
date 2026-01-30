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
    .phase-card { background-color: #f4f6f9; color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .news-card { background-color: #ffffff; color: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; }
    @media (prefers-color-scheme: dark) {
        .phase-card { background-color: #1e2129; color: #ffffff; border: 1px solid #3d414b; }
        .news-card { background-color: #262730; color: #ffffff; border-left: 5px solid #00b0ff; }
    }
    .impact-announcement { background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; border-left: 8px solid #ffc107; margin-bottom: 20px; font-weight: bold; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 24px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
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

def get_fundamental_health(ticker, suffix):
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1825)
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        
        # Fundamental Data (ROE + P/B)
        health = {"ROE": 0, "PB": 0}
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
            def fvz(label):
                td = soup.find('td', string=label)
                return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "-"
            health = {
                "ROE": float(fvz("ROE"))/100 if fvz("ROE")!="-" else 0,
                "PB": float(fvz("P/B")) if fvz("P/B")!="-" else 0
            }
        except: pass
        return df, health
    except: return None, None

def get_news_sentiment(ticker):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0, []
        parsed_news, score_total = [], 0
        rows = news_table.find_all('tr')
        for row in rows[:5]:
            text = row.a.text
            polarity = TextBlob(text).sentiment.polarity
            score_total += polarity
            parsed_news.append(f"{'🟢' if polarity > 0 else '🔴' if polarity < 0 else '⚪'} {text}")
        return (score_total / 5), parsed_news
    except: return 0, ["⚠️ News Feed Unavailable"]

# 4. SIDEBAR
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker", value="SAP")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

# 5. MAIN EXECUTION
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner("Analyzing Market and AI Vectors..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)
        
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            
            # MA Analytics
            df['MA50_Days'] = df['y'].rolling(window=50).mean()
            df['MA200_Days'] = df['y'].rolling(window=200).mean()
            
            # --- AI LOGIC (Conservative Sync) ---
            m = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.02).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            target_30 = forecast['yhat'].iloc[-1] * fx
            ai_roi_30 = ((target_30 - cur_p) / cur_p) * 100
            
            # Crossover Check
            crossover_msg = "Market stability detected. No major technical breaches."
            cross_point = None
            for i in range(len(df)-60, len(df)):
                prev = i-1
                if df['MA50_Days'].iloc[prev] < df['MA200_Days'].iloc[prev] and df['MA50_Days'].iloc[i] > df['MA200_Days'].iloc[i]:
                    cross_point = (df['ds'].iloc[i], df['MA50_Days'].iloc[i], "GOLDEN")
                    crossover_msg = "🚀 GOLDEN CROSS: 50-Day Moving Average crossed ABOVE 200-Day. Bullish momentum incoming."
                elif df['MA50_Days'].iloc[prev] > df['MA200_Days'].iloc[prev] and df['MA50_Days'].iloc[i] < df['MA200_Days'].iloc[i]:
                    cross_point = (df['ds'].iloc[i], df['MA50_Days'].iloc[i], "DEATH")
                    crossover_msg = "⚠️ DEATH CROSS: 50-Day Moving Average crossed BELOW 200-Day. Bearish breakdown risk."

            news_val, headlines = get_news_sentiment(ticker)
            
            # --- SCORING MATRIX ---
            score = 0
            if cur_p > (df['MA200_Days'].iloc[-1] * fx): score += 30
            if health['ROE'] > 0.15: score += 15
            if health['PB'] < 3.0: score += 15
            if ai_roi_30 > 2: score += 40
            elif ai_roi_30 < -1: score -= 50
            score = max(0, min(100, score))
            
            # VERDICT LABELS
            if score >= 75: verdict, v_col, action = "STRONG BUY", "v-green", "BULLISH TREND"
            elif score >= 45: verdict, v_col, action = "HOLD", "v-orange", "NEUTRAL / SIDEWAYS"
            else: verdict, v_col, action = "SELL / AVOID", "v-red", "BEARISH RISK"

            # DISPLAY
            st.subheader(f"📊 {name} Analysis ({ticker})")
            st.markdown(f'<div class="impact-announcement">{crossover_msg}</div>', unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"{sym}{cur_p:,.2f}")
            m2.metric("30d AI Growth", f"{ai_roi_30:.1f}%")
            m3.metric("30d Target", f"{sym}{target_30:,.2f}")
            m4.metric("Strategic Score", f"{score}/100")

            st.markdown(f'<div class="verdict-box {v_col}">Verdict: {verdict} | {action}</div>', unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 🏥 Company Health")
                st.table(pd.DataFrame({
                    "Metric": ["ROE (Return on Equity)", "P/B (Price-to-Book)"],
                    "Status": [f"{health['ROE']*100:.1f}%", f"{health['PB']} x"],
                    "Health": ["✅ Strong" if health['ROE']>0.15 else "⚠️ Low", "✅ Good Value" if health['PB']<3.0 else "⚠️ Expensive"]
                }))
                st.markdown("### 📰 Sentiment Feed")
                for h in headlines: st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown("### 🤖 Technical Analysis Graph")
                fig, ax = plt.subplots(figsize=(10, 5))
                forecast_plot = forecast.copy()
                forecast_plot[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
                m.plot(forecast_plot, ax=ax)
                ax.plot(df['ds'], df['MA50_Days']*fx, label='50-Day Moving Average', color='orange', alpha=0.7)
                ax.plot(df['ds'], df['MA200_Days']*fx, label='200-Day Moving Average', color='red', alpha=0.7)
                if cross_point:
                    ax.scatter(cross_point[0], cross_point[1]*fx, color='gold', s=200, marker='*', zorder=5)
                ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=120), datetime.datetime.now() + datetime.timedelta(days=35)])
                plt.legend(loc='upper left')
                st.pyplot(fig)
        else: st.error("Data Unavailable.")
