import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob 
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

# 1. UI SETUP & CSS
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    .phase-card { background-color: #f4f6f9; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 320px; }
    .tech-card { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 5px solid #2e7d32; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; }
    .news-card { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; }
    .fib-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 20px; text-align: center; color: white; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
    .disclaimer-container { background-color: #262730; color: #aaa; padding: 15px; border-radius: 5px; font-size: 12px; margin-bottom: 20px; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

# 2. DISCLAIMER
st.markdown('<div class="disclaimer-container">🚨 <b>LEGAL:</b> Educational Tool Only. Signals based on probabilistic AI & Technical models. Not financial advice.</div>', unsafe_allow_html=True)
st.title("🏛️ Strategic AI Investment Architect (Pro + News)")

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
            suffix_map = {'LSE': '.UK', 'GER': '.DE', 'FRA': '.DE', 'PAR': '.FR', 'AMS': '.NL', 'TSE': '.JP', 'HKG': '.HK'}
            suffix = suffix_map.get(exch, ".US")
            return ticker.split('.')[0], name, suffix, native_curr
    except: pass
    return user_input.upper(), user_input.upper(), ".US", "USD"

# 4. NEWS ENGINE
def get_news_sentiment(ticker):
    # Scrapes Finviz News Headlines
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(req.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if not news_table: return 0, []

        parsed_news = []
        sentiment_score = 0
        
        # Get last 5 headlines
        rows = news_table.findAll('tr')
        for index, row in enumerate(rows):
            if index > 4: break # Limit to 5 latest
            text = row.a.text
            timestamp = row.td.text.split()
            date = timestamp[0] if len(timestamp) == 1 else "Today"
            
            # TextBlob Sentiment Analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity # -1 to 1
            sentiment_score += polarity
            parsed_news.append(f"{'🟢' if polarity > 0 else '🔴' if polarity < 0 else '⚪'} {text}")
            
        avg_sentiment = sentiment_score / 5
        return avg_sentiment, parsed_news
    except:
        return 0, ["⚠️ News Feed Unavailable"]

# 5. TECHNICAL ANALYSIS ENGINE
def calculate_technicals(df):
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    k = df['y'].ewm(span=12, adjust=False, min_periods=12).mean()
    d = df['y'].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = k - d
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    
    recent_high = df['y'].tail(180).max()
    recent_low = df['y'].tail(180).min()
    diff = recent_high - recent_low
    fib_levels = {
        '0.5': recent_high - (diff * 0.5),
        '0.618': recent_high - (diff * 0.618)
    }
    return df['rsi'].iloc[-1], macd.iloc[-1], signal.iloc[-1], fib_levels

def get_options_sentiment(ticker):
    try:
        tk = yf.Ticker(ticker)
        dates = tk.options
        if not dates: return "N/A", 0.5
        opt = tk.option_chain(dates[0])
        puts = opt.puts['volume'].sum()
        calls = opt.calls['volume'].sum()
        if calls == 0: return "Bearish", 0.0
        pc_ratio = puts / calls
        sent = "Bullish" if pc_ratio < 0.7 else "Bearish" if pc_ratio > 1.0 else "Neutral"
        return sent, pc_ratio
    except: return "N/A", 0.5

def get_fundamental_health(ticker, suffix):
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095)
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'vol'}).sort_values('ds')
        
        health = {"ROE": 0, "Debt": 0}
        if suffix == ".US":
            try:
                url = f"https://finviz.com/quote.ashx?t={ticker}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                soup = BeautifulSoup(requests.get(url, headers=headers, timeout=5).text, 'html.parser')
                def fvz(label):
                    td = soup.find('td', string=label)
                    return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "-"
                health = {"ROE": float(fvz("ROE"))/100 if fvz("ROE")!="-" else 0,
                          "Debt": float(fvz("Debt/Eq")) if fvz("Debt/Eq")!="-" else 0}
            except: pass
        return df, health
    except: return None, None

# 6. SIDEBAR
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker / Company", value="Nvidia")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

# 7. MAIN EXECUTION
if st.sidebar.button("🚀 Run Deep Technical & News Audit"):
    with st.spinner("Analyzing AI Trend, Technicals, Options & LIVE News..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)
        
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            
            # 1. AI
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            ai_roi = ((forecast['yhat'].iloc[-1] - df['y'].iloc[-1]) / df['y'].iloc[-1]) * 100
            
            # 2. Techs
            rsi, macd, signal, fibs = calculate_technicals(df)
            
            # 3. Sentiment (News + Options)
            news_score, headlines = get_news_sentiment(ticker)
            opt_sent, pc_ratio = get_options_sentiment(ticker)
            
            # 4. SCORING (Max 100)
            score = 0
            if health['ROE'] > 0.15: score += 10
            if health['Debt'] < 1.0: score += 10
            if ai_roi > 15: score += 20
            elif ai_roi > 5: score += 10
            if rsi < 70 and rsi > 30: score += 10 
            if macd > signal: score += 10 
            if pc_ratio < 0.8: score += 10 
            
            # News Impact (Can add or subtract up to 20 points)
            if news_score > 0.1: score += 20 # Positive News
            elif news_score < -0.1: score -= 20 # Negative News penalizes the score
            else: score += 10 # Neutral News is better than no news
            
            score = max(0, min(100, score)) # Clamp 0-100

            # VERDICT
            if score >= 75: verdict, v_col, pct = "STRONG BUY (News Validated)", "v-green", 25
            elif score >= 50: verdict, v_col, pct = "ACCUMULATE (Monitor News)", "v-orange", 10
            else: verdict, v_col, pct = "AVOID / HIGH RISK", "v-red", 0

            # --- DISPLAY ---
            st.subheader(f"📊 {name} ({ticker})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Conviction", f"{score}/100")
            c2.metric("AI 180d ROI", f"{ai_roi:+.1f}%")
            c3.metric("News Sentiment", "Positive" if news_score > 0 else "Negative")
            c4.metric("Current Price", f"{sym}{cur_p:,.2f}")

            st.markdown(f'<div class="verdict-box {v_col}">{verdict}</div>', unsafe_allow_html=True)
            
            col_tech, col_phase = st.columns([1, 1])
            with col_tech:
                st.markdown("### 📰 Market Sentiment & Headlines")
                st.write(f"**News Analysis:** {'Bullish' if news_score > 0 else 'Bearish'}. Latest headlines analyzed:")
                for h in headlines:
                    st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)
                
                st.markdown("### 🛠️ Technical Signals")
                st.markdown(f"""<div class="tech-card">
                    <b>RSI (14):</b> {rsi:.1f}<br>
                    <b>MACD:</b> {"🟢 Bullish" if macd > signal else "🔴 Bearish"}<br>
                    <b>Options P/C Ratio:</b> {pc_ratio:.2f} ({opt_sent})
                </div>""", unsafe_allow_html=True)

            with col_phase:
                st.markdown("### ⚖️ Strategy")
                buy_amt = total_capital * (pct/100)
                st.markdown(f"""<div class="phase-card">
                    <h4>PHASE 1: IMMEDIATE</h4>
                    <p><b>Allocation:</b> <span style="font-size:20px; color:#2e7d32">{pct}%</span></p>
                    <p><b>Value:</b> {sym}{buy_amt:,.2f}</p>
                    <hr>
                    <h4>PHASE 2: FIBONACCI LIMITS</h4>
                    <p>Set buy orders at these technical support levels:</p>
                    <div class="fib-box">🎯 0.50 Level: {sym}{fibs['0.5']*fx:,.2f}</div>
                    <div class="fib-box">🏆 0.618 Level: {sym}{fibs['0.618']*fx:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🤖 AI Forecast")
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
            fig1 = m.plot(forecast)
            plt.title(f"{name} AI Forecast")
            st.pyplot(fig1)

        else: st.error("Data Unavailable.")
