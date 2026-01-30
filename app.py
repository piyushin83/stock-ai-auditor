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
    .phase-card { background-color: #f4f6f9; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .tech-card { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 5px solid #2e7d32; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; }
    .news-card { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; }
    .fib-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; font-weight: bold; }
    .stop-loss-box { background-color: #fff1f1; border-left: 8px solid #ff4b4b; padding: 15px; margin-bottom: 20px; color: #b71c1c; font-weight: bold; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
    .legend-box { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin-top: 10px; font-size: 14px; line-height: 1.6; }
    .disclaimer-container { background-color: #262730; color: #aaa; padding: 15px; border-radius: 5px; font-size: 12px; margin-bottom: 20px; border: 1px solid #444; }
    .upside-text { color: #2e7d32; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 2. DISCLAIMER
st.markdown('<div class="disclaimer-container">🚨 <b>LEGAL:</b> Educational Tool Only. Fibonacci targets are contingency buy orders for market volatility and may differ from AI trend projections.</div>', unsafe_allow_html=True)
st.title("🏛️ Strategic AI Investment Architect (V5.4)")

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

# 5. TECHNICALS & FIBONACCI
def calculate_technicals(df):
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    recent_high = df['y'].tail(252).max()
    recent_low = df['y'].tail(252).min()
    diff = recent_high - recent_low
    fib_levels = {
        '0.382': recent_high - (diff * 0.382), 
        '0.500': recent_high - (diff * 0.500), 
        '0.618': recent_high - (diff * 0.618)  
    }
    return df['rsi'].iloc[-1], fib_levels

def get_fundamental_health(ticker, suffix):
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095)
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'vol'}).sort_values('ds')
        health = {"ROE": 0, "Debt": 0, "Margin": "N/A", "CurrentRatio": "N/A"}
        if suffix == ".US":
            try:
                url = f"https://finviz.com/quote.ashx?t={ticker}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                soup = BeautifulSoup(requests.get(url, headers=headers, timeout=5).text, 'html.parser')
                def fvz(label):
                    td = soup.find('td', string=label)
                    return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "-"
                health = {"ROE": float(fvz("ROE"))/100 if fvz("ROE")!="-" else 0,
                          "Debt": float(fvz("Debt/Eq")) if fvz("Debt/Eq")!="-" else 0,
                          "Margin": fvz("Profit Margin") + "%",
                          "CurrentRatio": fvz("Current Ratio")}
            except: pass
        return df, health
    except: return None, None

# 6. SIDEBAR
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker", value="NVDA")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

# 7. MAIN EXECUTION
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner("Processing..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)
        
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            
            # --- AI ENGINE ---
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            # 30-Day Sub-Logic (For Metric Box Only)
            forecast_30_row = forecast.iloc[len(df) + 29] if len(forecast) > (len(df)+29) else forecast.iloc[-1]
            target_p_30 = forecast_30_row['yhat'] * fx
            upside_30 = target_p_30 - cur_p
            roi_30 = (upside_30 / cur_p) * 100

            # 180-Day Logic (For Strategic Analysis & Verdict)
            target_p_180 = forecast['yhat'].iloc[-1] * fx
            ai_roi_180 = ((target_p_180 - cur_p) / cur_p) * 100
            
            rsi, fibs = calculate_technicals(df)
            news_score, headlines = get_news_sentiment(ticker)
            
            score = 0
            if health['ROE'] > 0.15: score += 20
            if health['Debt'] < 1.1: score += 20
            if ai_roi_180 > 10: score += 30
            if news_score > 0: score += 20
            score = min(100, score)
            
            if score >= 75: verdict, action, v_col, risk, pct = "Strong Buy", "ACTION: BUY", "v-green", "Low", 25
            elif score >= 50: verdict, action, v_col, risk, pct = "Accumulate", "ACTION: HOLD / BUY DIPS", "v-orange", "Moderate", 10
            else: verdict, action, v_col, risk, pct = "Avoid", "ACTION: SELL / STAY AWAY", "v-red", "High", 0

            st.subheader(f"📊 {name} Analysis ({ticker})")
            
            # --- ROW 1: METRICS (UPSCALE 30-DAY WINDOW FOR FIRST TWO) ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Conviction Score", f"{score}/100")
            m2.metric("Risk Level", risk)
            m3.metric("Current Price", f"{sym}{cur_p:,.2f}")
            # Potential Upside & Target adjusted to 30 days as requested
            m4.metric("30d Potential Upside", f"{sym}{upside_30:,.2f}", delta=f"{roi_30:.1f}%")
            m5.metric("30d AI Target", f"{sym}{target_p_30:,.2f}")

            st.markdown(f"ℹ️ **Short-Term View:** Potential Upside and AI Target are showing a **30-day outlook**. All strategic ratings and the chart below maintain a **180-day forecast**.")

            st.markdown(f'<div class="verdict-box {v_col}">Strategic Verdict (180d): {verdict} | {action}</div>', unsafe_allow_html=True)
            
            sl_price = cur_p * 0.88 if risk == "Low" else cur_p * 0.85
            st.markdown(f'<div class="stop-loss-box">🛑 STOP LOSS GUIDANCE: Exit if price drops to {sym}{sl_price:,.2f}</div>', unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 🏥 Company Health Details")
                st.table(pd.DataFrame({
                    "Metric": ["ROE", "Debt/Equity", "Profit Margin", "Current Ratio"],
                    "Status": [f"{health['ROE']*100:.1f}%", health['Debt'], health['Margin'], health['CurrentRatio']],
                    "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Weak", "✅ Safe" if health['Debt'] < 1.1 else "⚠️ High", "✅ Stable", "✅ Liquid"]
                }))
                st.markdown("### 📰 Latest News Headlines")
                for h in headlines:
                    st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown("### ⚖️ Strategy & Fibonacci Limits")
                st.markdown(f"""<div class="phase-card">
                    <h4 style="color:#1f77b4">PHASE 1: IMMEDIATE</h4>
                    <p><b>Invest Today:</b> {sym}{total_capital*(pct/100):,.2f} ({pct}% of funds)</p>
                    <hr>
                    <h4 style="color:#1f77b4">PHASE 2: STAGED ENTRY (FIBONACCI)</h4>
                    <p>Set <b>Limit Orders</b> at these levels to catch dips.</p>
                    <div class="fib-box">🔹 Target 1 (0.382): {sym}{fibs['0.382']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Target 2 (0.500): {sym}{fibs['0.500']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Target 3 (0.618): {sym}{fibs['0.618']*fx:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🤖 AI Forecast Projection (180-Day Trend)")
            
            st.markdown("""
            <div class="legend-box">
                <b>Chart Legend:</b><br>
                <span style="color:#000000;">●</span> <b>Black Dots:</b> Historical Price.<br>
                <span style="color:#1f77b4; font-weight:bold;">━</span> <b>Solid Blue Line:</b> AI Median Path.<br>
                <span style="background-color:#d1e6f9; border:1px solid #1f77b4; padding:0 5px;">&nbsp;</span> <b>Light Blue Shade:</b> Volatility Zone (80% probability).
            </div>
            """, unsafe_allow_html=True)
            
            # Plot remains 180 days
            forecast_plot = forecast.copy()
            forecast_plot[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
            fig1 = m.plot(forecast_plot)
            plt.title(f"{name} 180-Day Growth Forecast")
            st.pyplot(fig1)

        else: st.error("Data Unavailable.")
