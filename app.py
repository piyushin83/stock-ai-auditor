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
    [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 800 !important; color: #1f77b4; }
    .phase-card { background-color: #f4f6f9; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .news-card { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; }
    .fib-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; font-weight: bold; }
    .stop-loss-box { background-color: #fff1f1; border-left: 8px solid #ff4b4b; padding: 15px; margin-bottom: 20px; color: #b71c1c; font-weight: bold; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
    .legend-box { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin-top: 10px; font-size: 14px; line-height: 1.6; }
    .disclaimer-container { background-color: #262730; color: #aaa; padding: 15px; border-radius: 5px; font-size: 12px; margin-bottom: 20px; border: 1px solid #444; }
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
            
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=30) # Changed to 30 days for requested view
            forecast = m.predict(future)
            
            target_p = forecast['yhat'].iloc[-1] * fx
            potential_upside = target_p - cur_p
            ai_roi = (potential_upside / cur_p) * 100
            
            rsi, fibs = calculate_technicals(df)
            news_score, headlines = get_news_sentiment(ticker)
            
            score = 0
            if health['ROE'] > 0.15: score += 20
            if health['Debt'] < 1.1: score += 20
            if ai_roi > 5: score += 30 # Adjusted for 30-day window
            if news_score > 0: score += 20
            score = min(100, score)
            
            if score >= 75: verdict, action, v_col, risk, pct = "Strong Buy", "ACTION: BUY", "v-green", "Low", 25
            elif score >= 50: verdict, action, v_col, risk, pct = "Accumulate", "ACTION: HOLD / BUY DIPS", "v-orange", "Moderate", 10
            else: verdict, action, v_col, risk, pct = "Avoid", "ACTION: SELL / STAY AWAY", "v-red", "High", 0

            st.subheader(f"📊 {name} Analysis ({ticker})")
            
            # --- ALIGNED METRICS ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Conviction", f"{score}/100")
            m2.metric("Risk Level", risk)
            m3.metric("Live Price", f"{sym}{cur_p:,.2f}")
            m4.metric("Potential Upside", f"{sym}{potential_upside:,.2f}", delta=f"{ai_roi:.2f}%")
            m5.metric("AI 30d Target", f"{sym}{target_p:,.2f}")

            st.info(f"**Potential Upside:** The predicted growth in value per share over the next 30 days based on AI median trend projections.")

            st.markdown(f'<div class="verdict-box {v_col}">Verdict: {verdict} | {action}</div>', unsafe_allow_html=True)
            
            sl_price = cur_p * 0.88 if risk == "Low" else cur_p * 0.85
            st.markdown(f'<div class="stop-loss-box">🛑 STOP LOSS: Exit if price drops below {sym}{sl_price:,.2f}</div>', unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 🏥 Company Health")
                st.table(pd.DataFrame({
                    "Metric": ["ROE", "Debt/Eq", "Margin", "Current Ratio"],
                    "Status": [f"{health['ROE']*100:.1f}%", health['Debt'], health['Margin'], health['CurrentRatio']],
                    "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Weak", "✅ Safe" if health['Debt'] < 1.1 else "⚠️ High", "✅ Stable", "✅ Liquid"]
                }))
                st.markdown("### 📰 Sentiment Headlines")
                for h in headlines:
                    st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown("### ⚖️ Fibonacci Entry Zones")
                st.markdown(f"""<div class="phase-card">
                    <h4 style="color:#1f77b4">PHASE 1: CAPITAL ALLOCATION</h4>
                    <p><b>Deploy Today:</b> {sym}{total_capital*(pct/100):,.2f} ({pct}% of total)</p>
                    <hr>
                    <h4 style="color:#1f77b4">PHASE 2: VOLATILITY TARGETS</h4>
                    <div class="fib-box">🔹 Support 1 (0.382): {sym}{fibs['0.382']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Support 2 (0.500): {sym}{fibs['0.500']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Support 3 (0.618): {sym}{fibs['0.618']*fx:,.2f}</div>
                    <br><small>Limit orders at Support 2/3 are recommended for long-term accumulation.</small>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🤖 AI 30-Day Forecast Projection")
            st.markdown("""
            <div class="legend-box">
                <b>Chart Legend:</b><br>
                <span style="color:#000000;">●</span> <b>Actuals:</b> Historical price closings.<br>
                <span style="color:#1f77b4; font-weight:bold;">━</span> <b>Trend:</b> AI Median Projection.<br>
                <span style="background-color:#d1e6f9; border:1px solid #1f77b4; padding:0 5px;">&nbsp;</span> <b>Zone:</b> 80% Volatility Range.
            </div>
            """, unsafe_allow_html=True)
            
            # Zoom to 30 days view
            forecast_zoom = forecast.tail(60) # Show last 30 days + 30 forecast days
            forecast_zoom[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
            fig1 = m.plot(forecast_zoom)
            plt.title(f"{name}: 30-Day Short Term Outlook")
            st.pyplot(fig1)

        else: st.error("Data Unavailable.")
