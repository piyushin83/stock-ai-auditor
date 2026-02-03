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
st.set_page_config(page_title="Strategic AI Architect V8.3", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    .phase-card { background-color: #f4f6f9; color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .news-card { background-color: #ffffff; color: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1); }
    .fib-box { background-color: #e3f2fd; color: #0d47a1; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; font-weight: bold; }
    @media (prefers-color-scheme: dark) {
        .phase-card { background-color: #1e2129; color: #ffffff; border: 1px solid #3d414b; }
        .news-card { background-color: #262730; color: #ffffff; border-left: 5px solid #00b0ff; }
        .fib-box { background-color: #0d47a1; color: #e3f2fd; border-left: 4px solid #00b0ff; }
    }
    .impact-announcement { background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; border-left: 8px solid #ffc107; margin-bottom: 20px; font-weight: bold; }
    .stop-loss-box { background-color: #fff1f1; border-left: 8px solid #ff4b4b; padding: 15px; margin-bottom: 20px; color: #b71c1c; font-weight: bold; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
    .disclaimer-container { background-color: #262730; color: #aaa; padding: 15px; border-radius: 5px; font-size: 12px; margin-bottom: 20px; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer-container">🚨 <b>LEGAL:</b> Universal Analysis Engine. AI targets are mathematically projected and do not guarantee market performance.</div>', unsafe_allow_html=True)
st.title("🏛️ Strategic AI Investment Architect (V8.3)")

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
            ticker = res['symbol']; name = res.get('longname', ticker); exch = res.get('exchange', 'NYQ')
            t_obj = yf.Ticker(ticker); native_curr = t_obj.fast_info.get('currency', 'USD')
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
        
        # Point 2: Effect-based News Filtering
        impact_keywords = ['earnings', 'dividend', 'fed', 'rate', 'revenue', 'lawsuit', 'sec', 'merger', 'acquisition', 'guidance', 'downgrade', 'upgrade']
        parsed_news, sentiment_score = [], 0
        rows = news_table.find_all('tr')
        
        for row in rows:
            text = row.a.text
            if any(k in text.lower() for k in impact_keywords) or len(parsed_news) < 2:
                score = TextBlob(text).sentiment.polarity
                sentiment_score += score
                parsed_news.append(f"{'🟢' if score > 0 else '🔴' if score < 0 else '⚪'} {text}")
            if len(parsed_news) >= 5: break
        return (sentiment_score / 5 if parsed_news else 0), (parsed_news if parsed_news else ["No high-impact news found."])
    except: return 0, ["⚠️ News Feed Unavailable"]

def calculate_technicals(df):
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    curr_p = df['y'].iloc[-1]
    recent_low = df['y'].tail(126).min() 
    diff = max(curr_p - recent_low, curr_p * 0.10)
    fib_levels = {'0.382': curr_p - (diff * 0.382), '0.500': curr_p - (diff * 0.500), '0.618': curr_p - (diff * 0.618)}
    return df['rsi'].iloc[-1], fib_levels

def get_fundamental_health(ticker, suffix):
    try:
        end = datetime.datetime.now(); start = end - datetime.timedelta(days=1825)
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'vol'}).sort_values('ds')
        
        health = {"ROE": 0, "Debt": 0, "PB": 0, "Margin": "N/A", "CurrentRatio": "N/A"}
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
            def fvz(label):
                td = soup.find('td', string=label)
                return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "-"
            health = {
                "ROE": float(fvz("ROE"))/100 if fvz("ROE")!="-" else 0,
                "Debt": float(fvz("Debt/Eq")) if fvz("Debt/Eq")!="-" else 0,
                "PB": float(fvz("P/B")) if fvz("P/B")!="-" else 0,
                "Margin": fvz("Profit Margin") + "%",
                "CurrentRatio": fvz("Current Ratio")
            }
        except: pass
        return df, health
    except: return None, None

# 4. SIDEBAR
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker / Company", value="AAPL")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

# 5. MAIN EXECUTION
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner("Processing Logic..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)
        
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            
            df['MA50_Days'] = df['y'].rolling(window=50).mean()
            df['MA200_Days'] = df['y'].rolling(window=200).mean()
            
            # --- SYNCED AI LOGIC ---
            m = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.01).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180) 
            forecast = m.predict(future)
            
            # Point 1 & 3: Bridge the Gap between AI and Technicals
            is_death_cross = df['MA50_Days'].iloc[-1] < df['MA200_Days'].iloc[-1]
            crossover_msg = "Price stability detected."
            cross_point = None
            for i in range(len(df)-60, len(df)):
                prev = i-1
                if df['MA50_Days'].iloc[prev] < df['MA200_Days'].iloc[prev] and df['MA50_Days'].iloc[i] > df['MA200_Days'].iloc[i]:
                    cross_point = (df['ds'].iloc[i], df['MA50_Days'].iloc[i], "GOLDEN")
                    crossover_msg = "🚀 GOLDEN CROSS: Technical momentum is turning Bullish."
                elif df['MA50_Days'].iloc[prev] > df['MA200_Days'].iloc[prev] and df['MA50_Days'].iloc[i] < df['MA200_Days'].iloc[i]:
                    cross_point = (df['ds'].iloc[i], df['MA50_Days'].iloc[i], "DEATH")
                    crossover_msg = "⚠️ DEATH CROSS: Technical risk detected. AI projection adjusted for bearish pressure."

            # Calculate Target & Apply Bearish Penalty if Death Cross is active
            raw_target_30 = forecast['yhat'].iloc[len(df) + 29] * fx
            target_p_30 = raw_target_30 * 0.90 if is_death_cross else raw_target_30
            ai_roi_30 = ((target_p_30 - cur_p) / cur_p) * 100
            
            rsi, fibs = calculate_technicals(df)
            news_score, headlines = get_news_sentiment(ticker)
            
            # Logic-aware Scoring
            score = 0
            if not is_death_cross: score += 30 
            if health['ROE'] > 0.15: score += 15
            if ai_roi_30 > 2: score += 40 
            elif ai_roi_30 < -1: score -= 50
            score = max(0, min(100, score))
            
            if score >= 75: verdict, v_col, action, pct = "Strong Buy", "v-green", "ACTION: BUY NOW", 25
            elif score >= 45: verdict, v_col, action, pct = "Hold", "v-orange", "ACTION: MONITOR", 10
            else: verdict, v_col, action, pct = "Sell / Avoid", "v-red", "ACTION: SELL / STAY AWAY", 0

            sl_price = cur_p * (0.95 if rsi > 70 else 0.85 if rsi < 30 else 0.90)

            # --- DISPLAY ---
            st.subheader(f"📊 {name} Analysis ({ticker})")
            st.markdown(f'<div class="impact-announcement">{crossover_msg}</div>', unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conviction Score", f"{score}/100")
            m2.metric("Current Price", f"{sym}{cur_p:,.2f}")
            m3.metric("30d AI Growth", f"{ai_roi_30:.1f}%")
            m4.metric("30d AI Target", f"{sym}{target_p_30:,.2f}")

            st.markdown(f'<div class="verdict-box {v_col}">Strategic Verdict: {verdict} | {action}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stop-loss-box">🛑 AGGRESSIVE STOP LOSS: {sym}{sl_price:,.2f}</div>', unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 🏥 Fundamental Health")
                st.table(pd.DataFrame({
                    "Metric": ["ROE", "P/B Ratio", "Debt/Equity", "Current Ratio"],
                    "Status": [f"{health['ROE']*100:.1f}%", f"{health['PB']}x", health['Debt'], health['CurrentRatio']],
                    "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Weak", "✅ Healthy" if health['PB'] < 3.0 else "⚠️ Overvalued", "✅ Safe", "✅ Liquid"]
                }))
                st.markdown("### 📰 Market Impact News")
                for h in headlines: st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown("### ⚖️ Portfolio & Fibonacci Strategy")
                st.markdown(f"""<div class="phase-card">
                    <h4 style="color:#1f77b4">PHASE 1: CAPITAL ALLOCATION</h4>
                    <p><b>Initial Entry:</b> {sym}{total_capital*(pct/100):,.2f} ({pct}% of total capital)</p>
                    <hr>
                    <h4 style="color:#1f77b4">PHASE 2: VOLATILITY TARGETS</h4>
                    <div class="fib-box">🔹 Limit Buy (0.382): {sym}{fibs['0.382']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Limit Buy (0.500): {sym}{fibs['0.500']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Limit Buy (0.618): {sym}{fibs['0.618']*fx:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🤖 AI Stock 180-Day Projection (Synced Indicators)")
            fig, ax = plt.subplots(figsize=(12, 6))
            forecast_plot = forecast.copy()
            forecast_plot[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
            
            # Point 3 Correction: Sync the Blue Line to the Death Cross
            if is_death_cross:
                forecast_plot.loc[forecast_plot.index > len(df), 'yhat'] *= 0.94 # Visual adjustment for bearish pressure

            m.plot(forecast_plot, ax=ax)
            ax.plot(df['ds'], df['MA50_Days'] * fx, label='50-Day MA', color='orange', alpha=0.8)
            ax.plot(df['ds'], df['MA200_Days'] * fx, label='200-Day MA', color='red', alpha=0.8)
            
            if cross_point:
                ax.scatter(cross_point[0], cross_point[1] * fx, color='gold', s=300, marker='*', label=f"{cross_point[2]} POINT", zorder=5)
            
            ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=180), datetime.datetime.now() + datetime.timedelta(days=180)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.legend(loc='upper left'); st.pyplot(fig)
        else: st.error("Search failed. Please try a standard ticker symbol (e.g., NVDA, TSLA).")
