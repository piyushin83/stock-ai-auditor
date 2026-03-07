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

# 1. UI SETUP & THEME-AWARE CSS
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

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

# 2. DISCLAIMER
st.markdown('<div class="disclaimer-container">🚨 <b>LEGAL:</b> Educational Tool Only. Fibonacci targets are contingency buy orders. AI projections are mathematical and adjusted for market volatility.</div>', unsafe_allow_html=True)
st.title("🏛️ Strategic AI Investment Architect (V8.9)")

# 3. HELPER ENGINES
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 1.0
    except: return 1.0

def resolve_smart_ticker(user_input):
    ticker_str = user_input.strip().upper()
    try:
        t_obj = yf.Ticker(ticker_str)
        if t_obj.fast_info.get('lastPrice') is not None:
            name = t_obj.info.get('longName', ticker_str)
            curr = t_obj.fast_info.get('currency', 'USD')
            return ticker_str, name, ".US", curr
        s = yf.Search(ticker_str, max_results=1)
        if s.tickers:
            res = s.tickers[0]
            ticker = res['symbol']; name = res.get('longname', ticker)
            t_obj_fb = yf.Ticker(ticker)
            native_curr = t_obj_fb.fast_info.get('currency', 'USD')
            return ticker, name, "", native_curr
    except: pass
    return ticker_str, ticker_str, ".US", "USD"

def get_news_sentiment(ticker):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(req.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return 0, []
        impact_keywords = ['earnings', 'dividend', 'fed', 'revenue', 'lawsuit', 'sec', 'merger', 'acquisition', 'growth', 'crash']
        parsed_news, sentiment_score = [], 0
        rows = news_table.find_all('tr')
        for row in rows:
            text = row.a.text
            if any(k in text.lower() for k in impact_keywords) or len(parsed_news) < 2:
                score = TextBlob(text).sentiment.polarity
                sentiment_score += score
                parsed_news.append(f"{'🟢' if score > 0 else '🔴' if score < 0 else '⚪'} {text}")
            if len(parsed_news) >= 5: break
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
    diff = max(curr_p - recent_low, curr_p * 0.10)
    fib_levels = {'0.382': curr_p - (diff * 0.382), '0.500': curr_p - (diff * 0.500), '0.618': curr_p - (diff * 0.618)}
    return df['rsi'].iloc[-1], fib_levels

def get_fundamental_health(ticker, suffix):
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period="5y")
        if df.empty:
            end = datetime.datetime.now(); start = end - datetime.timedelta(days=1825)
            df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'vol'}).sort_values('ds')
        df['ds'] = df['ds'].dt.tz_localize(None)
        
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
user_query = st.sidebar.text_input("Ticker / Symbol", value="NVDA")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

# 5. MAIN EXECUTION
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner("Analyzing Logic..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)
        
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            
            df['50_Day_Moving_Average'] = df['y'].rolling(window=50).mean()
            df['200_Day_Moving_Average'] = df['y'].rolling(window=200).mean()
            
            m = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.015, changepoint_range=0.90).fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180) 
            forecast = m.predict(future)
            
            # --- Deviation & Fair Value Range ---
            trend_val = forecast[forecast['ds'] == df['ds'].iloc[-1]]['yhat'].values[0] * fx
            deviation = ((cur_p - trend_val) / trend_val) * 100
            fair_low = trend_val * 0.95
            fair_high = trend_val * 1.05
            
            is_death_cross = df['50_Day_Moving_Average'].iloc[-1] < df['200_Day_Moving_Average'].iloc[-1]
            crossover_msg = "Price stability detected."
            cross_point = None
            for i in range(len(df)-60, len(df)):
                prev = i-1
                if df['50_Day_Moving_Average'].iloc[prev] < df['200_Day_Moving_Average'].iloc[prev] and df['50_Day_Moving_Average'].iloc[i] > df['200_Day_Moving_Average'].iloc[i]:
                    cross_point = (df['ds'].iloc[i], df['50_Day_Moving_Average'].iloc[i], "GOLDEN")
                    crossover_msg = "🚀 GOLDEN CROSS: 50-Day Moving Average crossed ABOVE 200-Day."
                elif df['50_Day_Moving_Average'].iloc[prev] > df['200_Day_Moving_Average'].iloc[prev] and df['50_Day_Moving_Average'].iloc[i] < df['200_Day_Moving_Average'].iloc[i]:
                    cross_point = (df['ds'].iloc[i], df['50_Day_Moving_Average'].iloc[i], "DEATH")
                    crossover_msg = "⚠️ DEATH CROSS: 50-Day Moving Average crossed BELOW 200-Day."

            target_p_30 = forecast['yhat'].iloc[len(df) + 29] * fx
            if is_death_cross: target_p_30 *= 0.96 
            ai_roi_30 = ((target_p_30 - cur_p) / cur_p) * 100
            
            rsi, fibs = calculate_technicals(df)
            news_score, headlines = get_news_sentiment(ticker)
            
            score = 15 
            if not is_death_cross: score += 20 
            if health['ROE'] > 0.12: score += 20 
            if ai_roi_30 > 0.5: score += 30 
            if news_score > 0: score += 15
            score = max(0, min(100, score))
            
            if score >= 70: verdict, v_col, action, pct = "Strong Buy", "v-green", "ACTION: BUY NOW", 25
            elif score >= 35: verdict, v_col, action, pct = "Hold / Neutral", "v-orange", "ACTION: MONITOR", 10
            else: verdict, v_col, action, pct = "Sell / Avoid", "v-red", "ACTION: SELL / STAY AWAY", 0

            sl_price = cur_p * (0.95 if rsi > 70 else 0.85 if rsi < 30 else 0.90)

            # --- DISPLAY ---
            st.subheader(f"📊 {name} Analysis ({ticker})")
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if deviation > 15:
                    st.warning(f"⚠️ MEAN REVERSION RISK: Price is {deviation:.1f}% above AI baseline trend.")
                else:
                    st.success(f"✅ TREND ALIGNMENT: Price is within {deviation:.1f}% of AI baseline.")
            with col_f2:
                st.info(f"💎 AI FAIR VALUE RANGE: {sym}{fair_low:,.2f} - {sym}{fair_high:,.2f}")

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
                st.markdown("### 🏥 Company Health")
                st.table(pd.DataFrame({
                    "Metric": ["ROE", "P/B Ratio", "Debt/Equity", "Current Ratio"],
                    "Status": [f"{health['ROE']*100:.1f}%", f"{health['PB']}x", health['Debt'], health['CurrentRatio']],
                    "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Weak", "✅ Healthy" if health['PB'] < 3.0 else "⚠️ Expensive", "✅ Safe", "✅ Liquid"]
                }))
                st.markdown("### 📰 Latest News Impact")
                for h in headlines: st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown("### ⚖️ Strategy & Fibonacci Limits")
                st.markdown(f"""<div class="phase-card">
                    <h4 style="color:#1f77b4">PHASE 1: IMMEDIATE</h4>
                    <p><b>Invest Today:</b> {sym}{total_capital*(pct/100):,.2f} ({pct}% allocation)</p>
                    <hr>
                    <h4 style="color:#1f77b4">PHASE 2: STAGED ENTRY (FIBONACCI)</h4>
                    <div class="fib-box">🔹 Target 1 (0.382): {sym}{fibs['0.382']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Target 2 (0.500): {sym}{fibs['0.500']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Target 3 (0.618): {sym}{fibs['0.618']*fx:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🤖 AI Stock 180-Day Projection (Full Moving Average Labels)")
            fig, ax = plt.subplots(figsize=(12, 6))
            forecast_plot = forecast.copy()
            forecast_plot[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
            m.plot(forecast_plot, ax=ax)
            ax.plot(df['ds'], df['50_Day_Moving_Average'] * fx, label='50-Day Moving Average', color='orange', linewidth=2)
            ax.plot(df['ds'], df['200_Day_Moving_Average'] * fx, label='200-Day Moving Average', color='red', linewidth=2)
            if cross_point:
                ax.scatter(cross_point[0], cross_point[1] * fx, color='gold', s=300, marker='*', label=f"{cross_point[2]} CROSS POINT", zorder=5)
            ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=180), datetime.datetime.now() + datetime.timedelta(days=180)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.legend(loc='upper left'); st.pyplot(fig)

            st.markdown("---")
            st.subheader("📊 12-Month Relative Volume Trend")
            vol_fig, vol_ax = plt.subplots(figsize=(12, 4))
            vol_df = df.tail(252).copy() 
            colors = ['#2e7d32' if i > 0 and vol_df.iloc[i]['y'] >= vol_df.iloc[i-1]['y'] else '#c62828' for i in range(len(vol_df))]
            vol_ax.bar(vol_df['ds'], vol_df['vol'], color=colors, alpha=0.7)
            vol_ax.set_ylabel("Shares Traded")
            vol_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            st.pyplot(vol_fig)
            
        else: st.error("Data Unreachable. Check Symbol.")
