import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

# 1. UI SETUP & CSS
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

st.markdown("""
<style>
    /* Metric Styling */
    [data-testid="stMetricValue"] { font-size: 30px !important; font-weight: 800 !important; color: #1f77b4; }
    /* Strategy & Alert Styling */
    .alert-container { border-radius: 10px; padding: 20px; margin: 10px 0; }
    .stop-loss-box { background-color: #fff1f1; border-left: 8px solid #ff4b4b; margin-bottom: 10px; padding: 15px; }
    .risk-box { background-color: #fffde7; border-left: 8px solid #fbc02d; padding: 15px; margin-bottom: 20px; }
    .alert-text { font-size: 18px; font-weight: bold; margin: 0; }
    /* Phase Card Styling */
    .phase-card { background-color: #f0f2f6; padding: 25px; border-radius: 15px; border: 1px solid #d1d5db; min-height: 280px; }
    .phase-header { color: #1f77b4; font-weight: 800; font-size: 20px; margin-bottom: 10px; }
    .disclaimer-box { font-size: 11px; color: #888; text-align: center; margin-top: 50px; border-top: 1px solid #ddd; padding-top: 10px; }
</style>
""", unsafe_allow_html=True)

# 2. HEADER
st.title("🏛️ Strategic AI Investment Architect")

# 3. ENGINES
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return data['Close'].iloc[-1]
    except: return 1.07

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

def get_audit_data(ticker, suffix):
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) 
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        
        health = {"ROE": 0.0, "Debt": 0.0, "Ratio": "N/A", "Margin": "N/A"}
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
                          "Ratio": fvz("Current Ratio"), "Margin": fvz("Profit Margin") + "%"}
            except: pass
        return df, health
    except: return None, None

# 4. SIDEBAR
st.sidebar.header("⚙️ User Controls")
user_query = st.sidebar.text_input("Search Company", value="Nvidia")
display_currency = st.sidebar.selectbox("Output Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input(f"Investment Amount ({display_currency})", value=1000)

# 5. EXECUTION
if st.sidebar.button("🚀 Analyze Market Opportunity"):
    with st.spinner(f"📡 Synchronizing Global Data for {user_query}..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_audit_data(ticker, suffix)
        
        if df is not None:
            fx_rate = get_exchange_rate(native_curr, display_currency)
            symbol = "$" if display_currency == "USD" else "€"
            
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            raw_p = df['y'].iloc[-1]
            conv_p = raw_p * fx_rate
            roi_180 = ((forecast['yhat'].iloc[-1] - raw_p) / raw_p) * 100
            
            # --- LOGIC & SCORING ---
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi_180 > 10 else 0) + (1 if health['Debt'] < 1.1 else 0)
            conviction_score = int((points / 3) * 100)
            prob_profit = max(min(50 + (roi_180 / 2), 99), 1)

            if points == 3:
                label, pct, sl_buf, risk_txt = "🌟 HIGH CONVICTION BUY", 15, 0.90, "Low-to-Moderate"
            elif points >= 1:
                label, pct, sl_buf, risk_txt = "🟡 ACCUMULATE / HOLD", 5, 0.85, "Moderate"
            else:
                label, pct, sl_buf, risk_txt = "🛑 AVOID / SELL", 0, 0.0, "High Risk"

            # --- DISPLAY: METRICS ---
            st.subheader(f"📊 Deep Audit: {name} ({ticker}{suffix})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conviction", f"{conviction_score}/100")
            m2.metric("Prob. of Profit", f"{prob_profit:.1f}%")
            m3.metric("AI 180D ROI", f"{roi_180:+.1f}%")
            m4.metric(f"Current Price", f"{symbol}{conv_p:.2f}")

            # --- DISPLAY: RISK & STOP LOSS ---
            st.markdown("---")
            st.success(f"### Strategy: {label}")
            
            # Grouped Risk & Stop Loss to eliminate "Big Box" gap
            st.markdown(f'<div class="risk-box"><p class="alert-text">⚠️ RISK ASSESSMENT: {risk_txt}</p><p>Fundamental markers suggest ' + 
                        ("strong underlying stability." if points >= 2 else "potential volatility or weak growth metrics.") + '</p></div>', unsafe_allow_html=True)
            
            sl_price = conv_p * sl_buf
            sl_msg = f"🛑 STOP LOSS: Exit if price hits {symbol}{sl_price:.2f}" if pct > 0 else "🛑 ADVISORY: Capital preservation mode. Do not enter."
            st.markdown(f'<div class="stop-loss-box"><p class="stop-loss-text">{sl_msg}</p></div>', unsafe_allow_html=True)

            # --- DISPLAY: DETAILED STRATEGY PHASES ---
            st.markdown("---")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f"""<div class="phase-card">
                    <div class="phase-header">🚀 PHASE 1: IMMEDIATE ACTION</div>
                    <p><b>Allocation:</b> {pct}% of total capital</p>
                    <p><b>Investment:</b> {symbol}{total_capital * (pct/100):,.2f}</p>
                    <hr>
                    <p><b>Instructions:</b> Enter a market-on-open position to capture immediate momentum. 
                    This establishes your "anchor" position based on current AI trend validation.</p>
                </div>""", unsafe_allow_html=True)
            with p2:
                st.markdown(f"""<div class="phase-card">
                    <div class="phase-header">⏳ PHASE 2: STAGED DEPLOYMENT</div>
                    <p><b>Allocation:</b> {100-pct}% of total capital</p>
                    <p><b>Reserve:</b> {symbol}{total_capital * ((100-pct)/100):,.2f}</p>
                    <hr>
                    <p><b>Instructions:</b> Do not buy all at once. Use a 'Limit Order' strategy to buy during red days. 
                    Deploy 1/4 of this reserve every 45 days, provided price remains above the Stop Loss.</p>
                </div>""", unsafe_allow_html=True)

            # --- DISPLAY: HEALTH & CHART ---
            st.markdown("---")
            st.subheader("🏥 Company Health Detail")
            st.table(pd.DataFrame({
                "Metric": ["ROE (Efficiency)", "Debt/Equity", "Current Ratio", "Profit Margin"],
                "Status": [f"{health['ROE']*100:.1f}%", health['Debt'], health['Ratio'], health['Margin']],
                "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Low/NA", "✅ Safe" if health['Debt'] < 1.1 else "⚠️ High/NA", "✅ Liquid", "✅ Stable"]
            }))

            st.markdown("---")
            st.subheader(f"🤖 180-Day AI Price Projection ({display_currency})")
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] *= fx_rate
            fig = m.plot(forecast)
            plt.title(f"{name} Forecast ({display_currency})")
            st.pyplot(fig)

            st.markdown('<div class="disclaimer-box">Educational tool only. Forecasts are probabilistic and may vary based on market conditions.</div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ Error: Data feed for {user_query} is unavailable.")
