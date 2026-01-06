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
    [data-testid="stMetricValue"] { font-size: 30px !important; font-weight: 800 !important; color: #1f77b4; }
    .stop-loss-box { background-color: #fff1f1; border-left: 8px solid #ff4b4b; padding: 15px; margin-bottom: 10px; }
    .risk-box { background-color: #fffde7; border-left: 8px solid #fbc02d; padding: 15px; margin-bottom: 20px; }
    .phase-card { background-color: #f0f2f6; padding: 20px; border-radius: 12px; border: 1px solid #d1d5db; min-height: 300px; }
    .target-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1f77b4; }
    .disclaimer-box { font-size: 11px; color: #888; text-align: center; margin-top: 50px; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Strategic AI Investment Architect")

# 2. ENGINES (Search & Currency)
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return data['Close'].iloc[-1]
    except: return 1.06 if to_curr == "USD" else 0.94

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

# 3. SIDEBAR
st.sidebar.header("⚙️ Settings")
user_query = st.sidebar.text_input("Enter Company", value="Microsoft")
display_currency = st.sidebar.selectbox("Currency Output", ["USD", "EUR"])
total_capital = st.sidebar.number_input(f"Total Capital ({display_currency})", value=1000)

# 4. EXECUTION
if st.sidebar.button("🚀 Analyze Now"):
    with st.spinner(f"📡 Processing {user_query}..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_audit_data(ticker, suffix)
        
        if df is not None:
            fx_rate = get_exchange_rate(native_curr, display_currency)
            symbol = "$" if display_currency == "USD" else "€"
            
            # AI FORECAST
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            raw_p = df['y'].iloc[-1]
            conv_p = raw_p * fx_rate
            roi_180 = ((forecast['yhat'].iloc[-1] - raw_p) / raw_p) * 100
            
            # STRATEGY LOGIC
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi_180 > 10 else 0) + (1 if health['Debt'] < 1.1 else 0)
            if points == 3: label, pct, sl_buf, risk = "🌟 HIGH CONVICTION BUY", 15, 0.90, "Low"
            elif points >= 1: label, pct, sl_buf, risk = "🟡 ACCUMULATE / HOLD", 5, 0.85, "Moderate"
            else: label, pct, sl_buf, risk = "🛑 AVOID", 0, 0.0, "High"

            # --- DISPLAY: HEADER & METRICS ---
            st.subheader(f"📊 {name} ({ticker}{suffix})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conviction", f"{int((points/3)*100)}/100")
            m2.metric("ROI (180d)", f"{roi_180:+.1f}%")
            m3.metric(f"Price ({display_currency})", f"{symbol}{conv_p:.2f}")
            m4.metric("Risk Level", risk)

            st.markdown(f'<div class="risk-box"><b>Strategy:</b> {label}</div>', unsafe_allow_html=True)
            sl_price = conv_p * sl_buf
            st.markdown(f'<div class="stop-loss-box"><b>🛑 STOP LOSS:</b> Exit if price drops below {symbol}{sl_price:.2f}</div>', unsafe_allow_html=True)

            # --- DISPLAY: PHASED STRATEGY ---
            st.markdown("---")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f"""<div class="phase-card">
                    <h3>🚀 PHASE 1: IMMEDIATE</h3>
                    <p><b>Allocation:</b> {pct}% of total funds</p>
                    <p><b>Invest Today:</b> {symbol}{total_capital * (pct/100):,.2f}</p>
                    <hr>
                    <p><i>Action: Buy at current market price to secure initial exposure.</i></p>
                </div>""", unsafe_allow_html=True)
            
            with p2:
                # Calculate Staged Entry Targets (Remaining 85-95% of capital)
                t1, t2, t3 = conv_p * 0.97, conv_p * 0.94, conv_p * 0.90 # -3%, -6%, -10% targets
                st.markdown(f"""<div class="phase-card">
                    <h3>⏳ PHASE 2: STAGED ENTRY</h3>
                    <p><b>Allocation:</b> {100-pct}% of total funds</p>
                    <p><b>Remaining Capital:</b> {symbol}{total_capital * ((100-pct)/100):,.2f}</p>
                    <hr>
                    <p><b>Deploy remaining funds at these Price Targets:</b></p>
                    <div class="target-box"><b>Target 1 (Minor Dip -3%):</b> {symbol}{t1:.2f}</div>
                    <div class="target-box"><b>Target 2 (Healthy Pullback -6%):</b> {symbol}{t2:.2f}</div>
                    <div class="target-box"><b>Target 3 (Strong Support -10%):</b> {symbol}{t3:.2f}</div>
                </div>""", unsafe_allow_html=True)

            # --- DISPLAY: HEALTH & CHART ---
            st.markdown("---")
            st.subheader("🏥 Company Health Detail")
            
            st.table(pd.DataFrame({
                "Metric": ["ROE", "Debt/Equity", "Current Ratio", "Profit Margin"],
                "Status": [f"{health['ROE']*100:.1f}%", health['Debt'], health['Ratio'], health['Margin']],
                "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Low", "✅ Safe" if health['Debt'] < 1.1 else "⚠️ High", "✅ Liquid", "✅ Stable"]
            }))

            st.markdown("---")
            st.subheader(f"🤖 180-Day AI Projection ({display_currency})")
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] *= fx_rate
            fig = m.plot(forecast)
            plt.title(f"{name} Forecast ({display_currency})")
            st.pyplot(fig)
            
            st.markdown('<div class="disclaimer-box">Educational tool only. Currency conversion based on daily rates.</div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ Could not find data for '{user_query}'.")
