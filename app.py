import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

# 1. UI SETUP & PROFESSIONAL STYLING
st.set_page_config(page_title="Global AI Investment Terminal", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; color: #1f77b4; }
    .stop-loss-box { background-color: #fff1f1; padding: 15px; border-radius: 8px; border-left: 6px solid #ff4b4b; margin: 10px 0; }
    .stop-loss-text { font-size: 18px; font-weight: bold; color: #d32f2f; margin: 0; }
    .disclaimer-box { background-color: #fef9e7; padding: 15px; border-radius: 5px; border: 1px solid #f1c40f; font-size: 12px; color: #7f8c8d; }
</style>
""", unsafe_allow_html=True)

# 2. LEGAL DISCLAIMER & HEADER
st.title("🏛️ Strategic AI Investment Architect")
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ LEGAL DISCLAIMER:</b> This terminal is an AI-driven analytical tool intended for educational purposes only. 
    It does not constitute financial, investment, or legal advice. Historical performance and AI projections (Prophet) 
    do not guarantee future results. All investments carry risk, and you may lose some or all of your principal capital. 
    <b>Human judgment is mandatory before any trade execution.</b>
</div>
""", unsafe_allow_html=True)

# 3. CURRENCY & TICKER ENGINE
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return data['Close'].iloc[-1]
    except:
        return 1.10 if to_curr == "USD" else 0.91

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

# 4. SIDEBAR CONFIG
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Enter Company (e.g. Microsoft, BMW)", value="Microsoft")
display_currency = st.sidebar.selectbox("Preferred Output Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input(f"Total Capital ({display_currency})", value=1000)

# 5. DATA AUDIT ENGINE
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

# 6. EXECUTION & VISUALIZATION
if st.sidebar.button("🚀 Execute Audit"):
    with st.spinner(f"📡 Resolving Global Markets for '{user_query}'..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_audit_data(ticker, suffix)
        
        if df is not None:
            # Currency Logic
            fx_rate = get_exchange_rate(native_curr, display_currency)
            symbol = "$" if display_currency == "USD" else "€"
            
            # AI MODEL (Prophet)
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            raw_p = df['y'].iloc[-1]
            conv_p = raw_p * fx_rate
            roi_180 = ((forecast['yhat'].iloc[-1] - raw_p) / raw_p) * 100
            
            # SCORING & STRATEGY
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi_180 > 10 else 0) + (1 if health['Debt'] < 1.1 else 0)
            mode, label, pct, sl_buf = ("Aggressive", "🌟 STRONG BUY", 0.15, 0.90) if points >= 2 else \
                                       ("Defensive", "🟡 ACCUMULATE", 0.05, 0.85) if points == 1 else \
                                       ("Preservation", "🛑 AVOID", 0.0, 0.0)

            # RESULTS UI
            st.subheader(f"📊 Audit Result: {name} ({ticker}{suffix})")
            st.markdown(f"**Native Currency:** {native_curr} | **Converted Price ({display_currency}):** {symbol}{conv_p:.2f}")
            st.success(f"### Strategy: {mode} - {label}")
            
            if pct > 0:
                st.markdown(f'<div class="stop-loss-box"><p class="stop-loss-text">STOP LOSS: Exit below {symbol}{conv_p * sl_buf:.2f}</p></div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Action: Buy", f"{symbol}{total_capital*pct:.2f}")
            c2.metric("Conviction Score", f"{int((points/3)*100)}%")
            c3.metric("180D AI ROI", f"{roi_180:+.1f}%")
            c4.metric(f"Price ({display_currency})", f"{symbol}{conv_p:.2f}")

            # AI CHART
            st.markdown("---")
            st.subheader(f"🤖 180-Day AI Price Projection ({display_currency})")
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] *= fx_rate
            fig = m.plot(forecast)
            plt.title(f"{name} 180D Trend Forecast")
            st.pyplot(fig)
            
            st.info("💡 **Note:** Graph prices have been adjusted for exchange rate parity.")
        else:
            st.error(f"❌ Connection Issue: Found '{ticker}' but data is unavailable. Please try a different stock.")

# 7. HELP TAB
with st.expander("📚 How to read this dashboard"):
    st.write("""
    1. **Strategy Mode:** Determined by Profit Probability, Return on Equity, and Debt levels.
    2. **Stop Loss:** The price at which the trade should be closed to protect your capital.
    3. **Action: Buy:** The dollar amount of your total capital to invest immediately today.
    4. **AI Projection:** The blue line represents the AI's 'best guess' for the next 6 months.
    """)
