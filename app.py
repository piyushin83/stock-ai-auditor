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
    [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] p { font-size: 14px !important; color: #555; }
    .stop-loss-box { background-color: #fff1f1; padding: 15px; border-radius: 8px; border-left: 6px solid #ff4b4b; margin-bottom: 20px; }
    .stop-loss-text { font-size: 18px; font-weight: bold; color: #d32f2f; margin: 0; }
    .phase-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-top: 4px solid #1f77b4; }
    .disclaimer-box { background-color: #fef9e7; padding: 15px; border-radius: 5px; border: 1px solid #f1c40f; font-size: 12px; color: #7f8c8d; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# 2. HEADER & DISCLAIMER
st.title("🏛️ Strategic AI Investment Architect")
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ LEGAL DISCLAIMER:</b> This terminal is an AI analytical tool for educational purposes only. 
    It is not financial advice. AI projections do not guarantee future results. 
    <b>Human judgment is mandatory before investing.</b>
</div>
""", unsafe_allow_html=True)

# 3. ENGINES (Search, Currency, Audit)
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return data['Close'].iloc[-1]
    except: return 1.07 if to_curr == "USD" else 0.93

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
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Company Name or Ticker", value="Microsoft")
display_currency = st.sidebar.selectbox("Display Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input(f"Total Capital ({display_currency})", value=1000)

# 5. EXECUTION
if st.sidebar.button("🚀 Execute Full Audit"):
    with st.spinner(f"📡 Resolving Global Markets for '{user_query}'..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_audit_data(ticker, suffix)
        
        if df is not None:
            fx_rate = get_exchange_rate(native_curr, display_currency)
            symbol = "$" if display_currency == "USD" else "€"
            
            # AI (180 DAYS)
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            raw_p = df['y'].iloc[-1]
            conv_p = raw_p * fx_rate
            roi_180 = ((forecast['yhat'].iloc[-1] - raw_p) / raw_p) * 100
            
            # CONVICTION & STRATEGY
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi_180 > 10 else 0) + (1 if health['Debt'] < 1.1 else 0)
            if points >= 2: mode, label, pct, sl_buf = "Aggressive", "🌟 STRONG BUY", 0.15, 0.90
            elif points == 1: mode, label, pct, sl_buf = "Defensive", "🟡 ACCUMULATE", 0.05, 0.85
            else: mode, label, pct, sl_buf = "Preservation", "🛑 AVOID", 0.0, 0.0

            imm_buy = total_capital * pct
            staging_amt = total_capital - imm_buy

            # --- OUTPUT: HEADER ---
            st.subheader(f"📊 Audit Result: {name} ({ticker}{suffix})")
            st.success(f"### Strategy Mode: {mode} - {label}")
            
            # STOP LOSS
            sl_price = conv_p * sl_buf
            sl_text = f"STOP LOSS: Exit immediately if price drops below {symbol}{sl_price:.2f} ({(1-sl_buf)*100:.0f}% Buffer)" if pct > 0 else "STOP LOSS: No entry recommended."
            st.markdown(f'<div class="stop-loss-box"><p class="stop-loss-text">{sl_text}</p></div>', unsafe_allow_html=True)

            # METRICS
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Action 1: Buy Today", f"{symbol}{imm_buy:.2f}")
            c2.metric("Conviction Score", f"{int((points/3)*100)}%")
            c3.metric("180-Day AI ROI", f"{roi_180:+.1f}%")
            c4.metric(f"Current Price ({display_currency})", f"{symbol}{conv_p:.2f}")

            # --- OUTPUT: PHASES ---
            st.markdown("---")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f'<div class="phase-card"><h4>🚀 PHASE 1: IMMEDIATE</h4><p><b>Action:</b> {"BUY - Invest " + symbol + f"{imm_buy:,.2f}" + " today." if imm_buy > 0 else "HOLD - Do not deploy capital."}</p></div>', unsafe_allow_html=True)
            with p2:
                st.markdown(f'<div class="phase-card"><h4>⏳ PHASE 2: STAGING</h4><p><b>Action:</b> {"Deploy remaining " + symbol + f"{staging_amt:,.2f}" + " over 180 days." if staging_amt > 0 else "Keep cash in reserves."}</p></div>', unsafe_allow_html=True)

            # --- OUTPUT: HEALTH TABLE ---
            st.markdown("---")
            st.subheader("🏥 Company Health Detail")
            st.table(pd.DataFrame({
                "Metric": ["ROE (Efficiency)", "Debt/Equity", "Current Ratio", "Profit Margin"],
                "Status": [f"{health['ROE']*100:.1f}%", health['Debt'], health['Ratio'], health['Margin']],
                "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Low", "✅ Safe" if health['Debt'] < 1.1 else "⚠️ High", "✅ Liquid", "✅ Stable"]
            }))

            # --- OUTPUT: 180-DAY GRAPH ---
            st.markdown("---")
            st.subheader(f"🤖 180-Day AI Price Projection ({display_currency})")
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] *= fx_rate
            fig = m.plot(forecast)
            plt.title(f"{name} 180-Day Forecast")
            st.pyplot(fig)
            
            st.warning("⚠️ Forecasts are probabilities. Human judgment required.")
        else:
            st.error(f"❌ Could not retrieve data for {user_query}. Check the name and try again.")
