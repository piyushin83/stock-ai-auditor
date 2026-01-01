import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime

# 1. UI SETUP
st.set_page_config(page_title="Master AI Terminal V3", layout="wide")
st.title("🏛️ Master AI Investment Terminal")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border: 1px solid #d1d5db;">
    🟢 <b>Source:</b> Stooq (Prices) + Finviz (Fundamentals) | <b>Status:</b> Unlimited Access
</div>
""", unsafe_allow_html=True)

# 2. SIDEBAR
st.sidebar.header("⚙️ Parameters")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)

# 3. DATA ENGINE (NO KEY REQUIRED)
def get_data_unlimited(ticker):
    try:
        # A. Fetch Prices from Stooq (No Limit)
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) # 3 years
        df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        df = df.sort_values('ds')
        
        # B. Fetch Fundamentals from Finviz (Scrape)
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        f_res = requests.get(url, headers=headers)
        soup = BeautifulSoup(f_res.text, 'html.parser')
        
        # Helper to find data in Finviz table
        def get_finviz_val(label):
            td = soup.find('td', string=label)
            if td:
                val = td.find_next_sibling('td').text
                return val.strip('%')
            return "0"

        roe = float(get_finviz_val("ROE")) / 100
        debt = float(get_finviz_val("Debt/Eq"))
        
        return df, roe, debt
    except Exception as e:
        st.error(f"Error: {e}")
        return None, 0, 0

# 4. EXECUTION
if st.sidebar.button("🚀 Run Full Audit"):
    with st.spinner(f"📊 Analyzing {stock_symbol}..."):
        df, roe, de = get_data_unlimited(stock_symbol)
        
        if df is not None and not df.empty:
            # AI Prediction
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            cur_p = df['y'].iloc[-1]
            target_roi = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # POINT SYSTEM
            f_score = 1 if (roe > 0.15 and de < 1.5) else 0
            ai_score = 1 if target_roi > 10 else 0
            points = f_score + ai_score # Points out of 2 for this version
            
            # ALLOCATION
            if points == 2:
                label, imm_pct = "🌟 HIGH CONVICTION BUY", 0.15
                strat = "Aggressive: Phase in over 2 months. If price drops 5%, double monthly buy."
            elif points == 1:
                label, imm_pct = "🟡 ACCUMULATE / HOLD", 0.05
                strat = "Defensive: Phase in over 4 months."
            else:
                label, imm_pct = "🛑 AVOID", 0.0
                strat = "Capital Preservation: Wait for better entry."

            imm_buy = total_capital * imm_pct

            # DASHBOARD
            if points == 2: st.success(f"### {label}")
            elif points == 1: st.warning(f"### {label}")
            else: st.error(f"### {label}")
            
            st.info(f"ROE: {roe*100:.1f}% | Debt/Eq: {de:.2f} | 90d Forecast: {target_roi:+.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${cur_p:.2f}")
            c2.metric("Immediate Buy", f"${imm_buy:.2f}")
            c3.metric("Shares", f"{imm_buy/cur_p:.2f}")

            st.markdown("---")
            st.write(f"**Action Plan:** {strat}")
            st.pyplot(m.plot(forecast))
            
            st.caption("⚠️ IT IS AI AND HUMAN SHOULD USE THEIR BRAIN BEFORE INVESTING.")
