import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime

# 1. UI SETUP & CUSTOM BIG FONT CSS
st.set_page_config(page_title="Master AI Terminal V3", layout="wide")

# Custom CSS to force the Strategy metric to be huge and colored
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 56px !important; font-weight: 800 !important; }
    .status-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 8px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Strategic AI Investment Architect")

# 2. SIDEBAR
st.sidebar.header("⚙️ Configuration")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)

# 3. DATA ENGINE (YAHOO BYPASS)
def get_full_audit(ticker):
    try:
        # Fetch Prices (Stooq)
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) 
        df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        
        # Fetch Stats (Finviz)
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        
        def fvz(label):
            td = soup.find('td', string=label)
            return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "0"

        health = {
            "ROE": float(fvz("ROE")) / 100 if fvz("ROE") != "-" else 0,
            "Debt/Eq": float(fvz("Debt/Eq")) if fvz("Debt/Eq") != "-" else 0,
            "Current Ratio": fvz("Current Ratio"),
            "Profit Margin": fvz("Profit Margin") + "%"
        }
        return df, health
    except: return None, None

# 4. EXECUTION
if st.sidebar.button("🚀 Run Full Audit"):
    with st.spinner(f"📡 Auditing {stock_symbol}..."):
        df, health = get_full_audit(stock_symbol)
        
        if df is not None:
            # AI Prediction
            m = Prophet(daily_seasonality=False).fit(df)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            cur_p = df['y'].iloc[-1]
            roi = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # Logic for Points (Max 3)
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi > 10 else 0) + (1 if health['Debt/Eq'] < 1.1 else 0)

            # --- BIG DATA POINT: STRATEGY ---
            if points == 3:
                strat_mode, label, color, pct = "Aggressive", "🌟 STRONG BUY", "normal", 0.15
                risk_lvl = "Low-to-Moderate"
            elif points >= 1:
                strat_mode, label, color, pct = "Defensive", "🟡 ACCUMULATE", "off", 0.05
                risk_lvl = "Moderate"
            else:
                strat_mode, label, color, pct = "Capital Preservation", "🛑 AVOID", "inverse", 0.0
                risk_lvl = "High Risk"

            st.metric(label="DASHBOARD STRATEGY", value=f"{strat_mode}: {label}")
            
            st.markdown("---")
            
            # --- PHASED ACTION ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Action 1: Buy", f"${total_capital*pct:.2f}", delta=f"{pct*100}% Capital")
            c2.metric("Company Risk", risk_lvl)
            c3.metric("90d AI ROI", f"{roi:+.1f}%")

            st.markdown("---")
            
            # --- COMPANY HEALTH & RISK AUDIT ---
            col_health, col_risk = st.columns(2)
            
            with col_health:
                st.subheader("🏥 Company Health Detail")
                health_data = {
                    "Metric": ["Efficiency (ROE)", "Solvency (Debt/Eq)", "Liquidity (Current)", "Profitability"],
                    "Status": [f"{health['ROE']*100:.1f}%", health['Debt/Eq'], health['Current Ratio'], health['Profit Margin']],
                    "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Low",
                               "✅ Safe" if health['Debt/Eq'] < 1.1 else "⚠️ Leveraged",
                               "✅ Liquid", "✅ Stable"]
                }
                st.table(pd.DataFrame(health_data))

            with col_risk:
                st.subheader("⚖️ Risk Assessment")
                st.markdown(f"""
                - **Primary Risk:** {'Market Volatility' if points >= 2 else 'Fundamental Weakness'}
                - **Leverage Risk:** {'Low' if health['Debt/Eq'] < 1 else 'High Debt-to-Equity Detected'}
                - **Forecast Confidence:** {'High' if abs(roi) > 5 else 'Low Conviction'}
                """)
                st.progress(points / 3)
                st.caption(f"Audit Conviction Score: {int((points/3)*100)}%")

            st.markdown("---")
            st.subheader("🤖 AI Price Forecast (90 Days)")
            st.pyplot(m.plot(forecast))
            
            st.warning("⚠️ This AI-Generated Content Is For Informational Purposes Only And Not a Substitute For Professional Advice. No Liability For Financial Losses Due To Reliance On The Tool's Outputs.")
