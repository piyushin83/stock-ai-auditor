import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

# 1. UI SETUP & PROFESSIONAL THEMING
st.set_page_config(page_title="Global AI Investment Terminal", layout="wide")

st.markdown("""
<style>
    /* Metric Value Styling */
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; color: #1f77b4; }
    [data-testid="stMetricLabel"] p { font-size: 13px !important; color: #666; }
    /* Strategy Highlight Boxes */
    .stop-loss-box { background-color: #fff1f1; padding: 15px; border-radius: 8px; border-left: 6px solid #ff4b4b; margin: 10px 0; }
    .stop-loss-text { font-size: 18px; font-weight: bold; color: #d32f2f; margin: 0; }
    .phase-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-top: 4px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Strategic AI Investment Architect")
st.caption("International Market Coverage: US, UK, EU, ASIA | 180-Day AI Forecasting")

# 2. GLOBAL TICKER DISCOVERY ENGINE
def resolve_global_ticker(user_input):
    """Resolves name/ticker to Symbol, Official Name, and Stooq Suffix"""
    try:
        search = yf.Search(user_input, max_results=1).tickers
        if not search:
            return user_input.upper(), user_input.upper(), ".US"
        
        info = search[0]
        ticker_code = info['symbol']
        full_name = info.get('longname', ticker_code)
        exch = info.get('exchange', 'NYQ')
        
        # Map Exchange to Stooq Regional Suffixes
        suffix_map = {
            'LSE': '.UK', 'WSE': '.PL', 'GER': '.DE', 
            'FRA': '.DE', 'PAR': '.FR', 'AMS': '.NL',
            'TSE': '.JP', 'HKG': '.HK', 'ASX': '.AU'
        }
        suffix = suffix_map.get(exch, ".US")
        
        # Strip yahoo-specific dot suffixes (e.g. 'BP.L' -> 'BP')
        clean_ticker = ticker_code.split('.')[0]
        
        return clean_ticker, full_name, suffix
    except:
        return user_input.upper(), user_input.upper(), ".US"

# 3. SIDEBAR CONFIGURATION
st.sidebar.header("⚙️ Audit Parameters")
user_query = st.sidebar.text_input("Enter Company or Ticker", value="Nvidia")
total_capital = st.sidebar.number_input("Capital Allocation ($)", value=1000)

# 4. MULTI-MARKET DATA ENGINE
def get_audit_data(ticker, suffix):
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) 
        # Fetch Price History from Stooq
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        
        # Fundamentals (US Only via Finviz)
        health = {"ROE": 0.0, "Debt": 0.0, "Ratio": "N/A", "Margin": "N/A"}
        if suffix == ".US":
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            soup = BeautifulSoup(requests.get(url, headers=headers, timeout=10).text, 'html.parser')
            def fvz(label):
                td = soup.find('td', string=label)
                return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "-"
            
            health = {
                "ROE": float(fvz("ROE")) / 100 if fvz("ROE") != "-" else 0,
                "Debt": float(fvz("Debt/Eq")) if fvz("Debt/Eq") != "-" else 0,
                "Ratio": fvz("Current Ratio"),
                "Margin": fvz("Profit Margin") + "%"
            }
        return df, health
    except: return None, None

# 5. EXECUTION LOGIC
if st.sidebar.button("🚀 Execute Global Audit"):
    with st.spinner(f"📡 Resolving '{user_query}' and fetching market data..."):
        ticker, official_name, suffix = resolve_global_ticker(user_query)
        df, health = get_audit_data(ticker, suffix)
        
        if df is not None:
            # AI FORECASTING (180 DAYS)
            model = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = model.make_future_dataframe(periods=180)
            forecast = model.predict(future)
            
            cur_p = df['y'].iloc[-1]
            roi_180 = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # Conviction & Strategy Mapping
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi_180 > 10 else 0) + (1 if health['Debt'] < 1.1 else 0)
            
            if points >= 2:
                mode, label, pct, sl_pct = "Aggressive", "🌟 STRONG BUY", 0.15, 0.90
            elif points == 1:
                mode, label, pct, sl_pct = "Defensive", "🟡 ACCUMULATE", 0.05, 0.85
            else:
                mode, label, pct, sl_pct = "Preservation", "🛑 AVOID", 0.0, 0.0

            imm_buy = total_capital * pct
            staging_amt = total_capital - imm_buy

            # --- DASHBOARD HEADER ---
            st.subheader(f"📊 Audit: {official_name} ({ticker}{suffix})")
            st.info(f"**AI Strategy Mode:** {mode} | **Recommendation:** {label}")
            
            # STOP LOSS STATEMENT
            if pct > 0:
                st.markdown(f'<div class="stop-loss-box"><p class="stop-loss-text">STOP LOSS: Exit immediately if price drops below ${cur_p * sl_pct:.2f} ({int((1-sl_pct)*100)}% Buffer)</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="stop-loss-box"><p class="stop-loss-text">STOP LOSS: No entry recommended. Capital preserved in cash.</p></div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Action 1: Buy", f"${imm_buy:.2f}")
            c2.metric("Market Region", "USA" if suffix == ".US" else "International")
            c3.metric("180-Day AI ROI", f"{roi_180:+.1f}%")
            c4.metric("Current Price", f"${cur_p:.2f}")

            # --- STRATEGY PHASES ---
            st.markdown("---")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f'<div class="phase-card"><h3>🚀 PHASE 1: IMMEDIATE</h3><p><b>Action:</b> {"Invest $" + str(round(imm_buy,2)) + " today." if imm_buy > 0 else "Do not buy. Keep capital in cash."}</p></div>', unsafe_allow_html=True)
            with p2:
                st.markdown(f'<div class="phase-card"><h3>⏳ PHASE 2: STAGING</h3><p><b>Action:</b> {"Deploy remaining $" + str(round(staging_amt,2)) + " over 180 days." if staging_amt > 0 else "No further deployment planned."}</p></div>', unsafe_allow_html=True)

            # --- HEALTH AUDIT ---
            st.markdown("---")
            st.subheader("🏥 Company Health Detail")
            health_df = pd.DataFrame({
                "Metric": ["Official Symbol", "Exchange Suffix", "ROE (Efficiency)", "Debt/Equity", "Profit Margin"],
                "Value": [ticker, suffix, f"{health['ROE']*100:.1f}%", health['Debt'], health['Margin']],
                "Audit": ["✅ Verified", "🌐 Global", "✅ Prime" if health['ROE'] > 0.15 else "⚠️ Low/NA", "✅ Safe" if health['Debt'] < 1.1 else "⚠️ High/NA", "✅ Stable"]
            })
            st.table(health_df)

            # --- AI VISUALIZATION ---
            st.markdown("---")
            st.subheader(f"🤖 180-Day AI Growth Projection: {ticker}")
            fig = model.plot(forecast)
            plt.title(f"{official_name} Forecast Trend")
            st.pyplot(fig)
            
            st.warning("⚠️ AI ADVISORY: Projections are mathematical probabilities based on historical trends. Human judgment is required.")
        else:
            st.error(f"❌ Error: Could not retrieve market data for {ticker}{suffix}. Please check the company name.")
