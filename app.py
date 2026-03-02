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
import time
import random
import io
import os
from pathlib import Path
import re

# Document extraction libraries
try:
    from pypdf import PdfReader
    import docx
    import reticker
    DOC_EXTRACTION_AVAILABLE = True
except ImportError:
    DOC_EXTRACTION_AVAILABLE = False
    st.warning("Document extraction libraries not installed. Install pypdf, python-docx, reticker for PDF/Word support.")

# 1. UI SETUP & THEME-AWARE CSS
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    
    .phase-card { background-color: #f4f6f9; color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .news-card { background-color: #ffffff; color: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1); }
    .fib-box { background-color: #e3f2fd; color: #0d47a1; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; font-weight: bold; }
    .impact-news { background-color: #fff3e0; border-left: 8px solid #ff9800; padding: 10px; margin-bottom: 8px; border-radius: 5px; }
    
    @media (prefers-color-scheme: dark) {
        .phase-card { background-color: #1e2129; color: #ffffff; border: 1px solid #3d414b; }
        .news-card { background-color: #262730; color: #ffffff; border-left: 5px solid #00b0ff; }
        .fib-box { background-color: #0d47a1; color: #e3f2fd; border-left: 4px solid #00b0ff; }
        .impact-news { background-color: #332e1f; color: #ffe0b2; border-left: 8px solid #ffb74d; }
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
st.title("🏛️ Strategic AI Investment Architect (V10.0 - Automatic Analysis)")

# 3. RATE LIMITER
class RateLimiter:
    def __init__(self, calls_per_second=0.3):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
    
    def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last + random.uniform(0.1, 0.3)
            time.sleep(sleep_time)
        self.last_call_time = time.time()

rate_limiter = RateLimiter(calls_per_second=0.3)

# 4. HELPER FUNCTIONS (same as your original, with rate limiting)
@st.cache_data(ttl=3600)
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        rate_limiter.wait()
        data = yf.download(pair, period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 1.0
    except: 
        return 1.0

@st.cache_data(ttl=3600)
def resolve_smart_ticker(user_input):
    ticker_str = user_input.strip().upper()
    try:
        rate_limiter.wait()
        t_obj = yf.Ticker(ticker_str)
        if t_obj.fast_info.get('lastPrice') is not None:
            name = t_obj.info.get('longName', ticker_str)
            curr = t_obj.fast_info.get('currency', 'USD')
            return ticker_str, name, ".US", curr
        rate_limiter.wait()
        s = yf.Search(ticker_str, max_results=1)
        if s.tickers:
            res = s.tickers[0]
            ticker = res['symbol']; name = res.get('longname', ticker)
            t_obj_fb = yf.Ticker(ticker)
            native_curr = t_obj_fb.fast_info.get('currency', 'USD')
            return ticker, name, "", native_curr
    except Exception as e:
        if '429' in str(e):
            time.sleep(5)
    return ticker_str, ticker_str, ".US", "USD"

@st.cache_data(ttl=1800)
def get_enhanced_news(ticker):
    headlines = []
    # Finviz
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        req = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(req.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if news_table:
            rows = news_table.find_all('tr')
            for row in rows[:10]:
                try:
                    a_tag = row.find('a')
                    if a_tag and a_tag.text:
                        headline = a_tag.text
                        date_td = row.find('td', class_='nn-date')
                        date_str = date_td.text if date_td else ""
                        if date_str:
                            try:
                                dt = datetime.datetime.strptime(date_str, "%b-%d-%y %I:%M%p")
                            except:
                                dt = datetime.datetime.now()
                        else:
                            dt = datetime.datetime.now()
                        sentiment = TextBlob(headline).sentiment.polarity
                        headlines.append({'date': dt, 'headline': headline, 'sentiment': sentiment, 'source': 'Finviz'})
                except:
                    continue
    except:
        pass
    # Yahoo Finance (rate limited)
    try:
        rate_limiter.wait()
        ticker_obj = yf.Ticker(ticker)
        yf_news = ticker_obj.news
        for item in yf_news[:10]:
            try:
                title = item.get('title', '')
                if not title: continue
                ts = item.get('providerPublishTime')
                if ts: dt = datetime.datetime.fromtimestamp(ts)
                else: dt = datetime.datetime.now()
                sentiment = TextBlob(title).sentiment.polarity
                headlines.append({'date': dt, 'headline': title, 'sentiment': sentiment, 'source': 'Yahoo'})
            except:
                continue
    except:
        pass
    # Remove duplicates
    unique = []
    seen = set()
    for h in headlines:
        if h['headline'] not in seen:
            seen.add(h['headline'])
            unique.append(h)
    unique.sort(key=lambda x: x['date'], reverse=True)
    return unique

def filter_impactful_news(news_list, threshold=0.3, keywords=None):
    if keywords is None:
        keywords = ['earnings', 'dividend', 'fed', 'revenue', 'lawsuit', 'sec',
                    'merger', 'acquisition', 'growth', 'crash', 'ai', 'claude',
                    'agent', 'product', 'launch', 'guidance', 'forecast', 'upgrade',
                    'downgrade', 'target', 'investor', 'conference']
    impactful = []
    for item in news_list:
        headline_lower = item['headline'].lower()
        if abs(item['sentiment']) > threshold or any(kw in headline_lower for kw in keywords):
            impactful.append(item)
    return impactful

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

@st.cache_data(ttl=3600)
def get_fundamental_health(ticker, suffix):
    try:
        rate_limiter.wait()
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
    except Exception as e:
        if '429' in str(e): time.sleep(10)
        return None, None

def volume_trend_message(df_vol):
    if df_vol.empty or len(df_vol) < 20: return "Insufficient volume data."
    df_vol = df_vol.copy()
    df_vol['price_change'] = df_vol['y'].diff()
    df_vol['vol_change'] = df_vol['vol'].diff()
    up_days = df_vol[df_vol['price_change'] > 0]
    down_days = df_vol[df_vol['price_change'] < 0]
    up_vol_up = (up_days['vol_change'] > 0).sum()
    down_vol_up = (down_days['vol_change'] > 0).sum()
    total_up = len(up_days)
    total_down = len(down_days)
    if total_up == 0 or total_down == 0: return "Neutral volume pattern."
    up_confirmation = up_vol_up / total_up
    down_confirmation = down_vol_up / total_down
    if up_confirmation > 0.6 and down_confirmation < 0.4:
        return "✅ Volume confirms uptrend – bullish."
    elif up_confirmation < 0.4 and down_confirmation > 0.6:
        return "⚠️ Volume divergence – bearish."
    elif up_confirmation > 0.6 and down_confirmation > 0.6:
        return "⚖️ Mixed volume – indecision."
    else:
        return "➡️ Neutral volume trend."

# 5. DOCUMENT EXTRACTION CLASS
class DocumentPortfolioExtractor:
    def __init__(self):
        if DOC_EXTRACTION_AVAILABLE:
            self.ticker_extractor = reticker.TickerExtractor(
                deduplicate=True,
                match_config=reticker.TickerMatchConfig(
                    prefixed_uppercase=True,
                    unprefixed_uppercase=True,
                    prefixed_lowercase=True,
                    prefixed_titlecase=True,
                    separators="-."
                )
            )
        else:
            self.ticker_extractor = None

    def extract_text_from_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except: return ""

    def extract_text_from_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except: return ""

    def find_tickers_regex(self, text):
        pattern = r'\b[A-Z]{1,5}\b'
        matches = re.findall(pattern, text)
        blacklist = {'ETF', 'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'THE', 'AND', 'FOR', 'INC', 'LTD'}
        return [m for m in matches if m not in blacklist][:20]

    def process_document(self, file_path):
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            text = self.extract_text_from_docx(file_path)
        else:
            return [], "Unsupported file type"
        if not text:
            return [], "No text could be extracted"
        # Find tickers
        if self.ticker_extractor:
            raw_tickers = self.ticker_extractor.extract(text)
        else:
            raw_tickers = self.find_tickers_regex(text)
        # Validate with yFinance
        holdings = []
        for ticker in raw_tickers[:10]:
            try:
                rate_limiter.wait()
                t_obj = yf.Ticker(ticker)
                info = t_obj.info
                if info.get('regularMarketPrice'):
                    holdings.append({
                        'ticker': ticker,
                        'name': info.get('longName', ticker),
                        'shares': 1,
                        'sector': info.get('sector', 'Unknown'),
                        'current_price': info.get('regularMarketPrice')
                    })
            except:
                continue
        return holdings, text[:500]

# 6. PORTFOLIO ANALYSIS FUNCTIONS
def analyze_ticker_basic(ticker, display_currency):
    """Simplified analysis for portfolio use (no Prophet)"""
    try:
        rate_limiter.wait()
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        price = info.get('regularMarketPrice')
        if not price:
            return None
        currency = info.get('currency', 'USD')
        fx = get_exchange_rate(currency, display_currency)
        price_display = price * fx
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'price': price_display,
            'currency': currency
        }
    except:
        return None

def process_uploaded_file(uploaded_file):
    """Handle any uploaded file: CSV, Excel, PDF, Word"""
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()

    # CSV / Excel
    if file_name.endswith('.csv') or file_name.endswith(('.xlsx', '.xls')):
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            if 'ticker' not in df.columns:
                st.error("CSV/Excel must contain a 'Ticker' column.")
                return None
            if 'shares' not in df.columns:
                df['shares'] = 1
            return {'type': 'tabular', 'data': df}
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

    # PDF / Word
    elif DOC_EXTRACTION_AVAILABLE and (file_name.endswith('.pdf') or file_name.endswith(('.docx', '.doc'))):
        # Save to temp file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        extractor = DocumentPortfolioExtractor()
        holdings, preview = extractor.process_document(temp_path)
        os.remove(temp_path)
        if holdings:
            df = pd.DataFrame(holdings)
            return {'type': 'tabular', 'data': df}
        else:
            st.warning("No holdings found in document.")
            return None
    else:
        st.error("Unsupported file type. Please upload CSV, Excel, PDF, or Word document.")
        return None

# 7. SIDEBAR
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker / Symbol", value="NVDA")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Portfolio / Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, PDF, or Word", 
    type=['csv', 'xlsx', 'xls', 'pdf', 'docx', 'doc']
)

# 8. MAIN EXECUTION - SINGLE TICKER (same as original, triggered by button)
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner(f"Analyzing {user_query}... (this may take a moment)"):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            
            df['50_Day_Moving_Average'] = df['y'].rolling(window=50).mean()
            df['200_Day_Moving_Average'] = df['y'].rolling(window=200).mean()
            
            # Prophet
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
                        changepoint_prior_scale=0.08, changepoint_range=0.9, seasonality_mode='additive')
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            hist_min, hist_max = df['y'].min(), df['y'].max()
            forecast['yhat'] = forecast['yhat'].clip(lower=hist_min*0.5, upper=hist_max*2)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=hist_min*0.5, upper=hist_max*2)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=hist_min*0.5, upper=hist_max*2)
            
            trend_val = forecast[forecast['ds'] == df['ds'].iloc[-1]]['yhat'].values[0] * fx
            deviation = ((cur_p - trend_val) / trend_val) * 100
            fair_low = trend_val * 0.95
            fair_high = trend_val * 1.05
            
            # Moving average cross detection
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
            all_news = get_enhanced_news(ticker)
            impactful_news = filter_impactful_news(all_news)
            headlines_display = [f"{'🟢' if n['sentiment']>0 else '🔴' if n['sentiment']<0 else '⚪'} {n['headline']}" for n in all_news[:5]]
            avg_news_sentiment = np.mean([n['sentiment'] for n in impactful_news]) if impactful_news else 0
            
            # Conviction score
            score = 15
            if not is_death_cross: score += 20
            if health['ROE'] > 0.12: score += 20
            if ai_roi_30 > 0.5: score += 30
            if avg_news_sentiment > 0: score += 15
            score = max(0, min(100, score))
            
            if score >= 70: verdict, v_col, action, pct = "Strong Buy", "v-green", "ACTION: BUY NOW", 25
            elif score >= 35: verdict, v_col, action, pct = "Hold / Neutral", "v-orange", "ACTION: MONITOR", 10
            else: verdict, v_col, action, pct = "Sell / Avoid", "v-red", "ACTION: SELL / STAY AWAY", 0
            
            sl_price = cur_p * (0.95 if rsi > 70 else 0.85 if rsi < 30 else 0.90)
            
            # --- DISPLAY (same as your original) ---
            st.subheader(f"📊 {name} Analysis ({ticker})")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if deviation > 15: st.warning(f"⚠️ MEAN REVERSION RISK: Price is {deviation:.1f}% above AI baseline trend.")
                else: st.success(f"✅ TREND ALIGNMENT: Price is within {deviation:.1f}% of AI baseline.")
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
                st.markdown("### 📰 Recent Headlines")
                for h in headlines_display: st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)
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
            if impactful_news:
                st.markdown("### 🔥 News Likely to Impact Price")
                for news in impactful_news[:7]:
                    emoji = '🟢' if news['sentiment'] > 0.1 else '🔴' if news['sentiment'] < -0.1 else '⚪'
                    date_str = news['date'].strftime("%b %d, %Y")
                    st.markdown(f'<div class="impact-news"><b>{emoji} {news["headline"]}</b><br><span style="font-size:0.85rem;">{date_str} · {news["source"]} · Sentiment: {news["sentiment"]:.2f}</span></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("🤖 AI Stock 180-Day Projection")
            fig, ax = plt.subplots(figsize=(12,6))
            forecast_plot = forecast.copy()
            forecast_plot[['yhat','yhat_lower','yhat_upper']] *= fx
            m.plot(forecast_plot, ax=ax)
            ax.plot(df['ds'], df['50_Day_Moving_Average']*fx, label='50-Day MA', color='orange', linewidth=2)
            ax.plot(df['ds'], df['200_Day_Moving_Average']*fx, label='200-Day MA', color='red', linewidth=2)
            if cross_point:
                ax.scatter(cross_point[0], cross_point[1]*fx, color='gold', s=300, marker='*', label=f"{cross_point[2]} CROSS POINT", zorder=5)
            ax.set_xlim([datetime.datetime.now()-datetime.timedelta(days=180), datetime.datetime.now()+datetime.timedelta(days=180)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.legend(loc='upper left')
            st.pyplot(fig)
            st.markdown("---")
            st.subheader("📊 12-Month Relative Volume Trend")
            vol_fig, vol_ax = plt.subplots(figsize=(12,4))
            vol_df = df.tail(252).copy()
            colors = ['#2e7d32' if i>0 and vol_df.iloc[i]['y'] >= vol_df.iloc[i-1]['y'] else '#c62828' for i in range(len(vol_df))]
            vol_ax.bar(vol_df['ds'], vol_df['vol'], color=colors, alpha=0.7)
            vol_ax.set_ylabel("Shares Traded")
            vol_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            st.pyplot(vol_fig)
            st.caption("🔵 **Volume color:** Green bars = price increased from previous day; Red bars = price decreased from previous day.")
            vol_message = volume_trend_message(vol_df)
            st.info(f"📈 **Volume Insight:** {vol_message}")
        else:
            st.error(f"Data Unreachable for {user_query}. Check symbol or try again later.")

# 9. AUTOMATIC PORTFOLIO/DOCUMENT ANALYSIS
if uploaded_file is not None:
    st.markdown("---")
    st.header("📁 Uploaded File Analysis")
    
    with st.spinner("Processing uploaded file..."):
        result = process_uploaded_file(uploaded_file)
    
    if result and result['type'] == 'tabular':
        df_holdings = result['data']
        
        # Analyze each ticker
        results_list = []
        total_value = 0
        progress_bar = st.progress(0)
        
        for idx, row in df_holdings.iterrows():
            ticker = str(row['ticker']).strip().upper()
            shares = float(row['shares']) if 'shares' in row else 1
            progress_bar.progress((idx + 1) / len(df_holdings))
            
            analysis = analyze_ticker_basic(ticker, display_currency)
            if analysis:
                value = shares * analysis['price']
                total_value += value
                results_list.append({
                    'Ticker': ticker,
                    'Name': analysis['name'],
                    'Sector': analysis['sector'],
                    'Shares': shares,
                    'Current Price': analysis['price'],
                    'Current Value': value
                })
        
        progress_bar.empty()
        
        if results_list:
            df_results = pd.DataFrame(results_list)
            if total_value > 0:
                df_results['Allocation %'] = (df_results['Current Value'] / total_value * 100)
            
            sym = "$" if display_currency == "USD" else "€"
            st.metric("Total Portfolio Value", f"{sym}{total_value:,.2f}")
            
            st.subheader("Holdings")
            display_df = df_results.copy()
            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"{sym}{x:,.2f}")
            display_df['Current Value'] = display_df['Current Value'].apply(lambda x: f"{sym}{x:,.2f}")
            display_df['Allocation %'] = display_df['Allocation %'].apply(lambda x: f"{x:.1f}%" if x>0 else "0%")
            st.dataframe(display_df)
            
            # Sector allocation
            if 'Sector' in df_results.columns:
                st.subheader("Sector Allocation")
                sector_data = df_results.groupby('Sector')['Current Value'].sum().reset_index()
                fig, ax = plt.subplots()
                ax.pie(sector_data['Current Value'], labels=sector_data['Sector'], autopct='%1.1f%%')
                ax.axis('equal')
                st.pyplot(fig)
        else:
            st.error("No valid tickers could be analyzed.")
