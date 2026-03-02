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
import io
import re
import tempfile
import os
from pathlib import Path

# Document extraction libraries
from pypdf import PdfReader
import docx
import reticker

# -------------------------------
# 1. UI SETUP & THEME-AWARE CSS
# -------------------------------
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

st.markdown('<div class="disclaimer-container">🚨 <b>LEGAL:</b> Educational Tool Only. Fibonacci targets are contingency buy orders. AI projections are mathematical and adjusted for market volatility.</div>', unsafe_allow_html=True)
st.title("🏛️ Strategic AI Investment Architect (V10.3)")

# -------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 1.0
    except:
        return 1.0

def resolve_smart_ticker(user_input):
    """Improved ticker resolution with better error handling"""
    ticker_str = user_input.strip().upper()
    try:
        # Try direct ticker first
        t_obj = yf.Ticker(ticker_str)
        info = t_obj.info
        if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            name = info.get('longName', ticker_str)
            curr = info.get('currency', 'USD')
            return ticker_str, name, "", curr
        
        # Try with search
        s = yf.Search(ticker_str, max_results=1)
        if s.quotes:
            ticker = s.quotes[0]['symbol']
            name = s.quotes[0].get('longname', ticker)
            t_obj = yf.Ticker(ticker)
            curr = t_obj.info.get('currency', 'USD')
            return ticker, name, "", curr
    except Exception as e:
        st.warning(f"Could not resolve {ticker_str}: {e}")
    
    return ticker_str, ticker_str, ".US", "USD"

def get_sector(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        return t_obj.info.get('sector', 'Unknown')
    except:
        return 'Unknown'

def get_enhanced_news(ticker):
    headlines = []
    # Finviz news
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=5)
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
                        headlines.append({
                            'date': dt,
                            'headline': headline,
                            'sentiment': sentiment,
                            'source': 'Finviz'
                        })
                except:
                    continue
    except:
        pass

    # Yahoo Finance news
    try:
        ticker_obj = yf.Ticker(ticker)
        yf_news = ticker_obj.news
        for item in yf_news[:10]:
            try:
                title = item.get('title', '')
                if not title:
                    continue
                ts = item.get('providerPublishTime')
                if ts:
                    dt = datetime.datetime.fromtimestamp(ts)
                else:
                    dt = datetime.datetime.now()
                sentiment = TextBlob(title).sentiment.polarity
                headlines.append({
                    'date': dt,
                    'headline': title,
                    'sentiment': sentiment,
                    'source': 'Yahoo'
                })
            except:
                continue
    except:
        pass

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
        if abs(item['sentiment']) > threshold:
            impactful.append(item)
        elif any(kw in headline_lower for kw in keywords):
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

def get_fundamental_health(ticker, suffix):
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period="5y")
        if df.empty:
            end = datetime.datetime.now()
            start = end - datetime.timedelta(days=1825)
            df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty:
            return None, None
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
        except:
            pass
        return df, health
    except:
        return None, None

def volume_trend_message(df_vol):
    if df_vol.empty or len(df_vol) < 20:
        return "Insufficient volume data."
    df_vol = df_vol.copy()
    df_vol['price_change'] = df_vol['y'].diff()
    df_vol['vol_change'] = df_vol['vol'].diff()

    up_days = df_vol[df_vol['price_change'] > 0]
    up_vol_up = (up_days['vol_change'] > 0).sum()
    down_days = df_vol[df_vol['price_change'] < 0]
    down_vol_up = (down_days['vol_change'] > 0).sum()

    total_up = len(up_days)
    total_down = len(down_days)

    if total_up == 0 or total_down == 0:
        return "Neutral volume pattern (recent price action too one‑sided)."

    up_confirmation = up_vol_up / total_up
    down_confirmation = down_vol_up / total_down

    if up_confirmation > 0.6 and down_confirmation < 0.4:
        return "✅ Volume confirms uptrend: rising prices on rising volume, falling prices on falling volume – bullish."
    elif up_confirmation < 0.4 and down_confirmation > 0.6:
        return "⚠️ Volume divergence: prices rise on low volume but fall on high volume – bearish signal."
    elif up_confirmation > 0.6 and down_confirmation > 0.6:
        return "⚖️ Mixed volume: both up and down days see rising volume – market indecision."
    else:
        return "➡️ Neutral volume trend; no strong confirmation or divergence."

# -------------------------------
# 3. DOCUMENT EXTRACTION CLASS
# -------------------------------
class DocumentPortfolioExtractor:
    def __init__(self):
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
        reticker.config.BLACKLIST.update(["ETF", "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"])

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            return ""

    def extract_text_from_file(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        return ""

    def find_tickers_in_text(self, text: str):
        return self.ticker_extractor.extract(text)

    def find_holdings_near_tickers(self, text: str, tickers: list):
        holdings = []
        for ticker in tickers:
            patterns = [
                rf'{ticker}\s+(\d+(?:\.\d+)?)',
                rf'(\d+(?:\.\d+)?)\s+{ticker}',
                rf'\${ticker}\s+(\d+(?:\.\d+)?)',
                rf'(\d+(?:\.\d+)?)\s+shares?.*{ticker}',
            ]
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    shares = float(match.group(1)) if match.groups() else None
                    holdings.append({
                        'ticker': ticker.upper(),
                        'shares': shares,
                        'source_text': match.group(0)
                    })
                    break
        return holdings

    def validate_tickers(self, tickers: list):
        validated = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info
                if info.get('regularMarketPrice') is not None:
                    validated.append({
                        'ticker': ticker,
                        'name': info.get('longName', ticker),
                        'sector': info.get('sector', 'Unknown'),
                        'current_price': info.get('regularMarketPrice'),
                        'valid': True
                    })
                else:
                    validated.append({'ticker': ticker, 'valid': False})
            except:
                validated.append({'ticker': ticker, 'valid': False})
        return validated

    def process_document(self, file_path: str):
        text = self.extract_text_from_file(file_path)
        if not text:
            return [], "No text could be extracted."

        raw_tickers = self.find_tickers_in_text(text)
        validated = self.validate_tickers(raw_tickers)
        valid_tickers = [v['ticker'] for v in validated if v.get('valid')]

        holdings = self.find_holdings_near_tickers(text, valid_tickers)

        enriched = []
        for h in holdings:
            val_info = next((v for v in validated if v['ticker'] == h['ticker']), {})
            enriched.append({**h, **val_info})

        return enriched, text[:500]

# -------------------------------
# 4. PORTFOLIO ANALYSIS FUNCTIONS
# -------------------------------
@st.cache_data(ttl=3600)
def analyze_single_ticker_for_portfolio(ticker, display_currency):
    try:
        # First resolve the ticker properly
        resolved_ticker, name, suffix, native_curr = resolve_smart_ticker(ticker)
        
        # Get historical data
        t_obj = yf.Ticker(resolved_ticker)
        df = t_obj.history(period="2y")
        
        if df.empty:
            return None
            
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)
        
        fx = get_exchange_rate(native_curr, display_currency)
        cur_p = df['y'].iloc[-1] * fx

        # Simple forecast using Prophet
        try:
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.08
            )
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            target_p_30 = forecast['yhat'].iloc[len(df) + 29] * fx
        except:
            # If Prophet fails, use simple moving average
            target_p_30 = cur_p * 1.05  # 5% growth estimate
        
        growth_30 = ((target_p_30 - cur_p) / cur_p) * 100
        sector = get_sector(resolved_ticker)

        return {
            'name': name,
            'sector': sector,
            'current_price': cur_p,
            'target_30': target_p_30,
            'growth_30': growth_30
        }
    except Exception as e:
        return None

def process_portfolio(uploaded_file, display_currency):
    df_in = None
    
    # Read the file
    try:
        if isinstance(uploaded_file, str):
            df_in = pd.read_csv(io.StringIO(uploaded_file))
        elif hasattr(uploaded_file, 'name'):
            if uploaded_file.name.endswith('.csv'):
                df_in = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df_in = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None, 0

    if df_in is None or df_in.empty:
        return None, None, None, 0

    # Normalize column names
    df_in.columns = df_in.columns.str.strip().str.lower()
    
    # Check for ticker column
    if 'ticker' not in df_in.columns:
        st.error("File must contain 'Ticker' column")
        return None, None, None, 0

    # Set defaults
    if 'shares' not in df_in.columns:
        df_in['shares'] = 1
    else:
        df_in['shares'] = pd.to_numeric(df_in['shares'], errors='coerce').fillna(1)

    if 'purchase price' in df_in.columns:
        df_in['purchase price'] = pd.to_numeric(df_in['purchase price'], errors='coerce')
    else:
        df_in['purchase price'] = np.nan

    if 'sector' not in df_in.columns:
        df_in['sector'] = 'Unknown'

    # Analyze each ticker
    results = []
    total_value = 0
    progress_bar = st.progress(0)
    
    for idx, row in df_in.iterrows():
        ticker = str(row['ticker']).strip().upper()
        shares = float(row['shares'])
        
        progress_bar.progress((idx + 1) / len(df_in))
        
        analysis = analyze_single_ticker_for_portfolio(ticker, display_currency)
        
        if analysis:
            curr_value = shares * analysis['current_price']
            total_value += curr_value
            
            results.append({
                'Ticker': ticker,
                'Name': analysis['name'],
                'Sector': analysis['sector'],
                'Shares': shares,
                'Current Price': analysis['current_price'],
                'Current Value': curr_value,
                '30d Target': analysis['target_30'],
                '30d Growth %': analysis['growth_30']
            })
        else:
            results.append({
                'Ticker': ticker,
                'Name': 'Error',
                'Sector': 'Unknown',
                'Shares': shares,
                'Current Price': 0,
                'Current Value': 0,
                '30d Target': 0,
                '30d Growth %': 0
            })
    
    progress_bar.empty()
    
    if not results:
        return None, None, None, 0
    
    df_portfolio = pd.DataFrame(results)
    
    # Calculate allocations
    valid_mask = df_portfolio['Current Value'] > 0
    if valid_mask.any():
        total_valid = df_portfolio.loc[valid_mask, 'Current Value'].sum()
        df_portfolio['Allocation %'] = 0
        df_portfolio.loc[valid_mask, 'Allocation %'] = (df_portfolio.loc[valid_mask, 'Current Value'] / total_valid * 100)
    else:
        df_portfolio['Allocation %'] = 0
    
    # Sector allocation
    sector_data = df_portfolio[valid_mask].groupby('Sector')['Current Value'].sum().reset_index()
    if not sector_data.empty:
        sector_data['Allocation %'] = (sector_data['Current Value'] / sector_data['Current Value'].sum() * 100)
        sector_data = sector_data.sort_values('Allocation %', ascending=False)
    else:
        sector_data = pd.DataFrame(columns=['Sector', 'Current Value', 'Allocation %'])
    
    # Generate suggestions
    suggestions = []
    
    # Check sector concentration
    high_conc = sector_data[sector_data['Allocation %'] > 30]
    if not high_conc.empty:
        for _, row in high_conc.iterrows():
            suggestions.append(f"⚠️ High concentration in {row['Sector']} ({row['Allocation %']:.1f}%)")
    
    # Check negative growth
    neg_growth = df_portfolio[valid_mask & (df_portfolio['30d Growth %'] < 0)]
    if not neg_growth.empty:
        for _, row in neg_growth.iterrows():
            suggestions.append(f"🔻 {row['Ticker']} has negative growth projection ({row['30d Growth %']:.1f}%)")
    
    # Check portfolio size
    if valid_mask.sum() < 3:
        suggestions.append("📊 Consider adding more holdings for better diversification")
    
    return df_portfolio, sector_data, suggestions, total_value

# -------------------------------
# 5. SIDEBAR CONFIGURATION
# -------------------------------
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Single Ticker / Symbol", value="AAPL")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital (for single ticker)", value=10000)

st.sidebar.markdown("---")
st.sidebar.header("📁 Portfolio Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

st.sidebar.markdown("---")
st.sidebar.header("📄 Document Upload")
uploaded_doc = st.sidebar.file_uploader("Upload PDF or Word", type=['pdf', 'docx', 'doc'])

# Initialize session state
if 'doc_holdings' not in st.session_state:
    st.session_state['doc_holdings'] = None
if 'show_doc_analysis' not in st.session_state:
    st.session_state['show_doc_analysis'] = False

# -------------------------------
# 6. SINGLE TICKER ANALYSIS
# -------------------------------
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner(f"Analyzing {user_query}..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)

        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx

            df['50_Day_MA'] = df['y'].rolling(window=50).mean()
            df['200_Day_MA'] = df['y'].rolling(window=200).mean()

            # Prophet forecast
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.08
            )
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)

            hist_min = df['y'].min()
            hist_max = df['y'].max()
            forecast['yhat'] = forecast['yhat'].clip(lower=hist_min * 0.5, upper=hist_max * 2)

            trend_val = forecast[forecast['ds'] == df['ds'].iloc[-1]]['yhat'].values[0] * fx
            deviation = ((cur_p - trend_val) / trend_val) * 100
            
            target_p_30 = forecast['yhat'].iloc[len(df) + 29] * fx
            ai_roi_30 = ((target_p_30 - cur_p) / cur_p) * 100

            rsi, fibs = calculate_technicals(df)
            
            # News and sentiment
            all_news = get_enhanced_news(ticker)
            impactful_news = filter_impactful_news(all_news)
            
            # Calculate score
            score = 50  # Base score
            if ai_roi_30 > 5:
                score += 20
            if rsi < 70:
                score += 10
            if deviation < 10:
                score += 10
            score = max(0, min(100, score))

            # Display results
            st.subheader(f"📊 {name} Analysis ({ticker})")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"{sym}{cur_p:,.2f}")
            with col2:
                st.metric("30d Target", f"{sym}{target_p_30:,.2f} ({ai_roi_30:.1f}%)")
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("RSI", f"{rsi:.1f}")
            with col4:
                st.metric("Conviction Score", f"{score}/100")
            
            # Forecast chart
            st.subheader("🤖 180-Day Price Forecast")
            fig, ax = plt.subplots(figsize=(12, 6))
            forecast_plot = forecast.copy()
            forecast_plot[['yhat']] *= fx
            ax.plot(df['ds'], df['y'] * fx, 'k.', label='Historical', markersize=2)
            ax.plot(forecast_plot['ds'], forecast_plot['yhat'], 'b-', label='Forecast')
            ax.fill_between(forecast_plot['ds'], 
                           forecast_plot['yhat_lower'] * fx, 
                           forecast_plot['yhat_upper'] * fx, 
                           alpha=0.2, color='b')
            ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=180), 
                        datetime.datetime.now() + datetime.timedelta(days=180)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # News section
            if impactful_news:
                st.subheader("📰 Recent News")
                for news in impactful_news[:5]:
                    emoji = '🟢' if news['sentiment'] > 0 else '🔴' if news['sentiment'] < 0 else '⚪'
                    st.markdown(f"{emoji} {news['headline']}")
        else:
            st.error(f"Could not fetch data for {user_query}")

# -------------------------------
# 7. PORTFOLIO ANALYSIS
# -------------------------------
if uploaded_file is not None:
    st.markdown("---")
    st.header("📁 Portfolio Analysis")
    
    with st.spinner("Analyzing portfolio..."):
        df_port, sector_data, suggestions, total_val = process_portfolio(uploaded_file, display_currency)
    
    if df_port is not None and not df_port.empty:
        sym = "$" if display_currency == "USD" else "€"
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Value", f"{sym}{total_val:,.2f}")
        with col2:
            st.metric("Holdings", len(df_port[df_port['Current Value'] > 0]))
        with col3:
            avg_growth = df_port[df_port['Current Value'] > 0]['30d Growth %'].mean()
            st.metric("Avg 30d Growth", f"{avg_growth:.1f}%")
        
        # Holdings table
        st.subheader("Holdings")
        display_df = df_port.copy()
        
        # Format columns for display
        for col in ['Current Price', 'Current Value', '30d Target']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{sym}{x:,.2f}" if x > 0 else "N/A")
        
        display_df['30d Growth %'] = display_df['30d Growth %'].apply(lambda x: f"{x:.1f}%" if x != 0 else "N/A")
        display_df['Allocation %'] = display_df['Allocation %'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0%")
        
        st.dataframe(display_df)
        
        # Sector allocation
        if not sector_data.empty:
            st.subheader("Sector Allocation")
            fig, ax = plt.subplots()
            ax.pie(sector_data['Current Value'], 
                   labels=sector_data['Sector'], 
                   autopct='%1.1f%%',
                   startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        
        # Suggestions
        if suggestions:
            st.subheader("💡 Suggestions")
            for s in suggestions:
                st.info(s)
    else:
        st.error("Could not analyze portfolio")

# -------------------------------
# 8. DOCUMENT PROCESSING
# -------------------------------
if uploaded_doc is not None:
    if st.sidebar.button("📄 Extract & Analyze Document"):
        with st.spinner("Extracting holdings from document..."):
            # Save temp file
            temp_path = f"temp_{uploaded_doc.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_doc.getbuffer())
            
            # Extract holdings
            extractor = DocumentPortfolioExtractor()
            holdings, preview = extractor.process_document(temp_path)
            os.remove(temp_path)
            
            if holdings:
                st.session_state['doc_holdings'] = holdings
                st.session_state['show_doc_analysis'] = True
                st.success(f"✅ Found {len(holdings)} holdings!")
            else:
                st.warning("No holdings found in document")

# -------------------------------
# 9. DOCUMENT ANALYSIS RESULTS
# -------------------------------
if st.session_state['show_doc_analysis'] and st.session_state['doc_holdings']:
    st.markdown("---")
    st.header("📄 Document Holdings")
    
    # Show extracted holdings
    holdings_df = pd.DataFrame(st.session_state['doc_holdings'])
    st.dataframe(holdings_df)
    
    # Analyze button
    if st.button("📊 Analyze These Holdings"):
        # Convert to portfolio format
        portfolio_data = pd.DataFrame({
            'Ticker': [h['ticker'] for h in st.session_state['doc_holdings']],
            'Shares': [h.get('shares', 1) for h in st.session_state['doc_holdings']],
            'Sector': [h.get('sector', 'Unknown') for h in st.session_state['doc_holdings']]
        })
        
        # Convert to CSV string
        csv_buffer = io.StringIO()
        portfolio_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Analyze
        with st.spinner("Analyzing holdings..."):
            df_port, sector_data, suggestions, total_val = process_portfolio(csv_buffer, display_currency)
        
        if df_port is not None and not df_port.empty:
            sym = "$" if display_currency == "USD" else "€"
            
            st.subheader("Analysis Results")
            st.metric("Total Value", f"{sym}{total_val:,.2f}")
            
            # Holdings table
            display_df = df_port.copy()
            for col in ['Current Price', 'Current Value', '30d Target']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{sym}{x:,.2f}" if x > 0 else "N/A")
            
            display_df['30d Growth %'] = display_df['30d Growth %'].apply(lambda x: f"{x:.1f}%" if x != 0 else "N/A")
            st.dataframe(display_df)
            
            # Sector allocation
            if not sector_data.empty:
                st.subheader("Sector Allocation")
                fig, ax = plt.subplots()
                ax.pie(sector_data['Current Value'], 
                       labels=sector_data['Sector'], 
                       autopct='%1.1f%%',
                       startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            
            # Suggestions
            if suggestions:
                st.subheader("💡 Suggestions")
                for s in suggestions:
                    st.info(s)
