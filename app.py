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
import time
import random

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
st.title("🏛️ Strategic AI Investment Architect (V10.4)")

# -------------------------------
# 2. RATE-LIMITED API FUNCTIONS
# -------------------------------
class RateLimiter:
    """Simple rate limiter to avoid Yahoo Finance blocking"""
    def __init__(self, calls_per_second=1):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
    
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last + random.uniform(0.1, 0.3)
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

# Global rate limiter
rate_limiter = RateLimiter(calls_per_second=0.5)  # Max 2 calls per second

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ticker_info(ticker):
    """Get ticker info with rate limiting and caching"""
    rate_limiter.wait_if_needed()
    
    try:
        # Try yfinance first
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        
        # Check if we got valid data
        if info and info.get('regularMarketPrice') is not None:
            return {
                'name': info.get('longName', ticker),
                'currency': info.get('currency', 'USD'),
                'sector': info.get('sector', 'Unknown'),
                'price': info.get('regularMarketPrice', 0),
                'success': True
            }
    except Exception as e:
        if '429' in str(e):
            st.warning(f"Rate limit hit for {ticker}, waiting longer...")
            time.sleep(5)  # Wait longer if rate limited
    
    return {'success': False, 'ticker': ticker}

@st.cache_data(ttl=3600)
def get_historical_data(ticker, period="2y"):
    """Get historical data with retries"""
    max_retries = 3
    
    for attempt in range(max_retries):
        rate_limiter.wait_if_needed()
        
        try:
            t_obj = yf.Ticker(ticker)
            df = t_obj.history(period=period)
            
            if not df.empty:
                return df
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt + random.uniform(1, 3)
                time.sleep(wait_time)
                continue
    
    return None

# -------------------------------
# 3. HELPER FUNCTIONS (Updated with rate limiting)
# -------------------------------
def resolve_smart_ticker(user_input):
    """Improved ticker resolution with rate limiting"""
    ticker_str = user_input.strip().upper()
    
    # Check cache first
    info = get_ticker_info(ticker_str)
    
    if info['success']:
        return ticker_str, info['name'], "", info['currency']
    
    # Try with common suffixes
    suffixes = ['', '.US', '.L', '.DE', '.PA']
    for suffix in suffixes:
        test_ticker = ticker_str + suffix
        info = get_ticker_info(test_ticker)
        if info['success']:
            return test_ticker, info['name'], "", info['currency']
    
    return ticker_str, ticker_str, ".US", "USD"

def get_sector(ticker):
    info = get_ticker_info(ticker)
    return info.get('sector', 'Unknown') if info['success'] else 'Unknown'

def get_enhanced_news(ticker):
    """Get news with rate limiting and multiple sources"""
    headlines = []
    
    # Try Finviz first (no rate limits)
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(req.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if news_table:
            rows = news_table.find_all('tr')
            for row in rows[:8]:  # Limit to 8 news items
                try:
                    a_tag = row.find('a')
                    if a_tag and a_tag.text:
                        headline = a_tag.text
                        sentiment = TextBlob(headline).sentiment.polarity
                        headlines.append({
                            'headline': headline,
                            'sentiment': sentiment,
                            'source': 'Finviz'
                        })
                except:
                    continue
    except:
        pass
    
    # Try Yahoo Finance with rate limiting
    try:
        rate_limiter.wait_if_needed()
        ticker_obj = yf.Ticker(ticker)
        yf_news = ticker_obj.news
        
        for item in yf_news[:5]:
            try:
                title = item.get('title', '')
                if title:
                    sentiment = TextBlob(title).sentiment.polarity
                    headlines.append({
                        'headline': title,
                        'sentiment': sentiment,
                        'source': 'Yahoo'
                    })
            except:
                continue
    except:
        pass
    
    return headlines[:10]  # Return top 10 news items

def calculate_technicals(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    curr_p = df['Close'].iloc[-1]
    recent_low = df['Close'].tail(126).min()
    diff = max(curr_p - recent_low, curr_p * 0.10)
    
    fib_levels = {
        '0.382': curr_p - (diff * 0.382),
        '0.500': curr_p - (diff * 0.500),
        '0.618': curr_p - (diff * 0.618)
    }
    
    return rsi.iloc[-1] if not rsi.empty else 50, fib_levels

def get_fundamental_health(ticker):
    """Get fundamental health metrics"""
    health = {"ROE": 0, "Debt": 0, "PB": 0, "Margin": "N/A", "CurrentRatio": "N/A"}
    
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(req.text, 'html.parser')
        
        def fvz(label):
            td = soup.find('td', string=label)
            if td:
                next_td = td.find_next_sibling('td')
                if next_td:
                    return next_td.text.strip('%').replace(',', '')
            return "-"
        
        health = {
            "ROE": float(fvz("ROE"))/100 if fvz("ROE") != "-" else 0,
            "Debt": float(fvz("Debt/Eq")) if fvz("Debt/Eq") != "-" else 0,
            "PB": float(fvz("P/B")) if fvz("P/B") != "-" else 0,
            "Margin": fvz("Profit Margin") + "%",
            "CurrentRatio": fvz("Current Ratio")
        }
    except:
        pass
    
    return health

def volume_trend_message(df):
    if df.empty or len(df) < 20:
        return "Insufficient volume data."
    
    df = df.copy()
    df['price_change'] = df['Close'].diff()
    df['vol_change'] = df['Volume'].diff()
    
    up_days = df[df['price_change'] > 0]
    up_vol_up = (up_days['vol_change'] > 0).sum()
    down_days = df[df['price_change'] < 0]
    down_vol_up = (down_days['vol_change'] > 0).sum()
    
    total_up = len(up_days)
    total_down = len(down_days)
    
    if total_up == 0 or total_down == 0:
        return "Neutral volume pattern"
    
    up_confirmation = up_vol_up / total_up
    down_confirmation = down_vol_up / total_down
    
    if up_confirmation > 0.6 and down_confirmation < 0.4:
        return "✅ Volume confirms uptrend – bullish"
    elif up_confirmation < 0.4 and down_confirmation > 0.6:
        return "⚠️ Volume divergence – bearish"
    else:
        return "➡️ Neutral volume trend"

# -------------------------------
# 4. DOCUMENT EXTRACTION CLASS
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
        reticker.config.BLACKLIST.update(["ETF", "USD", "EUR", "GBP", "JPY", "CAD"])

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except:
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except:
            return ""

    def process_document(self, file_path: str):
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
        raw_tickers = self.ticker_extractor.extract(text)
        
        # Validate tickers with rate limiting
        holdings = []
        for ticker in raw_tickers[:20]:  # Limit to first 20 tickers
            info = get_ticker_info(ticker)
            if info['success']:
                # Look for share count
                patterns = [
                    rf'{ticker}\s+(\d+(?:\.\d+)?)',
                    rf'(\d+(?:\.\d+)?)\s+{ticker}',
                ]
                
                shares = 1
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match and match.groups():
                        shares = float(match.group(1))
                        break
                
                holdings.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'shares': shares,
                    'sector': info['sector'],
                    'current_price': info['price']
                })
        
        return holdings, text[:500]

# -------------------------------
# 5. PORTFOLIO ANALYSIS FUNCTIONS
# -------------------------------
@st.cache_data(ttl=3600)
def analyze_ticker_for_portfolio(ticker, display_currency):
    """Analyze a single ticker for portfolio use"""
    
    # Get ticker info
    info = get_ticker_info(ticker)
    if not info['success']:
        return None
    
    # Get historical data for forecast
    df = get_historical_data(ticker, "1y")
    
    fx = get_exchange_rate(info['currency'], display_currency)
    cur_p = info['price'] * fx
    
    # Simple forecast (if we have data)
    if df is not None and len(df) > 30:
        try:
            # Use last 30 days for trend
            recent = df['Close'].tail(30)
            trend = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
            target_p_30 = cur_p * (1 + trend/100 * 0.5)  # Half the recent trend
        except:
            target_p_30 = cur_p * 1.05  # Default 5% growth
    else:
        target_p_30 = cur_p * 1.05
    
    growth_30 = ((target_p_30 - cur_p) / cur_p) * 100
    
    return {
        'name': info['name'],
        'sector': info['sector'],
        'current_price': cur_p,
        'target_30': target_p_30,
        'growth_30': growth_30
    }

def process_portfolio(uploaded_file, display_currency):
    """Process uploaded portfolio file"""
    
    # Read file
    try:
        if uploaded_file.name.endswith('.csv'):
            df_in = pd.read_csv(uploaded_file)
        else:
            df_in = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None, 0
    
    # Normalize columns
    df_in.columns = df_in.columns.str.strip().str.lower()
    
    if 'ticker' not in df_in.columns:
        st.error("File must contain 'Ticker' column")
        return None, None, None, 0
    
    # Set defaults
    if 'shares' not in df_in.columns:
        df_in['shares'] = 1
    
    # Analyze each ticker with progress
    results = []
    total_value = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df_in.iterrows():
        ticker = str(row['ticker']).strip().upper()
        shares = float(row['shares'])
        
        status_text.text(f"Analyzing {ticker}... ({idx+1}/{len(df_in)})")
        progress_bar.progress((idx + 1) / len(df_in))
        
        analysis = analyze_ticker_for_portfolio(ticker, display_currency)
        
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
                '30d Growth %': analysis['growth_30'],
                'Allocation %': 0  # Will calculate after
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
                '30d Growth %': 0,
                'Allocation %': 0
            })
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return None, None, None, 0
    
    df_portfolio = pd.DataFrame(results)
    
    # Calculate allocations
    valid_mask = df_portfolio['Current Value'] > 0
    if valid_mask.any():
        total_valid = df_portfolio.loc[valid_mask, 'Current Value'].sum()
        df_portfolio.loc[valid_mask, 'Allocation %'] = (
            df_portfolio.loc[valid_mask, 'Current Value'] / total_valid * 100
        )
    
    # Sector allocation
    sector_data = df_portfolio[valid_mask].groupby('Sector')['Current Value'].sum().reset_index()
    if not sector_data.empty:
        sector_data['Allocation %'] = (
            sector_data['Current Value'] / sector_data['Current Value'].sum() * 100
        )
        sector_data = sector_data.sort_values('Allocation %', ascending=False)
    
    # Generate suggestions
    suggestions = []
    
    # Check concentration
    high_conc = sector_data[sector_data['Allocation %'] > 30]
    for _, row in high_conc.iterrows():
        suggestions.append(f"⚠️ High concentration in {row['Sector']} ({row['Allocation %']:.1f}%)")
    
    # Check negative growth
    neg_growth = df_portfolio[valid_mask & (df_portfolio['30d Growth %'] < -5)]
    for _, row in neg_growth.iterrows():
        suggestions.append(f"🔻 {row['Ticker']} has negative outlook ({row['30d Growth %']:.1f}%)")
    
    # Check portfolio size
    if valid_mask.sum() < 3:
        suggestions.append("📊 Consider adding more holdings")
    
    return df_portfolio, sector_data, suggestions, total_value

# -------------------------------
# 6. SIDEBAR CONFIGURATION
# -------------------------------
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Single Ticker", value="AAPL")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

st.sidebar.markdown("---")
st.sidebar.header("📁 Portfolio Upload")
uploaded_file = st.sidebar.file_uploader("CSV or Excel", type=['csv', 'xlsx', 'xls'])

st.sidebar.markdown("---")
st.sidebar.header("📄 Document Upload")
uploaded_doc = st.sidebar.file_uploader("PDF or Word", type=['pdf', 'docx', 'doc'])

# Initialize session state
if 'doc_holdings' not in st.session_state:
    st.session_state['doc_holdings'] = None

# -------------------------------
# 7. SINGLE TICKER ANALYSIS
# -------------------------------
if st.sidebar.button("🚀 Analyze Single Stock"):
    with st.spinner(f"Analyzing {user_query}..."):
        
        # Get ticker info
        ticker, name, suffix, currency = resolve_smart_ticker(user_query)
        
        # Get historical data
        df = get_historical_data(ticker, "2y")
        
        if df is not None and not df.empty:
            fx = get_exchange_rate(currency, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['Close'].iloc[-1] * fx
            
            # Calculate moving averages
            df['50_MA'] = df['Close'].rolling(window=50).mean()
            df['200_MA'] = df['Close'].rolling(window=200).mean()
            
            # Simple forecast using trend
            recent_trend = df['Close'].tail(30).pct_change().mean() * 100
            target_p_30 = cur_p * (1 + recent_trend/100)
            growth_30 = ((target_p_30 - cur_p) / cur_p) * 100
            
            # Technicals
            rsi, fibs = calculate_technicals(df)
            
            # News
            news = get_enhanced_news(ticker)
            
            # Fundamentals
            health = get_fundamental_health(ticker)
            
            # Display
            st.subheader(f"📊 {name} Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"{sym}{cur_p:,.2f}")
            with col2:
                st.metric("30d Target", f"{sym}{target_p_30:,.2f}")
            with col3:
                st.metric("RSI", f"{rsi:.1f}")
            
            # Chart
            st.subheader("📈 180-Day Price History")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index[-180:], df['Close'].iloc[-180:] * fx, 'b-', label='Price')
            ax.plot(df.index[-180:], df['50_MA'].iloc[-180:] * fx, 'orange', label='50-day MA', alpha=0.7)
            ax.plot(df.index[-180:], df['200_MA'].iloc[-180:] * fx, 'red', label='200-day MA', alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Price ({sym})')
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # News
            if news:
                st.subheader("📰 Recent News")
                for item in news[:5]:
                    emoji = '🟢' if item['sentiment'] > 0 else '🔴' if item['sentiment'] < 0 else '⚪'
                    st.markdown(f"{emoji} {item['headline']} *({item['source']})*")
            
            # Fundamentals
            st.subheader("🏥 Fundamentals")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROE", f"{health['ROE']*100:.1f}%")
                st.metric("P/B Ratio", f"{health['PB']:.2f}")
            with col2:
                st.metric("Debt/Equity", f"{health['Debt']:.2f}")
                st.metric("Profit Margin", health['Margin'])
            
            # Fibonacci levels
            st.subheader("📊 Fibonacci Support Levels")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("0.382", f"{sym}{fibs['0.382']*fx:,.2f}")
            with col2:
                st.metric("0.500", f"{sym}{fibs['0.500']*fx:,.2f}")
            with col3:
                st.metric("0.618", f"{sym}{fibs['0.618']*fx:,.2f}")
            
        else:
            st.error(f"Could not fetch data for {user_query}")

# -------------------------------
# 8. PORTFOLIO ANALYSIS
# -------------------------------
if uploaded_file is not None:
    st.markdown("---")
    st.header("📁 Portfolio Analysis")
    
    with st.spinner("Analyzing portfolio (this may take a minute)..."):
        df_port, sector_data, suggestions, total_val = process_portfolio(uploaded_file, display_currency)
    
    if df_port is not None and not df_port.empty:
        sym = "$" if display_currency == "USD" else "€"
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Value", f"{sym}{total_val:,.2f}")
        with col2:
            valid_count = len(df_port[df_port['Current Value'] > 0])
            st.metric("Holdings", valid_count)
        with col3:
            avg_growth = df_port[df_port['Current Value'] > 0]['30d Growth %'].mean()
            st.metric("Avg 30d Growth", f"{avg_growth:.1f}%")
        
        # Holdings table
        st.subheader("Holdings")
        display_df = df_port.copy()
        
        # Format columns
        for col in ['Current Price', 'Current Value', '30d Target']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{sym}{x:,.2f}" if x > 0 else "N/A"
                )
        
        display_df['30d Growth %'] = display_df['30d Growth %'].apply(
            lambda x: f"{x:.1f}%" if x != 0 else "N/A"
        )
        display_df['Allocation %'] = display_df['Allocation %'].apply(
            lambda x: f"{x:.1f}%" if x > 0 else "0%"
        )
        
        st.dataframe(display_df)
        
        # Sector allocation
        if sector_data is not None and not sector_data.empty:
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
# 9. DOCUMENT PROCESSING
# -------------------------------
if uploaded_doc is not None:
    if st.sidebar.button("📄 Extract Holdings"):
        with st.spinner("Extracting from document..."):
            # Save temp file
            temp_path = f"temp_{uploaded_doc.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_doc.getbuffer())
            
            # Extract
            extractor = DocumentPortfolioExtractor()
            holdings, preview = extractor.process_document(temp_path)
            os.remove(temp_path)
            
            if holdings:
                st.session_state['doc_holdings'] = holdings
                st.success(f"✅ Found {len(holdings)} holdings!")
                
                # Show extracted holdings
                st.subheader("📄 Extracted Holdings")
                holdings_df = pd.DataFrame(holdings)
                st.dataframe(holdings_df)
                
                # Analyze button
                if st.button("📊 Analyze These Holdings"):
                    # Convert to portfolio format
                    portfolio_data = pd.DataFrame({
                        'Ticker': [h['ticker'] for h in holdings],
                        'Shares': [h['shares'] for h in holdings]
                    })
                    
                    # Save to temp CSV
                    temp_csv = io.StringIO()
                    portfolio_data.to_csv(temp_csv, index=False)
                    temp_csv.seek(0)
                    
                    # Analyze
                    with st.spinner("Analyzing holdings..."):
                        df_port, sector_data, suggestions, total_val = process_portfolio(
                            portfolio_data, display_currency
                        )
                    
                    if df_port is not None:
                        sym = "$" if display_currency == "USD" else "€"
                        
                        st.subheader("Analysis Results")
                        st.metric("Total Value", f"{sym}{total_val:,.2f}")
                        
                        # Display table
                        display_df = df_port.copy()
                        for col in ['Current Price', 'Current Value', '30d Target']:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(
                                    lambda x: f"{sym}{x:,.2f}" if x > 0 else "N/A"
                                )
                        
                        display_df['30d Growth %'] = display_df['30d Growth %'].apply(
                            lambda x: f"{x:.1f}%" if x != 0 else "N/A"
                        )
                        
                        st.dataframe(display_df)
                        
                        # Sector chart
                        if sector_data is not None and not sector_data.empty:
                            fig, ax = plt.subplots()
                            ax.pie(sector_data['Current Value'], 
                                   labels=sector_data['Sector'], 
                                   autopct='%1.1f%%')
                            ax.axis('equal')
                            st.pyplot(fig)
                        
                        # Suggestions
                        if suggestions:
                            st.subheader("💡 Suggestions")
                            for s in suggestions:
                                st.info(s)
            else:
                st.warning("No holdings found in document")
