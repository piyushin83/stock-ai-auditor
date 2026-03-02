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

# Document extraction libraries (pure Python, no system deps)
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
st.title("🏛️ Strategic AI Investment Architect (V10.2)")

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
    ticker_str = user_input.strip().upper()
    try:
        t_obj = yf.Ticker(ticker_str)
        if t_obj.fast_info.get('lastPrice') is not None:
            name = t_obj.info.get('longName', ticker_str)
            curr = t_obj.fast_info.get('currency', 'USD')
            return ticker_str, name, ".US", curr
        s = yf.Search(ticker_str, max_results=1)
        if s.tickers:
            res = s.tickers[0]
            ticker = res['symbol']; name = res.get('longname', ticker)
            t_obj_fb = yf.Ticker(ticker)
            native_curr = t_obj_fb.fast_info.get('currency', 'USD')
            return ticker, name, "", native_curr
    except:
        pass
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
    except Exception as e:
        st.warning(f"Finviz news unavailable: {e}")

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
    except Exception as e:
        st.warning(f"Yahoo Finance news unavailable: {e}")

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
            end = datetime.datetime.now(); start = end - datetime.timedelta(days=1825)
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
    """Extract portfolio holdings from digital PDFs and Word docs."""

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
        # Common false positives
        reticker.config.BLACKLIST.update(["ETF", "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"])

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF using pypdf."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            st.warning(f"PDF extraction error: {e}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a Word document using python-docx."""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            st.warning(f"DOCX extraction error: {e}")
            return ""

    def extract_text_from_file(self, file_path: str) -> str:
        """Route to appropriate extractor based on file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        else:
            st.warning(f"Unsupported file type: {ext}. Please upload PDF or Word document.")
            return ""

    def find_tickers_in_text(self, text: str):
        return self.ticker_extractor.extract(text)

    def find_holdings_near_tickers(self, text: str, tickers: list):
        holdings = []
        for ticker in tickers:
            # Look for patterns like "AAPL 100" or "100 AAPL" or "$AAPL 100"
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
                    break  # Take first match per ticker
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
            return [], "No text could be extracted. Ensure the file is a digital PDF (not scanned) or a Word document."

        preview = text[:500] + "..." if len(text) > 500 else text

        raw_tickers = self.find_tickers_in_text(text)
        validated = self.validate_tickers(raw_tickers)
        valid_tickers = [v['ticker'] for v in validated if v.get('valid')]

        holdings = self.find_holdings_near_tickers(text, valid_tickers)

        enriched = []
        for h in holdings:
            val_info = next((v for v in validated if v['ticker'] == h['ticker']), {})
            enriched.append({**h, **val_info})

        return enriched, preview

# -------------------------------
# 4. PORTFOLIO ANALYSIS FUNCTIONS
# -------------------------------
@st.cache_data(ttl=3600)
def analyze_single_ticker_for_portfolio(ticker, display_currency):
    try:
        ticker_resolved, name, suffix, native_curr = resolve_smart_ticker(ticker)
        df, health = get_fundamental_health(ticker_resolved, suffix)
        if df is None:
            return None
        fx = get_exchange_rate(native_curr, display_currency)
        cur_p = df['y'].iloc[-1] * fx

        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.08,
            changepoint_range=0.9,
            seasonality_mode='additive'
        )
        m.fit(df[['ds', 'y']])
        future = m.make_future_dataframe(periods=180)
        forecast = m.predict(future)
        hist_min = df['y'].min()
        hist_max = df['y'].max()
        forecast['yhat'] = forecast['yhat'].clip(lower=hist_min * 0.01, upper=hist_max * 3)

        target_p_30 = forecast['yhat'].iloc[len(df) + 29] * fx
        growth_30 = ((target_p_30 - cur_p) / cur_p) * 100

        sector = get_sector(ticker_resolved)

        return {
            'name': name,
            'sector': sector,
            'current_price': cur_p,
            'target_30': target_p_30,
            'growth_30': growth_30,
            'native_curr': native_curr,
            'fx': fx
        }
    except Exception as e:
        st.warning(f"Error analyzing {ticker}: {e}")
        return None

def process_portfolio(uploaded_file, display_currency):
    df_in = None
    try:
        if isinstance(uploaded_file, str):
            # Handle string input (CSV content)
            df_in = pd.read_csv(io.StringIO(uploaded_file))
        elif uploaded_file.name.endswith('.csv'):
            df_in = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            try:
                df_in = pd.read_excel(uploaded_file, engine='openpyxl')
            except ImportError:
                try:
                    df_in = pd.read_excel(uploaded_file, engine='xlrd')
                except ImportError:
                    st.error("Excel support not installed. Please install openpyxl or use CSV.")
                    return None, None, None, None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None, None

    if df_in is None:
        return None, None, None, None

    df_in.columns = df_in.columns.str.strip().str.lower()
    if 'ticker' not in df_in.columns:
        st.error("File must contain 'Ticker' column.")
        return None, None, None, None

    if 'shares' in df_in.columns:
        df_in['shares'] = pd.to_numeric(df_in['shares'], errors='coerce').fillna(1)
    else:
        df_in['shares'] = 1

    if 'purchase price' in df_in.columns:
        df_in['purchase price'] = pd.to_numeric(df_in['purchase price'], errors='coerce')
    else:
        df_in['purchase price'] = np.nan

    if 'sector' not in df_in.columns:
        df_in['sector'] = 'Unknown'

    results = []
    total_value_display = 0.0
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df_in.iterrows():
        ticker = str(row['ticker']).strip().upper()
        shares = row['shares']
        purchase_price = row['purchase price'] if pd.notna(row['purchase price']) else None
        sector_from_file = row['sector'] if pd.notna(row['sector']) else None

        progress = (idx + 1) / len(df_in)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing {ticker}... ({idx+1}/{len(df_in)})")

        analysis = analyze_single_ticker_for_portfolio(ticker, display_currency)

        if analysis is None:
            results.append({
                'Ticker': ticker,
                'Name': 'Error',
                'Sector': sector_from_file or 'Unknown',
                'Shares': shares,
                'Purchase Price': purchase_price,
                'Current Price': np.nan,
                'Current Value': np.nan,
                '30d Target': np.nan,
                '30d Growth %': np.nan,
                'Status': 'Failed'
            })
            continue

        curr_price = analysis['current_price']
        curr_value = shares * curr_price
        total_value_display += curr_value

        sector = sector_from_file if sector_from_file and sector_from_file != 'Unknown' else analysis.get('sector', 'Unknown')

        results.append({
            'Ticker': ticker,
            'Name': analysis['name'],
            'Sector': sector,
            'Shares': shares,
            'Purchase Price': purchase_price,
            'Current Price': curr_price,
            'Current Value': curr_value,
            '30d Target': analysis['target_30'],
            '30d Growth %': analysis['growth_30'],
            'Status': 'OK'
        })

    progress_bar.empty()
    status_text.empty()

    if not results:
        return None, None, None, None

    df_portfolio = pd.DataFrame(results)
    valid_df = df_portfolio[df_portfolio['Status'] == 'OK']

    if valid_df.empty:
        return df_portfolio, None, ["All tickers failed to analyze."], total_value_display

    total_valid_value = valid_df['Current Value'].sum()
    df_portfolio['Allocation %'] = df_portfolio.apply(
        lambda row: (row['Current Value'] / total_valid_value * 100) if row['Status'] == 'OK' else 0,
        axis=1
    )

    sector_alloc = valid_df.groupby('Sector')['Current Value'].sum().reset_index()
    sector_alloc['Allocation %'] = (sector_alloc['Current Value'] / total_valid_value) * 100
    sector_alloc = sector_alloc.sort_values('Allocation %', ascending=False)

    suggestions = []
    high_conc = sector_alloc[sector_alloc['Allocation %'] > 30]
    if not high_conc.empty:
        for _, row in high_conc.iterrows():
            suggestions.append(f"⚠️ **High concentration** in {row['Sector']} ({row['Allocation %']:.1f}%).")

    common_sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
                      'Industrials', 'Energy', 'Utilities', 'Real Estate',
                      'Communication Services', 'Consumer Defensive', 'Basic Materials']
    present_sectors = set(sector_alloc['Sector'].tolist()) - {'Unknown'}
    missing = [s for s in common_sectors if s not in present_sectors]
    if missing and len(present_sectors) < 5:
        suggestions.append(f"💡 Consider adding: {', '.join(missing[:3])}")

    negative_growth = valid_df[valid_df['30d Growth %'] < 0]
    if not negative_growth.empty:
        for _, row in negative_growth.iterrows():
            suggestions.append(f"🔻 **{row['Ticker']}** negative 30d growth ({row['30d Growth %']:.1f}%).")

    if len(valid_df) < 5:
        suggestions.append(f"📊 Only {len(valid_df)} holdings. Consider adding more.")

    failed_count = len(df_portfolio[df_portfolio['Status'] == 'Failed'])
    if failed_count > 0:
        suggestions.append(f"⚠️ {failed_count} ticker(s) failed: {', '.join(df_portfolio[df_portfolio['Status'] == 'Failed']['Ticker'].tolist())}")

    if not suggestions:
        suggestions.append("✅ Portfolio appears well-diversified.")

    return df_portfolio, sector_alloc, suggestions, total_value_display

# -------------------------------
# 5. SIDEBAR CONFIGURATION
# -------------------------------
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Single Ticker / Symbol", value="NVDA")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital (for single ticker)", value=10000)

st.sidebar.markdown("---")
st.sidebar.header("📁 Portfolio Upload (CSV/Excel)")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

st.sidebar.markdown("---")
st.sidebar.header("📄 Document Upload (PDF/Word)")
st.sidebar.info("Upload a **digital PDF** or Word document (no scanned images, as OCR is not supported on the cloud).")
uploaded_doc = st.sidebar.file_uploader(
    "Upload statement (PDF, DOCX, DOC)",
    type=['pdf', 'docx', 'doc']
)

# Initialize session state for document holdings
if 'doc_holdings' not in st.session_state:
    st.session_state['doc_holdings'] = None

# -------------------------------
# 6. MAIN EXECUTION: SINGLE TICKER
# -------------------------------
if st.sidebar.button("🚀 Run Deep Audit on Single Ticker"):
    with st.spinner("Analyzing..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_fundamental_health(ticker, suffix)

        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx

            df['50_Day_Moving_Average'] = df['y'].rolling(window=50).mean()
            df['200_Day_Moving_Average'] = df['y'].rolling(window=200).mean()

            # Prophet
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.08,
                changepoint_range=0.9,
                seasonality_mode='additive'
            )
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)

            hist_min = df['y'].min()
            hist_max = df['y'].max()
            forecast['yhat'] = forecast['yhat'].clip(lower=hist_min * 0.01, upper=hist_max * 3)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=hist_min * 0.01, upper=hist_max * 3)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=hist_min * 0.01, upper=hist_max * 3)

            trend_val = forecast[forecast['ds'] == df['ds'].iloc[-1]]['yhat'].values[0] * fx
            deviation = ((cur_p - trend_val) / trend_val) * 100
            fair_low = trend_val * 0.95
            fair_high = trend_val * 1.05

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
            if is_death_cross:
                target_p_30 *= 0.96
            ai_roi_30 = ((target_p_30 - cur_p) / cur_p) * 100

            rsi, fibs = calculate_technicals(df)

            all_news = get_enhanced_news(ticker)
            impactful_news = filter_impactful_news(all_news)
            headlines_display = [f"{'🟢' if n['sentiment']>0 else '🔴' if n['sentiment']<0 else '⚪'} {n['headline']}" for n in all_news[:5]]

            if impactful_news:
                avg_news_sentiment = np.mean([n['sentiment'] for n in impactful_news])
            else:
                avg_news_sentiment = 0

            score = 15
            if not is_death_cross:
                score += 20
            if health['ROE'] > 0.12:
                score += 20
            if ai_roi_30 > 0.5:
                score += 30
            if avg_news_sentiment > 0:
                score += 15
            score = max(0, min(100, score))

            if score >= 70:
                verdict, v_col, action, pct = "Strong Buy", "v-green", "ACTION: BUY NOW", 25
            elif score >= 35:
                verdict, v_col, action, pct = "Hold / Neutral", "v-orange", "ACTION: MONITOR", 10
            else:
                verdict, v_col, action, pct = "Sell / Avoid", "v-red", "ACTION: SELL / STAY AWAY", 0

            sl_price = cur_p * (0.95 if rsi > 70 else 0.85 if rsi < 30 else 0.90)

            st.subheader(f"📊 {name} Analysis ({ticker})")

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if deviation > 15:
                    st.warning(f"⚠️ MEAN REVERSION RISK: Price is {deviation:.1f}% above AI baseline trend.")
                else:
                    st.success(f"✅ TREND ALIGNMENT: Price is within {deviation:.1f}% of AI baseline.")
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
                for h in headlines_display:
                    st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)

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
                    st.markdown(f"""
                    <div class="impact-news">
                        <b>{emoji} {news['headline']}</b><br>
                        <span style="font-size:0.85rem;">{date_str} · {news['source']} · Sentiment: {news['sentiment']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🤖 AI Stock 180-Day Projection")
            fig, ax = plt.subplots(figsize=(12, 6))
            forecast_plot = forecast.copy()
            forecast_plot[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
            m.plot(forecast_plot, ax=ax)
            ax.plot(df['ds'], df['50_Day_Moving_Average'] * fx, label='50-Day Moving Average', color='orange', linewidth=2)
            ax.plot(df['ds'], df['200_Day_Moving_Average'] * fx, label='200-Day Moving Average', color='red', linewidth=2)
            if cross_point:
                ax.scatter(cross_point[0], cross_point[1] * fx, color='gold', s=300, marker='*', label=f"{cross_point[2]} CROSS POINT", zorder=5)
            ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=180), datetime.datetime.now() + datetime.timedelta(days=180)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.legend(loc='upper left')
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("📊 12-Month Relative Volume Trend")
            vol_fig, vol_ax = plt.subplots(figsize=(12, 4))
            vol_df = df.tail(252).copy()
            colors = ['#2e7d32' if i > 0 and vol_df.iloc[i]['y'] >= vol_df.iloc[i-1]['y'] else '#c62828' for i in range(len(vol_df))]
            vol_ax.bar(vol_df['ds'], vol_df['vol'], color=colors, alpha=0.7)
            vol_ax.set_ylabel("Shares Traded")
            vol_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            st.pyplot(vol_fig)

            st.caption("🔵 **Volume color:** Green bars = price increased from previous day; Red bars = price decreased from previous day. "
                       "High volume on green days confirms buying interest; high volume on red days signals selling pressure. "
                       "The message below summarises the volume‑price relationship over the last 12 months.")

            vol_message = volume_trend_message(vol_df)
            st.info(f"📈 **Volume Insight:** {vol_message}")

        else:
            st.error("Data Unreachable. Check Symbol.")

# -------------------------------
# 7. PORTFOLIO ANALYSIS (CSV/Excel)
# -------------------------------
if uploaded_file is not None:
    st.markdown("---")
    st.header("📁 Portfolio Analysis")
    with st.spinner("Processing portfolio..."):
        df_port, sector_alloc, suggestions, total_val = process_portfolio(uploaded_file, display_currency)

    if df_port is not None and not df_port.empty:
        sym = "$" if display_currency == "USD" else "€"
        st.metric("Total Portfolio Value", f"{sym}{total_val:,.2f}")

        st.subheader("Holdings Summary")
        display_cols = ['Ticker', 'Name', 'Sector', 'Shares', 'Current Price', 'Current Value', 'Allocation %', '30d Target', '30d Growth %']
        st.dataframe(df_port[display_cols].style.format({
            'Current Price': f'{sym}{{:.2f}}',
            'Current Value': f'{sym}{{:.2f}}',
            'Allocation %': '{:.1f}%',
            '30d Target': f'{sym}{{:.2f}}',
            '30d Growth %': '{:.1f}%'
        }))

        if sector_alloc is not None and not sector_alloc.empty:
            st.subheader("Sector Allocation")
            fig, ax = plt.subplots()
            ax.pie(sector_alloc['Current Value'], labels=sector_alloc['Sector'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        if suggestions:
            st.subheader("💡 Diversification Suggestions")
            for s in suggestions:
                st.markdown(f"- {s}")
        else:
            st.success("✅ Your portfolio appears well-diversified based on sector allocation.")

        st.caption("Note: 30‑day AI growth projections are based on historical trends and news sentiment. They are not guarantees.")
    else:
        st.error("Could not analyze portfolio. Please check your file format and tickers.")

# -------------------------------
# 8. DOCUMENT UPLOAD PROCESSING
# -------------------------------
if st.sidebar.button("🔍 Extract from Document") and uploaded_doc is not None:
    with st.spinner("Extracting holdings from document..."):
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_doc.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_doc.getbuffer())

        extractor = DocumentPortfolioExtractor()
        holdings, preview = extractor.process_document(temp_path)
        os.remove(temp_path)

        st.markdown("---")
        st.subheader("📄 Extracted Holdings")

        if holdings:
            df_doc = pd.DataFrame(holdings)
            display_cols = ['ticker', 'name', 'shares', 'sector', 'current_price']
            available = [c for c in display_cols if c in df_doc.columns]
            st.dataframe(df_doc[available])
            
            # Store in session state
            st.session_state['doc_holdings'] = holdings
            st.success(f"✅ Found {len(holdings)} holdings! Click '📊 Analyze Document Holdings' below to analyze them.")
        else:
            st.warning("No valid holdings found in document. Raw text preview:")
            st.text(preview)

# -------------------------------
# 9. DOCUMENT HOLDINGS ANALYSIS
# -------------------------------
if st.session_state['doc_holdings'] is not None:
    st.markdown("---")
    st.header("📊 Document Holdings Analysis")
    
    if st.button("📊 Analyze Document Holdings"):
        # Convert holdings to DataFrame format expected by process_portfolio
        holdings_df = pd.DataFrame(st.session_state['doc_holdings'])
        
        # Create portfolio format
        portfolio_data = pd.DataFrame({
            'Ticker': [h['ticker'] for h in st.session_state['doc_holdings']],
            'Shares': [h.get('shares', 1) for h in st.session_state['doc_holdings']],
            'Sector': [h.get('sector', 'Unknown') for h in st.session_state['doc_holdings']]
        })
        
        # Convert to CSV string for processing
        csv_buffer = io.StringIO()
        portfolio_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        with st.spinner("Analyzing document holdings..."):
            df_port, sector_alloc, suggestions, total_val = process_portfolio(csv_buffer, display_currency)
        
        if df_port is not None and not df_port.empty:
            sym = "$" if display_currency == "USD" else "€"
            st.metric("Total Portfolio Value", f"{sym}{total_val:,.2f}")
            
            st.subheader("Holdings Summary")
            display_cols = ['Ticker', 'Name', 'Sector', 'Shares', 'Current Price', 'Current Value', 'Allocation %', '30d Target', '30d Growth %']
            st.dataframe(df_port[display_cols].style.format({
                'Current Price': f'{sym}{{:.2f}}',
                'Current Value': f'{sym}{{:.2f}}',
                'Allocation %': '{:.1f}%',
                '30d Target': f'{sym}{{:.2f}}',
                '30d Growth %': '{:.1f}%'
            }))
            
            if sector_alloc is not None and not sector_alloc.empty:
                st.subheader("Sector Allocation")
                fig, ax = plt.subplots()
                ax.pie(sector_alloc['Current Value'], labels=sector_alloc['Sector'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            
            if suggestions:
                st.subheader("💡 Diversification Suggestions")
                for s in suggestions:
                    st.markdown(f"- {s}")
            else:
                st.success("✅ Portfolio appears well-diversified.")
        else:
            st.error("Could not analyze document holdings.")
