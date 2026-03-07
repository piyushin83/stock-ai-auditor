import streamlit as st

# ── set_page_config MUST be the absolute first st.* call ─────────────────────
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

# ── Multi-user auth gate — runs right after set_page_config ──────────────────
from auth import require_auth, logout_button
_current_user = require_auth()
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from prophet import Prophet
# pandas_datareader removed — not compatible with Python 3.14
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import yfinance as yf
import time
import random
import os
from pathlib import Path
import re
import traceback
import warnings
warnings.filterwarnings('ignore')

# ── Hide Streamlit chrome ─────────────────────────────────────────────────────
_HIDE_ST_STYLE = """
<style>
    #MainMenu {visibility:hidden !important; display:none !important;}
    footer {visibility:hidden !important; display:none !important;}
    header {visibility:hidden !important; display:none !important;}
    [data-testid="stToolbar"]      {visibility:hidden !important; display:none !important;}
    [data-testid="stDecoration"]   {display:none !important;}
    [data-testid="stStatusWidget"] {visibility:hidden !important; display:none !important;}
    [data-testid="manage-app-button"] {display:none !important;}
    .viewerBadge_container__1QSob {display:none !important;}
    .stDeployButton               {display:none !important;}
    ._profileContainer_gzau3_53   {display:none !important;}
    ._chatAvatarIcon_gzau3_37      {display:none !important;}
    .styles_viewerBadge__CvC9N    {display:none !important;}
    iframe[title="streamlit_feedback.streamlit_feedback"] {display:none !important;}
    div[class*="StatusWidget"]    {display:none !important;}
    div[class*="manageApp"]       {display:none !important;}
    button[kind="header"]         {display:none !important;}
    /* Hide bottom-right manage app bar entirely */
    .st-emotion-cache-h4xjwg      {display:none !important;}
    .st-emotion-cache-1dp5vir      {display:none !important;}
    section[data-testid="stBottom"] {display:none !important;}
    div[data-testid="stBottom"]   {display:none !important;}
</style>
"""

# ── Optional libs ─────────────────────────────────────────────────────────────
missing_libs = []
try:
    from pypdf import PdfReader
    import docx
    import reticker
    PDF_DOCX_AVAILABLE = True
except ImportError:
    PDF_DOCX_AVAILABLE = False
    missing_libs.append("pypdf python-docx reticker")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── Page config & CSS ─────────────────────────────────────────────────────────
st.markdown(_HIDE_ST_STYLE + """
<style>
    [data-testid="stMetricValue"] { font-size:26px !important; font-weight:800 !important; color:#1f77b4; }
    .phase-card { background-color:#f4f6f9; color:#1a1a1a; padding:20px; border-radius:10px; border:1px solid #dcdcdc; min-height:420px; }
    .news-card { background-color:#ffffff; color:#1a1a1a; padding:15px; border-radius:8px; border-left:5px solid #0288d1; margin-bottom:10px; font-size:14px; box-shadow:1px 1px 5px rgba(0,0,0,0.1); }
    .fib-box { background-color:#e3f2fd; color:#0d47a1; padding:10px; border-radius:5px; margin-top:5px; border-left:4px solid #1565c0; font-family:monospace; font-weight:bold; }
    .impact-news { background-color:#fff3e0; border-left:8px solid #ff9800; padding:10px; margin-bottom:8px; border-radius:5px; }
    .roadmap-card { background:linear-gradient(135deg,#1a237e 0%,#283593 50%,#1565c0 100%); color:#fff; padding:20px; border-radius:12px; margin-bottom:16px; box-shadow:0 4px 15px rgba(21,101,192,0.3); }
    .roadmap-phase { background:rgba(255,255,255,0.12); border-radius:8px; padding:14px; margin-bottom:10px; border-left:4px solid #64b5f6; }
    .roadmap-ticker { display:inline-block; background:rgba(100,181,246,0.2); border:1px solid #64b5f6; border-radius:4px; padding:2px 8px; margin:2px; font-family:monospace; font-size:12px; font-weight:bold; }
    .risk-badge-low { background:#1b5e20; color:#a5d6a7; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold; }
    .risk-badge-med { background:#e65100; color:#ffcc80; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold; }
    .risk-badge-high { background:#b71c1c; color:#ef9a9a; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold; }
    .impact-announcement { background-color:#fff3cd; color:#856404; padding:15px; border-radius:5px; border-left:8px solid #ffc107; margin-bottom:20px; font-weight:bold; }
    .stop-loss-box { background-color:#fff1f1; border-left:8px solid #ff4b4b; padding:15px; margin-bottom:20px; color:#b71c1c; font-weight:bold; }
    .verdict-box { padding:20px; border-radius:8px; margin-bottom:20px; font-weight:bold; font-size:22px; text-align:center; color:white; text-transform:uppercase; }
    .v-green { background-color:#2e7d32; } .v-orange { background-color:#f57c00; } .v-red { background-color:#c62828; }
    .disclaimer-container { background-color:#262730; color:#aaa; padding:15px; border-radius:5px; font-size:12px; margin-bottom:20px; border:1px solid #444; }
    @media (prefers-color-scheme:dark) {
        .phase-card { background-color:#1e2129; color:#ffffff; border:1px solid #3d414b; }
        .news-card { background-color:#262730; color:#ffffff; border-left:5px solid #00b0ff; }
        .fib-box { background-color:#0d47a1; color:#e3f2fd; border-left:4px solid #00b0ff; }
        .impact-news { background-color:#332e1f; color:#ffe0b2; border-left:8px solid #ffb74d; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer-container">🚨 <b>LEGAL:</b> Educational Tool Only. AI projections are mathematical estimates. Not financial advice.</div>', unsafe_allow_html=True)
st.title("🏛️ Strategic AI Investment Architect (V11.5)")

if missing_libs:
    st.sidebar.warning(f"Optional libs missing: {' '.join(missing_libs)}")

# Logout in sidebar
logout_button()

# ── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, cps=0.3):
        self.min_interval = 1.0 / cps
        self.last = 0
    def wait(self):
        elapsed = time.time() - self.last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed + random.uniform(0.1, 0.3))
        self.last = time.time()

rate_limiter = RateLimiter()
_DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"

def show_error(msg, detail=""):
    st.error(msg)
    if _DEV_MODE and detail:
        with st.expander("🔧 Dev details"):
            st.code(detail)

# ── Exchange rate ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        rate_limiter.wait()
        data = yf.download(f"{from_curr}{to_curr}=X", period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 1.0
    except:
        return 1.0

# ── Ticker resolution ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def resolve_smart_ticker(user_input):
    t = user_input.strip().upper()
    try:
        rate_limiter.wait()
        obj  = yf.Ticker(t)
        info = obj.info
        lp   = obj.fast_info.get('lastPrice')
        if lp:
            name = info.get('longName') or info.get('shortName') or t
            curr = obj.fast_info.get('currency', 'USD') or 'USD'
            return t, name.strip(), ".US", curr, None
        rate_limiter.wait()
        s = yf.Search(t, max_results=3)
        if s.tickers:
            r    = s.tickers[0]
            sym  = r['symbol']
            name = r.get('longname') or r.get('shortname') or sym
            curr = yf.Ticker(sym).fast_info.get('currency', 'USD') or 'USD'
            return sym, name.strip(), "", curr, None
    except Exception as e:
        return t, t, ".US", "USD", str(e)
    return t, t, ".US", "USD", None

# ── Sector lookup ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_ticker_sector_online(ticker):
    try:
        rate_limiter.wait()
        info = yf.Ticker(ticker).info
        s, i = info.get('sector',''), info.get('industry','')
        if s and s not in ('','N/A','None'): return s, i
    except: pass
    try:
        soup = BeautifulSoup(requests.get(
            f"https://finviz.com/quote.ashx?t={ticker}",
            headers={'User-Agent':'Mozilla/5.0'}, timeout=10).text, 'html.parser')
        def fv(l):
            td = soup.find('td', string=l)
            return td.find_next_sibling('td').text.strip() if td else None
        return fv('Sector') or 'Unknown', fv('Industry') or ''
    except: pass
    return 'Unknown', ''

# ── News ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_enhanced_news(ticker):
    headlines = []
    try:
        soup = BeautifulSoup(requests.get(
            f"https://finviz.com/quote.ashx?t={ticker}",
            headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=10).text, 'html.parser')
        tbl = soup.find(id='news-table')
        if tbl:
            for row in tbl.find_all('tr')[:10]:
                try:
                    a = row.find('a')
                    if not (a and a.text): continue
                    dtd = row.find('td', class_='nn-date')
                    try: dt = datetime.datetime.strptime(dtd.text, "%b-%d-%y %I:%M%p") if dtd else datetime.datetime.now()
                    except: dt = datetime.datetime.now()
                    headlines.append({'date':dt,'headline':a.text,'sentiment':TextBlob(a.text).sentiment.polarity,'source':'Finviz'})
                except: continue
    except: pass
    try:
        rate_limiter.wait()
        for item in yf.Ticker(ticker).news[:10]:
            try:
                title = item.get('title','')
                if not title: continue
                ts = item.get('providerPublishTime')
                dt = datetime.datetime.fromtimestamp(ts) if ts else datetime.datetime.now()
                headlines.append({'date':dt,'headline':title,'sentiment':TextBlob(title).sentiment.polarity,'source':'Yahoo'})
            except: continue
    except: pass
    seen, unique = set(), []
    for h in headlines:
        if h['headline'] not in seen:
            seen.add(h['headline']); unique.append(h)
    return sorted(unique, key=lambda x: x['date'], reverse=True)

def filter_impactful_news(news_list, threshold=0.3):
    kw = ['earnings','dividend','fed','revenue','lawsuit','sec','merger','acquisition',
          'growth','crash','ai','product','launch','guidance','forecast','upgrade','downgrade']
    return [n for n in news_list if abs(n['sentiment'])>threshold or any(k in n['headline'].lower() for k in kw)]

# ── Technicals ────────────────────────────────────────────────────────────────
def calculate_technicals(df):
    delta = df['y'].diff()
    gain  = (delta.where(delta>0,0)).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    df['rsi'] = 100 - (100/(1+gain/loss))
    cp  = df['y'].iloc[-1]
    low = df['y'].tail(126).min()
    d   = max(cp-low, cp*0.10)
    return df['rsi'].iloc[-1], {'0.382':cp-d*0.382,'0.500':cp-d*0.500,'0.618':cp-d*0.618}

# ── Fundamental health ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_fundamental_health(ticker, suffix):
    try:
        rate_limiter.wait()
        obj = yf.Ticker(ticker)
        df  = obj.history(period="5y")
        if df.empty:
            # pandas_datareader removed (incompatible with Python 3.14)
            # Try yfinance with longer period as fallback
            try:
                df = yf.Ticker(ticker).history(period="max")
            except Exception:
                pass
        if df is None or df.empty: return None, None, "No historical data"
        df = df.reset_index().rename(columns={'Date':'ds','Close':'y','Volume':'vol'}).sort_values('ds')
        df['ds'] = df['ds'].dt.tz_localize(None)
        health = {"ROE":0,"Debt":0,"PB":0,"Margin":"N/A","CurrentRatio":"N/A"}
        try:
            soup = BeautifulSoup(requests.get(
                f"https://finviz.com/quote.ashx?t={ticker}",
                headers={'User-Agent':'Mozilla/5.0'}).text,'html.parser')
            def fv(l):
                td = soup.find('td', string=l)
                return td.find_next_sibling('td').text.strip('%').replace(',','') if td else "-"
            health = {
                "ROE":   float(fv("ROE"))/100    if fv("ROE")!="-"    else 0,
                "Debt":  float(fv("Debt/Eq"))    if fv("Debt/Eq")!="-" else 0,
                "PB":    float(fv("P/B"))         if fv("P/B")!="-"    else 0,
                "Margin":fv("Profit Margin")+"%", "CurrentRatio":fv("Current Ratio")
            }
        except: pass
        return df, health, None
    except Exception as e:
        return None, None, str(e)

# ── Volume trend ──────────────────────────────────────────────────────────────
def volume_trend_message(df_vol):
    if df_vol.empty or len(df_vol)<20: return "Insufficient volume data."
    dv = df_vol.copy()
    dv['pc'] = dv['y'].diff(); dv['vc'] = dv['vol'].diff()
    up = dv[dv['pc']>0]; dn = dv[dv['pc']<0]
    if not len(up) or not len(dn): return "Neutral volume pattern."
    uc = (up['vc']>0).sum()/len(up); dc = (dn['vc']>0).sum()/len(dn)
    if uc>0.6 and dc<0.4: return "✅ Volume confirms uptrend – bullish."
    if uc<0.4 and dc>0.6: return "⚠️ Volume divergence – bearish."
    if uc>0.6 and dc>0.6: return "⚖️ Mixed volume – indecision."
    return "➡️ Neutral volume trend."

# ── 7-day news-weighted target ────────────────────────────────────────────────
def compute_7day_target(df, forecast, all_news, is_death_cross, fx):
    try:
        last_ds = df['ds'].iloc[-1]
        rows = forecast[forecast['ds']>last_ds].head(7)
        base_raw = float(rows['yhat'].iloc[-1]) if len(rows)>=7 else float(df['y'].iloc[-1])*1.005
    except:
        base_raw = float(df['y'].iloc[-1])*1.005
    now = datetime.datetime.now()
    s_sum = w_sum = 0.0
    for item in all_news:
        age = max((now-item['date']).total_seconds()/86400, 0.01)
        if age>7: continue
        w = 1.0/age; s_sum += item['sentiment']*w; w_sum += w
    news_adj   = (s_sum/w_sum if w_sum>0 else 0)*0.03
    mom_adj    = float(df['y'].pct_change().tail(5).mean())*0.5
    cross_pen  = -0.02 if is_death_cross else 0.0
    adj        = max(0.95, min(1.05, 1+news_adj+mom_adj+cross_pen))
    return base_raw*adj*fx

# ── Portfolio extractor ───────────────────────────────────────────────────────
class PortfolioExtractor:
    def __init__(self):
        self.ticker_extractor = None
        self.ocr_available = OCR_AVAILABLE
        self.ocr_reader = None
        if PDF_DOCX_AVAILABLE:
            try:
                self.ticker_extractor = reticker.TickerExtractor(
                    deduplicate=True,
                    match_config=reticker.TickerMatchConfig(
                        prefixed_uppercase=True, unprefixed_uppercase=True,
                        prefixed_lowercase=True, prefixed_titlecase=True, separators="-."))
            except: pass

    def _get_ocr(self):
        if not self.ocr_reader and self.ocr_available:
            try: self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except: self.ocr_available = False
        return self.ocr_reader

    def _pdf_text(self, path):
        try: return "\n".join(p.extract_text() for p in PdfReader(path).pages if p.extract_text()), None
        except Exception as e: return "", str(e)

    def _docx_text(self, path):
        try: return "\n".join(p.text for p in docx.Document(path).paragraphs), None
        except Exception as e: return "", str(e)

    def _img_text(self, path):
        r = self._get_ocr()
        if not r: return "", "OCR not available"
        try: return "\n".join(r.readtext(path, detail=0, paragraph=True)), None
        except Exception as e: return "", str(e)

    def _regex_tickers(self, text):
        bl = {'ETF','USD','EUR','GBP','JPY','CAD','AUD','CHF','THE','AND','FOR','INC','LTD','LLC','NYSE','NASDAQ'}
        return list(set(m for m in re.findall(r'\b[A-Z]{1,5}\b', text) if m not in bl))

    def process_file(self, path):
        ext = Path(path).suffix.lower()
        text, err = "", None
        if ext=='.pdf':           text, err = self._pdf_text(path)
        elif ext in ['.docx','.doc']: text, err = self._docx_text(path)
        elif ext in ['.png','.jpg','.jpeg','.tiff','.bmp']: text, err = self._img_text(path)
        else: return [], "Unsupported file type", ""
        if err: return [], f"Extraction error: {err}", ""
        if not text.strip(): return [], "No text extracted.", ""
        preview = text[:1000]+("..." if len(text)>1000 else "")
        raw = list(set(self.ticker_extractor.extract(text) if self.ticker_extractor else self._regex_tickers(text)))
        holdings, errs = [], []
        for t in raw[:15]:
            try:
                rate_limiter.wait()
                info = yf.Ticker(t).info
                if info.get('regularMarketPrice'):
                    holdings.append({'ticker':t,'name':info.get('longName',t),
                                     'shares':1.0,'sector':info.get('sector','Unknown'),
                                     'current_price':info.get('regularMarketPrice')})
                else: errs.append(f"{t}: no price")
            except Exception as e: errs.append(f"{t}: {e}")
        return holdings, preview, f"{len(raw)} tickers found, {len(holdings)} validated."

# ── Basic ticker analysis ─────────────────────────────────────────────────────
def analyze_ticker_basic(ticker, display_currency):
    try:
        rate_limiter.wait()
        info = yf.Ticker(ticker).info
        price = info.get('regularMarketPrice')
        if not price: return None, "No market price"
        currency = info.get('currency','USD')
        fx       = get_exchange_rate(currency, display_currency)
        sector   = info.get('sector','')
        industry = info.get('industry','')
        if not sector or sector in ('','N/A','None'):
            sector, industry = get_ticker_sector_online(ticker)
        return {'name':info.get('longName',ticker),'sector':sector or 'Unknown',
                'industry':industry or '','price':price*fx,'currency':currency}, None
    except Exception as e:
        return None, str(e)

# ── Batch scoring (live market intelligence) ──────────────────────────────────
@st.cache_data(ttl=3600)
def batch_score_candidates(candidates_tuple):
    tickers = list(candidates_tuple)
    results = {}
    try:
        batch = yf.Tickers(" ".join(tickers))
        for t in tickers:
            try:
                info = batch.tickers[t].info
                cur  = info.get('currentPrice') or info.get('regularMarketPrice') or 0
                tgt  = info.get('targetMeanPrice') or 0
                rec  = info.get('recommendationMean') or 3.0
                eg   = info.get('earningsGrowth')  or 0
                rg   = info.get('revenueGrowth')   or 0
                lo   = info.get('fiftyTwoWeekLow')  or cur*0.7
                hi   = info.get('fiftyTwoWeekHigh') or cur*1.3
                up   = ((tgt-cur)/cur*100) if cur>0 and tgt>0 else 0
                cs   = max(0,(5-rec)/4*100)
                gs   = min(100,max(0,(eg+rg)*100))
                ms   = ((cur-lo)/(hi-lo)*100) if hi>lo else 50
                us   = min(100,max(0,up*3))
                sc   = us*0.35 + cs*0.30 + gs*0.20 + ms*0.15
                lbl  = "Strong Buy" if rec<=1.5 else "Buy" if rec<=2.5 else "Hold" if rec<=3.5 else "Underperform"
                name = info.get('shortName') or info.get('longName') or t
                results[t] = {'ticker':t,'name':name,'score':round(sc,1),
                               'upside_pct':round(up,1),'analyst_label':lbl,
                               'summary':f"Analyst:{lbl} | Upside:{up:+.1f}% | EPS:{eg*100:.1f}% | Rev:{rg*100:.1f}%",
                               'valid':cur>0}
            except:
                results[t] = {'ticker':t,'name':t,'score':0,'upside_pct':0,
                               'analyst_label':'N/A','summary':'Data unavailable','valid':False}
    except:
        for t in tickers:
            results[t] = {'ticker':t,'name':t,'score':0,'upside_pct':0,
                           'analyst_label':'N/A','summary':'Data unavailable','valid':False}
    return results

def get_best_candidates(pool, owned, top_n=3):
    to_score = tuple(t for t in pool if t.upper() not in owned)
    if not to_score: return []
    scored = [v for v in batch_score_candidates(to_score).values() if v['valid']]
    return sorted(scored, key=lambda x: x['score'], reverse=True)[:top_n]

def format_candidates_html(candidates):
    if not candidates: return "<i style='opacity:.7'>No qualifying candidates.</i>"
    parts = []
    for c in candidates:
        col = "#a5d6a7" if c['upside_pct']>0 else "#ef9a9a"
        parts.append(
            f"<span class='roadmap-ticker' title='{c['summary']}'>{c['ticker']}</span>"
            f"<span style='font-size:11px;color:{col};margin-right:10px;'>"
            f"&nbsp;{c['analyst_label']} · {c['upside_pct']:+.1f}%</span>")
    return "".join(parts)

# ── Investment roadmap ────────────────────────────────────────────────────────
def generate_investment_roadmap(df_results, sector_data, total_value, display_currency):
    sym = "$" if display_currency=="USD" else "€"
    SB = {
        'Technology':             {'target':28,'etf':'XLK', 'risk':'High',
            'candidates':['MSFT','GOOGL','META','ORCL','NVDA','AVGO','AMD','CRM','ADBE','NOW']},
        'Healthcare':             {'target':13,'etf':'XLV', 'risk':'Low',
            'candidates':['UNH','LLY','ABBV','ISRG','TMO','DHR','BSX','ELV','HCA','VRTX']},
        'Financial Services':     {'target':13,'etf':'XLF', 'risk':'Medium',
            'candidates':['JPM','V','MA','BRK-B','GS','MS','AXP','SCHW','CME','ICE']},
        'Consumer Cyclical':      {'target':10,'etf':'XLY', 'risk':'High',
            'candidates':['AMZN','TSLA','NKE','MCD','BKNG','ABNB','CMG','TJX','LULU','ROST']},
        'Industrials':            {'target':8, 'etf':'XLI', 'risk':'Medium',
            'candidates':['CAT','DE','RTX','GE','ETN','EMR','UBER','FDX','LMT','NOC']},
        'Communication Services': {'target':8, 'etf':'XLC', 'risk':'Medium',
            'candidates':['GOOGL','META','NFLX','SPOT','DIS','CHTR','T','VZ','EA','TTWO']},
        'Consumer Defensive':     {'target':6, 'etf':'XLP', 'risk':'Low',
            'candidates':['COST','WMT','PG','KO','PEP','MDLZ','CL','KHC','GIS','SYY']},
        'Energy':                 {'target':4, 'etf':'XLE', 'risk':'High',
            'candidates':['XOM','CVX','SLB','EOG','PXD','MPC','PSX','COP','OXY','HAL']},
        'Utilities':              {'target':2, 'etf':'XLU', 'risk':'Low',
            'candidates':['NEE','DUK','SO','D','AES','PCG','EXC','XEL','AWK','ED']},
        'Real Estate':            {'target':2, 'etf':'XLRE','risk':'Low',
            'candidates':['PLD','AMT','EQIX','CCI','SPG','O','VICI','WY','AVB','EQR']},
        'Basic Materials':        {'target':2, 'etf':'XLB', 'risk':'Medium',
            'candidates':['LIN','APD','FCX','NEM','NUE','ALB','DOW','PPG','VMC','MLM']},
    }
    owned = set(df_results['Ticker'].str.upper().tolist())
    cur_s = {r['Sector']: r['Current Value']/total_value*100 for _,r in sector_data.iterrows()}
    overweight, underweight, missing = [], [], []
    for sector, info in SB.items():
        cur = cur_s.get(sector, 0); gap = cur - info['target']
        if gap>5:  overweight.append((sector, cur, info['target'], gap))
        elif cur==0: missing.append((sector, info['target'], info))
        elif gap<-3: underweight.append((sector, cur, info['target'], abs(gap), info))
    missing.sort(key=lambda x: x[1], reverse=True)
    underweight.sort(key=lambda x: x[3], reverse=True)
    tech_w   = cur_s.get('Technology', 0)
    income_w = sum(cur_s.get(s,0) for s in ['Consumer Defensive','Utilities','Real Estate'])
    conc     = (df_results.nlargest(3,'Current Value')['Current Value'].sum()/total_value*100) if total_value>0 else 0

    # Pre-warm batch cache
    rel_info = [i for _,_,i in missing[:6]] + [i for _,_,_,_,i in underweight[:2]]
    all_cands = list(dict.fromkeys(t for i in rel_info for t in i['candidates'] if t.upper() not in owned))
    p3 = ['NVDA','MSFT','GOOGL','AMD','AVGO','CRM','ADBE','NOW','SNOW','PLTR',
          'O','JNJ','PG','KO','VZ','NEE','VICI','ABBV','MO','T']
    all_cands += [t for t in p3 if t.upper() not in owned and t not in all_cands]
    if all_cands: batch_score_candidates(tuple(all_cands))

    roadmap = []

    # PHASE 1 — with specific share quantities
    p1 = []
    for sector, cur, tgt, gap in overweight[:3]:
        v_shed   = total_value*(gap/100); v_target = total_value*(tgt/100)
        sh       = df_results[df_results['Sector']==sector].copy().sort_values('Allocation %', ascending=False)
        lines, rem = [], v_shed
        for _, row in sh.iterrows():
            if rem<=0: break
            t2s = round(min(float(row['Shares']), rem/float(row['Current Price'])), 4) if float(row['Current Price'])>0 else 0
            sv  = t2s*float(row['Current Price']); pct = (t2s/float(row['Shares'])*100) if float(row['Shares'])>0 else 0
            if t2s>0:
                lines.append(f"<span class='roadmap-ticker'>{row['Ticker']}</span> "
                              f"sell <b>{t2s:,.2f} shares</b> ≈{sym}{sv:,.0f} "
                              f"<span style='font-size:11px;opacity:.8;'>({pct:.0f}% of position)</span>")
            rem -= sv
        detail = ("<br>&nbsp;&nbsp;🎯 Suggested trims: "+" &nbsp;·&nbsp; ".join(lines)) if lines else ""
        p1.append(f"<b>Trim {sector}</b>: currently {cur:.1f}% vs {tgt}% benchmark — "
                  f"sell ≈{sym}{v_shed:,.0f} to reach target {sym}{v_target:,.0f}{detail}")
    if conc>50:
        top3 = df_results.nlargest(3,'Current Value')['Ticker'].tolist()
        p1.append(f"<b>Reduce concentration</b>: top 3 ({', '.join(top3)}) = {conc:.0f}% of portfolio. Target <40%.")
    if not p1: p1.append("Portfolio is reasonably balanced — no urgent rebalancing needed.")
    roadmap.append({'phase':'PHASE 1 — IMMEDIATE (Now → 30 Days)','icon':'⚡','actions':p1})

    # PHASE 2
    p2 = []
    for sector, tpct, info in missing[:3]:
        rb  = 'low' if info['risk']=='Low' else 'med' if info['risk']=='Medium' else 'high'
        p2.append(f"<b>Add {sector}</b> <span class='risk-badge-{rb}'>{info['risk']} Risk</span>: "
                  f"0% → target {tpct}% (≈{sym}{total_value*(tpct/100):,.0f})<br>"
                  f"&nbsp;&nbsp;📡 Live picks: {format_candidates_html(get_best_candidates(info['candidates'],owned,3))} "
                  f"or ETF <span class='roadmap-ticker'>{info['etf']}</span>")
    for sector, cur, tgt, gap, info in underweight[:2]:
        rb  = 'low' if info['risk']=='Low' else 'med' if info['risk']=='Medium' else 'high'
        p2.append(f"<b>Increase {sector}</b> <span class='risk-badge-{rb}'>{info['risk']} Risk</span>: "
                  f"{cur:.1f}% → {tgt}% (add ≈{sym}{total_value*(gap/100):,.0f})<br>"
                  f"&nbsp;&nbsp;📡 Live picks: {format_candidates_html(get_best_candidates(info['candidates'],owned,3))}")
    if not p2: p2.append("Good sector coverage — focus on quality within existing positions.")
    roadmap.append({'phase':'PHASE 2 — CORE BUILD (1–3 Months)','icon':'🏗️','actions':p2})

    # PHASE 3
    p3a = []
    if tech_w<15:
        p3a.append(f"<b>Growth Layer — AI/Tech</b>: Only {tech_w:.1f}% tech. "
                   f"Top picks:<br>&nbsp;&nbsp;📡 {format_candidates_html(get_best_candidates(['NVDA','MSFT','GOOGL','AMD','AVGO','CRM','ADBE','NOW','SNOW','PLTR'],owned,4))}")
    if income_w<8:
        p3a.append(f"<b>Income Layer</b>: Only {income_w:.1f}% defensive/income. "
                   f"Top payers:<br>&nbsp;&nbsp;📡 {format_candidates_html(get_best_candidates(['O','JNJ','PG','KO','VZ','NEE','VICI','ABBV','MO','T'],owned,4))}")
    p3a.append("<b>International</b>: <span class='roadmap-ticker'>VEA</span> Developed · "
               "<span class='roadmap-ticker'>VWO</span> Emerging · <span class='roadmap-ticker'>EWJ</span> Japan")
    p3a.append("<b>Bonds / Hedge</b>: <span class='roadmap-ticker'>BND</span> Total Bond · "
               "<span class='roadmap-ticker'>GLD</span> Gold · <span class='roadmap-ticker'>TIP</span> TIPS")
    roadmap.append({'phase':'PHASE 3 — GROWTH & INCOME (3–6 Months)','icon':'📈','actions':p3a})

    # PHASE 4
    p4 = []
    for sector, tpct, info in missing[3:6]:
        p4.append(f"<b>Complete {sector}</b> (target {tpct}%):<br>"
                  f"&nbsp;&nbsp;📡 {format_candidates_html(get_best_candidates(info['candidates'],owned,2))} "
                  f"or ETF <span class='roadmap-ticker'>{info['etf']}</span>")
    p4.append("<b>Rebalance Annually</b>: Review each January. Trim >+5% overweight, add to laggards.")
    p4.append("<b>Tax-Loss Harvesting</b>: Before year-end review unrealised losses to offset gains.")
    p4.append("<b>Dollar-Cost Average</b>: Deploy new capital in 3 tranches over 6–8 weeks.")
    roadmap.append({'phase':'PHASE 4 — LONG-TERM (6–12 Months)','icon':'🎯','actions':p4})

    return roadmap, overweight, underweight, missing

def render_investment_roadmap(roadmap, overweight, underweight, missing, sym):
    st.markdown("---")
    st.subheader("🗺️ Personalised Investment Roadmap")
    st.caption("Phased action plan vs S&P 500 benchmark. Picks scored live: analyst targets, EPS/revenue growth & 52w momentum.")
    ca,cb,cc = st.columns(3)
    ca.metric("Sectors Covered",    f"{11-len(missing)}/11", f"-{len(missing)} missing")
    cb.metric("Overweight Sectors", str(len(overweight)),    "Need trimming" if overweight else "✅ OK")
    cc.metric("Underweight Sectors",str(len(underweight)),   "Need building" if underweight else "✅ OK")
    for pd_ in roadmap:
        st.markdown(f"""
<div class="roadmap-card">
  <h4 style="margin:0 0 12px 0;font-size:16px;">{pd_['icon']} {pd_['phase']}</h4>
  {''.join(f'<div class="roadmap-phase"><p style="margin:0;line-height:1.9;font-size:14px;">{a}</p></div>' for a in pd_['actions'])}
</div>""", unsafe_allow_html=True)
    if missing or underweight:
        st.markdown("#### 📊 Sector Gap Analysis")
        rows = []
        for s,tp,i in missing:
            rows.append({'Sector':s,'You Have':'0%','S&P Target':f'{tp}%','Gap':f'-{tp}%','Risk':i['risk'],'ETF':i['etf']})
        for s,c,t,g,i in underweight:
            rows.append({'Sector':s,'You Have':f'{c:.1f}%','S&P Target':f'{t}%','Gap':f'-{g:.1f}%','Risk':i['risk'],'ETF':i['etf']})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True)

def suggest_diversification(cur_s, total_value):
    ex = {'Technology':['AAPL','MSFT','QQQ'],'Healthcare':['JNJ','UNH','XLV'],
          'Financial Services':['JPM','BAC','XLF'],'Consumer Cyclical':['AMZN','TSLA','XLY'],
          'Industrials':['HON','CAT','XLI'],'Energy':['XOM','CVX','XLE'],
          'Utilities':['NEE','DUK','XLU'],'Real Estate':['PLD','AMT','XLRE'],
          'Communication Services':['META','GOOGL','XLC'],'Consumer Defensive':['PG','KO','XLP'],
          'Basic Materials':['LIN','BHP','XLB']}
    tips = []
    for s,e in ex.items():
        if s not in cur_s:
            tips.append(f"💡 No exposure to **{s}**. Consider: {', '.join(e[:3])}")
        elif cur_s[s]<10:
            tips.append(f"💡 Low {s} ({cur_s[s]:.1f}%). Consider increasing via {', '.join(e[:2])}")
    return tips[:5]

# ── File processing ───────────────────────────────────────────────────────────
def process_uploaded_file(uploaded_file):
    ext = Path(uploaded_file.name.lower()).suffix
    if ext in ['.csv','.xlsx','.xls']:
        try:
            df = pd.read_csv(uploaded_file) if ext=='.csv' else pd.read_excel(uploaded_file)
            orig = list(df.columns)
            df.columns = df.columns.astype(str).str.strip().str.lower()
            CM = {
                'ticker':['ticker','symbol','stock','scrip','security','code','asset','instrument','equity'],
                'shares':['shares','qty','quantity','units','amount','no of shares','holding','holdings','position','lots'],
                'purchase price':['purchase price','avg price','average price','avg cost','average cost',
                                  'cost price','buy price','cost basis','price paid','cost','bought at','acquisition price'],
            }
            rmap, det = {}, {}
            for canon, variants in CM.items():
                for col in df.columns:
                    if col.strip() in variants or any(v in col for v in variants):
                        rmap[col]=canon; det[canon]=col; break
            df.rename(columns=rmap, inplace=True)
            if 'ticker' not in df.columns:
                st.error(f"No ticker column found. Columns detected: {', '.join(orig)}")
                return None
            if det: st.success("✅ Auto-detected: "+" | ".join(f"**{v}**→{k}" for k,v in det.items()))
            if 'shares' not in df.columns:
                st.warning("No qty column found — defaulting to 1 share each."); df['shares']=1
            else:
                df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(1)
            if 'purchase price' in df.columns:
                df['purchase price'] = pd.to_numeric(df['purchase price'], errors='coerce')
            df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
            df = df[df['ticker'].str.len()<=6]
            df = df[df['ticker']!='NAN'].reset_index(drop=True)
            return {'type':'tabular','data':df}
        except Exception as e:
            show_error(f"File read error: {e}", traceback.format_exc()); return None
    elif ext in ['.pdf','.docx','.doc','.png','.jpg','.jpeg','.tiff','.bmp']:
        if ext in ['.pdf','.docx','.doc'] and not PDF_DOCX_AVAILABLE:
            st.error("PDF/Word support not installed. Run: pip install pypdf python-docx reticker"); return None
        ex = PortfolioExtractor()
        tmp = f"tmp_{uploaded_file.name}"
        try:
            with open(tmp,'wb') as f: f.write(uploaded_file.getbuffer())
            holdings, preview, msg = ex.process_file(tmp)
        finally:
            if os.path.exists(tmp): os.remove(tmp)
        with st.expander("📄 Extraction details"): st.text(msg); st.code(preview)
        if holdings: return {'type':'tabular','data':pd.DataFrame(holdings)}
        st.warning("No valid tickers found in document."); return None
    else:
        st.error("Unsupported file type."); return None

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Configuration")
user_query       = st.sidebar.text_input("Ticker / Symbol", value="NVDA")
display_currency = st.sidebar.selectbox("Currency", ["USD","EUR"])
total_capital    = st.sidebar.number_input("Capital", value=10000)
run_audit        = st.sidebar.button("🚀 Run Deep Audit")
st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Portfolio / Document")
uploaded_file    = st.sidebar.file_uploader(
    "CSV, Excel, PDF, Word, or Image",
    type=['csv','xlsx','xls','pdf','docx','doc','png','jpg','jpeg','tiff','bmp'])

# ════════════════════════════════════════════════════════════════════════════
# DEEP AUDIT
# ════════════════════════════════════════════════════════════════════════════
if run_audit:
    with st.spinner(f"Analysing {user_query}..."):
        ticker, name, suffix, native_curr, _ = resolve_smart_ticker(user_query)
        try:
            i2 = yf.Ticker(ticker).info
            display_name = i2.get('longName') or i2.get('shortName') or name
        except: display_name = name

        df, health, hist_err = get_fundamental_health(ticker, suffix)
        if hist_err: st.error(f"Historical data error: {hist_err}")

        if df is not None:
            fx    = get_exchange_rate(native_curr, display_currency)
            sym   = "$" if display_currency=="USD" else "€"
            cur_p = df['y'].iloc[-1]*fx
            df['50_Day_Moving_Average']  = df['y'].rolling(50).mean()
            df['200_Day_Moving_Average'] = df['y'].rolling(200).mean()

            forecast = None
            try:
                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
                            changepoint_prior_scale=0.08, changepoint_range=0.9, seasonality_mode='additive')
                m.fit(df[['ds','y']])
                future   = m.make_future_dataframe(periods=180)
                forecast = m.predict(future)
                hmin, hmax = df['y'].min(), df['y'].max()
                for col in ['yhat','yhat_lower','yhat_upper']:
                    forecast[col] = forecast[col].clip(hmin*0.5, hmax*2)
                tv   = forecast[forecast['ds']==df['ds'].iloc[-1]]['yhat'].values[0]*fx
                dev  = (cur_p-tv)/tv*100
                fl, fh = tv*0.95, tv*1.05
            except Exception as e:
                st.warning(f"Forecast failed: {e}")
                tv=cur_p; dev=0; fl=cur_p*0.95; fh=cur_p*1.05; forecast=None

            is_dc = df['50_Day_Moving_Average'].iloc[-1] < df['200_Day_Moving_Average'].iloc[-1]
            cross_msg, cross_pt = "Price stability detected.", None
            for i in range(len(df)-60, len(df)):
                p = i-1
                ma50  = df['50_Day_Moving_Average']
                ma200 = df['200_Day_Moving_Average']
                if ma50.iloc[p]<ma200.iloc[p] and ma50.iloc[i]>=ma200.iloc[i]:
                    cross_pt  = (df['ds'].iloc[i], ma50.iloc[i], "GOLDEN")
                    cross_msg = "🚀 GOLDEN CROSS: 50-Day MA crossed ABOVE 200-Day."
                elif ma50.iloc[p]>ma200.iloc[p] and ma50.iloc[i]<=ma200.iloc[i]:
                    cross_pt  = (df['ds'].iloc[i], ma50.iloc[i], "DEATH")
                    cross_msg = "⚠️ DEATH CROSS: 50-Day MA crossed BELOW 200-Day."

            all_news   = get_enhanced_news(ticker)
            target_7d  = compute_7day_target(df, forecast, all_news, is_dc, fx) if forecast is not None else \
                         cur_p*max(0.95,min(1.05,1+float(df['y'].pct_change().tail(5).mean())*0.5+(-0.02 if is_dc else 0)))
            roi_7d     = (target_7d-cur_p)/cur_p*100
            rsi, fibs  = calculate_technicals(df)
            imp_news   = filter_impactful_news(all_news)
            avg_sent   = float(np.mean([n['sentiment'] for n in imp_news])) if imp_news else 0

            score = 15
            if not is_dc:      score += 20
            if health['ROE']>0.12: score += 20
            if roi_7d>0.5:     score += 30
            if avg_sent>0:     score += 15
            score = max(0, min(100, score))

            if score>=70:   verdict,vcol,action,pct = "Strong Buy",   "v-green", "ACTION: BUY NOW",         25
            elif score>=35: verdict,vcol,action,pct = "Hold/Neutral", "v-orange","ACTION: MONITOR",          10
            else:           verdict,vcol,action,pct = "Sell / Avoid", "v-red",   "ACTION: SELL / STAY AWAY", 0

            sl = cur_p*(0.95 if rsi>70 else 0.85 if rsi<30 else 0.90)

            st.subheader(f"📊 {display_name} ({ticker})")
            c1,c2 = st.columns(2)
            with c1:
                (st.warning if dev>15 else st.success)(
                    f"{'⚠️ MEAN REVERSION RISK' if dev>15 else '✅ TREND ALIGNMENT'}: "
                    f"Price is {dev:.1f}% {'above' if dev>0 else 'below'} AI baseline.")
            with c2:
                st.info(f"💎 AI FAIR VALUE: {sym}{fl:,.2f} – {sym}{fh:,.2f}")

            st.markdown(f'<div class="impact-announcement">{cross_msg}</div>', unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Conviction Score",       f"{score}/100")
            m2.metric("Current Price",          f"{sym}{cur_p:,.2f}")
            m3.metric("7d AI Growth (News-Adj)",f"{roi_7d:.2f}%")
            m4.metric("7d AI Target",           f"{sym}{target_7d:,.2f}")

            st.markdown(f'<div class="verdict-box {vcol}">{verdict} | {action}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stop-loss-box">🛑 STOP LOSS: {sym}{sl:,.2f}</div>', unsafe_allow_html=True)

            cl,cr = st.columns(2)
            with cl:
                st.markdown("### 🏥 Company Health")
                st.table(pd.DataFrame({
                    "Metric": ["ROE","P/B Ratio","Debt/Equity","Current Ratio"],
                    "Value":  [f"{health['ROE']*100:.1f}%",f"{health['PB']}x",health['Debt'],health['CurrentRatio']],
                    "Rating": ["✅ Prime" if health['ROE']>0.15 else "⚠️ Weak",
                               "✅ OK" if health['PB']<3 else "⚠️ Pricey","✅ Safe","✅ Liquid"]
                }))
                st.markdown("### 📰 Headlines")
                for n in all_news[:5]:
                    e = '🟢' if n['sentiment']>0 else '🔴' if n['sentiment']<0 else '⚪'
                    st.markdown(f'<div class="news-card">{e} {n["headline"]}</div>', unsafe_allow_html=True)
            with cr:
                st.markdown("### ⚖️ Strategy & Fibonacci")
                st.markdown(f"""<div class="phase-card">
                    <h4 style="color:#1f77b4">PHASE 1: INVEST TODAY</h4>
                    <p><b>{sym}{total_capital*(pct/100):,.2f}</b> ({pct}% of capital)</p><hr>
                    <h4 style="color:#1f77b4">PHASE 2: STAGED ENTRY (FIBONACCI)</h4>
                    <div class="fib-box">🔹 0.382: {sym}{fibs['0.382']*fx:,.2f}</div>
                    <div class="fib-box">🔹 0.500: {sym}{fibs['0.500']*fx:,.2f}</div>
                    <div class="fib-box">🔹 0.618: {sym}{fibs['0.618']*fx:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            if imp_news:
                st.markdown("### 🔥 High-Impact News")
                for n in imp_news[:7]:
                    e = '🟢' if n['sentiment']>0.1 else '🔴' if n['sentiment']<-0.1 else '⚪'
                    st.markdown(f"""<div class="impact-news">
                        <b>{e} {n['headline']}</b><br>
                        <span style="font-size:.85rem;">{n['date'].strftime('%b %d, %Y')} · {n['source']} · Sentiment: {n['sentiment']:.2f}</span>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🤖 AI 180-Day Projection")
            fig, ax = plt.subplots(figsize=(12,6))
            try:
                fp = forecast.copy()
                fp[['yhat','yhat_lower','yhat_upper']] *= fx
                m.plot(fp, ax=ax)
            except: ax.plot(df['ds'], df['y']*fx, 'b-', label='Historical')
            ax.plot(df['ds'], df['50_Day_Moving_Average']*fx,  label='50-Day MA',  color='orange', linewidth=2)
            ax.plot(df['ds'], df['200_Day_Moving_Average']*fx, label='200-Day MA', color='red',    linewidth=2)
            if cross_pt:
                ax.scatter(cross_pt[0], cross_pt[1]*fx, color='gold', s=300, marker='*',
                           label=f"{cross_pt[2]} CROSS", zorder=5)
            ax.set_xlim([datetime.datetime.now()-datetime.timedelta(days=180),
                         datetime.datetime.now()+datetime.timedelta(days=180)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.legend(loc='upper left'); st.pyplot(fig)

            st.markdown("---")
            st.subheader("📊 12-Month Volume Trend")
            vdf = df.tail(252).copy()
            vcol2 = ['#2e7d32' if i>0 and vdf.iloc[i]['y']>=vdf.iloc[i-1]['y'] else '#c62828' for i in range(len(vdf))]
            vfig, vax = plt.subplots(figsize=(12,4))
            vax.bar(vdf['ds'], vdf['vol'], color=vcol2, alpha=0.7)
            vax.set_ylabel("Shares Traded")
            vax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            st.pyplot(vfig)
            st.caption("Green = price up vs prior day · Red = price down")
            st.info(f"📈 {volume_trend_message(vdf)}")
        else:
            st.error(f"Could not fetch data for {user_query}.")

# ════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    st.markdown("---")
    st.header("📁 Portfolio Analysis")
    with st.spinner("Processing file..."):
        result = process_uploaded_file(uploaded_file)

    if result and result['type']=='tabular':
        df_h = result['data']
        rlist, elist, tv, tc = [], [], 0, 0
        prog = st.progress(0); stxt = st.empty()

        for idx, row in df_h.iterrows():
            t  = str(row['ticker']).strip().upper()
            sh = float(row.get('shares', 1))
            pp = row.get('purchase price', None)
            stxt.text(f"Analysing {t}... ({idx+1}/{len(df_h)})")
            prog.progress((idx+1)/len(df_h))
            an, err = analyze_ticker_basic(t, display_currency)
            if an:
                val = sh*an['price']; tv += val
                gain = gain_pct = None
                if pd.notna(pp):
                    cost = sh*float(pp); tc += cost
                    gain = val-cost; gain_pct = gain/cost*100 if cost else 0
                rlist.append({'Ticker':t,'Name':an['name'],'Sector':an['sector'],
                              'Industry':an.get('industry',''),'Shares':sh,
                              'Current Price':an['price'],'Current Value':val,
                              'Purchase Price':float(pp) if pd.notna(pp) else None,
                              'Gain/Loss $':gain,'Gain/Loss %':gain_pct})
            else: elist.append(f"{t}: {err}")

        prog.empty(); stxt.empty()
        if elist:
            with st.expander("⚠️ Errors"):
                for e in elist: st.write(e)

        if rlist:
            dfr = pd.DataFrame(rlist)
            if tv>0: dfr['Allocation %'] = dfr['Current Value']/tv*100
            sym = "$" if display_currency=="USD" else "€"

            c1,c2,c3 = st.columns(3)
            c1.metric("Total Value", f"{sym}{tv:,.2f}")
            if tc>0:
                tg = tv-tc
                c2.metric("Total Cost", f"{sym}{tc:,.2f}")
                c3.metric("Total Gain/Loss", f"{sym}{tg:,.2f} ({tg/tc*100:.1f}%)")

            st.subheader("Holdings")
            dfd = dfr.copy()
            dfd['Current Price'] = dfd['Current Price'].map(lambda x: f"{sym}{x:,.2f}")
            dfd['Current Value'] = dfd['Current Value'].map(lambda x: f"{sym}{x:,.2f}")
            if 'Purchase Price' in dfd.columns:
                dfd['Purchase Price'] = dfd['Purchase Price'].map(lambda x: f"{sym}{x:,.2f}" if pd.notna(x) else "N/A")
            if 'Gain/Loss $' in dfd.columns:
                dfd['Gain/Loss $'] = dfd['Gain/Loss $'].map(lambda x: f"{sym}{x:,.2f}" if pd.notna(x) else "N/A")
                dfd['Gain/Loss %'] = dfd['Gain/Loss %'].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
            dfd['Allocation %'] = dfd['Allocation %'].map(lambda x: f"{x:.1f}%")
            cols = ['Ticker','Name','Sector','Industry','Shares','Current Price',
                    'Current Value','Purchase Price','Gain/Loss $','Gain/Loss %','Allocation %']
            st.dataframe(dfd[[c for c in cols if c in dfd.columns]])

            st.subheader("Sector Allocation")
            sd = dfr.groupby('Sector')['Current Value'].sum().reset_index()
            fig2, ax2 = plt.subplots()
            ax2.pie(sd['Current Value'], labels=sd['Sector'], autopct='%1.1f%%')
            ax2.axis('equal'); st.pyplot(fig2)

            tips = suggest_diversification(
                dict(zip(sd['Sector'], sd['Current Value']/tv*100)), tv)
            if tips:
                st.subheader("💡 Quick Tips")
                for t2 in tips: st.info(t2)

            with st.spinner("📡 Scoring live market data for all sectors..."):
                roadmap, ow, uw, ms = generate_investment_roadmap(dfr, sd, tv, display_currency)
            render_investment_roadmap(roadmap, ow, uw, ms, sym)
        else:
            st.error("No valid tickers could be analysed.")
    else:
        st.error("Failed to process the file.")
