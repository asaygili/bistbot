"""
BIST Hisse Analiz Robotu v5.0
─────────────────────────────
Mimari: Flask REST API + 7 Katmanlı Skor Motoru
Modüller:
  1. Teknik Analiz   (RSI, MACD, Bollinger, ATR, OBV, CCI, Stoch, Williams, ROC)
  2. Temel Analiz    (F/K, PD/DD, temettü, büyüme — Yahoo Finance)
  3. Duygu Analizi   (Google News + kelime tabanlı skor)
  4. KAP Bildirimleri (kap.org.tr proxy)
  5. Makro Analiz    (USD/TRY, Altın, BIST100, Petrol)
  6. Analist Konsensüsü (Yahoo Finance analyst data)
  7. ML Ensemble     (RF + GB + ExtraTrees + LogReg meta — 10 günlük hedef ±%1.5)

Ek:
  - Kelly Criterion  (pozisyon büyüklüğü hesabı)
  - ARIMA trend      (basit trend projeksiyonu)
  - Walk-forward backtest
  - KAP async proxy endpoint
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, sys, time, json, math, hashlib, threading, warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.parse import quote
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import joblib

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample

import yfinance as yf

warnings.filterwarnings("ignore")

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r'/api/*': {'origins': '*'}})

# ── Paths & Constants ─────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MODEL_DIR      = BASE_DIR / "bist_modeller"
MODEL_PATH     = MODEL_DIR / "evrensel_model.joblib"
PORTFOLYO_FILE = BASE_DIR / "portfolyo.json"
SINYAL_FILE    = BASE_DIR / "sinyal_gecmisi.json"
MODEL_DIR.mkdir(exist_ok=True)

CACHE_SURE     = 180   # saniye
MAKRO_SURE     = 1800  # 30 dk
KAP_SURE       = 1800  # 30 dk

# ML Hedef: 10 günlük ±%1.5
ML_HEDEF_GUN   = 10
ML_ESIK        = 0.015

# Birleşik skor ağırlıkları (toplam = 1.0)
W = {
    "teknik":   0.30,
    "ml":       0.30,
    "duygu":    0.10,
    "kap":      0.10,
    "temel":    0.10,
    "analist":  0.05,
    "arima":    0.05,
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Fiyat düzeltme çarpanları ─────────────────────────────────────────────────
FIYAT_DUZELTME = {
    "THYAO": 1, "TTKOM": 1, "TCELL": 1, "AKBNK": 1, "GARAN": 1,
    "ISCTR": 1, "YKBNK": 1, "HALKB": 1, "VAKBN": 1, "SAHOL": 1,
    "KCHOL": 1, "SISE":  1, "EREGL": 1, "PETKM": 1, "TUPRS": 1,
    "FROTO": 1, "TOASO": 1, "BIMAS": 1, "MGROS": 1, "ARCLK": 1,
    "ASELS": 1, "EKGYO": 1, "ENKAI": 1, "TKFEN": 1, "PGSUS": 1,
    "KOZAL": 1, "SASA":  1, "KRDMD": 1, "VESTL": 1, "SOKM":  1,
}
VARSAYILAN = [
    "AKBNK","ARCLK","ASELS","BIMAS","EKGYO","EREGL","FROTO","GARAN",
    "HALKB","ISCTR","KCHOL","KOZAL","KRDMD","MGROS","PETKM","PGSUS",
    "SAHOL","SASA","SISE","SOKM","TCELL","THYAO","TKFEN","TOASO",
    "TTKOM","TUPRS","VAKBN","VESTL","YKBNK","AKSA","ALFEN","ENKAI",
    "GUBRF","LOGO","MAVI","BMSCH","BMSTL","MEKAG","ALTINS1",
]

# BIST100 ek hisseler (BIST50'de olmayan)
BIST100_EK = [
    "AEFES","AFYON","AGESA","AGHOL","AGYO","AHGAZ","AKFEN","AKMGY","AKSEN",
    "ALBRK","ALFAS","ALTNY","ANSGR","APORT","ARKAS","ARMDA","ASGYO","ASTOR",
    "AYDEM","AYEN","BAGFS","BASGZ","BERA","BFREN","BIGCH","BIOEN","BIZIM",
    "BNTAS","BORDA","BRISA","BRYAT","BUCIM","BURCE","CANTE","CCOLA","CELHA",
    "CEMAS","CEMTS","CIMSA","CLEBI","COKAL","CRDFA","CWENE","DEVA","DGNMO",
    "DOAS","DOBUR","DOGUB","DOHOL","DORI","DURDO","DYOBY","DZGYO","ECILC",
    "EGEEN","EGGUB","EGPRO","EGSER","EMKEL","EMNIS","ENKAI","ENSRI","ERBOS",
    "ERSU","ESCAR","ESCOM","ETILR","ETYAT","EUPWR","EUREN","EYGYO","FADE",
    "FENER","FLAP","FMIZP","FONET","FORMT","FORTE","FTURZ","GARAN","GARFA",
    "GEDZA","GEREL","GLBMD","GLRYH","GMTAS","GOLTS","GOODY","GORBON","GSDDE",
    "GSDHO","GSRAY","GUBRF","GWIND","HALKB","HATEK","HDFGS","HEDEF","HEKTS",
    "HLGYO","HOROZ","HTTBT","HUNER","ICBCT","IDEAS","IEYHO","IHEVA","IHGZT",
    "IHLAS","IHLGM","IHYAY","IMASM","INDES","INFO","INTEM","INVEO","IPEKE",
    "ISATR","ISDMR","ISGSY","ISGYO","ISKPL","ISKUR","ISMEN","ISYAT","IZFAS",
    "IZINV","IZMDC","JANTS","KAPLM","KARSN","KATMR","KAYSE","KCAER","KENT",
    "KERVT","KFEIN","KGYO","KLGYO","KLKIM","KLMSN","KLNMA","KLRHO","KMPUR",
    "KNFRT","KONKA","KONTR","KONYA","KOPOL","KORDS","KRSTL","KRTEK","KRVGD",
    "KTLEV","KUTPO","KUYAS","LIDER","LIDFA","LKMNH","LOGO","LRSHO","LYKHO",
    "MAALT","MACKO","MAGEN","MAKIM","MAKTK","MANAS","MAVI","MEDTR","MEGAP",
    "MEGES","MEKAG","MERIT","MERKO","METRO","METUR","MIATK","MIGROS","MIPAZ",
    "MMCAS","MNDRS","MNVRL","MOBTL","MOGAN","MPARK","MRDIN","MSDOS","MTRKS",
    "NATEN","NETAS","NIBAS","NILYT","NTHOL","NTTUR","NUGYO","NUHCM","OBASE",
    "ODAS","OFISM","OLMIP","ONCSM","ORCAY","ORGE","ORMA","OSMKB","OSTIM",
    "OTKAR","OYAKC","OYLUM","OZGYO","OZKGY","PAPIL","PAREG","PEKGY","PENGD",
    "PENTA","PETUN","PINSU","PKART","PKENT","PLTUR","POLHO","POLTK","PRDGS",
    "PRZMA","PSDTC","PSGYO","QNBFB","QNBFL","RALYH","RAYSG","RGYAS","RHEAG",
    "RNPOL","RODRG","RTALB","RUBNS","RYSAS","SAFKR","SANEL","SANFM","SANKO",
    "SAYAS","SDTTR","SEGMN","SEKFK","SEKUR","SELEC","SELVA","SENTE","SEYKM",
    "SILVR","SNGYO","SNKRN","SNPAM","SOKE","SONME","SRVGY","SUMAS","SUNTK",
    "SUWEN","TABGD","TATGD","TDGYO","TEKTU","TETMT","TEZOL","TGSAS","TIRE",
    "TKFEN","TLMAN","TMPOL","TMSN","TNZTP","TOASO","TRCAS","TRGYO","TRILC",
    "TSPOR","TTRAK","TUCLK","TUKAS","TUMAS","TUREX","TURGG","TURSG","TUYAP",
    "ULUUN","ULUSE","UMPAS","UNLU","USAK","USDAP","UTPYA","UZERB","VAKFN",
    "VANGD","VBTS","VERTU","VERUS","VKGYO","VKFYO","YAPRK","YATAS","YBTAS",
    "YEOTK","YESIL","YGYO","YIGIT","YKSLN","YUNSA","ZEDUR","ZOREN","ZRGYO",
]

# Model eğitimi için kullanılacak tüm hisseler

BIST30 = [
    "AKBNK","ARCLK","ASELS","BIMAS","EKGYO","EREGL","FROTO","GARAN",
    "HALKB","ISCTR","KCHOL","KRDMD","KOZAL","MGROS","PETKM","PGSUS",
    "SAHOL","SASA","SISE","SOKM","TCELL","THYAO","TKFEN","TOASO",
    "TTKOM","TUPRS","VAKBN","VESTL","YKBNK","ENKAI",
]

BIST50 = [
    "AKBNK","ARCLK","ASELS","BIMAS","EKGYO","EREGL","FROTO","GARAN",
    "HALKB","ISCTR","KCHOL","KOZAA","KOZAL","KRDMD","MGROS","PETKM",
    "PGSUS","SAHOL","SASA","SISE","SOKM","TAVHL","TCELL","THYAO",
    "TKFEN","TOASO","TTKOM","TUPRS","VAKBN","VESTL","YKBNK","AKSA",
    "ALARK","ALFEN","ANHYT","AYGAZ","BAGFS","BRISA","CIMSA","DOAS",
    "EGEEN","ENKAI","GESAN","GUBRF","HEKTS","ISDMR","ISGYO","LOGO",
    "MAVI","NETAS",
]

BIST50 = [
    "AKBNK","ARCLK","ASELS","BIMAS","EKGYO","EREGL","FROTO","GARAN",
    "HALKB","ISCTR","KCHOL","KOZAA","KOZAL","KRDMD","MGROS","PETKM",
    "PGSUS","SAHOL","SASA","SISE","SOKM","TAVHL","TCELL","THYAO",
    "TKFEN","TOASO","TTKOM","TUPRS","VAKBN","VESTL","YKBNK","AKSA",
    "ALARK","ALFEN","ANHYT","AYGAZ","BAGFS","BRISA","CIMSA","DOAS",
    "EGEEN","ENKAI","GESAN","GUBRF","HEKTS","ISDMR","ISGYO","LOGO",
    "MAVI","NETAS",
]

BIST100_EK = [
    "AEFES","AFYON","AGESA","AGHOL","AGYO","AHGAZ","AKFEN","AKMGY","AKSEN",
    "ALBRK","ALFAS","ALTNY","ANSGR","APORT","ARKAS","ARMDA","ASGYO","ASTOR",
    "AYDEM","AYEN","BAGFS","BASGZ","BERA","BFREN","BIGCH","BIOEN","BIZIM",
    "BNTAS","BORDA","BRISA","BRYAT","BUCIM","BURCE","CANTE","CCOLA","CELHA",
    "CEMAS","CEMTS","CIMSA","CLEBI","COKAL","CRDFA","CWENE","DEVA","DGNMO",
    "DOAS","DOBUR","DOGUB","DOHOL","DORI","DURDO","DYOBY","DZGYO","ECILC",
    "EGEEN","EGGUB","EGPRO","EGSER","EMKEL","EMNIS","ENKAI","ENSRI","ERBOS",
    "ERSU","ESCAR","ESCOM","ETILR","ETYAT","EUPWR","EUREN","EYGYO","FADE",
    "FENER","FLAP","FMIZP","FONET","FORMT","FORTE","FTURZ","GARAN","GARFA",
    "GEDZA","GEREL","GLBMD","GLRYH","GMTAS","GOLTS","GOODY","GORBON","GSDDE",
    "GSDHO","GSRAY","GUBRF","GWIND","HALKB","HATEK","HDFGS","HEDEF","HEKTS",
    "HLGYO","HOROZ","HTTBT","HUNER","ICBCT","IDEAS","IEYHO","IHEVA","IHGZT",
    "IHLAS","IHLGM","IHYAY","IMASM","INDES","INFO","INTEM","INVEO","IPEKE",
    "ISATR","ISDMR","ISGSY","ISGYO","ISKPL","ISKUR","ISMEN","ISYAT","IZFAS",
    "IZINV","IZMDC","JANTS","KAPLM","KARSN","KATMR","KAYSE","KCAER","KENT",
    "KERVT","KFEIN","KGYO","KLGYO","KLKIM","KLMSN","KLNMA","KLRHO","KMPUR",
    "KNFRT","KONKA","KONTR","KONYA","KOPOL","KORDS","KRSTL","KRTEK","KRVGD",
    "KTLEV","KUTPO","KUYAS","LIDER","LIDFA","LKMNH","LOGO","LRSHO","LYKHO",
    "MAALT","MACKO","MAGEN","MAKIM","MAKTK","MANAS","MAVI","MEDTR","MEGAP",
    "MEGES","MEKAG","MERIT","MERKO","METRO","METUR","MIATK","MIGROS","MIPAZ",
    "MMCAS","MNDRS","MNVRL","MOBTL","MOGAN","MPARK","MRDIN","MSDOS","MTRKS",
    "NATEN","NETAS","NIBAS","NILYT","NTHOL","NTTUR","NUGYO","NUHCM","OBASE",
    "ODAS","OFISM","OLMIP","ONCSM","ORCAY","ORGE","ORMA","OSMKB","OSTIM",
    "OTKAR","OYAKC","OYLUM","OZGYO","OZKGY","PAPIL","PAREG","PEKGY","PENGD",
    "PENTA","PETUN","PINSU","PKART","PKENT","PLTUR","POLHO","POLTK","PRDGS",
    "PRZMA","PSDTC","PSGYO","QNBFB","QNBFL","RALYH","RAYSG","RGYAS","RHEAG",
    "RNPOL","RODRG","RTALB","RUBNS","RYSAS","SAFKR","SANEL","SANFM","SANKO",
    "SAYAS","SDTTR","SEGMN","SEKFK","SEKUR","SELEC","SELVA","SENTE","SEYKM",
    "SILVR","SNGYO","SNKRN","SNPAM","SOKE","SONME","SRVGY","SUMAS","SUNTK",
    "SUWEN","TABGD","TATGD","TDGYO","TEKTU","TETMT","TEZOL","TGSAS","TIRE",
    "TKFEN","TLMAN","TMPOL","TMSN","TNZTP","TOASO","TRCAS","TRGYO","TRILC",
    "TSPOR","TTRAK","TUCLK","TUKAS","TUMAS","TUREX","TURGG","TURSG","TUYAP",
    "ULUUN","ULUSE","UMPAS","UNLU","USAK","USDAP","UTPYA","UZERB","VAKFN",
    "VANGD","VBTS","VERTU","VERUS","VKGYO","VKFYO","YAPRK","YATAS","YBTAS",
    "YEOTK","YESIL","YGYO","YIGIT","YKSLN","YUNSA","ZEDUR","ZOREN","ZRGYO",
]

# Model eğitimi için kullanılacak tüm hisseler

YILDIZ_PAZAR = list(dict.fromkeys([
    "ACSEL","ADEL","ADESE","ADGYO","AKENR","AKGRT","AKMGY","AKSGY","ALCAR","ALGYO","ALKA","ALVES","ALYAG","ANGEN","APOLI","ARENA","ARSAN","ARTMS","ARZUM","ASGYO","ASLAN","ATEKS","ATSYH","AVHOL","AVOD","AYCES","AYES","AYGAZ","BABYO","BAKAB","BALAT","BANVT","BAYRK","BEGYO","BERA","BEYAZ","BFREN","BIGCH","BIOEN","BINHO","BITTM","BMELK","BMSCH","BMSTL","BNTAS","BORLS","BOSSA","BRKO","BSOKE","BUCIM","BURCE","BURVA","BVSAN","CASA","CEOEM","CMBTN","CMENT","CONSE","COSMO","CRDFA","CRFSA","CUSAN","DAGHL","DAPGM","DATA","DENGE","DERHL","DERIM","DESPC","DEVA","DGATE","DITAS","DMRGD","DNISI","DOBUR","DOCO","DOFER","DOGUB","DOHOL","DORE","DURDO","DYOBY","EDATA","EDIP","EFORC","EGPRO","EGGUB","EGSER","EKIZ","EKOS","ELITE","EMKEL","EMNIS","ENERY","ENSRI","EPLAS","ERBOS","ERSU","ESCAR","ESCOM","ESGYO","ETILR","ETYAT","EUREN","EYGYO","FADE","FENER","FLAP","FMIZP","FONET","FORMT","FORTE","FTURZ","GARFA","GEDZA","GEREL","GESAN","GLBMD","GLRYH","GLYHO","GMTAS","GOLTS","GOODY","GORBON","GSDDE","GSDHO","GSRAY","GWIND","HATEK","HDFGS","HLGYO","HOROZ","HTTBT","HUNER","ICBCT","IDEAS","IEYHO","IHEVA","IHGZT","IHLAS","IHLGM","IHYAY","IMASM","INFO","INTEM","INVEO","IPEKE","ISATR","ISGSY","ISKPL","ISKUR","ISMEN","ISYAT","IZFAS","IZINV","JANTS","KAPLM","KARSN","KATMR","KAYSE","KCAER","KENT","KERVT","KFEIN","KGYO","KLGYO","KLKIM","KLMSN","KLNMA","KLRHO","KMPUR","KNFRT","KONKA","KONTR","KONYA","KOPOL","KORDS","KRSTL","KRTEK","KRVGD","KTLEV","KUTPO","KUYAS","LIDER","LIDFA","LKMNH","LRSHO","LYKHO","MAALT","MACKO","MAGEN","MAKIM","MAKTK","MANAS","MEDTR","MEGAP","MEGES","MEKAG","MERIT","METUR","MIATK","MIGROS","MIPAZ","MMCAS","MNDRS","MNVRL","MOBTL","MOGAN","MRDIN","MSDOS","MTRKS","NATEN","NIBAS","NILYT","NTHOL","NTTUR","NUGYO","OBASE","OFISM","OLMIP","ONCSM","ORCAY","ORMA","OSMKB","OSTIM","OYLUM","OZGYO","OZKGY","PAPIL","PAREG","PEKGY","PINSU","PKART","PKENT","PLTUR","POLTK","PRDGS","PRZMA","PSDTC","QNBFB","QNBFL","RALYH","RAYSG","RGYAS","RHEAG","RNPOL","RODRG","RTALB","RUBNS","SAFKR","SAYAS","SDTTR","SEGMN","SEKFK","SEKUR","SELEC","SELVA","SEYKM","SNKRN","SNPAM","SOKE","SONME","SRVGY","SUMAS","SUNTK","SUWEN","TABGD","TATGD","TDGYO","TEKTU","TETMT","TEZOL","TGSAS","TIRE","TLMAN","TMPOL","TMSN","TNZTP","TRCAS","TRGYO","TRILC","TSPOR","TUCLK","TUMAS","TUREX","TURGG","TUYAP","ULUUN","UMPAS","UNLU","USDAP","UTPYA","UZERB","VANGD","VERTU","VERUS","VKFYO","YAPRK","YBTAS","YEOTK","YESIL","YGYO","YIGIT","YKSLN","ZEDUR","ZRGYO","ALTINS1","ASUZU","AFYON","AGYO","AHGAZ","APORT","ARMDA","BASGZ","BORDA",
]))

EGITIM_HISSELERI = list(dict.fromkeys(
    BIST30 + BIST50 + BIST100_EK + YILDIZ_PAZAR + VARSAYILAN
))



# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 2: CACHE & GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

_cache: dict       = {}
_cache_lock        = threading.Lock()
_makro_cache: dict = {}
_makro_lock        = threading.Lock()
_makro_zaman       = 0.0
_kap_cache: dict   = {}
_kap_lock          = threading.Lock()

# ML model state
_evrensel_model    = None
_model_lock        = threading.Lock()
_hisse_modeller: dict = {}
_hisse_model_lock  = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 3: YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def _safe(x, default=0.0):
    """NaN/inf güvenli float."""
    try:
        v = float(x)
        return default if not math.isfinite(v) else v
    except:
        return default

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n, min_periods=1).mean()
    l = (-d.clip(upper=0)).rolling(n, min_periods=1).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def _macd(s: pd.Series):
    m = _ema(s, 12) - _ema(s, 26)
    sig = _ema(m, 9)
    return m, sig, m - sig

def _boll(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = _sma(s, n)
    std = s.rolling(n, min_periods=1).std().fillna(0)
    return mid + k * std, mid, mid - k * std

def _stoch(h, l, c, n: int = 14):
    ll = l.rolling(n, min_periods=1).min()
    hh = h.rolling(n, min_periods=1).max()
    k  = 100 * (c - ll) / (hh - ll + 1e-9)
    return k, k.rolling(3, min_periods=1).mean()

def _atr(h, l, c, n: int = 14) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _obv(c: pd.Series, v: pd.Series) -> pd.Series:
    return (np.sign(c.diff()).fillna(0) * v).cumsum()

def _cci(h, l, c, n: int = 20) -> pd.Series:
    tp = (h + l + c) / 3
    ma = _sma(tp, n)
    md = tp.rolling(n, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - ma) / (0.015 * md + 1e-9)

def _wr(h, l, c, n: int = 14) -> pd.Series:
    hh = h.rolling(n, min_periods=1).max()
    ll = l.rolling(n, min_periods=1).min()
    return -100 * (hh - c) / (hh - ll + 1e-9)

def _roc(s: pd.Series, n: int = 10) -> pd.Series:
    return (s / s.shift(n).replace(0, np.nan) - 1) * 100



# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 4: MAKRO VERİ
# ══════════════════════════════════════════════════════════════════════════════

MAKRO_SEMBOLLER = {
    "usdtry": "USDTRY=X",
    "altin":  "GC=F",
    "bist100":"XU100.IS",
    "petrol": "CL=F",
    "eurusd": "EURUSD=X",
    "vix":    "^VIX",
}

def makro_guncelle():
    global _makro_zaman
    with _makro_lock:
        if time.time() - _makro_zaman < MAKRO_SURE:
            return dict(_makro_cache)
    sonuc = {}
    for ad, sem in MAKRO_SEMBOLLER.items():
        try:
            df = yf.Ticker(sem).history(period="3mo")
            if df.empty: continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            c = df["Close"].dropna()
            sonuc[ad] = {
                "son":    _safe(c.iloc[-1]),
                "ret1d":  _safe(c.iloc[-1]/c.iloc[-2]-1) if len(c)>1 else 0,
                "ret5d":  _safe(c.iloc[-1]/c.iloc[-6]-1) if len(c)>5 else 0,
                "ret20d": _safe(c.iloc[-1]/c.iloc[-21]-1) if len(c)>20 else 0,
                "vol20":  _safe(c.pct_change().rolling(20).std().iloc[-1]),
                "sma20":  _safe(_sma(c,20).iloc[-1]/c.iloc[-1]-1),
                "sma50":  _safe(_sma(c,50).iloc[-1]/c.iloc[-1]-1) if len(c)>50 else 0,
                "seri":   c.tolist()[-60:],
            }
        except: pass
    with _makro_lock:
        _makro_cache.update(sonuc)
        _makro_zaman = time.time()
    return sonuc

def piyasa_rejimi(bist_seri: list) -> dict:
    """Basit rejim tespiti: trend, volatilite, momentum."""
    if len(bist_seri) < 20:
        return {"trend": 0, "volatilite": 0, "momentum": 0, "etiket": "BELİRSİZ"}
    arr = np.array(bist_seri[-60:])
    ret  = np.diff(arr) / (arr[:-1] + 1e-9)
    vol  = float(np.std(ret[-20:]) * np.sqrt(252))
    mom  = float(arr[-1]/arr[-20]-1) if len(arr)>=20 else 0
    ma20 = float(arr[-20:].mean())
    trend = 1 if arr[-1] > ma20 * 1.02 else -1 if arr[-1] < ma20 * 0.98 else 0
    etiket = ("YÜKSELEN" if trend==1 and mom>0.03 else
              "DÜŞEN"    if trend==-1 and mom<-0.03 else
              "YATAY")
    return {"trend": trend, "volatilite": round(vol,4),
            "momentum": round(mom,4), "etiket": etiket}


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 5: TEKNİK ANALİZ
# ══════════════════════════════════════════════════════════════════════════════

def teknik_analiz(df: pd.DataFrame) -> dict:
    c = df["Close"]; h = df["High"]; l = df["Low"]; v = df["Volume"]
    n = len(c)

    rsi  = _rsi(c)
    macd_l, macd_s, macd_h = _macd(c)
    bbu, bbm, bbl = _boll(c)
    stk, std = _stoch(h, l, c)
    atr_s = _atr(h, l, c)
    obv_s = _obv(c, v)
    cci_s = _cci(h, l, c)
    wr_s  = _wr(h, l, c)
    roc_s = _roc(c)

    def v(s, i=-1): return _safe(s.iloc[i])

    son   = v(c);  prev = v(c, -2)
    rsi_v = v(rsi); macd_v = v(macd_l); macds_v = v(macd_s)
    bb_u  = v(bbu); bb_l = v(bbl); bb_m = v(bbm)
    stk_v = v(stk); std_v = v(std)
    atr_v = v(atr_s)
    cci_v = v(cci_s); wr_v = v(wr_s); roc_v = v(roc_s)
    obv_v = v(obv_s); obv_p = v(obv_s, -2)

    s20 = v(_sma(c,20)); s50 = v(_sma(c,50)); s200 = v(_sma(c,200))
    e9  = v(_ema(c,9));  e21 = v(_ema(c,21))

    puanlar = []

    # RSI
    if rsi_v < 30:   puanlar.append(+0.8)
    elif rsi_v < 45: puanlar.append(+0.3)
    elif rsi_v > 70: puanlar.append(-0.8)
    elif rsi_v > 55: puanlar.append(-0.3)
    else:            puanlar.append(0.0)

    # MACD
    macd_p = +0.6 if macd_v > macds_v else -0.6
    if abs(macd_v) < atr_v * 0.05: macd_p *= 0.5
    puanlar.append(macd_p)

    # Bollinger
    bb_w = (bb_u - bb_l) / (bb_m + 1e-9)
    if son < bb_l * 1.01:  puanlar.append(+0.7)
    elif son > bb_u * 0.99: puanlar.append(-0.7)
    elif son > bb_m and bb_w > 0.05: puanlar.append(+0.2)
    else: puanlar.append(0.0)

    # Stochastic
    if stk_v < 20:   puanlar.append(+0.6)
    elif stk_v > 80: puanlar.append(-0.6)
    else:            puanlar.append(0.0)

    # MA trend
    ma_puan = 0
    if son > s20: ma_puan += 0.2
    if son > s50: ma_puan += 0.3
    if son > s200: ma_puan += 0.5
    if e9 > e21:  ma_puan += 0.2
    puanlar.append(np.clip(ma_puan - 0.6, -1, 1))

    # CCI
    if cci_v < -100: puanlar.append(+0.5)
    elif cci_v > 100: puanlar.append(-0.5)
    else:             puanlar.append(cci_v / 200)

    # Williams %R
    if wr_v < -80: puanlar.append(+0.5)
    elif wr_v > -20: puanlar.append(-0.5)
    else: puanlar.append(0.0)

    # OBV trend
    puanlar.append(+0.4 if obv_v > obv_p else -0.4)

    # ROC momentum
    puanlar.append(np.clip(roc_v / 10, -0.5, 0.5))

    teknik_skor = float(np.clip(np.mean(puanlar), -1, 1))

    # Pivot noktaları
    pivot = (v(h) + v(l) + son) / 3
    r1 = 2 * pivot - v(l)
    s1 = 2 * pivot - v(h)

    return {
        "teknik_skor": round(teknik_skor, 4),
        "rsi": round(rsi_v, 2),
        "macd": round(macd_v, 4),
        "macd_sinyal": round(macds_v, 4),
        "macd_hist": round(v(macd_h), 4),
        "stoch_k": round(stk_v, 2),
        "stoch_d": round(std_v, 2),
        "cci": round(cci_v, 2),
        "williams_r": round(wr_v, 2),
        "roc": round(roc_v, 2),
        "atr": round(atr_v, 4),
        "bb_ust": round(bb_u, 2),
        "bb_alt": round(bb_l, 2),
        "sma20": round(s20, 2),
        "sma50": round(s50, 2),
        "sma200": round(s200, 2),
        "ema9": round(e9, 2),
        "ema21": round(e21, 2),
        "pivot": round(pivot, 2),
        "r1": round(r1, 2),
        "s1": round(s1, 2),
        "obv_trend": "YUKARI" if obv_v > obv_p else "ASAGI",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 6: TEMEL ANALİZ + ANALİST KONSENSÜSÜ
# ══════════════════════════════════════════════════════════════════════════════

def temel_analiz(ticker) -> dict:
    try:
        info = ticker.info or {}
    except:
        info = {}

    def g(k, default=None):
        v = info.get(k)
        return v if v is not None else default

    pe   = _safe(g("trailingPE"),    0)
    pb   = _safe(g("priceToBook"),   0)
    ps   = _safe(g("priceToSalesTrailingTwelveMonths"), 0)
    dy   = _safe(g("dividendYield"), 0) * 100
    roe  = _safe(g("returnOnEquity"),0) * 100
    roa  = _safe(g("returnOnAssets"),0) * 100
    debt = _safe(g("debtToEquity"),  0)
    mg   = _safe(g("profitMargins"), 0) * 100
    rev_g= _safe(g("revenueGrowth"), 0) * 100
    earn_g=_safe(g("earningsGrowth"),0) * 100
    curr = _safe(g("currentRatio"),  1)
    beta = _safe(g("beta"),          1)
    mc   = _safe(g("marketCap"),     0)
    sirket = g("shortName") or g("longName") or ""
    sektor = g("sector") or g("industry") or "—"

    puanlar = []

    # F/K
    if 0 < pe < 10:   puanlar.append(+0.8)
    elif 0 < pe < 20: puanlar.append(+0.4)
    elif pe > 40:     puanlar.append(-0.5)
    elif pe < 0:      puanlar.append(-0.3)

    # PD/DD
    if 0 < pb < 1:    puanlar.append(+0.7)
    elif 0 < pb < 2:  puanlar.append(+0.3)
    elif pb > 5:      puanlar.append(-0.3)

    # Temettü
    if dy > 5:        puanlar.append(+0.6)
    elif dy > 2:      puanlar.append(+0.3)

    # ROE
    if roe > 20:      puanlar.append(+0.6)
    elif roe > 10:    puanlar.append(+0.3)
    elif roe < 0:     puanlar.append(-0.5)

    # Büyüme
    if earn_g > 20:   puanlar.append(+0.5)
    elif earn_g > 0:  puanlar.append(+0.2)
    elif earn_g < -20: puanlar.append(-0.4)

    # Borç
    if debt < 50:     puanlar.append(+0.3)
    elif debt > 200:  puanlar.append(-0.4)

    temel_skor = float(np.clip(np.mean(puanlar), -1, 1)) if puanlar else 0.0

    detay = {}
    if pe > 0:   detay["F/K"] = f"{pe:.1f}x"
    if pb > 0:   detay["PD/DD"] = f"{pb:.1f}x"
    if ps > 0:   detay["F/S"] = f"{ps:.1f}x"
    if dy > 0:   detay["Temettü"] = f"%{dy:.2f}"
    if roe != 0: detay["ROE"] = f"%{roe:.1f}"
    if roa != 0: detay["ROA"] = f"%{roa:.1f}"
    if mg != 0:  detay["Kâr Marjı"] = f"%{mg:.1f}"
    if debt > 0: detay["Borç/Özkaynak"] = f"{debt:.0f}%"
    if rev_g != 0: detay["Gelir Büyüme"] = f"%{rev_g:.1f}"
    if beta != 0:  detay["Beta"] = f"{beta:.2f}"
    if mc > 0:   detay["Piy. Değer"] = f"{mc/1e9:.1f}B ₺"

    return {
        "temel_skor": round(temel_skor, 4),
        "sirket": sirket,
        "sektor": sektor,
        "detay": detay,
        "beta": round(beta, 2),
    }

def analist_analiz(ticker) -> dict:
    """
    Yahoo Finance analist konsensüsü.
    Döner: {skor, oneri, hedef_fiyat, analist_sayisi}
    """
    try:
        info = ticker.info or {}
    except:
        info = {}

    oneri    = (info.get("recommendationKey") or "").lower()
    hedef    = _safe(info.get("targetMeanPrice"), 0)
    analist  = _safe(info.get("numberOfAnalystOpinions"), 0)
    son_fiy  = _safe(info.get("currentPrice") or info.get("regularMarketPrice"), 0)

    # Potansiyel yükseliş
    potansiyel = (hedef / son_fiy - 1) if son_fiy > 0 and hedef > 0 else 0

    # Konsensüs skoru
    oneri_skor = {
        "strong_buy": +0.9, "buy": +0.6, "outperform": +0.5,
        "hold": 0.0, "neutral": 0.0, "underperform": -0.5,
        "sell": -0.6, "strong_sell": -0.9,
    }.get(oneri, 0.0)

    # Potansiyele göre de ağırlıklandır
    pot_skor = np.clip(potansiyel * 3, -0.5, 0.5)
    skor = oneri_skor * 0.6 + float(pot_skor) * 0.4 if hedef > 0 else oneri_skor

    # Analist sayısı azsa güven düşür
    if analist < 3: skor *= 0.5

    oneri_tr = {
        "strong_buy": "Güçlü Al", "buy": "Al", "outperform": "Üstün Performans",
        "hold": "Tut", "neutral": "Nötr", "underperform": "Düşük Performans",
        "sell": "Sat", "strong_sell": "Güçlü Sat",
    }.get(oneri, "Veri Yok")

    return {
        "skor": round(float(np.clip(skor, -1, 1)), 4),
        "oneri": oneri_tr,
        "hedef_fiyat": round(hedef, 2) if hedef > 0 else None,
        "potansiyel": round(potansiyel * 100, 1) if hedef > 0 else None,
        "analist_sayisi": int(analist),
    }



# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 7: DUYGU ANALİZİ (Google News + Kelime Tabanlı)
# ══════════════════════════════════════════════════════════════════════════════

POZ_KELIMELER = [
    "artış","yükseliş","büyüme","kâr","kazanç","rekor","güçlü","pozitif",
    "ihracat","sözleşme","temettü","geri alım","ortaklık","yatırım","ihale kazandı",
    "olumlu","başarılı","anlaşma","kapasitei artırım","sipariş","ihracat",
]
NEG_KELIMELER = [
    "düşüş","kayıp","zarar","negatif","risk","dava","ceza","soruşturma",
    "iflas","konkordato","haciz","istifa","fabrika kapatma","üretim durdu",
    "sözleşme fesih","borç","temerrüt","olumsuz","endişe","baskı",
]

def _kelime_puan(baslik: str) -> float:
    bl = baslik.lower()
    poz = sum(1 for k in POZ_KELIMELER if k in bl)
    neg = sum(1 for k in NEG_KELIMELER if k in bl)
    return float(np.clip((poz - neg) * 0.25, -1, 1))

def gnews_cek(sembol: str, sirket: str = "") -> list:
    aramalar = [f"{sembol} hisse borsa", f"{sembol} bist"]
    if sirket: aramalar.append(sirket[:30])
    haberler = []; gorulmus = set()
    for arama in aramalar:
        if len(haberler) >= 10: break
        try:
            url = f"https://news.google.com/rss/search?q={quote(arama)}&hl=tr&gl=TR&ceid=TR:tr"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=6) as r:
                root = ET.fromstring(r.read().decode("utf-8", errors="replace"))
            for item in root.findall(".//item")[:6]:
                b = (item.findtext("title") or "").strip()
                if not b or b in gorulmus: continue
                gorulmus.add(b)
                ts = item.findtext("pubDate") or ""
                try:
                    t = datetime.strptime(ts[:25], "%a, %d %b %Y %H:%M:%S")
                    if (datetime.utcnow() - t).days > 7: continue
                    tf = t.strftime("%d.%m.%Y")
                except: tf = "—"
                haberler.append({"baslik": b[:120], "tarih": tf})
        except: pass
    return haberler[:10]

def duygu_analizi(sembol: str, sirket: str, df: pd.DataFrame) -> dict:
    haberler = []
    try: haberler = gnews_cek(sembol, sirket)
    except: pass

    if not haberler:
        # Fallback: teknik momentum
        c = df["Close"]
        ret5 = _safe(c.iloc[-1]/c.iloc[-6]-1) if len(c)>5 else 0
        skor = float(np.clip(ret5 * 5, -0.5, 0.5))
        seviye = ("POZİTİF" if skor>0.1 else "NEGATİF" if skor<-0.1 else "NÖTR")
        return {"skor": round(skor,4), "seviye": seviye, "haberler": [],
                "pozitif": 0, "negatif": 0, "yontem": "momentum"}

    puanlar = [_kelime_puan(h["baslik"]) for h in haberler]
    agirlik = np.linspace(1.5, 1.0, len(puanlar))
    skor = float(np.clip(np.average(puanlar, weights=agirlik), -1, 1))
    poz = sum(1 for p in puanlar if p > 0.1)
    neg = sum(1 for p in puanlar if p < -0.1)

    if skor > 0.4:    seviye = "ÇOK POZİTİF"
    elif skor > 0.15: seviye = "POZİTİF"
    elif skor < -0.4: seviye = "ÇOK NEGATİF"
    elif skor < -0.15: seviye = "NEGATİF"
    else:             seviye = "NÖTR"

    etiketli = []
    for i, h in enumerate(haberler[:6]):
        p = puanlar[i] if i < len(puanlar) else 0
        etiketli.append({**h,
            "duygu": "😊 POZİTİF" if p>0.1 else "😟 NEGATİF" if p<-0.1 else "😐 NÖTR",
            "skor": round(p, 2)})

    return {"skor": round(skor,4), "seviye": seviye, "haberler": etiketli,
            "pozitif": poz, "negatif": neg, "yontem": "kelime"}


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 8: KAP BİLDİRİMLERİ
# ══════════════════════════════════════════════════════════════════════════════

KAP_BILDIRIM_TUR = [
    (["pay geri alım","hisse geri alım","geri alım programı"],  "🔵 Pay Geri Alımı",        +0.80),
    (["temettü","kar payı","karpayı","nakit temettü"],          "💰 Temettü",                +0.70),
    (["yeni sözleşme","önemli sözleşme","ihracat sözleşme"],    "📈 Yeni Sözleşme",          +0.65),
    (["ihale kazandı","ihale kazanma","proje kazanma"],         "🏗️ İhale/Proje",            +0.60),
    (["iş birliği","ortaklık","işbirliği","stratejik"],         "🤝 Ortaklık/Anlaşma",       +0.55),
    (["sermaye artır","bedelsiz","rüçhan hakkı"],               "📊 Sermaye Artırımı",       +0.40),
    (["yatırım","kapasite artır","yeni fabrika"],               "🏭 Yatırım",                +0.45),
    (["iflas","konkordato"],                                    "💀 İflas/Konkordato",       -0.90),
    (["haciz","icra"],                                         "🔒 Haciz/İcra",              -0.75),
    (["soruşturma","spk","inceleme"],                           "🔍 Soruşturma",             -0.65),
    (["zarar açıkladı","net zarar","dönem zararı","zarar etti"],"🔴 Zarar",                  -0.60),
    (["fabrika kapatma","üretim durdu","tesis kapatma"],        "🏭 Üretim Durma",           -0.60),
    (["sözleşme feshedildi","sözleşme iptal","fesih"],          "❌ Sözleşme Feshi",          -0.55),
    (["para cezası","idari para","vergi cezası","tazminat ödeyecek"], "🚨 Para Cezası",       -0.55),
    (["dava açıldı","hukuki","mahkeme"],                        "⚖️ Hukuki Süreç",           -0.45),
    (["genel müdür istifa","ceo ayrıldı","üst yönetim"],        "👤 Üst Yönetici Ayrılışı",  -0.40),
    (["ödeyecek","tazminat ödeyecek","borç ödeme"],             "💸 Ödeme Yükümlülüğü",      -0.40),
    (["istifa","ayrıldı","görevden"],                           "👤 Yönetici Değişikliği",   -0.25),
    (["genel kurul","olağan genel kurul"],                      "📅 Genel Kurul",            +0.05),
    (["finansal tablo","bilanço","gelir tablosu"],              "📊 Finansal Tablo",         +0.10),
]

def _kap_tur(baslik: str) -> tuple:
    bl = baslik.lower()
    for anahtar_list, tur, skor in KAP_BILDIRIM_TUR:
        if any(a in bl for a in anahtar_list):
            return tur, skor
    return "📄 Genel Bildirim", 0.05

def kap_analiz(sembol: str) -> dict:
    sembol = sembol.upper()
    with _kap_lock:
        if sembol in _kap_cache:
            c = _kap_cache[sembol]
            if time.time() - c["zaman"] < KAP_SURE:
                return c["data"]

    bildirimleri = []
    # KAP API
    try:
        req = Request(
            f"https://www.kap.org.tr/tr/api/disclosures/member/{sembol}",
            headers={"User-Agent":"Mozilla/5.0","Accept":"application/json",
                     "Referer":"https://www.kap.org.tr/"})
        with urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        items = data if isinstance(data, list) else data.get("data", [])
        for item in items[:15]:
            b = (item.get("title") or item.get("baslik") or "").strip()
            t = str(item.get("date") or item.get("publishDate") or "—")[:16]
            if b: bildirimleri.append({"baslik": b[:150], "tarih": t})
    except: pass

    # Google News KAP fallback
    if not bildirimleri:
        try:
            url = f"https://news.google.com/rss/search?q={quote(sembol+' KAP bildirim özel durum')}&hl=tr&gl=TR&ceid=TR:tr"
            with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0"}), timeout=6) as r:
                root = ET.fromstring(r.read().decode("utf-8", errors="replace"))
            kw = ["kap","bildirim","özel durum","kamuoyu","ödeyecek","fabrika",
                  "soruşturma","temettü","geri alım","istifa","zarar"]
            for item in root.findall(".//item")[:10]:
                b = (item.findtext("title") or "").strip()
                t = (item.findtext("pubDate") or "")[:16]
                if b and any(k in b.lower() for k in kw):
                    bildirimleri.append({"baslik": b, "tarih": t})
        except: pass

    islenmiş = []
    skorlar  = []
    for b in bildirimleri[:8]:
        tur, skor = _kap_tur(b["baslik"])
        islenmiş.append({"baslik": b["baslik"][:120], "tarih": b.get("tarih","—"),
                          "tur": tur, "skor": round(skor,2)})
        skorlar.append(skor)

    kap_skor = 0.0
    if skorlar:
        ag = np.linspace(1.5, 0.8, len(skorlar))
        kap_skor = float(np.clip(np.average(skorlar, weights=ag[:len(skorlar)]), -1, 1))

    en_onemli = max(islenmiş, key=lambda x: abs(x["skor"])) if islenmiş else None
    pos = sum(1 for s in skorlar if s > 0.1)
    neg = sum(1 for s in skorlar if s < -0.1)
    ozet = f"{len(bildirimleri)} bildirim — {pos} olumlu, {neg} olumsuz" if bildirimleri else "Bildirim bulunamadı"

    sonuc = {"skor": round(kap_skor,4), "bildirimleri": islenmiş[:6],
             "ozet": ozet, "en_onemli": en_onemli, "bildirim_sayisi": len(bildirimleri)}
    with _kap_lock:
        _kap_cache[sembol] = {"data": sonuc, "zaman": time.time()}
    return sonuc



# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 9: ARIMA TREND TAHMİNİ (statsmodels olmadan)
# ══════════════════════════════════════════════════════════════════════════════

def arima_trend(df: pd.DataFrame, gun: int = 10) -> dict:
    """
    Basit ARIMA(1,1,1) benzeri trend tahmini.
    statsmodels gerekmez — numpy ile AR(1) + trend.
    """
    try:
        c = df["Close"].dropna().values
        if len(c) < 30:
            return {"skor": 0.0, "tahmini_fiyat": None, "potansiyel": 0.0, "guven": 0.0}

        # 1. Fark serisi (I=1)
        diff = np.diff(c)
        if len(diff) < 10:
            return {"skor": 0.0, "tahmini_fiyat": None, "potansiyel": 0.0, "guven": 0.0}

        # 2. AR(1) katsayısı — OLS
        X_ar = diff[:-1].reshape(-1, 1)
        y_ar = diff[1:]
        phi  = float(np.dot(X_ar.flatten(), y_ar) / (np.dot(X_ar.flatten(), X_ar.flatten()) + 1e-9))
        phi  = np.clip(phi, -0.95, 0.95)

        # 3. Lineer trend bileşeni
        t    = np.arange(len(c[-60:]))
        A    = np.vstack([t, np.ones_like(t)]).T
        slope, intercept = np.linalg.lstsq(A, c[-60:], rcond=None)[0]

        # 4. gun adım ileriye tahmin
        son_diff = diff[-1]
        proj_diffs = []
        d = son_diff
        for _ in range(gun):
            d = phi * d
            proj_diffs.append(d)

        trend_katki  = slope * gun
        diff_katki   = sum(proj_diffs)
        tahmini_fiyat = c[-1] + diff_katki + trend_katki * 0.3

        potansiyel = (tahmini_fiyat / c[-1] - 1) if c[-1] > 0 else 0

        # ARIMA skoru: potansiyele göre [-1, +1]
        skor = float(np.clip(potansiyel * 8, -1, 1))

        # Güven: AR(1) katsayısının mutlak değeri (ne kadar açıklayıcı)
        fitted  = phi * X_ar.flatten()
        ss_res  = np.sum((y_ar - fitted)**2)
        ss_tot  = np.sum((y_ar - y_ar.mean())**2) + 1e-9
        r2      = max(0, 1 - ss_res/ss_tot)
        guven   = round(float(r2), 3)

        return {
            "skor":          round(skor, 4),
            "tahmini_fiyat": round(tahmini_fiyat, 2),
            "potansiyel":    round(potansiyel * 100, 2),
            "phi":           round(phi, 4),
            "guven":         guven,
            "gun":           gun,
        }
    except Exception as e:
        return {"skor": 0.0, "tahmini_fiyat": None, "potansiyel": 0.0, "guven": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 10: ML ÖZELLİK ÜRETİMİ
# ══════════════════════════════════════════════════════════════════════════════

def ozellik_uret(df: pd.DataFrame, makro: dict = None) -> pd.DataFrame:
    """56+ özellik — teknik + makro + rejim. Eğitim ve tahmin aynı fonksiyon."""
    if makro is None:
        makro = dict(_makro_cache)

    c = df["Close"]; h = df["High"]; l = df["Low"]; v = df["Volume"]

    f = pd.DataFrame(index=df.index)

    # Fiyat özellikleri
    f["ret1"]  = c.pct_change(1)
    f["ret3"]  = c.pct_change(3)
    f["ret5"]  = c.pct_change(5)
    f["ret10"] = c.pct_change(10)
    f["ret20"] = c.pct_change(20)

    # RSI
    f["rsi14"] = _rsi(c, 14) / 100
    f["rsi7"]  = _rsi(c, 7)  / 100
    f["rsi28"] = _rsi(c, 28) / 100

    # MACD
    ml, ms, mh = _macd(c)
    atr_s = _atr(h, l, c, 14)
    f["macd_norm"]  = ml / (atr_s + 1e-9)
    f["macds_norm"] = ms / (atr_s + 1e-9)
    f["macdh_norm"] = mh / (atr_s + 1e-9)

    # Bollinger
    bbu, bbm, bbl = _boll(c, 20)
    bb_w = (bbu - bbl) / (bbm + 1e-9)
    f["bb_pos"]  = (c - bbl) / (bbu - bbl + 1e-9)
    f["bb_width"]= bb_w

    # Stochastic
    stk, std = _stoch(h, l, c)
    f["stoch_k"] = stk / 100
    f["stoch_d"] = std / 100

    # ATR normalize
    f["atr_norm"] = atr_s / (c + 1e-9)

    # CCI, Williams, ROC
    f["cci_norm"] = _cci(h, l, c) / 200
    f["wr_norm"]  = _wr(h, l, c) / 100
    f["roc10"]    = _roc(c, 10) / 100

    # MA konumu
    for n in [5, 10, 20, 50, 200]:
        sma = _sma(c, n)
        f[f"ma{n}_pos"] = (c - sma) / (sma + 1e-9)

    # EMA trend
    e9 = _ema(c, 9); e21 = _ema(c, 21)
    f["ema_cross"] = (e9 - e21) / (e21 + 1e-9)

    # Volume
    v_ma20 = _sma(v, 20)
    f["vol_ratio"]  = v / (v_ma20 + 1e-9)
    f["vol_trend"]  = v_ma20.pct_change(5)
    f["obv_norm"]   = _obv(c, v).pct_change(10).fillna(0)

    # Volatilite
    f["vol5"]  = c.pct_change().rolling(5).std()
    f["vol20"] = c.pct_change().rolling(20).std()
    f["vol_ratio2"] = f["vol5"] / (f["vol20"] + 1e-9)

    # Fiyat örüntüleri
    f["higher_high"] = ((h > h.shift(1)) & (h.shift(1) > h.shift(2))).astype(int)
    f["lower_low"]   = ((l < l.shift(1)) & (l.shift(1) < l.shift(2))).astype(int)
    f["doji"]        = ((h - l) > 0).astype(int) * (abs(c - c.shift(1)) / (h - l + 1e-9))
    f["gap"]         = (c.shift(1) - c) / (c + 1e-9)

    # Makro özellikler
    for ad in ["usdtry","altin","bist100","petrol","eurusd","vix"]:
        m = makro.get(ad, {})
        f[f"makro_{ad}_ret1"]  = m.get("ret1d",  0.0)
        f[f"makro_{ad}_ret5"]  = m.get("ret5d",  0.0)
        f[f"makro_{ad}_vol"]   = m.get("vol20",  0.0)

    # BIST100 korelasyonu
    bist_seri = makro.get("bist100", {}).get("seri", [])
    if len(bist_seri) >= 20:
        bist_arr = np.array(bist_seri[-len(c):]) if len(bist_seri) >= len(c) else np.array(bist_seri)
        n_min = min(len(c), len(bist_arr))
        if n_min >= 20:
            hr = c.pct_change().values[-n_min:]
            br = np.diff(bist_arr[-n_min:]) / (bist_arr[-n_min:-1] + 1e-9)
            nc = min(len(hr), len(br))
            if nc >= 20:
                kor = float(np.corrcoef(hr[-nc:], br[-nc:])[0,1])
                f["bist_kor"] = kor if np.isfinite(kor) else 0.0
            else: f["bist_kor"] = 0.0
        else: f["bist_kor"] = 0.0
    else: f["bist_kor"] = 0.0

    # Piyasa rejimi
    rejim = piyasa_rejimi(bist_seri)
    f["rejim_trend"]     = rejim["trend"]
    f["rejim_volatilite"]= rejim["volatilite"]
    f["rejim_momentum"]  = rejim["momentum"]

    # Hedef — 10 günlük ±%1.5
    f["hedef"]  = c.shift(-ML_HEDEF_GUN) / c - 1
    f["sinyal"] = (f["hedef"] > ML_ESIK).astype(int) - (f["hedef"] < -ML_ESIK).astype(int)

    # Kısa vade hedef (5 günlük)
    f["hedef_5d"]  = c.shift(-5) / c - 1
    f["sinyal_5d"] = (f["hedef_5d"] > 0.01).astype(int) - (f["hedef_5d"] < -0.01).astype(int)

    return f.replace([np.inf, -np.inf], np.nan).dropna()



# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 11: ML MODEL (EnsembleModel v2)
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleModel:
    """
    7 katmanlı stacked ensemble.
    Eğitim: 10 günlük hedef, ±%1.5 eşiği, 1:1:1 denge, makro dahil.
    """
    VERSION = "5.0"

    def __init__(self):
        self.k1 = LinearRegression()
        self.k2 = Ridge(alpha=1.0)
        self.k3 = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1)
        self.k4 = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42)
        self.k5 = ExtraTreesClassifier(
            n_estimators=150, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1)
        self.meta = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")
        self.sc   = RobustScaler()
        self.fs   = None
        self.cols: list       = []
        self.cols_secili: list= []
        self.dogruluk: float  = 0.0
        self.metrikler: dict  = {}
        self.egitim_tarihi: str = "—"
        self.ornek_sayisi: int  = 0
        self.hedef_gun: int   = ML_HEDEF_GUN
        self.esik: float      = ML_ESIK

    def _meta_features(self, Xm):
        parts = [
            self.k1.predict(Xm).reshape(-1, 1),
            self.k2.predict(Xm).reshape(-1, 1),
            self.k3.predict_proba(Xm),
            self.k4.predict_proba(Xm),
            self.k5.predict_proba(Xm),
        ]
        return np.hstack(parts)

    def egit(self, X: np.ndarray, yr: np.ndarray, yc: np.ndarray, cols: list):
        self.cols = cols
        self.ornek_sayisi = len(X)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Özellik seçimi
        try:
            k_feat = min(35, X.shape[1])
            self.fs = SelectKBest(mutual_info_classif, k=k_feat)
            X = self.fs.fit_transform(X, yc)
            mask = self.fs.get_support()
            self.cols_secili = [c for c, m in zip(cols, mask) if m]
        except:
            self.fs = None
            self.cols_secili = cols

        # Scale
        Xs = self.sc.fit_transform(X)

        # Train/test — zaman sıralı %80/%20
        cut = int(len(Xs) * 0.80)
        Xtr, Xte = Xs[:cut], Xs[cut:]
        yctr, ycte = yc[:cut], yc[cut:]
        yrtr = yr[:cut]

        # Sınıf dengeleme: 1:1:1
        idx_al  = np.where(yctr ==  1)[0]
        idx_sat = np.where(yctr == -1)[0]
        idx_tut = np.where(yctr ==  0)[0]
        hedef_n = max(len(idx_al), len(idx_sat))
        hedef_tut = min(len(idx_tut), hedef_n)  # 1:1:1

        if hedef_tut > 0 and hedef_tut < len(idx_tut):
            idx_tut = resample(idx_tut, n_samples=hedef_tut, random_state=42, replace=False)

        idx_d = np.concatenate([idx_al, idx_sat, idx_tut])
        np.random.shuffle(idx_d)
        Xtr_d, yctr_d, yrtr_d = Xtr[idx_d], yctr[idx_d], yrtr[idx_d]

        print(f"    Denge: AL={len(idx_al)}, SAT={len(idx_sat)}, TUT={hedef_tut}")

        # Regresyon katmanları
        self.k1.fit(Xtr_d, yrtr_d)
        self.k2.fit(Xtr_d, yrtr_d)

        # Sınıflandırma katmanları
        self.k3.fit(Xtr_d, yctr_d)

        # GB — sample_weight (class_weight desteklemez)
        _cls, _cnt = np.unique(yctr_d, return_counts=True)
        _wmap = {c: len(yctr_d) / (len(_cls) * n) for c, n in zip(_cls, _cnt)}
        _sw   = np.array([_wmap[c] for c in yctr_d])
        self.k4.fit(Xtr_d, yctr_d, sample_weight=_sw)

        self.k5.fit(Xtr_d, yctr_d)

        # Meta-learner
        self.meta.fit(self._meta_features(Xtr_d), yctr_d)

        # Doğruluk — 5-fold CV + son %20 birleşik
        y_pred   = self.meta.predict(self._meta_features(Xte))
        acc_son  = float(accuracy_score(ycte, y_pred))

        try:
            skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_sc = cross_val_score(
                RandomForestClassifier(n_estimators=50, max_depth=6,
                    class_weight="balanced", random_state=42, n_jobs=-1),
                Xs, yc, cv=skf, scoring="accuracy", n_jobs=-1)
            acc_cv = float(cv_sc.mean())
            self.dogruluk = round(acc_cv * 0.6 + acc_son * 0.4, 4)
            print(f"    Doğruluk → son:{acc_son*100:.1f}% | CV:{acc_cv*100:.1f}% | birleşik:{self.dogruluk*100:.1f}%")
        except:
            self.dogruluk = acc_son

        # Sınıf bazlı metrikler
        siniflar = sorted(self.meta.classes_)
        pr, rc, f1, sup = precision_recall_fscore_support(
            ycte, y_pred, labels=siniflar, zero_division=0)
        self.metrikler = {
            int(c): {"precision": round(float(p),4), "recall": round(float(r),4),
                     "f1": round(float(f),4), "destek": int(s)}
            for c, p, r, f, s in zip(siniflar, pr, rc, f1, sup)
        }
        self.egitim_tarihi = datetime.now().strftime("%d.%m.%Y %H:%M")

    def tahmin(self, X_son: np.ndarray) -> dict:
        X_son = np.nan_to_num(X_son, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.fs is not None:
            try: X_son = self.fs.transform(X_son)
            except: pass
        Xs    = self.sc.transform(X_son)
        mf    = self._meta_features(Xs)
        proba = self.meta.predict_proba(mf)[0]
        ol    = {int(c): float(p) for c, p in zip(self.meta.classes_, proba)}
        ml_skor   = ol.get(1, 0) - ol.get(-1, 0)
        getiri    = float(self.k2.predict(Xs)[0])
        sinif     = self.meta.predict(mf)[0]
        ml_sinyal = {1: "AL", -1: "SAT", 0: "TUT"}.get(int(sinif), "TUT")
        return {
            "ml_skor":    round(float(ml_skor), 4),
            "ml_sinyal":  ml_sinyal,
            "olasilik":   {k: round(v,4) for k,v in ol.items()},
            "getiri_tahmini": round(getiri * 100, 2),
        }


# ── Model yükleme / kaydetme ──────────────────────────────────────────────────

def model_yukle() -> "EnsembleModel | None":
    if MODEL_PATH.exists():
        try:
            m = joblib.load(MODEL_PATH)
            print(f"  ✅ Model yüklendi: {m.egitim_tarihi} | "
                  f"{m.ornek_sayisi:,} örnek | %{m.dogruluk*100:.1f}")
            return m
        except Exception as e:
            print(f"  ⚠️  Model yüklenemedi: {e}")
    return None

def model_egit_ve_kaydet(hisseler: list = None) -> EnsembleModel:
    global _evrensel_model
    if hisseler is None:
        hisseler = EGITIM_HISSELERI

    print(f"\n🔄 Evrensel model eğitimi ({len(hisseler)} hisse)…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    makro_snap = dict(_makro_cache)
    tum_X, tum_yr, tum_yc = [], [], []
    cols_ref = None
    basarili = 0

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(_egitim_veri_cek, sem, makro_snap): sem for sem in hisseler}
        for fut in as_completed(futures):
            sem = futures[fut]
            try:
                result = fut.result()
                if result is None: continue
                X, yr, yc, cols = result
                if cols_ref is None:
                    cols_ref = cols
                elif cols != cols_ref:
                    ortak = [c for c in cols_ref if c in cols]
                    if len(ortak) < len(cols_ref) * 0.85: continue
                    cols_ref = ortak
                    tum_X = [x[:, :len(ortak)] for x in tum_X]
                    X = X[:, :len(ortak)]
                tum_X.append(X); tum_yr.append(yr); tum_yc.append(yc)
                basarili += 1
            except: pass

    if not tum_X:
        raise RuntimeError("Hiç veri toplanamadı")

    X_all  = np.vstack(tum_X)
    yr_all = np.concatenate(tum_yr)
    yc_all = np.concatenate(tum_yc)

    print(f"  📊 Ham: {len(X_all):,} örnek, {basarili} hisse, {len(cols_ref)} özellik")

    # Temizlik
    X_all = np.where(np.isinf(X_all), np.nan, X_all)
    col_med = np.nanmedian(X_all, axis=0)
    for j in range(X_all.shape[1]):
        mask = np.isnan(X_all[:, j])
        if mask.any(): X_all[mask, j] = col_med[j]

    col_std  = np.std(X_all, axis=0) + 1e-9
    col_mean = np.mean(X_all, axis=0)
    X_all    = np.clip(X_all, col_mean - 5*col_std, col_mean + 5*col_std)

    gecerli = np.isfinite(X_all).all(axis=1) & np.isfinite(yr_all) & np.isfinite(yc_all.astype(float))
    X_all = X_all[gecerli]; yr_all = yr_all[gecerli]; yc_all = yc_all[gecerli]
    print(f"  🧹 Temiz: {len(X_all):,} örnek")

    model = EnsembleModel()
    model.egit(X_all, yr_all, yc_all, cols_ref)

    joblib.dump(model, MODEL_PATH)
    print(f"  💾 Kaydedildi: {MODEL_PATH}")
    print(f"  ✅ Doğruluk: %{model.dogruluk*100:.1f}")

    with _model_lock:
        _evrensel_model = model
    return model

def _egitim_veri_cek(sem: str, makro_snap: dict):
    """Tek hisse için eğitim verisi çek — paralel çalışır."""
    try:
        ticker = yf.Ticker(f"{sem}.IS")
        df = ticker.history(period="2y")
        if df.empty or len(df) < 120: return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        duz = FIYAT_DUZELTME.get(sem, 1)
        if duz != 1:
            for col in ["Open","High","Low","Close"]:
                if col in df.columns: df[col] *= duz
        feat = ozellik_uret(df, makro_snap)
        if len(feat) < 60: return None
        cols = [c for c in feat.columns if c not in
                ("hedef","sinyal","hedef_5d","sinyal_5d")]
        return feat[cols].values, feat["hedef"].values, feat["sinyal"].values, cols
    except:
        return None



# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 12: ML ANALİZ + KELLY CRİTERİON
# ══════════════════════════════════════════════════════════════════════════════

def ml_analiz(df: pd.DataFrame, sembol: str = "") -> dict:
    """Evrensel modeli kullanarak ML tahmini üretir."""
    try:
        makro = dict(_makro_cache)
        feat  = ozellik_uret(df, makro)
        if len(feat) < 30:
            return {"ml_skor": 0, "ml_sinyal": "Yetersiz veri", "dogruluk": 0,
                    "olasilik": {}, "getiri_tahmini": 0}
        cols  = [c for c in feat.columns if c not in
                 ("hedef","sinyal","hedef_5d","sinyal_5d")]
        X_son = np.nan_to_num(feat[cols].values[-1:], nan=0.0, posinf=1e6, neginf=-1e6)

        with _model_lock:
            model = _evrensel_model

        if model is None:
            return {"ml_skor": 0, "ml_sinyal": "Model yok", "dogruluk": 0,
                    "olasilik": {}, "getiri_tahmini": 0}

        # Sütun uyumu
        if model.cols and cols != model.cols:
            ortak = [c for c in model.cols if c in cols]
            if not ortak:
                return {"ml_skor": 0, "ml_sinyal": "Sütun uyumsuz", "dogruluk": 0,
                        "olasilik": {}, "getiri_tahmini": 0}
            col_idx = [cols.index(c) for c in ortak]
            X_son = X_son[:, col_idx]
            if model.fs is not None:
                try: X_son = model.fs.transform(X_son)
                except: pass
            Xs = model.sc.transform(X_son)
            mf = model._meta_features(Xs)
        else:
            tahmin = model.tahmin(X_son)
            return {**tahmin, "dogruluk": model.dogruluk,
                    "top5": getattr(model, "cols_secili", [])[:5]}

        proba   = model.meta.predict_proba(mf)[0]
        ol      = {int(c): float(p) for c, p in zip(model.meta.classes_, proba)}
        ml_skor = ol.get(1, 0) - ol.get(-1, 0)
        sinif   = model.meta.predict(mf)[0]
        return {
            "ml_skor":        round(float(ml_skor), 4),
            "ml_sinyal":      {1:"AL",-1:"SAT",0:"TUT"}.get(int(sinif),"TUT"),
            "olasilik":       {k: round(v,4) for k,v in ol.items()},
            "getiri_tahmini": 0,
            "dogruluk":       model.dogruluk,
            "top5":           getattr(model, "cols_secili", [])[:5],
        }
    except Exception as e:
        return {"ml_skor": 0, "ml_sinyal": f"Hata", "dogruluk": 0,
                "olasilik": {}, "getiri_tahmini": 0}


def kelly_criterion(ml_olasilik: dict, risk_odul: float,
                    max_kelly: float = 0.25) -> dict:
    """
    Kelly Criterion ile optimal pozisyon büyüklüğü.
    f* = (p*(b+1) - 1) / b
    p: kazanma olasılığı, b: kazanç/kayıp oranı
    """
    p_al  = ml_olasilik.get(1,  0.33)
    p_sat = ml_olasilik.get(-1, 0.33)
    p_tut = ml_olasilik.get(0,  0.34)

    # En güçlü yönü al
    if p_al >= p_sat:
        p_win  = p_al
        p_loss = p_sat + p_tut * 0.5
        yon    = "AL"
    else:
        p_win  = p_sat
        p_loss = p_al + p_tut * 0.5
        yon    = "SAT"

    b = max(risk_odul, 0.5)  # risk/ödül oranı
    kelly_f = (p_win * (b + 1) - 1) / b if b > 0 else 0
    kelly_f = max(0.0, min(kelly_f, max_kelly))  # [0, max_kelly]

    # Yarım Kelly — daha güvenli
    yari_kelly = kelly_f * 0.5

    return {
        "kelly_f":    round(kelly_f, 4),
        "yari_kelly": round(yari_kelly, 4),
        "yon":        yon,
        "p_kazanma":  round(p_win, 4),
        "aciklama":   f"Sermayenin %{yari_kelly*100:.1f}'i pozisyon için önerilen",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 13: WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest(df: pd.DataFrame, sembol: str = "",
                          train_gun: int = 252, test_gun: int = 63,
                          komisyon: float = 0.001) -> dict:
    """
    Walk-forward backtest: 1 yıl eğit → 3 ay test → kaydır.
    Gerçekçi performans ölçümü.
    """
    c = df["Close"]
    n = len(c)
    if n < train_gun + test_gun:
        return {"hata": "Yeterli veri yok", "toplam_getiri": 0}

    sonuclar = []
    pozisyon = 0
    portfoy  = 1.0
    islemler = []

    pencere_bas = 0
    while pencere_bas + train_gun + test_gun <= n:
        train_end = pencere_bas + train_gun
        test_end  = min(train_end + test_gun, n)

        df_train = df.iloc[pencere_bas:train_end]
        df_test  = df.iloc[train_end:test_end]

        if len(df_train) < 60 or len(df_test) < 5:
            pencere_bas += test_gun
            continue

        # Bu pencere için mini model eğit (hafif RF)
        try:
            makro = dict(_makro_cache)
            feat_tr = ozellik_uret(df_train, makro)
            if len(feat_tr) < 30:
                pencere_bas += test_gun
                continue
            cols = [c for c in feat_tr.columns if c not in
                    ("hedef","sinyal","hedef_5d","sinyal_5d")]
            X_tr = np.nan_to_num(feat_tr[cols].values, nan=0.0)
            y_tr = feat_tr["sinyal"].values
            sc   = RobustScaler()
            X_tr = sc.fit_transform(X_tr)

            # Denge
            idx_al  = np.where(y_tr ==  1)[0]
            idx_sat = np.where(y_tr == -1)[0]
            idx_tut = np.where(y_tr ==  0)[0]
            hn = max(len(idx_al), len(idx_sat))
            idx_tut = resample(idx_tut, n_samples=min(len(idx_tut),hn), random_state=42)
            idx_d   = np.concatenate([idx_al, idx_sat, idx_tut])
            X_d, y_d = X_tr[idx_d], y_tr[idx_d]

            rf = RandomForestClassifier(n_estimators=50, max_depth=6,
                    class_weight="balanced", random_state=42, n_jobs=-1)
            rf.fit(X_d, y_d)

            # Test döneminde sinyal uygula
            feat_te = ozellik_uret(df_test, makro)
            if len(feat_te) < 2:
                pencere_bas += test_gun
                continue
            cols_te = [c for c in feat_te.columns if c not in
                       ("hedef","sinyal","hedef_5d","sinyal_5d")]
            ortak = [c for c in cols if c in cols_te]
            X_te  = sc.transform(np.nan_to_num(feat_te[ortak].values, nan=0.0)[:, :len(ortak)])

            sinyaller = rf.predict(X_te[:, :X_d.shape[1]] if X_te.shape[1] > X_d.shape[1] else X_te)
            fiyatlar  = df_test["Close"].values[:len(sinyaller)]

            for i in range(len(sinyaller)):
                sinyal = int(sinyaller[i])
                if i > 0:
                    ret = fiyatlar[i] / fiyatlar[i-1] - 1
                    if pozisyon == 1:
                        portfoy *= (1 + ret - komisyon * (sinyal != 1))
                    elif pozisyon == -1:
                        portfoy *= (1 - ret - komisyon * (sinyal != -1))

                if sinyal != pozisyon:
                    islemler.append({
                        "tarih": str(df_test.index[i])[:10],
                        "sinyal": {1:"AL",-1:"SAT",0:"TUT"}.get(sinyal,"TUT"),
                        "fiyat": round(float(fiyatlar[i]), 2),
                    })
                    pozisyon = sinyal

            pencere_getiri = portfoy - 1
            sonuclar.append({
                "bas":  str(df_test.index[0])[:10],
                "son":  str(df_test.index[-1])[:10],
                "getiri": round(float(pencere_getiri) * 100, 2),
            })
        except:
            pass

        pencere_bas += test_gun

    if not sonuclar:
        return {"hata": "Backtest başarısız", "toplam_getiri": 0}

    toplam_getiri = (portfoy - 1) * 100
    getiriler = [s["getiri"] for s in sonuclar]
    kazanan   = sum(1 for g in getiriler if g > 0)

    # Buy & Hold karşılaştırması
    bh_getiri = (_safe(df["Close"].iloc[-1]) / _safe(df["Close"].iloc[0]) - 1) * 100

    # Sharpe (yıllık)
    if len(getiriler) > 1:
        ort  = np.mean(getiriler)
        std  = np.std(getiriler) + 1e-9
        sharpe = round((ort / std) * np.sqrt(4), 3)  # 4 çeyrek/yıl
    else:
        sharpe = 0

    return {
        "toplam_getiri":  round(toplam_getiri, 2),
        "bh_getiri":      round(bh_getiri, 2),
        "kazanma_orani":  round(kazanan / len(sonuclar) * 100, 1) if sonuclar else 0,
        "pencere_sayisi": len(sonuclar),
        "sharpe":         sharpe,
        "son_islemler":   islemler[-10:],
        "pencereler":     sonuclar,
    }



# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 14: ANA ANALİZ MOTORU
# ══════════════════════════════════════════════════════════════════════════════

def hisse_analiz(sembol: str) -> dict:
    """
    7 modülü birleştirir, ağırlıklı skor üretir, karar verir.
    Sonuç cache'lenir (CACHE_SURE saniye).
    """
    with _cache_lock:
        if sembol in _cache:
            c = _cache[sembol]
            if time.time() - c["zaman"] < CACHE_SURE:
                return c["sonuc"]

    try:
        ticker = yf.Ticker(f"{sembol}.IS")
        df     = ticker.history(period="2y")
        if df.empty or len(df) < 100:
            return {"sembol": sembol, "hata": "Yeterli veri yok"}
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    except Exception as e:
        return {"sembol": sembol, "hata": str(e)}

    duz = FIYAT_DUZELTME.get(sembol, 1)
    if duz != 1:
        for col in ["Open","High","Low","Close"]:
            if col in df.columns: df[col] *= duz

    # Paralel hesapla — yavaş olanlar aynı anda
    tek = teknik_analiz(df)
    tem = temel_analiz(ticker)

    with ThreadPoolExecutor(max_workers=5) as ex:
        f_duy  = ex.submit(duygu_analizi, sembol, tem["sirket"], df)
        f_kap  = ex.submit(kap_analiz, sembol)
        f_ana  = ex.submit(analist_analiz, ticker)
        f_ml   = ex.submit(ml_analiz, df, sembol)
        f_arima= ex.submit(arima_trend, df, ML_HEDEF_GUN)

        duy   = f_duy.result()
        kap   = f_kap.result()
        ana   = f_ana.result()
        ml    = f_ml.result()
        arima = f_arima.result()

    # Birleşik skor
    bs = (tek["teknik_skor"] * W["teknik"] +
          ml["ml_skor"]      * W["ml"]     +
          duy["skor"]        * W["duygu"]  +
          kap["skor"]        * W["kap"]    +
          tem["temel_skor"]  * W["temel"]  +
          ana["skor"]        * W["analist"]+
          arima["skor"]      * W["arima"])

    if   bs >=  0.35: karar = "GÜÇLÜ AL"
    elif bs >=  0.15: karar = "AL"
    elif bs <= -0.35: karar = "GÜÇLÜ SAT"
    elif bs <= -0.15: karar = "SAT"
    else:             karar = "TUT"

    # Fiyat hesapları
    son  = _safe(df["Close"].iloc[-1])
    prev = _safe(df["Close"].iloc[-2]) if len(df)>1 else son
    k6   = _safe(df["Close"].iloc[-6])  if len(df)>5  else son
    k21  = _safe(df["Close"].iloc[-21]) if len(df)>20 else son

    atr_v  = tek["atr"]
    hedef  = son*(1+min(abs(bs)*0.30, 0.25)) if "AL" in karar else son*(1-min(abs(bs)*0.20, 0.20))
    stop   = son - 2*atr_v if "AL" in karar else son + 2*atr_v
    ro     = abs(hedef-son)/abs(son-stop) if abs(son-stop) > 0 else 0

    # Kelly pozisyon önerisi
    kelly = kelly_criterion(ml.get("olasilik",{}), ro)

    # Makro bilgisi
    makro = dict(_makro_cache)
    rejim = piyasa_rejimi(makro.get("bist100",{}).get("seri",[]))

    sonuc = {
        "sembol":      sembol,
        "sirket":      tem["sirket"],
        "sektor":      tem["sektor"],
        "fiyat":       round(son, 2),
        "onceki":      round(prev, 2),
        "gunluk":      round((son/prev-1)*100, 2) if prev else 0,
        "haftalik":    round((son/k6-1)*100, 2)   if k6   else 0,
        "aylik":       round((son/k21-1)*100, 2)  if k21  else 0,
        "karar":       karar,
        "birlesik":    round(bs, 4),
        # Alt skorlar
        "teknik_skor": round(tek["teknik_skor"], 4),
        "ml_skor":     round(ml["ml_skor"], 4),
        "duygu_skor":  round(duy["skor"], 4),
        "kap_skor":    round(kap["skor"], 4),
        "temel_skor":  round(tem["temel_skor"], 4),
        "analist_skor":round(ana["skor"], 4),
        "arima_skor":  round(arima["skor"], 4),
        # ML detay
        "ml_sinyal":   ml["ml_sinyal"],
        "ml_dogruluk": ml.get("dogruluk", 0),
        "ml_olasilik": ml.get("olasilik", {}),
        "ml_top5":     ml.get("top5", []),
        # Risk
        "hedef":       round(hedef, 2),
        "stop":        round(stop, 2),
        "risk_odul":   round(ro, 2),
        "kelly":       kelly,
        # Detay
        "teknik":   tek,
        "temel":    tem["detay"],
        "duygu":    duy,
        "kap":      kap,
        "analist":  ana,
        "arima":    arima,
        "piyasa_rejimi": rejim,
        "makro": {k: {"ret1d": round(v.get("ret1d",0)*100,2),
                      "ret5d": round(v.get("ret5d",0)*100,2)}
                  for k, v in makro.items() if k != "bist100" or True},
        "guncelleme": datetime.now().strftime("%H:%M:%S"),
    }

    with _cache_lock:
        _cache[sembol] = {"sonuc": sonuc, "zaman": time.time()}

    # Sinyal geçmişine kaydet
    _sinyal_kaydet(sembol, karar, round(son,2), round(bs,4))
    return sonuc


def _sinyal_kaydet(sembol, karar, fiyat, skor):
    try:
        gecmis = json.loads(SINYAL_FILE.read_text()) if SINYAL_FILE.exists() else []
        gecmis.append({
            "sembol": sembol, "karar": karar, "fiyat": fiyat,
            "skor": skor, "tarih": datetime.now().strftime("%Y-%m-%d %H:%M")})
        gecmis = gecmis[-500:]
        SINYAL_FILE.write_text(json.dumps(gecmis, ensure_ascii=False))
    except: pass



# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 15: PORTFÖY YÖNETİMİ
# ══════════════════════════════════════════════════════════════════════════════

def portfolyo_yukle() -> list:
    try:
        return json.loads(PORTFOLYO_FILE.read_text()) if PORTFOLYO_FILE.exists() else []
    except: return []

def portfolyo_kaydet(data: list):
    PORTFOLYO_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 16: FLASK API ENDPOINT'LERİ
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({"status": "BIST Robot API v5.0", "docs": "/api/durum"})

# ── Hisse analizi ─────────────────────────────────────────────────────────────
@app.route("/api/analiz/<sembol>")
def api_analiz(sembol):
    return jsonify(hisse_analiz(sembol.upper()))

@app.route("/api/toplu")
def api_toplu():
    preset    = request.args.get("preset", "varsayilan")
    liste_str = request.args.get("liste", "")

    if liste_str:
        hisseler = [h.strip().upper() for h in liste_str.split(",") if h.strip()]
    elif preset == "bist30":   hisseler = BIST30
    elif preset == "bist50":   hisseler = BIST50
    elif preset == "bist100":  hisseler = list(dict.fromkeys(BIST50 + BIST100_EK))
    elif preset == "yildiz":   hisseler = YILDIZ_PAZAR
    else:                      hisseler = VARSAYILAN

    maks = int(request.args.get("maks", 50))
    hisseler = hisseler[:maks]

    sonuclar = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(hisse_analiz, s): s for s in hisseler}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                r = fut.result()
                if "hata" not in r:
                    sonuclar[s] = r
            except: pass

    return jsonify(sonuclar)

@app.route("/api/liste")
def api_liste():
    return jsonify({
        "varsayilan": VARSAYILAN,
        "bist30":     BIST30,
        "bist50":     BIST50,
        "bist100":    list(dict.fromkeys(BIST50 + BIST100_EK)),
        "yildiz":     YILDIZ_PAZAR,
    })

# ── KAP ──────────────────────────────────────────────────────────────────────
@app.route("/api/kap/<sembol>")
def api_kap(sembol):
    return jsonify(kap_analiz(sembol.upper()))

# ── ML Model ─────────────────────────────────────────────────────────────────
@app.route("/api/ml/durum")
def api_ml_durum():
    with _model_lock:
        m = _evrensel_model
    if m is None:
        return jsonify({"durum": "Eğitilmemiş", "dogruluk": 0})
    return jsonify({
        "durum":          "Eğitilmiş",
        "dogruluk":       m.dogruluk,
        "egitim_tarihi":  m.egitim_tarihi,
        "ornek_sayisi":   m.ornek_sayisi,
        "ozellik_sayisi": len(m.cols_secili),
        "dosya":          str(MODEL_PATH),
        "hedef_gun":      m.hedef_gun,
        "esik":           m.esik,
        "metrikler":      m.metrikler,
        "mimari": {
            "K1": "Linear Regression (trend tahmini)",
            "K2": "Ridge Regression (fiyat tahmini)",
            "K3": "Random Forest 200 ağaç",
            "K4": "Gradient Boosting 200 iter",
            "K5": "ExtraTrees 150 ağaç",
            "Meta": "Logistic Meta-Learner",
        },
        "hedef_aciklama": f"{m.hedef_gun} günlük yön tahmini (±%{m.esik*100:.1f} eşiği)",
    })

@app.route("/api/ml/egit", methods=["POST"])
def api_ml_egit():
    def bg():
        try:
            makro_guncelle()
            model_egit_ve_kaydet()
        except Exception as e:
            print(f"  ❌ Eğitim hatası: {e}")
    threading.Thread(target=bg, daemon=True).start()
    return jsonify({"mesaj": "Eğitim başladı"})

# ── Backtest ──────────────────────────────────────────────────────────────────
@app.route("/api/backtest/<sembol>")
def api_backtest(sembol):
    try:
        ticker = yf.Ticker(f"{sembol.upper()}.IS")
        df = ticker.history(period="3y")
        if df.empty:
            return jsonify({"hata": "Veri yok"})
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        sonuc = walk_forward_backtest(df, sembol.upper())
        return jsonify(sonuc)
    except Exception as e:
        return jsonify({"hata": str(e)})

# ── Kelly Criterion ───────────────────────────────────────────────────────────
@app.route("/api/kelly/<sembol>")
def api_kelly(sembol):
    sonuc = hisse_analiz(sembol.upper())
    if "hata" in sonuc:
        return jsonify(sonuc)
    kelly = sonuc.get("kelly", {})
    return jsonify({**kelly,
        "sembol":    sembol.upper(),
        "karar":     sonuc["karar"],
        "birlesik":  sonuc["birlesik"],
        "risk_odul": sonuc["risk_odul"],
    })

# ── ARIMA ─────────────────────────────────────────────────────────────────────
@app.route("/api/arima/<sembol>")
def api_arima(sembol):
    try:
        ticker = yf.Ticker(f"{sembol.upper()}.IS")
        df = ticker.history(period="1y")
        if df.empty: return jsonify({"hata": "Veri yok"})
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return jsonify(arima_trend(df, ML_HEDEF_GUN))
    except Exception as e:
        return jsonify({"hata": str(e)})

# ── Makro ─────────────────────────────────────────────────────────────────────
@app.route("/api/makro")
def api_makro():
    makro_guncelle()
    return jsonify({k: {kk: vv for kk, vv in v.items() if kk != "seri"}
                    for k, v in _makro_cache.items()})

# ── Portföy ───────────────────────────────────────────────────────────────────
@app.route("/api/portfolyo", methods=["GET"])
def api_portfolyo_get():
    return jsonify(portfolyo_yukle())

@app.route("/api/portfolyo", methods=["POST"])
def api_portfolyo_post():
    portfolyo_kaydet(request.json or [])
    return jsonify({"ok": True})

@app.route("/api/portfolyo/ekle", methods=["POST"])
def api_portfolyo_ekle():
    p = portfolyo_yukle()
    yeni = request.json or {}
    if yeni.get("sembol"):
        # Zaten varsa güncelle
        p = [x for x in p if x.get("sembol") != yeni["sembol"]]
        p.append(yeni)
        portfolyo_kaydet(p)
    return jsonify({"ok": True, "toplam": len(p)})

@app.route("/api/portfolyo/sil/<sembol>", methods=["DELETE"])
def api_portfolyo_sil(sembol):
    p = [x for x in portfolyo_yukle() if x.get("sembol") != sembol.upper()]
    portfolyo_kaydet(p)
    return jsonify({"ok": True})

# ── Sinyal geçmişi ────────────────────────────────────────────────────────────
@app.route("/api/sinyal_gecmisi")
def api_sinyal_gecmisi():
    try:
        return jsonify(json.loads(SINYAL_FILE.read_text()) if SINYAL_FILE.exists() else [])
    except: return jsonify([])

# ── Cache ─────────────────────────────────────────────────────────────────────
@app.route("/api/cache/temizle", methods=["POST"])
def api_cache_temizle():
    with _cache_lock: _cache.clear()
    with _kap_lock:   _kap_cache.clear()
    return jsonify({"ok": True})

@app.route("/api/durum")
def api_durum():
    with _model_lock: m = _evrensel_model
    return jsonify({
        "cache_boyut":  len(_cache),
        "makro_sayi":   len(_makro_cache),
        "kap_cache":    len(_kap_cache),
        "model_var":    m is not None,
        "model_dogruluk": m.dogruluk if m else 0,
        "version":      "5.0",
    })


# ══════════════════════════════════════════════════════════════════════════════
#  BÖLÜM 17: BAŞLANGIÇ
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/ping")
def api_ping():
    return jsonify({"ok": True, "zaman": datetime.now().isoformat()})


def basla():
    global _evrensel_model
    print("\n" + "═"*60)
    print("  BIST Hisse Analiz Robotu v5.0")
    print("═"*60)

    # Makroyu arka planda güncelle
    threading.Thread(target=makro_guncelle, daemon=True).start()

    # Modeli yükle
    m = model_yukle()
    if m:
        with _model_lock:
            _evrensel_model = m
    else:
        print("  ⚠️  Kaydedilmiş model yok — 'Yeniden Eğit' ile eğitin")

    print("  🌐  http://localhost:5000")
    print("═"*60 + "\n")


if __name__ == "__main__":
    basla()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)



# ── Eski endpoint uyumluluk stub'ları ─────────────────────────────────────────
@app.route("/api/alarmlar", methods=["GET","POST","DELETE"])
def api_alarmlar_stub():
    return jsonify([])

@app.route("/api/alarmlar/temizle", methods=["POST","DELETE"])
def api_alarmlar_temizle_stub():
    return jsonify({"ok": True})

@app.route("/api/ml/finetune_toplu", methods=["POST","GET"])
def api_finetune_stub():
    return jsonify({"mesaj": "Fine-tune v5'te kaldırıldı"})

@app.route("/api/ml/hisseler")
def api_ml_hisseler_stub():
    return jsonify([])

@app.route("/api/ml/metrikler/<sembol>")
def api_ml_metrikler_stub(sembol):
    return jsonify({})

@app.route("/api/test/<path:alt>")
def api_test_stub(alt):
    return jsonify({})

@app.route("/api/rapor")
def api_rapor_stub():
    return jsonify({})
