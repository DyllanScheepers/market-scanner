"""
Market Scanner — futures-native, report-grade.
Run:   python market_scanner_report.py
Needs: pip install pandas numpy yfinance requests openpyxl reportlab matplotlib

Data (all free, all futures-native except DXY/DAX/FTSE which have no Yahoo futures feed):
  - COT: CFTC Public Reporting Environment (live, with disk cache fallback)
  - Prices: Yahoo continuous front-month futures. DXY = DX-Y.NYB, DAX = ^GDAXI, UK 100 = ^FTSE.
Output: scan_results.csv, scan_results.xlsx,
        "Weekly Outlook <date>.pdf" (full), "Weekly Outlook <date> (Lazy).pdf" (fast read),
        scan_state.json
"""
import datetime as dt
import hashlib
import io
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0

# ---------------------------------------------------------------------------
# Robustness layer (does not change research logic or report content)
# ---------------------------------------------------------------------------
SCRIPT_VERSION = "2026.10.09"
CACHE_DIR = Path(os.environ.get("SCANNER_CACHE_DIR", "scanner_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# How long cached data is considered fresh (hours). Override with env if needed.
PRICE_CACHE_HOURS = float(os.environ.get("SCANNER_PRICE_CACHE_HOURS", "12"))
COT_CACHE_HOURS = float(os.environ.get("SCANNER_COT_CACHE_HOURS", "6"))
NETWORK_RETRIES = int(os.environ.get("SCANNER_NETWORK_RETRIES", "4"))
NETWORK_TIMEOUT = int(os.environ.get("SCANNER_NETWORK_TIMEOUT", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            CACHE_DIR / f"scanner_{dt.date.today().isoformat()}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("scanner")


def _cache_path(kind: str, key: str = "") -> Path:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12] if key else "default"
    return CACHE_DIR / f"{kind}_{h}.pkl"


def _cache_is_fresh(path: Path, max_hours: float) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h <= max_hours


def _cache_load(path: Path):
    try:
        return pd.read_pickle(path)
    except Exception as e:
        log.warning("Cache read failed (%s): %s", path.name, e)
        return None


def _cache_save(path: Path, obj) -> None:
    try:
        pd.to_pickle(obj, path)
    except Exception as e:
        log.warning("Cache write failed (%s): %s", path.name, e)


def _retry(fn, *, retries=None, delay=2.0, label="request"):
    """Call fn() with retries + exponential backoff. Raises last exception."""
    retries = retries if retries is not None else NETWORK_RETRIES
    last = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < retries:
                wait = delay * attempt
                log.warning("%s failed (attempt %d/%d): %s — retrying in %.0fs",
                            label, attempt, retries, e, wait)
                time.sleep(wait)
    raise RuntimeError(f"{label} failed after {retries} attempts: {last}") from last


# ---------------------------------------------------------------------------
# Telegram alerts & delivery (optional — set env vars to enable)
#   TELEGRAM_BOT_TOKEN  — from @BotFather
#   TELEGRAM_CHAT_ID    — your chat/user id
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
# Max age (calendar days) of the latest COT report before we warn / flag stale.
# CFTC publishes weekly (usually Friday). >10 days almost always means holiday lag.
COT_STALE_DAYS = int(os.environ.get("SCANNER_COT_STALE_DAYS", "10"))
# If True, refuse to build the full report when COT is stale (still sends an alert).
COT_STALE_ABORT = os.environ.get("SCANNER_COT_STALE_ABORT", "0").strip() in {"1", "true", "yes"}


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def telegram_send_message(text: str, *, silent: bool = False) -> bool:
    """Send a plain-text Telegram message. Returns True on success."""
    if not telegram_enabled():
        log.debug("Telegram not configured — skip message")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram limit ~4096 chars; keep a safety margin
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text[:4000],
        "disable_notification": silent,
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("Telegram message failed: %s", e)
        return False


def telegram_send_document(path, caption: str = "") -> bool:
    """Send a file (PDF/xlsx) as a Telegram document. Returns True on success."""
    if not telegram_enabled():
        log.debug("Telegram not configured — skip document")
        return False
    path = Path(path)
    if not path.exists():
        log.warning("Telegram document missing: %s", path)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    try:
        with open(path, "rb") as f:
            r = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": (caption or "")[:1024]},
                files={"document": (path.name, f)},
                timeout=120,
            )
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("Telegram document failed (%s): %s", path.name, e)
        return False


def alert_failure(title: str, detail: str = "") -> None:
    """Log + Telegram notify on hard failures so the weekly run is never silent."""
    msg = f"🚨 Market Scanner FAILED\n{title}"
    if detail:
        msg += f"\n\n{detail[:1500]}"
    msg += f"\n\nversion {SCRIPT_VERSION} · {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    log.error("%s — %s", title, detail)
    telegram_send_message(msg)


def check_cot_freshness(cot_df, prior_state=None):
    """Return (latest_date, status, message).

    status: 'ok' | 'stale' | 'unchanged' | 'unknown'
    - stale: latest COT date is older than COT_STALE_DAYS (holiday / delayed publish)
    - unchanged: same COT date as the previous successful run in scan_state.json
    """
    if cot_df is None or len(cot_df) == 0 or "date" not in cot_df.columns:
        return None, "unknown", "No COT dates available"
    try:
        latest = pd.to_datetime(cot_df["date"]).max().date()
    except Exception:
        return None, "unknown", "Could not parse COT dates"
    today = dt.date.today()
    age = (today - latest).days
    prior_cot = None
    if isinstance(prior_state, dict):
        prior_cot = prior_state.get("cot_date")
        if prior_cot:
            try:
                prior_cot = dt.date.fromisoformat(str(prior_cot)[:10])
            except Exception:
                prior_cot = None

    if age > COT_STALE_DAYS:
        msg = (f"CFTC report is {age} days old (latest {latest}). "
               f"Likely a holiday lag or delayed publish — treat positioning with caution.")
        return latest, "stale", msg
    if prior_cot is not None and latest == prior_cot:
        msg = (f"CFTC has not published a new report since {latest} "
               f"(same as last run). Common after long weekends/holidays.")
        return latest, "unchanged", msg
    return latest, "ok", f"CFTC report date {latest} ({age}d ago) looks current"


# ============================== SAFE COERCION ==============================
def _safe_float(x):
    if x is None:
        return np.nan
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"", "-", "n/a", "na", "none", "null", "nan"}:
            return np.nan
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _fmt(v, digits=2, suffix=""):
    try:
        x = float(v)
        return "-" if not np.isfinite(x) else f"{x:.{digits}f}{suffix}"
    except Exception:
        return "-"


# ============================== SETTINGS ==============================
COT_LOOKBACK_SHORT = 26
COT_LOOKBACKS = {"6M": 26, "1Y": 52, "2Y": 104, "3Y": 156, "5Y": 260}
COT_EDGE_HORIZONS = (4, 8)
COT_EDGE_MIN_SAMPLES = 6
EXTREME_HI = 80
EXTREME_LO = 20
TURN_WEEKS = 4
TURN_MIN_CHANGE = 0
DIV_WEEKS = 4
DIV_PRICE_LOOKBACK = 26
INDICES_USE_RETAIL = False

ROC_SHORT = 10
ROC_SHORT_INDICES = 13
ROC_LONG = 30
RESCALE_LEN = 100
VAL_UPPER = 75
VAL_LOWER = -75
VAL_PRIMARY = "long"

SHAPE_CLOSE = 0.50
SHAPE_LOOSE = 0.20

SEASON_PRIMARY_YEARS = 15
SEASON_MAX_YEARS = 25
SEASON_PROJECT_DAYS = 60
SEASON_TRIM_PCT = 10
SEASON_SMOOTH_LEN = 3
SEASON_MIN_MOVE_PCT = 0.75
SEASON_MIN_RELIABILITY = 65.0
SEASON_AGREEMENT_THRESHOLD = 0.25
SEASON_NEAREST_DAYS = 2

SEASON_STRONG_WIN = 0.86
SEASON_STRONG_AVG = 0.30
SEASON_MODERATE_WIN = 0.73
SEASON_MODERATE_AVG = 0.20

REPORT_BRAND = ""
REPORT_MAX_CHAPTERS = 15
BUILD_LAZY_PDF = False
REPORT_ALWAYS_CHAPTER = {"DXY", "Gold", "Silver", "Platinum", "WTI Crude", "Natural Gas", "EURUSD", "GBPUSD", "USDJPY", "Corn", "Soybeans", "Coffee"}
REPORT_FORCE_MARKETS = []
REPORT_EDGE_MIN_CASES = 6
REPORT_EDGE_MIN_WIN = 60
REPORT_SEASON_YEARS = 15
REPORT_SEASON_BEHIND_DAYS = 14
REPORT_SEASON_AHEAD_DAYS = 56

CHART_ZOOM = {"M": 60, "W": 150, "D": 300}
CHART_RIGHT_PAD_DAYS = 70
CHART_RIGHT_PAD = {"M": 420, "W": 180, "D": 70}

STATE_FILE = "scan_state.json"


# ============================== VALUATION REFERENCES ==============================
REFS = {"RATES": "ZB=F", "GOLD": "GC=F", "SILVER": "SI=F", "DXY": "DX-Y.NYB"}

REFS_BY_GROUP = {
    "Indices": ["RATES"],
    "FX": ["DXY"],
    "Metals": ["GOLD", "DXY"],
    "Energies": ["DXY", "GOLD"],
    "Agriculturals": ["DXY"],
    "Dollar": ["RATES", "GOLD"],
}
REFS_DISPLAY_ONLY = {"FX": ["RATES", "GOLD"]}


# ============================== MARKETS ==============================
MARKETS = [
    ("EURUSD", "FX", "6E=F",  "099741", False),
    ("GBPUSD", "FX", "6B=F",  "096742", False),
    ("AUDUSD", "FX", "6A=F",  "232741", False),
    ("NZDUSD", "FX", "6N=F",  "112741", False),
    ("USDJPY", "FX", "6J=F",  "097741", True),
    ("USDCHF", "FX", "6S=F",  "092741", True),
    ("USDCAD", "FX", "6C=F",  "090741", True),
    ("DXY", "Dollar", "DX-Y.NYB", "098662", False),
    ("Gold",      "Metals", "GC=F", "088691", False),
    ("Silver",    "Metals", "SI=F", "084691", False),
    ("Platinum",  "Metals", "PL=F", "076651", False),
    ("Palladium", "Metals", "PA=F", "075651", False),
    ("Copper",    "Metals", "HG=F", "085692", False),
    ("WTI Crude",   "Energies", "CL=F", "067651", False),
    ("Natural Gas", "Energies", "NG=F", "023651", False),
    ("Heating Oil", "Energies", "HO=F", "022651", False),
    ("Corn",            "Agriculturals", "ZC=F", "002602", False),
    ("Wheat (Chicago)", "Agriculturals", "ZW=F", "001602", False),
    ("Wheat (KC)",      "Agriculturals", "KE=F", "001612", False),
    ("Soybeans",        "Agriculturals", "ZS=F", "005602", False),
    ("Soybean Oil",     "Agriculturals", "ZL=F", "007601", False),
    ("Sugar",           "Agriculturals", "SB=F", "080732", False),
    ("Coffee",          "Agriculturals", "KC=F", "083731", False),
    ("Cotton",          "Agriculturals", "CT=F", "033661", False),
    ("Cocoa",           "Agriculturals", "CC=F", "073732", False),
    ("S&P 500",      "Indices", "ES=F",  "13874A", False),
    ("Nasdaq 100",   "Indices", "NQ=F",  "209742", False),
    ("Dow Jones",    "Indices", "YM=F",  "124603", False),
    ("Russell 2000", "Indices", "RTY=F", "239742", False),
    ("Nikkei 225",   "Indices", "NKD=F", None,     False),
    ("DAX",    "Indices", "^GDAXI", None, False),
    ("UK 100", "Indices", "^FTSE",  None, False),
]

CORE = {"Gold", "Silver", "Platinum", "WTI Crude", "Natural Gas", "EURUSD", "GBPUSD",
        "USDJPY", "DXY", "Corn", "Soybeans", "Coffee"}

SECTION_ORDER = ["Majors", "Indices", "Precious Metals", "Energies", "Agriculturals"]
_PRECIOUS = {"Gold", "Silver", "Platinum", "Palladium"}
MARKET_INDEX = {m[0]: i for i, m in enumerate(MARKETS)}


def _rpt_section(name, group):
    if group in ("FX", "Dollar"):
        return "Majors"
    if group == "Indices":
        return "Indices"
    if group == "Metals":
        return "Precious Metals"
    if group == "Energies":
        return "Energies"
    return "Agriculturals"


def _rpt_sort_key(name, group):
    sec = _rpt_section(name, group)
    return (SECTION_ORDER.index(sec), MARKET_INDEX.get(name, 999))


CHART_FUTURES_LABEL = {
    "EURUSD": "Euro FX futures (CME 6E)", "GBPUSD": "British Pound futures (CME 6B)",
    "AUDUSD": "Australian Dollar futures (CME 6A)", "NZDUSD": "NZ Dollar futures (CME 6N)",
    "USDJPY": "Japanese Yen futures (CME 6J), inverted to USDJPY",
    "USDCHF": "Swiss Franc futures (CME 6S), inverted to USDCHF",
    "USDCAD": "Canadian Dollar futures (CME 6C), inverted to USDCAD",
    "DXY": "ICE US Dollar Index (cash, no free futures feed on Yahoo)",
    "Gold": "Gold futures (COMEX GC)", "Silver": "Silver futures (COMEX SI)",
    "Platinum": "Platinum futures (NYMEX PL)", "Palladium": "Palladium futures (NYMEX PA)",
    "Copper": "Copper futures (COMEX HG)", "WTI Crude": "WTI Crude futures (NYMEX CL)",
    "Natural Gas": "Natural Gas futures (NYMEX NG)", "Heating Oil": "Heating Oil / ULSD futures (NYMEX HO)",
    "Corn": "Corn futures (CBOT ZC)", "Wheat (Chicago)": "Chicago Wheat futures (CBOT ZW)",
    "Wheat (KC)": "KC Wheat futures (CBOT KE)", "Soybeans": "Soybean futures (CBOT ZS)",
    "Soybean Oil": "Soybean Oil futures (CBOT ZL)", "Sugar": "Sugar #11 futures (ICE SB)",
    "Coffee": "Coffee C futures (ICE KC)", "Cotton": "Cotton #2 futures (ICE CT)",
    "Cocoa": "Cocoa futures (ICE CC)", "S&P 500": "E-mini S&P 500 futures (CME ES)",
    "Nasdaq 100": "E-mini Nasdaq-100 futures (CME NQ)", "Dow Jones": "E-mini Dow futures (CBOT YM)",
    "Russell 2000": "E-mini Russell 2000 futures (CME RTY)", "Nikkei 225": "Nikkei 225 futures (CME NKD)",
    "DAX": "cash index (no free futures feed on Yahoo)",
    "UK 100": "cash index (no free futures feed on Yahoo)",
}

ZONE_DRAW_MODE = "Preferred only"
ZONE_EXTEND_BARS = 150
SD = dict(legInBodyPct=0.40, baseBodyPct=0.40, legOutBodyPct=0.55, maxBase=8, legInFirst=False,
          absorbSmall=True, smallCandleMult=1.0, minStickOutPct=0.30, maxZoneVsLegOut=1.25,
          breakoutClearanceATR=0.25, avgLen=10, allowRejectionBase=True, rejectionWickPct=0.30,
          allowGapZones=True, minGapATR=0.50, displacementMultiple=1.5, skipTestBars=2, maxZonesKept=200)


def compute_sd_zones(df, mode=None):
    mode = mode or ZONE_DRAW_MODE
    P = SD
    o = df["Open"].values.astype(float); h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float); c = df["Close"].values.astype(float)
    n = len(o)
    rng = h - l
    body = np.where(rng > 0, np.abs(c - o) / np.where(rng > 0, rng, 1.0), 0.0)
    avg = pd.Series(rng).rolling(P["avgLen"]).mean().values
    idx = df.index

    def is_base(j, avg_range, demand_cand, base_avg_h):
        cr = rng[j]; bp = body[j]
        bhi = max(o[j], c[j]); blo = min(o[j], c[j])
        lower = (blo - l[j]) / cr if cr > 0 else 0.0
        upper = (h[j] - bhi) / cr if cr > 0 else 0.0
        normal = bp <= P["baseBodyPct"]
        small = P["absorbSmall"] and cr <= P["smallCandleMult"] * max(base_avg_h, avg_range * 0.6)
        absorb = small and bp >= P["legInBodyPct"]
        rej = P["allowRejectionBase"] and (lower if demand_cand else upper) >= P["rejectionWickPct"]
        return bool(normal or absorb or rej)

    pending, zones = [], []
    for i in range(n):
        if i > P["avgLen"] + P["maxBase"] + 3:
            leg_rng = rng[i]
            prior = avg[i - 1] if np.isfinite(avg[i - 1]) else leg_rng
            bull = c[i] > o[i] and body[i] >= P["legOutBodyPct"]
            bear = c[i] < o[i] and body[i] >= P["legOutBodyPct"]
            gap_up_q = P["allowGapZones"] and l[i] > h[i - 1]
            guess = bull or gap_up_q
            base_len, still, run_sum = 0, True, 0.0
            for k in range(1, P["maxBase"] + 1):
                if not still:
                    break
                j = i - k
                cr = rng[j]
                prov = run_sum / base_len if base_len > 0 else cr
                if is_base(j, prior, guess, prov):
                    base_len = k; run_sum += cr
                else:
                    still = False
            if base_len >= 1:
                seg = slice(i - base_len, i)
                base_high = h[seg].max(); base_low = l[seg].min()
                body_high = np.maximum(o[seg], c[seg]).max(); body_low = np.minimum(o[seg], c[seg]).min()
                base_height = base_high - base_low
                li = i - base_len - 1
                leg_in_bull = c[li] > o[li]
                leg_in_ok = (not P["legInFirst"]) or body[li] >= P["legInBodyPct"]
                zone_ok = base_height <= P["maxZoneVsLegOut"] * leg_rng
                stick_bull = (c[i] - base_high) / max(leg_rng, 0.0001) if bull else 0.0
                stick_bear = (base_low - c[i]) / max(leg_rng, 0.0001) if bear else 0.0
                stick_ok = (bull and stick_bull >= P["minStickOutPct"]) or (bear and stick_bear >= P["minStickOutPct"])
                clear_bull = bull and c[i] > base_high + P["breakoutClearanceATR"] * prior
                clear_bear = bear and c[i] < base_low - P["breakoutClearanceATR"] * prior
                strong_bull = bull and zone_ok and (stick_ok or clear_bull)
                strong_bear = bear and zone_ok and (stick_ok or clear_bear)
                gap_up = P["allowGapZones"] and l[i] > base_high + P["minGapATR"] * prior
                gap_dn = P["allowGapZones"] and h[i] < base_low - P["minGapATR"] * prior
                above = strong_bull or gap_up
                below = strong_bear or gap_dn
                was_gap = (above and gap_up) or (below and gap_dn)
                if leg_in_ok and (above or below):
                    demand = bool(above)
                    tag = " • Gap" if was_gap else ""
                    if demand:
                        wide, bprox, dist = base_high, body_high, base_low
                        setup = ("RBR Demand" if leg_in_bull else "DBR Demand") + tag
                    else:
                        wide, bprox, dist = base_low, body_low, base_high
                        setup = ("RBD Supply" if leg_in_bull else "DBD Supply") + tag
                    size = abs(wide - dist)
                    target = wide + P["displacementMultiple"] * size if demand else wide - P["displacementMultiple"] * size
                    pending.append(dict(demand=demand, wide=wide, body=bprox, distal=dist, setup=setup,
                                        left=i - base_len, right=i, target=target))
        for p in range(len(pending) - 1, -1, -1):
            pz = pending[p]
            reached = h[i] >= pz["target"] if pz["demand"] else l[i] <= pz["target"]
            failed = l[i] < pz["distal"] if pz["demand"] else h[i] > pz["distal"]
            if reached:
                eff = pz["body"] if mode == "Preferred only" else pz["wide"]
                top = eff if pz["demand"] else pz["distal"]
                bot = pz["distal"] if pz["demand"] else eff
                zones.append(dict(
                    demand=pz["demand"], top=top, bottom=bot, body=pz["body"], distal=pz["distal"],
                    setup=pz["setup"], left=pz["left"], right=pz["right"], born=i,
                    left_date=idx[pz["left"]], right_date=idx[pz["right"]], born_date=idx[i],
                    tested=False, mitigated=False))
                pending.pop(p)
                while len(zones) > P["maxZonesKept"]:
                    zones.pop(0)
            elif failed:
                pending.pop(p)
        for z in zones:
            if z["mitigated"]:
                continue
            can_test = i > z["born"] + P["skipTestBars"]
            entered = l[i] <= z["body"] if z["demand"] else h[i] >= z["body"]
            if can_test and entered and not z["tested"]:
                z["tested"] = True
            invalid = l[i] < z["distal"] if z["demand"] else h[i] > z["distal"]
            if i > z["born"] and invalid:
                z["mitigated"] = True
    return [z for z in zones if not z["mitigated"]]


# ============================== COT API ==============================
CFTC_API_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
CFTC_API_LOOKBACK_DAYS = 16 * 365
CFTC_API_PAGE_SIZE = 1000
CFTC_API_RETRIES = NETWORK_RETRIES

COT_NAME_ALIASES = {
    "EURUSD": ["EURO FX"], "GBPUSD": ["BRITISH POUND", "BRITISH POUND STERLING"],
    "AUDUSD": ["AUSTRALIAN DOLLAR"], "NZDUSD": ["NZ DOLLAR", "NEW ZEALAND DOLLAR"],
    "USDJPY": ["JAPANESE YEN"], "USDCHF": ["SWISS FRANC"], "USDCAD": ["CANADIAN DOLLAR"],
    "DXY": ["USD INDEX", "U.S. DOLLAR INDEX", "US DOLLAR INDEX", "DOLLAR INDEX"],
    "Gold": ["GOLD"], "Silver": ["SILVER"], "Platinum": ["PLATINUM"], "Palladium": ["PALLADIUM"],
    "Copper": ["COPPER"],
    "WTI Crude": ["WTI CRUDE OIL", "WTI CRUDE", "CRUDE OIL"],
    "Natural Gas": ["NATURAL GAS"],
    "Heating Oil": ["NY HARB ULSD", "NY HARBOR ULSD", "HEATING OIL", "ULSD"],
    "Corn": ["CORN"],
    "Wheat (Chicago)": ["WHEAT-SRW", "WHEAT FUTURE(CBT)", "WHEAT"],
    "Wheat (KC)": ["WHEAT-HRW", "KC HRW WHEAT", "HARD RED WINTER WHEAT", "KC WHEAT"],
    "Soybeans": ["SOYBEANS"], "Soybean Oil": ["SOYBEAN OIL"],
    "Sugar": ["SUGAR NO. 11", "SUGAR #11", "SUGAR 11", "SUGAR #11 (WORLD)"],
    "Coffee": ["COFFEE C", "COFFEE 'C'"],
    "Cotton": ["COTTON NO. 2", "COTTON NO.2", "COTTON"],
    "Cocoa": ["COCOA"],
    "S&P 500": ["E-MINI S&P 500", "S&P 500"],
    "Nasdaq 100": ["NASDAQ-100 STOCK INDEX (MINI)", "E-MINI NASDAQ-100", "NASDAQ-100", "NASDAQ MINI"],
    "Dow Jones": ["DJIA X $5", "DOW JONES INDUSTRIAL AVG- X $5", "DOW JONES INDUSTRIAL AVERAGE", "DJIA", "DOW JONES"],
    "Russell 2000": ["RUSSELL 2000 STOCK INDEX (MINI)", "E-MINI RUSSELL 2000", "RUSSELL E-MINI", "RUSSELL 2000"],
}

COT_EXPECTED_CODES = {
    "EURUSD": "099741", "GBPUSD": "096742", "AUDUSD": "232741", "NZDUSD": "112741",
    "USDJPY": "097741", "USDCHF": "092741", "USDCAD": "090741", "DXY": "098662",
    "Gold": "088691", "Silver": "084691", "Platinum": "076651", "Palladium": "075651",
    "Copper": "085692", "WTI Crude": "067651", "Natural Gas": "023651", "Heating Oil": "022651",
    "Corn": "002602", "Wheat (Chicago)": "001602",
    "Wheat (KC)": "001612", "Soybeans": "005602", "Soybean Oil": "007601",
    "Sugar": "080732", "Coffee": "083731", "Cotton": "033661", "Cocoa": "073732",
    "S&P 500": "13874A", "Nasdaq 100": "209742", "Dow Jones": "124603", "Russell 2000": "239742",
}


def _api_get(params):
    def _do():
        r = requests.get(
            CFTC_API_URL, params=params,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36",
                     "Accept": "application/json"},
            timeout=NETWORK_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    return _retry(_do, retries=CFTC_API_RETRIES, delay=2.0, label="CFTC API")


def _api_rows_for_code(code, date_from):
    safe_code = str(code).strip().replace("'", "''")
    where = (f"report_date_as_yyyy_mm_dd >= '{date_from}' "
             f"AND cftc_contract_market_code = '{safe_code}'")
    return _api_get({
        "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,contract_market_name,open_interest_all,comm_positions_long_all,comm_positions_short_all,noncomm_positions_long_all,noncomm_positions_short_all,nonrept_positions_long_all,nonrept_positions_short_all",
        "$where": where, "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": str(CFTC_API_PAGE_SIZE),
    })


def _api_rows_for_alias(alias, date_from):
    where = (f"report_date_as_yyyy_mm_dd >= '{date_from}' "
             f"AND upper(contract_market_name) like '%{alias.upper().replace(chr(39), chr(39)*2)}%'")
    return _api_get({
        "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,contract_market_name,open_interest_all,comm_positions_long_all,comm_positions_short_all,noncomm_positions_long_all,noncomm_positions_short_all,nonrept_positions_long_all,nonrept_positions_short_all",
        "$where": where, "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": str(CFTC_API_PAGE_SIZE),
    })


def load_cot(years=None, use_cache=True):
    """Download current CFTC COT data. Uses disk cache with TTL; falls back to
    last-known-good cache if the live API is completely unavailable."""
    cache_p = _cache_path("cot", dt.date.today().isoformat())
    stale_p = _cache_path("cot", "last_good")

    if use_cache and _cache_is_fresh(cache_p, COT_CACHE_HOURS):
        cached = _cache_load(cache_p)
        if cached is not None and len(cached) > 0:
            log.info("COT: using fresh cache (%s rows)", f"{len(cached):,}")
            return cached

    date_from = (dt.date.today() - dt.timedelta(days=CFTC_API_LOOKBACK_DAYS)).isoformat()
    rows = []; resolved = {}; failures = []
    log.info("Loading CURRENT CFTC COT data via Public Reporting API ...")
    for market_name, aliases in COT_NAME_ALIASES.items():
        market_rows = []
        expected_code = COT_EXPECTED_CODES.get(market_name)
        if expected_code:
            try:
                got = _api_rows_for_code(expected_code, date_from)
                if got:
                    market_rows = got
            except Exception as e:
                failures.append((market_name, str(e)))
        if not market_rows:
            for alias in aliases:
                try:
                    got = _api_rows_for_alias(alias, date_from)
                    if not got:
                        continue
                    if expected_code:
                        cm = [r for r in got if str(r.get("cftc_contract_market_code", "")).strip() == expected_code]
                        if cm:
                            market_rows = cm
                            break
                    market_rows = got
                    break
                except Exception as e:
                    failures.append((market_name, str(e)))
                    continue
        if not market_rows:
            log.warning("%s: live CFTC rows not found", market_name)
            continue
        if expected_code:
            exact = [r for r in market_rows if str(r.get("cftc_contract_market_code", "")).strip() == expected_code]
            if exact:
                market_rows = exact
        names = {}
        for r in market_rows:
            nm = r.get("contract_market_name")
            if nm:
                names[nm] = names.get(nm, 0) + 1
        chosen_name = max(names, key=names.get) if names else (aliases[0] if aliases else market_name)
        actual_codes = [str(r.get("cftc_contract_market_code", "")).strip() for r in market_rows if r.get("cftc_contract_market_code")]
        actual_code = actual_codes[0] if actual_codes else expected_code
        if expected_code and actual_code != expected_code:
            log.warning("%s: expected code %s, got %s; skipping", market_name, expected_code, actual_code)
            continue
        if actual_code:
            resolved[market_name] = actual_code.lstrip("0") or "0"
            log.info("CFTC %s: %s -> code %s (%d rows)", market_name, chosen_name, actual_code, len(market_rows))
        for r in market_rows:
            rows.append({
                "date": r.get("report_date_as_yyyy_mm_dd"),
                "code": str(r.get("cftc_contract_market_code", "")).strip().lstrip("0"),
                "market_name": r.get("contract_market_name", ""),
                "oi": r.get("open_interest_all"),
                "comm_long": r.get("comm_positions_long_all"), "comm_short": r.get("comm_positions_short_all"),
                "spec_long": r.get("noncomm_positions_long_all"), "spec_short": r.get("noncomm_positions_short_all"),
                "ret_long": r.get("nonrept_positions_long_all"), "ret_short": r.get("nonrept_positions_short_all"),
            })
    if not rows:
        # Fall back to last-known-good cache rather than hard-failing the whole run
        stale = _cache_load(stale_p)
        if stale is not None and len(stale) > 0:
            log.error("Live COT download failed; using last-known-good cache (%s rows). "
                      "Report will still build but COT may be stale.", f"{len(stale):,}")
            return stale
        detail = failures[-1][1] if failures else "no live CFTC rows"
        raise RuntimeError("CURRENT CFTC COT DATA COULD NOT BE DOWNLOADED. " + detail)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.strip().str.lstrip("0")
    for c in ["oi", "comm_long", "comm_short", "spec_long", "spec_short", "ret_long", "ret_short"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["date", "code"]).drop_duplicates(["code", "date"])
            .sort_values(["code", "date"]).reset_index(drop=True))
    df.attrs["resolved_codes"] = resolved
    log.info("CFTC API: downloaded %s rows for %d matched markets", f"{len(df):,}", len(resolved))
    missing = [m for m in COT_EXPECTED_CODES if m not in resolved]
    if missing:
        log.warning("COT markets not matched this run: %s", ", ".join(missing))
    _cache_save(cache_p, df)
    _cache_save(stale_p, df)  # always keep a last-good snapshot
    return df


# ============================== FUTURES PRICE DATA ==============================
def get_market_data(markets=None, use_cache=True):
    """Download futures OHLC via yfinance. Caches the inverted result to disk
    so a flaky Yahoo session does not block the weekly run."""
    import yfinance as yf
    markets = markets or MARKETS
    market_tickers = sorted({m[2] for m in markets if m[2]})
    ref_tickers = sorted({v for v in REFS.values()})
    all_tickers = sorted(set(market_tickers) | set(ref_tickers))
    cache_key = ",".join(all_tickers) + f"|{dt.date.today().isoformat()}"
    cache_p = _cache_path("prices", cache_key)
    stale_p = _cache_path("prices", "last_good")

    if use_cache and _cache_is_fresh(cache_p, PRICE_CACHE_HOURS):
        cached = _cache_load(cache_p)
        if isinstance(cached, dict) and "Close" in cached and not cached["Close"].empty:
            log.info("Prices: using fresh cache (%d tickers, %d bars)",
                     len(cached["Close"].columns), len(cached["Close"]))
            return cached

    def _download():
        raw = yf.download(
            all_tickers,
            start=(dt.date.today() - dt.timedelta(days=25 * 365 + 10)),
            interval="1d", auto_adjust=True, progress=False, group_by="column",
            threads=True,
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError("yfinance returned no data for any ticker.")
        return raw

    try:
        raw = _retry(_download, retries=NETWORK_RETRIES, delay=3.0, label="yfinance download")
    except Exception as e:
        stale = _cache_load(stale_p)
        if isinstance(stale, dict) and "Close" in stale and not stale["Close"].empty:
            log.error("Price download failed; using last-known-good cache. Error: %s", e)
            return stale
        raise

    ohlc = {}
    if isinstance(raw.columns, pd.MultiIndex):
        fields_present = set(raw.columns.get_level_values(0))
        for field in ("Open", "High", "Low", "Close"):
            if field not in fields_present:
                continue
            frame = raw[field]
            if isinstance(frame, pd.Series):
                frame = frame.to_frame(all_tickers[0])
            ohlc[field] = frame.copy()
    else:
        flat = set(raw.columns)
        price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if flat.issubset(price_fields):
            for field in ("Open", "High", "Low", "Close"):
                if field in raw.columns:
                    ohlc[field] = raw[[field]].rename(columns={field: all_tickers[0]})
        else:
            for field in ("Open", "High", "Low", "Close"):
                cols = {}
                for t in all_tickers:
                    if (field, t) in raw.columns:
                        cols[t] = raw[(field, t)]
                if cols:
                    ohlc[field] = pd.DataFrame(cols)

    if "Close" not in ohlc or ohlc["Close"].empty:
        stale = _cache_load(stale_p)
        if isinstance(stale, dict) and "Close" in stale and not stale["Close"].empty:
            log.error("Could not parse yfinance response; using last-known-good cache. Columns: %s",
                      list(raw.columns)[:8])
            return stale
        raise RuntimeError(f"Could not parse yfinance response. Columns: {list(raw.columns)[:8]}")

    for name, group, ticker, code, invert in markets:
        if not invert or ticker is None or ticker not in ohlc["Close"].columns:
            continue
        o = ohlc["Open"][ticker].copy(); h = ohlc["High"][ticker].copy()
        l = ohlc["Low"][ticker].copy();  c = ohlc["Close"][ticker].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ohlc["Open"][ticker]  = 1.0 / o
            ohlc["High"][ticker]  = 1.0 / l
            ohlc["Low"][ticker]   = 1.0 / h
            ohlc["Close"][ticker] = 1.0 / c

    result = {f: ohlc[f].replace([np.inf, -np.inf], np.nan) for f in ("Open", "High", "Low", "Close")}
    _cache_save(cache_p, result)
    _cache_save(stale_p, result)
    return result


def get_chart_data(ohlc=None):
    if ohlc is None:
        ohlc = get_market_data()
    out = {}
    for name, group, ticker, code, invert in MARKETS:
        if ticker is None or ticker not in ohlc["Close"].columns:
            continue
        df = pd.DataFrame({f: ohlc[f][ticker] for f in ("Open", "High", "Low", "Close")}).dropna()
        df = df[(df > 0).all(axis=1)]
        if len(df) < 60:
            continue
        df.index = pd.to_datetime(df.index)
        try:
            df.index = df.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        out[name] = {
            "df": df, "ticker": ticker, "inverted": invert,
            "desc": CHART_FUTURES_LABEL.get(name, "cash series (no futures feed)"),
            "futures": ticker.endswith("=F"),
            "zones": compute_sd_zones(df),
        }
    return out


# ============================== MATH ==============================
def roc(s, n):
    return 100 * (s - s.shift(n)) / s.shift(n)


def rescale(s, n):
    lo, hi = s.rolling(n).min(), s.rolling(n).max()
    out = (s - lo) / (hi - lo) * 200 - 100
    return out.where(hi != lo, 0)


def valuation(asset, ref, roc_len):
    ref = ref.reindex(asset.index, method="ffill")
    df = pd.concat([asset, ref], axis=1).dropna()
    if len(df) < roc_len + RESCALE_LEN:
        return np.nan
    rel = roc(df.iloc[:, 0], roc_len) - roc(df.iloc[:, 1], roc_len)
    v = rescale(rel, RESCALE_LEN).dropna()
    return float(v.iloc[-1]) if len(v) else np.nan


def val_signal(values):
    used = [v for v in values if not np.isnan(v)]
    n = len(used)
    if not n:
        return 0
    if sum(v <= VAL_LOWER for v in used) > n / 2:
        return 1
    if sum(v >= VAL_UPPER for v in used) > n / 2:
        return -1
    return 0
# ============================== COT SCORING ==============================
def spec_divergence(spec, weekly_px, invert):
    if weekly_px is None or len(spec) < DIV_WEEKS + 1:
        return "-"
    p = weekly_px.dropna()
    if len(p) < DIV_PRICE_LOOKBACK:
        return "-"
    if invert:
        p = 1.0 / p
    w = p.tail(DIV_PRICE_LOOKBACK).values
    bars_since_high = len(w) - 1 - int(np.argmax(w))
    bars_since_low = len(w) - 1 - int(np.argmin(w))
    chg = spec.iloc[-1] - spec.iloc[-1 - DIV_WEEKS]
    out = "-"
    if bars_since_high < DIV_WEEKS and chg <= 0:
        out = "Bearish"
    elif bars_since_low < DIV_WEEKS and chg >= 0:
        out = "Bullish"
    if invert and out != "-":
        out = "Bullish" if out == "Bearish" else "Bearish"
    return out


def _cot_index(series, weeks):
    x = pd.Series(series).dropna()
    if len(x) < max(4, weeks // 2):
        return np.nan
    w = x.tail(weeks)
    if len(w) < 4:
        return np.nan
    cur = float(w.iloc[-1])
    n_le = int((w <= cur).sum())
    return float(n_le) / float(len(w)) * 100.0


def _cot_extreme_direction(index_value):
    if not np.isfinite(index_value):
        return 0
    return 1 if index_value >= EXTREME_HI else -1 if index_value <= EXTREME_LO else 0


def _edge_stats_for_signal(cot_sub, weekly_px, lookback_weeks, signal_direction, invert=False, role="commercial"):
    empty = {}
    for h in COT_EDGE_HORIZONS:
        empty.update({f"Edge{h}Win": np.nan, f"Edge{h}Wins": 0, f"Edge{h}Total": 0,
                      f"Edge{h}AvgMove": np.nan, f"Edge{h}Samples": 0, f"Edge{h}AvgWeeksIn": np.nan})
    if weekly_px is None or len(weekly_px) < lookback_weeks + 10:
        return empty
    c = cot_sub.sort_values("date").copy().reset_index(drop=True)
    oi = c["oi"].replace(0, np.nan)
    if role == "commercial":
        net = c["comm_long"] - c["comm_short"]
    elif role == "retail":
        net = c["ret_long"] - c["ret_short"]
    else:
        net = c["spec_long"] - c["spec_short"]
    basis = net / oi * 100.0
    minp = max(4, lookback_weeks // 2)
    idx = basis.rolling(lookback_weeks, min_periods=minp).apply(
        lambda a: float((a <= a[-1]).sum()) / float(len(a)) * 100.0, raw=True)
    px = weekly_px.dropna().copy(); px.index = pd.to_datetime(px.index)
    dates = pd.to_datetime(c["date"])
    rows = []
    last_extreme = 0
    for i, iv in enumerate(idx):
        d = _cot_extreme_direction(iv)
        if d == 0:
            last_extreme = 0
            continue
        if d == last_extreme:
            continue
        last_extreme = d
        native_price = (d if role == "commercial" else -d)
        price_dir = -native_price if invert else native_price
        if price_dir != signal_direction:
            continue
        dt0 = dates.iloc[i]
        before = px[px.index <= dt0]
        if before.empty:
            continue
        pos0 = px.index.searchsorted(before.index[-1]); p0 = float(before.iloc[-1])
        rows.append((pos0, p0))
    for h in COT_EDGE_HORIZONS:
        moves = []; weeks_in = []
        for pos0, p0 in rows:
            j = pos0 + h
            if j >= len(px):
                continue
            move = (float(px.iloc[j]) - p0) / p0 * 100.0
            favorable = (px.iloc[pos0 + 1:j + 1] - p0) / p0 * 100.0 * signal_direction
            hit = np.where(favorable.values >= 0.5)[0]
            weeks_in.append(float(hit[0] + 1) if len(hit) else float(h))
            moves.append(move)
        if moves:
            wins = sum(m * signal_direction > 0 for m in moves); total = len(moves)
            empty[f"Edge{h}Win"] = wins / total * 100.0
            empty[f"Edge{h}Wins"] = int(wins); empty[f"Edge{h}Total"] = int(total)
            empty[f"Edge{h}AvgMove"] = float(np.mean(moves)); empty[f"Edge{h}Samples"] = int(total)
            empty[f"Edge{h}AvgWeeksIn"] = float(np.mean(weeks_in))
    return empty


def cot_metrics(sub, invert, weekly_px=None):
    sub = sub.sort_values("date").reset_index(drop=True)
    comm = sub["comm_long"] - sub["comm_short"]
    spec = sub["spec_long"] - sub["spec_short"]
    ret = sub["ret_long"] - sub["ret_short"]
    oi = sub["oi"].replace(0, np.nan)
    comm_basis = comm / oi * 100.0
    ret_basis = ret / oi * 100.0
    comm_idx = {k: _cot_index(comm_basis, w) for k, w in COT_LOOKBACKS.items()}
    ret_idx = {k: _cot_index(ret_basis, w) for k, w in COT_LOOKBACKS.items()}
    spec26 = _cot_index(spec, 26)

    comm_ext = {k: _cot_extreme_direction(v) for k, v in comm_idx.items()}
    ret_ext = {k: _cot_extreme_direction(v) for k, v in ret_idx.items()}
    comm_price = {k: (d if not invert else -d) for k, d in comm_ext.items() if d}
    ret_price = {k: (-d if not invert else d) for k, d in ret_ext.items() if d}
    pos = sum(v == 1 for v in comm_price.values()); neg = sum(v == -1 for v in comm_price.values())
    comm_direction = 1 if pos > neg else -1 if neg > pos else 0
    comm_count = max(pos, neg)
    agree_layers = [k for k, v in comm_price.items() if v == comm_direction] if comm_direction else []

    horizon = "-"; comm_score = 0
    for k, label, weight in (("5Y", "Multi-year", 3), ("3Y", "3-year", 2), ("2Y", "2-year", 2), ("1Y", "1-year", 1), ("6M", "Six-month", 1)):
        if comm_price.get(k) == comm_direction and comm_direction:
            horizon = label; comm_score = weight * comm_direction; break
    if comm_score == 0:
        for k, label, weight in (("5Y", "Multi-year", 3), ("3Y", "3-year", 2), ("2Y", "2-year", 2), ("1Y", "1-year", 1), ("6M", "Six-month", 1)):
            if k in comm_price:
                horizon = label; comm_score = weight * comm_price[k]; break

    spec_dir = _cot_extreme_direction(spec26)
    spec_price = (-spec_dir if invert else spec_dir) if spec_dir else 0
    spec_score = 1 if spec_price == comm_direction and comm_direction else 0

    ret_direction = 0; ret_layer = "-"; ret_score = 0
    for k, weight in (("5Y", 2), ("3Y", 2), ("2Y", 1), ("1Y", 1), ("6M", 1)):
        if k in ret_price:
            ret_direction = ret_price[k]; ret_layer = k; ret_score = weight * ret_direction; break

    prev = _cot_index(comm_basis.iloc[:-TURN_WEEKS] if len(comm_basis) > TURN_WEEKS else comm_basis, 26)
    comm26 = comm_idx["6M"]
    change = comm26 - prev if np.isfinite(prev) and np.isfinite(comm26) else np.nan
    turning = bool(comm_direction and np.isfinite(change) and change * (-comm_direction) >= TURN_MIN_CHANGE)

    out = {
        "comm_score": int(comm_score), "spec_score": int(spec_score), "ret_score": int(ret_score),
        "comm26": comm_idx["6M"], "comm1y": comm_idx["1Y"], "comm2y": comm_idx["2Y"],
        "comm3y": comm_idx["3Y"], "comm5y": comm_idx["5Y"], "spec26": spec26,
        "ret26": ret_idx["6M"], "ret1y": ret_idx["1Y"], "ret2y": ret_idx["2Y"],
        "ret3y": ret_idx["3Y"], "ret5y": ret_idx["5Y"],
        "comm_extreme_count": comm_count, "ret_extreme_count": sum(bool(v) for v in ret_ext.values()),
        "comm_direction": comm_direction, "ret_direction": ret_direction, "ret_layer": ret_layer,
        "horizon": horizon, "turning": turning,
        "div": spec_divergence(spec, weekly_px, invert), "date": sub["date"].iloc[-1], "weeks": len(sub),
        "CommAgreeLayers": " / ".join(agree_layers) if agree_layers else "-",
        "CommAgreement": f"{len(agree_layers)} of {len(COT_LOOKBACKS)}" if agree_layers else "0 of 5",
        "RetailAgreeLayers": " / ".join(k for k, v in ret_price.items() if v == ret_direction) if ret_direction else "-",
    }
    for layer, w in COT_LOOKBACKS.items():
        cd = comm_price.get(layer, 0); rd = ret_price.get(layer, 0)
        if cd:
            out[f"{layer}_EdgeDirection"] = "Bullish" if cd == 1 else "Bearish"
            out.update({f"{layer}_{k}": v for k, v in _edge_stats_for_signal(sub, weekly_px, w, cd, invert, role="commercial").items()})
        else:
            out[f"{layer}_EdgeDirection"] = "-"
            for h in COT_EDGE_HORIZONS:
                for suffix, val in (("Win", np.nan), ("Wins", 0), ("Total", 0), ("AvgMove", np.nan), ("Samples", 0), ("AvgWeeksIn", np.nan)):
                    out[f"{layer}_Edge{h}{suffix}"] = val
        if rd:
            out[f"{layer}_RetailEdgeDirection"] = "Bullish" if rd == 1 else "Bearish"
            out.update({f"{layer}_RetailEdge{k}": v for k, v in _edge_stats_for_signal(sub, weekly_px, w, rd, invert, role="retail").items()})
        else:
            out[f"{layer}_RetailEdgeDirection"] = "-"
            for h in COT_EDGE_HORIZONS:
                for suffix, val in (("Win", np.nan), ("Wins", 0), ("Total", 0), ("AvgMove", np.nan), ("Samples", 0), ("AvgWeeksIn", np.nan)):
                    out[f"{layer}_RetailEdge{h}{suffix}"] = val
    return out


def combine(cot_bias, val_bias, season_bias="Neutral", season_conflict=False,
            season_window_conflict=False, season_strength=None, season_in_play=False):
    if cot_bias in ("Bullish", "Bearish"):
        parts = [f"COT-led {cot_bias}"]
        if val_bias == cot_bias: parts.append("valuation aligned")
        elif val_bias in ("Bullish", "Bearish"): parts.append(f"valuation {val_bias.lower()}")
        if season_bias == cot_bias: parts.append("seasonality confirms")
        elif season_conflict or season_window_conflict: parts.append("seasonality tempers conviction")
        return " • ".join(parts)
    if season_bias in ("Bullish", "Bearish") and season_strength == "STRONG":
        parts = [f"Seasonality-led {season_bias} (COT neutral)"]
        if season_in_play: parts.append("window live now")
        if val_bias == season_bias: parts.append("valuation aligned")
        elif val_bias in ("Bullish", "Bearish"): parts.append(f"valuation {val_bias.lower()}")
        return " • ".join(parts)
    if val_bias in ("Bullish", "Bearish"):
        return f"Valuation-only {val_bias} (no valid COT side)"
    return "Neutral"


def valuation_state(d, w):
    if w != 0:
        if d == w: return w, 2, "Strong W+D"
        if d == -w: return w, 2, "Strong W (D opposes)"
        return w, 2, "Strong W"
    if d != 0:
        return d, 1, "Daily"
    return 0, 0, "-"


def _season_nearest(series, idx, radius=SEASON_NEAREST_DAYS):
    if series is None or len(series) == 0 or idx is None or idx < 0:
        return np.nan
    if idx < len(series) and pd.notna(series.iloc[idx]):
        return float(series.iloc[idx])
    for k in range(1, radius + 1):
        if idx + k < len(series) and pd.notna(series.iloc[idx + k]):
            return float(series.iloc[idx + k])
        if idx - k >= 0 and pd.notna(series.iloc[idx - k]):
            return float(series.iloc[idx - k])
    return np.nan


def _trimmed_mean(values, pct=SEASON_TRIM_PCT):
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return np.nan
    vals.sort()
    cut = int(round(len(vals) * pct / 100.0))
    lo, hi = cut, len(vals) - cut
    if lo >= hi:
        return float(np.mean(vals))
    return float(np.mean(vals[lo:hi]))


def _smooth(values, length=SEASON_SMOOTH_LEN):
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0 or length <= 1:
        return vals
    half = length // 2
    out = np.full(len(vals), np.nan, dtype=float)
    for i in range(len(vals)):
        a, b = max(0, i - half), min(len(vals), i + half + 1)
        w = vals[a:b]
        if np.isfinite(w).any():
            out[i] = np.nanmean(w)
    return out


def _season_year_series(close, year):
    if close is None or close.empty:
        return pd.Series(dtype=float)
    x = close.dropna().copy()
    idx = pd.to_datetime(x.index)
    x.index = idx
    return x[x.index.year == year].sort_index()


def _season_next_trading_date(ts, all_dates):
    t = pd.Timestamp(ts).normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(all_dates)).normalize()
    pos = dates.searchsorted(t, side="right")
    return dates[pos] if pos < len(dates) else t + pd.Timedelta(days=1)


def _season_projection(close, lookback=SEASON_PRIMARY_YEARS, horizon=SEASON_PROJECT_DAYS,
                       trim_pct=SEASON_TRIM_PCT, smooth_len=SEASON_SMOOTH_LEN):
    x = close.dropna().copy()
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    if len(x) < 252 * 5:
        return None
    current_year = int(x.index[-1].year)
    current_year_data = _season_year_series(x, current_year)
    if current_year_data.empty:
        return None
    t_idx = len(current_year_data) - 1
    anchor_close = float(x.iloc[-1])
    anchor_date = pd.Timestamp(x.index[-1])
    years = {}
    for y in range(current_year - SEASON_MAX_YEARS, current_year + 2):
        ys = _season_year_series(x, y)
        if not ys.empty:
            years[y] = ys
    raw_returns = {h: [] for h in range(horizon + 1)}
    det_returns = {h: [] for h in range(horizon + 1)}
    samples = []
    for offset in range(1, min(lookback, SEASON_MAX_YEARS) + 1):
        y = current_year - offset
        ys = years.get(y)
        if ys is None or len(ys) < 2:
            continue
        p0 = _season_nearest(ys, t_idx)
        if not np.isfinite(p0) or p0 == 0:
            continue
        y_start = _season_nearest(ys, 0)
        y_end = _season_nearest(ys, len(ys) - 1)
        if not np.isfinite(y_start) or not np.isfinite(y_end) or y_start == 0:
            continue
        drift = ((y_end - y_start) / y_start * 100.0) / len(ys)
        samples.append((offset, y, ys, p0, drift))
        for h in range(horizon + 1):
            target_idx = t_idx + h
            target_year = y
            while target_year in years and target_idx >= len(years[target_year]):
                target_idx -= len(years[target_year])
                target_year += 1
            target_series = years.get(target_year)
            ph = _season_nearest(target_series, target_idx) if target_series is not None else np.nan
            if np.isfinite(ph):
                raw = (ph - p0) / p0 * 100.0
                raw_returns[h].append(raw)
                det_returns[h].append(raw - drift * h)
    if not samples:
        return None
    path_det = np.array([_trimmed_mean(det_returns[h], trim_pct) for h in range(horizon + 1)], dtype=float)
    coverage = np.array([len(det_returns[h]) for h in range(horizon + 1)], dtype=int)
    path_det = _smooth(path_det, smooth_len)
    finite = np.isfinite(path_det)
    if not finite.any():
        return None
    peak_h = int(np.nanargmax(path_det))
    trough_h = int(np.nanargmin(path_det))
    horizon_vals = det_returns[horizon]
    up_count = sum(1 for v in horizon_vals if v > 0)
    total_count = len(horizon_vals)
    reliability = (up_count / total_count * 100.0) if total_count else np.nan
    move = float(path_det[horizon]) if np.isfinite(path_det[horizon]) else np.nan
    if np.isfinite(move) and np.isfinite(reliability):
        if move > SEASON_MIN_MOVE_PCT and reliability >= SEASON_MIN_RELIABILITY:
            verdict = "Bullish"
        elif move < -SEASON_MIN_MOVE_PCT and reliability >= SEASON_MIN_RELIABILITY:
            verdict = "Bearish"
        else:
            verdict = "Neutral"
    else:
        verdict = "Neutral"
    return {"anchor_close": anchor_close, "anchor_date": anchor_date, "move": move, "reliability": reliability,
            "verdict": verdict, "peak_h": peak_h, "trough_h": trough_h,
            "avg_coverage": float(np.mean(coverage)), "min_coverage": int(np.min(coverage)), "samples": len(samples)}


def _season_window_stats(close, direction, center_date=None):
    x = close.dropna().copy(); x.index = pd.to_datetime(x.index); x = x.sort_index()
    empty = {"SeasonWindow": "-", "SeasonWindowDays": np.nan, "SeasonWindowShape": "-",
             "SeasonWindowReliability": np.nan, "SeasonWindowSamples": 0, "SeasonWindowWins": 0,
             "SeasonWindowTotal": 0, "SeasonWindowAvgMove": np.nan, "SeasonWindowDriver": "-",
             "SeasonWindowNeighbour": "-", "SeasonWindowWholeYear": "-",
             "SeasonWindowStrength": None, "SeasonWindowInPlay": False}
    if len(x) < 252 * 8 or direction not in (1, -1):
        return empty
    center = pd.Timestamp(center_date or x.index[-1]); years = sorted(set(x.index.year))
    candidates = []
    for start_off in range(-56, -13, 7):
        for end_off in range(14, 57, 7):
            if end_off - start_off < 21:
                continue
            vals = []
            for y in years:
                try:
                    sdate = pd.Timestamp(year=y, month=(center + pd.Timedelta(days=start_off)).month,
                                         day=min((center + pd.Timedelta(days=start_off)).day, 28))
                    edate = pd.Timestamp(year=y, month=(center + pd.Timedelta(days=end_off)).month,
                                         day=min((center + pd.Timedelta(days=end_off)).day, 28))
                except Exception:
                    continue
                if edate <= sdate:
                    edate += pd.DateOffset(years=1)
                yy = x[(x.index >= sdate) & (x.index <= edate)]
                if len(yy) < 2:
                    continue
                vals.append((float(yy.iloc[-1]) - float(yy.iloc[0])) / float(yy.iloc[0]) * 100.0)
            if len(vals) < 8:
                continue
            wins = sum(v * direction > 0 for v in vals); total = len(vals); rel = wins / total * 100.0
            trimmed = float(_trimmed_mean(vals, SEASON_TRIM_PCT)); avg = float(np.mean(vals))
            if rel < 60 or trimmed * direction <= 0:
                continue
            score = rel * 0.55 + min(abs(trimmed) * 12, 40) + min(abs(avg) * 8, 20) - max(0, (21 - (end_off - start_off)) * 0.25)
            candidates.append((score, start_off, end_off, rel, wins, total, avg, trimmed))
    if not candidates:
        return empty
    best = max(candidates, key=lambda z: z[0])
    _, so, eo, rel, wins, total, avg, trimmed = best
    s0 = center + pd.Timedelta(days=so); e0 = center + pd.Timedelta(days=eo)

    def hist_move(shift):
        vals = []
        for y in years:
            try:
                ss = s0 + pd.Timedelta(days=shift); ee = e0 + pd.Timedelta(days=shift)
                a = pd.Timestamp(year=y, month=ss.month, day=min(ss.day, 28))
                b = pd.Timestamp(year=y, month=ee.month, day=min(ee.day, 28))
                if b <= a:
                    b += pd.DateOffset(years=1)
                yy = x[(x.index >= a) & (x.index <= b)]
                if len(yy) >= 2:
                    vals.append((float(yy.iloc[-1]) - float(yy.iloc[0])) / float(yy.iloc[0]) * 100.0)
            except Exception:
                pass
        return _trimmed_mean(vals, SEASON_TRIM_PCT) if vals else np.nan

    left, right = hist_move(-7), hist_move(7)
    neighbour = "Pass" if all(np.isfinite(v) and abs(trimmed) >= abs(v) for v in (left, right)) else "Shift-sensitive"
    days = int((e0 - s0).days + 1)
    shape = "sharp" if days <= 21 else "broad"
    label = f"{s0.strftime('%d %b')} – {e0.strftime('%d %b')} · {days}d · {shape} · {rel:.0f}% · {wins} of {total} · {trimmed:+.1f}%"
    rec_vals = []
    for k in range(1, 5):
        y = center.year - k
        try:
            a = pd.Timestamp(year=y, month=s0.month, day=min(s0.day, 28))
            b = pd.Timestamp(year=y, month=e0.month, day=min(e0.day, 28))
            if b <= a:
                b += pd.DateOffset(years=1)
            yy = x[(x.index >= a) & (x.index <= b)]
            if len(yy) >= 2:
                rec_vals.append((float(yy.iloc[-1]) - float(yy.iloc[0])) / float(yy.iloc[0]) * 100.0)
        except Exception:
            pass
    rec_wins = sum(v * direction > 0 for v in rec_vals)
    rec_cases = len(rec_vals)
    rel_frac = rel / 100.0
    if (rel_frac >= SEASON_STRONG_WIN and abs(trimmed) >= SEASON_STRONG_AVG
            and (rec_cases == 0 or rec_wins >= max(1, rec_cases - 1))):
        strength = "STRONG"
    elif rel_frac >= SEASON_MODERATE_WIN and abs(trimmed) >= SEASON_MODERATE_AVG:
        strength = "MODERATE"
    else:
        strength = None
    in_play = bool(pd.Timestamp(center) >= s0 and pd.Timestamp(center) <= e0)
    return {**empty, "SeasonWindow": label, "SeasonWindowDays": days, "SeasonWindowShape": shape,
            "SeasonWindowReliability": rel, "SeasonWindowSamples": total, "SeasonWindowWins": wins,
            "SeasonWindowTotal": total, "SeasonWindowAvgMove": trimmed, "SeasonWindowDriver": "Window + whole-year",
            "SeasonWindowNeighbour": neighbour, "SeasonWindowWholeYear": "Pending",
            "SeasonWindowStrength": strength, "SeasonWindowInPlay": in_play}


def _season_volatility(close, lookback_days=20):
    x = close.dropna().copy(); x.index = pd.to_datetime(x.index); x = x.sort_index()
    if len(x) < 252 * 8:
        return {"SeasonVolCurrent": np.nan, "SeasonVol60D": np.nan, "SeasonVolSeasonalAvg": np.nan,
                "SeasonVolSeasonalPath": np.nan, "SeasonVolState": "n/a", "SeasonVolPercentile": np.nan,
                "SeasonVolYears": 0, "SeasonVolCoverage": np.nan, "SeasonVolSizing": "Normal"}
    rv = x.pct_change().rolling(lookback_days, min_periods=lookback_days).std() * np.sqrt(252) * 100.0
    current = float(rv.iloc[-1]) if np.isfinite(rv.iloc[-1]) else np.nan
    current_year = int(x.index[-1].year)
    cy = rv[rv.index.year == current_year].dropna()
    if cy.empty:
        return {"SeasonVolCurrent": current, "SeasonVol60D": np.nan, "SeasonVolSeasonalAvg": np.nan,
                "SeasonVolSeasonalPath": np.nan, "SeasonVolState": "n/a", "SeasonVolPercentile": np.nan,
                "SeasonVolYears": 0, "SeasonVolCoverage": np.nan, "SeasonVolSizing": "Normal"}
    t_idx = len(cy) - 1
    max_h = int(SEASON_PROJECT_DAYS)
    paths = []
    years = sorted(set(rv.index.year))
    for y in years:
        if y >= current_year:
            continue
        ys = rv[rv.index.year == y].dropna()
        if len(ys) < 180:
            continue
        path = []
        for h in range(max_h + 1):
            idx = t_idx + h
            yy = y
            while yy in years and idx >= len(rv[rv.index.year == yy].dropna()):
                idx -= len(rv[rv.index.year == yy].dropna()); yy += 1
            target = rv[rv.index.year == yy].dropna() if yy in years else pd.Series(dtype=float)
            path.append(float(target.iloc[idx]) if idx < len(target) else np.nan)
        paths.append(path)
    if not paths:
        return {"SeasonVolCurrent": current, "SeasonVol60D": np.nan, "SeasonVolSeasonalAvg": np.nan,
                "SeasonVolSeasonalPath": np.nan, "SeasonVolState": "n/a", "SeasonVolPercentile": np.nan,
                "SeasonVolYears": 0, "SeasonVolCoverage": np.nan, "SeasonVolSizing": "Normal"}
    arr = np.asarray(paths, dtype=float)
    seasonal_path = np.nanmean(arr, axis=0)
    coverage = np.sum(np.isfinite(arr), axis=0)
    seasonal_60 = float(np.nanmean(seasonal_path[:max_h + 1]))
    seasonal_now = float(seasonal_path[0]) if np.isfinite(seasonal_path[0]) else np.nan
    all_hist = arr[np.isfinite(arr)]
    pct = float(np.mean(all_hist <= current) * 100.0) if np.isfinite(current) and len(all_hist) else np.nan
    if np.isfinite(current) and np.isfinite(seasonal_now):
        state = "Loud" if current > seasonal_now * 1.25 else "Calm" if current < seasonal_now * 0.80 else "Normal"
    else:
        state = "n/a"
    sizing = {"Loud": "Reduce exposure / allow wider stop context",
              "Calm": "Normal exposure / cleaner volatility context",
              "Normal": "Normal exposure"}.get(state, "Normal")
    path_summary = "; ".join(f"{i}:{v:.2f}" for i, v in enumerate(seasonal_path[:max_h + 1]) if np.isfinite(v))
    return {"SeasonVolCurrent": current, "SeasonVol60D": seasonal_60, "SeasonVolSeasonalAvg": seasonal_now,
            "SeasonVolSeasonalPath": path_summary, "SeasonVolState": state, "SeasonVolPercentile": pct,
            "SeasonVolYears": len(paths), "SeasonVolCoverage": float(np.mean(coverage)), "SeasonVolSizing": sizing}


def seasonality_metrics(close):
    primary = _season_projection(close, SEASON_PRIMARY_YEARS)
    ten = _season_projection(close, 10)
    fifteen = _season_projection(close, 15)
    if primary is None:
        base = {"Seasonality": "n/a", "SeasonalityMove": np.nan, "SeasonalityReliability": np.nan,
                "SeasonalityYears": 0, "Seasonality10YMove": np.nan, "Seasonality15YMove": np.nan,
                "Seasonality10Y": "n/a", "Seasonality15Y": "n/a", "SeasonalityAgreement": "Insufficient Data",
                "SeasonalHighDays": np.nan, "SeasonalLowDays": np.nan, "SeasonalHighDate": None,
                "SeasonalLowDate": None, "SeasonalCoverage": np.nan, "SeasonalMinCoverage": np.nan,
                "SeasonalityConfluence": "n/a", "SeasonalityScore": 0,
                "SeasonWindow": "-", "SeasonWindowDays": np.nan, "SeasonWindowShape": "-",
                "SeasonWindowReliability": np.nan, "SeasonWindowSamples": 0, "SeasonWindowWins": 0,
                "SeasonWindowTotal": 0, "SeasonWindowAvgMove": np.nan, "SeasonWindowDriver": "-",
                "SeasonWindowNeighbour": "-", "SeasonWindowWholeYear": "-",
                "SeasonWindowStrength": None, "SeasonWindowInPlay": False}
        base.update(_season_window_stats(close, 0))
        base.update(_season_volatility(close))
        return base

    def sign(v):
        return 1 if np.isfinite(v) and v > SEASON_AGREEMENT_THRESHOLD else -1 if np.isfinite(v) and v < -SEASON_AGREEMENT_THRESHOLD else 0

    s10 = sign(ten["move"] if ten else np.nan)
    s15 = sign(fifteen["move"] if fifteen else np.nan)
    agreement = ("Aligned Bullish" if s10 == 1 and s15 == 1 else
                 "Aligned Bearish" if s10 == -1 and s15 == -1 else
                 "One Neutral" if s10 == 0 or s15 == 0 else "Conflicting")
    all_dates = pd.DatetimeIndex(pd.to_datetime(close.dropna().index))

    def future_date(h):
        d = primary["anchor_date"]
        for _ in range(int(h)):
            d = _season_next_trading_date(d, all_dates)
        return d.date()

    bias = primary["verdict"]
    direction = 1 if bias == "Bullish" else -1 if bias == "Bearish" else 0
    out = {"Seasonality": bias, "SeasonalityMove": primary["move"], "SeasonalityReliability": primary["reliability"],
           "SeasonalityYears": primary["samples"], "Seasonality10YMove": ten["move"] if ten else np.nan,
           "Seasonality15YMove": fifteen["move"] if fifteen else np.nan,
           "Seasonality10Y": ten["verdict"] if ten else "n/a",
           "Seasonality15Y": fifteen["verdict"] if fifteen else "n/a",
           "SeasonalityAgreement": agreement, "SeasonalHighDays": primary["peak_h"],
           "SeasonalLowDays": primary["trough_h"], "SeasonalHighDate": future_date(primary["peak_h"]),
           "SeasonalLowDate": future_date(primary["trough_h"]), "SeasonalCoverage": primary["avg_coverage"],
           "SeasonalMinCoverage": primary["min_coverage"], "SeasonalityConfluence": "n/a",
           "SeasonalityScore": direction}
    out.update(_season_window_stats(close, direction, primary["anchor_date"]))
    if out.get("SeasonWindow") != "-":
        out["SeasonWindowWholeYear"] = ("Match" if agreement in ("Aligned Bullish", "Aligned Bearish") and direction == s10 == s15
                                        else "Partial" if agreement == "One Neutral" else "Mixed")
        out["SeasonWindowDriver"] = "10Y/15Y + calendar window" if out["SeasonWindowWholeYear"] == "Match" else "Calendar window"
    out.update(_season_volatility(close))
    return out


def _seasonality_fallback():
    return {"Seasonality": "n/a", "SeasonalityMove": np.nan, "SeasonalityReliability": np.nan,
            "SeasonalityYears": 0, "Seasonality10YMove": np.nan, "Seasonality15YMove": np.nan,
            "Seasonality10Y": "n/a", "Seasonality15Y": "n/a", "SeasonalityAgreement": "Insufficient Data",
            "SeasonalHighDays": np.nan, "SeasonalLowDays": np.nan, "SeasonalHighDate": None,
            "SeasonalLowDate": None, "SeasonalCoverage": np.nan, "SeasonalMinCoverage": np.nan,
            "SeasonalityConfluence": "n/a", "SeasonalityScore": 0,
            "SeasonWindow": "-", "SeasonWindowDays": np.nan, "SeasonWindowShape": "-",
            "SeasonWindowReliability": np.nan, "SeasonWindowSamples": 0, "SeasonWindowWins": 0,
            "SeasonWindowTotal": 0, "SeasonWindowAvgMove": np.nan, "SeasonWindowDriver": "-",
            "SeasonWindowNeighbour": "-", "SeasonWindowWholeYear": "-",
            "SeasonWindowStrength": None, "SeasonWindowInPlay": False,
            "SeasonVolCurrent": np.nan, "SeasonVol60D": np.nan, "SeasonVolSeasonalAvg": np.nan,
            "SeasonVolSeasonalPath": "", "SeasonVolPercentile": np.nan, "SeasonVolYears": 0,
            "SeasonVolCoverage": np.nan, "SeasonVolState": "n/a", "SeasonVolSizing": "Normal"}


def safe_seasonality_metrics(close):
    try:
        out = seasonality_metrics(close)
        base = _seasonality_fallback()
        if isinstance(out, dict):
            base.update(out)
            return base
        base["SeasonalityError"] = "seasonality_metrics returned no dictionary"
        return base
    except Exception as exc:
        base = _seasonality_fallback()
        base["SeasonalityError"] = str(exc)
        return base


# ============================== SCAN ==============================
def _attention_for_row(r):
    reasons = []; score = 0
    cot = str(r.get("COTBias", "Neutral"))
    val = str(r.get("Valuation", "Neutral"))
    seas = str(r.get("Seasonality", "n/a"))
    if cot in ("Bullish", "Bearish"): score += 2
    if val in ("Bullish", "Bearish") and val == cot:
        reasons.append("COT and valuation align"); score += 2
    elif val in ("Bullish", "Bearish") and val != cot:
        reasons.append("valuation challenges COT"); score += 3
    if seas in ("Bullish", "Bearish") and cot in ("Bullish", "Bearish") and seas != cot:
        reasons.append("seasonality conflicts with COT"); score += 3
    elif seas == cot and seas in ("Bullish", "Bearish"):
        reasons.append("seasonality confirms COT"); score += 1
    if cot == "Neutral" and seas in ("Bullish", "Bearish") and str(r.get("SeasonWindowStrength")) == "STRONG":
        reasons.append("seasonality-led; COT has no side"); score += 3
    if str(r.get("FundDivergence", "-")) not in ("-", "Neutral", "n/a"):
        reasons.append(f"fund divergence {r.get('FundDivergence')}"); score += 2
    if str(r.get("EdgeFinder", "-")) != "-":
        reasons.append("qualifying historical edge"); score += 2
    if str(r.get("SeasonWindow", "-")) != "-":
        reasons.append("timed seasonal window"); score += 1
    if str(r.get("SeasonVolState", "n/a")) == "Loud":
        reasons.append("seasonal volatility is loud"); score += 1
    if cot == "Neutral" and val in ("Bullish", "Bearish"):
        reasons.append("valuation-only read; no commercial COT side"); score += 2
    return score, "; ".join(reasons) if reasons else "no major conflict; routine chart review"


def _build_attention_narrative(r):
    cot = str(r.get("COTBias", "Neutral")); val = str(r.get("Valuation", "Neutral"))
    seas = str(r.get("Seasonality", "n/a")); parts = []
    if cot in ("Bullish", "Bearish"):
        parts.append(f"FOLLOW / FADE LOGIC: The primary COT read is {cot.lower()}. "
                     f"{r.get('CommAgreement', '0 of 5')} commercial lookbacks agree, with "
                     f"{r.get('Horizon', '-')} as the strongest active layer. COT remains the directional anchor.")
        if r.get("RetailAgreeLayers") not in (None, "-"):
            parts.append(f"Retail fade: {r.get('RetailAgreeLayers')} agrees with the COT price direction.")
        if r.get("FundDivergence") not in (None, "-"):
            parts.append(f"Fund divergence: {r.get('FundDivergence')}. Warning layer, not a reversal.")
    else:
        parts.append("FOLLOW / FADE LOGIC: There is no valid commercial COT side right now. "
                     "The scanner deliberately does not manufacture a bullish or bearish COT signal from retail positioning alone.")
    if val in ("Bullish", "Bearish"):
        relation = "supports" if val == cot else "challenges" if cot in ("Bullish", "Bearish") else "stands alone"
        parts.append(f"Valuation is {val.lower()} ({r.get('ValStrength', '-')}) and {relation} the primary COT read.")
    else:
        parts.append("Valuation is neutral.")
    if seas in ("Bullish", "Bearish"):
        move = _fmt(r.get("SeasonalityMove"), 2, "%"); rel = _fmt(r.get("SeasonalityReliability"), 1, "%")
        if cot in ("Bullish", "Bearish") and seas != cot:
            parts.append(f"Seasonality points {seas.lower()} ({move}, {rel}) so it tempers conviction rather than reversing COT.")
        elif seas == cot:
            parts.append(f"Seasonality confirms the COT side ({move}, {rel}).")
        else:
            parts.append(f"Seasonality is {seas.lower()} ({move}, {rel}) with no COT side; "
                         f"{'promoted to a single-engine read' if r.get('SeasonWindowStrength') == 'STRONG' else 'not promoted to a trade direction'}.")
    else:
        parts.append("Seasonality is neutral/insufficient for a directional seasonal verdict.")
    if str(r.get("SeasonWindow", "-")) != "-":
        parts.append(f"Timing window: {r.get('SeasonWindow')}. Neighbour test = {r.get('SeasonWindowNeighbour', '-')}; "
                     f"whole-year shape = {r.get('SeasonWindowWholeYear', '-')}.")
    if str(r.get("SeasonVolState", "n/a")) not in ("n/a", "-"):
        parts.append(f"Volatility seasonality: {r.get('SeasonVolState')}. Current vol {_fmt(r.get('SeasonVolCurrent'), 2, '%')} "
                     f"vs seasonal {_fmt(r.get('SeasonVolSeasonalAvg'), 2, '%')}; sizing note = {r.get('SeasonVolSizing', 'Normal')}.")
    if str(r.get("EdgeFinder", "-")) != "-":
        parts.append(f"Historical edge: {r.get('EdgeFinder')}.")
    return " ".join(parts)


def scan_one(name, group, ticker, code, invert, prices, weekly, cot, seasonality=True):
    if ticker is None or ticker not in prices.columns or prices[ticker].dropna().empty:
        raise ValueError(f"no price data for {ticker}")
    refs = [r for r in REFS_BY_GROUP[group] if REFS[r] != ticker]
    if name == "Gold":
        refs = ["SILVER", "DXY"]
    if not refs:
        refs = [r for r in REFS if REFS[r] != ticker]
    extras = [r for r in REFS_DISPLAY_ONLY.get(group, []) if r not in refs and REFS[r] != ticker]
    short_len = ROC_SHORT_INDICES if group == "Indices" else ROC_SHORT
    lens = {"short": short_len, "long": ROC_LONG}
    combos = [("D", "short"), ("D", "long"), ("W", "short"), ("W", "long")]
    reading = {r: {} for r in refs + extras}
    for tf, px in {"D": prices, "W": weekly}.items():
        if ticker not in px.columns:
            continue
        a_px = px[ticker].dropna()
        for r in refs + extras:
            t = REFS[r]
            if t not in px.columns:
                continue
            for hz, n in lens.items():
                try:
                    reading[r][(tf, hz)] = valuation(a_px, px[t].dropna(), n)
                except Exception:
                    reading[r][(tf, hz)] = np.nan
    sigs = {c: val_signal([reading[r].get(c, np.nan) for r in refs]) for c in combos}
    vsig, vweight, val_label = valuation_state(sigs[("D", VAL_PRIMARY)], sigs[("W", VAL_PRIMARY)])
    val_bias = {1: "Bullish", -1: "Bearish", 0: "Neutral"}[vsig]

    def pair(r):
        vals = reading.get(r, {})
        if not vals or not any(np.isfinite(_safe_float(v)) for v in vals.values()):
            return "-"
        f = lambda v: "-" if not np.isfinite(_safe_float(v)) else f"{_safe_float(v):.0f}"
        return f"D {f(vals.get(('D','short')))}/{f(vals.get(('D','long')))}  W {f(vals.get(('W','short')))}/{f(vals.get(('W','long')))}" + (" (info)" if r in extras else "")

    c = None
    if code:
        sub = cot[cot["code"] == code.lstrip("0")]
        if len(sub) >= COT_LOOKBACK_SHORT:
            c = cot_metrics(sub, invert, weekly[ticker] if ticker in weekly.columns else None)
    is_index = group == "Indices"
    scored = bool(c) and (not is_index or INDICES_USE_RETAIL)
    cot_total = 0
    if scored:
        cot_total = c["comm_score"] + c["spec_score"]
    if c and not is_index and c.get("comm_direction", 0) == 1:
        cot_bias = "Bullish"
    elif c and not is_index and c.get("comm_direction", 0) == -1:
        cot_bias = "Bearish"
    else:
        cot_bias = "n/a" if not c or is_index else "Neutral"
    show_cot = bool(c) and not is_index
    seas = safe_seasonality_metrics(prices[ticker].dropna()) if seasonality else _seasonality_fallback()
    season_conflict = (seas.get("Seasonality") in ("Bullish", "Bearish")
                       and cot_bias in ("Bullish", "Bearish")
                       and seas.get("Seasonality") != cot_bias)

    edge_rows = []
    if c:
        for layer in COT_LOOKBACKS:
            for role, prefix in (("commercial", ""), ("retail", "RetailEdge")):
                direction_label = c.get(f"{layer}_{prefix}EdgeDirection" if prefix else f"{layer}_EdgeDirection", "-")
                if direction_label not in ("Bullish", "Bearish"):
                    continue
                for h in COT_EDGE_HORIZONS:
                    base = f"{layer}_{prefix}Edge{h}" if prefix else f"{layer}_Edge{h}"
                    wp = _safe_float(c.get(base + "Win"))
                    avg = _safe_float(c.get(base + "AvgMove"))
                    total = int(_safe_float(c.get(base + "Total")) or 0)
                    if (np.isfinite(wp) and np.isfinite(avg) and total >= COT_EDGE_MIN_SAMPLES
                            and wp >= 60 and ((avg > 0 and direction_label == "Bullish") or (avg < 0 and direction_label == "Bearish"))):
                        wins = int(_safe_float(c.get(base + "Wins")) or 0)
                        edge_rows.append((wp + min(abs(avg) * 5, 25), role, layer, h, wp, wins, avg, total, direction_label))
    edge_rows.sort(reverse=True)
    if c and edge_rows:
        _, erole, elayer, eh, ewin, ewins, eavg, etotal, edir = edge_rows[0]
        c.update({"EdgeFinderLabel": f"{erole.title()} {elayer} {eh}W {edir} · {ewins} of {etotal} · {ewin:.0f}% · {eavg:+.2f}% avg",
                  "EdgeFinderRole": erole.title(), "EdgeFinderLayer": elayer, "EdgeFinderHorizon": eh,
                  "EdgeFinderWin": ewin, "EdgeFinderWins": ewins, "EdgeFinderTotal": etotal, "EdgeFinderAvgMove": eavg})
    elif c:
        c.update({"EdgeFinderLabel": "-", "EdgeFinderRole": "-", "EdgeFinderLayer": "-", "EdgeFinderHorizon": "-",
                  "EdgeFinderWin": np.nan, "EdgeFinderWins": 0, "EdgeFinderTotal": 0, "EdgeFinderAvgMove": np.nan})

    fundamentals = combine(
        cot_bias, val_bias, seas.get("Seasonality", "Neutral"),
        season_conflict,
        seas.get("SeasonWindowWholeYear") == "Mixed",
        season_strength=seas.get("SeasonWindowStrength"),
        season_in_play=bool(seas.get("SeasonWindowInPlay")),
    )
    if seas.get("Seasonality") in ("Bullish", "Bearish") and val_bias in ("Bullish", "Bearish"):
        seas["SeasonalityConfluence"] = "Aligned with Valuation" if seas["Seasonality"] == val_bias else "Conflicts with Valuation"
    elif seas.get("Seasonality") in ("Bullish", "Bearish") and cot_bias in ("Bullish", "Bearish"):
        seas["SeasonalityConfluence"] = "Aligned with COT" if seas["Seasonality"] == cot_bias else "Conflicts with COT"
    else:
        seas["SeasonalityConfluence"] = "Neutral / Mixed"

    row = {"Market": name + (" (inv)" if invert else ""), "Group": group,
           "Core": "yes" if name in CORE else "",
           "Fundamentals": fundamentals, "COT": cot_total if scored else np.nan,
           "COTBias": cot_bias, "Horizon": c["horizon"] if show_cot else "-",
           "Turning": ("yes" if c["turning"] else "no") if show_cot else "-",
           "CommAgreement": c.get("CommAgreement", "0 of 5") if show_cot else "-",
           "RetailAgreeLayers": c.get("RetailAgreeLayers", "-") if c else "-",
           "Comm26": c["comm26"] if show_cot else np.nan, "Comm3y": c["comm3y"] if show_cot else np.nan,
           "Comm1y": c.get("comm1y", np.nan) if show_cot else np.nan,
           "Comm2y": c.get("comm2y", np.nan) if show_cot else np.nan,
           "Comm5y": c.get("comm5y", np.nan) if show_cot else np.nan,
           "CommExtremeCount": c.get("comm_extreme_count", np.nan) if show_cot else np.nan,
           "Spec26": c["spec26"] if show_cot else np.nan,
           "SpecDiv": c["div"] if show_cot else "-",
           "Retail26": c["ret26"] if c else np.nan, "Retail3y": c["ret3y"] if c else np.nan,
           "Retail1y": c.get("ret1y", np.nan) if c else np.nan,
           "Retail2y": c.get("ret2y", np.nan) if c else np.nan,
           "Retail5y": c.get("ret5y", np.nan) if c else np.nan,
           "RetailExtremeCount": c.get("ret_extreme_count", np.nan) if c else np.nan}
    if c:
        for layer in COT_LOOKBACKS:
            for k, v in c.items():
                if k.startswith(layer + "_"):
                    row[f"COT_{layer}_{k}"] = v

    row.update({
        "Valuation": val_bias, "ValStrength": val_label,
        "Seasonality": seas.get("Seasonality", "n/a"),
        "SeasonalityMove": seas.get("SeasonalityMove", np.nan),
        "SeasonalityReliability": seas.get("SeasonalityReliability", np.nan),
        "SeasonalityYears": seas.get("SeasonalityYears", 0),
        "Seasonality10YMove": seas.get("Seasonality10YMove", np.nan),
        "Seasonality15YMove": seas.get("Seasonality15YMove", np.nan),
        "Seasonality10Y": seas.get("Seasonality10Y", "n/a"),
        "Seasonality15Y": seas.get("Seasonality15Y", "n/a"),
        "SeasonalityAgreement": seas.get("SeasonalityAgreement", "Insufficient Data"),
        "SeasonalHighDays": seas.get("SeasonalHighDays", np.nan),
        "SeasonalLowDays": seas.get("SeasonalLowDays", np.nan),
        "SeasonalHighDate": seas.get("SeasonalHighDate"),
        "SeasonalLowDate": seas.get("SeasonalLowDate"),
        "SeasonalCoverage": seas.get("SeasonalCoverage", np.nan),
        "SeasonalMinCoverage": seas.get("SeasonalMinCoverage", np.nan),
        "SeasonalityConfluence": seas.get("SeasonalityConfluence", "n/a"),
        "SeasonWindow": seas.get("SeasonWindow", "-"),
        "SeasonWindowDays": seas.get("SeasonWindowDays", np.nan),
        "SeasonWindowShape": seas.get("SeasonWindowShape", "-"),
        "SeasonWindowReliability": seas.get("SeasonWindowReliability", np.nan),
        "SeasonWindowSamples": seas.get("SeasonWindowSamples", 0),
        "SeasonWindowWins": seas.get("SeasonWindowWins", 0),
        "SeasonWindowTotal": seas.get("SeasonWindowTotal", 0),
        "SeasonWindowAvgMove": seas.get("SeasonWindowAvgMove", np.nan),
        "SeasonWindowDriver": seas.get("SeasonWindowDriver", "-"),
        "SeasonWindowNeighbour": seas.get("SeasonWindowNeighbour", "-"),
        "SeasonWindowWholeYear": seas.get("SeasonWindowWholeYear", "-"),
        "SeasonWindowStrength": seas.get("SeasonWindowStrength"),
        "SeasonWindowInPlay": seas.get("SeasonWindowInPlay", False),
        "SeasonVolCurrent": seas.get("SeasonVolCurrent", np.nan),
        "SeasonVol60D": seas.get("SeasonVol60D", np.nan),
        "SeasonVolSeasonalAvg": seas.get("SeasonVolSeasonalAvg", np.nan),
        "SeasonVolPercentile": seas.get("SeasonVolPercentile", np.nan),
        "SeasonVolYears": seas.get("SeasonVolYears", 0),
        "SeasonVolCoverage": seas.get("SeasonVolCoverage", np.nan),
        "SeasonVolState": seas.get("SeasonVolState", "n/a"),
        "SeasonVolSizing": seas.get("SeasonVolSizing", "Normal"),
        "DXY": pair("DXY"), "RATES": pair("RATES"), "GOLD": pair("GOLD"), "SILVER": pair("SILVER"),
        "COT_date": c["date"].date() if c else None,
        "EdgeFinder": c.get("EdgeFinderLabel", "-") if c else "-",
        "EdgeFinderWin": c.get("EdgeFinderWin", np.nan) if c else np.nan,
        "EdgeFinderWins": c.get("EdgeFinderWins", 0) if c else 0,
        "EdgeFinderTotal": c.get("EdgeFinderTotal", 0) if c else 0,
        "EdgeFinderAvgMove": c.get("EdgeFinderAvgMove", np.nan) if c else np.nan,
        "FundDivergence": c.get("div", "-") if c else "-",
    })
    row["CombinedVerdict"] = _build_attention_narrative(row)
    row["AttentionScore"], row["AttentionReason"] = _attention_for_row(row)
    row["_strength"] = (abs(cot_total) * 2.0 + abs(vsig) * vweight
                        + (0.75 if seas.get("Seasonality") == cot_bias and seas.get("Seasonality") in ("Bullish", "Bearish") else
                           -0.50 if season_conflict else 0.0))
    row["_core"] = 1 if name in CORE else 0
    return row


def _fallback_row(name, group, invert, error_msg):
    return {
        "Market": name + (" (inv)" if invert else ""), "Group": group,
        "Core": "yes" if name in CORE else "", "Fundamentals": "Error - review",
        "COT": np.nan, "COTBias": "Neutral", "Horizon": "-", "Turning": "-",
        "CommAgreement": "0 of 5", "RetailAgreeLayers": "-",
        "Comm26": np.nan, "Comm1y": np.nan, "Comm2y": np.nan, "Comm3y": np.nan, "Comm5y": np.nan,
        "CommExtremeCount": np.nan, "Spec26": np.nan, "SpecDiv": "-",
        "Retail26": np.nan, "Retail1y": np.nan, "Retail2y": np.nan, "Retail3y": np.nan,
        "Retail5y": np.nan, "RetailExtremeCount": np.nan,
        "Valuation": "Neutral", "ValStrength": "-",
        "Seasonality": "n/a", "SeasonalityMove": np.nan, "SeasonalityReliability": np.nan,
        "SeasonalityYears": 0, "Seasonality10YMove": np.nan, "Seasonality15YMove": np.nan,
        "Seasonality10Y": "n/a", "Seasonality15Y": "n/a",
        "SeasonalityAgreement": "Insufficient Data",
        "SeasonalHighDays": np.nan, "SeasonalLowDays": np.nan,
        "SeasonalHighDate": None, "SeasonalLowDate": None,
        "SeasonalCoverage": np.nan, "SeasonalMinCoverage": np.nan,
        "SeasonalityConfluence": "n/a",
        "SeasonWindow": "-", "SeasonWindowDays": np.nan, "SeasonWindowShape": "-",
        "SeasonWindowReliability": np.nan, "SeasonWindowSamples": 0, "SeasonWindowWins": 0,
        "SeasonWindowTotal": 0, "SeasonWindowAvgMove": np.nan,
        "SeasonWindowDriver": "-", "SeasonWindowNeighbour": "-", "SeasonWindowWholeYear": "-",
        "SeasonWindowStrength": None, "SeasonWindowInPlay": False,
        "SeasonVolCurrent": np.nan, "SeasonVol60D": np.nan, "SeasonVolSeasonalAvg": np.nan,
        "SeasonVolPercentile": np.nan, "SeasonVolYears": 0, "SeasonVolCoverage": np.nan,
        "SeasonVolState": "n/a", "SeasonVolSizing": "Normal",
        "DXY": "-", "RATES": "-", "GOLD": "-", "SILVER": "-",
        "COT_date": None,
        "EdgeFinder": "-", "EdgeFinderWin": np.nan, "EdgeFinderWins": 0,
        "EdgeFinderTotal": 0, "EdgeFinderAvgMove": np.nan,
        "FundDivergence": "-",
        "CombinedVerdict": f"The market could not be fully calculated: {error_msg}",
        "AttentionScore": 99, "AttentionReason": f"calculation error: {error_msg}",
        "_strength": -999, "_core": 1 if name in CORE else 0,
    }


def run_scan(prices, cot, seasonality=True):
    rows = []
    weekly = prices.resample("W-FRI").last()
    for name, group, ticker, code, invert in MARKETS:
        try:
            rows.append(scan_one(name, group, ticker, code, invert, prices, weekly, cot, seasonality))
        except Exception as e:
            log.warning("%s: retained with fallback row (%s)", name, e)
            rows.append(_fallback_row(name, group, invert, str(e)))
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["AttentionScore", "_strength", "_core", "Market"],
                             ascending=[False, False, False, True]).reset_index(drop=True)


# ============================== STATE ==============================
def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(table, extra=None):
    state = {
        "run_date": dt.date.today().isoformat(),
        "script_version": SCRIPT_VERSION,
        "markets": {},
    }
    if extra:
        state.update(extra)
    for _, r in table.iterrows():
        nm = str(r.get("Market", "")).replace(" (inv)", "")
        if not nm:
            continue
        _v = _safe_float(r.get("CommExtremeCount"))
        _q = _safe_float(r.get("EdgeFinderWin"))
        state["markets"][nm] = {
            "bias": str(r.get("COTBias", "Neutral")),
            "cot_lit": int(_v) if np.isfinite(_v) else 0,
            "season_window": str(r.get("SeasonWindow", "-")),
            "season_strength": r.get("SeasonWindowStrength") if r.get("SeasonWindowStrength") else None,
            "qualifying_rows": 1 if (r.get("EdgeFinder", "-") != "-" and np.isfinite(_q)) else 0,
            "combined": str(r.get("Fundamentals", "-")),
        }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        log.info("State saved to %s", STATE_FILE)
    except Exception as e:
        log.warning("Could not save state: %s", e)


# ============================== EXCEL ==============================
def _make_excel_report(table):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    if table.empty:
        return None
    out = table.drop(columns=[c for c in ["_strength", "_core"] if c in table.columns]).copy()
    for c in ["COT", "Comm26", "Comm1y", "Comm2y", "Comm3y", "Comm5y", "CommExtremeCount",
              "Spec26", "Retail26", "Retail1y", "Retail2y", "Retail3y", "Retail5y", "RetailExtremeCount"]:
        if c in out:
            out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{float(v):.0f}")

    wb = Workbook()
    ws = wb.active
    ws.title = "DASHBOARD"
    title_fill = PatternFill("solid", fgColor="1F4E78")
    section_fill = PatternFill("solid", fgColor="D9EAF7")
    header_fill = PatternFill("solid", fgColor="5B9BD5")
    green_fill = PatternFill("solid", fgColor="E2F0D9")
    red_fill = PatternFill("solid", fgColor="FCE4D6")
    white_font = Font(color="FFFFFF", bold=True, size=14)
    bold = Font(bold=True)
    thin = Side(style="thin", color="B7B7B7")

    ws.merge_cells("A1:J1")
    ws["A1"] = "MARKET SCANNER — QUICK DECISION DASHBOARD"
    ws["A1"].fill = title_fill
    ws["A1"].font = white_font
    ws["A1"].alignment = Alignment(horizontal="center")
    ws["A2"] = "Latest COT report"
    if "COT_date" in out.columns and out["COT_date"].notna().any():
        ws["B2"] = str(out["COT_date"].dropna().iloc[0])
    else:
        ws["B2"] = "-"
    ws["A3"] = "How to use"
    ws["B3"] = "Read COT first, then valuation, seasonality, historical edge, divergence and volatility."
    ws.merge_cells("B3:H3")

    aligned = out[out["Fundamentals"].astype(str).str.contains("valuation aligned", case=False, na=False)]
    candidates = out[out["Fundamentals"].astype(str).str.contains("Bullish|Bearish", regex=True)].copy()
    candidates = candidates.sort_values(["AttentionScore", "Core", "Market"], ascending=[False, False, True])

    ws["A5"] = "BEST ALIGNED MARKETS"
    ws["A5"].font = bold; ws["A5"].fill = section_fill
    ws.merge_cells("A5:H5")
    dash_cols = ["Market", "Group", "Core", "Fundamentals", "COT", "COTBias", "Horizon", "CommExtremeCount",
                 "Valuation", "ValStrength", "Seasonality", "SeasonalityReliability", "SeasonWindow",
                 "SeasonVolState", "EdgeFinder", "AttentionReason"]
    for j, c in enumerate(dash_cols, 1):
        cell = ws.cell(6, j, c); cell.fill = header_fill; cell.font = bold; cell.alignment = Alignment(horizontal="center")
    for i, (_, r) in enumerate(aligned.iterrows(), 7):
        for j, c in enumerate(dash_cols, 1):
            ws.cell(i, j, r.get(c, "-"))

    start = 7 + max(len(aligned), 1) + 2
    ws.cell(start, 1, "TOP ATTENTION MARKETS").font = bold
    ws.cell(start, 1).fill = section_fill
    ws.merge_cells(start_row=start, start_column=1, end_row=start, end_column=10)
    for j, c in enumerate(dash_cols, 1):
        cell = ws.cell(start + 1, j, c); cell.fill = header_fill; cell.font = bold; cell.alignment = Alignment(horizontal="center")
    for i, (_, r) in enumerate(candidates.head(10).iterrows(), start + 2):
        for j, c in enumerate(dash_cols, 1):
            ws.cell(i, j, r.get(c, "-"))

    wa = wb.create_sheet("ALL MARKETS")
    for j, c in enumerate(out.columns, 1):
        cell = wa.cell(1, j, c); cell.fill = header_fill; cell.font = bold; cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, (_, r) in enumerate(out.iterrows(), 2):
        for j, c in enumerate(out.columns, 1):
            wa.cell(i, j, r[c])
    wa.freeze_panes = "A2"; wa.auto_filter.ref = wa.dimensions

    wc = wb.create_sheet("COT FOCUS")
    cot_cols = ["Market", "Group", "Core", "Fundamentals", "COT", "Horizon", "Turning", "CommExtremeCount",
                "Comm5y", "Comm3y", "Comm2y", "Comm1y", "Comm26", "Spec26",
                "RetailExtremeCount", "Retail5y", "Retail3y", "Retail2y", "Retail1y", "Retail26",
                "SpecDiv", "CommAgreeLayers", "CommAgreement", "RetailAgreeLayers", "EdgeFinder", "COT_date"]
    for j, c in enumerate(cot_cols, 1):
        cell = wc.cell(1, j, c); cell.fill = header_fill; cell.font = bold; cell.alignment = Alignment(horizontal="center", wrap_text=True)
    cot_out = out[out["Group"] != "Indices"].copy()
    for i, (_, r) in enumerate(cot_out.iterrows(), 2):
        for j, c in enumerate(cot_cols, 1):
            wc.cell(i, j, r.get(c, "-"))
    wc.freeze_panes = "A2"; wc.auto_filter.ref = wc.dimensions

    wv = wb.create_sheet("VALUATION")
    val_cols = ["Market", "Group", "Fundamentals", "Valuation", "ValStrength", "DXY", "RATES", "GOLD", "SILVER"]
    for j, c in enumerate(val_cols, 1):
        cell = wv.cell(1, j, c); cell.fill = header_fill; cell.font = bold; cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for i, (_, r) in enumerate(out.iterrows(), 2):
        for j, c in enumerate(val_cols, 1):
            wv.cell(i, j, r.get(c, "-"))
    wv.freeze_panes = "A2"; wv.auto_filter.ref = wv.dimensions

    wsx = wb.create_sheet("SEASONALITY")
    season_cols = ["Market", "Group", "Core", "Seasonality", "SeasonalityMove", "SeasonalityReliability",
                   "SeasonalityYears", "Seasonality10Y", "Seasonality10YMove", "Seasonality15Y",
                   "Seasonality15YMove", "SeasonalityAgreement", "SeasonalityConfluence",
                   "SeasonWindow", "SeasonWindowDays", "SeasonWindowShape", "SeasonWindowReliability",
                   "SeasonWindowWins", "SeasonWindowTotal", "SeasonWindowSamples", "SeasonWindowAvgMove",
                   "SeasonWindowNeighbour", "SeasonWindowDriver", "SeasonWindowWholeYear",
                   "SeasonWindowStrength", "SeasonVolCurrent", "SeasonVol60D", "SeasonVolSeasonalAvg",
                   "SeasonVolPercentile", "SeasonVolYears", "SeasonVolCoverage", "SeasonVolState",
                   "SeasonVolSizing", "SeasonalHighDays", "SeasonalHighDate", "SeasonalLowDays",
                   "SeasonalLowDate", "SeasonalCoverage", "SeasonalMinCoverage"]
    for j, c in enumerate(season_cols, 1):
        cell = wsx.cell(1, j, c); cell.fill = header_fill; cell.font = bold; cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, (_, r) in enumerate(out.iterrows(), 2):
        for j, c in enumerate(season_cols, 1):
            wsx.cell(i, j, r.get(c, "-"))
    wsx.freeze_panes = "A2"; wsx.auto_filter.ref = wsx.dimensions

    we = wb.create_sheet("EDGE - TRACK RECORD")
    edge_cols = ["Market", "Group", "Core", "Fundamentals", "COT", "Layer", "Role", "ExtremeIndex", "Direction",
                 "Edge4Win", "Edge4Wins", "Edge4Total", "Edge4AvgMove", "Edge4AvgWeeksIn",
                 "Edge8Win", "Edge8Wins", "Edge8Total", "Edge8AvgMove", "Edge8AvgWeeksIn", "EdgeFinder"]
    for j, c in enumerate(edge_cols, 1):
        cell = we.cell(1, j, c); cell.fill = header_fill; cell.font = bold; cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    erow = 2
    for _, r in out.iterrows():
        for layer, idxcol in [("6M", "Comm26"), ("1Y", "Comm1y"), ("2Y", "Comm2y"), ("3Y", "Comm3y"), ("5Y", "Comm5y")]:
            idx = _safe_float(r.get(idxcol, np.nan))
            if not np.isfinite(idx) or not (idx <= EXTREME_LO or idx >= EXTREME_HI):
                continue
            for role, prefix in (("Commercial", ""), ("Retail", "RetailEdge")):
                direction = r.get(f"{layer}_{prefix}EdgeDirection" if prefix else f"{layer}_EdgeDirection", "-")
                if direction not in ("Bullish", "Bearish"):
                    continue
                vals = {"Market": r.get("Market", "-"), "Group": r.get("Group", "-"), "Core": r.get("Core", ""),
                        "Fundamentals": r.get("Fundamentals", "-"), "COT": r.get("COT", "-"),
                        "Layer": layer, "Role": role, "ExtremeIndex": round(idx, 1), "Direction": direction,
                        "EdgeFinder": r.get("EdgeFinder", "-")}
                for h in (4, 8):
                    pre = f"{layer}_{prefix}Edge{h}" if prefix else f"{layer}_Edge{h}"
                    vals[f"Edge{h}Win"] = r.get(pre + "Win", np.nan)
                    vals[f"Edge{h}Wins"] = r.get(pre + "Wins", 0)
                    vals[f"Edge{h}Total"] = r.get(pre + "Total", 0)
                    vals[f"Edge{h}AvgMove"] = r.get(pre + "AvgMove", np.nan)
                    vals[f"Edge{h}AvgWeeksIn"] = r.get(pre + "AvgWeeksIn", np.nan)
                for j, cname in enumerate(edge_cols, 1):
                    v = vals.get(cname, "-")
                    if isinstance(v, float) and np.isnan(v):
                        v = "-"
                    we.cell(erow, j, v)
                erow += 1
    we.freeze_panes = "A2"; we.auto_filter.ref = we.dimensions

    wcv = wb.create_sheet("COMBINED VERDICTS")
    cv_cols = ["Market", "Group", "Core", "Fundamentals", "COT", "COTBias", "Valuation", "ValStrength",
               "Seasonality", "SeasonalityAgreement", "SeasonWindow", "SeasonVolState", "SeasonVolSizing",
               "EdgeFinder", "AttentionReason", "CombinedVerdict"]
    for j, cname in enumerate(cv_cols, 1):
        cell = wcv.cell(1, j, cname); cell.fill = header_fill; cell.font = bold; cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, (_, r) in enumerate(out.iterrows(), 2):
        for j, cname in enumerate(cv_cols, 1):
            wcv.cell(i, j, r.get(cname, "-"))
    wcv.freeze_panes = "A2"; wcv.auto_filter.ref = wcv.dimensions

    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = False
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for col_cells in sheet.columns:
            letter = get_column_letter(col_cells[0].column)
            max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells[:100])
            sheet.column_dimensions[letter].width = min(max(max_len + 2, 10), 32)
        for row in sheet.iter_rows():
            for cell in row:
                cell.border = Border(bottom=thin)

    for sheet_name in ["DASHBOARD", "ALL MARKETS"]:
        sh = wb[sheet_name]
        for row in sh.iter_rows():
            for cell in row:
                if "Bullish" in str(cell.value):
                    cell.fill = green_fill
                elif "Bearish" in str(cell.value):
                    cell.fill = red_fill

    path = "scan_results.xlsx"
    wb.save(path)
    return path
# ============================== PDF ==============================
RPT = dict(bg="#faf7f4", ink="#1a1a1a", muted="#8a8a8a", rule="#e3ddd6", green="#2e7d32", red="#c62828",
           amber="#b8860b", blue="#2563eb", card="#ffffff", lav="#eeeef7", greentint="#e4efe4",
           redtint="#fadadd", grey="#ececec")
RPT_TABLE_ORDER = [("6M", "6 mo"), ("1Y", "1 yr"), ("2Y", "2 yr"), ("3Y", "3 yr"), ("5Y", "Multi-yr")]
RPT_CHIP_ORDER = list(reversed(RPT_TABLE_ORDER))
RPT_DIRWORD = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
RPT_NUMWORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
               8: "eight", 9: "nine", 10: "ten"}

RPT_MARKET_LABELS = {
    "EURUSD": ("EURUSD", "Euro FX"), "GBPUSD": ("GBPUSD", "British Pound"),
    "AUDUSD": ("AUDUSD", "Australian Dollar"), "NZDUSD": ("NZDUSD", "NZ Dollar"),
    "USDJPY": ("USDJPY", "Japanese Yen"), "USDCHF": ("USDCHF", "Swiss Franc"),
    "USDCAD": ("USDCAD", "Canadian Dollar"), "DXY": ("USD Index", "DXY"),
    "Gold": ("Gold", "Gold"), "Silver": ("Silver", "Silver"), "Platinum": ("Platinum", "Platinum"),
    "Palladium": ("Palladium", "Palladium"), "Copper": ("Copper", "Copper"),
    "WTI Crude": ("WTI Crude", "WTI Crude"), "Natural Gas": ("Natural Gas", "Natural Gas"),
    "S&P 500": ("S&P 500", "S&P 500"), "Nasdaq 100": ("Nasdaq 100", "Nasdaq 100"),
    "Dow Jones": ("Dow Jones", "Dow Jones"), "Russell 2000": ("Russell 2000", "Russell 2000"),
}
RPT_FX_PAIRS = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"}


_TV_EXCH = {"6E": "CME", "6B": "CME", "6A": "CME", "6N": "CME", "6J": "CME", "6S": "CME", "6C": "CME",
            "GC": "COMEX", "SI": "COMEX", "HG": "COMEX", "PL": "NYMEX", "PA": "NYMEX", "CL": "NYMEX",
            "NG": "NYMEX", "HO": "NYMEX", "ZC": "CBOT", "ZW": "CBOT", "KE": "CBOT", "ZS": "CBOT", "ZL": "CBOT",
            "SB": "ICE", "KC": "ICE", "CT": "ICE", "CC": "ICE", "ES": "CME", "NQ": "CME", "YM": "CBOT",
            "RTY": "CME", "NKD": "CME"}


def _rpt_tvlabel(pk_or_ticker):
    """TradingView-style continuous contract label, e.g. 6B=F -> '6B1! · CME'."""
    t = pk_or_ticker["ticker"] if isinstance(pk_or_ticker, dict) else pk_or_ticker
    if not t:
        return "-"
    if t == "DX-Y.NYB":
        return "DX1! · ICE"
    if t.endswith("=F"):
        root = t[:-2]
        return f"{root}1! · {_TV_EXCH.get(root, 'futures')}"
    return t.lstrip("^") + " · cash"


def _rpt_market_display(nm):
    if nm in RPT_FX_PAIRS:
        return nm, nm, f"{nm} ({RPT_MARKET_LABELS[nm][1]})"
    if nm in RPT_MARKET_LABELS:
        short, long = RPT_MARKET_LABELS[nm]
        if nm == "DXY":
            return short, long, f"{short} ({long})"
        if short != long:
            return short, long, f"{short} ({long})"
        return short, long, short
    return nm, nm, nm


def _nw(n):
    return RPT_NUMWORD.get(int(n), str(int(n)))


def _rpt_dircolor(d):
    return RPT["green"] if d == 1 else RPT["red"] if d == -1 else RPT["muted"]


def _esc(s):
    from xml.sax.saxutils import escape
    return escape(str(s))


def _rpt_price_dir(d, role, invert):
    if d == 0:
        return 0
    p = d if role == "commercial" else -d
    return -p if invert else p


def _rpt_role_state(sub, role, invert):
    c = sub.sort_values("date").reset_index(drop=True)
    oi = c["oi"].replace(0, np.nan)
    net = (c["comm_long"] - c["comm_short"]) if role == "commercial" else (c["ret_long"] - c["ret_short"])
    basis = pd.Series((net / oi * 100.0).values, index=pd.to_datetime(c["date"]))
    out = {"basis": basis, "layers": {}}

    def _idx(a):
        cur = a[-1]
        return float((a <= cur).sum()) / float(len(a)) * 100.0

    for key, _label in RPT_TABLE_ORDER:
        w = COT_LOOKBACKS[key]
        idx = basis.rolling(w, min_periods=max(4, w // 2)).apply(_idx, raw=True)
        d = idx.map(lambda v: 0 if not np.isfinite(v) else 1 if v >= EXTREME_HI else -1 if v <= EXTREME_LO else 0).astype(int)
        now = int(d.iloc[-1]) if len(d) else 0
        weeks_in = 0
        if now != 0:
            run = 0
            for v in d.values[::-1]:
                if v == now:
                    run += 1
                else:
                    break
            weeks_in = run - 1
        out["layers"][key] = {"idx": idx, "d": d, "now": now, "pdir": _rpt_price_dir(now, role, invert),
                              "index_now": float(idx.iloc[-1]) if len(idx) and np.isfinite(idx.iloc[-1]) else np.nan,
                              "weeks_in": weeks_in, "w": w, "valid": int(idx.notna().sum())}
    pd_list = [L["pdir"] for L in out["layers"].values()]
    pos, neg = sum(v == 1 for v in pd_list), sum(v == -1 for v in pd_list)
    out["pdir"] = 1 if pos > neg else -1 if neg > pos else 0
    out["lit"] = max(pos, neg) if out["pdir"] else 0
    ds = [L["now"] for L in out["layers"].values() if L["pdir"] == out["pdir"] and out["pdir"]]
    out["side"] = ds[0] if ds else 0
    return out


def _rpt_track(layer, weekly_px, role, invert, h):
    now = layer["pdir"]
    if now == 0 or weekly_px is None:
        return None
    px = weekly_px.dropna().copy()
    px.index = pd.to_datetime(px.index)
    d = layer["d"]; dates = d.index; vals = d.values
    moves = []
    for i in range(len(vals)):
        if vals[i] == 0 or (i > 0 and vals[i] == vals[i - 1]):
            continue
        if _rpt_price_dir(int(vals[i]), role, invert) != now:
            continue
        pos0 = px.index.searchsorted(dates[i], side="right") - 1
        if pos0 < 0 or pos0 + h >= len(px):
            continue
        move = (float(px.iloc[pos0 + h]) / float(px.iloc[pos0]) - 1.0) * 100.0 * now
        moves.append(move)
    moves = moves[-6:]
    cases = len(moves); wins = sum(m > 0 for m in moves)
    return {"live": True, "past": layer["weeks_in"] >= h, "weeks_in": layer["weeks_in"],
            "cases": cases, "wins": wins,
            "win": (wins / cases * 100.0) if cases else np.nan,
            "avg": float(np.mean(moves)) if cases else np.nan,
            "history": max(1, int(round(layer["valid"] / 52.0))), "dir": now}


def _rpt_cot_pack(sub, weekly_px, invert):
    comm = _rpt_role_state(sub, "commercial", invert)
    ret = _rpt_role_state(sub, "retail", invert)
    tracks = {}
    for role, st in (("commercial", comm), ("retail", ret)):
        for key, _ in RPT_TABLE_ORDER:
            for h in COT_EDGE_HORIZONS:
                tracks[(role, key, h)] = _rpt_track(st["layers"][key], weekly_px, role, invert, h)
    bias = comm["pdir"]
    lit = (comm["lit"] if comm["pdir"] == bias else 0) + (ret["lit"] if ret["pdir"] == bias and bias else 0)
    conflict = bool(bias and ret["pdir"] == -bias)
    h0 = COT_EDGE_HORIZONS[0]
    cands = []
    for (role, key, h), t in tracks.items():
        if h != h0 or not t or t["past"]:
            continue
        if t["cases"] >= REPORT_EDGE_MIN_CASES and t["win"] >= REPORT_EDGE_MIN_WIN and t["avg"] > 0:
            cands.append((t["win"], t["cases"], t["avg"], role, key))
    cands.sort(reverse=True)
    edge = None
    if cands:
        _, _, _, er, ek = cands[0]
        edge = {"role": er, "key": ek, "h": h0, "t": tracks[(er, ek, h0)],
                "t8": tracks[(er, ek, COT_EDGE_HORIZONS[-1])]}
    qualifying = [(r, k) for (_, _, _, r, k) in cands]
    return {"comm": comm, "ret": ret, "tracks": tracks, "bias": bias, "lit": lit, "conflict": conflict,
            "edge": edge, "qualifying": qualifying}


def _rpt_daily(close):
    x = close.dropna().copy()
    x.index = pd.to_datetime(x.index)
    x = x.sort_index()
    x = x[~x.index.duplicated(keep="last")]
    return x.resample("D").last().ffill()


def _rpt_ret_matrix(d, bases, lens, years):
    pv = d.values.astype(float)
    t0 = d.index[0]
    R = np.full((years, len(bases), len(lens)), np.nan)
    lens = np.asarray(lens, dtype=int)
    for k in range(1, years + 1):
        for e, b in enumerate(bases):
            s = pd.Timestamp(b) - pd.DateOffset(years=k)
            si = (s - t0).days
            if si < 0 or si >= len(pv):
                continue
            ei = si + lens
            ok = ei < len(pv)
            if not ok.any() or not np.isfinite(pv[si]) or pv[si] == 0:
                continue
            R[k - 1, e, ok] = (pv[ei[ok]] / pv[si] - 1.0) * 100.0
    return R


def _rpt_window_stats(R, direction):
    valid = np.isfinite(R)
    cases = valid.sum(axis=0)
    wins = ((R * direction > 0) & valid).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        win = np.where(cases > 0, wins / cases, np.nan)
        avg = np.nanmean(np.where(valid, R, np.nan), axis=0)
    rv = valid[:4]
    rec_cases = rv.sum(axis=0)
    rec_wins = ((R[:4] * direction > 0) & rv).sum(axis=0)
    return win, cases, wins, avg, rec_wins, rec_cases


def _rpt_strength(win, cases, rec_wins, rec_cases, avg):
    if cases < max(8, int(REPORT_SEASON_YEARS * 0.6)):
        return None
    if win >= SEASON_STRONG_WIN and rec_cases and rec_wins >= max(1, rec_cases - 1) and abs(avg) >= SEASON_STRONG_AVG:
        return "STRONG"
    if win >= SEASON_MODERATE_WIN and abs(avg) >= SEASON_MODERATE_AVG:
        return "MODERATE"
    return None


def _rpt_season_pack(close, today=None):
    try:
        d = _rpt_daily(close)
    except Exception:
        return None
    if len(d) < 365 * 9:
        return None
    today = pd.Timestamp(today or d.index[-1]).normalize()
    years = min(REPORT_SEASON_YEARS, int(len(d) / 365.25) - 1)
    if years < 8:
        return None
    entries = [today + pd.Timedelta(days=o) for o in range(-REPORT_SEASON_BEHIND_DAYS, REPORT_SEASON_AHEAD_DAYS + 1)]
    lens = list(range(7, 61))
    R = _rpt_ret_matrix(d, entries, lens, years)
    best = {}
    # Prefer nearer entries so the Seasonal Overview does not surface windows
    # that are still weeks away when a comparable nearer window exists.
    # entries[0] = today - BEHIND, entries[-1] = today + AHEAD.
    entry_offsets = np.array([(e - today).days for e in entries], dtype=float)
    # Soft proximity bonus: full credit for starts already underway or within ~3 weeks;
    # gently decays toward the outer edge of the +8 week horizon.
    prox = np.clip(1.0 - np.maximum(entry_offsets - 21, 0) / max(REPORT_SEASON_AHEAD_DAYS - 21, 1), 0.55, 1.0)
    for direction in (1, -1):
        win, cases, wins, avg, rw, rc = _rpt_window_stats(R, direction)
        okmask = np.isfinite(win) & (cases >= max(8, int(years * 0.6))) & (avg * direction > 0)
        lenbonus = np.array([min(L, 21) / 21.0 * 0.04 for L in lens])[None, :]
        prox_bonus = prox[:, None] * 0.06  # small tilt; strength/win still dominate
        score = np.where(okmask, win + lenbonus + np.minimum(np.abs(avg), 3) * 0.01 + prox_bonus, -9.0)
        if not np.isfinite(score).any() or score.max() < 0:
            continue
        e, l = np.unravel_index(int(np.argmax(score)), score.shape)
        st = _rpt_strength(win[e, l], cases[e, l], rw[e, l], rc[e, l], avg[e, l])
        if st is None:
            continue
        L = lens[l]; s0 = entries[e]
        nb_entries = [s0 + pd.Timedelta(days=o) for o in (-14, -7, 7, 14)]
        Rn = _rpt_ret_matrix(d, nb_entries, [L], years)
        nwin, _, _, _, _, _ = _rpt_window_stats(Rn, direction)
        nb = float(np.nanmean(nwin)) if np.isfinite(nwin).any() else np.nan
        w = float(win[e, l])
        timing = "tight" if np.isfinite(nb) and nb <= w - 0.30 else "forgiving" if np.isfinite(nb) and nb >= w - 0.12 else "moderate"
        best[direction] = {"dir": direction, "start": s0, "end": s0 + pd.Timedelta(days=L), "len": L,
                           "shape": "sharp" if L <= 21 else "broad", "win": w,
                           "cases": int(cases[e, l]), "wins": int(wins[e, l]),
                           "avg": float(avg[e, l]), "rec_wins": int(rw[e, l]), "rec_cases": int(rc[e, l]),
                           "strength": st, "neighbour": nb, "timing": timing, "years": years}
    main = None
    if best:
        # Overview only surfaces windows whose entry is already underway or starts
        # within the next ~5 weeks. Farther candidates (still inside the +8w search
        # band) are kept in best[] for chapter pages but do not drive the overview.
        DISPLAY_AHEAD_DAYS = 35
        near = {
            d: b for d, b in best.items()
            if (b["start"] - today).days <= DISPLAY_AHEAD_DAYS
        }
        pool = near if near else best  # fall back only if nothing nearer exists
        def _main_key(b):
            days_out = (b["start"] - today).days
            return (
                b["strength"] == "STRONG",
                b["win"],
                abs(b.get("avg", 0)),
                -days_out,
            )
        main = sorted(pool.values(), key=_main_key, reverse=True)[0]
    shape = np.nan
    try:
        paths = []
        yr = int(today.year)
        for k in range(1, years + 1):
            seg = d[(d.index >= pd.Timestamp(year=yr - k, month=1, day=1)) & (d.index <= pd.Timestamp(year=yr - k, month=12, day=31))]
            if len(seg) >= 360:
                paths.append((seg.values[:365] / seg.values[0] - 1.0) * 100.0)
        cur = d[(d.index >= pd.Timestamp(year=yr, month=1, day=1)) & (d.index <= today)]
        if paths and len(cur) > 30:
            m = min(len(p) for p in paths)
            avgp = np.mean([p[:m] for p in paths], axis=0)
            n = min(len(cur), m)
            cp = (cur.values[:n] / cur.values[0] - 1.0) * 100.0
            shape = float(np.corrcoef(avgp[:n], cp)[0, 1])
    except Exception:
        shape = np.nan
    return {"d": d, "today": today, "best": best, "main": main, "shape": shape, "years": years}


def _rpt_fig_to_png(fig, dpi=170):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return buf


def _rpt_style_ax(ax):
    ax.set_facecolor(RPT["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cfc8bf")
    ax.tick_params(colors=RPT["muted"], labelsize=7)
    ax.grid(True, color="#ece6de", lw=0.6)


def _rpt_cot_chart(state, role, years=5):
    import matplotlib.pyplot as plt
    basis = state["basis"]
    end = basis.index[-1]
    start = end - pd.DateOffset(years=years)
    fig, ax = plt.subplots(figsize=(11.2, 1.0), facecolor=RPT["bg"])
    _rpt_style_ax(ax)
    zone_col = RPT["red"] if role == "retail" else RPT["green"]
    side = state["side"] if state["side"] else (-1 if role == "retail" else 1)
    lit = 0
    for key, _ in reversed(RPT_TABLE_ORDER):
        L = state["layers"][key]
        w = L["w"]
        hi = basis.rolling(w, min_periods=max(4, w // 2)).max()
        lo = basis.rolling(w, min_periods=max(4, w // 2)).min()
        rng = hi - lo
        if side == 1:
            lower, upper = hi - (100 - EXTREME_HI) / 100.0 * rng, hi
        else:
            lower, upper = lo, lo + EXTREME_LO / 100.0 * rng
        m = basis.index >= start
        ax.fill_between(basis.index[m], lower[m], upper[m], color=zone_col, alpha=0.20, lw=0)
    for key, _ in RPT_TABLE_ORDER:
        if state["layers"][key]["pdir"] == state["pdir"] and state["pdir"]:
            lit += 1
    m = basis.index >= start
    ax.plot(basis.index[m], basis[m], color="#222222" if role == "retail" else RPT["blue"], lw=1.4)
    ax.plot([basis.index[-1]], [basis.iloc[-1]], marker="o", ms=4, color="#222222" if role == "retail" else RPT["blue"])
    ax.set_xlim(start, end + pd.Timedelta(days=60))
    ax.set_ylabel("Retail" if role == "retail" else "Commercial", fontsize=8, fontweight="bold",
                  color="#222222" if role == "retail" else RPT["blue"])
    for i in range(5):
        ax.add_patch(plt.Rectangle((0.0 + i * 0.016, 1.04), 0.012, 0.15, transform=ax.transAxes,
                                   facecolor=zone_col if i < lit else "#ffffff", edgecolor=zone_col, lw=0.6,
                                   clip_on=False))
    ax.text(0.09, 1.115, f"Layered extreme: {lit} of 5", transform=ax.transAxes, fontsize=7,
            color=RPT["muted"], va="center")
    return _rpt_fig_to_png(fig)


def _px_dec(v):
    v = abs(float(v))
    return 0 if v >= 10000 else 1 if v >= 1000 else 2 if v >= 100 else 3 if v >= 10 else 4 if v >= 1 else 5


def _rpt_draw_zones(ax, df, zones, ylo, yhi, dec, tf="D"):
    """Draw zones computed on the SAME timeframe as `df`.
    Prefer integer bar indices (left/right/born); fall back to dates only if needed.
    Zones extend ZONE_EXTEND_BARS bars to the right of the last bar (clipped at the chart edge)."""
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgba
    GOLD = "#FFD700"
    xlo, xhi = ax.get_xlim()
    n = len(df)
    if n < 2:
        return

    x_nums = mdates.date2num(df.index.to_pydatetime())
    avg_gap = float(np.median(np.diff(x_nums)))
    days_per_bar = float((x_nums[-1] - x_nums[0]) / max(1, n - 1))
    right_x = x_nums[-1] + days_per_bar * ZONE_EXTEND_BARS

    if tf == "M":
        half_w = max(avg_gap * 0.35, 6.0)
        label_offset = max(avg_gap * 0.45, 8.0)
        label_fs = 6.0
        extend_bars = 8
    elif tf == "W":
        half_w = max(avg_gap * 0.32, 1.8)
        label_offset = max(avg_gap * 0.40, 3.0)
        label_fs = 6.0
        extend_bars = 8
    else:
        half_w = min(0.45, avg_gap * 0.40)
        label_offset = 3.0
        label_fs = 6.5
        extend_bars = 60

    def _bar_x(i):
        i = int(max(0, min(i, n - 1)))
        return float(x_nums[i])

    # Keep only unmitigated zones that are still relevant to the visible window.
    # Prefer freshest zones (by born bar), cap at 6 so the chart stays readable.
    visible_i0 = 0
    try:
        # first bar whose date is >= chart left edge
        visible_i0 = int(np.searchsorted(x_nums, xlo))
    except Exception:
        pass
    filtered = []
    for z in zones:
        born = int(z.get("born", 0))
        # drop tested zones born well before the visible window
        if z.get("tested") and born < max(0, visible_i0 - extend_bars):
            continue
        filtered.append(z)
    filtered = sorted(filtered, key=lambda z: int(z.get("born", 0)))[-6:]

    used_ys = []
    for z in filtered:
        top, bot = float(z["top"]), float(z["bottom"])
        if top < ylo or bot > yhi:
            continue
        # Prefer integer bar indices from the TF the zones were computed on
        left_i = int(z.get("left", 0))
        right_i = int(z.get("right", left_i))
        born_i = int(z.get("born", right_i))
        left_i = max(0, min(left_i, n - 1))
        right_i = max(left_i, min(right_i, n - 1))
        born_i = max(0, min(born_i, n - 1))

        xl = _bar_x(left_i) - half_w
        xr = _bar_x(right_i) + half_w
        # proximal must be at least one half-bar wide
        if xr - xl < half_w * 1.5:
            mid = (_bar_x(left_i) + _bar_x(right_i)) / 2.0
            xl, xr = mid - half_w, mid + half_w

        if z.get("tested"):
            face, edge = to_rgba("#1a1a1a", 0.28), to_rgba(GOLD, 0.50)
            proximal = to_rgba(GOLD, 0.40)
        else:
            face = to_rgba("#00c9a7" if z["demand"] else "#e63946", 0.18)
            edge = to_rgba(GOLD, 0.75)
            proximal = to_rgba(GOLD, 0.65)

        # Extended zone (faded, dashed) from proximal left out to the right pad
        ax.add_patch(Rectangle((xl, bot), right_x - xl, top - bot,
                               facecolor=face, edgecolor=edge, lw=0.9, ls="--", zorder=1))
        # Proximal base (solid gold)
        ax.add_patch(Rectangle((xl, bot), xr - xl, top - bot,
                               facecolor=proximal, edgecolor=to_rgba(GOLD, 0.90),
                               lw=1.15, ls="-", zorder=2))

        # Label near proximal, dodge vertical overlaps
        my = (top + bot) / 2.0
        for uy in used_ys:
            if abs(my - uy) < (yhi - ylo) * 0.045:
                my = uy + (yhi - ylo) * 0.05
        used_ys.append(my)
        label_x = min(xr + label_offset, right_x - avg_gap)
        if xlo <= label_x <= xhi and ylo <= my <= yhi:
            tag = "TESTED" if z.get("tested") else "FRESH"
            ax.text(label_x, my, f"{z.get('setup', 'Zone')} · {tag}",
                    color=GOLD, alpha=0.92, fontsize=label_fs, va="center", ha="left",
                    zorder=4, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#000000", ec="none", alpha=0.55))


def _sd_nearest_fresh(zones, last, bias):
    if bias == 0:
        return None
    best = None
    for z in zones:
        if z["tested"] or z["demand"] != (bias == 1):
            continue
        d = (last - z["top"]) if bias == 1 else (z["bottom"] - last)
        if d < 0:
            continue
        if best is None or d < best[0]:
            best = (d, z)
    return None if best is None else (best[1], best[0] / last * 100.0)


def _resample_safe(df, rule):
    try:
        return df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    except Exception:
        fallback = rule.replace("ME", "M").replace("YE", "A")
        return df.resample(fallback).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()


def _rpt_price_chart(item, label, tf="D", fig_h=4.4):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter

    df = item["df"]
    if tf == "W":
        df = _resample_safe(df, "W-FRI")
    elif tf == "M":
        df = _resample_safe(df, "ME")
    w = df.tail(CHART_ZOOM[tf])
    if len(w) < 15:
        return None

    end = w.index[-1]
    start = w.index[0]
    BG, UP, DN = "#000000", "#26a69a", "#ef5350"
    x = mdates.date2num(w.index.to_pydatetime())
    o, h, l, c = (w[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    rng = float(h.max() - l.min())
    if rng <= 0:
        rng = max(abs(float(c[-1])) * 0.01, 1e-9)
    up = c >= o

    # Adaptive candle body width (matplotlib date units = days).
    # Fixed 0.65 is fine for daily but collapses to hairlines on W/M.
    if len(x) > 1:
        avg_gap = float(np.median(np.diff(x)))
    else:
        avg_gap = 1.0 if tf == "D" else (7.0 if tf == "W" else 30.0)
    if tf == "M":
        body_w = avg_gap * 0.58
        wick_lw = 1.5
        min_body = rng * 0.004
    elif tf == "W":
        body_w = avg_gap * 0.62
        wick_lw = 1.15
        min_body = rng * 0.0025
    else:
        body_w = min(0.70, avg_gap * 0.72)
        wick_lw = 0.90
        min_body = rng * 0.0012

    fig, ax = plt.subplots(figsize=(11.2, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    for mask, col in ((up, UP), (~up, DN)):
        if not mask.any():
            continue
        ax.vlines(x[mask], l[mask], h[mask], color=col, lw=wick_lw, zorder=3, capstyle="round")
        body = np.maximum(np.abs(c[mask] - o[mask]), min_body)
        ax.bar(x[mask], body, bottom=np.minimum(o[mask], c[mask]),
               width=body_w, color=col, lw=0, zorder=3, align="center")
    pad = rng * 0.07
    ax.set_ylim(float(l.min()) - pad, float(h.max()) + pad)
    right_pad = CHART_RIGHT_PAD.get(tf, CHART_RIGHT_PAD_DAYS)
    ax.set_xlim(mdates.date2num(start), mdates.date2num(end) + right_pad)
    dec = _px_dec(c[-1]); last = float(c[-1])
    # Compute zones on the SAME timeframe as the candles so left/right/born
    # indices and dates line up with the bars being drawn. Projecting daily
    # zones onto W/M puts them in the wrong place (daily bar indices ≠ weekly bars).
    zones = compute_sd_zones(df)
    _rpt_draw_zones(ax, df, zones, float(l.min()) - pad, float(h.max()) + pad, dec, tf=tf)
    ax.axhline(last, color="#888888", lw=0.7, ls=":", zorder=3)
    ax.annotate(f"{last:.{dec}f}", xy=(1, last), xycoords=("axes fraction", "data"), xytext=(4, 0),
                textcoords="offset points", va="center", ha="left", fontsize=7.5, color="white",
                bbox=dict(boxstyle="square,pad=0.25", fc=DN, ec=DN), annotation_clip=False)
    ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{dec}f"))
    if tf == "M":
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: str(mdates.num2date(v).year)))
    elif tf == "W":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, p: (lambda d: str(d.year) if d.month == 1 else d.strftime("%b"))(mdates.num2date(v))))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, p: (lambda d: str(d.year) if d.month == 1 else d.strftime("%b"))(mdates.num2date(v))))
    ax.tick_params(colors="#d1d4dc", labelsize=7.5, length=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.text(0.005, 0.985, f"{label}  ·  {tf}  ·  {_rpt_tvlabel(item).split(' · ')[0]}", transform=ax.transAxes,
            color="white", fontsize=9, fontweight="bold", va="top")
    return _rpt_fig_to_png(fig)


def _rpt_strip_chart(sp):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    today = sp["today"]
    x0, x1 = today - pd.Timedelta(days=14), today + pd.Timedelta(days=56)
    fig, ax = plt.subplots(figsize=(11.2, 1.6), facecolor=RPT["bg"])
    ax.set_facecolor(RPT["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 3.3)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors=RPT["muted"], labelsize=8)
    row = 1.75
    for dirn, b_ in sorted(sp["best"].items(), key=lambda kv: kv[1]["start"]):
        col = RPT["green"] if dirn == 1 else RPT["red"]
        tint = "#dcebdc" if dirn == 1 else "#f5d9d9"
        xs = max(mdates.date2num(b_["start"]), mdates.date2num(x0))
        xe = min(mdates.date2num(b_["end"]), mdates.date2num(x1))
        ax.add_patch(plt.Rectangle((xs, row), xe - xs, 0.55, fc=tint, ec=col, lw=1.4))
        ax.text(xs, row + 0.66, f"{'BUY' if dirn == 1 else 'SELL'} {b_['win'] * 100:.0f}%   "
                f"{b_['start'].strftime('%d %b')} – {b_['end'].strftime('%d %b')}  ·  {b_['shape']}  ·  recent {b_['rec_wins']}/{b_['rec_cases']}",
                fontsize=7.5, color=col, va="bottom", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", fc=RPT["bg"], ec="none", alpha=0.92))
        row = 0.45
    ax.axvline(mdates.date2num(today), color="#222", lw=2, zorder=3)
    ax.text(mdates.date2num(today) + 0.6, 3.15, "now", ha="left", va="top", fontsize=8, fontweight="bold")
    return _rpt_fig_to_png(fig)


def _rpt_map_chart(sp):
    import matplotlib.pyplot as plt
    d, today = sp["d"], sp["today"]
    yr = int(today.year)

    def path(y):
        seg = d[(d.index >= pd.Timestamp(year=y, month=1, day=1)) & (d.index <= pd.Timestamp(year=y, month=12, day=31))]
        return None if len(seg) < 300 else (seg.values[:365] / seg.values[0] - 1.0) * 100.0

    avgs = {}
    for n, col, lab in ((15, "#8a8a8a", "15-year average"), (10, RPT["blue"], "10-year average"), (5, RPT["green"], "5-year average")):
        ps = [p for p in (path(yr - k) for k in range(1, n + 1)) if p is not None]
        if ps:
            m = min(len(p) for p in ps)
            avgs[lab] = (np.mean([p[:m] for p in ps], axis=0), col)
    cur = d[(d.index >= pd.Timestamp(year=yr, month=1, day=1)) & (d.index <= today)]
    fig, ax = plt.subplots(figsize=(11.2, 2.2), facecolor=RPT["bg"])
    _rpt_style_ax(ax)
    for lab, (arr, col) in avgs.items():
        ax.plot(range(len(arr)), arr, color=col, lw=1.6, label=lab)
    if len(cur) > 1:
        ax.plot(range(len(cur)), (cur.values / cur.values[0] - 1.0) * 100.0, color="#222", lw=1.8, label=f"{yr} so far")
    ax.axhline(0, color="#777", lw=0.8)
    ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("% from year start", fontsize=8, color=RPT["muted"])
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.13))
    return _rpt_fig_to_png(fig)


def _rpt_vol_chart(sp):
    import matplotlib.pyplot as plt
    d, today = sp["d"], sp["today"]
    rv = d.pct_change().rolling(20, min_periods=20).std() * np.sqrt(365) * 100.0
    yr = int(today.year)
    typ = []
    for k in range(1, sp["years"] + 1):
        seg = rv[(rv.index >= pd.Timestamp(year=yr - k, month=1, day=1)) & (rv.index <= pd.Timestamp(year=yr - k, month=12, day=31))]
        if len(seg) >= 300:
            typ.append(seg.values[:365])
    if not typ:
        return None, np.nan, np.nan
    m = min(len(t) for t in typ)
    typical = pd.Series(np.nanmean([t[:m] for t in typ], axis=0)).rolling(5, min_periods=1).mean().values
    cur = rv[(rv.index >= pd.Timestamp(year=yr, month=1, day=1)) & (rv.index <= today)].dropna()
    fig, ax = plt.subplots(figsize=(11.2, 2.6), facecolor=RPT["bg"])
    _rpt_style_ax(ax)
    ax.plot(range(len(typical)), typical, color="#8a8a8a", lw=2, label="typical (15-yr)")
    ax.fill_between(range(len(typical)), 0, typical, color="#8a8a8a", alpha=0.08)
    if len(cur):
        pos = [(i - pd.Timestamp(year=yr, month=1, day=1)).days for i in cur.index]
        ax.plot(pos, cur.values, color="#222", lw=1.8, label=f"{yr} so far")
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("annualized vol %", fontsize=8, color=RPT["muted"])
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    doy = (today - pd.Timestamp(year=yr, month=1, day=1)).days
    typ_now = float(typical[min(doy, len(typical) - 1)])
    cur_now = float(cur.iloc[-1]) if len(cur) else np.nan
    return _rpt_fig_to_png(fig), cur_now, typ_now


def _rpt_analog_chart(sp):
    import matplotlib.pyplot as plt
    d, today = sp["d"], sp["today"]
    yr = int(today.year)
    current = d[(d.index >= pd.Timestamp(year=yr, month=1, day=1)) & (d.index <= today)]
    if len(current) < 30:
        return None, []
    cur_path = (current.values / current.values[0] - 1.0) * 100.0
    matches = []
    for k in range(1, 26):
        y = yr - k
        seg = d[(d.index >= pd.Timestamp(year=y, month=1, day=1)) & (d.index <= pd.Timestamp(year=y, month=12, day=31))]
        if len(seg) < 360:
            continue
        hist_ytd = seg.values[:len(cur_path)]
        if len(hist_ytd) < len(cur_path):
            continue
        h = (hist_ytd / hist_ytd[0] - 1.0) * 100.0
        if np.std(h) == 0 or np.std(cur_path) == 0:
            continue
        corr = float(np.corrcoef(h, cur_path)[0, 1])
        full = (seg.values / seg.values[0] - 1.0) * 100.0
        rest = float(full[-1] - full[len(cur_path) - 1])
        matches.append((corr, y, full, len(cur_path), rest))
    matches.sort(reverse=True)
    top = matches[:4]
    fig, ax = plt.subplots(figsize=(11.2, 2.6), facecolor=RPT["bg"])
    _rpt_style_ax(ax)
    ax.plot(range(len(cur_path)), cur_path, color="#222", lw=2.2, label=f"{yr} (this year)")
    cols = [RPT["blue"], RPT["green"], RPT["amber"], "#9b59b6"]
    for (corr, y, full, cut, rest), col in zip(top, cols):
        ax.plot(range(len(full)), full, color=col, lw=1.3, alpha=0.85, label=f"{y} ({corr:+.2f})")
    if top:
        ax.axvline(len(cur_path), color="#666", lw=1, ls="--")
        ax.text(len(cur_path) + 1, ax.get_ylim()[1] * 0.9, "today", fontsize=7.5, color=RPT["muted"])
    ax.axhline(0, color="#777", lw=0.6)
    ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("% from year start", fontsize=8, color=RPT["muted"])
    ax.legend(loc="upper center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.14))
    return _rpt_fig_to_png(fig), top


def _rpt_styles():
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    H = colors.HexColor
    S_ = {}
    S_["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=9.6, leading=14.2, textColor=H(RPT["ink"]), spaceAfter=6)
    S_["small"] = ParagraphStyle("small", fontName="Helvetica", fontSize=7.4, leading=10, textColor=H("#9a9a9a"))
    S_["note"] = ParagraphStyle("note", fontName="Helvetica", fontSize=8, leading=11.5, textColor=H("#777777"))
    S_["label"] = ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=H(RPT["muted"]))
    S_["kicker"] = ParagraphStyle("kicker", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=H(RPT["muted"]))
    S_["title"] = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=H(RPT["ink"]))
    S_["right"] = ParagraphStyle("right", fontName="Helvetica", fontSize=8.6, leading=12.5, textColor=H(RPT["muted"]), alignment=2)
    S_["rightb"] = ParagraphStyle("rightb", fontName="Helvetica-Bold", fontSize=9, leading=12.5, textColor=H(RPT["ink"]), alignment=2)
    S_["h"] = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=13.5, leading=24, textColor=H(RPT["ink"]), spaceAfter=6)
    S_["cell"] = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, leading=11, textColor=H(RPT["ink"]))
    S_["big"] = ParagraphStyle("big", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=H(RPT["ink"]))
    return S_



def _rpt_combined_essay(pk, cp, sp, r, bias, nm, packs):
    """Long mentor-style combined-bias narrative built from live evidence."""
    paras = []
    full = _rpt_market_display(nm)[1]
    rolename = {"commercial": "commercials", "retail": "retail"}

    # ---- Positioning paragraph ----
    if cp is None:
        paras.append("Positioning data is unavailable this week, so the combined bias rests on seasonality alone if a strong window is present.")
    else:
        comm, ret = cp["comm"], cp["ret"]
        if cp["bias"] == 0:
            # Neutral COT - explain why extremes may exist but no side
            ret_lit = ret["lit"] if ret else 0
            comm_lit = comm["lit"] if comm else 0
            if ret_lit or comm_lit:
                paras.append(
                    f"<b>Where positioning stands, and the group that matters has nothing.</b> "
                    f"Commercials are at an extreme on {comm_lit} of 5 lookbacks"
                    f"{', the commercial panel is thin' if comm_lit == 0 else ''}. "
                    f"Retail is lit on {ret_lit} of 5"
                    f"{', but their treatment is not qualified on this market, so it carries no weight' if ret_lit and not cp.get('edge') else ''}. "
                    f"Extreme positioning is not the same thing as an edge. The picture can look dramatic; "
                    f"the history does not support acting on it. Positioning gives no side."
                )
            else:
                paras.append(
                    "<b>Positioning is neutral this week.</b> Neither group is at an extreme that gives a side, "
                    "so COT has nothing to say."
                )
        else:
            # Has a COT side - walk the qualifying rows
            edge = cp.get("edge")
            bits = []
            if edge:
                lab = dict(RPT_TABLE_ORDER).get(edge["key"], edge["key"])
                t4 = edge["t"]
                bits.append(
                    f"The strongest read is the <b>{lab}</b> lookback on {rolename[edge['role']]}: "
                    f"{t4['win']:.0f}% over {edge['h']}w, {t4['wins']} of {t4['cases']} cases, "
                    f"{t4['avg']:+.2f}% average."
                )
            # Walk precomputed track records at the primary horizon
            pos_rows = []
            weak_rows = []
            h0 = COT_EDGE_HORIZONS[0]
            tracks = cp.get("tracks") or {}
            for role_k, st in (("commercial", comm), ("retail", ret)):
                for key, lab in RPT_TABLE_ORDER:
                    L = st["layers"].get(key)
                    if not L or L.get("pdir") == 0:
                        continue
                    tr = tracks.get((role_k, key, h0))
                    if not tr:
                        continue
                    if tr["cases"] >= REPORT_EDGE_MIN_CASES and tr["win"] >= REPORT_EDGE_MIN_WIN and tr["avg"] > 0:
                        pos_rows.append((role_k, lab, tr))
                    elif tr["cases"] >= 2 and tr["avg"] <= 0:
                        weak_rows.append((role_k, lab, tr))
            if pos_rows:
                desc = "; ".join(
                    f"{lab} on {rolename[rk]} {tr['win']:.0f}% of {tr['cases']} ({tr['avg']:+.2f}%)"
                    for rk, lab, tr in pos_rows[:5]
                )
                paras.append(
                    f"<b>Qualifying evidence.</b> {_nw(len(pos_rows)).capitalize()} row"
                    f"{'s clear' if len(pos_rows) != 1 else ' clears'} our bar"
                    f"{' from both groups' if len({rk for rk,_,_ in pos_rows}) > 1 else ''}: {desc}."
                )
            if bits:
                paras.append(bits[0])
            if weak_rows:
                w = weak_rows[0]
                paras.append(
                    f"<b>One honest caution.</b> The {w[1]} on {rolename[w[0]]} is the weak row: "
                    f"{w[2]['win']:.0f}% of {w[2]['cases']} with {w[2]['avg']:+.2f}% at four weeks. "
                    f"Several supporting rows rest on small samples. The floor is six cases — not a comfortable sample."
                )

    # ---- Cross-market compare if EUR/GBP style pair exists ----
    peer = None
    if nm in ("GBPUSD", "British Pound", "GBP"):
        for cand in ("EURUSD", "Euro FX", "EUR"):
            if cand in packs and packs[cand].get("cot"):
                peer = (cand, packs[cand])
                break
    elif nm in ("EURUSD", "Euro FX", "EUR"):
        for cand in ("GBPUSD", "British Pound", "GBP"):
            if cand in packs and packs[cand].get("cot"):
                peer = (cand, packs[cand])
                break
    if peer and cp and cp.get("bias"):
        pnm, ppk = peer
        pcp = ppk["cot"]
        pfull = _rpt_market_display(pnm)[1]
        if pcp and pcp.get("bias") == cp["bias"]:
            paras.append(
                f"<b>Read this against the {pfull} page.</b> Both carry the same badge this week. "
                f"Compare the qualifying rows and sample sizes before sizing either leg."
            )
        elif pcp and pcp.get("bias") == -cp["bias"]:
            paras.append(
                f"<b>Read this against the {pfull} page.</b> Same group of majors, opposite evidence this week. "
                f"That contrast is worth holding in view before treating either as a standalone."
            )
        elif pcp and pcp.get("bias") == 0:
            paras.append(
                f"<b>Read this against the {pfull} page.</b> {pfull} has no COT side this week while "
                f"{full} does. The divergence is part of the read."
            )

    # ---- Seasonality line-up ----
    if sp and sp.get("main"):
        m = sp["main"]
        sdir = m["dir"]
        days_left = max(0, (m["end"] - sp["today"]).days) if hasattr(m["end"], "toordinal") else 0
        try:
            days_left = max(0, (pd.Timestamp(m["end"]) - pd.Timestamp(sp["today"])).days)
        except Exception:
            days_left = 0
        if cp and cp.get("bias") and sdir == -cp["bias"]:
            paras.append(
                f"<b>How the two engines line up, and the disagreement has {days_left} day"
                f"{'s' if days_left != 1 else ''} left.</b> Seasonality reads "
                f"{'bearish' if sdir == -1 else 'bullish'} at {m['strength'].lower()}: a {m['shape']} "
                f"{'sell' if sdir == -1 else 'buy'} from {m['start'].strftime('%d %B')} to "
                f"{m['end'].strftime('%d %B')}, closing in its direction {m['win']*100:.0f}% of the last "
                f"{m['cases']} years, averaging {m['avg']:+.1f}%. "
                f"COT carries the side; the seasonal opponent is a reason for patience on entry, not a reason to have no view."
            )
        elif cp and cp.get("bias") and sdir == cp["bias"]:
            paras.append(
                f"<b>How the two engines line up.</b> Seasonality confirms the COT side on a {m['shape']} window "
                f"({m['start'].strftime('%d %b')}–{m['end'].strftime('%d %b')}), "
                f"{m['win']*100:.0f}% of {m['cases']}, average {m['avg']:+.1f}%. The two inputs reinforce each other."
            )
        elif (not cp or not cp.get("bias")) and m["strength"] == "STRONG":
            paras.append(
                f"<b>Seasonality is the only read.</b> A {m['strength'].lower()}, "
                f"{'still-firing' if sp['today'] >= m['start'] and sp['today'] <= m['end'] else 'upcoming'} "
                f"window: {m['win']*100:.0f}% of {m['cases']}, average {m['avg']:+.1f}%. "
                f"Size it as a single-engine call, not a confirmed one."
            )

    # ---- Closing instruction ----
    word = {1: "buy", -1: "sell", 0: "stand aside"}[bias]
    demand = {1: "Demand", -1: "Supply", 0: "a zone"}[bias]
    if bias:
        paras.append(
            f"<b>The combined bias.</b> {RPT_DIRWORD[bias].title()}."
        )
    else:
        paras.append(
            "<b>The combined bias.</b> Neutral. Stand aside until COT or a strong seasonal window gives a side. "
            "A neutral read is not an excuse to trade the middle."
        )
    return paras


def _rpt_build(table, prices, cot, latest_cot, only=None, chart_data=None, prior_state=None, mode="full"):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
                                    TableStyle, PageBreak, Image, KeepTogether, HRFlowable, Flowable)
    H = colors.HexColor
    S_ = _rpt_styles()
    today = dt.date.today()
    lazy = (mode == "lazy")
    date_str = today.strftime("%d %B %Y").lstrip("0")
    path = (f"Weekly Outlook {date_str} (Lazy).pdf" if lazy
            else f"Weekly Outlook {date_str}.pdf")
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            stamp = dt.datetime.now().strftime("%H%M%S")
            path = (f"Weekly Outlook {date_str} (Lazy) {stamp}.pdf" if lazy
                    else f"Weekly Outlook {date_str} {stamp}.pdf")
    PW, PH = landscape(A4)
    M = 14 * mm
    CW = PW - 2 * M
    state = {"section": "Weekly Outlook"}
    # Outline / bookmark tracking for the PDF sidebar (section → market hierarchy)
    _outline = {"seen": set(), "seq": 0}

    class SetSection(Flowable):
        """Zero-size flowable: updates footer label and optionally registers a clickable outline entry."""
        def __init__(self, txt, level=0, key=None, bookmark=True):
            Flowable.__init__(self)
            self.txt = txt
            self.level = max(0, min(int(level), 2))
            self.key = key
            self.bookmark = bookmark

        def wrap(self, aw, ah):
            return (0, 0)

        def draw(self):
            state["section"] = self.txt
            if not self.bookmark:
                return
            _outline["seq"] += 1
            bkey = self.key or f"bm_{_outline['seq']}"
            if bkey in _outline["seen"]:
                bkey = f"{bkey}_{_outline['seq']}"
            _outline["seen"].add(bkey)
            try:
                self.canv.bookmarkPage(bkey)
                # closed=1 for top-level folders keeps the sidebar tidy
                self.canv.addOutlineEntry(self.txt, bkey, level=self.level, closed=(self.level == 0))
            except Exception:
                pass

    def on_bg(c, doc):
        c.saveState()
        c.setFillColor(H(RPT["bg"]))
        c.rect(0, 0, PW, PH, stroke=0, fill=1)
        c.restoreState()

    def on_page(c, doc):
        c.saveState()
        c.setStrokeColor(H(RPT["rule"]))
        c.setLineWidth(0.6)
        c.line(M, 10 * mm, PW - M, 10 * mm)
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(H(RPT["ink"]))
        c.drawString(M, 6 * mm, "WEEKLY OUTLOOK")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(H(RPT["muted"]))
        c.drawRightString(PW - M, 6 * mm, f"{state['section']}  |  page {doc.page}")
        c.restoreState()

    doc = BaseDocTemplate(path, pagesize=(PW, PH), leftMargin=M, rightMargin=M, topMargin=11 * mm, bottomMargin=14 * mm,
                          title="Weekly Outlook " + today.strftime("%d %B %Y").lstrip("0"))
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(M, 14 * mm, CW, PH - 25 * mm, leftPadding=0, rightPadding=0,
                                                              topPadding=0, bottomPadding=0)], onPage=on_bg, onPageEnd=on_page)])

    def P(t, style="body"):
        return Paragraph(t, S_[style])

    # Unicode arrow support: register a TTF that has U+2192 if one exists, else fall back to ">>".
    ARROW = "&gt;&gt;"
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        for fp in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                   "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
            if os.path.exists(fp):
                try:
                    pdfmetrics.registerFont(TTFont("RptArrow", fp))
                    ARROW = '<font name="RptArrow">\u2192</font>'
                    break
                except Exception:
                    continue
    except Exception:
        pass

    def header(kicker, title, r1, r2="", r3=""):
        left = [P(_esc(kicker).upper(), "kicker"), P(_esc(title), "title")]
        right = ([P(_esc(r1), "rightb")] if r1 else []) + ([P(_esc(r2), "right")] if r2 else []) + ([P(_esc(r3), "right")] if r3 else [])
        t = Table([[left, right]], colWidths=[CW * 0.66, CW * 0.34])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        return [t, Spacer(1, 2), HRFlowable(width="100%", thickness=2.4, color=H(RPT["ink"]), spaceAfter=7)]

    def section(label, right=""):
        t = Table([[P(_esc(label).upper(), "label"), P(_esc(right), "right")]], colWidths=[CW * 0.55, CW * 0.45])
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.8, H("#d8d2ca")), ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        return [Spacer(1, 4), t, Spacer(1, 6)]

    def card(content, accent=None, pad=11):
        t = Table([[content]], colWidths=[CW])
        st = [("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])), ("BOX", (0, 0), (-1, -1), 0.7, H("#ddd6cd")),
              ("LEFTPADDING", (0, 0), (-1, -1), pad + (5 if accent else 0)), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
              ("TOPPADDING", (0, 0), (-1, -1), pad - 2), ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 4)]
        if accent:
            st.append(("LINEBEFORE", (0, 0), (0, -1), 4.5, H(accent)))
        t.setStyle(TableStyle(st))
        return t

    def note(text):
        t = Table([[P(text, "note")]], colWidths=[CW])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])), ("BOX", (0, 0), (-1, -1), 0.7, H("#ddd6cd")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                               ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        return t

    def img(buf, width=CW):
        from reportlab.lib.utils import ImageReader
        ir = ImageReader(buf)
        w, h = ir.getSize()
        buf.seek(0)
        return Image(buf, width=width, height=width * h / w)

    def colored(txt, col, bold=True):
        return f'<font color="{col}">{"<b>" if bold else ""}{txt}{"</b>" if bold else ""}</font>'

    story = []
    out = table.copy()
    wanted = {o.lower() for o in only} if only else {o.lower() for o in REPORT_FORCE_MARKETS}
    weekly = prices.resample("W-FRI").last()
    mk = {m[0]: m for m in MARKETS}

    log.info("Building %s report: seasonal windows and COT packs ...", "lazy" if mode == "lazy" else "full")
    packs = {}
    for _, r in out.iterrows():
        nm = str(r["Market"]).replace(" (inv)", "")
        if nm not in mk:
            continue
        _, group, ticker, code, invert = mk[nm]
        pk = {"row": r, "name": nm, "group": group, "ticker": ticker, "invert": invert,
              "cot": None, "season": None}
        try:
            if ticker and ticker in prices.columns:
                pk["season"] = _rpt_season_pack(prices[ticker], today)
        except Exception as e:
            log.warning("Seasonal pack failed for %s: %s", nm, e)
        try:
            if code and group != "Indices":
                sub = cot[cot["code"] == code.lstrip("0")]
                if len(sub) >= COT_LOOKBACK_SHORT:
                    pk["cot"] = _rpt_cot_pack(sub, weekly[ticker] if ticker in weekly.columns else None, invert)
        except Exception as e:
            log.warning("COT pack failed for %s: %s", nm, e)
        packs[nm] = pk

    # ===== PAGE 1: Seasonal Overview =====
    story.append(SetSection("Seasonal overview", level=0, key="seasonal_overview"))
    from reportlab.lib.styles import ParagraphStyle as _PS
    from reportlab.lib.enums import TA_RIGHT as _TAR
    strong = [p for p in packs.values() if p["season"] and p["season"]["main"] and p["season"]["main"]["strength"] == "STRONG"]
    buys = [p for p in strong if p["season"]["main"]["dir"] == 1]
    sells = [p for p in strong if p["season"]["main"]["dir"] == -1]
    if len(strong) >= 3 and (len(buys) >= len(strong) * 0.7 or len(sells) >= len(strong) * 0.7):
        theme = f"{len(strong)} markets carry a strong seasonal window right now and {max(len(buys), len(sells))} of them lean the same way."
    elif strong:
        theme = "No single dominant cross-market seasonal theme this week, read each market on its own window below."
    else:
        theme = "No market has a strong seasonal window entering inside the horizon this week."
    _ov_cell = _PS("ovc", parent=S_["cell"], fontSize=10.5, leading=13)
    _ov_cell_r = _PS("ovcr", parent=_ov_cell, alignment=_TAR)
    _ov_lab = S_["label"]
    _ov_lab_r = _PS("ovlr", parent=_ov_lab, alignment=_TAR)
    _ov_body = _PS("ovb", parent=S_["body"], fontSize=12, leading=17)
    first_ov = True
    for g in SECTION_ORDER:
        members = [p for p in packs.values() if _rpt_section(p["name"], p["group"]) == g]
        if not members:
            continue
        def _rank_key(p):
            m = p["season"]["main"] if p.get("season") else None
            if not m:
                return (2, 0, 0)
            return (0 if m["strength"] == "STRONG" else 1, -m["win"], -abs(m.get("avg", 0)))
        members = sorted(members, key=_rank_key)
        story += header("Weekly market outlook", "Seasonal Overview", REPORT_BRAND,
                        "Bias · Seasonality", today.strftime("%d %B %Y").lstrip("0"))
        if first_ov:
            story.append(card([P("<font size=22><b>Seasonal overview</b></font>", "h"), Spacer(1, 3),
                               Paragraph(_esc(theme), _ov_body)], RPT["green"]))
            story.append(Spacer(1, 6))
            first_ov = False
        g_label = "Currencies" if g == "Majors" else g
        story += section("Best seasonal windows in play",
                         f"{g_label} · entry within -2 to +8 weeks · {REPORT_SEASON_YEARS}-year basis")
        rows = [[Paragraph("MARKET", _ov_lab), Paragraph("TRADE", _ov_lab), Paragraph("WINDOW", _ov_lab_r),
                 Paragraph("LENGTH", _ov_lab), Paragraph("WIN", _ov_lab_r), Paragraph("CASES", _ov_lab_r),
                 Paragraph("SEASONAL STRENGTH", _ov_lab_r)]]
        spans = []
        for p in members:
            sp2 = p["season"]; m2 = sp2["main"] if sp2 else None
            _, _, ov_label = _rpt_market_display(p["name"])
            if not m2:
                rows.append([Paragraph(_esc(ov_label), _ov_cell),
                             Paragraph('<font color="#8a8a8a">no window in the -2/+8wk horizon</font>', _ov_cell),
                             "", "", "", "", ""])
                spans.append(("SPAN", (1, len(rows) - 1), (6, len(rows) - 1)))
                continue
            tc = RPT["green"] if m2["dir"] == 1 else RPT["red"]
            sc = RPT["ink"] if m2["strength"] == "STRONG" else RPT["amber"]
            rows.append([Paragraph(f"<b>{_esc(ov_label)}</b>", _ov_cell),
                         Paragraph(colored("BUY" if m2["dir"] == 1 else "SELL", tc), _ov_cell),
                         Paragraph(f"{m2['start'].strftime('%d %b')} – {m2['end'].strftime('%d %b')}", _ov_cell_r),
                         Paragraph(f'<font color="#8a8a8a">{m2["len"]}d · {m2["shape"]}</font>', _ov_cell),
                         Paragraph(f"<b>{m2['win'] * 100:.0f}%</b>", _ov_cell_r),
                         Paragraph(f"{m2['wins']} of {m2['cases']}", _ov_cell_r),
                         Paragraph(colored(m2["strength"], sc), _ov_cell_r)])
        t = Table(rows, colWidths=[CW * w for w in (0.27, 0.08, 0.15, 0.13, 0.08, 0.10, 0.19)], repeatRows=1)
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, H("#e6e0d8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                               ("LINEBELOW", (0, 0), (-1, 0), 1.0, H("#cfc8bf"))] + spans))
        story += [t, Spacer(1, 10)]
        story.append(note("<b>How to read this overview.</b> Seasonality is the <b>add-on</b> to the COT bias, it never trades alone. "
                          "This page shows, for every market in the outlook, the strongest seasonal window whose entry falls in the "
                          "coming weeks, ranked by how much conviction sits behind it: <b>hit-rate</b>, <b>average move</b>, its "
                          "<b>recent record</b>, and whether the <b>surrounding weeks</b> work too. That last check matters most, if a "
                          "window only pays on its exact dates it was likely an accident of two dates; if the whole period leans the "
                          "same way, the edge is broader than that."))
        story.append(PageBreak())

    # ===== PAGE 2: Dashboard =====
    story.append(SetSection("Quick decision dashboard", level=0, key="dashboard"))
    story += header("Weekly market outlook", "Quick Decision Dashboard", REPORT_BRAND,
                    "COT + valuation + seasonality", today.strftime("%d %B %Y").lstrip("0"))
    def _edge_dir(lbl):
        lbl = str(lbl)
        return "Bullish" if "Bullish" in lbl else "Bearish" if "Bearish" in lbl else None
    _ed = out["EdgeFinder"].map(_edge_dir)
    aligned = out[_ed.notna() & (out["COTBias"] == _ed)]
    story += section("Best aligned markets", f"latest COT report {latest_cot or '-'}")
    hdr = ["MARKET", "SECTION", "CORE", "COT SIDE", "COT", "VALUATION", "VAL STRENGTH", "SEASONALITY", "EDGE FINDER"]
    rows = [[P(h, "label") for h in hdr]]
    def tint(v):
        return colored(_esc(v), RPT["green"] if v == "Bullish" else RPT["red"] if v == "Bearish" else RPT["muted"])
    shown = aligned.sort_values(["AttentionScore"], ascending=False).head(14)
    for _, r in sorted(shown.iterrows(), key=lambda it: _rpt_sort_key(str(it[1]["Market"]).replace(" (inv)", ""), it[1]["Group"])):
        nm = str(r["Market"]).replace(" (inv)", "")
        rows.append([P(f"<b>{_esc(nm)}</b>", "cell"),
                     P(_esc(_rpt_section(nm, r["Group"])), "cell"),
                     P("yes" if r["Core"] == "yes" else "", "cell"),
                     P(tint(r["COTBias"]), "cell"), P(_esc(_fmt(r.get("COT"), 0)), "cell"),
                     P(tint(r["Valuation"]), "cell"), P(_esc(r["ValStrength"]), "cell"),
                     P(tint(r["Seasonality"]), "cell"),
                     P(_esc(str(r.get("EdgeFinder", "-"))), "cell")])
    if len(rows) == 1:
        rows.append([P("No market has COT and valuation aligned this week.", "cell")] + [""] * 8)
    t = Table(rows, colWidths=[CW * w for w in (0.12, 0.10, 0.05, 0.09, 0.05, 0.10, 0.12, 0.11, 0.26)], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])), ("LINEBELOW", (0, 0), (-1, -1), 0.4, H("#e6e0d8")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t, Spacer(1, 8)]
    story += section("Reading the dashboard")
    story.append(note("<b>Aligned</b> = the Edge Finder's best historical setup points the same way as the commercial COT side. "
                      "<b>Seasonality</b> only tempers conviction unless COT has no side and the window is STRONG."))
    story.append(PageBreak())

    # ===== What changed since last week =====
    # Uses scan_state.json from the previous successful run (bias, edge rows, season window).
    prior_mkts = (prior_state or {}).get("markets") or {}
    changes = []  # (sort_key, market, section, lines)
    if prior_mkts:
        for _, r in out.iterrows():
            nm = str(r.get("Market", "")).replace(" (inv)", "")
            if not nm:
                continue
            prev = prior_mkts.get(nm)
            if not prev:
                continue
            lines = []
            cur_bias = str(r.get("COTBias", "Neutral"))
            prev_bias = str(prev.get("bias", "Neutral"))
            if cur_bias != prev_bias:
                lines.append(f"COT side flipped from <b>{_esc(prev_bias)}</b> to <b>{_esc(cur_bias)}</b>")
            _q = _safe_float(r.get("EdgeFinderWin"))
            cur_edge = 1 if (r.get("EdgeFinder", "-") != "-" and np.isfinite(_q)) else 0
            prev_edge = int(prev.get("qualifying_rows") or 0)
            if cur_edge and not prev_edge:
                lines.append(f"Gained a qualifying edge: {_esc(str(r.get('EdgeFinder', '-')))}")
            elif prev_edge and not cur_edge:
                lines.append("Lost its qualifying historical edge")
            cur_sw = str(r.get("SeasonWindow", "-"))
            prev_sw = str(prev.get("season_window", "-"))
            cur_ss = r.get("SeasonWindowStrength")
            prev_ss = prev.get("season_strength")
            if cur_sw != prev_sw and (cur_sw != "-" or prev_sw != "-"):
                if prev_sw in ("-", "", "None", "none") and cur_sw not in ("-", "", "None", "none"):
                    lines.append(f"New seasonal window: {_esc(cur_sw)}"
                                 + (f" · {_esc(str(cur_ss))}" if cur_ss else ""))
                elif cur_sw in ("-", "", "None", "none") and prev_sw not in ("-", "", "None", "none"):
                    lines.append("Seasonal window dropped out of the horizon")
                else:
                    lines.append(f"Seasonal window changed ({_esc(prev_sw)} → {_esc(cur_sw)})")
            elif cur_ss and prev_ss and str(cur_ss) != str(prev_ss):
                lines.append(f"Seasonal strength moved from {_esc(str(prev_ss))} to {_esc(str(cur_ss))}")
            prev_lit = int(prev.get("cot_lit") or 0)
            cur_lit = int(_safe_float(r.get("CommExtremeCount")) or 0) if np.isfinite(_safe_float(r.get("CommExtremeCount"))) else 0
            if cur_lit != prev_lit and (cur_lit > 0 or prev_lit > 0):
                if cur_lit > prev_lit:
                    lines.append(f"Commercial extremes expanded ({prev_lit} → {cur_lit} of 5)")
                else:
                    lines.append(f"Commercial extremes contracted ({prev_lit} → {cur_lit} of 5)")
            if lines:
                sec = _rpt_section(nm, r.get("Group", ""))
                changes.append((_rpt_sort_key(nm, r.get("Group", "")), nm, sec, lines))
        changes.sort(key=lambda z: z[0])

    story.append(SetSection("What changed since last week", level=0, key="what_changed"))
    story += header("Weekly market outlook", "What changed since last week", REPORT_BRAND,
                    "Bias · edge · seasonality", today.strftime("%d %B %Y").lstrip("0"))
    prior_run = (prior_state or {}).get("run_date") or "—"
    story += section("Week-over-week moves", f"compared with run of {prior_run}")
    if not prior_mkts:
        story.append(note(
            "No prior run is stored yet. After the next weekly scan, this page will list markets whose "
            "COT side flipped, that gained or lost a qualifying historical edge, or whose seasonal window changed."
        ))
    elif not changes:
        story.append(note(
            f"No material changes versus the run of {prior_run}. "
            "No COT side flipped, no qualifying edge was gained or lost, and no seasonal window moved."
        ))
    else:
        story.append(note(
            f"<b>{len(changes)} market{'s' if len(changes) != 1 else ''}</b> moved versus the run of {prior_run}. "
            "Only flips in COT side, qualifying edge, commercial extreme count, or the seasonal window are listed — "
            "routine noise is left out."
        ))
        story.append(Spacer(1, 6))
        ch_rows = [[P(h, "label") for h in ("MARKET", "SECTION", "WHAT CHANGED")]]
        for _, nm, sec, lines in changes:
            ch_rows.append([
                P(f"<b>{_esc(nm)}</b>", "cell"),
                P(_esc(sec), "cell"),
                P(" · ".join(lines), "cell"),
            ])
        ct = Table(ch_rows, colWidths=[CW * w for w in (0.18, 0.14, 0.68)], repeatRows=1)
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, H("#e6e0d8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(ct)
    story.append(PageBreak())

    # ===== Shape Scan =====
    story.append(SetSection("Seasonal shape scan", level=0, key="shape_scan"))
    story += header("Weekly market outlook", "Seasonal Overview, Shape Scan", REPORT_BRAND,
                    "Bias · Seasonality", today.strftime("%d %B %Y").lstrip("0"))
    story += section("Shape scan", "is THIS year following each market's seasonal shape?")
    ranked = []
    for nm, pk in packs.items():
        sp = pk.get("season")
        if not sp or not np.isfinite(sp.get("shape", np.nan)):
            continue
        ranked.append((float(sp["shape"]), nm, pk))
    ranked.sort(reverse=True)
    # Keep top majors + any with shape >= LOOSE, cap list length
    rows = [[P(x, "label") for x in ("MARKET", "SHAPE MATCH", "", "READ")]]
    track_w = CW * 0.30
    for sh, nm, pk in ranked[:14]:
        _, full, _ = _rpt_market_display(nm)
        if sh >= SHAPE_CLOSE:
            read, col = "tracking closely", RPT["ink"]
        elif sh >= SHAPE_LOOSE:
            read, col = "loosely tracking", RPT["amber"]
        else:
            read, col = "off-pattern", "#9a9a9a"
        # filled fraction mapped from [-1, +1] -> [0, 1]
        frac = max(0.02, min(1.0, (sh + 1.0) / 2.0))
        fill_w = track_w * frac
        rest_w = max(0.5, track_w - fill_w)
        fill = Table([[""]], colWidths=[fill_w], rowHeights=[8])
        fill.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), H(col)),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        rest = Table([[""]], colWidths=[rest_w], rowHeights=[8])
        rest.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), H("#ece7e0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        bar = Table([[fill, rest]], colWidths=[fill_w, rest_w])
        bar.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        score_col = col if sh >= SHAPE_LOOSE else "#9a9a9a"
        rows.append([
            P(f"<b>{_esc(full)}</b>", "cell"),
            P(f'<font color="{score_col}"><b>{sh:+.2f}</b></font>', "cell"),
            bar,
            P(f'<font color="{col}">{read}</font>', "cell"),
        ])
    t = Table(rows, colWidths=[CW * w for w in (0.30, 0.10, 0.32, 0.28)])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, H("#e6e0d8")),
        ("BACKGROUND", (0, 0), (-1, 0), H(RPT["card"])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(card([
        P("<font color='#2563eb'><b>WHY THE SHAPE SCAN MATTERS</b></font>", "label"),
        P("A seasonal average can have a brilliant 15-year record and still be the wrong guide "
          "<b>this year</b>, if price is not actually following the pattern. The shape scan measures "
          "exactly that: how closely this year's path tracks its own seasonal rhythm."),
        P(f"<b>High and positive</b> (top of the list) {ARROW} the seasonal is in control, so its forward window "
          f"is worth weighting. <b>Low or negative</b> {ARROW} the market is off-pattern this year, so we "
          "<b>discount</b> the seasonal no matter how strong its history, and lean harder on COT."),
    ], RPT["blue"]))
    story.append(PageBreak())

    # ===== CHAPTERS =====
    cands = []
    filler = []
    for nm, p in packs.items():
        cp = p["cot"]; r = p["row"]
        if wanted and nm.lower() not in wanted:
            continue
        if not wanted and (not cp or cp["bias"] == 0) and not (p["season"] and p["season"]["main"]
                                                              and p["season"]["main"]["strength"] == "STRONG") \
                and nm not in REPORT_ALWAYS_CHAPTER:
            filler.append((float(r.get("AttentionScore", 0) or 0), (cp["lit"] if cp else 0), nm))
            continue
        cands.append((float(r.get("AttentionScore", 0) or 0), (cp["lit"] if cp else 0), nm))
    cands.sort(reverse=True)
    filler.sort(reverse=True)
    if not wanted and not lazy:
        # keep coverage broad: top up to the chapter cap with the next-most-interesting markets
        cands += filler[: max(0, REPORT_MAX_CHAPTERS - len(cands))]
    max_ch = REPORT_MAX_CHAPTERS if not lazy else min(8, REPORT_MAX_CHAPTERS)
    ordered = ([c for c in cands if c[2] in REPORT_ALWAYS_CHAPTER] +
               [c for c in cands if c[2] not in REPORT_ALWAYS_CHAPTER])
    chosen = [c[2] for c in ordered][: (len(cands) if wanted else max_ch)]
    chosen.sort(key=lambda n: _rpt_sort_key(n, packs[n]["group"]))
    log.info("Chapters: %s", ", ".join(chosen) if chosen else "none")
    _bookmarked_sections = set()

    for nm in chosen:
        pk = packs[nm]
        cp = pk["cot"]; sp = pk["season"]; r = pk["row"]
        _short, full, _ov = _rpt_market_display(nm)
        title_pair = f"{_short} ({full})" if _short != full else full
        bias = cp["bias"] if cp else 0
        season_promoted = (bias == 0 and sp and sp.get("main") and sp["main"]["strength"] == "STRONG")
        if season_promoted:
            bias = sp["main"]["dir"]
        word = {1: "buy", -1: "sell", 0: "stand aside"}[bias]
        demand = {1: "Demand", -1: "Supply", 0: "a zone"}[bias]
        rolename = {"commercial": "commercials", "retail": "retail"}
        in_play = bool(sp and sp.get("main") and sp["today"] >= sp["main"]["start"] and sp["today"] <= sp["main"]["end"])
        sec_name = _rpt_section(nm, pk["group"])
        # One top-level outline folder per section, then each market underneath
        if sec_name not in _bookmarked_sections:
            story.append(SetSection(sec_name, level=0, key=f"sec_{sec_name}"))
            _bookmarked_sections.add(sec_name)
        mkt_key = f"mkt_{nm}"
        mkt_bookmarked = False

        def _ensure_mkt_bookmark():
            nonlocal mkt_bookmarked
            if not mkt_bookmarked:
                story.append(SetSection(full, level=1, key=mkt_key))
                mkt_bookmarked = True

        # ========== LAZY MODE: one compact page per market ==========
        if lazy:
            _ensure_mkt_bookmark()
            story += header(f"{full}", f"{full}", REPORT_BRAND, "Quick read",
                            today.strftime("%d %B %Y").lstrip("0"))
            cot_txt = "no COT side"
            if cp:
                cot_txt = (f"Commercials {RPT_DIRWORD[cp['comm']['pdir']].lower()} on {cp['comm']['lit']} of 5, "
                           f"retail {RPT_DIRWORD[cp['ret']['pdir']].lower()} on {cp['ret']['lit']} of 5"
                           if cp["comm"]["pdir"] or cp["ret"]["pdir"]
                           else "no extreme on either group")
            sea_txt = "no strong window"
            if sp and sp.get("main"):
                m = sp["main"]
                sea_txt = (f"{'BUY' if m['dir']==1 else 'SELL'} {m['start'].strftime('%d %b')}–{m['end'].strftime('%d %b')} "
                           f"· {m['strength']} · {m['win']*100:.0f}% of {m['cases']} · {m['shape']}")
            lines = [
                P(f"<b>{_esc(full)}</b>  <font color='{_rpt_dircolor(bias)}' size=16><b>{RPT_DIRWORD[bias]}</b></font>", "h"),
                P(f"<b>COT.</b> {cot_txt}.", "body"),
                P(f"<b>Seasonality.</b> {sea_txt}.", "body"),
            ]
            if bias:
                lines.append(P(f"<b>Action.</b> Look to {word} in <b>{demand}</b> only. "
                               f"A bias is not a trade — wait for price to reach a fresh zone.", "body"))
            else:
                lines.append(P("<b>Action.</b> Stand aside. Neither COT nor seasonality gives a side this week.", "body"))
            if cp and cp.get("edge"):
                ed = cp["edge"]
                t4 = ed["t"]
                lab = dict(RPT_TABLE_ORDER).get(ed["key"], ed["key"])
                lines.append(P(f"<b>Edge.</b> Best track record: {lab} on {ed['role']} — "
                               f"{t4['win']:.0f}% over {ed['h']}w, {t4['wins']} of {t4['cases']}, {t4['avg']:+.2f}% avg.", "body"))
            # Nearest daily zone (if any)
            item = (chart_data or {}).get(nm)
            if item and bias:
                if "zones" not in item:
                    item["zones"] = compute_sd_zones(item["df"])
                near = _sd_nearest_fresh(item["zones"], float(item["df"]["Close"].iloc[-1]), bias)
                if near:
                    nz_, dist_ = near
                    dd = _px_dec(item["df"]["Close"].iloc[-1])
                    lines.append(P(f"<b>Nearest {demand}.</b> {_esc(nz_['setup'])}, "
                                   f"{nz_['bottom']:.{dd}f}–{nz_['top']:.{dd}f} "
                                   f"({dist_:.1f}% {'below' if bias == 1 else 'above'} price).", "body"))
            story.append(card(lines, _rpt_dircolor(bias)))
            story.append(Spacer(1, 6))
            # Daily chart only
            if item:
                try:
                    cbuf = _rpt_price_chart(item, full, tf="D")
                    if cbuf is not None:
                        story.append(img(cbuf))
                        story.append(P(f"Daily · {item['desc']} · preferred proximal zones.", "small"))
                except Exception as e:
                    log.warning("Lazy daily chart failed for %s: %s", nm, e)
            if bias == 1:
                read_txt = ("<b>THE READ</b>  Long in daily Demand only. A bias is not a trade — wait for the zone.")
            elif bias == -1:
                read_txt = ("<b>THE READ</b>  Short in daily Supply only. A bias is not a trade — wait for the zone.")
            else:
                read_txt = ("<b>THE READ</b>  No side this week. Stand aside.")
            story.append(Spacer(1, 4))
            story.append(card([P(read_txt, "body")], _rpt_dircolor(bias)))
            story.append(PageBreak())
            continue

        if cp:
            comm, ret = cp["comm"], cp["ret"]
            _ensure_mkt_bookmark()
            try:
                cot_day = pd.Timestamp(latest_cot)
                cot_txt_hdr = f"COT as of {cot_day.strftime('%A')} {cot_day.strftime('%Y-%m-%d')}"
            except Exception:
                cot_txt_hdr = f"COT as of {latest_cot or '-'}"
            story += header("Weekly market outlook", f"Week of {date_str}", REPORT_BRAND,
                            "Step 1 · Bias from COT", cot_txt_hdr)
            nr, nc = ret["lit"], comm["lit"]
            if comm["pdir"] and ret["pdir"] == comm["pdir"]:
                relation = "and here both point the same way"
            elif comm["pdir"] and ret["pdir"] == -comm["pdir"]:
                relation = "and here they point in opposite directions, so commercials carry the side"
            elif comm["pdir"]:
                relation = "and only the commercials have an extreme, so they carry the side"
            else:
                relation = "and the commercials have no extreme, so there is no valid COT side"
            if nr and nc:
                intro = (f"Retail and commercials are <b>both at an extreme</b>, on <b>{_nw(nr)} of five</b> lookbacks "
                         f"(retail) and <b>{_nw(nc)} of five</b> (commercials). We fade the crowd, we follow the "
                         f"commercials, {relation}. ")
            else:
                intro = (f"Retail is at an extreme on <b>{_nw(nr)} of five</b> lookbacks, commercials on "
                         f"<b>{_nw(nc)} of five</b>. We fade the crowd, we follow the commercials, {relation}. ")
            if cp["bias"]:
                intro += (f"<b>Step 1 gives us the side: we are looking to {word}.</b> "
                          f"Step 2 gives us the zone, and we only {word} in {demand}.")
            elif season_promoted:
                intro += f"<b>Step 1 has no side, but seasonality is STRONG and takes the lead. We are looking to {word}.</b>"
            else:
                intro += "<b>Step 1 gives us no side this week.</b>"
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.enums import TA_RIGHT
            _nm_style = ParagraphStyle("vnm", parent=S_["h"], fontSize=22, leading=25)
            _dir_style = ParagraphStyle("vdir", parent=S_["h"], fontSize=22, leading=25, alignment=TA_RIGHT)
            _pill_style = ParagraphStyle("vpill", parent=S_["cell"], fontSize=9, leading=11, textColor=H(RPT["muted"]))
            pill = Table([[Paragraph(f"<b>{_esc(_rpt_tvlabel(pk))}</b>", _pill_style)]], colWidths=[84], hAlign="LEFT")
            pill.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, H("#cfc8bf")),
                                      ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                                      ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            top_row = Table([[Paragraph(f"<b>{_esc(full)}</b>", _nm_style), pill,
                              Paragraph(f"<font color='{_rpt_dircolor(bias)}'><b>{RPT_DIRWORD[bias]}</b></font>", _dir_style)]],
                            colWidths=[CW * 0.30, CW * 0.22, CW * 0.40])
            top_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                         ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("ALIGN", (1, 0), (1, 0), "LEFT")]))
            story.append(card([top_row, Spacer(1, 4), P(intro)], _rpt_dircolor(bias)))
            story.append(Spacer(1, 8))

            def role_card(title, st, role, action, actcol):
                side_txt = ("LONG extreme" if st["side"] == 1 else "SHORT extreme") if st["pdir"] else "no extreme"
                side_col = (RPT["green"] if st["side"] == 1 else RPT["red"]) if st["pdir"] else RPT["muted"]
                if role == "retail":
                    side_col = RPT["red"] if st["pdir"] else RPT["muted"]
                accent = RPT["red"] if role == "retail" else RPT["green"]
                tint = RPT["redtint"] if role == "retail" else RPT["greentint"]

                # ---- Row 1: 5 small colored square chips + "N of 5 lookbacks agree" ----
                chip_size = 12
                chip_gap = 4
                n_lit = sum(1 for (key, lab) in RPT_CHIP_ORDER
                            if bool(st["pdir"]) and st["layers"][key]["pdir"] == st["pdir"])
                chip_cells = [""] * 9
                chip_styles = []
                for i in range(5):
                    col_i = i * 2
                    chip_styles.append(("BACKGROUND", (col_i, 0), (col_i, 0), H(accent if i < n_lit else "#ffffff")))
                    chip_styles.append(("BOX", (col_i, 0), (col_i, 0), 0.9, H(accent)))
                chip_table = Table([chip_cells], colWidths=[chip_size, chip_gap] * 4 + [chip_size], rowHeights=[chip_size])
                chip_table.setStyle(TableStyle(chip_styles + [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]))
                agree_txt = f"<b>{st['lit'] if st['pdir'] else 0} of 5</b> lookbacks agree"
                chips_row = Table(
                    [[chip_table, P(f"<font color='#666666' size='8'>{agree_txt}</font>", "cell")]],
                    colWidths=[chip_size * 5 + chip_gap * 4 + 6, (CW / 2 - 32) - (chip_size * 5 + chip_gap * 4 + 6)]
                )
                chips_row.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("LEFTPADDING", (1, 0), (1, 0), 6),
                ]))

                # ---- Row 2: wide labeled lookback boxes (Multi-yr … 6 mo) ----
                cells, styles_ = [], []
                box_w = (CW / 2 - 32) / 5
                for i, (key, lab) in enumerate(RPT_CHIP_ORDER):
                    L = st["layers"][key]
                    lit_ = bool(st["pdir"]) and L["pdir"] == st["pdir"]
                    cells.append(P(f'<font color="{accent if lit_ else "#999999"}" size="7.5"><b>{lab}</b></font>', "cell"))
                    styles_.append(("BACKGROUND", (i, 0), (i, 0), H(tint if lit_ else "#f0ebe4")))
                    styles_.append(("BOX", (i, 0), (i, 0), 0.6, H(accent if lit_ else "#d4cdc3")))
                ct = Table([cells], colWidths=[box_w] * 5)
                ct.setStyle(TableStyle(styles_ + [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 1),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ]))

                inner = [P(f"<font color='#8a8a8a'><b>{title.upper()}</b></font>", "label"),
                         chips_row,
                         Spacer(1, 5),
                         P(f"<font color='{side_col}'><b>{side_txt}</b></font>", "big"),
                         Spacer(1, 3),
                         ct,
                         Spacer(1, 5),
                         P(f"<font color='{actcol}'><b>{action}</b></font>", "cell")]
                t_ = Table([[inner]], colWidths=[CW / 2 - 4])
                t_.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                        ("BOX", (0, 0), (-1, -1), 0.7, H("#ddd6cd")),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
                return t_

            rdir, cdir = ret["pdir"], comm["pdir"]
            ract = f"FADE the crowd  {ARROW}  {RPT_DIRWORD[rdir]}" if rdir else f"FADE the crowd  {ARROW}  no read"
            cact = f"FOLLOW the smart money  {ARROW}  {RPT_DIRWORD[cdir]}" if cdir else f"FOLLOW the smart money  {ARROW}  no read"
            pair = Table([[role_card("Retail | small traders", ret, "retail", ract, _rpt_dircolor(rdir)),
                           role_card("Commercials", comm, "commercial", cact, _rpt_dircolor(cdir))]],
                         colWidths=[CW / 2, CW / 2])
            pair.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                      ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [pair]
            story += section("Positioning", "CFTC data | 5-year window | nested lookback bands")
            story.append(img(_rpt_cot_chart(ret, "retail")))
            story.append(Spacer(1, 4))
            story.append(img(_rpt_cot_chart(comm, "commercial")))
            story.append(PageBreak())

            def track_table(role, st, h):
                rows = [[P(x, "label") for x in ("LOOKBACK", "WEEKS IN", "WIN RATE", "CASES", "AVG", "HISTORY")]]
                sty = [("LINEBELOW", (0, 0), (-1, -1), 0.4, H("#e6e0d8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                       ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                       ("LINEBELOW", (0, 0), (-1, 0), 1.0, H("#cfc8bf"))]
                best_key = cp["edge"]["key"] if cp["edge"] and cp["edge"]["role"] == role and h == COT_EDGE_HORIZONS[0] else None
                for i, (key, lab) in enumerate(RPT_TABLE_ORDER, start=1):
                    tr = cp["tracks"].get((role, key, h))
                    if not tr:
                        rows.append([P(lab, "cell"), P('<font color="#9a9a9a">not at an extreme</font>', "cell"), "", "", "", ""])
                        sty.append(("SPAN", (1, i), (-1, i)))
                        continue
                    badge = (' <font color="#9a9a9a" size=7><i>past</i></font>' if tr["past"]
                               else ' <font color="#2563eb" size=7><b>live</b></font>')
                    if tr["cases"] == 0:
                        rows.append([P(lab + badge, "cell"), P('<font color="#9a9a9a">no comparable cases</font>', "cell"), "", "", "", ""])
                        sty.append(("SPAN", (1, i), (-1, i)))
                        sty.append(("BACKGROUND", (0, i), (-1, i), H(RPT["lav"])))
                        continue
                    wcol = "#9a9a9a" if tr["win"] <= 33 else RPT["ink"]
                    wtxt = f"<b>{tr['win']:.0f}%</b>" if tr["win"] >= 60 else f"{tr['win']:.0f}%"
                    acol = RPT["green"] if tr["avg"] > 0 else RPT["red"]
                    rows.append([P(lab + badge, "cell"), P(f"<b>{tr['weeks_in']}w</b>", "cell"),
                                 P(f'<font color="{wcol}">{wtxt}</font>', "cell"),
                                 P(f"{tr['wins']} of {tr['cases']}", "cell"),
                                 P(colored(f"{tr['avg']:+.2f}%", acol), "cell"),
                                 P(f'<font color="#9a9a9a">{tr["history"]} yrs</font>', "cell")])
                    sty.append(("BACKGROUND", (0, i), (-1, i), H(RPT["lav"])))
                    if best_key == key:
                        sty.append(("BOX", (0, i), (-1, i), 1.2, H(RPT["blue"])))
                tt = Table(rows, colWidths=[CW / 2 * f for f in (0.20, 0.15, 0.17, 0.16, 0.16, 0.16)])
                tt.setStyle(TableStyle(sty))
                return tt

            def pair_of_tables(h):
                left = [P(f"<b>RETAIL | FADE | {h}-WEEK RESULT</b>", "cell"), Spacer(1, 3), track_table("retail", ret, h)]
                right = [P(f"<b>COMMERCIALS | FOLLOW | {h}-WEEK RESULT</b>", "cell"), Spacer(1, 3), track_table("commercial", comm, h)]
                tt = Table([[left, right]], colWidths=[CW / 2, CW / 2])
                tt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, 0), 0),
                                        ("RIGHTPADDING", (0, 0), (0, 0), 8), ("LEFTPADDING", (1, 0), (1, 0), 8),
                                        ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
                return tt

            story += header(f"{full} · {_rpt_tvlabel(pk)}", "The evidence", REPORT_BRAND, "Track Record",
                            "What happened last time this extreme showed up")
            story += section("Track record", "every past case, capped at the 6 most recent")
            story.append(pair_of_tables(COT_EDGE_HORIZONS[0]))
            story += [Spacer(1, 8), note("<b>Weeks in</b> = how long that lookback's extreme has been running. "
                                         "Rows still inside their window carry a <b>live</b> badge; older rows are marked <b>past</b>. "
                                         "Each row caps at the 6 most recent comparable cases."),
                      Spacer(1, 8)]
            story.append(pair_of_tables(COT_EDGE_HORIZONS[-1]))
            story.append(PageBreak())

            story += header(f"{full} · {_rpt_tvlabel(pk)}", "Does it pay?", REPORT_BRAND, "Edge Finder",
                            "Every case traceable to a row on this page")
            story += section("Edge Finder", "the setups this market has actually paid for")
            ed = cp["edge"]
            paras = []
            if ed:
                t4 = ed["t"]; lab = dict(RPT_TABLE_ORDER)[ed["key"]]
                paras.append(f"<b>The strongest read is the {lab} lookback, on {rolename[ed['role']]}.</b> "
                             + (f"A commercials {'long' if comm['side'] == 1 else 'short'} extreme has led to a "
                                f"{'BUY' if bias == 1 else 'SELL'} " if ed["role"] == "commercial" else
                                f"A retail {'long' if ret['side'] == 1 else 'short'} extreme, faded, has led to a "
                                f"{'BUY' if bias == 1 else 'SELL'} ")
                             + f"<b>{t4['win']:.0f}% of the time over {ed['h']} weeks</b>, "
                               f"{t4['wins']} of {t4['cases']} cases, {t4['avg']:+.2f}% average move. "
                               f"That is the highlighted row in the table below: every case is on the page.")
                near = []
                for (role_k, key_k, h_k), tr_k in cp["tracks"].items():
                    if h_k != COT_EDGE_HORIZONS[0] or not tr_k or tr_k["past"]:
                        continue
                    if role_k == ed["role"] and key_k == ed["key"]:
                        continue
                    if tr_k["cases"] >= 3 and tr_k["win"] >= REPORT_EDGE_MIN_WIN and tr_k["avg"] > 0:
                        near.append((role_k, key_k, tr_k))
                if near:
                    txt_parts = []
                    for role_k, key_k, tr_k in near[:3]:
                        lab_k = dict(RPT_TABLE_ORDER)[key_k]
                        txt_parts.append(f"the {lab_k} on {rolename[role_k]} reads <b>{tr_k['win']:.0f}%</b>, "
                                         f"{tr_k['wins']} of {tr_k['cases']} cases")
                    paras.append(f"<b>{_nw(len(near)).capitalize()} more lookback{'s' if len(near) != 1 else ''} point the same way"
                                 f"{', and both are below the case floor' if len(near) == 2 else ''}.</b> "
                                 + "; ".join(txt_parts) + ". Both are in the table below.")
                    paras.append("No rule may point at any of these, each one is too thin on its own, and thin "
                                 "evidence lies in the direction that flatters you. But two lookbacks agreeing is a "
                                 "stronger statement than one, and pretending they are not there would be its own kind "
                                 "of dishonesty. They are all on the table below. If price reaches a good zone in that "
                                 "direction, they add to the trade. Step 1 does not have to carry the trade on its own.")
            else:
                paras.append(f"<b>No single row clears our bar this week.</b> A row needs at least "
                             f"{_nw(REPORT_EDGE_MIN_CASES)} cases, {REPORT_EDGE_MIN_WIN}% win rate, positive average, and to be live.")
            story.append(card([P("<font color='#2e7d32'><b>EDGE FINDER, THE STRONGEST READ IN THIS MARKET</b></font>", "label")]
                              + [P(p_) for p_ in paras], RPT["green"] if ed else RPT["muted"]))
            story += section("Where this number comes from", f"Track Record at {COT_EDGE_HORIZONS[0]} weeks")
            story.append(pair_of_tables(COT_EDGE_HORIZONS[0]))
            story.append(PageBreak())

            # ===== Combined bias two-engine page =====
            story += header(f"{full} · {_rpt_tvlabel(pk)}", "Combined bias, the verdict", REPORT_BRAND,
                            "COT + Seasonality", "the conclusion of Step 1")
            story += section("The two engines", "each read on its own, before they are weighed")

            cot_dir_word = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}[cp["bias"]]
            cot_col = _rpt_dircolor(cp["bias"])
            cot_conv = ("High" if cp["lit"] >= 7 and not cp["conflict"]
                        else "Moderate" if cp["lit"] >= 4 else "None")
            sea_dir_word = "NEUTRAL"; sea_col = RPT["muted"]; sea_conv = "None"; sea_win = 0.05
            if sp and sp.get("main"):
                m2 = sp["main"]
                sea_dir_word = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}[m2["dir"]]
                sea_col = _rpt_dircolor(m2["dir"])
                sea_conv = f"{m2['strength'].title()} · {m2['shape']} window"
                sea_win = m2["win"]

            def _engine_box(title_txt, dir_txt, colr, conv_txt, bar_frac):
                bar_cols = 20
                filled = max(1, int(round(bar_frac * bar_cols))) if dir_txt != "NEUTRAL" else 1
                cells = [""] * bar_cols
                bar = Table([cells], colWidths=[(CW / 2 - 44) / bar_cols] * bar_cols, rowHeights=[8])
                bst = []
                for i in range(bar_cols):
                    bst.append(("BACKGROUND", (i, 0), (i, 0), H(colr if i < filled else "#e5e5e5")))
                bar.setStyle(TableStyle(bst + [("LEFTPADDING", (0, 0), (-1, -1), 0),
                                               ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                               ("TOPPADDING", (0, 0), (-1, -1), 0),
                                               ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
                return [P(f"<font color='#8a8a8a'><b>{title_txt}</b></font>", "label"),
                        P(f"<font color='{colr}' size=22><b>{dir_txt}</b></font>", "h"),
                        bar, Spacer(1, 3),
                        P(f"<font color='#777777'>Conviction: <b>{conv_txt}</b></font>", "cell")]

            cot_inner = _engine_box("COT POSITIONING, THE DRIVER", cot_dir_word, cot_col, cot_conv,
                                    (cp["lit"] / 5.0) if cp["bias"] else 0.05)
            sea_inner = _engine_box("SEASONALITY, THE ADD-ON", sea_dir_word, sea_col, sea_conv, sea_win)

            cot_box = Table([[cot_inner]], colWidths=[CW / 2 - 6])
            cot_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                         ("BOX", (0, 0), (-1, -1), 0.7, H("#ddd6cd")),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                         ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
            sea_box = Table([[sea_inner]], colWidths=[CW / 2 - 6])
            sea_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                         ("BOX", (0, 0), (-1, -1), 0.7, H("#ddd6cd")),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                         ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
            pair2 = Table([[cot_box, sea_box]], colWidths=[CW / 2, CW / 2])
            pair2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 4),
                                       ("LEFTPADDING", (1, 0), (1, 0), 4), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
            story.append(pair2)
            story.append(Spacer(1, 8))

            story += section("How they line up", "the rule")
            if cp["bias"] and sp and sp.get("main") and sp["main"]["dir"] == cp["bias"]:
                line1 = (f"<b>COT AND SEASONALITY AGREE</b><br/>Both engines point {RPT_DIRWORD[cp['bias']].lower()}. "
                         f"The seasonality window adds timing to a COT-led side, so the two inputs reinforce each other.")
            elif cp["bias"] and sp and sp.get("main") and sp["main"]["dir"] == -cp["bias"]:
                line1 = (f"<b>THE ENGINES DISAGREE</b><br/>COT reads {RPT_DIRWORD[cp['bias']].lower()}, "
                         f"seasonality reads {RPT_DIRWORD[sp['main']['dir']].lower()}. COT carries the side; "
                         f"the seasonal opponent tempers conviction rather than reversing it.")
            elif cp["bias"]:
                line1 = ("<b>COT HAS A SIDE, SEASONALITY IS SILENT</b><br/>No seasonal window is in play in the "
                         "coming weeks, so the bias rests on positioning alone.")
            elif sp and sp.get("main") and sp["main"]["strength"] == "STRONG":
                line1 = ("<b>COT IS NEUTRAL, SEASONALITY IS THE ONLY READ</b><br/>Positioning is neutral this week, "
                         "neither group is at an extreme that gives a side, so COT has nothing to say. There is no "
                         "conflict to resolve because there is no second opinion: the seasonal window is the only "
                         "read on the table, and it should be sized as a single-engine call, not a confirmed one.")
            else:
                line1 = ("<b>NEITHER ENGINE HAS A SIDE</b><br/>The scanner stands aside. A neutral read is not an "
                         "excuse to trade the middle.")
            story.append(card([P(line1)], RPT["blue"]))
            story.append(Spacer(1, 8))

            story += section("The combined bias", "the conclusion of Step 1")
            essay = _rpt_combined_essay(pk, cp, sp, r, bias, nm, packs)
            from reportlab.lib.styles import ParagraphStyle as _PS2
            from reportlab.lib.enums import TA_RIGHT as _TAR2
            _vbody = _PS2("vbody", parent=S_["body"], fontSize=11, leading=16.5)
            _vhead = _PS2("vhead", parent=S_["h"], fontSize=19, leading=23, textColor=H("#8a8a8a"))
            _vdir = _PS2("vdir2", parent=S_["h"], fontSize=26, leading=30, alignment=_TAR2)

            def _verdict_card(paras):
                head = Table([[Paragraph("<b>Combined verdict</b>", _vhead),
                               Paragraph(f"<font color='{_rpt_dircolor(bias)}'><b>{RPT_DIRWORD[bias]}</b></font>", _vdir)]],
                             colWidths=[CW * 0.5 - 20, CW * 0.5 - 20])
                head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                          ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
                rws = [[head]] + [[Paragraph(x, _vbody)] for x in paras]
                vt_ = Table(rws, colWidths=[CW])
                vt_.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 22), ("RIGHTPADDING", (0, 0), (-1, -1), 18),
                                         ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                                         ("TOPPADDING", (0, 0), (-1, 0), 14), ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                                         ("BOTTOMPADDING", (0, -1), (-1, -1), 16),
                                         ("LINEBEFORE", (0, 0), (0, -1), 4.5, H(_rpt_dircolor(bias))),
                                         ("LINEABOVE", (0, 0), (-1, 0), 0.7, H("#ddd6cd")),
                                         ("LINEBELOW", (0, -1), (-1, -1), 0.7, H("#ddd6cd")),
                                         ("LINEAFTER", (-1, 0), (-1, -1), 0.7, H("#ddd6cd"))]))
                return vt_

            closing = ""
            if bias:
                _w = {1: "buy", -1: "sell"}[bias]; _d = {1: "Demand", -1: "Supply"}[bias]
                _m = sp["main"] if sp and sp.get("main") else None
                _dl = 0
                if _m and _m["dir"] == -bias:
                    try:
                        _dl = max(0, (pd.Timestamp(_m["end"]) - pd.Timestamp(sp["today"])).days)
                    except Exception:
                        _dl = 0
                closing = "So work the four to eight week horizon, "
                if _dl > 0:
                    closing += (f"treat the next {_nw(_dl)} day{'s' if _dl != 1 else ''} as a reason to be patient on entry "
                                f"rather than absent, and let the zone decide. ")
                else:
                    closing += "and let the zone decide. "
                closing += f"We look to {_w} in {_d} only, and only when price reaches a fresh zone. A bias is not a trade."
            _disc = ("Seasonality is an analytical add-on, not a signal. A bias is not a trade, wait for price to reach "
                     "the zone. Educational content only. Not financial advice.")
            _kick = f"{title_pair} · {_rpt_tvlabel(pk)}"
            _today_txt = today.strftime("%d %B %Y").lstrip("0")
            FH_ = PH - 25 * mm
            _SAFE = 22

            def _hh(fl):
                try:
                    return fl.wrap(CW, 10000)[1]
                except Exception:
                    return 0

            _k = len(story) - 1
            while _k >= 0 and not isinstance(story[_k], PageBreak):
                _k -= 1
            used_first = sum(_hh(f_) for f_ in story[_k + 1:])
            cont_top = header(_kick, "Combined bias, the verdict", "Bias · the conclusion", "continued", _today_txt) + \
                section("The combined bias", "continued, how to read this week's pages")
            used_cont = sum(_hh(f_) for f_ in cont_top)
            _head_h = 62
            _pw = CW - 22 - 18 - 6

            def _ph(x):
                return Paragraph(x, _vbody).wrap(_pw, 10000)[1] + 13

            pages_ = []; cur_ = []; h_ = _head_h; avail_ = FH_ - used_first - _SAFE
            for x in essay:
                ph_ = _ph(x)
                if cur_ and h_ + ph_ > avail_:
                    pages_.append(cur_); cur_ = []; h_ = _head_h; avail_ = FH_ - used_cont - _SAFE
                cur_.append(x); h_ += ph_
            if cur_:
                pages_.append(cur_)
            for pi_, chunk in enumerate(pages_):
                if pi_ > 0:
                    story.append(PageBreak())
                    story += cont_top
                story.append(_verdict_card(chunk))
            if closing:
                close_h = _head_h + _ph(closing) + 30
                left_ = avail_ - h_
                if left_ < close_h + 10:
                    story.append(PageBreak())
                    story += cont_top
                else:
                    story.append(Spacer(1, 10))
                story.append(_verdict_card([closing]))
            story.append(Spacer(1, 8))
            story.append(P(_disc, "small"))
            story.append(PageBreak())

        if sp is not None and sp.get("main"):
            m = sp["main"]; sdir = m["dir"]
            sw = "bullish" if sdir == 1 else "bearish"
            scol = _rpt_dircolor(sdir)
            in_play_local = sp["today"] >= m["start"] and sp["today"] <= m["end"]
            _ensure_mkt_bookmark()
            story.append(SetSection(f"{full} · Seasonality", bookmark=False))
            story += header(f"{title_pair} · {_rpt_tvlabel(pk)}", "Seasonality, the horizon", REPORT_BRAND,
                            "Add-on to the COT read", today.strftime("%d %B %Y").lstrip("0"))
            story.append(card([P(f"<b>{_esc(full)}</b>   <font color='{scol}'><b>{sw.upper()}</b></font>", "h"),
                               P(f"Seasonality points <b>{sw}</b> over the coming weeks, on a <b>{m['shape']}</b> window "
                                 f"({m['start'].strftime('%d %b')} – {m['end'].strftime('%d %b')}). It closed in its direction "
                                 f"<b>{m['win'] * 100:.0f}%</b> of the last {m['cases']} years and has hit "
                                 f"<b>{m['rec_wins']} of its last {m['rec_cases']}</b>, for an average of "
                                 f"<b>{m['avg']:+.1f}%</b>. Edge: <b>{m['strength']}</b> · timing {m['timing']}.")], scol))
            story += section("Seasonal horizon", "-2 to +8 weeks · every window in play, drawn to scale")
            story.append(img(_rpt_strip_chart(sp)))
            story += section("Seasonal map", "5 / 10 / 15-year average path")
            story.append(img(_rpt_map_chart(sp), width=CW * 0.80))
            story.append(PageBreak())

            story += header(f"{title_pair} · {_rpt_tvlabel(pk)}", "Seasonality, the evidence", REPORT_BRAND,
                            "Add-on to the COT read", today.strftime("%d %B %Y").lstrip("0"))
            story += section("Windows in play", "every seasonal window entering the -2 to +8 week horizon")
            rows = [[P(x, "label") for x in ("ENTRY", "TRADE", "WINDOW", "LENGTH", "WIN", "CASES", "AVG")]]
            for dirn, b in sorted(sp["best"].items(), key=lambda kv: kv[1]["start"]):
                tag = ' <font color="#2563eb" size=7>driver</font>' if b is m else ""
                rows.append([P(f"<b>{b['start'].strftime('%d %b')}</b>{tag}", "cell"),
                             P(colored("BUY" if dirn == 1 else "SELL", _rpt_dircolor(dirn)), "cell"),
                             P(f"{b['start'].strftime('%d %b')} – {b['end'].strftime('%d %b')}", "cell"),
                             P(f"{b['len']}d · {b['shape']}", "cell"),
                             P(f"<b>{b['win'] * 100:.0f}%</b>", "cell"),
                             P(f"{b['wins']} of {b['cases']}", "cell"),
                             P(f"{b['avg']:+.1f}%", "cell")])
            t = Table(rows, colWidths=[CW * w for w in (0.14, 0.10, 0.22, 0.16, 0.12, 0.13, 0.13)])
            t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, H("#e6e0d8")),
                                   ("BACKGROUND", (0, 1), (-1, -1), H(RPT["lav"])),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            story += [t, Spacer(1, 6),
                      note("<b>Sharp beats broad.</b> A <b>sharp</b> window (≤3 weeks) is a concentrated edge.")]
            story += section("How strong is this window?", "edge · and how forgiving the timing is")
            shape_txt = "not available" if not np.isfinite(sp["shape"]) else f"{sp['shape']:+.2f}"
            # Left card: EDGE strength bar + TIMING
            str_col = RPT["ink"] if m["strength"] == "STRONG" else RPT["amber"]
            bar_cols = 16
            filled = 14 if m["strength"] == "STRONG" else (10 if m["strength"] == "MODERATE" else 5)
            bar_cells = [""] * bar_cols
            edge_bar = Table([bar_cells], colWidths=[(CW / 2 - 50) / bar_cols] * bar_cols, rowHeights=[9])
            ebst = [("BACKGROUND", (i, 0), (i, 0), H(str_col if i < filled else "#e8e2da")) for i in range(bar_cols)]
            edge_bar.setStyle(TableStyle(ebst + [
                ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            timing_label = m["timing"]
            timing_sub = ("most of the edge sits in these dates, timing matters" if timing_label == "tight"
                          else "clearly better than chance around these dates" if timing_label == "forgiving"
                          else "room on either side of these dates")
            left_inner = [
                P("<font color='#8a8a8a'><b>EDGE, HOW GOOD IS IT</b></font>", "label"),
                P(f"<font color='{str_col}' size=18><b>{m['strength']}</b></font>", "h"),
                edge_bar, Spacer(1, 8),
                P("<font color='#8a8a8a'><b>TIMING, HOW FORGIVING</b></font>", "label"),
                P(f"<font color='#333333' size=14><b>{timing_label}</b></font>", "h"),
                P(f"<font color='#777777' size=8>{timing_sub}</font>", "cell"),
                Spacer(1, 3),
                P(f"<font color='#999999' size=7.5>Whole-year shape match {shape_txt}"
                  f" · {'tracking closely' if np.isfinite(sp['shape']) and sp['shape'] >= SHAPE_CLOSE else 'loosely tracking' if np.isfinite(sp['shape']) and sp['shape'] >= SHAPE_LOOSE else 'off-pattern'}, context only</font>", "cell"),
            ]
            left_box = Table([[left_inner]], colWidths=[CW / 2 - 8])
            left_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                ("BOX", (0, 0), (-1, -1), 0.7, H("#ddd6cd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]))
            # Right card: VERDICT
            verdict = (f"A <b>{'strong, ' if m['strength'] == 'STRONG' else 'moderate, '}"
                       f"{'still-firing ' if in_play_local else 'upcoming '}</b>seasonal window: closed in its direction "
                       f"<b>{m['win'] * 100:.0f}%</b> of the last {m['cases']} years, averages <b>{m['avg']:+.1f}%</b>, "
                       f"and has hit <b>{m['rec_wins']} of its last {m['rec_cases']}</b>. "
                       f"{'Weight it.' if m['strength'] == 'STRONG' else 'Note it.'} "
                       f"For background, the year is "
                       f"{'broadly on-script' if np.isfinite(sp['shape']) and sp['shape'] >= SHAPE_CLOSE else 'only loosely on-script' if np.isfinite(sp['shape']) and sp['shape'] >= SHAPE_LOOSE else 'off-pattern'} "
                       f"({shape_txt}). Timing is <b>{m['timing']}</b>.")
            right_inner = [P("<font color='#8a8a8a'><b>VERDICT</b></font>", "label"), P(verdict)]
            right_box = Table([[right_inner]], colWidths=[CW / 2 - 8])
            right_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                ("BOX", (0, 0), (-1, -1), 0.7, H("#ddd6cd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]))
            strength_pair = Table([[left_box, right_box]], colWidths=[CW / 2, CW / 2])
            strength_pair.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4), ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ]))
            story.append(strength_pair)
            story.append(Spacer(1, 8))

            # Neighbour test with shifted windows
            story += section("The neighbour test", "how tight is the timing on these dates?")
            nb = m["neighbour"]
            nb_label = ("narrow" if m["timing"] == "tight"
                        else "wide" if m["timing"] == "forgiving" else "supported")
            nb_txt = (f"The surrounding weeks are clearly better than chance, so there is some room either side of these dates."
                      if m["timing"] == "forgiving"
                      else f"Most of the move sits inside these dates, so timing matters here."
                      if m["timing"] == "tight"
                      else f"The surrounding weeks are only just better than an even split.")
            story.append(card([
                P(f"<font color='#555555' size=14><b>{nb * 100:.0f}% average · {nb_label}</b></font>", "big")
                if np.isfinite(nb) else P("not available"),
                P(f"Randomly chosen windows land around 49% here, no directional lean. {nb_txt}"),
                P(f"The same window shifted ±1 and ±2 weeks, over the same {m['years']} years, hits on "
                  f"average <b>{nb * 100:.0f}%</b> against <b>{m['win'] * 100:.0f}%</b> on the actual dates. "
                  f"Whole-year shape match: <b>{shape_txt}</b>."),
            ], None))
            story.append(PageBreak())

            analog_buf, analog_top = _rpt_analog_chart(sp)
            if analog_buf is not None and analog_top:
                story.append(SetSection(f"{full} · Analogues", bookmark=False))
                story += header(f"{title_pair} · {_rpt_tvlabel(pk)}", "Seasonality, the analog cross-check", REPORT_BRAND,
                                "Cross-check", today.strftime("%d %B %Y").lstrip("0"))
                story += section("Analog years", "the historical years whose YTD path most resembles this one")
                story.append(img(analog_buf, width=CW * 0.86))
                ups = sum(1 for _, _, _, _, rest in analog_top if rest > 0)
                avg_rest = float(np.mean([rest for _, _, _, _, rest in analog_top]))
                rows = [[P(x, "label") for x in ("YEAR", "MATCH", "REST OF YEAR")]]
                for corr, y, _full, _cut, rest in analog_top:
                    rc = RPT["green"] if rest > 0 else RPT["red"]
                    rows.append([P(f"<b>{y}</b>", "cell"), P(f"{corr:+.2f}", "cell"),
                                 P(f'<font color="{rc}"><b>{rest:+.1f}%</b></font>', "cell")])
                t = Table(rows, colWidths=[CW * 0.10, CW * 0.10, CW * 0.14])
                t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, H("#e6e0d8")),
                                       ("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
                                       ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story.append(Spacer(1, 6))
                story.append(Table([[t, P(f"<b>THE READ.</b> They mostly went "
                                           f"<b>{'up' if avg_rest > 0 else 'down'}</b> from here, {ups} of "
                                           f"{len(analog_top)} higher, averaging <b>{avg_rest:+.1f}%</b>. "
                                           f"{'This cuts against the seasonal window above.' if (m and np.sign(avg_rest) != np.sign(m['avg'])) else 'This supports the seasonal window above.'}",
                                           "body")]],
                                    colWidths=[CW * 0.36, CW * 0.64],
                                    style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                      ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                                      ("RIGHTPADDING", (0, 0), (-1, -1), 0)])))
                story.append(PageBreak())

            vbuf, vcur, vtyp = _rpt_vol_chart(sp)
            if vbuf is not None:
                story += header(f"{title_pair} · {_rpt_tvlabel(pk)}", "Seasonality, volatility & sizing", REPORT_BRAND,
                                "Add-on to the COT read", today.strftime("%d %B %Y").lstrip("0"))
                story += section("Volatility season", f"when {full} gets loud, and how to size for it")
                story.append(img(vbuf))
                state_txt = "loud" if vcur > vtyp * 1.25 else "calm" if vcur < vtyp * 0.8 else "normal"
                story.append(Spacer(1, 4))
                story.append(P("Volatility seasonality sizes the trade. Where volatility is high, widen the stop and size smaller. "
                               "Where it is calm, a tighter stop is safe.", "note"))
                story.append(Spacer(1, 6))
                story.append(card([P("<font color='#2563eb'><b>USING THIS WITH THE WINDOW ABOVE</b></font>", "label"),
                                   P(f"Right now volatility is <b>{vcur:.1f}%</b> against a typical <b>{vtyp:.1f}%</b> "
                                     f"for this time of year, which is <b>{state_txt}</b>."),
                                   P("Volatility does not change the direction, it changes the size.")], RPT["blue"]))
                story.append(PageBreak())

        item = (chart_data or {}).get(nm)
        if item:
            _ensure_mkt_bookmark()
            story.append(SetSection(f"{full} · Step 2", bookmark=False))
            # Bias card ONCE, before the first chart — sets the side for all three TFs
            story += header(f"{title_pair} · {_rpt_tvlabel(item)}", "Step 2, the zones",
                            REPORT_BRAND, "Proprietary Supply & Demand",
                            "Monthly for reference · Weekly is HTF · Daily is LTF")
            if bias:
                s2 = (f"Step 1 gave us the side: <b>{RPT_DIRWORD[bias]}</b>. We look to {word} in "
                      f"<b>{demand}</b> only, so we wait for price to reach a fresh {demand} zone. "
                      f"A bias is not a trade.")
                if "zones" not in item:
                    item["zones"] = compute_sd_zones(item["df"])
                near = _sd_nearest_fresh(item["zones"], float(item["df"]["Close"].iloc[-1]), bias)
                dd = _px_dec(item["df"]["Close"].iloc[-1])
                if near:
                    nz_, dist_ = near
                    s2 += (f" Nearest fresh {demand} zone: <b>{_esc(nz_['setup'])}</b>, "
                           f"{nz_['bottom']:.{dd}f} to {nz_['top']:.{dd}f}, "
                           f"{dist_:.1f}% {'below' if bias == 1 else 'above'} price.")
                else:
                    s2 += f" There is no fresh {demand} zone on the daily chart right now."
                story.append(card([P(s2)], _rpt_dircolor(bias)))
                story.append(Spacer(1, 6))
            else:
                story.append(card([P(
                    "Step 1 gave us <b>no side</b> this week. Stand aside on zones until COT or a strong "
                    "seasonal window gives a direction. A neutral read is not an excuse to trade the middle."
                )], RPT["muted"]))
                story.append(Spacer(1, 6))

            for tf, subtitle, note_txt in (
                ("M", "Monthly — reference only, where are we in the big picture?",
                 "Monthly is context. We do not enter on monthly zones."),
                ("W", "Weekly, HTF — is this a zone worth trading into?",
                 "Weekly tells us whether a zone is worth trading into."),
                ("D", "Daily, LTF — execution: where do we actually get in?",
                 "Daily is the execution chart. This is where the trade is taken."),
            ):
                try:
                    cbuf = _rpt_price_chart(item, full, tf=tf, fig_h=4.4)
                    if cbuf is None:
                        continue
                    # Re-state TF header on each page after the first (first page already has the intro header)
                    if tf != "M":
                        story += header(f"{title_pair} · {_rpt_tvlabel(item)}", f"Step 2, the zones — {tf}",
                                        REPORT_BRAND, "Proprietary Supply & Demand", subtitle)
                    else:
                        story += section(f"Monthly", "reference only, where are we in the big picture?")
                    story.append(P(note_txt, "note"))
                    story.append(img(cbuf))
                    src = (f"Chart source: {item['desc']}, Yahoo continuous front-month, {tf} candles. "
                           f"Zones: Supply and Demand indicator ({ZONE_DRAW_MODE.lower()}, extend {ZONE_EXTEND_BARS} bars).")
                    if not item["futures"]:
                        src += " Cash/spot series (no futures feed available)."
                    story.append(P(src, "small"))
                    if tf == "D":
                        if bias == 1:
                            read_txt = ("<b>THE READ</b><br/>Long in daily Demand only, if price comes down to a fresh "
                                        "Demand zone. A bias is not a trade: wait for the zone. If the market runs "
                                        "without a pullback, there is no entry this week.")
                        elif bias == -1:
                            read_txt = ("<b>THE READ</b><br/>Short in daily Supply only, if price rallies into a fresh "
                                        "Supply zone. A bias is not a trade: wait for the zone. If the market falls "
                                        "without a bounce, there is no entry this week.")
                        else:
                            read_txt = ("<b>THE READ</b><br/>No side this week. Stand aside until the COT or the "
                                        "seasonal window gives a directional bias. A neutral read is not an excuse "
                                        "to trade the middle.")
                        story.append(Spacer(1, 4))
                        story.append(card([P(read_txt, "body")], _rpt_dircolor(bias)))
                        story.append(Spacer(1, 4))
                    story.append(PageBreak())
                except Exception as e:
                    try:
                        import matplotlib.pyplot as plt
                        plt.close("all")
                    except Exception:
                        pass
                    log.warning("Step 2 %s chart failed for %s: %s", tf, nm, e)

    rest = sorted((n for n in packs if n not in chosen), key=lambda n: _rpt_sort_key(n, packs[n]["group"]))
    if rest and not lazy:
        story.append(SetSection("Other markets", level=0, key="other_markets"))
        story += header("Weekly market outlook", "Other markets", REPORT_BRAND, "No full chapter this week",
                        today.strftime("%d %B %Y").lstrip("0"))
        rows = [[P(h, "label") for h in ("MARKET", "SECTION", "FUNDAMENTALS", "VALUATION", "SEASONALITY", "WHY NOT A CHAPTER")]]
        for n in rest:
            r_ = packs[n]["row"]
            if not packs[n]["cot"] or packs[n]["cot"]["bias"] == 0:
                if not (packs[n]["season"] and packs[n]["season"]["main"] and packs[n]["season"]["main"]["strength"] == "STRONG"):
                    why = "no commercial COT side and no strong seasonal window"
                else:
                    why = "not selected this run"
            else:
                why = "ranked below the chapter limit" if not wanted else "not selected for this run"
            rows.append([P(f"<b>{_esc(n)}</b>", "cell"),
                         P(_esc(_rpt_section(n, r_["Group"])), "cell"),
                         P(_esc(r_["Fundamentals"]), "cell"),
                         P(_esc(r_["Valuation"]), "cell"),
                         P(_esc(r_["Seasonality"]), "cell"),
                         P(why, "cell")])
        t = Table(rows, colWidths=[CW * w for w in (0.14, 0.11, 0.30, 0.10, 0.12, 0.23)], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, H("#e6e0d8")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(t)

    doc.build(story)
    return path


def _make_pdf_report_plain(table, latest_cot=None):
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"scan_report_{stamp}.pdf"
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="H3x", parent=styles["Heading3"], fontSize=10.5, leading=13, spaceBefore=5, spaceAfter=3))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.5, leading=12, spaceAfter=4))
    story = [Paragraph("MARKET SCANNER — REPORT BUILD FAILED", styles["Title2"]),
             Paragraph(f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | COT {latest_cot or '-'}", styles["Bodyx"]),
             Paragraph("The full report engine failed to build. See console traceback.", styles["Bodyx"])]
    if not table.empty:
        for _, r in table.head(20).iterrows():
            story.append(Paragraph(f"{r.get('Market', '-')}: {r.get('Fundamentals', '-')}", styles["H3x"]))
            story.append(Paragraph(str(r.get("AttentionReason", "-")), styles["Bodyx"]))
    doc.build(story)
    return path


def _make_pdf_report(table, latest_cot=None, prices=None, cot=None, only=None, chart_data=None, prior_state=None):
    """Build both the full (nerdy) and lazy PDFs. Returns (full_path, lazy_path)."""
    full_path = lazy_path = None
    if prices is not None and cot is not None:
        try:
            full_path = _rpt_build(table, prices, cot, latest_cot, only, chart_data, prior_state, mode="full")
        except Exception as exc:
            traceback.print_exc()
            log.error("Full report engine failed (%s); writing the plain fallback PDF instead.", exc)
            full_path = _make_pdf_report_plain(table, latest_cot)
        try:
            lazy_path = (_rpt_build(table, prices, cot, latest_cot, only, chart_data, prior_state, mode="lazy")
                         if BUILD_LAZY_PDF else None)
        except Exception as exc:
            traceback.print_exc()
            log.error("Lazy report engine failed (%s); skipping lazy PDF.", exc)
            lazy_path = None
        return full_path, lazy_path
    plain = _make_pdf_report_plain(table, latest_cot)
    return plain, None


def show(table, latest_cot=None, prices=None, cot=None, only=None, chart_data=None,
         prior_state=None, cot_status="ok", cot_msg=""):
    if table.empty:
        log.warning("Nothing to show.")
        return None
    out = table.drop(columns=[c for c in ["_strength", "_core"] if c in table.columns]).copy()
    for c in ["COT", "Comm26", "Comm1y", "Comm2y", "Comm3y", "Comm5y", "CommExtremeCount",
              "Spec26", "Retail26", "Retail1y", "Retail2y", "Retail3y", "Retail5y", "RetailExtremeCount"]:
        if c in out:
            out[c] = out[c].map(lambda v: "-" if not np.isfinite(_safe_float(v)) else f"{_safe_float(v):.0f}")
    out.to_csv("scan_results.csv", index=False)
    excel_path = _make_excel_report(table)
    full_path, lazy_path = _make_pdf_report(table, latest_cot, prices, cot, only, chart_data, prior_state)
    print("\n=== PRIORITY ATTENTION ===")
    for _, r in out.head(12).iterrows():
        print(f"\n{r.get('Market', '-')}: {r.get('AttentionReason', '-')}\n  {r.get('CombinedVerdict', '-')}")
    print("\n=== FULL SCAN saved to scan_results.csv ===")
    print(f"Excel report: {excel_path}")
    print(f"Full PDF  (nerdy): {full_path}")
    if lazy_path:
        print(f"Lazy PDF (fast):   {lazy_path}")

    # ---- Telegram delivery (success path) ----
    if telegram_enabled() and full_path:
        caption = f"Weekly Outlook · COT {latest_cot or '-'}"
        if cot_status in ("stale", "unchanged"):
            caption += f"\n⚠️ {cot_msg}"
        # Prefer the full PDF; also send Excel if present
        ok = telegram_send_document(full_path, caption=caption)
        if ok:
            log.info("Telegram: full PDF sent")
        if excel_path and Path(excel_path).exists():
            telegram_send_document(excel_path, caption=f"scan_results.xlsx · COT {latest_cot or '-'}")
        if lazy_path and Path(lazy_path).exists():
            telegram_send_document(lazy_path, caption=f"Lazy Outlook · COT {latest_cot or '-'}")
        # Short summary message with top attention markets
        top = []
        for _, r in out.head(6).iterrows():
            top.append(f"• {r.get('Market', '-')}: {r.get('COTBias', '-')} / {r.get('Fundamentals', '-')}")
        summary = (
            f"✅ Market Scanner complete\n"
            f"COT: {latest_cot or '-'}"
            + (f"\n⚠️ {cot_msg}" if cot_status in ("stale", "unchanged") else "")
            + "\n\nTop attention:\n" + "\n".join(top)
        )
        telegram_send_message(summary, silent=True)

    return full_path


# ============================== MAIN ==============================
def main():
    t0 = time.time()
    log.info("Market Scanner %s starting", SCRIPT_VERSION)
    if telegram_enabled():
        log.info("Telegram alerts enabled (chat %s)", TELEGRAM_CHAT_ID)
    else:
        log.info("Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable alerts/delivery")
    prior_state = load_state()
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    meta = {"script_version": SCRIPT_VERSION, "argv": sys.argv[1:]}

    # ---- Prices ----
    log.info("Downloading FUTURES prices (25-year history) ...")
    try:
        market_data = get_market_data()
        prices = market_data["Close"]
        log.info("Prices loaded for %d tickers, %d bars each",
                 len(prices.columns), len(prices))
        meta["price_tickers"] = len(prices.columns)
        meta["price_bars"] = len(prices)
    except Exception as e:
        alert_failure("Price data unavailable (Yahoo/yfinance)", str(e))
        traceback.print_exc()
        sys.exit(1)

    # ---- COT ----
    log.info("Loading CURRENT CFTC COT data via Public Reporting API ...")
    try:
        cot = load_cot()
    except Exception as e:
        alert_failure("CFTC COT data unavailable", str(e))
        traceback.print_exc()
        sys.exit(1)

    latest, cot_status, cot_msg = check_cot_freshness(cot, prior_state)
    log.info("Latest COT report date: %s [%s]", latest, cot_status)
    if cot_status in ("stale", "unchanged"):
        log.warning(cot_msg)
        telegram_send_message(f"⚠️ Market Scanner — COT freshness\n{cot_msg}")
        if COT_STALE_ABORT and cot_status == "stale":
            alert_failure("Aborting: CFTC report too old", cot_msg)
            sys.exit(2)
    meta["cot_date"] = str(latest) if latest else None
    meta["cot_status"] = cot_status
    meta["cot_rows"] = len(cot)

    # ---- Charts (non-fatal) ----
    log.info("Building chart data (same futures OHLC as the analysis) ...")
    try:
        chart_data = get_chart_data(market_data)
        log.info("Charts prepared for %d markets", len(chart_data))
        meta["chart_markets"] = len(chart_data)
    except Exception as e:
        log.warning("Chart data failed (%s); Step 2 pages will be skipped", e)
        traceback.print_exc()
        chart_data = {}
        meta["chart_markets"] = 0

    # ---- Scan ----
    log.info("Running full scan ...")
    try:
        table = run_scan(prices, cot, seasonality=True)
    except Exception as e:
        alert_failure("Scan engine failed", str(e))
        traceback.print_exc()
        sys.exit(1)

    # ---- Reports ----
    try:
        show(table, latest, prices, cot, only, chart_data, prior_state,
             cot_status=cot_status, cot_msg=cot_msg)
    except Exception as e:
        alert_failure("Report generation failed", str(e))
        traceback.print_exc()

    meta["runtime_sec"] = round(time.time() - t0, 1)
    meta["markets_scanned"] = len(table) if table is not None else 0
    save_state(table if table is not None else pd.DataFrame(), extra=meta)
    log.info("Done in %.1f seconds. State saved for next week's comparisons.", meta["runtime_sec"])

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        alert_failure("Unhandled exception", str(e))
        traceback.print_exc()
        sys.exit(1)
