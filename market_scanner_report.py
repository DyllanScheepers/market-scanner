"""
Market Scanner — futures-native, report-grade, with AI trade-idea lifecycle.
Run:
    python market_scanner_report.py --mode sunday       (full scan + ideas + PDF)
    python market_scanner_report.py --mode wednesday    (Wednesday check: re-validates open ideas)
    python market_scanner_report.py --mode daily        (silent unless a material event)
    python market_scanner_report.py                     (auto by weekday)

Needs: pip install pandas numpy yfinance requests reportlab matplotlib
Output: scan_results.csv, "Invariant Weekly <date> — <category>.pdf",
        scan_state.json, trade_ideas.json, season_windows.json

Reports are split into one PDF per category (Currencies, Commodities,
Agriculturals, Indices) so each can breathe without hiding data. Set
REPORT_SPLIT_BY_CATEGORY = False to fall back to one combined PDF.

bp25 — seasonal lock · persistence · shape tier
bp26 — neighbour shifts ±5/±10 · absolute timing rule
bp27 — peer comparison (conservative PEER_MAP) + persistence clause
bp28 — walk season candidates past the top-scored winner if it fails persistence ·
       always create season_windows.json · Harvest Profile now shows both the
       4-week and the 8-week strongest reads side by side
bp29 — fixed _season_persistence (always returned False, so no seasonal window ever
       survived) · sharp / mid-length / broad labels · STAND ASIDE verdict word ·
       stand-aside rationale naming both reasons · argumentative combined-bias essay ·
       "why these rows" note on the Harvest Profile page · overview column widths ·
       SHAPE_DETREND and RPT_TRACK_CASE_CAP switches + uncapped-case logging
bp30 — reference-alignment pass (numbers now track the Online Trading Campus book):
       · track record counts one case per extreme EPISODE (CASE_DEDUP_WEEKS 4 -> 13)
         and only on fully-formed lookback windows (min_periods = full lookback),
         which cuts the inflated case counts (Gold 6 -> ~3, Silver 6 -> ~4)
       · RPT_TRACK_CASE_CAP removed: every past case on the page, like the reference
       · metals track-record + seasonality intended on SPOT when a deep Yahoo feed
         exists; Yahoo's XAUUSD=X / XAGUSD=X / XPTUSD=X / XPDUSD=X are dead (404),
         so measurement stays on futures (GC=F / SI=F / PL=F / PA=F). SPOT_TICKERS
         left empty until a usable free feed appears.
       · seasonal window search extended to 70 days (reference prints 69d windows)
       · neighbour-test shifts scale with window length (~L/4 and ~L/2, capped 10d),
         matching the reference's ±4/±9d on 18d and ±6/±10d on 25d windows
       · STRONG win bar 0.86 -> 0.85 (reference grades 12-of-14 as STRONG)
       · timing "tight" band narrowed (55% neighbour average = "very tight")
       · shape match no longer detrended + labels re-banded to the reference scale
         (>=0.40 loosely tracking, 0..0.40 off-pattern, <0 counter-seasonal)
       · Seasonal Overview lists MODERATE windows too, like the reference
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

SCRIPT_VERSION = "2026.10.15-bp30e"
CACHE_DIR = Path(os.environ.get("SCANNER_CACHE_DIR", "scanner_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
        logging.FileHandler(CACHE_DIR / f"scanner_{dt.date.today().isoformat()}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("scanner")


def _cache_path(kind: str, key: str = "") -> Path:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12] if key else "default"
    return CACHE_DIR / f"{kind}_{h}.pkl"


def _cache_is_fresh(path: Path, max_hours: float) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) / 3600.0 <= max_hours


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


def _json_write_atomic(path, obj):
    import shutil
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    if os.path.exists(path):
        try:
            shutil.copyfile(path, f"{path}.bak")
        except Exception:
            pass
    os.replace(tmp, path)


def _json_read_safe(path, label):
    if not os.path.exists(path):
        return None
    last_err = None
    for cand in (path, f"{path}.bak"):
        if not os.path.exists(cand):
            continue
        try:
            with open(cand, "r") as f:
                data = json.load(f)
            if cand != path:
                log.warning("%s was unreadable; restored from its backup copy.", label)
                telegram_send_message(f"⚠️ Market Scanner: {label} was unreadable, restored from the previous backup.")
            return data
        except Exception as e:
            last_err = e
    try:
        os.replace(path, f"{path}.corrupt")
    except Exception:
        pass
    log.error("%s unreadable (%s); quarantined as %s.corrupt and starting empty.", label, last_err, path)
    telegram_send_message(f"⚠️ Market Scanner: {label} was unreadable and has been reset ({str(last_err)[:120]}). "
                          f"The bad copy is kept as {os.path.basename(path)}.corrupt. Open ideas may need re-checking.")
    return None


def _retry(fn, *, retries=None, delay=2.0, label="request"):
    retries = retries if retries is not None else NETWORK_RETRIES
    last = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(delay * attempt)
    raise RuntimeError(f"{label} failed after {retries} attempts: {last}") from last


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
COT_STALE_DAYS = int(os.environ.get("SCANNER_COT_STALE_DAYS", "10"))
COT_STALE_ABORT = os.environ.get("SCANNER_COT_STALE_ABORT", "0").strip() in {"1", "true", "yes"}


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _telegram_chunks(text: str, limit: int = 3800):
    text = str(text or "")
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(line[:limit]); line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            chunks.append(cur); cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def telegram_send_message(text: str, *, silent: bool = False) -> bool:
    if not telegram_enabled():
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    ok_all = True
    for part in _telegram_chunks(text):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": part, "disable_notification": silent}
        try:
            def _post():
                r = requests.post(url, json=payload, timeout=30)
                r.raise_for_status()
                return True
            _retry(_post, retries=3, delay=2.0, label="telegram sendMessage")
        except Exception as e:
            log.warning("Telegram message failed: %s", e)
            ok_all = False
    return ok_all


def telegram_send_document(path, caption: str = "") -> bool:
    if not telegram_enabled():
        return False
    path = Path(path)
    if not path.exists():
        log.warning("Telegram document missing: %s", path)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    try:
        def _post():
            with open(path, "rb") as f:
                r = requests.post(url,
                                  data={"chat_id": TELEGRAM_CHAT_ID, "caption": (caption or "")[:1024]},
                                  files={"document": (path.name, f)}, timeout=180)
            r.raise_for_status()
            return True
        return _retry(_post, retries=3, delay=3.0, label=f"telegram sendDocument {path.name}")
    except Exception as e:
        log.warning("Telegram document failed (%s): %s", path.name, e)
        return False


PRICE_STALE_DAYS = int(os.environ.get("SCANNER_PRICE_STALE_DAYS", "6"))
PRICE_STALE_ABORT_SHARE = 0.25
SCAN_FAIL_ABORT = 8


def check_price_freshness(prices):
    today = pd.Timestamp(dt.date.today())
    names = {m[2]: m[0] for m in MARKETS if m[2]}
    names.update({v: k for k, v in REFS.items()})
    names.update({v: f"{k} (spot)" for k, v in SPOT_TICKERS.items()})
    stale = []
    for t in prices.columns:
        last = prices[t].last_valid_index()
        age = 9999 if last is None else (today - pd.Timestamp(last)).days
        if age > PRICE_STALE_DAYS:
            stale.append((names.get(t, t), age))
    return stale, (len(stale) / max(1, len(prices.columns)))


def _price_guard(prices, mode):
    stale, share = check_price_freshness(prices)
    if share >= PRICE_STALE_ABORT_SHARE:
        alert_failure("Prices look stale",
                      f"{len(stale)} of {len(prices.columns)} tickers have no bar in the last {PRICE_STALE_DAYS} days "
                      f"(Yahoo may be serving old data). Nothing was reported so you are not misled.")
        sys.exit(1)
    if stale and mode == "sunday":
        shown = ", ".join(f"{n} ({'no data' if a >= 9999 else str(a) + 'd old'})" for n, a in stale[:10])
        telegram_send_message(f"⚠️ Market Scanner: stale prices for {len(stale)} market(s): {shown}. "
                              f"Treat those rows with caution.")
    return stale


def alert_failure(title: str, detail: str = "") -> None:
    msg = f"🚨 Market Scanner FAILED\n{title}"
    if detail:
        msg += f"\n\n{detail[:1500]}"
    msg += f"\n\nversion {SCRIPT_VERSION} · {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    log.error("%s — %s", title, detail)
    telegram_send_message(msg)


def _expected_cot_date(today=None, now_utc=None):
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    today = today or now.date()
    back = (today.weekday() - 4) % 7
    if back == 0 and today == now.date() and now.hour < 21:
        back = 7
    return today - dt.timedelta(days=back + 3)


def check_cot_freshness(cot_df, prior_state=None, today=None, now_utc=None):
    if cot_df is None or len(cot_df) == 0 or "date" not in cot_df.columns:
        return None, "unknown", "No COT dates available"
    try:
        latest = pd.to_datetime(cot_df["date"]).max().date()
    except Exception:
        return None, "unknown", "Could not parse COT dates"
    today = today or dt.date.today()
    expected = _expected_cot_date(today, now_utc)
    age = (today - latest).days
    if latest < expected:
        weeks = max(1, round((expected - latest).days / 7))
        return latest, "stale", (
            f"CFTC data is {weeks} week{'s' if weeks != 1 else ''} behind: the newest report is dated {latest}, "
            f"but the one dated {expected} should already be published (CFTC releases on Fridays; a holiday or "
            f"shutdown can delay it). Positioning in this report is out of date, treat COT reads with caution.")
    return latest, "ok", f"CFTC report date {latest} ({age}d ago) is the current one"


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


def _safe_int(x, default=0):
    v = _safe_float(x)
    return int(v) if np.isfinite(v) else default


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
COT_BASIS = "raw"
COT_RANK_MODE = "strict"
TURN_WEEKS = 4
TURN_MIN_CHANGE = 0
DIV_WEEKS = 4
DIV_PRICE_LOOKBACK = 26
INDICES_USE_RETAIL = False

# bp30: the reference counts one case per extreme EPISODE. A re-entry into the
# zone within a few weeks of the last case is the same episode flaring, not a new
# case. 13 weeks (~ a quarter) out of the zone before a new case is counted.
CASE_DEDUP_WEEKS = 13
# bp30d: log-only comparison of episode rules (current gap vs out-of-zone 13/20).
# Does not change verdicts — prints CASE-DIAG lines for live extremes so Gold
# Multi-yr and Silver 2yr can be checked against the reference (1 of 3 / 1 of 4).
CASE_DEDUP_DIAG = True
CASE_DEDUP_DIAG_OUT_WEEKS = (13, 20)

ROC_SHORT = 10
ROC_SHORT_INDICES = 13
ROC_LONG = 30
RESCALE_LEN = 100
VAL_UPPER = 75
VAL_LOWER = -75
VAL_PRIMARY = "long"

# bp30: bands matched to the reference's labels — their best read in the book
# (+0.49) prints "loosely tracking", +0.25/+0.11 print "off-pattern", -0.18
# prints "counter-seasonal".
SHAPE_CLOSE = 0.70
SHAPE_LOOSE = 0.40

SEASON_PRIMARY_YEARS = 15
SEASON_MAX_YEARS = 25
SEASON_PROJECT_DAYS = 60
SEASON_TRIM_PCT = 10
SEASON_SMOOTH_LEN = 3
SEASON_MIN_MOVE_PCT = 0.75
SEASON_MIN_RELIABILITY = 65.0
SEASON_AGREEMENT_THRESHOLD = 0.25
SEASON_NEAREST_DAYS = 2
SEASON_STRONG_WIN = 0.85   # bp30: reference grades a 12-of-14 (85.7%) window STRONG
SEASON_STRONG_AVG = 0.30
SEASON_MODERATE_WIN = 0.73
SEASON_MODERATE_AVG = 0.20

# bp25 seasonal lock + persistence + shape tier
SEASON_WINDOW_STATE_FILE = "season_windows.json"
SEASON_PERSISTENCE_MIN_WINS = 3
SEASON_PERSISTENCE_WINDOW = 5
SEASON_SHAPE_FULL = 0.40
SEASON_SHAPE_WATCH = 0.00

# bp26 neighbour shifts + absolute timing bands
# bp30: the reference scales the shifts with window length (±4/±9d on an 18d
# window, ±6/±10d on a 25d window) -> ~L/4 and ~L/2, outer capped at 10d.
# SEASON_NEIGHBOUR_SHIFTS remains only as a fallback for fixed-shift callers.
SEASON_NEIGHBOUR_SHIFTS = (-10, -5, 5, 10)
SEASON_TIMING_FORGIVING_BAR = 0.60
SEASON_TIMING_TIGHT_BAR = 0.56   # bp30: reference calls a 55% neighbour average "very tight"


def _neighbour_shifts_for(length_days):
    """Reference-style neighbour shifts, scaled to the window being tested."""
    L = max(7, int(length_days))
    inner = max(2, int(round(L * 0.25)))
    outer = max(inner + 1, min(10, int(round(L * 0.50))))
    return (-outer, -inner, inner, outer)

# bp28 walk candidates past the top-scored winner if it fails persistence
CANDIDATE_WALK_LIMIT = 200

REPORT_BRAND = ""
REPORT_MAX_CHAPTERS = 15
BUILD_LAZY_PDF = False
REPORT_ALWAYS_CHAPTER = {"DXY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
                         "USDJPY", "USDCHF", "USDCAD",
                         "Gold", "Silver", "Natural Gas"}
REPORT_FORCE_MARKETS = []
REPORT_EDGE_MIN_CASES = 6
REPORT_EDGE_MIN_WIN = 60
REPORT_SEASON_YEARS = 15
# Window-length labels: sharp <= 21d, mid-length 22-45d, broad > 45d.
# (Reference shows 41d as mid-length and 58d as broad, so the true edge sits
#  somewhere in 42-57d; 45 is a placeholder, tune SEASON_MID_MAX_DAYS to taste.)
SEASON_SHARP_MAX_DAYS = 21
SEASON_MID_MAX_DAYS = 45
# bp30: reference prints windows up to 69d (WTI 23 Sep - 01 Dec), so the search
# must reach past 60d.
SEASON_MAX_WINDOW_DAYS = 70
# Engagement Record: the reference prints "every past case, on the page" — no cap.
RPT_TRACK_CASE_CAP = None
# Shape match: the reference correlates the raw paths (their values run ~0.3 below
# the detrended ones), so no detrending.
SHAPE_DETREND = False
REPORT_SEASON_BEHIND_DAYS = 14
REPORT_SEASON_AHEAD_DAYS = 56

REPORT_CATEGORIES = {
    "Currencies":    ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
                      "USDJPY", "USDCHF", "USDCAD", "DXY"],
    "Commodities":   ["Gold", "Silver", "Platinum", "Palladium", "Copper",
                      "WTI Crude", "Natural Gas", "Heating Oil"],
    "Agriculturals": ["Corn", "Wheat (KC)", "Soybeans", "Sugar",
                      "Coffee", "Cotton", "Cocoa"],
    "Indices":       ["S&P 500", "Nasdaq 100", "Dow Jones", "Russell 2000",
                      "Nikkei 225", "DAX", "UK 100"],
}
REPORT_CATEGORY_ORDER = ["Currencies", "Commodities", "Agriculturals", "Indices"]
REPORT_CATEGORY_ICONS = {"Currencies": "💱", "Commodities": "🏭",
                         "Agriculturals": "🌾", "Indices": "📈",
                         "Other": "•"}
REPORT_SPLIT_BY_CATEGORY = True


def _market_category(nm: str) -> str:
    for cat, mkts in REPORT_CATEGORIES.items():
        if nm in mkts:
            return cat
    return "Other"


PEER_MAP = {
    "EURUSD":    ["GBPUSD"],
    "GBPUSD":    ["EURUSD"],
    "AUDUSD":    ["NZDUSD"],
    "NZDUSD":    ["AUDUSD"],
    "USDJPY":    ["USDCHF"],
    "USDCHF":    ["USDJPY"],
    "DXY":       ["EURUSD", "GBPUSD"],
    "Gold":      ["Silver"],
    "Silver":    ["Gold"],
    "Platinum":  ["Palladium"],
    "Palladium": ["Platinum"],
    "WTI Crude":   ["Heating Oil"],
    "Heating Oil": ["WTI Crude"],
    "Corn":       ["Wheat (KC)", "Soybeans"],
    "Wheat (KC)": ["Corn"],
    "Soybeans":   ["Corn"],
    "Sugar":      ["Coffee"],
    "Coffee":     ["Sugar"],
    "S&P 500":     ["Nasdaq 100", "Dow Jones"],
    "Nasdaq 100":  ["S&P 500"],
    "Dow Jones":   ["S&P 500"],
    "Russell 2000":["S&P 500"],
    "Nikkei 225":  ["S&P 500"],
    "DAX":         ["UK 100"],
    "UK 100":      ["DAX"],
}


CHART_ZOOM = {"M": 60, "W": 150, "D": 300}
CHART_RIGHT_PAD_DAYS = 70
CHART_RIGHT_PAD = {"M": 420, "W": 180, "D": 70}

CONFLUENCE_MAX_DIST_PCT = 4.0
CONFLUENCE_MAX_ROWS = 20
WATCHLIST_ZONE_REACH_PCT = 5.0
WATCHLIST_STANDASIDE_CAP = 6
FRONT_PARTIAL_CAP = 8
FRONT_DORMANT_CAP = 3
FRONT_DASHBOARD_CAP = 10
FRONT_SEASON_STRONG_ONLY = False   # bp30: the reference overview lists MODERATE windows too

DAILY_HISTORY_YEARS = 5
GREEN_NEEDS_CONFIRMATION = True
SHOW_CHART_PAGES = False
STATE_FILE = "scan_state.json"
TRADE_IDEAS_FILE = "trade_ideas.json"

IDEA_MAX_AGE_DAYS = 28
IDEA_TRIGGERED_MAX_AGE_DAYS = 90

CORRELATION_CLUSTERS = {
    "long USD": ["DXY", "USDCHF", "USDJPY", "USDCAD"],
    "short USD": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
    "precious metals": ["Gold", "Silver", "Platinum", "Palladium"],
    "energy": ["WTI Crude", "Natural Gas", "Heating Oil"],
    "grains": ["Corn", "Wheat (KC)", "Soybeans"],
    "softs": ["Sugar", "Coffee", "Cotton", "Cocoa"],
    "equity indices": ["S&P 500", "Nasdaq 100", "Dow Jones", "Russell 2000",
                       "Nikkei 225", "DAX", "UK 100"],
}

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

# bp30b: Yahoo no longer serves usable spot series for the four precious metals
# (XAUUSD=X / XAGUSD=X / XPTUSD=X / XPDUSD=X return 404). Measurement, seasonality
# and the Engagement Record therefore stay on the futures continuous series
# (GC=F / SI=F / PL=F / PA=F). Zones, charts and location reads were always on
# futures. Leave SPOT_TICKERS empty; if a deep free spot feed appears later,
# add the symbols here and the existing _measure_close path will prefer them.
SPOT_TICKERS = {}
SPOT_MIN_HISTORY_YEARS = 15.5   # need at least the 15-year seasonal window + margin


def _measure_close(nm, ticker, prices):
    """Daily close series used to MEASURE moves (track record, seasonality).

    Spot where a sufficiently deep feed exists for this market, otherwise the
    market's own futures series. Never returns an empty series when the futures
    series has data. (SPOT_TICKERS is currently empty — Yahoo spot metals are dead.)
    """
    fut = pd.Series(dtype=float)
    if prices is not None and ticker in prices.columns:
        fut = prices[ticker].dropna()
    st = SPOT_TICKERS.get(nm)
    if st and prices is not None and st in prices.columns:
        spot = prices[st].dropna()
        deep = len(spot) >= SPOT_MIN_HISTORY_YEARS * 252
        if deep and (fut.empty or len(spot) >= 0.9 * len(fut)):
            log.info("%s: measuring on spot (%s, %d bars) instead of %s", nm, st, len(spot), ticker)
            return spot
        if st in (prices.columns if prices is not None else []):
            log.info("%s: spot feed %s too short (%d bars) — measuring on futures %s",
                     nm, st, len(spot), ticker)
    return fut


def _measure_weekly(nm, ticker, prices):
    """Weekly (W-FRI) version of the measurement series, for the Engagement Record."""
    d = _measure_close(nm, ticker, prices)
    if d is None or len(d) == 0:
        return None
    w = d.resample("W-FRI").last().dropna()
    return w if len(w) else None


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
    ("Wheat (KC)",      "Agriculturals", "KE=F", "001612", False),
    ("Soybeans",        "Agriculturals", "ZS=F", "005602", False),
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
    sec_idx = SECTION_ORDER.index(sec)
    if sec == "Majors" and name == "DXY":
        return (sec_idx, -1)
    return (sec_idx, MARKET_INDEX.get(name, 999))


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
    "Corn": "Corn futures (CBOT ZC)",
    "Wheat (KC)": "KC Wheat futures (CBOT KE)", "Soybeans": "Soybean futures (CBOT ZS)",
    "Sugar": "Sugar #11 futures (ICE SB)",
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


def _compute_sd_zones_raw(df, mode=None):
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


FRESH_MAX_PEN = 0.25
_ZONE_MEMO = {}


def _annotate_zones(df, zones):
    try:
        h = df["High"].values.astype(float); l = df["Low"].values.astype(float)
        n = len(h); rng = h - l
    except Exception:
        return zones
    for z in zones:
        try:
            r = int(z["right"])
            height = max(float(z["top"]) - float(z["bottom"]), 1e-12)
            after = slice(r + 1, n)
            if r + 1 < n:
                if z["demand"]:
                    pen = max(0.0, (float(z["top"]) - float(l[after].min())) / height)
                    mfe = (float(h[after].max()) - float(z["top"])) / height
                else:
                    pen = max(0.0, (float(h[after].max()) - float(z["bottom"])) / height)
                    mfe = (float(z["bottom"]) - float(l[after].min())) / height
            else:
                pen, mfe = 0.0, 0.0
            z["pen"] = float(pen); z["mfe"] = float(max(mfe, 0.0))
        except Exception:
            z.setdefault("pen", 0.0); z.setdefault("mfe", 0.0)
    return zones


def compute_sd_zones(df, mode=None):
    mode = mode or ZONE_DRAW_MODE
    key = None
    try:
        key = (mode, len(df), df.index[0], df.index[-1], float(df["Close"].iloc[-1]),
               float(df["High"].iloc[-1]), float(df["Low"].iloc[0]), tuple(sorted(SD.items())))
        if key in _ZONE_MEMO:
            return [dict(z) for z in _ZONE_MEMO[key]]
    except Exception:
        key = None
    zones = _annotate_zones(df, _compute_sd_zones_raw(df, mode))
    if key is not None:
        if len(_ZONE_MEMO) > 600:
            _ZONE_MEMO.clear()
        _ZONE_MEMO[key] = [dict(z) for z in zones]
    return zones


def zone_is_fresh(z):
    return (not z.get("mitigated")) and float(z.get("pen", 0.0) or 0.0) <= FRESH_MAX_PEN


# ============================== LOCATION READ ==============================
MATRIX_ACTIONS = {
    "UPTREND": {
        "VERY LOW":    "look to buy (LTF) with the trend",
        "LOW":         "look to buy (LTF) with the trend",
        "EQUILIBRIUM": "look to buy (LTF) with the trend",
        "HIGH":        "no trade",
        "VERY HIGH":   "look to sell (LTF) counter-trend",
    },
    "DOWNTREND": {
        "VERY LOW":    "look to buy (LTF) counter-trend",
        "LOW":         "no trade",
        "EQUILIBRIUM": "look to short (LTF) with the trend",
        "HIGH":        "look to short (LTF) with the trend",
        "VERY HIGH":   "look to short (LTF) with the trend",
    },
    "SIDEWAYS": {
        "VERY LOW":    "look to buy (LTF) fade on extremes",
        "LOW":         "no trade",
        "EQUILIBRIUM": "no trade",
        "HIGH":        "no trade",
        "VERY HIGH":   "look to short (LTF) fade on extremes",
    },
}
TRADEABLE_ACTIONS = {
    "look to buy (LTF) with the trend", "look to buy (LTF) counter-trend",
    "look to buy (LTF) fade on extremes", "look to short (LTF) with the trend",
    "look to sell (LTF) counter-trend", "look to short (LTF) fade on extremes",
}


def _pivots_hl(df, wing=2, keep=4):
    if df is None or len(df) < 20:
        return []
    h = df["High"].values.astype(float); l = df["Low"].values.astype(float); idx = df.index
    out = []
    for i in range(wing, len(df) - wing):
        if h[i] == h[i - wing:i + wing + 1].max() and h[i] > h[i - wing] and h[i] > h[i + wing]:
            out.append((idx[i], float(h[i]), "H"))
        if l[i] == l[i - wing:i + wing + 1].min() and l[i] < l[i - wing] and l[i] < l[i + wing]:
            out.append((idx[i], float(l[i]), "L"))
    highs = [p for p in out if p[2] == "H"][-keep:]
    lows = [p for p in out if p[2] == "L"][-keep:]
    return sorted(highs + lows, key=lambda p: p[0])


def _trend_from_pivots(piv):
    highs = [p for p in piv if p[2] == "H"]; lows = [p for p in piv if p[2] == "L"]
    if len(highs) < 3 or len(lows) < 3:
        return "SIDEWAYS"
    up = lows[-1][1] > lows[-2][1] > lows[-3][1]
    down = highs[-1][1] < highs[-2][1] < highs[-3][1]
    if up and down:
        return "SIDEWAYS"
    if up:
        return "UPTREND"
    if down:
        return "DOWNTREND"
    return "SIDEWAYS"


def _phase(close):
    s = close.dropna()
    if len(s) < 40:
        return "CONSOLIDATION"
    r = s.tail(20); p = s.tail(40).head(20)
    rr = float(r.max() - r.min()); pr = float(p.max() - p.min())
    if pr <= 0:
        return "CONSOLIDATION"
    ratio = rr / pr
    if ratio > 1.5:
        return "IMPULSE"
    if ratio < 0.6:
        return "CORRECTION"
    return "CONSOLIDATION"


def _location_label(price, demand_distal, demand_proximal, supply_proximal, supply_distal):
    if price <= demand_proximal:
        return "VERY LOW"
    if price >= supply_proximal:
        return "VERY HIGH"
    rng = supply_distal - demand_distal
    if rng <= 0:
        return "EQUILIBRIUM"
    pct = (price - demand_distal) / rng
    if pct < 0.33:
        return "LOW"
    if pct < 0.66:
        return "EQUILIBRIUM"
    return "HIGH"


def _matrix_action(location_label, trend):
    return MATRIX_ACTIONS[trend][location_label]


def compute_location_read(df, zones, why_out=None):
    def _why(msg):
        if why_out is not None:
            why_out.append(msg)
    close = df["Close"].dropna()
    if len(close) < 100:
        _why("not enough weekly history")
        return None
    last = float(close.iloc[-1])
    fresh = [z for z in zones if zone_is_fresh(z)]
    demand_below = [z for z in fresh if z["demand"] and z["bottom"] < last]
    supply_above = [z for z in fresh if (not z["demand"]) and z["top"] > last]
    if not demand_below and not supply_above:
        _why("no clean HTF range this week")
        return None
    if not demand_below:
        _why("no fresh demand below price, range cannot be defined")
        return None
    if not supply_above:
        _why("no fresh supply above price (price in open air), range cannot be defined")
        return None
    dem = max(demand_below, key=lambda z: z["top"])
    sup = min(supply_above, key=lambda z: z["bottom"])
    demand_distal = float(dem["bottom"]); demand_proximal = float(dem["top"])
    supply_proximal = float(sup["bottom"]); supply_distal = float(sup["top"])
    if supply_distal <= demand_distal:
        _why("nearest fresh supply sits below nearest fresh demand")
        return None
    loc = _location_label(last, demand_distal, demand_proximal, supply_proximal, supply_distal)
    trend = _trend_from_pivots(_pivots_hl(df))
    phase = _phase(close)
    action = _matrix_action(loc, trend)
    return {
        "price": last, "location": loc, "trend": trend, "phase": phase,
        "action": action, "tradeable": action in TRADEABLE_ACTIONS,
        "demand_zone": (demand_distal, demand_proximal),
        "supply_zone": (supply_proximal, supply_distal),
        "range": (demand_distal, supply_distal),
    }


def _row_invert(row):
    nm = str(row.get("Market", "")).replace(" (inv)", "")
    m = next((x for x in MARKETS if x[0] == nm), None)
    return bool(m[4]) if m else False


def _has_countertrend_case(row, side=None):
    inv = _row_invert(row)
    def _ext_dir(v):
        v = _safe_float(v)
        if not np.isfinite(v):
            return 0
        d = 1 if v >= EXTREME_HI else -1 if v <= EXTREME_LO else 0
        return -d if inv else d
    for key, label in (("Comm3y", "3Y commercial extreme"), ("Comm5y", "multi-year commercial extreme")):
        d = _ext_dir(row.get(key))
        if d and (side is None or d == side):
            return label
    cnt = _safe_float(row.get("CommExtremeCount"))
    cot_dir = _DIRS.get(str(row.get("COTBias", "Neutral")), 0)
    if np.isfinite(cnt) and cnt >= 4 and cot_dir and (side is None or cot_dir == side):
        return "commercial panel aligned"
    nc = str(row.get("NonCommState", "n/a"))
    nc_dir = _safe_int(row.get("NonCommDir"))
    if nc.endswith("divergence") and nc_dir and (side is None or nc_dir == side):
        return f"non-commercial {nc.lower()}"
    if str(row.get("ValStrength", "-")) == "Strong W+D":
        vd = _DIRS.get(str(row.get("Valuation", "Neutral")), 0)
        if vd and (side is None or vd == side):
            return "valuation extreme"
    return None


def _display_action(action_str):
    return action_str.replace("counter-trend", "counterflow")


# ============================== CONFLUENCE ==============================
def _zone_intersection(a_bot, a_top, b_bot, b_top):
    lo = max(a_bot, b_bot); hi = min(a_top, b_top)
    return (lo, hi) if hi > lo else None


def compute_confluence(m_zones, w_zones, d_zones, price, max_dist_pct=CONFLUENCE_MAX_DIST_PCT):
    def _fresh(lst):
        return [z for z in (lst or []) if zone_is_fresh(z)]
    pools = [("M", _fresh(m_zones)), ("W", _fresh(w_zones)), ("D", _fresh(d_zones))]
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            ta, za_list = pools[i]; tb, zb_list = pools[j]
            for za in za_list:
                for zb in zb_list:
                    if za["demand"] != zb["demand"]:
                        continue
                    ov = _zone_intersection(za["bottom"], za["top"], zb["bottom"], zb["top"])
                    if ov:
                        pairs.append({"tfs": [ta, tb], "demand": za["demand"],
                                      "bottom": ov[0], "top": ov[1], "zones": {ta: za, tb: zb}})
    triples = []
    for p in list(pairs):
        if set(p["tfs"]) != {"M", "W"}:
            continue
        for zd in pools[2][1]:
            if zd["demand"] != p["demand"]:
                continue
            ov = _zone_intersection(p["bottom"], p["top"], zd["bottom"], zd["top"])
            if ov:
                triples.append({"tfs": ["M", "W", "D"], "demand": p["demand"],
                                "bottom": ov[0], "top": ov[1], "zones": {**p["zones"], "D": zd}})
    def _inside(p, t):
        return (p["bottom"] >= t["bottom"] - 1e-9 and p["top"] <= t["top"] + 1e-9
                and p["demand"] == t["demand"])
    pairs = [p for p in pairs if not any(_inside(p, t) for t in triples)]
    def _dedupe(items):
        seen = set(); out = []
        for c in items:
            key = (c["demand"], round(c["bottom"], 6), round(c["top"], 6), tuple(sorted(c["tfs"])))
            if key in seen:
                continue
            seen.add(key); out.append(c)
        return out
    triples = _dedupe(triples); pairs = _dedupe(pairs)
    out = triples + pairs
    keep = []
    if not np.isfinite(price) or price <= 0:
        return []
    for c in out:
        mid = (c["bottom"] + c["top"]) / 2
        dist = abs(mid - price) / price * 100
        if dist > max_dist_pct:
            continue
        c["distance_pct"] = dist
        c["rank"] = 0 if len(c["tfs"]) == 3 else 1
        keep.append(c)
    keep.sort(key=lambda c: (c["rank"], c["distance_pct"]))
    return keep


def _zone_dist_txt(bottom, top, price):
    if price < bottom:
        return f"{(bottom - price) / price * 100:.1f}% above"
    if price > top:
        return f"{(price - top) / price * 100:.1f}% below"
    return "at price"


# ============================== UNIFIED DECISION ENGINE ==============================
_DIRS = {"Bullish": 1, "Bearish": -1}
ENGINE_RULES = {
    "Indices":       {"side": "valuation", "must_not_oppose": ("seasonality",)},
    "FX":            {"side": "cot", "must_not_oppose": ("valuation", "seasonality")},
    "Dollar":        {"side": "cot", "must_not_oppose": ("valuation", "seasonality")},
    "Metals":        {"side": "cot", "must_not_oppose": ("seasonality",)},
    "Energies":      {"side": "cot", "must_not_oppose": ("seasonality",)},
    "Agriculturals": {"side": "cot", "must_not_oppose": ("seasonality",)},
}


def _effective_row_for_decision(pk):
    row = pk.get("row")
    if row is None:
        return None
    try:
        r = dict(row)
    except Exception:
        return row
    sp = pk.get("season")
    if not sp:
        return r
    m = sp.get("main")
    if not m:
        return r
    try:
        # bp30c: opposing windows of similar strength cancel the calendar side
        if sp.get("cancelled"):
            r["Seasonality"] = "Neutral"
            r["SeasonWindowStrength"] = None
            r["SeasonWindowCancelled"] = True
            try:
                r["SeasonWindowInPlay"] = bool(sp["today"] >= m["start"] and sp["today"] <= m["end"])
            except Exception:
                pass
            r["SeasonWindow"] = (f"{m['start'].strftime('%d %b')} – {m['end'].strftime('%d %b')} · "
                                 f"{m['len']}d · cancelled (opposing window)")
            r["SeasonWindowTier"] = sp.get("tier", "FULL")
            return r
        r["Seasonality"] = "Bullish" if m["dir"] == 1 else "Bearish"
        _tier = sp.get("tier", "FULL")
        if _tier == "DROP":
            r["SeasonWindowStrength"] = None
        else:
            r["SeasonWindowStrength"] = m["strength"]
        try:
            r["SeasonWindowInPlay"] = bool(sp["today"] >= m["start"] and sp["today"] <= m["end"])
        except Exception:
            pass
        r["SeasonWindow"] = (f"{m['start'].strftime('%d %b')} – {m['end'].strftime('%d %b')} · "
                             f"{m['len']}d · {m['shape']} · {m['win'] * 100:.0f}% · "
                             f"{m['wins']} of {m['cases']} · {m['avg']:+.1f}%")
        r["SeasonWindowDays"] = m.get("len")
        r["SeasonWindowShape"] = m.get("shape")
        r["SeasonWindowReliability"] = (m.get("win") or 0) * 100 if m.get("win") is not None else None
        r["SeasonWindowWins"] = m.get("wins")
        r["SeasonWindowTotal"] = m.get("cases")
        r["SeasonWindowSamples"] = m.get("cases")
        r["SeasonWindowAvgMove"] = m.get("avg")
        r["SeasonWindowTier"] = _tier
    except Exception as e:
        log.debug("Effective-row reconcile failed for %s: %s", pk.get("name"), e)
    return r


def fundamental_view(row, group=None):
    group = group or str(row.get("Group", ""))
    rules = ENGINE_RULES.get(group, ENGINE_RULES["Metals"])
    cot = str(row.get("COTBias", "Neutral"))
    val = str(row.get("Valuation", "Neutral"))
    seas = str(row.get("Seasonality", "n/a"))
    strong = row.get("SeasonWindowStrength") == "STRONG"
    primary = rules["side"]
    side = _DIRS.get(cot if primary == "cot" else val, 0)
    source = "Flow" if primary == "cot" else "Relative Value"
    season_led = False
    if side == 0 and strong and seas in _DIRS:
        side, source, season_led = _DIRS[seas], "Calendar Flow", True
    votes = {"cot": _DIRS.get(cot, 0), "valuation": _DIRS.get(val, 0), "seasonality": _DIRS.get(seas, 0)}
    confirms, conflicts, info_conflicts = [], [], []
    if side != 0 and not season_led:
        for eng, d in votes.items():
            if d == 0 or eng == primary:
                continue
            if d == side:
                confirms.append("calendar flow confirms" if eng == "seasonality" and strong
                                else f"{eng} agrees")
            elif eng in rules["must_not_oppose"]:
                conflicts.append(eng)
            else:
                info_conflicts.append(eng)
    funds = {"support": None, "text": ""}
    nc = str(row.get("NonCommState", "n/a"))
    nc_dir = _safe_int(row.get("NonCommDir"))
    if side != 0 and not season_led and nc_dir:
        if nc.endswith("divergence"):
            if nc_dir == side:
                funds = {"support": "for", "text": "speculators diverge in favour"}
            else:
                funds = {"support": "against", "text": "speculators diverge against it"}
        elif nc == "Follow-through":
            if nc_dir == side:
                funds = {"support": "for", "text": "speculators follow through"}
            else:
                funds = {"support": "against", "text": "speculators are still trending the other way"}
    if (side != 0 and not season_led and _safe_int(row.get("NonCommShift")) >= 3
            and _DIRS.get(cot, 0) == side and funds["support"] != "against"):
        funds = {"support": "for", "text": "trend-shift set-up: hedgers at an extreme, speculators trimming the old trend, the masses on the wrong end"}
    return {"side": side, "source": source, "primary": primary, "season_led": season_led,
            "confirms": confirms, "conflicts": conflicts, "info_conflicts": info_conflicts,
            "funds": funds, "cot": cot, "val": val, "seas": seas, "strong": strong}


def _action_dir(action):
    a = str(action or "")
    if "buy" in a:
        return 1
    if "short" in a or "sell" in a:
        return -1
    return 0


def _action_is_counter(action):
    a = str(action or "")
    return ("counter" in a) or ("fade" in a)


def _zone_gap_pct(location_read, price, side):
    if not location_read or not price:
        return None
    if side == 1:
        z = location_read.get("demand_zone")
        return None if not z else max(0.0, (price - float(z[1])) / price * 100)
    z = location_read.get("supply_zone")
    return None if not z else max(0.0, (float(z[0]) - price) / price * 100)


def decide_market(row, location_read, price, location_why=None):
    group = str(row.get("Group", ""))
    fv = fundamental_view(row, group)
    side = fv["side"]
    name = "flow positioning" if fv["primary"] == "cot" else "relative value"
    out = {"side": side, "source": fv["source"], "season_led": fv["season_led"], "fv": fv,
           "bucket": "DORMANT", "blocker": None, "why": "", "short": "", "trade_type": None}

    if side == 0:
        if group == "Indices":
            if fv["seas"] in _DIRS:
                out["why"] = (f"relative value vs rates gives no side · calendar flow "
                              f"{fv['seas'].lower()} but cannot lead an index")
            else:
                out["why"] = "relative value vs rates gives no side · no active calendar window"
        elif fv["seas"] in _DIRS:
            out["why"] = f"calendar flow {fv['seas'].lower()} but {name} has no side — dormant"
        else:
            out["why"] = f"no {name} side · no active calendar window"
        out["short"] = "no side"
        return out

    parts = ([f"{fv['source']} {'bullish' if side == 1 else 'bearish'}"] if not fv["season_led"]
             else [f"calendar flow {'bullish' if side == 1 else 'bearish'} (strong window) is the only engine"])
    parts += fv["confirms"]
    if fv["funds"]["support"] == "for":
        parts.append(fv["funds"]["text"])

    blocker = None
    if fv["conflicts"]:
        blocker = (" and ".join(fv["conflicts"])
                   + (" opposes" if len(fv["conflicts"]) == 1 else " oppose") + " the side")
    elif fv["season_led"]:
        blocker = ("indices need relative value as the driver" if group == "Indices"
                   else "calendar flow never engages alone, wait for the main engine")
    elif GREEN_NEEDS_CONFIRMATION and not fv["confirms"]:
        blocker = "no independent confirmation yet (relative value or calendar flow)"

    caution = fv["funds"]["text"] if fv["funds"]["support"] == "against" else None
    if caution:
        out["caution"] = caution
    if blocker is None:
        out["bucket"] = "READY"
        if caution:
            parts.append("CAUTION: " + caution)
        out["why"] = " · ".join(parts)
        out["short"] = "ready"
    else:
        out["bucket"] = "PARTIAL"
        out["blocker"] = blocker
        out["why"] = " · ".join(list(parts) + [blocker])
        out["short"] = blocker
    return out


def _confluence_aligned_with_bias(c, row):
    fv = fundamental_view(row)
    if fv["side"] == 0:
        return False
    return bool((c["demand"] and fv["side"] == 1) or ((not c["demand"]) and fv["side"] == -1))


# ============================== WATCHLIST ==============================
def _zone_within_reach(location_read, price, bias, threshold_pct=WATCHLIST_ZONE_REACH_PCT):
    if not location_read or price is None or not np.isfinite(price):
        return False
    if bias == 1:
        zone = location_read.get("demand_zone")
        if zone:
            top = float(zone[1])
            if top > 0 and price > 0:
                return (price - top) / price * 100 <= threshold_pct
    elif bias == -1:
        zone = location_read.get("supply_zone")
        if zone:
            bot = float(zone[0])
            if bot > 0 and price > 0:
                return (bot - price) / price * 100 <= threshold_pct
    return False


def _classify_watchlist(row, location_read, season_main=None, price=None, location_why=None):
    return decide_market(row, location_read, price, location_why)["bucket"]


def _watchlist_sentence(row, location_read, season_main=None, bucket=None, price=None, location_why=None):
    return decide_market(row, location_read, price, location_why)["why"]


def _watchlist_telegram_message(packs):
    by_cat = {cat: {"READY": [], "PARTIAL": [], "DORMANT": []} for cat in REPORT_CATEGORY_ORDER}
    by_cat.setdefault("Other", {"READY": [], "PARTIAL": [], "DORMANT": []})

    for nm, pk in packs.items():
        row = _effective_row_for_decision(pk)
        if row is None:
            continue
        d = decide_market(row, pk.get("location_read"),
                          pk.get("last_price"), pk.get("location_why"))
        cat = _market_category(nm)
        by_cat.setdefault(cat, {"READY": [], "PARTIAL": [], "DORMANT": []})
        by_cat[cat][d["bucket"]].append((nm, d))

    def _side(d):
        return "Long" if d["side"] == 1 else "Short" if d["side"] == -1 else "—"

    lines = ["📋 ACTIVATION BOARD"]
    for cat in REPORT_CATEGORY_ORDER + ["Other"]:
        buckets = by_cat.get(cat)
        if not buckets or not any(buckets[b] for b in ("READY", "PARTIAL", "DORMANT")):
            continue
        icon = REPORT_CATEGORY_ICONS.get(cat, "•")
        lines.append(f"\n{icon} {cat.upper()}")

        if buckets["READY"]:
            lines.append("🟢 READY — stack coherent")
            for nm, d in sorted(buckets["READY"], key=lambda t: t[0]):
                _, full, _ = _rpt_market_display(nm)
                lines.append(f"· {full} — {_side(d)}")
        if buckets["PARTIAL"]:
            lines.append("👀 PARTIAL — one layer missing")
            for nm, d in sorted(buckets["PARTIAL"], key=lambda t: t[0]):
                _, full, _ = _rpt_market_display(nm)
                tag = "Calendar-led" if d["season_led"] else _side(d)
                lines.append(f"· {full} — {tag} · {d['short']}")
        if buckets["DORMANT"]:
            names = [_rpt_market_display(nm)[1]
                     for nm, _ in sorted(buckets["DORMANT"], key=lambda t: t[0])]
            shown = names[:WATCHLIST_STANDASIDE_CAP]
            extra = len(names) - len(shown)
            lines.append("⏸ Dormant: " + ", ".join(shown) +
                         (f" (+{extra} more)" if extra > 0 else ""))
    return "\n".join(lines)


# ============================== COT API ==============================
CFTC_API_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
CFTC_API_LOOKBACK_DAYS = 40 * 365 + 60
CFTC_API_PAGE_SIZE = 50000
CFTC_API_MAX_PAGES = 10
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
    "Wheat (KC)": ["WHEAT-HRW", "KC HRW WHEAT", "HARD RED WINTER WHEAT", "KC WHEAT"],
    "Soybeans": ["SOYBEANS"],
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
    "Corn": "002602",
    "Wheat (KC)": "001612", "Soybeans": "005602",
    "Sugar": "080732", "Coffee": "083731", "Cotton": "033661", "Cocoa": "073732",
    "S&P 500": "13874A", "Nasdaq 100": "209742", "Dow Jones": "124603", "Russell 2000": "239742",
}


def _api_get(params):
    def _do():
        r = requests.get(CFTC_API_URL, params=params,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/154 Safari/537.36",
                                  "Accept": "application/json"},
                         timeout=NETWORK_TIMEOUT)
        r.raise_for_status()
        return r.json()
    return _retry(_do, retries=CFTC_API_RETRIES, delay=2.0, label="CFTC API")


def _api_get_all(params):
    out = []
    offset = 0
    base = dict(params)
    base.pop("$limit", None)
    base.pop("$offset", None)
    for _ in range(CFTC_API_MAX_PAGES):
        p = dict(base)
        p["$limit"] = str(CFTC_API_PAGE_SIZE)
        p["$offset"] = str(offset)
        page = _api_get(p)
        if not page:
            break
        out.extend(page)
        if len(page) < CFTC_API_PAGE_SIZE:
            break
        offset += CFTC_API_PAGE_SIZE
    return out


def _api_rows_for_code(code, date_from):
    safe_code = str(code).strip().replace("'", "''")
    where = f"report_date_as_yyyy_mm_dd >= '{date_from}' AND cftc_contract_market_code = '{safe_code}'"
    return _api_get_all({
        "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,contract_market_name,open_interest_all,comm_positions_long_all,comm_positions_short_all,noncomm_positions_long_all,noncomm_positions_short_all,nonrept_positions_long_all,nonrept_positions_short_all",
        "$where": where, "$order": "report_date_as_yyyy_mm_dd DESC"})


def _api_rows_for_alias(alias, date_from):
    where = f"report_date_as_yyyy_mm_dd >= '{date_from}' AND upper(contract_market_name) like '%{alias.upper().replace(chr(39), chr(39)*2)}%'"
    return _api_get_all({
        "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,contract_market_name,open_interest_all,comm_positions_long_all,comm_positions_short_all,noncomm_positions_long_all,noncomm_positions_short_all,nonrept_positions_long_all,nonrept_positions_short_all",
        "$where": where, "$order": "report_date_as_yyyy_mm_dd DESC"})


def load_cot(years=None, use_cache=True):
    cache_p = _cache_path("cot", dt.date.today().isoformat())
    stale_p = _cache_path("cot", "last_good")
    if use_cache and _cache_is_fresh(cache_p, COT_CACHE_HOURS):
        cached = _cache_load(cache_p)
        if cached is not None and len(cached) > 0:
            log.info("COT: using fresh cache (%s rows)", f"{len(cached):,}")
            return cached
    date_from = (dt.date.today() - dt.timedelta(days=CFTC_API_LOOKBACK_DAYS)).isoformat()
    rows = []; resolved = {}; failures = []
    log.info("Loading CURRENT CFTC COT data via Public Reporting API (last %.0f years) ...",
             CFTC_API_LOOKBACK_DAYS / 365.0)
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
                            market_rows = cm; break
                    market_rows = got; break
                except Exception as e:
                    failures.append((market_name, str(e))); continue
        if not market_rows:
            log.warning("%s: live CFTC rows not found", market_name); continue
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
            log.warning("%s: expected code %s, got %s; skipping", market_name, expected_code, actual_code); continue
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
                "ret_long": r.get("nonrept_positions_long_all"), "ret_short": r.get("nonrept_positions_short_all")})
    if not rows:
        stale = _cache_load(stale_p)
        if stale is not None and len(stale) > 0:
            log.error("Live COT download failed; using last-known-good cache (%s rows).", f"{len(stale):,}")
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
    _cache_save(cache_p, df); _cache_save(stale_p, df)
    return df


# ============================== FUTURES PRICE DATA ==============================
def get_market_data(markets=None, use_cache=True, years=25):
    import yfinance as yf
    markets = markets or MARKETS
    market_tickers = sorted({m[2] for m in markets if m[2]})
    ref_tickers = sorted({v for v in REFS.values()})
    spot_tickers = sorted({v for k, v in SPOT_TICKERS.items()
                           if k in {m[0] for m in markets}})
    all_tickers = sorted(set(market_tickers) | set(ref_tickers) | set(spot_tickers))
    cache_key = ",".join(all_tickers) + f"|{dt.date.today().isoformat()}|{years}"
    cache_p = _cache_path("prices", cache_key)
    stale_p = _cache_path("prices", f"last_good_{years}")
    if use_cache and _cache_is_fresh(cache_p, PRICE_CACHE_HOURS):
        cached = _cache_load(cache_p)
        if isinstance(cached, dict) and "Close" in cached and not cached["Close"].empty:
            log.info("Prices: using fresh cache (%d tickers, %d bars)", len(cached["Close"].columns), len(cached["Close"]))
            return cached

    def _download():
        raw = yf.download(all_tickers, start=(dt.date.today() - dt.timedelta(days=years * 365 + 10)),
                          interval="1d", auto_adjust=True, progress=False, group_by="column", threads=True)
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
            return stale
        raise RuntimeError(f"Could not parse yfinance response. Columns: {list(raw.columns)[:8]}")
    for name, group, ticker, code, invert in markets:
        if not invert or ticker is None or ticker not in ohlc["Close"].columns:
            continue
        o = ohlc["Open"][ticker].copy(); h = ohlc["High"][ticker].copy()
        l = ohlc["Low"][ticker].copy();  c = ohlc["Close"][ticker].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ohlc["Open"][ticker] = 1.0 / o
            ohlc["High"][ticker] = 1.0 / l
            ohlc["Low"][ticker] = 1.0 / h
            ohlc["Close"][ticker] = 1.0 / c
    result = {f: ohlc[f].replace([np.inf, -np.inf], np.nan) for f in ("Open", "High", "Low", "Close")}
    _cache_save(cache_p, result); _cache_save(stale_p, result)
    return result


def get_chart_data(ohlc=None, with_zones=True):
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
        }
        if with_zones:
            out[name]["zones"] = compute_sd_zones(df)
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
NC_LOOKBACK = 26
NC_RECENT = 4
NC_TOL = 0.05
NC_UNWIND = 0.10


def _align_spec_to_price(spec, dates, weekly_px):
    sp = pd.Series(np.asarray(pd.Series(spec).astype(float)), index=pd.to_datetime(pd.Series(dates).values))
    sp = sp.dropna()
    sp.index = sp.index + pd.to_timedelta((4 - sp.index.weekday()) % 7, unit="D")
    sp = sp[~sp.index.duplicated(keep="last")].sort_index()
    p = pd.Series(weekly_px).dropna().astype(float)
    p.index = pd.to_datetime(p.index)
    if getattr(p.index, "tz", None) is not None:
        p.index = p.index.tz_localize(None)
    p = p.sort_index()
    p_on = p.reindex(sp.index, method="nearest", tolerance=pd.Timedelta(days=10))
    ok = p_on.notna()
    return sp[ok], p_on[ok]


def noncomm_read(spec, weekly_px, invert, comm_dir=0, ret_dir=0, dates=None):
    out = {"state": "n/a", "dir": 0, "detail": "not enough positioning or price history", "idx52": np.nan,
           "chg4": np.nan, "unwinding": False, "fuel": "-", "shift_conditions": 0, "shift_setup": False}
    try:
        if weekly_px is None:
            return out
        if dates is not None:
            s, p = _align_spec_to_price(spec, dates, weekly_px)
            s = s.reset_index(drop=True)
            p = p.reset_index(drop=True)
        else:
            s = pd.Series(spec).dropna().astype(float).reset_index(drop=True)
            p = pd.Series(weekly_px).dropna().astype(float)
        if len(s) < NC_LOOKBACK + NC_RECENT or len(p) < NC_LOOKBACK + NC_RECENT:
            return out
        s = s * (-1.0 if invert else 1.0)
        sw = s.tail(NC_LOOKBACK).values
        pw = p.tail(NC_LOOKBACK).values
        rec = NC_RECENT
        rng_s = max(float(sw.max() - sw.min()), 1e-9)
        p_hi = pw[-rec:].max() >= pw[:-rec].max()
        p_lo = pw[-rec:].min() <= pw[:-rec].min()
        gap_hi = (sw[:-rec].max() - sw[-rec:].max()) / rng_s
        gap_lo = (sw[:-rec].min() - sw[-rec:].min()) / rng_s
        conf_hi = gap_hi <= NC_TOL
        conf_lo = gap_lo >= -NC_TOL
        chg4 = float(sw[-1] - sw[-1 - rec])
        out["chg4"] = chg4
        if p_hi and not conf_hi:
            out.update(state="Bearish divergence", dir=-1, detail=(
                f"Price made a {NC_LOOKBACK}-week high, but speculator net stayed {gap_hi * 100:.0f}% of its range "
                f"below its prior peak. Speculators are taking profit into the rally."))
        elif p_lo and not conf_lo:
            out.update(state="Bullish divergence", dir=1, detail=(
                f"Price made a {NC_LOOKBACK}-week low, but speculator net stayed {-gap_lo * 100:.0f}% of its range "
                f"above its prior trough. Speculators are covering into the decline."))
        elif p_hi:
            out.update(state="Follow-through", dir=1, detail=(
                f"Price made a {NC_LOOKBACK}-week high and speculator net confirmed it. The up-trend still has fuel."))
        elif p_lo:
            out.update(state="Follow-through", dir=-1, detail=(
                f"Price made a {NC_LOOKBACK}-week low and speculator net confirmed it. The down-trend still has fuel."))
        else:
            out.update(state="Neutral", dir=0, detail=(
                f"Price is not at a {NC_LOOKBACK}-week extreme, so there is nothing to confirm or diverge from."))
        idx52 = _cot_index(s, 52)
        out["idx52"] = idx52
        if np.isfinite(idx52):
            if idx52 >= EXTREME_HI:
                out["fuel"] = "Speculators stretched long"
            elif idx52 <= EXTREME_LO:
                out["fuel"] = "Speculators stretched short"
        net_dir = 1 if sw[-1] >= float(np.mean(sw)) else -1
        out["unwinding"] = bool(chg4 * net_dir < 0 and abs(chg4) >= NC_UNWIND * rng_s)
        if comm_dir:
            conds = [True, bool(chg4 * comm_dir >= NC_UNWIND * rng_s), bool(ret_dir == comm_dir)]
            out["shift_conditions"] = int(sum(conds))
            out["shift_setup"] = bool(all(conds))
    except Exception as e:
        log.debug("noncomm_read failed: %s", e)
    return out


def spec_divergence(spec, weekly_px, invert, dates=None):
    st = noncomm_read(spec, weekly_px, invert, dates=dates)["state"]
    return "Bearish" if st == "Bearish divergence" else "Bullish" if st == "Bullish divergence" else "-"


def _rank_pct(a):
    a = np.asarray(a, dtype=float)
    cur = a[-1]
    n = float(len(a))
    if COT_RANK_MODE == "weak":
        return float((a <= cur).sum()) / n * 100.0
    return float((a < cur).sum()) / n * 100.0


def _cot_basis(net, oi):
    if COT_BASIS == "pct_oi":
        return net / oi * 100.0
    return net.astype(float)


def _cot_index(series, weeks):
    x = pd.Series(series).dropna()
    if len(x) < max(4, weeks // 2):
        return np.nan
    w = x.tail(weeks)
    if len(w) < 4:
        return np.nan
    return _rank_pct(w.values)


def _cot_extreme_direction(index_value):
    if not np.isfinite(index_value):
        return 0
    return 1 if index_value >= EXTREME_HI else -1 if index_value <= EXTREME_LO else 0


def _edge_stats_for_signal(cot_sub, weekly_px, lookback_weeks, signal_direction, invert=False, role="commercial",
                           history_years=None):
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
    basis = _cot_basis(net, oi)
    # bp30: only fully-formed lookback windows may print an extreme. Half-built
    # windows made early history look "extreme" far more often than the reference
    # (their Multi-yr Gold row: 3 cases in 15 yrs, not 6).
    minp = lookback_weeks
    idx = basis.rolling(lookback_weeks, min_periods=minp).apply(_rank_pct, raw=True)
    px = weekly_px.dropna().copy(); px.index = pd.to_datetime(px.index)
    dates = pd.to_datetime(c["date"])
    cutoff = None
    if history_years is not None and len(px) > 0:
        cutoff = pd.Timestamp(px.index[-1]) - pd.DateOffset(years=history_years)
    rows = []; last_extreme = 0; last_case_i = -9999
    for i, iv in enumerate(idx):
        d = _cot_extreme_direction(iv)
        if d == 0:
            last_extreme = 0; continue
        if d == last_extreme:
            continue
        last_extreme = d
        native_price = (d if role == "commercial" else -d)
        price_dir = -native_price if invert else native_price
        if price_dir != signal_direction:
            continue
        if i - last_case_i < CASE_DEDUP_WEEKS:
            continue
        last_case_i = i
        dt0 = dates.iloc[i]
        if cutoff is not None and pd.Timestamp(dt0) < cutoff:
            continue
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
    sub = sub.copy().sort_values("date").reset_index(drop=True)
    comm = sub["comm_long"] - sub["comm_short"]
    spec = sub["spec_long"] - sub["spec_short"]
    ret = sub["ret_long"] - sub["ret_short"]
    oi = sub["oi"].replace(0, np.nan)
    comm_basis = _cot_basis(comm, oi)
    ret_basis = _cot_basis(ret, oi)
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
        "div": spec_divergence(spec, weekly_px, invert, dates=sub["date"]), "date": sub["date"].iloc[-1], "weeks": len(sub),
        "nc": noncomm_read(spec, weekly_px, invert, comm_direction, ret_direction, dates=sub["date"]),
        "CommAgreeLayers": " / ".join(agree_layers) if agree_layers else "-",
        "CommAgreement": f"{len(agree_layers)} of {len(COT_LOOKBACKS)}" if agree_layers else "0 of 5",
        "RetailAgreeLayers": " / ".join(k for k, v in ret_price.items() if v == ret_direction) if ret_direction else "-",
    }
    for layer, w in COT_LOOKBACKS.items():
        _hy = RPT_LOOKBACK_HISTORY_YEARS.get(layer)
        cd = comm_price.get(layer, 0); rd = ret_price.get(layer, 0)
        if cd:
            out[f"{layer}_EdgeDirection"] = "Bullish" if cd == 1 else "Bearish"
            out.update({f"{layer}_{k}": v for k, v in _edge_stats_for_signal(sub, weekly_px, w, cd, invert, role="commercial", history_years=_hy).items()})
        else:
            out[f"{layer}_EdgeDirection"] = "-"
            for h in COT_EDGE_HORIZONS:
                for suffix, val in (("Win", np.nan), ("Wins", 0), ("Total", 0), ("AvgMove", np.nan), ("Samples", 0), ("AvgWeeksIn", np.nan)):
                    out[f"{layer}_Edge{h}{suffix}"] = val
        if rd:
            out[f"{layer}_RetailEdgeDirection"] = "Bullish" if rd == 1 else "Bearish"
            out.update({f"{layer}_RetailEdge{k}": v for k, v in _edge_stats_for_signal(sub, weekly_px, w, rd, invert, role="retail", history_years=_hy).items()})
        else:
            out[f"{layer}_RetailEdgeDirection"] = "-"
            for h in COT_EDGE_HORIZONS:
                for suffix, val in (("Win", np.nan), ("Wins", 0), ("Total", 0), ("AvgMove", np.nan), ("Samples", 0), ("AvgWeeksIn", np.nan)):
                    out[f"{layer}_RetailEdge{h}{suffix}"] = val
    return out


def combine(cot_bias, val_bias, season_bias="Neutral", season_conflict=False,
            season_window_conflict=False, season_strength=None, season_in_play=False):
    if cot_bias in ("Bullish", "Bearish"):
        parts = [f"Flow-led {cot_bias}"]
        if val_bias == cot_bias: parts.append("relative value aligned")
        elif val_bias in ("Bullish", "Bearish"): parts.append(f"relative value {val_bias.lower()}")
        if season_bias == cot_bias: parts.append("calendar flow confirms")
        elif season_conflict or season_window_conflict: parts.append("calendar flow tempers conviction")
        return " • ".join(parts)
    if season_bias in ("Bullish", "Bearish") and season_strength == "STRONG":
        parts = [f"Calendar-led {season_bias} (flow neutral)"]
        if season_in_play: parts.append("window live now")
        if val_bias == season_bias: parts.append("relative value aligned")
        elif val_bias in ("Bullish", "Bearish"): parts.append(f"relative value {val_bias.lower()}")
        return " • ".join(parts)
    if val_bias in ("Bullish", "Bearish"):
        return f"Relative-value-only {val_bias} (no valid flow side)"
    return "Neutral"


def valuation_state(d, w):
    if w != 0:
        if d == w: return w, 2, "Strong W+D"
        if d == -w: return w, 2, "Strong W (D opposes)"
        return w, 2, "Strong W"
    if d != 0:
        return d, 1, "Daily"
    return 0, 0, "-"


# ============================== SEASONALITY ==============================
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
    if pos < len(dates):
        return dates[pos]
    nxt = t + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


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
        y_start = _season_nearest(ys, 0); y_end = _season_nearest(ys, len(ys) - 1)
        if not np.isfinite(y_start) or not np.isfinite(y_end) or y_start == 0:
            continue
        drift = ((y_end - y_start) / y_start * 100.0) / len(ys)
        samples.append((offset, y, ys, p0, drift))
        for h in range(horizon + 1):
            target_idx = t_idx + h; target_year = y
            while target_year in years and target_idx >= len(years[target_year]):
                target_idx -= len(years[target_year]); target_year += 1
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
    peak_h = int(np.nanargmax(path_det)); trough_h = int(np.nanargmin(path_det))
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


def _season_window_stats(close, direction, center_date=None, market_name=None):
    empty = {"SeasonWindow": "-", "SeasonWindowDays": np.nan, "SeasonWindowShape": "-",
             "SeasonWindowReliability": np.nan, "SeasonWindowSamples": 0, "SeasonWindowWins": 0,
             "SeasonWindowTotal": 0, "SeasonWindowAvgMove": np.nan, "SeasonWindowDriver": "-",
             "SeasonWindowNeighbour": "-", "SeasonWindowWholeYear": "-",
             "SeasonWindowStrength": None, "SeasonWindowInPlay": False,
             "SeasonWindowStart": None, "SeasonWindowEnd": None, "SeasonWindowDir": 0,
             "SeasonWindowTier": "FULL"}
    try:
        sp = _rpt_season_pack(close, center_date, market_name=market_name)
    except Exception:
        return empty
    m = sp.get("main") if sp else None
    if not m:
        return empty
    today = pd.Timestamp(sp["today"])
    label = (f"{m['start'].strftime('%d %b')} \u2013 {m['end'].strftime('%d %b')} \u00b7 {m['len']}d \u00b7 "
             f"{m['shape']} \u00b7 {m['win'] * 100:.0f}% \u00b7 {m['wins']} of {m['cases']} \u00b7 {m['avg']:+.1f}%")
    in_play = bool(today >= m["start"] and today <= m["end"])
    return {**empty, "SeasonWindow": label, "SeasonWindowDays": m["len"], "SeasonWindowShape": m["shape"],
            "SeasonWindowReliability": m["win"] * 100.0, "SeasonWindowSamples": m["cases"],
            "SeasonWindowWins": m["wins"], "SeasonWindowTotal": m["cases"],
            "SeasonWindowAvgMove": m["avg"], "SeasonWindowDriver": "Window + whole-year",
            "SeasonWindowNeighbour": "Shift-sensitive" if m.get("timing") == "tight" else "Pass",
            "SeasonWindowWholeYear": "Pending", "SeasonWindowStrength": m["strength"],
            "SeasonWindowInPlay": in_play,
            "SeasonWindowStart": m["start"].date().isoformat(),
            "SeasonWindowEnd": m["end"].date().isoformat(),
            "SeasonWindowDir": int(m["dir"]),
            "SeasonWindowTier": sp.get("tier", "FULL")}


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
            idx = t_idx + h; yy = y
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
    sizing = "-"
    path_summary = "; ".join(f"{i}:{v:.2f}" for i, v in enumerate(seasonal_path[:max_h + 1]) if np.isfinite(v))
    return {"SeasonVolCurrent": current, "SeasonVol60D": seasonal_60, "SeasonVolSeasonalAvg": seasonal_now,
            "SeasonVolSeasonalPath": path_summary, "SeasonVolState": state, "SeasonVolPercentile": pct,
            "SeasonVolYears": len(paths), "SeasonVolCoverage": float(np.mean(coverage)), "SeasonVolSizing": sizing}


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
            "SeasonWindowStart": None, "SeasonWindowEnd": None,
            "SeasonWindowTier": "FULL",
            "SeasonVolCurrent": np.nan, "SeasonVol60D": np.nan, "SeasonVolSeasonalAvg": np.nan,
            "SeasonVolSeasonalPath": "", "SeasonVolPercentile": np.nan, "SeasonVolYears": 0,
            "SeasonVolCoverage": np.nan, "SeasonVolState": "n/a", "SeasonVolSizing": "Normal"}


def seasonality_metrics(close, market_name=None):
    primary = _season_projection(close, SEASON_PRIMARY_YEARS)
    ten = _season_projection(close, 10)
    fifteen = _season_projection(close, 15)
    if primary is None:
        base = _seasonality_fallback()
        base.update(_season_window_stats(close, 0, market_name=market_name))
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
    out.update(_season_window_stats(close, direction, primary["anchor_date"], market_name=market_name))
    if out.get("SeasonWindow") != "-":
        _wd = int(out.get("SeasonWindowDir") or 0)
        if _wd:
            out["Seasonality"] = "Bullish" if _wd == 1 else "Bearish"
            out["SeasonalityScore"] = _wd
            direction = _wd
        out["SeasonWindowWholeYear"] = ("Match" if agreement in ("Aligned Bullish", "Aligned Bearish") and direction == s10 == s15
                                        else "Partial" if agreement == "One Neutral" else "Mixed")
        out["SeasonWindowDriver"] = "10Y/15Y + calendar window" if out["SeasonWindowWholeYear"] == "Match" else "Calendar window"
    out.update(_season_volatility(close))
    return out


def safe_seasonality_metrics(close, market_name=None):
    try:
        out = seasonality_metrics(close, market_name=market_name)
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
    cot = str(r.get("COTBias", "Neutral")); val = str(r.get("Valuation", "Neutral"))
    seas = str(r.get("Seasonality", "n/a"))
    if cot in ("Bullish", "Bearish"): score += 2
    if val in ("Bullish", "Bearish") and val == cot:
        reasons.append("flow and relative value align"); score += 2
    elif val in ("Bullish", "Bearish") and val != cot:
        reasons.append("relative value challenges flow"); score += 3
    if seas in ("Bullish", "Bearish") and cot in ("Bullish", "Bearish") and seas != cot:
        reasons.append("calendar flow conflicts with positioning"); score += 3
    elif seas == cot and seas in ("Bullish", "Bearish"):
        reasons.append("calendar flow confirms positioning"); score += 1
    if cot == "Neutral" and seas in ("Bullish", "Bearish") and str(r.get("SeasonWindowStrength")) == "STRONG":
        reasons.append("calendar-led; flow has no side"); score += 3
    if str(r.get("FundDivergence", "-")) not in ("-", "Neutral", "n/a"):
        reasons.append(f"speculator divergence {r.get('FundDivergence')}"); score += 2
    if str(r.get("EdgeFinder", "-")) != "-":
        reasons.append("qualifying historical edge"); score += 2
    if str(r.get("SeasonWindow", "-")) != "-":
        reasons.append("timed calendar window"); score += 1
    if str(r.get("SeasonVolState", "n/a")) == "Loud":
        reasons.append("seasonal volatility is loud"); score += 1
    if cot == "Neutral" and val in ("Bullish", "Bearish"):
        reasons.append("relative-value-only read; no hedger side"); score += 2
    return score, "; ".join(reasons) if reasons else "no major conflict; routine chart review"


def _build_attention_narrative(r):
    cot = str(r.get("COTBias", "Neutral")); val = str(r.get("Valuation", "Neutral"))
    seas = str(r.get("Seasonality", "n/a")); parts = []
    if cot in ("Bullish", "Bearish"):
        parts.append(f"FOLLOW / FADE LOGIC: The primary flow read is {cot.lower()}. "
                     f"{r.get('CommAgreement', '0 of 5')} hedger lookbacks agree, with "
                     f"{r.get('Horizon', '-')} as the strongest active layer. Flow positioning remains the directional anchor.")
        if r.get("RetailAgreeLayers") not in (None, "-"):
            parts.append(f"Mass fade: {r.get('RetailAgreeLayers')} agrees with the flow price direction.")
        if r.get("FundDivergence") not in (None, "-"):
            parts.append(f"Speculator divergence: {r.get('FundDivergence')}. Warning layer, not a reversal.")
    else:
        parts.append("FOLLOW / FADE LOGIC: There is no valid hedger flow side right now. "
                     "The scanner deliberately does not manufacture a bullish or bearish flow signal from mass positioning alone.")
    if val in ("Bullish", "Bearish"):
        relation = "supports" if val == cot else "challenges" if cot in ("Bullish", "Bearish") else "stands alone"
        parts.append(f"Relative value is {val.lower()} ({r.get('ValStrength', '-')}) and {relation} the primary flow read.")
    else:
        parts.append("Relative value is neutral.")
    if seas in ("Bullish", "Bearish"):
        move = _fmt(r.get("SeasonalityMove"), 2, "%"); rel = _fmt(r.get("SeasonalityReliability"), 1, "%")
        if cot in ("Bullish", "Bearish") and seas != cot:
            parts.append(f"Calendar flow points {seas.lower()} ({move}, {rel}) so it tempers conviction rather than reversing flow.")
        elif seas == cot:
            parts.append(f"Calendar flow confirms the flow side ({move}, {rel}).")
        else:
            parts.append(f"Calendar flow is {seas.lower()} ({move}, {rel}) with no flow side; "
                         f"{'promoted to a single-engine read' if r.get('SeasonWindowStrength') == 'STRONG' else 'not promoted to a trade direction'}.")
    else:
        parts.append("Calendar flow is neutral/insufficient for a directional verdict.")
    if str(r.get("SeasonWindow", "-")) != "-":
        parts.append(f"Timing window: {r.get('SeasonWindow')}. Neighbour test = {r.get('SeasonWindowNeighbour', '-')}; "
                     f"whole-year shape = {r.get('SeasonWindowWholeYear', '-')}.")
    if str(r.get("SeasonVolState", "n/a")) not in ("n/a", "-"):
        parts.append(f"Volatility seasonality: {r.get('SeasonVolState')}. Current vol {_fmt(r.get('SeasonVolCurrent'), 2, '%')} "
                     f"vs seasonal {_fmt(r.get('SeasonVolSeasonalAvg'), 2, '%')}.")
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
            c = cot_metrics(sub, invert, _measure_weekly(name, ticker, prices))
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
    seas = safe_seasonality_metrics(_measure_close(name, ticker, prices), market_name=name) if seasonality else _seasonality_fallback()
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
                    total = _safe_int(c.get(base + "Total"))
                    if (np.isfinite(wp) and np.isfinite(avg) and total >= COT_EDGE_MIN_SAMPLES
                            and wp >= 60 and ((avg > 0 and direction_label == "Bullish") or (avg < 0 and direction_label == "Bearish"))):
                        wins = _safe_int(c.get(base + "Wins"))
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
        season_in_play=bool(seas.get("SeasonWindowInPlay")))
    if seas.get("Seasonality") in ("Bullish", "Bearish") and val_bias in ("Bullish", "Bearish"):
        seas["SeasonalityConfluence"] = "Aligned with Relative Value" if seas["Seasonality"] == val_bias else "Conflicts with Relative Value"
    elif seas.get("Seasonality") in ("Bullish", "Bearish") and cot_bias in ("Bullish", "Bearish"):
        seas["SeasonalityConfluence"] = "Aligned with Flow" if seas["Seasonality"] == cot_bias else "Conflicts with Flow"
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
           "NonCommState": c["nc"]["state"] if show_cot else "n/a",
           "NonCommDir": c["nc"]["dir"] if show_cot else 0,
           "NonCommDetail": c["nc"]["detail"] if show_cot else "-",
           "NonCommFuel": c["nc"]["fuel"] if show_cot else "-",
           "NonCommShift": c["nc"]["shift_conditions"] if show_cot else 0,
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
        "SeasonWindowStart": seas.get("SeasonWindowStart"),
        "SeasonWindowEnd": seas.get("SeasonWindowEnd"),
        "SeasonWindowTier": seas.get("SeasonWindowTier", "FULL"),
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
        "NonCommState": "n/a", "NonCommDir": 0, "NonCommDetail": "-", "NonCommFuel": "-", "NonCommShift": 0,
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
        "SeasonWindowStart": None, "SeasonWindowEnd": None,
        "SeasonWindowTier": "FULL",
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
def _read_state_raw():
    data = _json_read_safe(STATE_FILE, "scan_state.json")
    return data if isinstance(data, dict) else {}


def load_state():
    s = _read_state_raw()
    base = s.get("baseline")
    if s.get("run_date") == dt.date.today().isoformat() and isinstance(base, dict) and base.get("markets"):
        return dict(base)
    return s


def save_state(table, extra=None, peer_states=None):
    prev = _read_state_raw()
    today_iso = dt.date.today().isoformat()
    if prev.get("run_date") == today_iso:
        baseline = prev.get("baseline")
    elif prev.get("markets"):
        baseline = {k: prev.get(k) for k in ("run_date", "cot_date", "markets")}
    else:
        baseline = None
    state = {"run_date": today_iso, "script_version": SCRIPT_VERSION, "markets": {}, "baseline": baseline}
    if extra:
        state.update(extra)
    for _, r in table.iterrows():
        nm = str(r.get("Market", "")).replace(" (inv)", "")
        if not nm:
            continue
        _v = _safe_float(r.get("CommExtremeCount"))
        _q = _safe_float(r.get("EdgeFinderWin"))
        entry = {
            "bias": str(r.get("COTBias", "Neutral")),
            "cot_lit": int(_v) if np.isfinite(_v) else 0,
            "season_window": str(r.get("SeasonWindow", "-")),
            "season_strength": r.get("SeasonWindowStrength") if r.get("SeasonWindowStrength") else None,
            "qualifying_rows": 1 if (r.get("EdgeFinder", "-") != "-" and np.isfinite(_q)) else 0,
            "combined": str(r.get("Fundamentals", "-")),
        }
        if peer_states and nm in peer_states:
            entry["peer_relation"] = peer_states[nm]
        state["markets"][nm] = entry
    try:
        _json_write_atomic(STATE_FILE, state)
        log.info("State saved to %s", STATE_FILE)
    except Exception as e:
        log.warning("Could not save state: %s", e)


# ============================== TRADE IDEAS ==============================
CLOSED_STATUSES = ("CLOSED", "EXPIRED", "INVALIDATED")


def _idea_id(market: str, date_str: str) -> str:
    return f"{market.replace(' ', '_')}-{date_str}"


def load_ideas():
    data = _json_read_safe(TRADE_IDEAS_FILE, "trade_ideas.json")
    if not isinstance(data, dict):
        return {"run_date": None, "ideas": {}}
    if not isinstance(data.get("ideas"), dict):
        data["ideas"] = {}
    return data


def save_ideas(data):
    try:
        _json_write_atomic(TRADE_IDEAS_FILE, data)
    except Exception as e:
        log.warning("Could not save %s: %s", TRADE_IDEAS_FILE, e)
        telegram_send_message(f"⚠️ Market Scanner could not save {TRADE_IDEAS_FILE}: {str(e)[:120]}")


def _idea_why_block(row, location_read, group=None):
    pieces = []
    group = group or str(row.get("Group", ""))
    cot = str(row.get("COTBias", "Neutral"))
    val = str(row.get("Valuation", "Neutral"))
    seas = str(row.get("Seasonality", "n/a"))
    if group == "Indices":
        if val in ("Bullish", "Bearish"):
            pieces.append(f"relative value vs rates {val.lower()}")
        if seas in ("Bullish", "Bearish"):
            pieces.append(f"calendar flow {seas.lower()}")
    else:
        if cot in ("Bullish", "Bearish"):
            pieces.append(f"flow {cot.lower()}")
        if val in ("Bullish", "Bearish"):
            if cot in ("Bullish", "Bearish"):
                pieces.append("relative value confirms" if val == cot else "relative value opposes")
            else:
                pieces.append(f"relative value {val.lower()}")
        if seas in ("Bullish", "Bearish"):
            if cot in ("Bullish", "Bearish"):
                pieces.append("calendar flow confirms" if seas == cot else "calendar flow opposes")
            else:
                pieces.append(f"calendar flow {seas.lower()}")
    return " · ".join(pieces) if pieces else "-"


def _fundamental_snapshot(row):
    return {
        "cot_bias": str(row.get("COTBias", "Neutral")),
        "comm_agreement": str(row.get("CommAgreement", "0 of 5")),
        "comm_extreme_count": _safe_int(row.get("CommExtremeCount")),
        "valuation": str(row.get("Valuation", "Neutral")),
        "val_strength": str(row.get("ValStrength", "-")),
        "seasonality": str(row.get("Seasonality", "n/a")),
        "season_window": str(row.get("SeasonWindow", "-")),
        "season_strength": row.get("SeasonWindowStrength"),
        "season_tier": str(row.get("SeasonWindowTier", "FULL")),
        "season_in_play": bool(row.get("SeasonWindowInPlay", False)),
        "edge_win": (_safe_float(row.get("EdgeFinderWin")) if np.isfinite(_safe_float(row.get("EdgeFinderWin"))) else None),
        "group": str(row.get("Group", "")),
    }


def _is_open(idea):
    return idea.get("status") not in CLOSED_STATUSES


def _fmt_px(v):
    try:
        return f"{float(v):.5g}"
    except Exception:
        return "-"


def _pick_trade_zone(pk, direction, lr):
    price = pk.get("last_price")
    if price is None or not np.isfinite(price):
        return None
    want_demand = direction == "long"
    htf = None
    if lr:
        htf = lr["demand_zone"] if want_demand else lr["supply_zone"]
    cands = []
    for z in pk.get("daily_zones") or []:
        if z["demand"] != want_demand or not zone_is_fresh(z):
            continue
        height_pct = (z["top"] - z["bottom"]) / price * 100
        gap = (price - z["top"]) / price * 100 if want_demand else (z["bottom"] - price) / price * 100
        if gap < -height_pct or gap > WATCHLIST_ZONE_REACH_PCT:
            continue
        inside = bool(htf) and _zone_intersection(z["bottom"], z["top"], float(htf[0]), float(htf[1])) is not None
        cands.append((inside, -max(gap, 0.0), z))
    if cands:
        cands.sort(key=lambda c: c[:2], reverse=True)
        return cands[0][2]
    fresh = [z for z in (pk.get("weekly_zones") or []) if zone_is_fresh(z) and z["demand"] == want_demand]
    if want_demand:
        fresh = [z for z in fresh if z["bottom"] < price]
        if not fresh:
            return None
        return max(fresh, key=lambda x: x["top"])
    fresh = [z for z in fresh if z["top"] > price]
    if not fresh:
        return None
    return min(fresh, key=lambda x: x["bottom"])


def _expire_old_ideas(data, today_iso):
    today = dt.date.fromisoformat(today_iso)
    armed_cutoff = today - dt.timedelta(days=IDEA_MAX_AGE_DAYS)
    triggered_cutoff = today - dt.timedelta(days=IDEA_TRIGGERED_MAX_AGE_DAYS)
    expired = []
    for idea in data["ideas"].values():
        if not _is_open(idea):
            continue
        if idea.get("triggered_at"):
            try:
                d = dt.date.fromisoformat(str(idea["triggered_at"])[:10])
            except Exception:
                continue
            if d < triggered_cutoff:
                idea["status"] = "EXPIRED"
                idea["closed_at"] = today_iso
                idea.setdefault("status_history", []).append(
                    {"date": today_iso, "status": "EXPIRED",
                     "note": f"{IDEA_TRIGGERED_MAX_AGE_DAYS} days past trigger"})
                expired.append(idea["market"])
        else:
            try:
                d = dt.date.fromisoformat(str(idea.get("walk_from") or idea.get("opened_at"))[:10])
            except Exception:
                continue
            if d < armed_cutoff:
                idea["status"] = "EXPIRED"
                idea["closed_at"] = today_iso
                idea.setdefault("status_history", []).append(
                    {"date": today_iso, "status": "EXPIRED", "note": f"{IDEA_MAX_AGE_DAYS} days without trigger"})
                expired.append(idea["market"])
    return expired


def _migrate_ideas(data):
    for i in data.get("ideas", {}).values():
        if not _is_open(i):
            continue
        if i.get("status") == "TRIGGERED":
            i["status"] = "ARMED"
            i["triggered_at"] = None
        if i.get("zones"):
            i["zones"] = {}


def open_ideas_from_watchlist(packs):
    data = load_ideas()
    _migrate_ideas(data)
    today = dt.date.today().isoformat()
    data["run_date"] = today
    opened, refreshed, demoted, replaced, opened_ideas = [], [], [], [], []
    for nm, pk in packs.items():
        row = _effective_row_for_decision(pk)
        if row is None:
            continue
        lr = pk.get("location_read")
        d = decide_market(row, lr, pk.get("last_price"), pk.get("location_why"))
        existing = next((i for i in data["ideas"].values() if _is_open(i) and i.get("market") == nm), None)
        if d["bucket"] != "READY":
            if existing and existing.get("status") == "ARMED":
                existing["status"] = "WATCH"
                existing.setdefault("status_history", []).append(
                    {"date": today, "status": "WATCH", "note": f"No longer ready: {d['short']}"})
                demoted.append(nm)
            continue
        direction = "long" if d["side"] == 1 else "short"
        if existing and existing.get("direction") != direction:
            if existing.get("status") in ("ARMED", "WATCH"):
                existing["status"] = "INVALIDATED"
                existing["closed_at"] = today
                existing.setdefault("status_history", []).append(
                    {"date": today, "status": "INVALIDATED", "note": "Fundamental side flipped, engagement replaced"})
                replaced.append(nm)
                existing = None
            else:
                existing.setdefault("status_history", []).append(
                    {"date": today, "status": existing.get("status"),
                     "note": f"Fundamentals now point {direction}; open engagement is against them"})
                continue
        why = _idea_why_block(row, None, pk.get("group"))
        if existing:
            existing["zones"] = {}
            existing["walk_from"] = today
            if existing.get("status") in ("ARMED", "WATCH"):
                existing["status"] = "ARMED"
            existing["why"] = why
            existing["fundamental_snapshot"] = _fundamental_snapshot(row)
            existing["score_current"] = float(row.get("AttentionScore", 0) or 0)
            existing.setdefault("status_history", []).append(
                {"date": today, "status": existing.get("status"), "note": "Re-validated Sunday"})
            refreshed.append(nm)
        else:
            idea_id = _idea_id(nm, today)
            idea = {
                "id": idea_id, "market": nm, "direction": direction,
                "opened_at": today, "walk_from": today,
                "status": "ARMED",
                "side_source": d["source"],
                "why": why,
                "zones": {},
                "fundamental_snapshot": _fundamental_snapshot(row),
                "score_at_open": float(row.get("AttentionScore", 0) or 0),
                "score_current": float(row.get("AttentionScore", 0) or 0),
                "status_history": [{"date": today, "status": "ARMED", "note": "Opened from Sunday Activation Board"}],
                "triggered_at": None, "closed_at": None, "outcome": None,
            }
            data["ideas"][idea_id] = idea
            opened.append(nm)
            opened_ideas.append(idea)
    expired = _expire_old_ideas(data, today)
    save_ideas(data)
    log.info("Ideas: %d opened, %d refreshed, %d demoted, %d replaced, %d expired",
             len(opened), len(refreshed), len(demoted), len(replaced), len(expired))
    total_open = sum(1 for i in data["ideas"].values() if _is_open(i))
    return {"opened": opened, "refreshed": refreshed, "expired": expired, "demoted": demoted,
            "replaced": replaced, "opened_ideas": opened_ideas, "total_open": total_open}


def correlation_guard(ideas):
    warnings = []
    live = [i for i in ideas.values() if i.get("status") in ("ARMED", "TRIGGERED")]
    for cluster_name, members in CORRELATION_CLUSTERS.items():
        cluster_ideas = [i for i in live if i["market"] in members]
        if len(cluster_ideas) < 2:
            continue
        for direction in ("long", "short"):
            same_dir = [i for i in cluster_ideas if i["direction"] == direction]
            if len(same_dir) >= 2:
                names = ", ".join(i["market"] for i in same_dir)
                warnings.append(
                    f"ℹ️ {len(same_dir)} {direction} {cluster_name} markets are ready ({names}). "
                    f"They sit in the same correlation cluster and must be sized as one engagement.")
    return warnings


def _walk_idea(idea, df, today=None):
    events = []
    z = idea.get("zones") or {}
    lo, hi = z.get("entry_low"), z.get("entry_high")
    if lo is None or hi is None or df is None or len(df) == 0:
        return events
    long_ = idea["direction"] == "long"
    nm = idea["market"]
    today_ts = pd.Timestamp(today or dt.date.today()).normalize()
    status = idea.get("status")
    start = idea.get("triggered_at") if status == "TRIGGERED" else (idea.get("walk_from") or idea.get("opened_at"))
    try:
        start_ts = pd.Timestamp(str(start)[:10]).normalize()
    except Exception:
        return events
    idx = pd.to_datetime(df.index).normalize()
    bars = df[idx > start_ts]
    bar_dates = idx[idx > start_ts]
    struct_level = float(lo) if long_ else float(hi)
    for (ts, bar), bd in zip(bars.iterrows(), bar_dates):
        h, l, c = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        done = bd < today_ts
        date_iso = bd.date().isoformat()
        if status in ("ARMED", "WATCH"):
            breached = done and ((long_ and c < struct_level) or ((not long_) and c > struct_level))
            if breached:
                status = "INVALIDATED"
                idea["status"] = status; idea["closed_at"] = date_iso; idea["outcome"] = "never filled"
                note = f"Closed through the {('demand' if long_ else 'supply')} reference before the engagement filled"
                idea.setdefault("status_history", []).append({"date": date_iso, "status": status, "note": note})
                events.append({"kind": "INVALIDATION_BREACH", "market": nm, "date": date_iso, "idea": idea, "text": note})
                break
            if status == "ARMED" and l <= hi and h >= lo:
                status = "TRIGGERED"
                idea["status"] = status; idea["triggered_at"] = date_iso
                note = f"Price engaged the reference {_fmt_px(lo)}–{_fmt_px(hi)}"
                idea.setdefault("status_history", []).append({"date": date_iso, "status": status, "note": note})
                events.append({"kind": "ZONE_TOUCHED", "market": nm, "date": date_iso, "idea": idea, "text": note})
                continue
        elif status == "TRIGGERED":
            breached = done and ((long_ and c < struct_level) or ((not long_) and c > struct_level))
            if breached:
                status = "INVALIDATED"
                idea["status"] = status; idea["closed_at"] = date_iso
                idea["outcome"] = "structural invalidation"
                note = f"Closed back through the {('demand' if long_ else 'supply')} reference — set-up is dead"
                idea.setdefault("status_history", []).append({"date": date_iso, "status": status, "note": note})
                events.append({"kind": "INVALIDATION_BREACH", "market": nm, "date": date_iso, "idea": idea, "text": note})
                break
    return events


def valuation_bias_for(name, group, ticker, prices, weekly):
    refs = [r for r in REFS_BY_GROUP.get(group, []) if REFS[r] != ticker]
    if name == "Gold":
        refs = ["SILVER", "DXY"]
    if not refs:
        refs = [r for r in REFS if REFS[r] != ticker]
    short_len = ROC_SHORT_INDICES if group == "Indices" else ROC_SHORT
    n = short_len if VAL_PRIMARY == "short" else ROC_LONG
    sig = {}
    for tf, px in (("D", prices), ("W", weekly)):
        vals = []
        if ticker in px.columns:
            a_px = px[ticker].dropna()
            for r in refs:
                t = REFS[r]
                if t not in px.columns:
                    continue
                try:
                    vals.append(valuation(a_px, px[t].dropna(), n))
                except Exception:
                    vals.append(np.nan)
        sig[tf] = val_signal(vals) if vals else 0
    vsig, _, label = valuation_state(sig["D"], sig["W"])
    return {1: "Bullish", -1: "Bearish", 0: "Neutral"}[vsig], label


def _recheck_fundamentals(idea, item, ctx):
    out = {"weakened": [], "strengthened": []}
    direction = 1 if idea["direction"] == "long" else -1
    snap = idea.get("fundamental_snapshot", {}) or {}
    nm = idea["market"]
    m = next((x for x in MARKETS if x[0] == nm), None)
    prices = (ctx or {}).get("prices")
    weekly = (ctx or {}).get("weekly")
    if m is None:
        return out
    _, group, ticker, _, _ = m
    if prices is not None and weekly is not None and ticker in prices.columns:
        try:
            vb, _ = valuation_bias_for(nm, group, ticker, prices, weekly)
            old = snap.get("valuation", "Neutral")
            if vb != old and vb in _DIRS:
                if _DIRS[vb] == -direction:
                    out["weakened"].append(f"relative value now {vb.lower()}, against the engagement")
                elif _DIRS[vb] == direction:
                    out["strengthened"].append(f"relative value now {vb.lower()}, supports the engagement")
        except Exception as e:
            log.warning("Valuation recheck failed for %s: %s", nm, e)
        try:
            ser = prices[ticker].dropna()
            _sp = _rpt_season_pack(ser, dt.date.today(), market_name=nm)
            _m = _sp.get("main") if _sp else None
            seas = ("Bullish" if _m["dir"] == 1 else "Bearish") if _m else "n/a"
            old = snap.get("seasonality", "n/a")
            if seas != old and seas in _DIRS:
                if _DIRS[seas] == -direction:
                    out["weakened"].append(f"calendar flow now points {seas.lower()}, against the engagement")
                elif _DIRS[seas] == direction and old not in _DIRS:
                    out["strengthened"].append(f"calendar flow now {seas.lower()}, supports the engagement")
        except Exception as e:
            log.warning("Seasonality recheck failed for %s: %s", nm, e)
    return out


def _board_line(idea, price=None):
    s = idea.get("fundamental_snapshot", {}) or {}
    bits = [f"flow {str(s.get('cot_bias', 'n/a')).lower()}", f"relative value {str(s.get('valuation', 'n/a')).lower()}",
            f"calendar flow {str(s.get('seasonality', 'n/a')).lower()}"]
    flag = f" · ⚠️ {'; '.join(idea['flags'])}" if idea.get("flags") else ""
    state = "ready" if idea.get("status") == "ARMED" else "on watch (no longer ready)"
    return f"{idea['market']} {idea['direction'].title()} · {state} since {idea.get('opened_at', '-')} · " + " · ".join(bits) + flag


def check_ideas(chart_data, ctx=None):
    data = load_ideas()
    _migrate_ideas(data)
    today = dt.date.today().isoformat()
    if not any(_is_open(i) for i in data.get("ideas", {}).values()):
        return None
    expired = _expire_old_ideas(data, today)
    events, weakened, strengthened, board = [], [], [], []
    for idea in data["ideas"].values():
        if not _is_open(idea):
            continue
        nm = idea["market"]
        item = (chart_data or {}).get(nm)
        if not item:
            board.append(f"{nm} {idea['direction'].title()} · no price data this run")
            continue
        if _is_open(idea):
            r = _recheck_fundamentals(idea, item, ctx)
            for w in r["weakened"]:
                weakened.append((nm, idea, w))
                idea.setdefault("status_history", []).append({"date": today, "status": idea.get("status"), "note": f"Weakened: {w}"})
            for s_ in r["strengthened"]:
                strengthened.append((nm, idea, s_))
                idea.setdefault("status_history", []).append({"date": today, "status": idea.get("status"), "note": f"Strengthened: {s_}"})
            idea["flags"] = r["weakened"]
            board.append(_board_line(idea))
    data["run_date"] = today
    data["last_wednesday"] = today
    save_ideas(data)
    return {"events": events, "weakened": weakened, "strengthened": strengthened, "expired": expired,
            "board": board,
            "total": sum(1 for i in data["ideas"].values() if _is_open(i))}


def daily_watch(chart_data, ctx=None):
    data = load_ideas()
    _migrate_ideas(data)
    events = []
    changed = False
    today = dt.date.today().isoformat()
    prices, weekly = (ctx or {}).get("prices"), (ctx or {}).get("weekly")
    for idea in data.get("ideas", {}).values():
        if not _is_open(idea):
            continue
        nm = idea["market"]
        m = next((x for x in MARKETS if x[0] == nm), None)
        if m is None or prices is None or weekly is None or m[2] not in prices.columns:
            continue
        try:
            vb, vlabel = valuation_bias_for(nm, m[1], m[2], prices, weekly)
        except Exception as e:
            log.warning("Daily valuation check failed for %s: %s", nm, e)
            continue
        snap = idea.setdefault("fundamental_snapshot", {})
        old = snap.get("valuation", "Neutral")
        if vb == old:
            continue
        direction = 1 if idea["direction"] == "long" else -1
        snap["valuation"] = vb
        changed = True
        if vb in _DIRS and _DIRS[vb] == -direction:
            kind, text = "VALUATION_AGAINST", f"relative value flipped to {vb.lower()} ({vlabel}), against the engagement"
        elif vb in _DIRS and _DIRS[vb] == direction:
            kind, text = "VALUATION_SUPPORT", f"relative value now {vb.lower()} ({vlabel}), supports the engagement"
        else:
            continue
        idea.setdefault("status_history", []).append({"date": today, "status": idea.get("status"), "note": text})
        events.append({"kind": kind, "market": nm, "date": today, "idea": idea, "text": text})
    if events or changed:
        save_ideas(data)
    return events


_EVENT_ICON = {"VALUATION_AGAINST": "⚠️", "VALUATION_SUPPORT": "✅"}


def _event_lines(events):
    lines = []
    for e in events:
        idea = e["idea"]
        icon = _EVENT_ICON.get(e["kind"], "•")
        lines.append(f"{icon} {e['market']} {idea['direction'].title()} — {e['text']}")
    return lines


def _ideas_telegram_wednesday(delta):
    today = dt.date.today().strftime("%d %b %Y")
    if delta is None:
        return f"📊 Mid-Week Check — {today}\nNo engagements under watch."
    lines = [f"📊 Mid-Week Check — {today}"]
    ev = _event_lines(delta["events"])
    if ev:
        lines.append("\nSince the last check:")
        lines += ev
    if delta["weakened"]:
        lines.append("\n⚠️ WEAKENED — re-check before acting")
        for nm, idea, w in delta["weakened"]:
            lines.append(f"• {nm} {idea['direction'].title()} — {w}")
    if delta["strengthened"]:
        lines.append("\n✅ STRENGTHENED")
        for nm, idea, s_ in delta["strengthened"]:
            lines.append(f"• {nm} {idea['direction'].title()} — {s_}")
    if delta["expired"]:
        lines.append(f"\n⏸ Expired: {', '.join(delta['expired'])}")
    if not (ev or delta["weakened"] or delta["strengthened"]):
        lines.append("\nNo engine flipped for or against a live engagement.")
    if delta["board"]:
        lines.append(f"\nLive engagements ({delta['total']}):")
        lines += [f"• {b}" for b in delta["board"]]
    lines.append("\nFlow positioning is frozen until the Friday report; relative value and calendar flow were re-read today.")
    return "\n".join(lines)


def _ideas_telegram_daily(events):
    if not events:
        return None
    lines = [f"⚡ Between-scan alert — {dt.date.today().strftime('%d %b')}"]
    lines += _event_lines(events)
    return "\n".join(lines)


def _ideas_telegram_sunday(ideas_summary, corr_warnings):
    if not ideas_summary:
        return None
    lines = ["🟢 Live engagements"]

    new_by_cat = {}
    for idea in ideas_summary.get("opened_ideas", []):
        new_by_cat.setdefault(_market_category(idea["market"]), []).append(idea)
    for cat in REPORT_CATEGORY_ORDER + ["Other"]:
        ideas = new_by_cat.get(cat)
        if not ideas:
            continue
        icon = REPORT_CATEGORY_ICONS.get(cat, "•")
        lines.append(f"\n{icon} {cat.upper()}")
        for idea in ideas:
            lines.append(f"✅ NEW · {idea['market']} {idea['direction'].title()}")
            lines.append(f"    {idea.get('why', '-')}")

    if ideas_summary["refreshed"]:
        ref_by_cat = {}
        for nm in ideas_summary["refreshed"]:
            ref_by_cat.setdefault(_market_category(nm), []).append(nm)
        lines.append("\n🔄 Re-validated")
        for cat in REPORT_CATEGORY_ORDER + ["Other"]:
            if cat in ref_by_cat:
                lines.append(f"· {cat}: {', '.join(ref_by_cat[cat])}")

    if ideas_summary.get("demoted"):
        lines.append(f"⏬ No longer ready (kept on watch): {', '.join(ideas_summary['demoted'])}")
    if ideas_summary.get("replaced"):
        lines.append(f"🔁 Replaced (side flipped): {', '.join(ideas_summary['replaced'])}")
    if ideas_summary["expired"]:
        lines.append(f"⏸ Expired: {', '.join(ideas_summary['expired'])}")
    lines.append(f"\nTotal live: {ideas_summary['total_open']}")
    if corr_warnings:
        lines.append("")
        lines.extend(corr_warnings)
    return "\n".join(lines)


def render_idea_text(idea):
    return f"{idea['market']} — {idea['direction'].title()} — ready\n  Why: {idea.get('why', '-')}"


# ============================== BLUEPRINT QUALIFIER (CSV) ==============================
def _blueprint_season_state(r, bias):
    season = str(r.get("Seasonality", "Neutral"))
    in_play = bool(r.get("SeasonWindowInPlay", False))
    window = str(r.get("SeasonWindow", "-")).strip()
    if season not in ("Bullish", "Bearish") or not window or window == "-":
        return "NEUTRAL", "No clear directional calendar window is available."
    if in_play and season == bias:
        return "ALIGNED", f"The active calendar tendency supports the {bias.lower()} bias."
    if in_play:
        return "OPPOSING", f"The active calendar tendency points {season.lower()}, against the {bias.lower()} bias."
    return "UPCOMING / NOT ACTIVE", "A calendar tendency exists, but the scanner does not mark its window as active."


def _blueprint_qualify_row(r):
    fv = fundamental_view(r, str(r.get("Group", "")))
    if fv["side"] == 0:
        return ("DORMANT", "Neutral", "Core directional evidence is inconclusive.",
                "No action unless new evidence changes the picture.", "NEUTRAL")
    bias = "Bullish" if fv["side"] == 1 else "Bearish"
    ss, sr = _blueprint_season_state(r, bias)
    src = fv["source"]
    if fv["season_led"]:
        return ("PARTIAL — calendar-only", bias,
                f"Only calendar flow has a side ({bias.lower()}). It never engages alone. {sr}",
                "Wait for the main engine before considering a setup.", ss)
    if fv["conflicts"]:
        engines = " and ".join(fv["conflicts"])
        status = "PARTIAL — calendar conflict" if fv["conflicts"] == ["seasonality"] else "PARTIAL — conflict"
        return (status, bias, f"{src} supports {bias.lower()}, while {engines} point the other way. {sr}",
                "Resolve the evidence conflict and verify the chart before considering a setup.", ss)
    if fv["confirms"]:
        return ("ALIGNED", bias, f"{src} supports {bias.lower()} and {', '.join(fv['confirms'])}. {sr}",
                "Fundamentals line up. The chart work and the execution are your call.", ss)
    return ("PARTIAL — evidence incomplete", bias,
            f"{src} supports {bias.lower()}, but independent confirmation is incomplete. {sr}",
            "Review higher-timeframe context and the missing confirmation.", ss)


def apply_blueprint_qualifier(table):
    if table is None or table.empty:
        return table
    out = table.copy()
    vals = out.apply(_blueprint_qualify_row, axis=1, result_type="expand")
    vals.columns = ["BlueprintStatus", "BlueprintBias", "BlueprintReason", "BlueprintNextStep", "BlueprintSeasonState"]
    for c in vals.columns:
        out[c] = vals[c]
    out["BlueprintRank"] = out["BlueprintStatus"].map(
        {"ALIGNED": 0, "PARTIAL — calendar conflict": 1, "PARTIAL — conflict": 2,
         "PARTIAL — evidence incomplete": 3, "PARTIAL — calendar-only": 4, "DORMANT": 5}).fillna(6)
    return out


# ============================== PDF ==============================
RPT = dict(
    bg="#0d1117",
    card="#161b22",
    ink="#e6edf3",
    muted="#8b949e",
    rule="#21262d",
    green="#2ea043",
    red="#f85149",
    amber="#d29922",
    blue="#58a6ff",
    navy="#0d1117",
    lav="#1c2128",
    greentint="#0f2a1a",
    redtint="#2a0f0f",
    grey="#21262d",
)
RPT_TABLE_ORDER = [("6M", "6 mo"), ("1Y", "1 yr"), ("2Y", "2 yr"), ("3Y", "3 yr"), ("5Y", "Multi-yr")]
RPT_CHIP_ORDER = list(reversed(RPT_TABLE_ORDER))
RPT_LOOKBACK_HISTORY_YEARS = {
    "6M": 5,
    "1Y": 5,
    "2Y": 8,
    "3Y": 12,
    "5Y": 15,
}
RPT_DIRWORD = {1: "LONG", -1: "SHORT", 0: "STAND ASIDE"}
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
    if nm in RPT_FX_PAIRS and len(nm) == 6:
        pair = f"{nm[:3]}/{nm[3:]}"
        return nm, nm, f"{pair} ({RPT_MARKET_LABELS[nm][1]})"
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
    basis = pd.Series(_cot_basis(net, oi).values, index=pd.to_datetime(c["date"]))
    out = {"basis": basis, "layers": {}}
    def _idx(a):
        return _rank_pct(a)
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


def _rpt_track_moves(vals, dates, px, role, invert, now, h, cutoff, mode="gap", out_weeks=None):
    """Collect post-horizon moves under a given episode rule.

    mode='gap'  — current live rule: >= CASE_DEDUP_WEEKS since last *case index*
    mode='out'  — must be non-extreme for >= out_weeks consecutive weeks before a
                  new entry counts as a new episode (reference-style).
    """
    moves = []
    if mode == "gap":
        last_case_i = -9999
        for i in range(len(vals)):
            if vals[i] == 0 or (i > 0 and vals[i] == vals[i - 1]):
                continue
            if _rpt_price_dir(int(vals[i]), role, invert) != now:
                continue
            if i - last_case_i < CASE_DEDUP_WEEKS:
                continue
            last_case_i = i
            if cutoff is not None and pd.Timestamp(dates[i]) < cutoff:
                continue
            pos0 = px.index.searchsorted(dates[i], side="right") - 1
            if pos0 < 0 or pos0 + h >= len(px):
                continue
            move = (float(px.iloc[pos0 + h]) / float(px.iloc[pos0]) - 1.0) * 100.0 * now
            moves.append(move)
        return moves

    # out-of-zone episode rule
    n_out = int(out_weeks if out_weeks is not None else CASE_DEDUP_WEEKS)
    weeks_out = 10 ** 9  # first entry always allowed
    last_ext = 0
    for i in range(len(vals)):
        v = int(vals[i])
        if v == 0:
            weeks_out += 1
            last_ext = 0
            continue
        is_entry = (last_ext != v)
        last_ext = v
        if not is_entry:
            continue
        if _rpt_price_dir(v, role, invert) != now:
            continue
        if weeks_out < n_out:
            continue
        if cutoff is not None and pd.Timestamp(dates[i]) < cutoff:
            weeks_out = 0
            continue
        pos0 = px.index.searchsorted(dates[i], side="right") - 1
        if pos0 < 0 or pos0 + h >= len(px):
            weeks_out = 0
            continue
        move = (float(px.iloc[pos0 + h]) / float(px.iloc[pos0]) - 1.0) * 100.0 * now
        moves.append(move)
        weeks_out = 0
    return moves


def _rpt_track(layer, weekly_px, role, invert, h, history_years=None):
    now = layer["pdir"]
    if now == 0 or weekly_px is None:
        return None
    px = weekly_px.dropna().copy(); px.index = pd.to_datetime(px.index)
    d = layer["d"]; dates = d.index; vals = d.values
    cutoff = None
    if history_years is not None and len(px) > 0:
        cutoff = pd.Timestamp(px.index[-1]) - pd.DateOffset(years=history_years)
    moves = _rpt_track_moves(vals, dates, px, role, invert, now, h, cutoff, mode="gap")
    moves_all = list(moves)
    if RPT_TRACK_CASE_CAP:
        moves = moves[-int(RPT_TRACK_CASE_CAP):]
    if len(moves_all) > len(moves):
        log.info("Track %s %dw lookback=%dw: showing last %d of %d cases (uncapped: %.0f%% win, %+.2f%% avg)",
                 role, h, layer.get("w", 0), len(moves), len(moves_all),
                 sum(m_ > 0 for m_ in moves_all) / len(moves_all) * 100.0, float(np.mean(moves_all)))
    # bp30d: log-only episode-rule comparison for live extremes (does not change cases)
    if CASE_DEDUP_DIAG and not (layer["weeks_in"] >= h) and moves_all:
        try:
            def _summ(mv):
                c = len(mv)
                if not c:
                    return "0 cases"
                w = sum(m_ > 0 for m_ in mv)
                return f"{w} of {c} · {w/c*100:.0f}% · {float(np.mean(mv)):+.2f}%"
            parts = [f"gap{CASE_DEDUP_WEEKS}={_summ(moves_all)}"]
            for ow in CASE_DEDUP_DIAG_OUT_WEEKS:
                mv_o = _rpt_track_moves(vals, dates, px, role, invert, now, h, cutoff,
                                        mode="out", out_weeks=ow)
                parts.append(f"out{ow}={_summ(mv_o)}")
            log.info("CASE-DIAG %s lb=%sw h=%sw | %s",
                     role, layer.get("w", 0), h, " | ".join(parts))
        except Exception as e:
            log.debug("CASE-DIAG failed: %s", e)
    cases = len(moves); wins = sum(m > 0 for m in moves)
    actual_years = max(1, int(round(layer["valid"] / 52.0)))
    display_years = min(history_years, actual_years) if history_years is not None else actual_years
    return {"live": True, "past": layer["weeks_in"] >= h, "weeks_in": layer["weeks_in"],
            "cases": cases, "wins": wins, "cases_all": len(moves_all),
            "win": (wins / cases * 100.0) if cases else np.nan,
            "avg": float(np.mean(moves)) if cases else np.nan,
            "history": display_years, "dir": now}


def _rpt_best_candidate_at(tracks, h):
    """Return the single strongest qualifying row at horizon h, or None."""
    cands = []
    for (role, key, hh), t in tracks.items():
        if hh != h or not t or t["past"]:
            continue
        if t["cases"] >= REPORT_EDGE_MIN_CASES and t["win"] >= REPORT_EDGE_MIN_WIN and t["avg"] > 0:
            cands.append((t["win"], t["cases"], t["avg"], role, key))
    cands.sort(reverse=True)
    if not cands:
        return None
    _, _, _, r, k = cands[0]
    return {"role": r, "key": k, "h": h, "t": tracks[(r, k, h)]}


def _rpt_cot_pack(sub, weekly_px, invert):
    sub = sub.sort_values("date").reset_index(drop=True)
    comm = _rpt_role_state(sub, "commercial", invert)
    ret = _rpt_role_state(sub, "retail", invert)
    tracks = {}
    for role, st in (("commercial", comm), ("retail", ret)):
        for key, _ in RPT_TABLE_ORDER:
            _hy = RPT_LOOKBACK_HISTORY_YEARS.get(key)
            for h in COT_EDGE_HORIZONS:
                tracks[(role, key, h)] = _rpt_track(st["layers"][key], weekly_px, role, invert, h,
                                                    history_years=_hy)
    bias = comm["pdir"]
    lit = (comm["lit"] if comm["pdir"] == bias else 0) + (ret["lit"] if ret["pdir"] == bias and bias else 0)
    conflict = bool(bias and ret["pdir"] == -bias)
    h4 = COT_EDGE_HORIZONS[0]
    h8 = COT_EDGE_HORIZONS[-1]
    edge = _rpt_best_candidate_at(tracks, h4)
    edge8 = _rpt_best_candidate_at(tracks, h8)
    qualifying = []
    for (role, key, h), t in tracks.items():
        if h != h4 or not t or t["past"]:
            continue
        if t["cases"] >= REPORT_EDGE_MIN_CASES and t["win"] >= REPORT_EDGE_MIN_WIN and t["avg"] > 0:
            qualifying.append((role, key))
    funds = None
    try:
        spec_net = (sub["spec_long"] - sub["spec_short"]).astype(float)
        funds = noncomm_read(spec_net, weekly_px, invert, comm["pdir"], ret["pdir"],
                             dates=sub["date"])
        if funds is None or funds.get("state") in (None, "n/a"):
            funds = noncomm_read(spec_net, weekly_px, invert, comm["pdir"], ret["pdir"], dates=None)
    except Exception as e:
        log.debug("_rpt_cot_pack noncomm_read failed: %s", e)
        funds = None
    return {"comm": comm, "ret": ret, "tracks": tracks, "bias": bias, "lit": lit, "conflict": conflict,
            "edge": edge, "edge8": edge8, "qualifying": qualifying, "funds": funds}


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


def _rpt_ret_matrix_td(close, bases, cal_lens, years, cur_year):
    c = close.dropna()
    c = c[~c.index.duplicated(keep="last")].sort_index()
    R = np.full((years, len(bases), len(cal_lens)), np.nan)
    cur = c[c.index >= pd.Timestamp(year=cur_year, month=1, day=1)]
    if len(cur) == 0:
        return R
    last = cur.index[-1]
    n_cur = len(cur)
    cur_idx = cur.index.values.astype("datetime64[D]")
    last_d = np.datetime64(last.date(), "D")

    def tdn_many(dates):
        d = np.asarray(dates, dtype="datetime64[D]")
        inside = np.searchsorted(cur_idx, d, side="right")
        beyond = n_cur + np.busday_count(last_d + np.timedelta64(1, "D"), d + np.timedelta64(1, "D"))
        return np.where(d > last_d, beyond, inside).astype(int)

    bases_d = np.array([np.datetime64(pd.Timestamp(b).date(), "D") for b in bases])
    lens_d = np.asarray(cal_lens, dtype="int64")
    N0 = tdn_many(bases_d)
    ends = bases_d[:, None] + lens_d[None, :].astype("timedelta64[D]")
    N1 = tdn_many(ends.ravel()).reshape(ends.shape)
    N0b = np.broadcast_to(N0[:, None], N1.shape)
    for k in range(1, years + 1):
        sg = c[c.index >= pd.Timestamp(year=cur_year - k, month=1, day=1)].values.astype(float)
        ok = (N0b >= 1) & (N1 > N0b) & (N1 <= len(sg))
        if not ok.any():
            continue
        i0 = np.where(ok, N0b - 1, 0)
        i1 = np.where(ok, N1 - 1, 0)
        p0 = sg[i0]
        p1 = sg[i1]
        good = ok & np.isfinite(p0) & np.isfinite(p1) & (p0 != 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            R[k - 1] = np.where(good, (p1 / p0 - 1.0) * 100.0, np.nan)
    return R


def _season_shape_label(length_days):
    try:
        L = int(length_days)
    except Exception:
        return "broad"
    if L <= SEASON_SHARP_MAX_DAYS:
        return "sharp"
    if L <= SEASON_MID_MAX_DAYS:
        return "mid-length"
    return "broad"


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


def _season_persistence(R, e_idx, l_idx, direction, min_wins=None, window=None):
    min_wins = min_wins if min_wins is not None else SEASON_PERSISTENCE_MIN_WINS
    window = window if window is not None else SEASON_PERSISTENCE_WINDOW
    try:
        series = np.asarray(R[:window, e_idx, l_idx], dtype=float)
        valid = np.isfinite(series)
        if int(valid.sum()) < min(min_wins, window):
            return False
        wins = int((((series * direction) > 0) & valid).sum())
        return wins >= min_wins
    except Exception:
        return False


def _season_shape_tier(shape_match):
    try:
        s = float(shape_match)
    except Exception:
        return "WATCH"
    if not np.isfinite(s):
        return "WATCH"
    if s >= SEASON_SHAPE_FULL:
        return "FULL"
    if s >= SEASON_SHAPE_WATCH:
        return "WATCH"
    return "DROP"


def _season_display_strength(sp):
    if not sp or not sp.get("main"):
        return ("-", "FULL")
    m = sp["main"]
    strength = m.get("strength", "MODERATE") or "MODERATE"
    tier = sp.get("tier", "FULL")
    if tier == "FULL":
        return (strength, "FULL")
    if tier == "WATCH":
        return (f"{strength} (off-pattern this year)", "WATCH")
    return (f"{strength} (off-pattern, chapter only)", "DROP")


def _load_season_windows():
    data = _json_read_safe(SEASON_WINDOW_STATE_FILE, "season_windows.json")
    if not isinstance(data, dict):
        return {"locked": {}, "version": SCRIPT_VERSION}
    if not isinstance(data.get("locked"), dict):
        data["locked"] = {}
    # bp30: windows locked by an older detection method (futures basis, 60d cap,
    # 4-week episode gap) must not survive the methodology change — re-detect.
    if data.get("version") != SCRIPT_VERSION:
        if data["locked"]:
            log.info("Seasonal locks reset: written by %s, running %s — windows will be re-detected.",
                     data.get("version", "an older version"), SCRIPT_VERSION)
        data["locked"] = {}
        data["version"] = SCRIPT_VERSION
    return data


def _save_season_windows(data):
    try:
        data["version"] = SCRIPT_VERSION
        _json_write_atomic(SEASON_WINDOW_STATE_FILE, data)
    except Exception as e:
        log.warning("Could not save %s: %s", SEASON_WINDOW_STATE_FILE, e)


def _ensure_season_windows_file():
    """bp28: guarantee the file exists so a missing file is never confused with
    'no locks were written'."""
    p = Path(SEASON_WINDOW_STATE_FILE)
    if not p.exists():
        try:
            _json_write_atomic(SEASON_WINDOW_STATE_FILE, {"locked": {}})
            log.info("Created empty %s", SEASON_WINDOW_STATE_FILE)
        except Exception as e:
            log.warning("Could not create %s: %s", SEASON_WINDOW_STATE_FILE, e)


def _rpt_score_window(close, start, length, direction, today_ts):
    try:
        d = _rpt_daily(close)
    except Exception:
        return None
    if len(d) < 365 * 9:
        return None
    years = min(REPORT_SEASON_YEARS, int(len(d) / 365.25) - 1)
    if years < 8:
        return None
    today = pd.Timestamp(today_ts or d.index[-1]).normalize()
    s0 = pd.Timestamp(start).normalize()
    L = int(length)
    entries = [s0]
    lens = [L]
    R = _rpt_ret_matrix_td(close, entries, lens, years, int(today.year))
    win, cases, wins, avg, rw, rc = _rpt_window_stats(R, direction)
    if cases[0, 0] < 1:
        return None
    st = _rpt_strength(win[0, 0], cases[0, 0], rw[0, 0], rc[0, 0], avg[0, 0])
    if st is None:
        return None
    nb_entries = [s0 + pd.Timedelta(days=o) for o in _neighbour_shifts_for(L)]
    Rn = _rpt_ret_matrix_td(close, nb_entries, [L], years, int(today.year))
    nwin, _, _, _, _, _ = _rpt_window_stats(Rn, direction)
    nb = float(np.nanmean(nwin)) if np.isfinite(nwin).any() else np.nan
    w = float(win[0, 0])
    timing = _timing_label_from_nb(nb)
    return {"dir": direction, "start": s0, "end": s0 + pd.Timedelta(days=L), "len": L,
            "shape": _season_shape_label(L), "win": w,
            "cases": int(cases[0, 0]), "wins": int(wins[0, 0]),
            "avg": float(avg[0, 0]), "rec_wins": int(rw[0, 0]), "rec_cases": int(rc[0, 0]),
            "strength": st, "neighbour": nb, "timing": timing, "years": years,
            "persistent": True}


def _timing_label_from_nb(nb):
    if not np.isfinite(nb):
        return "moderate"
    if nb >= SEASON_TIMING_FORGIVING_BAR:
        return "forgiving"
    if nb >= SEASON_TIMING_TIGHT_BAR:
        return "tight"
    return "very tight"


def _rpt_season_pack_raw(close, today=None):
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
    entries = [today + pd.Timedelta(days=o) for o in range(-REPORT_SEASON_BEHIND_DAYS, REPORT_SEASON_AHEAD_DAYS + 1)
               if (today + pd.Timedelta(days=o)).weekday() < 5]
    lens = list(range(7, SEASON_MAX_WINDOW_DAYS + 1))
    R = _rpt_ret_matrix_td(close, entries, lens, years, int(today.year))
    best = {}
    entry_offsets = np.array([(e - today).days for e in entries], dtype=float)
    prox = np.clip(1.0 - np.maximum(entry_offsets - 21, 0) / max(REPORT_SEASON_AHEAD_DAYS - 21, 1), 0.55, 1.0)
    for direction in (1, -1):
        win, cases, wins, avg, rw, rc = _rpt_window_stats(R, direction)
        okmask = np.isfinite(win) & (cases >= max(8, int(years * 0.6))) & (avg * direction > 0)
        lenbonus = np.array([min(L, 21) / 21.0 * 0.04 for L in lens])[None, :]
        prox_bonus = prox[:, None] * 0.06
        score = np.where(okmask, win + lenbonus + np.minimum(np.abs(avg), 3) * 0.01 + prox_bonus, -9.0)
        if not np.isfinite(score).any() or score.max() < 0:
            continue
        # bp28: walk candidates past the top-scored winner if it fails persistence.
        flat_order = np.argsort(score.ravel())[::-1]
        picked = None
        for flat_idx in flat_order[:CANDIDATE_WALK_LIMIT]:
            e_i, l_i = np.unravel_index(int(flat_idx), score.shape)
            if not np.isfinite(score[e_i, l_i]) or score[e_i, l_i] < 0:
                break
            st_i = _rpt_strength(win[e_i, l_i], cases[e_i, l_i], rw[e_i, l_i], rc[e_i, l_i], avg[e_i, l_i])
            if st_i is None:
                continue
            if _season_persistence(R, e_i, l_i, direction):
                picked = (e_i, l_i, st_i)
                break
        if picked is None:
            log.debug("Season: no persistent candidate for direction %+d", direction)
            continue
        e, l, st = picked
        persistent = True
        L = lens[l]; s0 = entries[e]
        nb_entries = [s0 + pd.Timedelta(days=o) for o in _neighbour_shifts_for(L)]
        Rn = _rpt_ret_matrix_td(close, nb_entries, [L], years, int(today.year))
        nwin, _, _, _, _, _ = _rpt_window_stats(Rn, direction)
        nb = float(np.nanmean(nwin)) if np.isfinite(nwin).any() else np.nan
        w = float(win[e, l])
        timing = _timing_label_from_nb(nb)
        best[direction] = {"dir": direction, "start": s0, "end": s0 + pd.Timedelta(days=L), "len": L,
                           "shape": _season_shape_label(L), "win": w,
                           "cases": int(cases[e, l]), "wins": int(wins[e, l]),
                           "avg": float(avg[e, l]), "rec_wins": int(rw[e, l]), "rec_cases": int(rc[e, l]),
                           "strength": st, "neighbour": nb, "timing": timing, "years": years,
                           "persistent": bool(persistent)}
    main = None
    cancelled = False
    if best:
        DISPLAY_AHEAD_DAYS = 35
        near = {d_: b for d_, b in best.items() if (b["start"] - today).days <= DISPLAY_AHEAD_DAYS}
        pool = near if near else best
        def _main_key(b):
            days_out = (b["start"] - today).days
            return (b["strength"] == "STRONG", b["win"], abs(b.get("avg", 0)), -days_out)
        main = sorted(pool.values(), key=_main_key, reverse=True)[0]
        # bp30c: opposing-window cancel — when both directions have a live window
        # of similar strength in the horizon, neither earns a calendar side
        # (reference: "the calendar cancels itself out this week").
        if main and len(best) >= 2:
            opp = best.get(-int(main["dir"]))
            if opp and opp.get("strength") in ("STRONG", "MODERATE"):
                close_records = abs(float(main["win"]) - float(opp["win"])) <= 0.10
                both_strong = main.get("strength") == "STRONG" and opp.get("strength") == "STRONG"
                if close_records or both_strong:
                    cancelled = True
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
            n = min(len(cur), m)

            def _detrend(arr):
                arr = np.asarray(arr, dtype=float)
                if len(arr) < 2:
                    return arr
                slope = (arr[-1] - arr[0]) / (len(arr) - 1)
                return arr - slope * np.arange(len(arr), dtype=float)

            if SHAPE_DETREND:
                avgp = np.mean([_detrend(p[:n]) for p in paths], axis=0)
                cp = _detrend((cur.values[:n] / cur.values[0] - 1.0) * 100.0)
            else:
                avgp = np.mean([np.asarray(p[:n], dtype=float) for p in paths], axis=0)
                cp = (cur.values[:n] / cur.values[0] - 1.0) * 100.0
            if np.std(avgp) > 1e-9 and np.std(cp) > 1e-9:
                shape = float(np.corrcoef(avgp, cp)[0, 1])
    except Exception:
        shape = np.nan
    tier = _season_shape_tier(shape)
    return {"d": d, "today": today, "best": best, "main": main, "shape": shape, "years": years,
            "tier": tier, "cancelled": bool(cancelled)}


def _rpt_season_pack_locked(close, market_name, today=None):
    lock_data = _load_season_windows()
    today_ts = pd.Timestamp(today or dt.date.today()).normalize()
    entry = lock_data["locked"].get(market_name)
    if entry:
        try:
            if pd.Timestamp(entry.get("end")) < today_ts:
                lock_data["locked"].pop(market_name, None)
                _save_season_windows(lock_data)
                entry = None
        except Exception:
            lock_data["locked"].pop(market_name, None)
            _save_season_windows(lock_data)
            entry = None

    if entry:
        best = {}
        for dir_key in ("1", "-1"):
            spec = entry.get(dir_key)
            if not spec:
                continue
            dirn = int(dir_key)
            scored = None
            try:
                scored = _rpt_score_window(close, spec["start"], spec["len"], dirn, today_ts)
            except Exception as e:
                log.debug("Locked window re-score failed for %s/%s: %s", market_name, dir_key, e)
            if scored is None:
                try:
                    s0 = pd.Timestamp(spec["start"])
                    L = int(spec["len"])
                    scored = {
                        "dir": dirn, "start": s0, "end": pd.Timestamp(spec["end"]), "len": L,
                        "shape": _season_shape_label(L),
                        "win": float(spec.get("win", np.nan)),
                        "cases": int(spec.get("cases", 0)),
                        "wins": int(spec.get("wins", 0)),
                        "avg": float(spec.get("avg", np.nan)),
                        "rec_wins": int(spec.get("rec_wins", 0)),
                        "rec_cases": int(spec.get("rec_cases", 0)),
                        "strength": spec.get("strength", "MODERATE"),
                        "neighbour": np.nan,
                        "timing": "moderate",
                        "years": int(spec.get("years", REPORT_SEASON_YEARS)),
                        "persistent": True,
                        "degraded": True,
                    }
                except Exception:
                    continue
            best[dirn] = scored
        if best:
            pack = _rpt_season_pack_raw_memo(close, today_ts)
            if pack is None:
                return None
            pack["best"] = best
            DISPLAY_AHEAD_DAYS = 35
            near = {d_: b for d_, b in best.items() if (b["start"] - today_ts).days <= DISPLAY_AHEAD_DAYS}
            pool = near if near else best
            def _main_key(b):
                days_out = (b["start"] - today_ts).days
                w = b.get("win", 0)
                if w is None or not np.isfinite(w):
                    w = 0
                return (b["strength"] == "STRONG", w, abs(b.get("avg", 0) or 0), -days_out)
            pack["main"] = sorted(pool.values(), key=_main_key, reverse=True)[0]
            pack["locked"] = True
            # same opposing-window cancel as the unlocked path
            cancelled = False
            m0 = pack["main"]
            if m0 and len(best) >= 2:
                opp = best.get(-int(m0["dir"]))
                if opp and opp.get("strength") in ("STRONG", "MODERATE"):
                    close_records = abs(float(m0.get("win") or 0) - float(opp.get("win") or 0)) <= 0.10
                    both_strong = m0.get("strength") == "STRONG" and opp.get("strength") == "STRONG"
                    if close_records or both_strong:
                        cancelled = True
            pack["cancelled"] = cancelled
            return pack

    pack = _rpt_season_pack_raw_memo(close, today_ts)
    if pack is None or not pack.get("best"):
        return pack
    locked = {}
    for dirn, b in pack["best"].items():
        if b.get("persistent"):
            locked[str(dirn)] = {
                "start": b["start"].date().isoformat(),
                "end": b["end"].date().isoformat(),
                "len": int(b["len"]),
                "dir": int(dirn),
                "strength": b.get("strength"),
                "win": float(b["win"]) if b.get("win") is not None and np.isfinite(b["win"]) else None,
                "cases": int(b.get("cases", 0)),
                "wins": int(b.get("wins", 0)),
                "avg": float(b["avg"]) if b.get("avg") is not None and np.isfinite(b["avg"]) else None,
                "rec_wins": int(b.get("rec_wins", 0)),
                "rec_cases": int(b.get("rec_cases", 0)),
                "years": int(b.get("years", REPORT_SEASON_YEARS)),
                "locked_at": today_ts.date().isoformat(),
            }
    if locked:
        lock_data["locked"][market_name] = locked
        _save_season_windows(lock_data)
        log.info("Seasonal window locked for %s: %s", market_name, list(locked.keys()))
    else:
        log.debug("No persistent seasonal window for %s — skipping lock", market_name)
    return pack


_SEASON_PACK_MEMO = {}


def _rpt_season_pack_raw_memo(close, today=None):
    try:
        key = (len(close), close.index[-1], float(close.dropna().iloc[-1]),
               None if today is None else str(pd.Timestamp(today).normalize().date()))
    except Exception:
        return _rpt_season_pack_raw(close, today)
    if key not in _SEASON_PACK_MEMO:
        _SEASON_PACK_MEMO[key] = _rpt_season_pack_raw(close, today)
    return _SEASON_PACK_MEMO[key]


def _rpt_season_pack(close, today=None, market_name=None):
    if market_name:
        return _rpt_season_pack_locked(close, market_name, today)
    return _rpt_season_pack_raw_memo(close, today)


def _rpt_fig_to_png(fig, dpi=170):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return buf


_CHART_BG = "#0d1117"
_CHART_FG = "#e6edf3"
_CHART_MUTED = "#8b949e"
_CHART_GRID = "#1c2128"
_CHART_BLUE = "#58a6ff"
_CHART_GREEN = "#2ea043"
_CHART_RED = "#f85149"
_CHART_AMBER = "#d29922"


def _rpt_style_ax(ax):
    ax.set_facecolor(_CHART_BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#21262d")
    ax.tick_params(colors=_CHART_MUTED, labelsize=7)
    ax.grid(True, color=_CHART_GRID, lw=0.6)


def _rpt_cot_chart(state, role, years=5):
    import matplotlib.pyplot as plt
    basis = state["basis"]
    end = basis.index[-1]
    start = end - pd.DateOffset(years=years)
    fig, ax = plt.subplots(figsize=(11.2, 2.6), facecolor=_CHART_BG)
    _rpt_style_ax(ax)
    zone_col = _CHART_RED if role == "retail" else _CHART_GREEN
    side = state["side"] if state["side"] else (-1 if role == "retail" else 1)
    lit = 0
    for key, _ in reversed(RPT_TABLE_ORDER):
        L = state["layers"][key]; w = L["w"]
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
    ax.plot(basis.index[m], basis[m], color=_CHART_MUTED if role == "retail" else _CHART_BLUE, lw=1.4)
    ax.plot([basis.index[-1]], [basis.iloc[-1]], marker="o", ms=4, color=_CHART_MUTED if role == "retail" else _CHART_BLUE)
    ax.set_xlim(start, end + pd.Timedelta(days=60))
    ax.set_ylabel("Masses" if role == "retail" else "Hedgers", fontsize=8, fontweight="bold",
                  color=_CHART_MUTED if role == "retail" else _CHART_BLUE)
    _chip_h = 0.055
    _chip_w = 0.012
    for i in range(5):
        ax.add_patch(plt.Rectangle((0.0 + i * 0.017, 1.015), _chip_w, _chip_h, transform=ax.transAxes,
                                   facecolor=zone_col if i < lit else _CHART_BG, edgecolor=zone_col, lw=0.6,
                                   clip_on=False))
    ax.text(0.095, 1.045, f"Layered extreme: {lit} of 5", transform=ax.transAxes, fontsize=7,
            color=_CHART_MUTED, va="center")
    return _rpt_fig_to_png(fig)


def _px_dec(v):
    v = abs(float(v))
    return 0 if v >= 10000 else 1 if v >= 1000 else 2 if v >= 100 else 3 if v >= 10 else 4 if v >= 1 else 5


def _rpt_draw_zones(ax, df, zones, ylo, yhi, dec, tf="D"):
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgba
    GOLD = "#ffd700"
    xlo, xhi = ax.get_xlim()
    n = len(df)
    if n < 2:
        return
    x_nums = mdates.date2num(df.index.to_pydatetime())
    avg_gap = float(np.median(np.diff(x_nums)))
    days_per_bar = float((x_nums[-1] - x_nums[0]) / max(1, n - 1))
    right_x = x_nums[-1] + days_per_bar * ZONE_EXTEND_BARS
    if tf == "M":
        half_w = max(avg_gap * 0.35, 6.0); label_offset = max(avg_gap * 0.45, 8.0); label_fs = 6.0; extend_bars = 8
    elif tf == "W":
        half_w = max(avg_gap * 0.32, 1.8); label_offset = max(avg_gap * 0.40, 3.0); label_fs = 6.0; extend_bars = 8
    else:
        half_w = min(0.45, avg_gap * 0.40); label_offset = 3.0; label_fs = 6.5; extend_bars = 60
    def _bar_x(i):
        i = int(max(0, min(i, n - 1)))
        return float(x_nums[i])
    visible_i0 = 0
    try:
        visible_i0 = int(np.searchsorted(x_nums, xlo))
    except Exception:
        pass
    filtered = []
    for z in zones:
        born = int(z.get("born", 0))
        if z.get("tested") and born < max(0, visible_i0 - extend_bars):
            continue
        filtered.append(z)
    filtered = sorted(filtered, key=lambda z: int(z.get("born", 0)))[-6:]
    used_ys = []
    for z in filtered:
        top, bot = float(z["top"]), float(z["bottom"])
        if top < ylo or bot > yhi:
            continue
        left_i = int(z.get("left", 0)); right_i = int(z.get("right", left_i))
        born_i = int(z.get("born", right_i))
        left_i = max(0, min(left_i, n - 1)); right_i = max(left_i, min(right_i, n - 1))
        born_i = max(0, min(born_i, n - 1))
        xl = _bar_x(left_i) - half_w; xr = _bar_x(right_i) + half_w
        if xr - xl < half_w * 1.5:
            mid = (_bar_x(left_i) + _bar_x(right_i)) / 2.0
            xl, xr = mid - half_w, mid + half_w
        if z.get("tested"):
            face, edge = to_rgba("#1a1a1a", 0.28), to_rgba(GOLD, 0.50); proximal = to_rgba(GOLD, 0.40)
        else:
            face = to_rgba("#00c9a7" if z["demand"] else "#e63946", 0.18)
            edge = to_rgba(GOLD, 0.75); proximal = to_rgba(GOLD, 0.65)
        ax.add_patch(Rectangle((xl, bot), right_x - xl, top - bot,
                               facecolor=face, edgecolor=edge, lw=0.9, ls="--", zorder=1))
        ax.add_patch(Rectangle((xl, bot), xr - xl, top - bot,
                               facecolor=proximal, edgecolor=to_rgba(GOLD, 0.90), lw=1.15, ls="-", zorder=2))
        my = (top + bot) / 2.0
        for uy in used_ys:
            if abs(my - uy) < (yhi - ylo) * 0.045:
                my = uy + (yhi - ylo) * 0.05
        used_ys.append(my)
        label_x = min(xr + label_offset, right_x - avg_gap)
        if xlo <= label_x <= xhi and ylo <= my <= yhi:
            tag = "TESTED" if z.get("tested") else "FRESH"
            ax.text(label_x, my, f"{z.get('setup', 'Reference')} · {tag}",
                    color=GOLD, alpha=0.92, fontsize=label_fs, va="center", ha="left",
                    zorder=4, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#000000", ec="none", alpha=0.55))


def _sd_nearest_fresh(zones, last, bias):
    if bias == 0:
        return None
    best = None
    for z in zones:
        if (not zone_is_fresh(z)) or z["demand"] != (bias == 1):
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
    end = w.index[-1]; start = w.index[0]
    BG, UP, DN = _CHART_BG, _CHART_GREEN, _CHART_RED
    x = mdates.date2num(w.index.to_pydatetime())
    o, h, l, c = (w[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    rng = float(h.max() - l.min())
    if rng <= 0:
        rng = max(abs(float(c[-1])) * 0.01, 1e-9)
    up = c >= o
    if len(x) > 1:
        avg_gap = float(np.median(np.diff(x)))
    else:
        avg_gap = 1.0 if tf == "D" else (7.0 if tf == "W" else 30.0)
    if tf == "M":
        body_w = avg_gap * 0.58; wick_lw = 1.5; min_body = rng * 0.004
    elif tf == "W":
        body_w = avg_gap * 0.62; wick_lw = 1.15; min_body = rng * 0.0025
    else:
        body_w = min(0.70, avg_gap * 0.72); wick_lw = 0.90; min_body = rng * 0.0012
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
    zones = compute_sd_zones(df)
    _rpt_draw_zones(ax, df, zones, float(l.min()) - pad, float(h.max()) + pad, dec, tf=tf)
    ax.axhline(last, color="#6e7681", lw=0.7, ls=":", zorder=3)
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
    ax.tick_params(colors=_CHART_MUTED, labelsize=7.5, length=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.text(0.005, 0.985, f"{label}  ·  {tf}  ·  {_rpt_tvlabel(item).split(' · ')[0]}", transform=ax.transAxes,
            color=_CHART_FG, fontsize=9, fontweight="bold", va="top")
    return _rpt_fig_to_png(fig)


def _rpt_strip_chart(sp):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    today = sp["today"]
    x0, x1 = today - pd.Timedelta(days=14), today + pd.Timedelta(days=56)
    fig, ax = plt.subplots(figsize=(11.2, 1.6), facecolor=_CHART_BG)
    ax.set_facecolor(_CHART_BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#21262d")
    ax.set_xlim(x0, x1); ax.set_ylim(0, 3.3); ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors=_CHART_MUTED, labelsize=8)
    row = 1.75
    for dirn, b_ in sorted(sp["best"].items(), key=lambda kv: kv[1]["start"]):
        col = _CHART_GREEN if dirn == 1 else _CHART_RED
        tint = "#0f2a1a" if dirn == 1 else "#2a0f0f"
        xs = max(mdates.date2num(b_["start"]), mdates.date2num(x0))
        xe = min(mdates.date2num(b_["end"]), mdates.date2num(x1))
        ax.add_patch(plt.Rectangle((xs, row), xe - xs, 0.55, fc=tint, ec=col, lw=1.4))
        ax.text(xs, row + 0.66, f"{'LONG' if dirn == 1 else 'SHORT'} {b_['win'] * 100:.0f}%   "
                f"{b_['start'].strftime('%d %b')} – {b_['end'].strftime('%d %b')}  ·  {b_['shape']}  ·  recent {b_['rec_wins']}/{b_['rec_cases']}",
                fontsize=7.5, color=col, va="bottom", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", fc=_CHART_BG, ec="none", alpha=0.92))
        row = 0.45
    ax.axvline(mdates.date2num(today), color=_CHART_FG, lw=2, zorder=3)
    ax.text(mdates.date2num(today) + 0.6, 3.15, "now", ha="left", va="top", fontsize=8, fontweight="bold", color=_CHART_FG)
    return _rpt_fig_to_png(fig)


def _rpt_map_chart(sp):
    import matplotlib.pyplot as plt
    d, today = sp["d"], sp["today"]
    yr = int(today.year)
    def path(y):
        seg = d[(d.index >= pd.Timestamp(year=y, month=1, day=1)) & (d.index <= pd.Timestamp(year=y, month=12, day=31))]
        return None if len(seg) < 300 else (seg.values[:365] / seg.values[0] - 1.0) * 100.0
    avgs = {}
    for n, col, lab in ((15, _CHART_MUTED, "15-year average"), (10, _CHART_BLUE, "10-year average"), (5, _CHART_GREEN, "5-year average")):
        ps = [p for p in (path(yr - k) for k in range(1, n + 1)) if p is not None]
        if ps:
            m = min(len(p) for p in ps)
            avgs[lab] = (np.mean([p[:m] for p in ps], axis=0), col)
    cur = d[(d.index >= pd.Timestamp(year=yr, month=1, day=1)) & (d.index <= today)]
    fig, ax = plt.subplots(figsize=(11.2, 2.2), facecolor=_CHART_BG)
    _rpt_style_ax(ax)
    for lab, (arr, col) in avgs.items():
        ax.plot(range(len(arr)), arr, color=col, lw=1.6, label=lab)
    if len(cur) > 1:
        ax.plot(range(len(cur)), (cur.values / cur.values[0] - 1.0) * 100.0, color=_CHART_FG, lw=1.8, label=f"{yr} so far")
    ax.axhline(0, color="#6e7681", lw=0.8)
    ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("% from year start", fontsize=8, color=_CHART_MUTED)
    leg = ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.13))
    for t in leg.get_texts():
        t.set_color(_CHART_MUTED)
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
    fig, ax = plt.subplots(figsize=(11.2, 2.6), facecolor=_CHART_BG)
    _rpt_style_ax(ax)
    ax.plot(range(len(typical)), typical, color=_CHART_MUTED, lw=2, label="typical (15-yr)")
    ax.fill_between(range(len(typical)), 0, typical, color=_CHART_MUTED, alpha=0.08)
    if len(cur):
        pos = [(i - pd.Timestamp(year=yr, month=1, day=1)).days for i in cur.index]
        ax.plot(pos, cur.values, color=_CHART_FG, lw=1.8, label=f"{yr} so far")
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("annualized vol %", fontsize=8, color=_CHART_MUTED)
    leg = ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(_CHART_MUTED)
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
    matches.sort(key=lambda t: (-t[0], -t[1]))
    top = matches[:4]
    fig, ax = plt.subplots(figsize=(11.2, 2.6), facecolor=_CHART_BG)
    _rpt_style_ax(ax)
    ax.plot(range(len(cur_path)), cur_path, color=_CHART_FG, lw=2.2, label=f"{yr} (this year)")
    cols = [_CHART_BLUE, _CHART_GREEN, _CHART_AMBER, "#9b59b6"]
    for (corr, y, full, cut, rest), col in zip(top, cols):
        ax.plot(range(len(full)), full, color=col, lw=1.3, alpha=0.85, label=f"{y} ({corr:+.2f})")
    if top:
        ax.axvline(len(cur_path), color="#6e7681", lw=1, ls="--")
        ax.text(len(cur_path) + 1, ax.get_ylim()[1] * 0.9, "today", fontsize=7.5, color=_CHART_MUTED)
    ax.axhline(0, color="#6e7681", lw=0.6)
    ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("% from year start", fontsize=8, color=_CHART_MUTED)
    leg = ax.legend(loc="upper center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.14))
    for t in leg.get_texts():
        t.set_color(_CHART_MUTED)
    return _rpt_fig_to_png(fig), top


def _rpt_styles():
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    H = colors.HexColor
    S_ = {}
    S_["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=9.6, leading=14.2, textColor=H(RPT["ink"]), spaceAfter=6)
    S_["small"] = ParagraphStyle("small", fontName="Helvetica", fontSize=7.4, leading=10, textColor=H(RPT["muted"]))
    S_["note"] = ParagraphStyle("note", fontName="Helvetica", fontSize=8, leading=11.5, textColor=H(RPT["muted"]))
    S_["label"] = ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=8.2, leading=11, textColor=H(RPT["blue"]),
                                 spaceBefore=0, spaceAfter=0)
    S_["kicker"] = ParagraphStyle("kicker", fontName="Helvetica-Bold", fontSize=8.0, leading=10.5,
                                  textColor=H(RPT["blue"]), spaceBefore=0, spaceAfter=1)
    S_["title"] = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26, leading=30,
                                 textColor=H(RPT["ink"]), spaceBefore=0, spaceAfter=0)
    S_["right"] = ParagraphStyle("right", fontName="Helvetica", fontSize=8.4, leading=12, textColor=H(RPT["muted"]), alignment=2)
    S_["rightb"] = ParagraphStyle("rightb", fontName="Helvetica-Bold", fontSize=8.8, leading=12, textColor=H(RPT["ink"]), alignment=2)
    S_["h"] = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=13.5, leading=24, textColor=H(RPT["ink"]), spaceAfter=6)
    S_["cell"] = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, leading=11, textColor=H(RPT["ink"]))
    S_["big"] = ParagraphStyle("big", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=H(RPT["ink"]))
    return S_


def _peer_snapshot(pk):
    cp = pk.get("cot") if pk else None
    if not cp:
        return None
    comm, ret = cp["comm"], cp["ret"]
    bias = cp["bias"]
    lit = 0
    if bias != 0:
        if comm["pdir"] == bias:
            lit += comm["lit"]
        if ret["pdir"] == bias:
            lit += ret["lit"]
    conflict = bool(cp.get("conflict"))
    conviction = "high" if (lit >= 7 and not conflict) else "moderate" if lit >= 4 else "low"
    qual = weak = 0
    h0 = COT_EDGE_HORIZONS[0]
    tracks = cp.get("tracks") or {}
    for role_k in ("commercial", "retail"):
        for key, _lab in RPT_TABLE_ORDER:
            tr = tracks.get((role_k, key, h0))
            if not tr or tr.get("past"):
                continue
            cases = tr.get("cases", 0) or 0
            win = tr.get("win")
            avg = tr.get("avg")
            if (cases >= REPORT_EDGE_MIN_CASES and win is not None and np.isfinite(win)
                    and win >= REPORT_EDGE_MIN_WIN
                    and avg is not None and np.isfinite(avg) and avg > 0):
                qual += 1
            elif cases >= 2 and avg is not None and np.isfinite(avg) and avg <= 0:
                weak += 1
    return {"bias": bias, "lit": lit, "conviction": conviction,
            "qualifying": qual, "weak": weak, "conflict": conflict}


def _peer_evidence_tone(s):
    q, w = s["qualifying"], s["weak"]
    if q >= 2 and q > w:
        return "positive"
    if w >= 2 and w > q:
        return "negative"
    if q == 0 and w == 0:
        return "empty"
    return "mixed"


def _describe_side(name, s):
    bits = []
    if s["lit"] > 0:
        bits.append(f"{_nw(s['lit'])} lit tier{'s' if s['lit'] != 1 else ''} at {s['conviction']} conviction")
    else:
        bits.append("no lit tiers on either group")
    if s["qualifying"] > 0 and s["weak"] > 0:
        bits.append(f"{_nw(s['qualifying'])} row{'s' if s['qualifying'] != 1 else ''} clearing the bar "
                    f"against {_nw(s['weak'])} losing at both horizons")
    elif s["qualifying"] > 0:
        bits.append(f"{_nw(s['qualifying'])} row{'s' if s['qualifying'] != 1 else ''} clearing the bar")
    elif s["weak"] > 0:
        bits.append("its only real samples losing at both horizons")
    else:
        bits.append("no qualifying rows yet")
    return f"<b>{_esc(name)}</b> has " + " and ".join(bits)


def _peer_relation_signature(a, b):
    if a is None or b is None:
        return None
    if a["bias"] == b["bias"] and a["bias"] != 0:
        badge = "same"
    elif a["bias"] == -b["bias"] and a["bias"] != 0:
        badge = "opposite"
    elif a["bias"] == 0 or b["bias"] == 0:
        badge = "unread"
    else:
        badge = "neither"
    ta, tb = _peer_evidence_tone(a), _peer_evidence_tone(b)
    if ta == tb:
        evidence = "same"
    elif "empty" in (ta, tb):
        evidence = "empty"
    elif (ta, tb) in (("positive", "negative"), ("negative", "positive")):
        evidence = "opposite"
    else:
        evidence = "mixed"
    return (badge, evidence)


def _peer_relation_state(nm, pk, packs, prior_state=None):
    peers = PEER_MAP.get(nm)
    if not peers:
        return None
    a = _peer_snapshot(pk)
    if a is None:
        return None
    for pname in peers:
        ppk = packs.get(pname)
        if ppk is None:
            continue
        b = _peer_snapshot(ppk)
        if b is None:
            continue
        sig = _peer_relation_signature(a, b)
        if sig is None:
            continue
        badge, evidence = sig
        prior_rel = ((prior_state or {}).get("markets") or {}).get(nm, {}).get("peer_relation")
        if (prior_rel
                and prior_rel.get("peer") == pname
                and prior_rel.get("badge") == badge
                and prior_rel.get("evidence") == evidence):
            streak = int(prior_rel.get("streak") or 1) + 1
        else:
            streak = 1
        return {"peer": pname, "badge": badge, "evidence": evidence, "streak": streak}
    return None


def _compute_peer_states(packs, prior_state=None):
    out = {}
    for nm, pk in packs.items():
        try:
            rel = _peer_relation_state(nm, pk, packs, prior_state=prior_state)
        except Exception as e:
            log.debug("Peer state failed for %s: %s", nm, e)
            rel = None
        if rel:
            out[nm] = rel
    return out


def _peer_comparison_paragraphs(nm, pk, peer_nm, peer_pk, prior_state=None):
    a = _peer_snapshot(pk)
    b = _peer_snapshot(peer_pk)
    if a is None or b is None:
        return []
    if a["qualifying"] == 0 and a["weak"] == 0 and b["qualifying"] == 0 and b["weak"] == 0:
        return []
    full = _rpt_market_display(nm)[1]
    pfull = _rpt_market_display(peer_nm)[1]
    a_txt = _describe_side(full, a)
    b_txt = _describe_side(pfull, b)
    sig = _peer_relation_signature(a, b)
    if sig is None:
        return []
    badge, evidence = sig
    badge_phrase = {
        "same": "Same badge",
        "opposite": "Opposite badges",
        "unread": "One side unread",
        "neither": "Neither side carrying a badge",
    }.get(badge, "Same badge")
    ta, tb = _peer_evidence_tone(a), _peer_evidence_tone(b)
    if evidence == "same":
        evidence_phrase = f"same evidence ({ta})"
    elif evidence == "empty":
        evidence_phrase = "one side has nothing to say"
    elif evidence == "opposite":
        evidence_phrase = "opposite evidence"
    else:
        evidence_phrase = "mixed evidence on both sides"
    streak_clause = ""
    try:
        prior_rel = ((prior_state or {}).get("markets") or {}).get(nm, {}).get("peer_relation")
        if (prior_rel
                and prior_rel.get("peer") == peer_nm
                and prior_rel.get("badge") == badge
                and prior_rel.get("evidence") == evidence):
            cur_streak = int(prior_rel.get("streak") or 1) + 1
            if cur_streak >= 2:
                streak_clause = f" This is the {_nw(cur_streak)} week running the contrast has held."
    except Exception:
        pass
    para = (f"<b>Read this against the {_esc(pfull)} page.</b> {a_txt}, "
            f"while {b_txt}. {badge_phrase}, {evidence_phrase}.{streak_clause}")
    return [para]


def _rpt_pdate(ts):
    """'2 October' style date (no leading zero)."""
    try:
        return pd.Timestamp(ts).strftime("%d %B").lstrip("0")
    except Exception:
        return "-"


def _rpt_art(pct):
    """'an' before 8, 11, 18 (eighty, eleven, eighteen), else 'a'."""
    try:
        n = int(round(float(pct)))
    except Exception:
        return "a"
    return "an" if (n // 10 == 8 or n in (8, 11, 18)) else "a"


def _rpt_lit_layers(st):
    """How many of the five lookbacks are at an extreme for one group (either direction)."""
    try:
        return sum(1 for L in st["layers"].values() if L.get("now"))
    except Exception:
        return 0


def _rpt_standaside_text(full, cp):
    """Step 1 rationale when positioning gives no side. Names the two possible reasons
    (no track record / evidence points both ways) and says which one applies when the data can tell."""
    comm, ret = cp["comm"], cp["ret"]
    n_m, n_h = _rpt_lit_layers(ret), _rpt_lit_layers(comm)
    if not (n_m or n_h):
        return (f"Neither group is at an extreme on any of the five lookbacks, so positioning has nothing to put on trial "
                f"in {full} this week. <b>No side from positioning.</b>")
    if n_m and n_h:
        s1 = (f"Positioning is stretched, the masses on <b>{_nw(n_m)} of five</b> lookbacks and the hedgers on "
              f"<b>{_nw(n_h)} of five</b>, but it does <b>not</b> turn into a read.")
    elif n_m:
        s1 = (f"Positioning is stretched, the masses on <b>{_nw(n_m)} of five</b> lookbacks, "
              f"but it does <b>not</b> turn into a read.")
    else:
        s1 = (f"Positioning is stretched, the hedgers on <b>{_nw(n_h)} of five</b> lookbacks, "
              f"but it does <b>not</b> turn into a read.")
    s2 = (f"Either the extreme has no track record in {full}, or what record there is points both ways.")
    h0, h1 = COT_EDGE_HORIZONS[0], COT_EDGE_HORIZONS[-1]
    rolename_q = {"commercial": "hedgers'", "retail": "masses'"}
    tracks = cp.get("tracks") or {}
    rows4 = [(k, t) for k, t in tracks.items() if t and k[2] == h0]
    diag = ""
    _q = cp.get("edge") or cp.get("edge8")
    if _q:
        _qt = _q["t"]
        diag = (f" Here it is neither: the {rolename_q[_q['role']]} record clears the bar ({_qt['win']:.0f}% over {_qt['cases']} cases), "
                f"but only the hedgers can carry a side in this book, so it stays context rather than a call.")
    elif not rows4 or all(t["cases"] == 0 for _, t in rows4):
        diag = " Here it is the first: there is no comparable past case to lean on."
    else:
        (role_k, key_k, _h), deep = max(rows4, key=lambda kt: (kt[1]["cases"], kt[1]["win"] if np.isfinite(kt[1]["win"]) else 0))
        t8 = tracks.get((role_k, key_k, h1))
        flips = bool(t8 and t8["cases"] and np.isfinite(deep["avg"]) and np.isfinite(t8["avg"])
                     and deep["avg"] * t8["avg"] < 0)
        if flips:
            diag = (f" Here it is the second: the deepest sample, {deep['wins']} of {deep['cases']} cases, averages "
                    f"{deep['avg']:+.2f}% at {h0} weeks and {t8['avg']:+.2f}% at {h1}, so the horizons disagree.")
        elif deep["cases"] < REPORT_EDGE_MIN_CASES:
            diag = (f" Here the record is too thin to lean on: the deepest sample is {deep['wins']} of {deep['cases']} cases, "
                    f"against a floor of {_nw(REPORT_EDGE_MIN_CASES)}.")
        else:
            diag = (f" Here the record exists but has not paid: the deepest sample is {deep['win']:.0f}% over "
                    f"{deep['cases']} cases at {deep['avg']:+.2f}% average.")
    s3 = "A stretched chart is a description, not an edge."
    s4 = "<b>No side from positioning.</b>"
    return f"{s1} {s2}{diag} {s3} {s4}"


def _rpt_combined_essay(pk, cp, sp, r, bias, nm, packs, prior_state=None):
    """The long-form argument on the verdict page: a narrator who weighs the case for the side,
    the case against it, and what would change the call."""
    paras = []
    full = _rpt_market_display(nm)[1]
    rolename = {"commercial": "hedgers", "retail": "the masses"}
    labels = dict(RPT_TABLE_ORDER)
    h0, h1 = COT_EDGE_HORIZONS[0], COT_EDGE_HORIZONS[-1]
    fb = int(cp["bias"]) if cp else 0
    mood = {1: "bullish", -1: "bearish"}
    act = {1: "buy", -1: "sell"}
    m = sp["main"] if (sp and sp.get("main")) else None
    tier = (sp or {}).get("tier", "FULL")
    cal_dir = int(m["dir"]) if m else 0
    days_left, live_now = 0, False
    if m:
        try:
            days_left = max(0, (pd.Timestamp(m["end"]) - pd.Timestamp(sp["today"])).days)
            live_now = pd.Timestamp(sp["today"]) >= pd.Timestamp(m["start"])
        except Exception:
            pass

    def _dword(n):
        return f"{_nw(n)} day{'s' if n != 1 else ''}"

    def _win_txt(b):
        return (f"{b['win'] * 100:.0f}% of the last {b['cases']} years, {b['rec_wins']} of its last {b['rec_cases']}, "
                f"averaging {b['avg']:+.1f}%")

    edge = cp.get("edge") if cp else None
    edge8 = cp.get("edge8") if cp else None
    tracks = (cp.get("tracks") if cp else None) or {}
    pos_rows, weak_rows = [], []
    if cp:
        for role_k, st in (("commercial", cp["comm"]), ("retail", cp["ret"])):
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

    # ---------------- positioning ----------------
    if cp is None:
        paras.append("<b>Positioning is missing this week.</b> Without it the coordinated bias can rest on the calendar alone, "
                     "and only if the calendar is strong enough to stand alone.")
    elif fb == 0:
        paras.append("<b>Start with what positioning does not say.</b> " + _rpt_standaside_text(full, cp))
        if edge or edge8:
            paras.append(
                "<b>The masses are the tempting witness.</b> Their fade has a record that clears the bar, which is exactly why it "
                "needs handling with care: in this book only the hedgers carry a side, so the masses inform the picture without "
                "being allowed to decide it.")
        elif _rpt_lit_layers(cp["comm"]) or _rpt_lit_layers(cp["ret"]):
            paras.append(
                "<b>This is the trap a stretched chart sets.</b> A crowd at an extreme feels like a trade, and the pull is to take "
                "the other side of it on instinct. But the rows on the evidence page are either empty or in disagreement, and "
                "neither earns the right to size up. Acting on this would mean acting on a feeling.")
    else:
        if edge:
            lab = labels.get(edge["key"], edge["key"]); t4 = edge["t"]
            lead = (f"the {lab.lower()} lookback on {rolename[edge['role']]} has paid <b>{t4['win']:.0f}%</b> of the time over "
                    f"{h0} weeks, {t4['wins']} of {t4['cases']} cases, {t4['avg']:+.2f}% on average")
        else:
            lead = (f"positioning is stretched, but no row reaches the floor of {_nw(REPORT_EDGE_MIN_CASES)} cases, so the "
                    f"record is thin rather than proven")
        if m and cal_dir == fb:
            paras.append(
                f"<b>Both engines speak, and they agree.</b> Flow reads {mood[fb]}: {lead}. The calendar adds a {m['shape']} "
                f"{act[cal_dir]} from {_rpt_pdate(m['start'])} to {_rpt_pdate(m['end'])}, {_win_txt(m)}. "
                f"Agreement buys timing, not a new verdict; flow still carries the side.")
        elif m and cal_dir == -fb:
            paras.append(
                f"<b>The engines disagree, and the disagreement has a date.</b> Flow reads {mood[fb]}: {lead}. The calendar reads "
                f"{mood[cal_dir]} on a {m['shape']} {act[cal_dir]} window from {_rpt_pdate(m['start'])} to {_rpt_pdate(m['end'])}, "
                f"{_win_txt(m)}, with {_dword(days_left)} {'left to run' if live_now else 'until it closes'}. "
                f"Flow outranks it, because positioning drives and the calendar only adds, but the objection is real.")
        else:
            paras.append(
                f"<b>Flow is the whole page.</b> Flow reads {mood[fb]}: {lead}. No calendar window is in play to confirm or "
                f"contradict it, so this is a single-engine side, and it should be sized as one.")
        if edge8:
            lab8 = labels.get(edge8["key"], edge8["key"]); t8_ = edge8["t"]
            same = bool(edge) and (edge8["role"], edge8["key"]) == (edge["role"], edge["key"])
            if same:
                paras.append(
                    f"<b>The longer horizon does not weaken the case.</b> At {h1} weeks the same {lab8.lower()} lookback holds: "
                    f"{t8_['win']:.0f}% over {t8_['cases']} cases, {t8_['avg']:+.2f}% average.")
            else:
                paras.append(
                    f"<b>The longer horizon changes the witness.</b> At {h1} weeks the strongest read moves to the {lab8.lower()} "
                    f"lookback on {rolename[edge8['role']]}: {t8_['win']:.0f}% over {t8_['cases']} cases, {t8_['avg']:+.2f}% average.")
        if len(pos_rows) > 1:
            desc = "; ".join(f"{lab} on {rolename[rk]} {tr['win']:.0f}% of {tr['cases']} ({tr['avg']:+.2f}%)"
                             for rk, lab, tr in pos_rows[:5])
            paras.append(f"<b>It is not standing alone.</b> {_nw(len(pos_rows)).capitalize()} rows clear the bar at {h0} weeks: {desc}.")
        if weak_rows:
            w = weak_rows[0]
            paras.append(
                f"<b>Now the part that argues back.</b> The {w[1]} lookback on {rolename[w[0]]} is the weak row: "
                f"{w[2]['win']:.0f}% of {w[2]['cases']} with {w[2]['avg']:+.2f}% at {h0} weeks. A side that ignores its own weakest "
                f"row is a hope, not a read, so it stays in the argument.")

    # peer comparison (unchanged)
    try:
        for pname in PEER_MAP.get(nm, []):
            ppk = packs.get(pname)
            if ppk is None:
                continue
            peer_paras = _peer_comparison_paragraphs(nm, pk, pname, ppk, prior_state=prior_state)
            if peer_paras:
                paras.extend(peer_paras)
                break
    except Exception as e:
        log.debug("Peer comparison failed for %s: %s", nm, e)

    # ---------------- calendar ----------------
    cal_promoted = bool(m and fb == 0 and m["strength"] == "STRONG" and tier != "DROP" and bias == cal_dir)
    if m and fb == 0:
        if cal_promoted:
            paras.append(
                f"<b>One engine is silent and the other is the whole page.</b> Positioning offers nothing, and the calendar reads "
                f"{mood[cal_dir]} at {m['strength'].lower()}: a {m['shape']} {act[cal_dir]} from {_rpt_pdate(m['start'])} to "
                f"{_rpt_pdate(m['end'])}, {_win_txt(m)}. "
                + (f"{_dword(days_left).capitalize()} left to run." if live_now else f"It opens on {_rpt_pdate(m['start'])}."))
        else:
            paras.append(
                f"<b>The calendar has a window, but it does not get to lead.</b> A {m['shape']} {act[cal_dir]} from "
                f"{_rpt_pdate(m['start'])} to {_rpt_pdate(m['end'])}, {_win_txt(m)}, graded {m['strength'].lower()}"
                + (", and this market is off-pattern this year" if tier != "FULL" else "")
                + ". In this book a calendar window only leads when it is strong. As background it counts; as a trade it does not.")
    if m and fb and cal_dir == -fb:
        paras.append(
            f"<b>Now the caution that matters more than the headline, and it has a date.</b> The {act[cal_dir]} window "
            f"{'runs to' if live_now else 'opens on ' + _rpt_pdate(m['start']) + ' and runs to'} {_rpt_pdate(m['end'])}. "
            f"It is not a veto. But if price reaches a zone inside it, price arrives against the grain, so patience on entry "
            f"is the price of respecting both engines.")

    # timing / shape qualifications
    if m and (cal_dir == bias or cal_promoted or (fb and cal_dir == fb)):
        tm = m.get("timing", "moderate"); nb = m.get("neighbour", np.nan)
        nb_txt = f" (the surrounding weeks score {nb * 100:.0f}% against {m['win'] * 100:.0f}% on the exact dates)" if np.isfinite(nb) else ""
        if tm == "very tight":
            tq = (f"The timing is graded very tight, the weakest grade the engine issues: shift the dates and the tendency "
                  f"fades{nb_txt}. Be on time or leave it.")
        elif tm == "tight":
            tq = f"The timing is graded tight: most of the move sits inside these dates{nb_txt}, so being late costs real edge."
        elif tm == "forgiving":
            tq = f"The timing is graded forgiving: the surrounding weeks lean the same way{nb_txt}, so the edge is broader than two dates."
        else:
            tq = "The timing could not be graded cleanly, so treat the exact dates with caution."
        sh = sp.get("shape", np.nan) if sp else np.nan
        if np.isfinite(sh):
            if sh >= SHAPE_CLOSE:
                sq = f"The whole-year shape match is {sh:+.2f}, tracking closely: this year is following the market's seasonal script."
            elif sh >= SHAPE_LOOSE:
                sq = f"The whole-year shape match is {sh:+.2f}, loosely tracking: encouraging context, not proof."
            else:
                sq = (f"The whole-year shape match is only {sh:+.2f}, off-pattern: price is not following its seasonal rhythm this "
                      f"year, so the calendar deserves a discount.")
        else:
            sq = "The whole-year shape match is not available."
        paras.append(f"<b>Two qualifications on the window itself.</b> {tq} {sq}")

    # funds
    if cp and bias:
        fnd = cp.get("funds") if isinstance(cp, dict) else None
        st_ = str((fnd or {}).get("state", "n/a") or "n/a"); f_dir = int((fnd or {}).get("dir") or 0)
        if st_ in ("Bearish divergence", "Bullish divergence") and f_dir:
            if f_dir == bias:
                paras.append("<b>Funds lean the same way.</b> The large speculators are showing a divergence that supports the "
                             "coordinated side, which is one more voice for it, though not a trigger.")
            else:
                paras.append("<b>Funds argue against it.</b> The large speculators are showing a divergence that cuts across the "
                             "coordinated side. It is a caution layer rather than a reversal call, but it belongs in the weighing.")
        elif st_ == "Follow-through":
            paras.append("<b>No fund divergence.</b> The large speculators are still moving with price, so nothing argues against "
                         "the side from that direction and there is no divergence page for this market.")
        elif st_ == "Neutral":
            paras.append("<b>No fund divergence.</b> Price is not at a multi-week extreme, so the large speculators have nothing to "
                         "confirm or contradict this week.")

    # ---------------- why a side / combined bias ----------------
    if bias:
        if fb:
            hn, mn = _rpt_lit_layers(cp["comm"]), _rpt_lit_layers(cp["ret"])
            if cp["ret"]["pdir"] == -fb:
                masses = "The masses point the other way, and we have already decided which group wins that argument."
            elif cp["ret"]["pdir"] == fb:
                masses = "The masses, faded, point the same way, so the two groups reinforce each other."
            else:
                masses = "The masses are not at an extreme, so nothing in positioning opposes it."
            rec = ("The record behind it clears the bar." if edge else
                   "The record does not clear the bar yet, so read this as stretched rather than proven.")
            paras.append(f"<b>Why this is a side at all.</b> Hedgers are at an extreme on {_nw(hn)} of five lookbacks, and hedgers "
                         f"are the group we follow. {rec} {masses}")
        else:
            paras.append("<b>Why this is a side at all.</b> One engine is speaking and it is saying something real, while "
                         "positioning is not opposing it, only absent. A single clear seasonal is a side in this book, graded by "
                         "the quality of its window.")
        carries, honest = [], []
        if fb and edge:
            carries.append(f"the {labels.get(edge['key'], edge['key']).lower()} lookback on {rolename[edge['role']]} at "
                           f"{edge['t']['win']:.0f}% of {edge['t']['cases']} ({edge['t']['avg']:+.2f}%)")
        elif fb:
            carries.append("hedgers stretched with no qualifying row behind them")
            honest.append(f"no row reaches the {_nw(REPORT_EDGE_MIN_CASES)}-case floor")
        if m and cal_dir == bias:
            carries.append(f"{'an open' if live_now else 'an upcoming'} {act[bias]} at {m['win'] * 100:.0f}% of {m['cases']}, "
                           f"{m['rec_wins']} of its last {m['rec_cases']}")
        if weak_rows:
            honest.append(f"a weak {weak_rows[0][1]} row on {rolename[weak_rows[0][0]]}")
        if m and cal_dir == -bias:
            honest.append(f"{_rpt_art(m['win'] * 100)} {m['win'] * 100:.0f}% {act[cal_dir]} window {'running to' if live_now else 'opening'} "
                          f"{_rpt_pdate(m['end'] if live_now else m['start'])}")
        if m and cal_dir == bias and m.get("timing") in ("very tight", "tight"):
            honest.append(f"{m['timing']} timing")
        if m and cal_dir == bias and tier != "FULL":
            honest.append("a market that is off-pattern this year")
        engines = "both engines" if (fb and m and cal_dir == fb) else "single engine"
        dated = ", and dated" if (cal_promoted or (m and cal_dir == -bias)) else ""
        paras.append(f"<b>The combined bias.</b> {mood[bias].capitalize()}, {engines}{dated}. "
                     f"What carries it: {'; '.join(carries) or 'the calendar window alone'}. "
                     f"What keeps it honest: {'; '.join(honest) or 'nothing in the record argues back, which deserves a second look rather than comfort'}.")
    else:
        paras.append("<b>The combined bias.</b> Stand aside. Neither engine has earned a side this week, and a neutral read "
                     "is not an excuse to trade the middle.")
    return paras


def _rpt_trade_ideas_page(SetSection, header, note, CW):
    return None


def _rpt_build(table, prices, cot, latest_cot, only=None, chart_data=None, prior_state=None,
               mode="full", category=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
                                    TableStyle, PageBreak, Image, HRFlowable, Flowable)
    H = colors.HexColor
    S_ = _rpt_styles()
    today = dt.date.today()
    lazy = (mode == "lazy")
    date_str = today.strftime("%d %B %Y").lstrip("0")
    base_name = f"Invariant Weekly {date_str}"
    if category:
        base_name += f" — {category}"
    path = f"{base_name} (Lazy).pdf" if lazy else f"{base_name}.pdf"
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            stamp = dt.datetime.now().strftime("%H%M%S")
            path = (f"{base_name} (Lazy) {stamp}.pdf" if lazy
                    else f"{base_name} {stamp}.pdf")
    PW, PH = landscape(A4)
    M = 14 * mm
    CW = PW - 2 * M
    state = {"section": "Invariant Weekly"}
    _outline = {"seen": set(), "seq": 0}

    class SetSection(Flowable):
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
                self.canv.addOutlineEntry(self.txt, bkey, level=self.level, closed=(self.level == 0))
            except Exception:
                pass

    def on_bg(c, doc):
        c.saveState(); c.setFillColor(H(RPT["bg"])); c.rect(0, 0, PW, PH, stroke=0, fill=1); c.restoreState()

    def on_page(c, doc):
        c.saveState()
        c.setStrokeColor(H(RPT["rule"])); c.setLineWidth(0.6)
        c.line(M, 10 * mm, PW - M, 10 * mm)
        c.setFont("Helvetica-Bold", 8); c.setFillColor(H(RPT["ink"]))
        c.drawString(M, 6 * mm, "INVARIANT WEEKLY")
        c.setFont("Helvetica", 7.5); c.setFillColor(H(RPT["muted"]))
        _right = state['section']
        if category:
            _right = f"{category}  |  {_right}"
        c.drawRightString(PW - M, 6 * mm, f"{_right}  |  page {doc.page}")
        c.restoreState()

    doc = BaseDocTemplate(path, pagesize=(PW, PH), leftMargin=M, rightMargin=M, topMargin=11 * mm,
                          bottomMargin=14 * mm,
                          title="Invariant Weekly " + today.strftime("%d %B %Y").lstrip("0")
                                + (f" — {category}" if category else ""))
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(M, 14 * mm, CW, PH - 25 * mm, leftPadding=0,
                                                              rightPadding=0, topPadding=0, bottomPadding=0)],
                                       onPage=on_bg, onPageEnd=on_page)])

    def P(t, style="body"):
        return Paragraph(t, S_[style])

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
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        bar = HRFlowable(width="100%", thickness=2.0, color=H(RPT["blue"]), spaceBefore=3, spaceAfter=9)
        return [t, bar]

    def section(label, right=""):
        t = Table([[P(_esc(label).upper(), "label"), P(_esc(right), "right")]], colWidths=[CW * 0.55, CW * 0.45])
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.8, H(RPT["blue"])),
                               ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 1)]))
        return [Spacer(1, 5), t, Spacer(1, 6)]

    def card(content, accent=None, pad=11):
        t = Table([[content]], colWidths=[CW])
        st = [("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])), ("BOX", (0, 0), (-1, -1), 0.6, H(RPT["rule"])),
              ("LEFTPADDING", (0, 0), (-1, -1), pad + (5 if accent else 0)), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
              ("TOPPADDING", (0, 0), (-1, -1), pad - 2), ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 4)]
        if accent:
            st.append(("LINEBEFORE", (0, 0), (0, -1), 4.0, H(accent)))
        t.setStyle(TableStyle(st))
        return t

    def note(text):
        t = Table([[P(text, "note")]], colWidths=[CW])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["lav"])),
                               ("BOX", (0, 0), (-1, -1), 0.6, H(RPT["rule"])),
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

    def timing_display(t):
        return {"forgiving": "fairly forgiving", "tight": "tight",
                "very tight": "very tight", "moderate": "moderate"}.get(t, t)

    def nb_label_for(t):
        return {"forgiving": "supported", "tight": "narrow",
                "very tight": "very narrow", "moderate": "unknown"}.get(t, "supported")

    story = []
    out = table.copy()

    if category:
        _cat_mkts = set(REPORT_CATEGORIES.get(category, []))
        if _cat_mkts and "Market" in out.columns:
            out = out[
                out["Market"].apply(lambda v: str(v).replace(" (inv)", "") in _cat_mkts)
            ].reset_index(drop=True)
    if only:
        wanted = {o.lower() for o in only}
    elif category:
        wanted = {str(v).replace(" (inv)", "").lower() for v in out["Market"].tolist()}
    else:
        wanted = {o.lower() for o in REPORT_FORCE_MARKETS}

    weekly = prices.resample("W-FRI").last()
    mk = {m[0]: m for m in MARKETS}

    if category:
        _cat_size = len(REPORT_CATEGORIES.get(category, []))
        _partial_cap = max(FRONT_PARTIAL_CAP, _cat_size + 5)
        _dormant_cap = max(FRONT_DORMANT_CAP, _cat_size)
        _dashboard_cap = max(FRONT_DASHBOARD_CAP, _cat_size + 5)
    else:
        _partial_cap = FRONT_PARTIAL_CAP
        _dormant_cap = FRONT_DORMANT_CAP
        _dashboard_cap = FRONT_DASHBOARD_CAP

    log.info("Building %s report%s: calendar windows and flow packs ...",
             "lazy" if mode == "lazy" else "full",
             f" [{category}]" if category else "")
    packs = {}
    for _, r in out.iterrows():
        nm = str(r["Market"]).replace(" (inv)", "")
        if nm not in mk:
            continue
        _, group, ticker, code, invert = mk[nm]
        pk = {"row": r, "name": nm, "group": group, "ticker": ticker, "invert": invert,
              "cot": None, "season": None, "last_price": None, "location_read": None, "location_why": None,
              "weekly_df": None, "weekly_zones": None, "daily_zones": None, "monthly_zones": None}
        try:
            if ticker and ticker in prices.columns:
                pk["season"] = _rpt_season_pack(_measure_close(nm, ticker, prices), today, market_name=nm)
        except Exception as e:
            log.warning("Seasonal pack failed for %s: %s", nm, e)
        try:
            if code and group != "Indices":
                sub = cot[cot["code"] == code.lstrip("0")]
                if len(sub) >= COT_LOOKBACK_SHORT:
                    pk["cot"] = _rpt_cot_pack(sub, _measure_weekly(nm, ticker, prices), invert)
        except Exception as e:
            log.warning("COT pack failed for %s: %s", nm, e)
        try:
            item = (chart_data or {}).get(nm)
            if item and item.get("df") is not None and len(item["df"]) >= 120:
                daily_df = item["df"]
                pk["last_price"] = float(daily_df["Close"].iloc[-1])
                pk["daily_zones"] = compute_sd_zones(daily_df) if len(daily_df) >= 60 else []
                wk_df = _resample_safe(daily_df, "W-FRI")
                if len(wk_df) >= 30:
                    wk_zones = compute_sd_zones(wk_df)
                    pk["weekly_df"] = wk_df
                    pk["weekly_zones"] = wk_zones
                    _lw = []
                    pk["location_read"] = compute_location_read(wk_df, wk_zones, why_out=_lw)
                    pk["location_why"] = _lw[0] if _lw else None
                mo_df = _resample_safe(daily_df, "ME")
                pk["monthly_zones"] = compute_sd_zones(mo_df) if len(mo_df) >= 12 else []
        except Exception as e:
            log.warning("Tool 1/2 packs failed for %s: %s", nm, e)
        packs[nm] = pk

    if not lazy:
        ideas_page = _rpt_trade_ideas_page(SetSection, header, note, CW)
        if ideas_page:
            story.extend(ideas_page)

    def _focus_page():
        _ideas_now = load_ideas().get("ideas", {})
        _aging = set()
        for _i in _ideas_now.values():
            if not _is_open(_i):
                continue
            try:
                _opened = dt.date.fromisoformat(str(_i.get("opened_at") or _i.get("walk_from"))[:10])
                if (today - _opened).days > 28:
                    _aging.add(_i.get("market"))
            except Exception:
                pass
        buckets = {"READY": [], "PARTIAL": [], "DORMANT": []}
        for nm_wl, pk_wl in packs.items():
            if nm_wl in _aging:
                continue
            row = _effective_row_for_decision(pk_wl)
            if row is None:
                continue
            d = decide_market(row, pk_wl.get("location_read"), pk_wl.get("last_price"), pk_wl.get("location_why"))
            pk_wl["_decision"] = d
            buckets[d["bucket"]].append((nm_wl, pk_wl))
        if not (buckets["READY"] or buckets["PARTIAL"] or buckets["DORMANT"]):
            return None
        _board_title = "Activation Board" + (f" — {category}" if category else "")
        story_wl = [SetSection(_board_title, level=0, key="activation_board")]
        story_wl += header("Invariant Weekly", _board_title, REPORT_BRAND,
                           "Stack at a glance", today.strftime("%d %B %Y").lstrip("0"))
        for bucket, title_txt, subtitle_txt in (
            ("READY", "Ready — activation stack coherent", "the layers agree; location, execution and size remain the operator's call"),
            ("PARTIAL", "Partial — one layer missing or opposing", "a side exists, the missing layer is named in the rationale"),
            ("DORMANT", "Dormant — notable only", "no engine side; only CORE / always-chapter names are listed")):
            items = list(buckets[bucket])
            if not items:
                continue
            if bucket == "PARTIAL":
                def _partial_rank(t):
                    nm, pk = t
                    dec = pk.get("_decision") or {}
                    side = dec.get("side") or 0
                    core = 0 if (nm in CORE or nm in REPORT_ALWAYS_CHAPTER) else 1
                    reach = 0 if _zone_within_reach(pk.get("location_read"), pk.get("last_price"), side) else 1
                    try:
                        att = -float(pk["row"].get("AttentionScore") or 0)
                    except Exception:
                        att = 0.0
                    return (core, reach, att, nm)
                items = sorted(items, key=_partial_rank)[:_partial_cap]
            elif bucket == "DORMANT":
                if not category:
                    items = [t for t in items if t[0] in CORE or t[0] in REPORT_ALWAYS_CHAPTER]
                items = items[:_dormant_cap]
            if not items:
                continue
            story_wl += section(title_txt, subtitle_txt)
            rows_wl = [[P(h, "label") for h in ("MARKET", "SIDE", "RATIONALE")]]
            for nm_wl, pk_wl in sorted(items, key=lambda t: _rpt_sort_key(t[0], t[1]["group"])):
                _, full_wl, _ = _rpt_market_display(nm_wl)
                dec_wl = pk_wl["_decision"]
                side_txt = "LONG" if dec_wl["side"] == 1 else "SHORT" if dec_wl["side"] == -1 else "—"
                side_col = RPT["green"] if side_txt == "LONG" else RPT["red"] if side_txt == "SHORT" else RPT["muted"]
                rows_wl.append([P(f"<b>{_esc(full_wl)}</b>", "cell"),
                                P(colored(side_txt, side_col), "cell"),
                                P(_esc(dec_wl["why"]), "cell")])
            t_wl = Table(rows_wl, colWidths=[CW * w for w in (0.18, 0.08, 0.74)], repeatRows=1)
            t_wl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
                                      ("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                                      ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                      ("TOPPADDING", (0, 0), (-1, -1), 5),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story_wl.append(t_wl)
            story_wl.append(Spacer(1, 6))
        _dormant_howto = ("no engine side — every market in the category is listed for completeness"
                          if category else
                          "no engine side; only CORE / always-watched names are listed")
        story_wl.append(note(
            f"<b>How to read this.</b> This board is triage, not a shopping list. <b>Ready</b> — the activation stack is coherent: "
            "a side from flow positioning (relative value for indices), at least one independent engine confirming it, and nothing "
            "opposing. That is the system's whole verdict; work those names first. <b>Partial</b> — a side exists but one layer is "
            "missing or opposing (CORE and zone-within-reach names are prioritised). <b>Dormant</b> — "
            f"{_dormant_howto}. A neutral read is not an excuse to trade the middle. Full evidence for every market "
            "lives in its chapter."))
        story_wl.append(PageBreak())
        return story_wl

    focus_page = _focus_page()
    if focus_page:
        story.extend(focus_page)

    story.append(SetSection("Seasonal Overview", level=0, key="seasonal_overview"))
    from reportlab.lib.styles import ParagraphStyle as _PS
    from reportlab.lib.enums import TA_RIGHT as _TAR
    def _front_season_keep(p):
        sp = p.get("season") or {}
        m = sp.get("main")
        if not m:
            return False
        # bp30: the reference overview lists off-pattern and counter-seasonal
        # markets too (Copper at -0.18 is on theirs), so no shape-tier exclusion.
        if category:
            return True
        if FRONT_SEASON_STRONG_ONLY and m.get("strength") != "STRONG":
            return False
        nm = p.get("name")
        if nm in CORE or nm in REPORT_ALWAYS_CHAPTER:
            return True
        dec = p.get("_decision") or {}
        if dec.get("bucket") in ("READY", "PARTIAL"):
            return True
        return False

    strong = [p for p in packs.values() if _front_season_keep(p)]
    buys = [p for p in strong if p["season"]["main"]["dir"] == 1]
    sells = [p for p in strong if p["season"]["main"]["dir"] == -1]
    if len(strong) >= 3 and (len(buys) >= len(strong) * 0.7 or len(sells) >= len(strong) * 0.7):
        theme = (f"{len(strong)} actionable markets carry a strong calendar window in the -2/+8wk horizon; "
                 f"{max(len(buys), len(sells))} of them lean the same way.")
    elif strong:
        theme = ("A handful of actionable markets have a calendar window this week — "
                 "read each on its own merits below."
                 if not category else
                 "Every market in this category with an active calendar window is listed below.")
    else:
        theme = ("No actionable market has a calendar window in the -2/+8wk horizon this week. "
                 "Any moderate windows still appear in individual chapters.")
    _ov_cell = _PS("ovc", parent=S_["cell"], fontSize=8.4, leading=10.5)
    _ov_cell_r = _PS("ovcr", parent=_ov_cell, alignment=_TAR)
    _ov_lab = _PS("ovlab", parent=S_["label"], fontSize=7.2, leading=9)
    _ov_lab_r = _PS("ovlabr", parent=_ov_lab, alignment=_TAR)
    _ov_body = _PS("ovb", parent=S_["body"], fontSize=11, leading=15)

    story += header("Weekly Market Outlook", "Seasonal Overview", REPORT_BRAND,
                    "Bias · Seasonality", today.strftime("%d %B %Y").lstrip("0"))
    story.append(card([P("<font size=17><b>Seasonal overview</b></font>", "h"), Spacer(1, 2),
                       Paragraph(_esc(theme), _ov_body)], RPT["green"]))
    story.append(Spacer(1, 6))
    story += section("Best seasonal windows in play",
                     f"entry within -2 to +8 weeks · {REPORT_SEASON_YEARS}-year basis")

    def _build_col(sections, col_w):
        rows = []; styles = []
        for g in sections:
            members = [p for p in packs.values()
                       if _rpt_section(p["name"], p["group"]) == g and _front_season_keep(p)]
            if not members:
                continue
            def _rank_key(p):
                m = p["season"]["main"] if p.get("season") else None
                if not m:
                    return (2, 0, 0)
                return (0 if m["strength"] == "STRONG" else 1, -m["win"], -abs(m.get("avg", 0)))
            members = sorted(members, key=_rank_key)
            if g == "Majors":
                members.sort(key=lambda p: (0 if p["name"] == "DXY" else 1, _rpt_sort_key(p["name"], p["group"])[1]))
            g_label = "Currencies" if g == "Majors" else g
            rows.append([Paragraph(f"<font color='{RPT['blue']}'><b>{_esc(g_label).upper()}</b></font>", _ov_lab)] + [""] * 6)
            styles.append(("SPAN", (0, len(rows) - 1), (6, len(rows) - 1)))
            styles.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), H(RPT["lav"])))
            styles.append(("TOPPADDING", (0, len(rows) - 1), (-1, len(rows) - 1), 4))
            for p in members:
                sp2 = p["season"]; m2 = sp2["main"] if sp2 else None
                _, _, ov_label = _rpt_market_display(p["name"])
                if not m2:
                    rows.append([Paragraph(_esc(ov_label), _ov_cell),
                                 Paragraph('<font color="#8b949e">no window in the -2/+8wk horizon</font>', _ov_cell),
                                 "", "", "", "", ""])
                    styles.append(("SPAN", (1, len(rows) - 1), (6, len(rows) - 1)))
                    continue
                tc = RPT["green"] if m2["dir"] == 1 else RPT["red"]
                sc = RPT["ink"] if m2["strength"] == "STRONG" else RPT["amber"]
                _disp_strength, _disp_tier = _season_display_strength(sp2)
                rows.append([
                    Paragraph(f"<b>{_esc(ov_label)}</b>", _ov_cell),
                    Paragraph(colored("BUY" if m2["dir"] == 1 else "SELL", tc), _ov_cell),
                    Paragraph(f"{m2['start'].strftime('%d %b')} – {m2['end'].strftime('%d %b')}", _ov_cell_r),
                    Paragraph(f'<font color="#8b949e">{m2["len"]}d · {m2["shape"]}</font>', _ov_cell),
                    Paragraph(f"<b>{m2['win'] * 100:.0f}%</b>", _ov_cell_r),
                    Paragraph(f"{m2['wins']} of {m2['cases']}", _ov_cell_r),
                    Paragraph(colored(_disp_strength, sc), _ov_cell_r),
                ])
        hdr = [Paragraph("MARKET", _ov_lab), Paragraph("TRADE", _ov_lab),
               Paragraph("WINDOW", _ov_lab_r), Paragraph("LENGTH", _ov_lab),
               Paragraph("WIN", _ov_lab_r), Paragraph("CASES", _ov_lab_r),
               Paragraph("SEASONAL STRENGTH", _ov_lab_r)]
        rows.insert(0, hdr)
        t = Table(rows, colWidths=[col_w * w for w in (0.19, 0.10, 0.19, 0.20, 0.07, 0.10, 0.15)], repeatRows=1)
        t.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, H(RPT["rule"])),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, H(RPT["blue"])),
        ] + styles))
        return t

    _col_w = CW / 2 - 6
    left_t = _build_col(["Majors", "Indices"], _col_w)
    right_t = _build_col(["Precious Metals", "Energies", "Agriculturals"], _col_w)
    pair = Table([[left_t, right_t]], colWidths=[CW / 2, CW / 2])
    pair.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 6),
        ("LEFTPADDING", (1, 0), (1, 0), 6), ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ]))
    story.append(pair)
    story.append(Spacer(1, 8))
    story.append(note("<b>How to read this overview.</b> Seasonality is the <b>add-on</b> to the flow bias; it never "
                      "engages alone. Every market with an active calendar window in the horizon is listed below, "
                      "STRONG first. Ranked by hit-rate, average move, recent record, and neighbour support. "
                      "The window is <b>locked</b> the first time a market enters the horizon and only re-scored from then on. "
                      "Markets off-pattern this year carry a suffix."))
    story.append(PageBreak())

    story.append(SetSection("Engagement Dashboard", level=0, key="dashboard"))
    story += header("Invariant Weekly", "Engagement Dashboard", REPORT_BRAND,
                    "Flow + relative value + calendar flow", today.strftime("%d %B %Y").lstrip("0"))
    def _edge_dir(lbl):
        lbl = str(lbl)
        return "Bullish" if "Bullish" in lbl else "Bearish" if "Bearish" in lbl else None
    _ed = out["EdgeFinder"].map(_edge_dir)
    aligned = out[_ed.notna() & (out["COTBias"] == _ed)].copy()
    _active = {nm for nm, pk in packs.items()
               if (pk.get("_decision") or {}).get("bucket") in ("READY", "PARTIAL")}
    if _active and "Market" in aligned.columns:
        def _mkt(v):
            return str(v).replace(" (inv)", "")
        prefer = aligned[aligned["Market"].map(_mkt).isin(_active)]
        if len(prefer) >= 4:
            aligned = prefer
        else:
            core_extra = aligned[aligned["Market"].map(_mkt).isin(CORE) & ~aligned["Market"].map(_mkt).isin(_active)]
            aligned = pd.concat([prefer, core_extra]).drop_duplicates(subset=["Market"])
    story += section("Best aligned markets", f"latest positioning report {latest_cot or '-'}")
    hdr = ["MARKET", "SECTION", "CORE", "FLOW SIDE", "FLOW", "REL VALUE", "VAL STRENGTH", "CAL FLOW", "HARVEST PROFILE"]
    rows = [[P(h, "label") for h in hdr]]
    def tint(v):
        return colored(_esc(v), RPT["green"] if v == "Bullish" else RPT["red"] if v == "Bearish" else RPT["muted"])
    shown = aligned.sort_values(["AttentionScore"], ascending=False).head(_dashboard_cap)
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
        rows.append([P("No market has flow and relative value aligned this week.", "cell")] + [""] * 8)
    t = Table(rows, colWidths=[CW * w for w in (0.12, 0.10, 0.05, 0.09, 0.05, 0.10, 0.12, 0.11, 0.26)], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])), ("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t, Spacer(1, 8)]
    story += section("Reading the dashboard")
    story.append(note("<b>Aligned</b> = the harvest profile's best historical setup points the same way as the hedger side. "
                      "<b>Calendar flow</b> tempers conviction unless flow has no side and the window is STRONG."))
    story.append(PageBreak())

    prior_mkts = (prior_state or {}).get("markets") or {}
    changes = []
    if prior_mkts:
        for _, r in out.iterrows():
            nm = str(r.get("Market", "")).replace(" (inv)", "")
            if not nm:
                continue
            prev = prior_mkts.get(nm)
            if not prev:
                continue
            lines = []
            cur_bias = str(r.get("COTBias", "Neutral")); prev_bias = str(prev.get("bias", "Neutral"))
            if cur_bias != prev_bias:
                lines.append(f"flow side flipped from <b>{_esc(prev_bias)}</b> to <b>{_esc(cur_bias)}</b>")
            _q = _safe_float(r.get("EdgeFinderWin"))
            cur_edge = 1 if (r.get("EdgeFinder", "-") != "-" and np.isfinite(_q)) else 0
            prev_edge = int(prev.get("qualifying_rows") or 0)
            if cur_edge and not prev_edge:
                lines.append(f"gained a qualifying harvest profile: {_esc(str(r.get('EdgeFinder', '-')))}")
            elif prev_edge and not cur_edge:
                lines.append("lost its qualifying harvest profile")
            cur_sw = str(r.get("SeasonWindow", "-")); prev_sw = str(prev.get("season_window", "-"))
            cur_ss = r.get("SeasonWindowStrength"); prev_ss = prev.get("season_strength")
            if cur_sw != prev_sw and (cur_sw != "-" or prev_sw != "-"):
                if prev_sw in ("-", "", "None", "none") and cur_sw not in ("-", "", "None", "none"):
                    lines.append(f"new calendar window: {_esc(cur_sw)}" + (f" · {_esc(str(cur_ss))}" if cur_ss else ""))
                elif cur_sw in ("-", "", "None", "none") and prev_sw not in ("-", "", "None", "none"):
                    lines.append("calendar window dropped out of the horizon")
                else:
                    lines.append(f"calendar window changed ({_esc(prev_sw)} → {_esc(cur_sw)})")
            elif cur_ss and prev_ss and str(cur_ss) != str(prev_ss):
                lines.append(f"calendar strength moved from {_esc(str(prev_ss))} to {_esc(str(cur_ss))}")
            prev_lit = int(prev.get("cot_lit") or 0)
            cur_lit = _safe_int(r.get("CommExtremeCount"))
            if cur_lit != prev_lit and (cur_lit > 0 or prev_lit > 0):
                if cur_lit > prev_lit:
                    lines.append(f"hedger extremes expanded ({prev_lit} → {cur_lit} of 5)")
                else:
                    lines.append(f"hedger extremes contracted ({prev_lit} → {cur_lit} of 5)")
            if lines:
                sec = _rpt_section(nm, r.get("Group", ""))
                changes.append((_rpt_sort_key(nm, r.get("Group", "")), nm, sec, lines))
        changes.sort(key=lambda z: z[0])

    story.append(SetSection("Stack Deltas", level=0, key="stack_deltas"))
    story += header("Invariant Weekly", "Stack Deltas", REPORT_BRAND,
                    "Week-over-week movement in the activation stack", today.strftime("%d %B %Y").lstrip("0"))
    prior_run = (prior_state or {}).get("run_date") or "—"
    story += section("Week-over-week moves", f"compared with run of {prior_run}")
    if not prior_mkts:
        story.append(note("No prior run is stored yet. After the next weekly scan, this page will list markets whose "
                          "flow side flipped, that gained or lost a qualifying harvest profile, or whose calendar window changed."))
    elif not changes:
        story.append(note(f"No material changes versus the run of {prior_run}. "
                          "No flow side flipped, no qualifying harvest profile was gained or lost, and no calendar window moved."))
    else:
        story.append(note(f"<b>{len(changes)} market{'s' if len(changes) != 1 else ''}</b> moved versus the run of {prior_run}. "
                          "Only flips in flow side, qualifying harvest profile, hedger extreme count, or the calendar window are listed."))
        story.append(Spacer(1, 6))
        ch_rows = [[P(h, "label") for h in ("MARKET", "SECTION", "WHAT CHANGED")]]
        for _, nm, sec, lines in changes:
            ch_rows.append([P(f"<b>{_esc(nm)}</b>", "cell"), P(_esc(sec), "cell"), P(" · ".join(lines), "cell")])
        ct = Table(ch_rows, colWidths=[CW * w for w in (0.18, 0.14, 0.68)], repeatRows=1)
        ct.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
                                ("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(ct)
    story.append(PageBreak())

    rows_data = []
    for nm_lr, pk_lr in packs.items():
        lr = pk_lr.get("location_read")
        if not lr or not lr.get("tradeable"):
            continue
        action = lr["action"]
        is_counter = ("counter-trend" in action) or ("fade" in action)
        reason = ""
        if is_counter:
            reason = _has_countertrend_case(pk_lr["row"], _action_dir(action)) or ""
            if not reason:
                continue
        rows_data.append((nm_lr, pk_lr, lr, is_counter, reason))
    if not SHOW_CHART_PAGES:
        rows_data = []
    if rows_data:
        def _rank(r):
            action = r[2]["action"]
            if "with the trend" in action:
                return 0
            if "counter-trend" in action:
                return 1
            return 2
        rows_data.sort(key=lambda r: (_rank(r), r[0]))
        story.append(SetSection("Location Read", level=0, key="location_read"))
        story += header("Invariant Weekly", "Location Read", REPORT_BRAND,
                        "HTF location · direction · Action Matrix", today.strftime("%d %B %Y").lstrip("0"))
        story += section("Where price sits in the HTF range",
                         "Only markets where location and direction align into a tradeable action")
        hdr_lr = ["MARKET", "TREND", "PHASE", "LOCATION", "RANGE", "ACTION"]
        rows_lr = [[P(h, "label") for h in hdr_lr]]
        for nm_lr, pk_lr, lr, is_counter, reason in rows_data:
            _, full_lr, _ = _rpt_market_display(nm_lr)
            tcol = RPT["green"] if lr["trend"] == "UPTREND" else RPT["red"] if lr["trend"] == "DOWNTREND" else RPT["muted"]
            acol = RPT["green"] if "buy" in lr["action"] else RPT["red"] if ("short" in lr["action"] or "sell" in lr["action"]) else RPT["muted"]
            range_txt = f"{lr['range'][0]:.4g} – {lr['range'][1]:.4g}"
            action_txt = _display_action(lr["action"]).upper()
            action_html = colored(action_txt, acol)
            if is_counter and reason:
                action_html += f' <font color="#8b949e" size="7.5">· {_esc(reason)}</font>'
            rows_lr.append([P(f"<b>{_esc(full_lr)}</b>", "cell"),
                            P(colored(lr["trend"].title(), tcol), "cell"),
                            P(f'<font color="#8b949e">{lr["phase"].title()}</font>', "cell"),
                            P(f"<b>{lr['location'].title()}</b>", "cell"),
                            P(range_txt, "cell"),
                            P(action_html, "cell")])
        t_lr = Table(rows_lr, colWidths=[CW * w for w in (0.16, 0.12, 0.12, 0.16, 0.20, 0.24)], repeatRows=1)
        t_lr.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
                                  ("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                  ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story.append(t_lr)
        story.append(Spacer(1, 8))
        story.append(note("<b>How to read this.</b> Location is context, not a signal. A <b>Low</b> location inside an "
                          "<b>Uptrend</b> means we look to buy, but only in a fresh mapped demand reference. The Action column "
                          "is the Action Matrix applied verbatim."))
        story.append(PageBreak())

    rows_data = []
    for nm_cf, pk_cf in packs.items():
        m_zones = pk_cf.get("monthly_zones") or []
        w_zones = pk_cf.get("weekly_zones") or []
        d_zones = pk_cf.get("daily_zones") or []
        price = pk_cf.get("last_price")
        if price is None or not np.isfinite(price):
            continue
        conf = compute_confluence(m_zones, w_zones, d_zones, price)
        if not conf:
            continue
        row = pk_cf["row"]
        conf = [c for c in conf if _confluence_aligned_with_bias(c, row)]
        if conf:
            rows_data.append((nm_cf, pk_cf, conf, price))
    if not SHOW_CHART_PAGES:
        rows_data = []
    if rows_data:
        triples = []; doubles = []
        for nm_cf, pk_cf, conf, price in rows_data:
            for c in conf:
                (triples if len(c["tfs"]) == 3 else doubles).append((nm_cf, pk_cf, c, price))
        triples.sort(key=lambda t: t[2]["distance_pct"])
        doubles.sort(key=lambda t: t[2]["distance_pct"])
        total = len(triples) + len(doubles)
        if total > CONFLUENCE_MAX_ROWS:
            keep_triples = triples[:min(len(triples), CONFLUENCE_MAX_ROWS)]
            remaining = CONFLUENCE_MAX_ROWS - len(keep_triples)
            keep_doubles = doubles[:max(0, remaining)]
            triples, doubles = keep_triples, keep_doubles
        story.append(SetSection("Precision Mapping", level=0, key="precision_mapping"))
        story += header("Invariant Weekly", "Precision Mapping", REPORT_BRAND,
                        "Monthly · Weekly · Daily reference overlaps", today.strftime("%d %B %Y").lstrip("0"))
        def _render_section(title, subtitle, items, is_triple):
            if not items:
                return []
            block = []
            block += section(title, subtitle)
            hdr_cf = ["MARKET", "SIDE", "TIMEFRAMES", "REFERENCE", "DISTANCE"]
            rows_cf = [[P(h, "label") for h in hdr_cf]]
            for nm_cf, pk_cf, c, price in items:
                _, full_cf, _ = _rpt_market_display(nm_cf)
                dec = _px_dec(price)
                side_col = RPT["green"] if c["demand"] else RPT["red"]
                tf_label = "+".join(c["tfs"])
                zone_txt = f"{c['bottom']:.{dec}f} – {c['top']:.{dec}f}"
                dist_txt = _zone_dist_txt(c["bottom"], c["top"], price)
                tf_html = f'<b>{tf_label}</b>' if is_triple else f'<font color="#8b949e">{tf_label}</font>'
                rows_cf.append([P(f"<b>{_esc(full_cf)}</b>", "cell"),
                                P(colored("DEMAND" if c["demand"] else "SUPPLY", side_col), "cell"),
                                P(tf_html, "cell"), P(zone_txt, "cell"), P(dist_txt, "cell")])
            t_cf = Table(rows_cf, colWidths=[CW * w for w in (0.20, 0.12, 0.14, 0.30, 0.24)], repeatRows=1)
            t_cf.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
                                      ("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                                      ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                      ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            block.append(t_cf)
            block.append(Spacer(1, 8))
            return block
        story += _render_section("Triple mapping", "Monthly · Weekly · Daily — every timeframe agrees",
                                 triples, is_triple=True)
        story += _render_section("Double mapping", "Any two timeframes agreeing near price",
                                 doubles, is_triple=False)
        story.append(note("<b>How to read this.</b> Only references where the market's own engine side agrees with the "
                          "reference direction are shown — a flow-led bias, or a STRONG calendar window when flow is neutral."))
        story.append(PageBreak())

    story.append(SetSection("Pattern Coherence", level=0, key="pattern_coherence"))
    story += header("Invariant Weekly", "Pattern Coherence", REPORT_BRAND,
                    "Is this year tracking each market's seasonal shape?", today.strftime("%d %B %Y").lstrip("0"))
    story += section("Shape scan", "is THIS year following each market's seasonal shape?")
    ranked = []
    for nm, pk in packs.items():
        sp = pk.get("season")
        if not sp or not np.isfinite(sp.get("shape", np.nan)):
            continue
        ranked.append((float(sp["shape"]), nm, pk))
    ranked.sort(reverse=True)
    rows = [[P(x, "label") for x in ("MARKET", "SHAPE MATCH", "", "READ")]]
    track_w = CW * 0.30
    for sh, nm, pk in ranked[:14]:
        _, full, _ = _rpt_market_display(nm)
        if sh >= SHAPE_CLOSE:
            read, col = "tracking closely", RPT["blue"]
        elif sh >= SHAPE_LOOSE:
            read, col = "loosely tracking", RPT["amber"]
        elif sh >= 0.0:
            read, col = "off-pattern", RPT["muted"]
        else:
            read, col = "counter-seasonal", RPT["muted"]
        frac = max(0.02, min(1.0, (sh + 1.0) / 2.0))
        fill_w = track_w * frac; rest_w = max(0.5, track_w - fill_w)
        fill = Table([[""]], colWidths=[fill_w], rowHeights=[8])
        fill.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(col)),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                  ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        rest = Table([[""]], colWidths=[rest_w], rowHeights=[8])
        rest.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["grey"])),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                  ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        bar = Table([[fill, rest]], colWidths=[fill_w, rest_w])
        bar.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                 ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        score_col = col if sh >= SHAPE_LOOSE else RPT["muted"]
        rows.append([P(f"<b>{_esc(full)}</b>", "cell"),
                     P(f'<font color="{score_col}"><b>{sh:+.2f}</b></font>', "cell"),
                     bar,
                     P(f'<font color="{col}">{read}</font>', "cell")])
    t = Table(rows, colWidths=[CW * w for w in (0.30, 0.10, 0.32, 0.28)])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                           ("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(card([P("<font color='#58a6ff'><b>WHY THE SHAPE SCAN MATTERS</b></font>", "label"),
                       P("A seasonal average can have a brilliant 15-year record and still be the wrong guide "
                         "<b>this year</b>, if price is not actually following the pattern. The shape scan measures "
                         "exactly that: how closely this year's path tracks its own seasonal rhythm."),
                       P(f"<b>High and positive</b> (top of the list) {ARROW} the calendar is in control, so its forward window "
                         f"is worth weighting. <b>Low or negative</b> {ARROW} the market is off-pattern this year, so we "
                         "<b>discount</b> the calendar no matter how strong its history, and lean harder on flow.")],
                      RPT["blue"]))
    story.append(PageBreak())

    cands = []; filler = []; must_chapter = set()
    try:
        for nm_m, p_m in packs.items():
            _eff = _effective_row_for_decision(p_m)
            if _eff is None:
                continue
            if decide_market(_eff, p_m.get("location_read"), p_m.get("last_price"),
                             p_m.get("location_why"))["bucket"] == "READY":
                must_chapter.add(nm_m)
            _ef_label = str(_eff.get("EdgeFinder", "-") or "-").strip()
            if _ef_label and _ef_label not in ("-", "nan", "None", ""):
                must_chapter.add(nm_m)
        must_chapter |= {i["market"] for i in load_ideas().get("ideas", {}).values()
                         if i.get("status") in ("ARMED", "TRIGGERED")}
    except Exception as e:
        log.warning("Could not build the must-have chapter list: %s", e)
    for nm, p in packs.items():
        cp = p["cot"]; r = p["row"]
        if wanted and nm.lower() not in wanted:
            continue
        if not wanted and (not cp or cp["bias"] == 0) and not (p["season"] and p["season"]["main"]
                                                              and p["season"]["main"]["strength"] == "STRONG") \
                and nm not in REPORT_ALWAYS_CHAPTER and nm not in must_chapter:
            filler.append((float(r.get("AttentionScore", 0) or 0), (cp["lit"] if cp else 0), nm))
            continue
        cands.append((float(r.get("AttentionScore", 0) or 0), (cp["lit"] if cp else 0), nm))
    cands.sort(reverse=True); filler.sort(reverse=True)
    if not wanted and not lazy:
        cands += filler[: max(0, REPORT_MAX_CHAPTERS - len(cands))]
    max_ch = REPORT_MAX_CHAPTERS if not lazy else min(8, REPORT_MAX_CHAPTERS)
    ordered = ([c for c in cands if c[2] in must_chapter] +
               [c for c in cands if c[2] in REPORT_ALWAYS_CHAPTER and c[2] not in must_chapter] +
               [c for c in cands if c[2] not in REPORT_ALWAYS_CHAPTER and c[2] not in must_chapter])
    _guaranteed = len({c[2] for c in cands if c[2] in must_chapter or c[2] in REPORT_ALWAYS_CHAPTER})
    max_ch = max(max_ch, _guaranteed)
    chosen = [c[2] for c in ordered][: (len(cands) if wanted else max_ch)]
    chosen.sort(key=lambda n: _rpt_sort_key(n, packs[n]["group"]))
    log.info("Chapters%s: %s", f" [{category}]" if category else "",
             ", ".join(chosen) if chosen else "none")
    _bookmarked_sections = set()

    for nm in chosen:
        pk = packs[nm]
        cp = pk["cot"]; sp = pk["season"]; r = pk["row"]
        _short, full, _ov = _rpt_market_display(nm)
        title_pair = f"{_short} ({full})" if _short != full else full
        bias = cp["bias"] if cp else 0
        season_promoted = (bias == 0 and sp and sp.get("main") and sp["main"]["strength"] == "STRONG"
                           and sp.get("tier", "FULL") != "DROP" and not sp.get("cancelled"))
        if season_promoted:
            bias = sp["main"]["dir"]
        try:
            _eff_row = _effective_row_for_decision(pk)
            _dec_ch = decide_market(_eff_row, pk.get("location_read"), pk.get("last_price"), pk.get("location_why"))
            if _dec_ch["side"] != 0 and not _dec_ch["season_led"]:
                bias = _dec_ch["side"]
        except Exception:
            pass
        word = {1: "buy", -1: "sell", 0: "stand aside"}[bias]
        demand = {1: "Demand", -1: "Supply", 0: "a reference"}[bias]
        rolename = {"commercial": "hedgers", "retail": "the masses"}
        in_play = bool(sp and sp.get("main") and sp["today"] >= sp["main"]["start"] and sp["today"] <= sp["main"]["end"])
        sec_name = _rpt_section(nm, pk["group"])
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

        if lazy:
            _ensure_mkt_bookmark()
            story += header(f"{full}", f"{full}", REPORT_BRAND, "Quick read",
                            today.strftime("%d %B %Y").lstrip("0"))
            cot_txt = "no flow side"
            if cp:
                cot_txt = (f"Hedgers {RPT_DIRWORD[cp['comm']['pdir']].lower()} on {cp['comm']['lit']} of 5, "
                           f"the masses {RPT_DIRWORD[cp['ret']['pdir']].lower()} on {cp['ret']['lit']} of 5"
                           if cp["comm"]["pdir"] or cp["ret"]["pdir"] else "no extreme on either group")
            sea_txt = "no strong window"
            if sp and sp.get("main"):
                m = sp["main"]
                _disp, _ = _season_display_strength(sp)
                sea_txt = (f"{'LONG' if m['dir']==1 else 'SHORT'} {m['start'].strftime('%d %b')}–{m['end'].strftime('%d %b')} "
                           f"· {_disp} · {m['win']*100:.0f}% of {m['cases']} · {m['shape']}")
            lines = [P(f"<b>{_esc(full)}</b>  <font color='{_rpt_dircolor(bias)}' size=16><b>{RPT_DIRWORD[bias]}</b></font>", "h"),
                     P(f"<b>Flow positioning.</b> {cot_txt}.", "body"),
                     P(f"<b>Calendar flow.</b> {sea_txt}.", "body")]
            if bias:
                lines.append(P(f"<b>Directive.</b> Look to {word} in <b>{demand}</b> only. "
                               f"A bias is not a trade — wait for price to reach a live mapped reference.", "body"))
            else:
                lines.append(P("<b>Directive.</b> Stand dormant. Neither flow nor calendar flow gives a side this week.", "body"))
            if cp and cp.get("edge"):
                ed = cp["edge"]; t4 = ed["t"]
                lab = dict(RPT_TABLE_ORDER).get(ed["key"], ed["key"])
                lines.append(P(f"<b>Harvest profile.</b> Best track record: {lab} on {ed['role']} — "
                               f"{t4['win']:.0f}% over {ed['h']}w, {t4['wins']} of {t4['cases']}, {t4['avg']:+.2f}% avg.", "body"))
            item = (chart_data or {}).get(nm)
            if item and bias:
                if "zones" not in item:
                    item["zones"] = compute_sd_zones(item["df"])
                near = _sd_nearest_fresh(item["zones"], float(item["df"]["Close"].iloc[-1]), bias)
                if near:
                    nz_, dist_ = near
                    dd = _px_dec(item["df"]["Close"].iloc[-1])
                    lines.append(P(f"<b>Nearest {demand} reference.</b> {_esc(nz_['setup'])}, "
                                   f"{nz_['bottom']:.{dd}f}–{nz_['top']:.{dd}f} "
                                   f"({dist_:.1f}% {'below' if bias == 1 else 'above'} price).", "body"))
            story.append(card(lines, _rpt_dircolor(bias)))
            story.append(Spacer(1, 6))
            if item:
                try:
                    cbuf = _rpt_price_chart(item, full, tf="D")
                    if cbuf is not None:
                        story.append(img(cbuf))
                        story.append(P(f"Daily · {item['desc']} · preferred proximal references.", "small"))
                except Exception as e:
                    log.warning("Lazy daily chart failed for %s: %s", nm, e)
            if bias == 1:
                read_txt = "<b>THE READ</b>  Long in daily Demand only. A bias is not a trade — wait for the reference."
            elif bias == -1:
                read_txt = "<b>THE READ</b>  Short in daily Supply only. A bias is not a trade — wait for the reference."
            else:
                read_txt = "<b>THE READ</b>  No side this week. Stand dormant."
            story.append(Spacer(1, 4))
            story.append(card([P(read_txt, "body")], _rpt_dircolor(bias)))
            story.append(PageBreak())
            continue

        if cp:
            comm, ret = cp["comm"], cp["ret"]
            _ensure_mkt_bookmark()
            try:
                cot_day = pd.Timestamp(latest_cot)
                cot_txt_hdr = f"Positioning as of {cot_day.strftime('%A')} {cot_day.strftime('%Y-%m-%d')}"
            except Exception:
                cot_txt_hdr = f"Positioning as of {latest_cot or '-'}"
            story += header("Invariant Weekly", f"Week of {date_str}", REPORT_BRAND,
                            "Layer 1 · Bias from flow positioning", cot_txt_hdr)
            nr, nc = ret["lit"], comm["lit"]
            if comm["pdir"] and ret["pdir"] == comm["pdir"]:
                relation = "and here both point the same way"
            elif comm["pdir"] and ret["pdir"] == -comm["pdir"]:
                relation = "and here they point in opposite directions, so the hedgers carry the side"
            elif comm["pdir"]:
                relation = "and only the hedgers have an extreme, so they carry the side"
            else:
                relation = "and the hedgers have no extreme, so there is no valid flow side"
            if nr and nc:
                intro = (f"The masses and the hedgers are <b>both at an extreme</b>, on <b>{_nw(nr)} of five</b> lookbacks "
                         f"(masses) and <b>{_nw(nc)} of five</b> (hedgers). We fade the masses, we follow the "
                         f"hedgers, {relation}. ")
            else:
                intro = (f"The masses are at an extreme on <b>{_nw(nr)} of five</b> lookbacks, hedgers on "
                         f"<b>{_nw(nc)} of five</b>. We fade the masses, we follow the hedgers, {relation}. ")
            if cp["bias"]:
                intro += (f"<b>Layer 1 gives us the side: we are looking to {word}.</b> "
                          f"Layer 2 gives us the reference, and we only {word} in {demand}.")
            elif season_promoted:
                intro += f"<b>Layer 1 has no side, but calendar flow is STRONG and takes the lead. We are looking to {word}.</b>"
            else:
                intro = _rpt_standaside_text(full, cp)
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.enums import TA_RIGHT
            _nm_style = ParagraphStyle("vnm", parent=S_["h"], fontSize=22, leading=25)
            _dir_style = ParagraphStyle("vdir", parent=S_["h"], fontSize=22, leading=25, alignment=TA_RIGHT)
            _pill_style = ParagraphStyle("vpill", parent=S_["cell"], fontSize=9, leading=11, textColor=H(RPT["muted"]))
            pill = Table([[Paragraph(f"<b>{_esc(_rpt_tvlabel(pk))}</b>", _pill_style)]], colWidths=[84], hAlign="LEFT")
            pill.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, H(RPT["rule"])),
                                      ("BACKGROUND", (0, 0), (-1, -1), H(RPT["lav"])),
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
                chip_size = 12; chip_gap = 4
                n_lit = sum(1 for (key, lab) in RPT_CHIP_ORDER
                            if bool(st["pdir"]) and st["layers"][key]["pdir"] == st["pdir"])
                chip_cells = [""] * 9; chip_styles = []
                for i in range(5):
                    col_i = i * 2
                    chip_styles.append(("BACKGROUND", (col_i, 0), (col_i, 0), H(accent if i < n_lit else RPT["card"])))
                    chip_styles.append(("BOX", (col_i, 0), (col_i, 0), 0.9, H(accent)))
                chip_table = Table([chip_cells], colWidths=[chip_size, chip_gap] * 4 + [chip_size], rowHeights=[chip_size])
                chip_table.setStyle(TableStyle(chip_styles + [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
                agree_txt = f"<b>{st['lit'] if st['pdir'] else 0} of 5</b> lookbacks agree"
                chips_row = Table([[chip_table, P(f"<font color='#8b949e' size='8'>{agree_txt}</font>", "cell")]],
                                  colWidths=[chip_size * 5 + chip_gap * 4 + 6, (CW / 2 - 32) - (chip_size * 5 + chip_gap * 4 + 6)])
                chips_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                               ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                               ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                               ("LEFTPADDING", (1, 0), (1, 0), 6)]))
                cells, styles_ = [], []
                box_w = (CW / 2 - 32) / 5
                for i, (key, lab) in enumerate(RPT_CHIP_ORDER):
                    L = st["layers"][key]
                    lit_ = bool(st["pdir"]) and L["pdir"] == st["pdir"]
                    cells.append(P(f'<font color="{accent if lit_ else "#8b949e"}" size="7.5"><b>{lab}</b></font>', "cell"))
                    styles_.append(("BACKGROUND", (i, 0), (i, 0), H(tint if lit_ else RPT["grey"])))
                    styles_.append(("BOX", (i, 0), (i, 0), 0.6, H(accent if lit_ else RPT["rule"])))
                ct = Table([cells], colWidths=[box_w] * 5)
                ct.setStyle(TableStyle(styles_ + [("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                                   ("TOPPADDING", (0, 0), (-1, -1), 6),
                                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                                   ("LEFTPADDING", (0, 0), (-1, -1), 1),
                                                   ("RIGHTPADDING", (0, 0), (-1, -1), 1)]))
                inner = [P(f"<font color='#8b949e'><b>{title.upper()}</b></font>", "label"),
                         chips_row, Spacer(1, 5),
                         P(f"<font color='{side_col}'><b>{side_txt}</b></font>", "big"),
                         Spacer(1, 3), ct, Spacer(1, 5),
                         P(f"<font color='{actcol}'><b>{action}</b></font>", "cell")]
                t_ = Table([[inner]], colWidths=[CW / 2 - 4])
                t_.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                        ("BOX", (0, 0), (-1, -1), 0.7, H(RPT["rule"])),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
                return t_

            rdir, cdir = ret["pdir"], comm["pdir"]
            ract = f"FADE the masses  {ARROW}  {RPT_DIRWORD[rdir]}" if rdir else f"FADE the masses  {ARROW}  no read"
            cact = f"FOLLOW the hedgers  {ARROW}  {RPT_DIRWORD[cdir]}" if cdir else f"FOLLOW the hedgers  {ARROW}  no read"
            pair = Table([[role_card("Masses | small traders", ret, "retail", ract, _rpt_dircolor(rdir)),
                           role_card("Hedgers", comm, "commercial", cact, _rpt_dircolor(cdir))]],
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
                sty = [("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                       ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                       ("LINEBELOW", (0, 0), (-1, 0), 0.8, H(RPT["blue"]))]
                best_key = None
                if cp["edge"] and cp["edge"]["role"] == role and h == COT_EDGE_HORIZONS[0]:
                    best_key = cp["edge"]["key"]
                if cp.get("edge8") and cp["edge8"]["role"] == role and h == COT_EDGE_HORIZONS[-1]:
                    best_key = cp["edge8"]["key"]
                for i, (key, lab) in enumerate(RPT_TABLE_ORDER, start=1):
                    tr = cp["tracks"].get((role, key, h))
                    if not tr:
                        rows.append([P(lab, "cell"), P('<font color="#8b949e">not at an extreme</font>', "cell"), "", "", "", ""])
                        sty.append(("SPAN", (1, i), (-1, i)))
                        continue
                    badge = (' <font color="#8b949e" size=7><i>past</i></font>' if tr["past"]
                             else ' <font color="#58a6ff" size=7><b>live</b></font>')
                    if tr["cases"] == 0:
                        rows.append([P(lab + badge, "cell"),
                                     P('<font color="#8b949e">no comparable cases</font>', "cell"), "", "", "", ""])
                        sty.append(("SPAN", (1, i), (-1, i)))
                        sty.append(("BACKGROUND", (0, i), (-1, i), H(RPT["lav"])))
                        continue
                    wcol = "#8b949e" if tr["win"] <= 33 else RPT["ink"]
                    wtxt = f"<b>{tr['win']:.0f}%</b>" if tr["win"] >= 60 else f"{tr['win']:.0f}%"
                    acol = RPT["green"] if tr["avg"] > 0 else RPT["red"]
                    rows.append([P(lab + badge, "cell"), P(f"<b>{tr['weeks_in']}w</b>", "cell"),
                                 P(f'<font color="{wcol}">{wtxt}</font>', "cell"),
                                 P(f"{tr['wins']} of {tr['cases']}", "cell"),
                                 P(colored(f"{tr['avg']:+.2f}%", acol), "cell"),
                                 P(f'<font color="#8b949e">{tr["history"]} yrs</font>', "cell")])
                    sty.append(("BACKGROUND", (0, i), (-1, i), H(RPT["lav"])))
                    if best_key == key:
                        sty.append(("BOX", (0, i), (-1, i), 1.2, H(RPT["blue"])))
                tt = Table(rows, colWidths=[CW / 2 * f for f in (0.20, 0.15, 0.17, 0.16, 0.16, 0.16)])
                tt.setStyle(TableStyle(sty))
                return tt

            def pair_of_tables(h):
                left = [P(f"<b>MASSES | FADE | {h}-WEEK RESULT</b>", "cell"), Spacer(1, 3), track_table("retail", ret, h)]
                right = [P(f"<b>HEDGERS | FOLLOW | {h}-WEEK RESULT</b>", "cell"), Spacer(1, 3), track_table("commercial", comm, h)]
                tt = Table([[left, right]], colWidths=[CW / 2, CW / 2])
                tt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, 0), 0),
                                        ("RIGHTPADDING", (0, 0), (0, 0), 8), ("LEFTPADDING", (1, 0), (1, 0), 8),
                                        ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
                return tt

            story += header(f"{full} · {_rpt_tvlabel(pk)}", "The evidence", REPORT_BRAND, "Engagement Record",
                            "What happened last time this extreme showed up")
            story += section("Engagement record", (f"every past case, capped at the {RPT_TRACK_CASE_CAP} most recent" if RPT_TRACK_CASE_CAP else "every past case"))
            story.append(pair_of_tables(COT_EDGE_HORIZONS[0]))
            story += [Spacer(1, 8), note("<b>Weeks in</b> = how long that lookback's extreme has been running. "
                                         "Rows still inside their window carry a <b>live</b> badge; older rows are marked <b>past</b>."),
                      Spacer(1, 8)]
            story.append(pair_of_tables(COT_EDGE_HORIZONS[-1]))
            story.append(PageBreak())

            story += header(f"{full} · {_rpt_tvlabel(pk)}", "Does it pay?", REPORT_BRAND, "Harvest Profile",
                            "Every case traceable to a row on this page")
            story += section("Harvest profile", "the setups this market has actually paid for")
            ed = cp.get("edge")
            ed8 = cp.get("edge8")
            paras = []
            if ed:
                t4 = ed["t"]; lab = dict(RPT_TABLE_ORDER)[ed["key"]]
                dir_word = "LONG" if bias == 1 else "SHORT"
                if ed["role"] == "commercial":
                    role_who = f"A hedger {'long' if comm['side'] == 1 else 'short'} extreme"
                    role_act = f"has led to a {dir_word}"
                else:
                    role_who = f"A mass {'long' if ret['side'] == 1 else 'short'} extreme, faded"
                    role_act = f"has led to a {dir_word}"
                paras.append(
                    f"<b>The strongest read at 4 weeks is the {lab.lower()} lookback, on {rolename[ed['role']]}.</b> "
                    f"{role_who} {role_act} <b>{t4['win']:.0f}% of the time over {ed['h']} weeks</b>, "
                    f"{t4['wins']} of {t4['cases']} cases, {t4['avg']:+.2f}% average move. "
                    f"That is the highlighted row in the 4-week table below.")
                paras.append(
                    f"<b>The table is the reason this side is on the page.</b> Our bar is "
                    f"{_nw(REPORT_EDGE_MIN_CASES)} cases, {REPORT_EDGE_MIN_WIN}% and a positive average — "
                    f"this row clears it. Read the deepest samples first; they carry the most weight, "
                    f"and the whole table is here so you can check it rather than take our word for it.")
                if ed8:
                    t8_ = ed8["t"]; lab8 = dict(RPT_TABLE_ORDER)[ed8["key"]]
                    same_as_4w = (ed8["role"], ed8["key"]) == (ed["role"], ed["key"])
                    if same_as_4w:
                        paras.append(
                            f"<b>At 8 weeks, the same {lab8.lower()} lookback on {rolename[ed8['role']]} remains the strongest read:</b> "
                            f"{t8_['win']:.0f}% of the time, {t8_['wins']} of {t8_['cases']} cases, "
                            f"{t8_['avg']:+.2f}% average move.")
                    else:
                        if ed8["role"] == "commercial":
                            role_line8 = f"A hedger {'long' if comm['side'] == 1 else 'short'} extreme has led to a {dir_word} "
                        else:
                            role_line8 = f"A mass {'long' if ret['side'] == 1 else 'short'} extreme, faded, has led to a {dir_word} "
                        paras.append(
                            f"<b>At 8 weeks, the strongest read shifts to the {lab8.lower()} lookback on {rolename[ed8['role']]}.</b> "
                            + role_line8
                            + f"<b>{t8_['win']:.0f}% of the time over {ed8['h']} weeks</b>, "
                              f"{t8_['wins']} of {t8_['cases']} cases, {t8_['avg']:+.2f}% average move. "
                              f"That is the highlighted row in the 8-week table below.")

                above_floor = []
                for (role_k, key_k, h_k), tr_k in cp["tracks"].items():
                    if h_k != COT_EDGE_HORIZONS[0] or not tr_k or tr_k["past"]:
                        continue
                    if role_k == ed["role"] and key_k == ed["key"]:
                        continue
                    if (tr_k["cases"] >= REPORT_EDGE_MIN_CASES
                            and tr_k["win"] >= REPORT_EDGE_MIN_WIN and tr_k["avg"] > 0):
                        above_floor.append((role_k, key_k, tr_k))
                if above_floor:
                    parts = []
                    for role_k, key_k, tr_k in above_floor[:3]:
                        lab_k = dict(RPT_TABLE_ORDER)[key_k]
                        parts.append(f"the {lab_k} on {rolename[role_k]} reads <b>{tr_k['win']:.0f}%</b>, "
                                     f"{tr_k['wins']} of {tr_k['cases']} cases")
                    paras.append(f"<b>{_nw(len(above_floor)).capitalize()} more 4-week lookback"
                                 f"{'s' if len(above_floor) != 1 else ''} also clear the bar.</b> "
                                 + "; ".join(parts) + ".")

                subfloor = []
                for (role_k, key_k, h_k), tr_k in cp["tracks"].items():
                    if h_k != COT_EDGE_HORIZONS[0] or not tr_k or tr_k["past"]:
                        continue
                    if role_k == ed["role"] and key_k == ed["key"]:
                        continue
                    if (2 <= tr_k["cases"] < REPORT_EDGE_MIN_CASES
                            and tr_k["win"] >= REPORT_EDGE_MIN_WIN and tr_k["avg"] > 0):
                        subfloor.append((role_k, key_k, tr_k))
                if subfloor:
                    sub_n = len(subfloor)
                    sub_nw = _nw(sub_n)
                    sub_desc = "; ".join(
                        f"the {dict(RPT_TABLE_ORDER)[key_k]} on {rolename[role_k]} reads "
                        f"<b>{tr_k['win']:.0f}%</b>, {tr_k['wins']} of {tr_k['cases']} cases"
                        for role_k, key_k, tr_k in subfloor[:3])
                    verb_all = "both are" if sub_n == 2 else ("all are" if sub_n > 2 else "is")
                    pronoun = "Both are" if sub_n == 2 else ("All are" if sub_n > 2 else "It is")
                    paras.append(
                        f"<b>{sub_nw.capitalize()} separate 4-week lookback"
                        f"{'s' if sub_n != 1 else ''} say{'s' if sub_n == 1 else ''} the same thing, "
                        f"and {verb_all} below the case floor.</b> "
                        f"{sub_desc[0].upper() + sub_desc[1:]}. "
                        f"{pronoun} in the table below.")
                    paras.append(
                        f"<b>No rule may point at any of these</b>, each one is too thin on its own, "
                        f"and thin evidence lies in the direction that flatters you. But "
                        f"{sub_nw} lookback{'s' if sub_n != 1 else ''} agreeing is a stronger statement than one, "
                        f"and pretending they are not there would be its own kind of dishonesty. "
                        f"They are all on the table below. If price reaches a good zone in that direction, "
                        f"they add to the trade. Step 1 does not have to carry a trade on its own.")
            else:
                subfloor_any = []
                for (role_k, key_k, h_k), tr_k in cp["tracks"].items():
                    if h_k != COT_EDGE_HORIZONS[0] or not tr_k or tr_k["past"]:
                        continue
                    if (2 <= tr_k["cases"] and tr_k["win"] >= REPORT_EDGE_MIN_WIN and tr_k["avg"] > 0):
                        subfloor_any.append((role_k, key_k, tr_k))
                if cp["bias"] == 0:
                    _nl = _rpt_lit_layers(comm) + _rpt_lit_layers(ret)
                    if _nl:
                        paras.append(
                            f"The positioning read is neutral this week, so there is no direction to test, "
                            f"<b>and that is not because the page is empty.</b> {_nw(_nl)} lookback"
                            f"{'s are' if _nl != 1 else ' is'} lit, and neither group's treatment qualified on this market.")
                        paras.append(
                            f"<b>The table below is the reason.</b> When this extreme has appeared before in {_esc(full)} "
                            f"it has not reliably paid, or the horizons pointed in different directions. "
                            f"Read the deepest samples first: that is what a stretched chart the history does not support "
                            f"actually looks like. Our bar is {_nw(REPORT_EDGE_MIN_CASES)} cases, "
                            f"{REPORT_EDGE_MIN_WIN}% and a positive average, and the whole table is here so you can "
                            f"check it rather than take our word for it.")
                    else:
                        paras.append(
                            "The positioning read is neutral this week, so there is no direction to test. "
                            "No lookback is lit, so there is nothing to put on trial.")
                else:
                    paras.append(
                        f"<b>No single row clears our bar at 4 weeks this week.</b> A row needs at least "
                        f"{_nw(REPORT_EDGE_MIN_CASES)} cases, {REPORT_EDGE_MIN_WIN}% win rate, a positive average, "
                        f"and to be live.")
                if subfloor_any:
                    sub_n = len(subfloor_any)
                    sub_nw = _nw(sub_n)
                    sub_desc = "; ".join(
                        f"the {dict(RPT_TABLE_ORDER)[key_k]} on {rolename[role_k]} reads "
                        f"<b>{tr_k['win']:.0f}%</b>, {tr_k['wins']} of {tr_k['cases']} cases"
                        for role_k, key_k, tr_k in subfloor_any[:3])
                    verb_all = "both are" if sub_n == 2 else ("all are" if sub_n > 2 else "is")
                    pronoun = "Both are" if sub_n == 2 else ("All are" if sub_n > 2 else "It is")
                    paras.append(
                        f"<b>{sub_nw.capitalize()} 4-week lookback{'s' if sub_n != 1 else ''} still lean the same way, "
                        f"and {verb_all} below the case floor.</b> {sub_desc[0].upper() + sub_desc[1:]}. "
                        f"{pronoun} in the table below.")
                    paras.append(
                        f"<b>None of these may point at a trade on their own</b>, each one is too thin, "
                        f"and thin evidence lies in the direction that flatters you. But {sub_nw} lookback"
                        f"{'s' if sub_n != 1 else ''} leaning the same way is a stronger statement than none, "
                        f"and pretending they are not there would be its own kind of dishonesty. "
                        f"They are all on the table below. If price reaches a good zone in that direction, "
                        f"they add to the trade. Layer 1 does not have to carry a trade on its own.")
                if ed8:
                    t8_ = ed8["t"]; lab8 = dict(RPT_TABLE_ORDER)[ed8["key"]]
                    paras.append(
                        f"<b>At 8 weeks, however, the {lab8.lower()} lookback on {rolename[ed8['role']]} does clear the bar:</b> "
                        f"{t8_['win']:.0f}% of the time, {t8_['wins']} of {t8_['cases']} cases, {t8_['avg']:+.2f}% average move. "
                        f"Only the 4-week table is empty; the 8-week evidence is on the page.")
            story.append(card([P("<font color='#2ea043'><b>HARVEST PROFILE — THE STRONGEST READ IN THIS MARKET</b></font>", "label")]
                              + [P(p_) for p_ in paras], RPT["green"] if (ed or ed8) else RPT["muted"]))
            story += section("Where this number comes from", f"Engagement Record at {COT_EDGE_HORIZONS[0]} weeks")
            story.append(pair_of_tables(COT_EDGE_HORIZONS[0]))
            if cp["bias"] == 0 and not ed:
                story.append(Spacer(1, 6))
                story.append(card([
                    P("<font color='#58a6ff'><b>WHY THESE ROWS AND NOT A HARVEST PROFILE NUMBER</b></font>", "label"),
                    P(f"The Harvest Profile tests one fixed horizon per market, and that horizon is only chosen once a "
                      f"direction exists to test. With no direction there is none to choose, so the table below is the "
                      f"{COT_EDGE_HORIZONS[0]}-week Engagement Record — the same rows the engine actually read when it "
                      f"decided neither treatment qualified."),
                    P(f"A row is only eligible if it has at least <b>{_nw(REPORT_EDGE_MIN_CASES)} cases</b>, a win rate of "
                      f"at least <b>{REPORT_EDGE_MIN_WIN}%</b>, and a <b>positive average move</b>. "
                      f"Read the deepest samples first: they carry the most weight and they are the reason there is no side here."),
                ], RPT["blue"]))
            story.append(PageBreak())

            story += header(f"{full} · {_rpt_tvlabel(pk)}", "Coordinated bias — the verdict", REPORT_BRAND,
                            "Flow positioning + calendar flow", "the conclusion of Layer 1")
            story += section("The two engines", "each read on its own, before they are weighed")
            cot_dir_word = {1: "LONG", -1: "SHORT", 0: "STAND ASIDE"}[cp["bias"]]
            cot_col = _rpt_dircolor(cp["bias"])
            cot_conv = ("High" if cp["lit"] >= 7 and not cp["conflict"]
                        else "Moderate" if cp["lit"] >= 4 else "None")
            sea_dir_word = "NEUTRAL"; sea_col = RPT["muted"]; sea_conv = "None"; sea_win = 0.05
            if sp and sp.get("cancelled"):
                sea_dir_word = "CANCELS"
                sea_col = RPT["muted"]
                sea_conv = "opposing windows · no side"
                sea_win = 0.05
            elif sp and sp.get("main"):
                m2 = sp["main"]
                sea_dir_word = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}[m2["dir"]]
                sea_col = _rpt_dircolor(m2["dir"])
                _disp_strength, _ = _season_display_strength(sp)
                sea_conv = f"{_disp_strength} · {m2['shape']} window"
                sea_win = m2["win"]
            def _engine_box(title_txt, dir_txt, colr, conv_txt, bar_frac):
                bar_cols = 20
                filled = max(1, int(round(bar_frac * bar_cols))) if dir_txt not in ("NEUTRAL", "STAND ASIDE") else 1
                cells = [""] * bar_cols
                bar = Table([cells], colWidths=[(CW / 2 - 44) / bar_cols] * bar_cols, rowHeights=[8])
                bst = []
                for i in range(bar_cols):
                    bst.append(("BACKGROUND", (i, 0), (i, 0), H(colr if i < filled else RPT["grey"])))
                bar.setStyle(TableStyle(bst + [("LEFTPADDING", (0, 0), (-1, -1), 0),
                                               ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                               ("TOPPADDING", (0, 0), (-1, -1), 0),
                                               ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
                return [P(f"<font color='#8b949e'><b>{title_txt}</b></font>", "label"),
                        P(f"<font color='{colr}' size=22><b>{dir_txt}</b></font>", "h"),
                        bar, Spacer(1, 3),
                        P(f"<font color='#8b949e'>Conviction: <b>{conv_txt}</b></font>", "cell")]
            cot_inner = _engine_box("FLOW POSITIONING — THE DRIVER", cot_dir_word, cot_col, cot_conv,
                                    (cp["lit"] / 5.0) if cp["bias"] else 0.05)
            sea_inner = _engine_box("CALENDAR FLOW — THE ADD-ON", sea_dir_word, sea_col, sea_conv, sea_win)
            cot_box = Table([[cot_inner]], colWidths=[CW / 2 - 6])
            cot_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                         ("BOX", (0, 0), (-1, -1), 0.7, H(RPT["rule"])),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                         ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
            sea_box = Table([[sea_inner]], colWidths=[CW / 2 - 6])
            sea_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                         ("BOX", (0, 0), (-1, -1), 0.7, H(RPT["rule"])),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                         ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
            pair2 = Table([[cot_box, sea_box]], colWidths=[CW / 2, CW / 2])
            pair2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 4),
                                       ("LEFTPADDING", (1, 0), (1, 0), 4), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
            story.append(pair2)
            story.append(Spacer(1, 8))
            story += section("How they line up", "the rule")
            if sp and sp.get("cancelled"):
                line1 = ("<b>FLOW HAS A SIDE, CALENDAR CANCELS</b><br/>" if cp["bias"] else
                         "<b>NEITHER ENGINE HAS A SIDE — CALENDAR CANCELS</b><br/>")
                line1 += ("Two opposing calendar windows of similar strength sit in the same horizon, so "
                          "seasonality adds nothing this week. "
                          + ("Flow carries the side." if cp["bias"] else
                             "The scanner stands dormant. A neutral read is not an excuse to trade the middle."))
            elif cp["bias"] and sp and sp.get("main") and sp["main"]["dir"] == cp["bias"]:
                line1 = (f"<b>FLOW AND CALENDAR AGREE</b><br/>Both engines point {RPT_DIRWORD[cp['bias']].lower()}. "
                         f"The calendar window adds timing to a flow-led side.")
            elif cp["bias"] and sp and sp.get("main") and sp["main"]["dir"] == -cp["bias"]:
                line1 = (f"<b>THE ENGINES DISAGREE</b><br/>Flow reads {RPT_DIRWORD[cp['bias']].lower()}, "
                         f"calendar flow reads {RPT_DIRWORD[sp['main']['dir']].lower()}. Flow carries the side; "
                         f"the calendar opponent tempers conviction.")
            elif cp["bias"]:
                line1 = ("<b>FLOW HAS A SIDE, CALENDAR IS SILENT</b><br/>No calendar window is in play in the "
                         "coming weeks, so the bias rests on positioning alone.")
            elif sp and sp.get("main") and sp["main"]["strength"] == "STRONG" and sp.get("tier", "FULL") != "DROP":
                line1 = ("<b>FLOW IS NEUTRAL, CALENDAR IS THE ONLY READ</b><br/>Positioning is neutral. The calendar "
                         "window is the only read on the table, and it should be sized as a single-engine call.")
            else:
                line1 = ("<b>NEITHER ENGINE HAS A SIDE</b><br/>The scanner stands dormant. A neutral read is not an "
                         "excuse to trade the middle.")
            story.append(card([P(line1)], RPT["blue"]))
            story.append(Spacer(1, 8))

            try:
                fnd = (cp or {}).get("funds") if isinstance(cp, dict) else None
                row_state = str((r.get("NonCommState") if r is not None else None) or "n/a")
                row_dir = _safe_int(r.get("NonCommDir") if r is not None else 0)
                row_detail = str((r.get("NonCommDetail") if r is not None else None) or "-")
                row_fuel = str((r.get("NonCommFuel") if r is not None else None) or "-")
                row_shift = _safe_int(r.get("NonCommShift") if r is not None else 0)

                if not isinstance(fnd, dict) or str(fnd.get("state", "n/a")) in ("n/a", "None", ""):
                    fnd = {
                        "state": row_state,
                        "dir": row_dir,
                        "detail": row_detail,
                        "fuel": row_fuel,
                        "shift_conditions": row_shift,
                        "shift_setup": bool(row_shift >= 3),
                        "unwinding": False,
                        "idx52": float("nan"),
                        "chg4": float("nan"),
                    }

                story += section("Non-Commercials (Funds / Trend Followers)",
                                 "divergence · fuel · trend-shift confirmation")
                nc_state = str(fnd.get("state", "n/a") or "n/a")
                if nc_state in ("n/a", "None", ""):
                    story.append(card([
                        P("<font color='#8b949e'><b>SPECULATORS | NON-COMMERCIALS</b></font>", "label"),
                        Spacer(1, 3),
                        P("Not enough positioning or price history this week to read non-commercial "
                          "divergence or follow-through.", "body"),
                    ], RPT.get("muted")))
                else:
                    f_dir = int(fnd.get("dir") or 0)
                    f_col = _rpt_dircolor(f_dir) if f_dir else RPT.get("muted")
                    f_word = {
                        "Bearish divergence": "BEARISH DIVERGENCE",
                        "Bullish divergence": "BULLISH DIVERGENCE",
                        "Follow-through": "FOLLOW-THROUGH",
                        "Neutral": "NO SIGNAL",
                    }.get(nc_state, nc_state.upper())

                    who = ("Hedge funds and large asset managers - capital deployers that follow trends "
                           "and add as trends strengthen, until they reach relative extremes and may "
                           "run out of fuel.")
                    blueprint_read = {
                        "Bearish divergence": (
                            "Price made a new high but non-commercials did not confirm it in net "
                            "positioning - they are taking profit into rising prices. The up-trend is "
                            "more likely to stall or turn."),
                        "Bullish divergence": (
                            "Price made a new low but non-commercials did not confirm it in net "
                            "positioning - they are covering into declining prices. The down-trend is "
                            "more likely to stall or turn."),
                        "Follow-through": (
                            "Price and non-commercial net moved together. Speculators are still adding "
                            "in the direction of the move, so the trend still has fuel."),
                        "Neutral": (
                            "Price is not at a multi-week extreme, so there is nothing for non-commercials "
                            "to confirm or diverge from this week."),
                    }.get(nc_state, "")

                    detail = str(fnd.get("detail") or "-")
                    f_lines = [
                        P(f"<font color='{RPT.get('muted')}' size=8><b>SPECULATORS | NON-COMMERCIALS</b></font>  "
                          f"<font color='{f_col}'><b>{_esc(f_word)}</b></font>", "label"),
                        Spacer(1, 4),
                        P(_esc(who), "body"),
                        Spacer(1, 4),
                        P(f"<b>This week:</b> {_esc(detail)}", "body"),
                    ]
                    if blueprint_read:
                        f_lines.append(P(f"<b>Blueprint read:</b> {_esc(blueprint_read)}", "body"))

                    extras = []
                    fuel = str(fnd.get("fuel") or "-")
                    if fuel not in ("-", "n/a", "None", ""):
                        extras.append(f"{fuel} (1-year COT index)")
                    try:
                        idx52 = float(fnd.get("idx52"))
                        if idx52 == idx52:
                            extras.append(f"1-year percentile rank: {idx52:.0f}")
                    except Exception:
                        pass
                    try:
                        chg4 = float(fnd.get("chg4"))
                        if chg4 == chg4:
                            extras.append(f"4-week net change: {chg4:+.0f} contracts")
                    except Exception:
                        pass
                    if fnd.get("unwinding"):
                        extras.append("unwinding: trimmed position over the last 4 weeks")
                    shift_n = int(fnd.get("shift_conditions") or 0)
                    if fnd.get("shift_setup") or shift_n >= 3:
                        extras.append(
                            "TREND-SHIFT SET-UP active: hedgers at an extreme, "
                            "speculators trimming the old trend, masses on the wrong end")
                    elif shift_n == 2:
                        extras.append("2 of 3 trend-shift conditions present")
                    elif shift_n == 1:
                        extras.append("1 of 3 trend-shift conditions present")
                    if extras:
                        f_lines.append(P("<b>Also:</b> " + _esc("; ".join(extras)) + ".", "body"))

                    if bias and f_dir:
                        if f_dir == bias:
                            align = ("supports the flow side - non-commercials are aligned with the "
                                     "coordinated bias this week.")
                        else:
                            align = ("works against the flow side - treat as a caution layer, not a "
                                     "standalone reversal.")
                        f_lines.append(P(f"<b>Vs coordinated bias:</b> {_esc(align)}", "body"))

                    story.append(card(f_lines, f_col))
                story.append(Spacer(1, 8))
            except Exception as e:
                log.warning("Non-Commercials section failed for %s: %s", nm, e)
                try:
                    story.append(Spacer(1, 4))
                except Exception:
                    pass

            story += section("The combined bias", "the conclusion of Layer 1")
            essay = _rpt_combined_essay(pk, cp, sp, r, bias, nm, packs, prior_state=prior_state)
            from reportlab.lib.styles import ParagraphStyle as _PS2
            from reportlab.lib.enums import TA_RIGHT as _TAR2
            _vbody = _PS2("vbody", parent=S_["body"], fontSize=11, leading=16.5)
            _vhead = _PS2("vhead", parent=S_["h"], fontSize=19, leading=23, textColor=H(RPT["muted"]))
            _vdir = _PS2("vdir2", parent=S_["h"], fontSize=26, leading=30, alignment=_TAR2)
            def _verdict_card(paras):
                head = Table([[Paragraph("<b>Coordinated verdict</b>", _vhead),
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
                                         ("LINEABOVE", (0, 0), (-1, 0), 0.7, H(RPT["rule"])),
                                         ("LINEBELOW", (0, -1), (-1, -1), 0.7, H(RPT["rule"])),
                                         ("LINEAFTER", (-1, 0), (-1, -1), 0.7, H(RPT["rule"]))]))
                return vt_
            closing = ""
            if bias:
                _w = {1: "buy", -1: "sell"}[bias]; _d = {1: "Demand", -1: "Supply"}[bias]
                _m = sp["main"] if sp and sp.get("main") else None
                _dl = 0
                _cal_led = bool(_m and _m["dir"] == bias and not (cp and cp.get("bias")))
                if _m and (_m["dir"] == -bias or _cal_led):
                    try:
                        _dl = max(0, (pd.Timestamp(_m["end"]) - pd.Timestamp(sp["today"])).days)
                    except Exception:
                        _dl = 0
                if _cal_led and _dl > 0:
                    _end_txt = pd.Timestamp(_m["end"]).strftime("%d %B").lstrip("0")
                    closing = (f"So work the {_nw(_dl)} day{'s' if _dl != 1 else ''} and no longer. If price has not reached a fresh "
                               f"{_d} by {_end_txt}, let it go rather than chasing it once the window has closed. "
                               f"We look to {_w} in {_d} only. A bias is not a trade.")
                else:
                    closing = "So work the four to eight week horizon, "
                    if _dl > 0:
                        closing += (f"treat the next {_nw(_dl)} day{'s' if _dl != 1 else ''} as a reason to be patient on entry "
                                    f"rather than absent, and let the reference decide. ")
                    else:
                        closing += "and let the reference decide. "
                    closing += f"We look to {_w} in {_d} only, and only when price reaches a live mapped reference. A bias is not a trade."
            _disc = ("Calendar flow is an analytical add-on, not a signal. A bias is not a trade, wait for price to reach "
                     "the reference. Educational content only. Not financial advice.")
            _kick = f"{title_pair} · {_rpt_tvlabel(pk)}"
            _today_txt = today.strftime("%d %B %Y").lstrip("0")
            FH_ = PH - 25 * mm; _SAFE = 22
            def _hh(fl):
                try:
                    return fl.wrap(CW, 10000)[1]
                except Exception:
                    return 0
            _k = len(story) - 1
            while _k >= 0 and not isinstance(story[_k], PageBreak):
                _k -= 1
            used_first = sum(_hh(f_) for f_ in story[_k + 1:])
            cont_top = header(_kick, "Coordinated bias — the verdict", "Bias · the conclusion", "continued", _today_txt) + \
                section("The coordinated bias", "continued, how to read this week's pages")
            used_cont = sum(_hh(f_) for f_ in cont_top)
            _head_h = 62; _pw = CW - 22 - 18 - 6
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
            _cancelled = bool(sp.get("cancelled"))
            sw = "bullish" if sdir == 1 else "bearish"
            scol = RPT["muted"] if _cancelled else _rpt_dircolor(sdir)
            in_play_local = sp["today"] >= m["start"] and sp["today"] <= m["end"]
            _disp_strength_label, _disp_tier = _season_display_strength(sp)
            _ensure_mkt_bookmark()
            story.append(SetSection(f"{full} · Calendar flow", bookmark=False))
            story += header(f"{title_pair} · {_rpt_tvlabel(pk)}", "Calendar flow — the horizon", REPORT_BRAND,
                            "Add-on to the flow read", today.strftime("%d %B %Y").lstrip("0"))
            if _cancelled:
                _opp = (sp.get("best") or {}).get(-int(sdir))
                _opp_line = ""
                if _opp:
                    _opp_line = (f" A {_esc('buy' if _opp['dir'] == 1 else 'sell')} from "
                                 f"{_opp['start'].strftime('%d %b')} and a {_esc('buy' if sdir == 1 else 'sell')} from "
                                 f"{m['start'].strftime('%d %b')} open within days of each other on similar records.")
                story.append(card([P(f"<b>{_esc(full)}</b>   <font color='{RPT['muted']}'><b>CANCELS</b></font>", "h"),
                                   P(f"The calendar <b>cancels itself out</b> this week.{_opp_line} "
                                     f"Neither outranks the other by enough to call a direction, so seasonality "
                                     f"adds nothing here — the flow read carries this market.")], RPT["muted"]))
            else:
                story.append(card([P(f"<b>{_esc(full)}</b>   <font color='{scol}'><b>{sw.upper()}</b></font>", "h"),
                                   P(f"Calendar flow points <b>{sw}</b> over the coming weeks, on a <b>{m['shape']}</b> window "
                                     f"({m['start'].strftime('%d %b')} – {m['end'].strftime('%d %b')}). It closed in its direction "
                                     f"<b>{m['win'] * 100:.0f}%</b> of the last {m['cases']} years and has hit "
                                     f"<b>{m['rec_wins']} of its last {m['rec_cases']}</b>, for an average of "
                                     f"<b>{m['avg']:+.1f}%</b>. Edge: <b>{_disp_strength_label}</b> · timing {timing_display(m['timing'])}.")], scol))
            story += section("Calendar horizon", "-2 to +8 weeks · every window in play, drawn to scale")
            story.append(img(_rpt_strip_chart(sp)))
            story += section("Seasonal map", "5 / 10 / 15-year average path")
            story.append(img(_rpt_map_chart(sp), width=CW * 0.80))
            story.append(PageBreak())
            story += header(f"{title_pair} · {_rpt_tvlabel(pk)}", "Calendar flow — the evidence", REPORT_BRAND,
                            "Add-on to the flow read", today.strftime("%d %B %Y").lstrip("0"))
            story += section("Windows in play", "every calendar window entering the -2 to +8 week horizon")
            rows = [[P(x, "label") for x in ("ENTRY", "TRADE", "WINDOW", "LENGTH", "WIN", "CASES", "AVG")]]
            for dirn, b in sorted(sp["best"].items(), key=lambda kv: kv[1]["start"]):
                tag = ' <font color="#58a6ff" size=7>driver</font>' if b is m else ""
                rows.append([P(f"<b>{b['start'].strftime('%d %b')}</b>{tag}", "cell"),
                             P(colored("LONG" if dirn == 1 else "SHORT", _rpt_dircolor(dirn)), "cell"),
                             P(f"{b['start'].strftime('%d %b')} – {b['end'].strftime('%d %b')}", "cell"),
                             P(f"{b['len']}d · {b['shape']}", "cell"),
                             P(f"<b>{b['win'] * 100:.0f}%</b>", "cell"),
                             P(f"{b['wins']} of {b['cases']}", "cell"),
                             P(f"{b['avg']:+.1f}%", "cell")])
            t = Table(rows, colWidths=[CW * w for w in (0.14, 0.10, 0.22, 0.16, 0.12, 0.13, 0.13)])
            t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                                   ("BACKGROUND", (0, 1), (-1, -1), H(RPT["lav"])),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            story += [t, Spacer(1, 6),
                      note("<b>Sharp beats broad.</b> A <b>sharp</b> window (about three weeks or less) packs its move into a few dates, "
                           "so a high hit-rate means more. A <b>mid-length</b> window (up to about six weeks) is a looser edge. "
                           "A <b>broad</b> window drifting across two months is the weakest: the same hit-rate spread thin. "
                           "The <b>driver</b> row is the one carrying this market's calendar read.")]
            story += section("How strong is this window?", "edge · and how forgiving the timing is")
            shape_txt = "not available" if not np.isfinite(sp["shape"]) else f"{sp['shape']:+.2f}"
            str_col = RPT["blue"] if m["strength"] == "STRONG" else RPT["amber"]
            bar_cols = 16
            filled = 14 if m["strength"] == "STRONG" else (10 if m["strength"] == "MODERATE" else 5)
            bar_cells = [""] * bar_cols
            edge_bar = Table([bar_cells], colWidths=[(CW / 2 - 50) / bar_cols] * bar_cols, rowHeights=[9])
            ebst = [("BACKGROUND", (i, 0), (i, 0), H(str_col if i < filled else RPT["grey"])) for i in range(bar_cols)]
            edge_bar.setStyle(TableStyle(ebst + [("LEFTPADDING", (0, 0), (-1, -1), 0),
                                                  ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                                  ("TOPPADDING", (0, 0), (-1, -1), 0),
                                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
            timing_label = timing_display(m["timing"])
            timing_sub = ("most of the edge sits in these dates, timing matters" if m["timing"] == "tight"
                          else "clearly better than chance around these dates" if m["timing"] == "forgiving"
                          else "the edge barely survives any shift; timing is critical" if m["timing"] == "very tight"
                          else "room on either side of these dates")
            left_inner = [P("<font color='#8b949e'><b>EDGE — HOW GOOD IS IT</b></font>", "label"),
                          P(f"<font color='{str_col}' size=18><b>{_disp_strength_label}</b></font>", "h"),
                          edge_bar, Spacer(1, 8),
                          P("<font color='#8b949e'><b>TIMING — HOW FORGIVING</b></font>", "label"),
                          P(f"<font color='#e6edf3' size=14><b>{timing_label}</b></font>", "h"),
                          P(f"<font color='#8b949e' size=8>{timing_sub}</font>", "cell"),
                          Spacer(1, 3),
                          P(f"<font color='#8b949e' size=7.5>Whole-year shape match {shape_txt}</font>", "cell")]
            left_box = Table([[left_inner]], colWidths=[CW / 2 - 8])
            left_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                          ("BOX", (0, 0), (-1, -1), 0.7, H(RPT["rule"])),
                                          ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                          ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
            if _cancelled:
                _opp = (sp.get("best") or {}).get(-int(sdir))
                if _opp:
                    verdict = (f"<b>The calendar cancels itself out this week.</b> A "
                               f"{'buy' if m['dir'] == 1 else 'sell'} from {m['start'].strftime('%d %b')} and a "
                               f"{'buy' if _opp['dir'] == 1 else 'sell'} from {_opp['start'].strftime('%d %b')} "
                               f"open within days of each other on similar records "
                               f"({m['win']*100:.0f}% vs {_opp['win']*100:.0f}%). "
                               f"Neither outranks the other by enough to call a direction, so seasonality "
                               f"adds nothing here — the flow read carries this market.")
                else:
                    verdict = ("<b>The calendar cancels itself out this week.</b> Opposing windows of similar "
                               "strength sit in the same horizon, so seasonality adds nothing — the flow read carries this market.")
            else:
                verdict = (f"A <b>{'strong, ' if m['strength'] == 'STRONG' else 'moderate, '}"
                           f"{'still-firing ' if in_play_local else 'upcoming '}</b>calendar window: closed in its direction "
                           f"<b>{m['win'] * 100:.0f}%</b> of the last {m['cases']} years, averages <b>{m['avg']:+.1f}%</b>, "
                           f"and has hit <b>{m['rec_wins']} of its last {m['rec_cases']}</b>. "
                           f"{'Weight it.' if m['strength'] == 'STRONG' else 'Note it.'} "
                           f"Timing is <b>{timing_label}</b>.")
            right_inner = [P("<font color='#8b949e'><b>VERDICT</b></font>", "label"), P(verdict)]
            right_box = Table([[right_inner]], colWidths=[CW / 2 - 8])
            right_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), H(RPT["card"])),
                                           ("BOX", (0, 0), (-1, -1), 0.7, H(RPT["rule"])),
                                           ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                           ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
            strength_pair = Table([[left_box, right_box]], colWidths=[CW / 2, CW / 2])
            strength_pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                               ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 4),
                                               ("LEFTPADDING", (1, 0), (1, 0), 4), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
            story.append(strength_pair)
            story.append(Spacer(1, 8))
            story += section("The neighbour test", "how tight is the timing on these dates?")
            nb = m["neighbour"]
            nb_label = nb_label_for(m["timing"])
            nb_txt = (f"The surrounding weeks are clearly better than chance." if m["timing"] == "forgiving"
                      else f"Most of the move sits inside these dates, so timing matters here." if m["timing"] == "tight"
                      else f"The edge barely survives any shift; you have to be on these dates or you are trading noise." if m["timing"] == "very tight"
                      else f"The surrounding weeks are only just better than an even split.")
            story.append(card([P(f"<font color='#e6edf3' size=14><b>{nb * 100:.0f}% average · {nb_label}</b></font>", "big")
                               if np.isfinite(nb) else P("not available"),
                               P(f"Randomly chosen windows land around 49% here, no directional lean. {nb_txt}"),
                               P(f"The same window shifted ±1 and ±2 weeks hits on average <b>{nb * 100:.0f}%</b> "
                                 f"against <b>{m['win'] * 100:.0f}%</b> on the actual dates. "
                                 f"Whole-year shape match: <b>{shape_txt}</b>.")], None))
            story.append(PageBreak())
            analog_buf, analog_top = _rpt_analog_chart(sp)
            if analog_buf is not None and analog_top:
                story.append(SetSection(f"{full} · Analogues", bookmark=False))
                story += header(f"{title_pair} · {_rpt_tvlabel(pk)}", "Calendar flow — the analog cross-check", REPORT_BRAND,
                                "Cross-check", today.strftime("%d %B %Y").lstrip("0"))
                story += section("Analog years", "the historical years whose YTD path most resembles this one")
                story.append(img(analog_buf, width=CW * 0.86))
                ups = sum(1 for _, _, _, _, rest in analog_top if rest > 0)
                downs = len(analog_top) - ups
                avg_rest = float(np.mean([rest for _, _, _, _, rest in analog_top]))
                count_leaning_up = ups > downs
                count_leaning_dn = downs > ups
                count_disagrees = (count_leaning_up and avg_rest < 0) or (count_leaning_dn and avg_rest > 0)
                supports = bool(m) and np.sign(avg_rest) == np.sign(m["avg"])
                if count_disagrees:
                    tail = (" The average and the count disagree, so weigh this as a soft cross-check only.")
                elif supports:
                    tail = " This supports the calendar window above."
                else:
                    tail = " This cuts against the calendar window above."
                rows = [[P(x, "label") for x in ("YEAR", "MATCH", "REST OF YEAR")]]
                for corr, y, _full, _cut, rest in analog_top:
                    rc = RPT["green"] if rest > 0 else RPT["red"]
                    rows.append([P(f"<b>{y}</b>", "cell"), P(f"{corr:+.2f}", "cell"),
                                 P(f'<font color="{rc}"><b>{rest:+.1f}%</b></font>', "cell")])
                t = Table(rows, colWidths=[CW * 0.10, CW * 0.10, CW * 0.14])
                t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                                       ("BACKGROUND", (0, 0), (-1, 0), H(RPT["lav"])),
                                       ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story.append(Spacer(1, 6))
                read_txt = (f"<b>THE READ.</b> Of the {len(analog_top)} closest analogues, "
                            f"{ups} finished higher and {downs} lower, averaging "
                            f"<b>{avg_rest:+.1f}%</b>.{tail}")
                story.append(Table([[t, P(read_txt, "body")]],
                                    colWidths=[CW * 0.36, CW * 0.64],
                                    style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                      ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                                      ("RIGHTPADDING", (0, 0), (-1, -1), 0)])))
                story.append(PageBreak())
            vbuf, vcur, vtyp = _rpt_vol_chart(sp)
            if vbuf is not None:
                story += header(f"{title_pair} · {_rpt_tvlabel(pk)}", "Calendar flow & volatility", REPORT_BRAND,
                                "Add-on to the flow read", today.strftime("%d %B %Y").lstrip("0"))
                story += section("Volatility season", f"when {full} gets loud")
                story.append(img(vbuf))
                state_txt = "loud" if vcur > vtyp * 1.25 else "calm" if vcur < vtyp * 0.8 else "normal"
                story.append(Spacer(1, 4))
                story.append(P("Volatility seasonality shows when this market tends to get loud or calm through the year.", "note"))
                story.append(Spacer(1, 6))
                story.append(card([P("<font color='#58a6ff'><b>USING THIS WITH THE WINDOW ABOVE</b></font>", "label"),
                                   P(f"Right now volatility is <b>{vcur:.1f}%</b> against a typical <b>{vtyp:.1f}%</b> "
                                     f"for this time of year, which is <b>{state_txt}</b>."),
                                   P("Volatility does not change the direction.")], RPT["blue"]))
                story.append(PageBreak())

        item = (chart_data or {}).get(nm)
        if item:
            _ensure_mkt_bookmark()
            story.append(SetSection(f"{full} · Layer 2", bookmark=False))
            story += header(f"{title_pair} · {_rpt_tvlabel(item)}", "Layer 2 — precision mapping",
                            REPORT_BRAND, "Proprietary supply & demand",
                            "Monthly for reference · Weekly is HTF · Daily is LTF")
            if bias:
                s2 = (f"Layer 1 gave us the side: <b>{RPT_DIRWORD[bias]}</b>. We look to {word} in "
                      f"<b>{demand}</b> only, so we wait for price to reach a live mapped {demand} reference. "
                      f"A bias is not a trade.")
                if "zones" not in item:
                    item["zones"] = compute_sd_zones(item["df"])
                near = _sd_nearest_fresh(item["zones"], float(item["df"]["Close"].iloc[-1]), bias)
                dd = _px_dec(item["df"]["Close"].iloc[-1])
                if near:
                    nz_, dist_ = near
                    s2 += (f" Nearest live {demand} reference: <b>{_esc(nz_['setup'])}</b>, "
                           f"{nz_['bottom']:.{dd}f} to {nz_['top']:.{dd}f}, "
                           f"{dist_:.1f}% {'below' if bias == 1 else 'above'} price.")
                else:
                    s2 += f" There is no live {demand} reference on the daily chart right now."
                story.append(card([P(s2)], _rpt_dircolor(bias)))
                story.append(Spacer(1, 6))
            else:
                story.append(card([P(
                    "Layer 1 gave us <b>no side</b> this week. Stand dormant on references until flow or a strong "
                    "calendar window gives a direction.")], RPT["muted"]))
                story.append(Spacer(1, 6))
            for tf, subtitle, note_txt in (
                ("M", "Monthly — reference only, where are we in the big picture?",
                 "Monthly is context. We do not engage on monthly references."),
                ("W", "Weekly, HTF — is this a reference worth trading into?",
                 "Weekly tells us whether a reference is worth trading into."),
                ("D", "Daily, LTF — execution: where do we actually engage?",
                 "Daily is the execution chart. This is where the engagement is taken.")):
                try:
                    cbuf = _rpt_price_chart(item, full, tf=tf, fig_h=4.4)
                    if cbuf is None:
                        continue
                    if tf != "M":
                        story += header(f"{title_pair} · {_rpt_tvlabel(item)}", f"Layer 2 — precision mapping — {tf}",
                                        REPORT_BRAND, "Proprietary supply & demand", subtitle)
                    else:
                        story += section(f"Monthly", "reference only, where are we in the big picture?")
                    story.append(P(note_txt, "note"))
                    story.append(img(cbuf))
                    src = (f"Chart source: {item['desc']}, Yahoo continuous front-month, {tf} candles. "
                           f"References: supply and demand indicator ({ZONE_DRAW_MODE.lower()}, extend {ZONE_EXTEND_BARS} bars).")
                    if not item["futures"]:
                        src += " Cash/spot series (no futures feed available)."
                    story.append(P(src, "small"))
                    if tf == "D":
                        if bias == 1:
                            read_txt = ("<b>THE READ</b><br/>Long in daily Demand only, if price comes down to a live "
                                        "Demand reference. A bias is not a trade: wait for the reference.")
                        elif bias == -1:
                            read_txt = ("<b>THE READ</b><br/>Short in daily Supply only, if price rallies into a live "
                                        "Supply reference. A bias is not a trade: wait for the reference.")
                        else:
                            read_txt = ("<b>THE READ</b><br/>No side this week. Stand dormant until flow or the "
                                        "calendar window gives a directional bias.")
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
        story += header("Invariant Weekly", "Other markets", REPORT_BRAND, "No full chapter this week",
                        today.strftime("%d %B %Y").lstrip("0"))
        rows = [[P(h, "label") for h in ("MARKET", "SECTION", "COORDINATED BIAS", "REL VALUE", "CAL FLOW", "WHY NOT A CHAPTER")]]
        for n in rest:
            r_ = packs[n]["row"]
            if not packs[n]["cot"] or packs[n]["cot"]["bias"] == 0:
                if not (packs[n]["season"] and packs[n]["season"]["main"] and packs[n]["season"]["main"]["strength"] == "STRONG"):
                    why = "no hedger flow side and no strong calendar window"
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
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, H(RPT["rule"])),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(t)

    doc.build(story)
    return path


def _make_pdf_report_plain(table, latest_cot=None):
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"invariant_report_{stamp}.pdf"
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            backColor=colors.HexColor("#0d1117"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=20, leading=24,
                              alignment=TA_CENTER, spaceAfter=8,
                              textColor=colors.HexColor("#e6edf3")))
    styles.add(ParagraphStyle(name="H3x", parent=styles["Heading3"], fontSize=10.5, leading=13,
                              spaceBefore=5, spaceAfter=3, textColor=colors.HexColor("#58a6ff")))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.5, leading=12, spaceAfter=4,
                              textColor=colors.HexColor("#e6edf3")))
    story = [Paragraph("INVARIANT WEEKLY — REPORT BUILD FAILED", styles["Title2"]),
             Paragraph(f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Positioning {latest_cot or '-'}", styles["Bodyx"]),
             Paragraph("The full report engine failed to build. See console traceback.", styles["Bodyx"])]
    if not table.empty:
        for _, r in table.head(20).iterrows():
            story.append(Paragraph(f"{r.get('Market', '-')}: {r.get('Fundamentals', '-')}", styles["H3x"]))
            story.append(Paragraph(str(r.get("AttentionReason", "-")), styles["Bodyx"]))
    doc.build(story)
    return path


def _make_pdf_report(table, latest_cot=None, prices=None, cot=None, only=None,
                     chart_data=None, prior_state=None):
    if prices is None or cot is None:
        return [("Full", _make_pdf_report_plain(table, latest_cot))], []

    full_list, lazy_list = [], []
    split = REPORT_SPLIT_BY_CATEGORY and not only

    if split:
        for cat in REPORT_CATEGORY_ORDER:
            try:
                p = _rpt_build(table, prices, cot, latest_cot, None,
                               chart_data, prior_state, mode="full", category=cat)
                if p:
                    full_list.append((cat, p))
            except Exception as exc:
                traceback.print_exc()
                log.error("Category %s full report failed (%s)", cat, exc)
            if BUILD_LAZY_PDF:
                try:
                    p = _rpt_build(table, prices, cot, latest_cot, None,
                                   chart_data, prior_state, mode="lazy", category=cat)
                    if p:
                        lazy_list.append((cat, p))
                except Exception as exc:
                    traceback.print_exc()
                    log.error("Category %s lazy report failed (%s)", cat, exc)
        if not full_list:
            full_list.append(("Full", _make_pdf_report_plain(table, latest_cot)))
    else:
        try:
            p = _rpt_build(table, prices, cot, latest_cot, only,
                           chart_data, prior_state, mode="full")
            full_list.append(("Full", p))
        except Exception as exc:
            traceback.print_exc()
            log.error("Full report engine failed (%s); writing plain fallback.", exc)
            full_list.append(("Full", _make_pdf_report_plain(table, latest_cot)))
        if BUILD_LAZY_PDF:
            try:
                p = _rpt_build(table, prices, cot, latest_cot, only,
                               chart_data, prior_state, mode="lazy")
                lazy_list.append(("Full", p))
            except Exception as exc:
                traceback.print_exc()
                log.error("Lazy report engine failed (%s); skipping lazy PDF.", exc)

    return full_list, lazy_list


def _build_decision_packs(out, prices, cot, chart_data):
    packs = {}
    weekly = prices.resample("W-FRI").last() if prices is not None else None
    for _, r in out.iterrows():
        nm = str(r["Market"]).replace(" (inv)", "")
        m = next((x for x in MARKETS if x[0] == nm), None)
        if not m:
            continue
        _, group, ticker, code, invert = m
        pk = {"row": r, "name": nm, "group": group, "ticker": ticker, "invert": invert,
              "last_price": None, "location_read": None, "location_why": None, "season": None, "cot": None,
              "weekly_zones": None, "daily_zones": None}
        item = (chart_data or {}).get(nm)
        if item and item.get("df") is not None and len(item["df"]) >= 120:
            pk["last_price"] = float(item["df"]["Close"].iloc[-1])
            pk["daily_zones"] = compute_sd_zones(item["df"])
            wk_df = _resample_safe(item["df"], "W-FRI")
            if len(wk_df) >= 30:
                wk_zones = compute_sd_zones(wk_df)
                pk["weekly_zones"] = wk_zones
                _lw = []
                pk["location_read"] = compute_location_read(wk_df, wk_zones, why_out=_lw)
                pk["location_why"] = _lw[0] if _lw else None
            else:
                pk["location_why"] = "not enough weekly history"
        else:
            pk["location_why"] = "no futures price history"
        try:
            if ticker and prices is not None and ticker in prices.columns:
                pk["season"] = _rpt_season_pack(_measure_close(nm, ticker, prices), dt.date.today(), market_name=nm)
        except Exception:
            pass
        try:
            if code and group != "Indices" and cot is not None and weekly is not None:
                sub = cot[cot["code"] == code.lstrip("0")]
                if len(sub) >= COT_LOOKBACK_SHORT:
                    pk["cot"] = _rpt_cot_pack(sub, _measure_weekly(nm, ticker, prices), invert)
        except Exception:
            pass
        packs[nm] = pk
    return packs


def show(table, latest_cot=None, prices=None, cot=None, only=None, chart_data=None,
         prior_state=None, cot_status="ok", cot_msg="", mode="sunday"):
    table = apply_blueprint_qualifier(table)
    if table.empty:
        log.warning("Nothing to show.")
        return None, {}
    out = table.drop(columns=[c for c in ["_strength", "_core"] if c in table.columns]).copy()
    for c in ["COT", "Comm26", "Comm1y", "Comm2y", "Comm3y", "Comm5y", "CommExtremeCount",
              "Spec26", "Retail26", "Retail1y", "Retail2y", "Retail3y", "Retail5y", "RetailExtremeCount"]:
        if c in out:
            out[c] = out[c].map(lambda v: "-" if not np.isfinite(_safe_float(v)) else f"{_safe_float(v):.0f}")
    out.to_csv("scan_results.csv", index=False)
    packs_for_wl = _build_decision_packs(out, prices, cot, chart_data)
    watchlist_msg = ""
    ideas_summary = None
    corr_warnings = []
    try:
        watchlist_msg = _watchlist_telegram_message(packs_for_wl)
    except Exception as e:
        log.warning("Watchlist message build failed: %s", e)
    try:
        ideas_summary = open_ideas_from_watchlist(packs_for_wl)
        corr_warnings = correlation_guard(load_ideas().get("ideas", {}))
    except Exception as e:
        traceback.print_exc()
        log.warning("Idea lifecycle update failed: %s", e)
    full_list, lazy_list = _make_pdf_report(table, latest_cot, prices, cot, only, chart_data, prior_state)

    peer_states = {}
    try:
        peer_states = _compute_peer_states(packs_for_wl, prior_state=prior_state)
    except Exception as e:
        log.warning("Peer state computation failed: %s", e)

    print("\n=== PRIORITY ATTENTION ===")
    for _, r in out.head(12).iterrows():
        print(f"\n{r.get('Market', '-')}: {r.get('AttentionReason', '-')}\n  {r.get('CombinedVerdict', '-')}")
    print("\n=== FULL SCAN saved to scan_results.csv ===")
    print(f"PDFs generated: {len(full_list)}")
    for name, p in full_list:
        print(f"  {name}: {p}")
    if lazy_list:
        print("Lazy PDFs:")
        for name, p in lazy_list:
            print(f"  {name}: {p}")
    if peer_states:
        print(f"Peer states tracked: {len(peer_states)}")
        for nm_p, rp in sorted(peer_states.items()):
            if rp.get("streak", 1) >= 2:
                print(f"  · {nm_p} vs {rp['peer']}: {rp['badge']}/{rp['evidence']} streak={rp['streak']}")

    # Season lock status
    try:
        _locks = _load_season_windows().get("locked", {})
        if _locks:
            print(f"Seasonal windows locked: {len(_locks)}")
        else:
            print("Seasonal windows locked: 0 (no persistent candidate this run)")
    except Exception:
        pass

    if telegram_enabled() and full_list:
        for name, path in full_list:
            if not path or not Path(path).exists():
                continue
            caption = f"Invariant Weekly · {name} · positioning {latest_cot or '-'}"
            if cot_status in ("stale", "unchanged"):
                caption += f"\n⚠️ {cot_msg}"
            if telegram_send_document(path, caption=caption):
                log.info("Telegram: %s PDF sent", name)
        for name, path in lazy_list:
            if path and Path(path).exists():
                telegram_send_document(
                    path,
                    caption=f"Invariant Weekly (Lazy) · {name} · positioning {latest_cot or '-'}")

        idea_msg = _ideas_telegram_sunday(ideas_summary, corr_warnings) if ideas_summary else ""
        summary = (f"✅ Market Scanner complete\n"
                   f"Positioning: {latest_cot or '-'}"
                   + (f"\n⚠️ {cot_msg}" if cot_status in ("stale", "unchanged") else "")
                   + ("\n\n" + watchlist_msg if watchlist_msg else "")
                   + ("\n\n" + idea_msg if idea_msg else ""))
        telegram_send_message(summary, silent=True)
    return full_list, peer_states


# ============================== MODE DETECTION ==============================
def detect_mode():
    valid = ("sunday", "wednesday", "daily", "full")
    argv = [a.lower() for a in sys.argv[1:]]
    for i, a in enumerate(argv):
        if a.startswith("--mode="):
            v = a.split("=", 1)[1].strip()
            if v in valid:
                return v
        if a == "--mode" and i + 1 < len(argv) and argv[i + 1] in valid:
            return argv[i + 1]
    for a in argv:
        if a in valid:
            return a
    env = os.environ.get("SCANNER_MODE", "").strip().lower()
    if env in valid:
        return env
    wd = dt.date.today().weekday()
    if wd in (5, 6, 0):
        return "sunday"
    if wd == 2:
        return "wednesday"
    return "daily"


# ============================== MAIN ==============================
def main():
    t0 = time.time()
    mode = detect_mode()
    log.info("Market Scanner %s starting · mode=%s", SCRIPT_VERSION, mode)
    if telegram_enabled():
        log.info("Telegram alerts enabled (chat %s)", TELEGRAM_CHAT_ID)
    else:
        log.info("Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    if mode == "daily":
        if not any(_is_open(i) for i in load_ideas().get("ideas", {}).values()):
            log.info("Daily watchdog: no live engagements. Silent.")
            return
        try:
            market_data = get_market_data(years=DAILY_HISTORY_YEARS)
            chart_data = get_chart_data(market_data, with_zones=False)
        except Exception as e:
            alert_failure("Daily watchdog could not load prices", str(e))
            traceback.print_exc()
            sys.exit(1)
        _px = market_data["Close"]
        _price_guard(_px, "daily")
        events = daily_watch(chart_data, {"prices": _px, "weekly": _px.resample("W-FRI").last()})
        msg = _ideas_telegram_daily(events)
        if msg:
            telegram_send_message(msg)
            log.info("Daily watchdog: %d event(s) sent to Telegram", len(events))
        else:
            log.info("Daily watchdog: nothing material. Silent.")
        return
    if mode == "wednesday":
        try:
            market_data = get_market_data()
            chart_data = get_chart_data(market_data, with_zones=False)
        except Exception as e:
            alert_failure("Mid-week check could not load prices", str(e))
            traceback.print_exc()
            sys.exit(1)
        px = market_data["Close"]
        _price_guard(px, "wednesday")
        ctx = {"prices": px, "weekly": px.resample("W-FRI").last()}
        try:
            delta = check_ideas(chart_data, ctx)
        except Exception as e:
            alert_failure("Mid-week check failed", str(e))
            traceback.print_exc()
            sys.exit(1)
        telegram_send_message(_ideas_telegram_wednesday(delta))
        log.info("Mid-week check sent to Telegram")
        return
    prior_state = load_state()
    only = [a for a in sys.argv[1:] if not a.startswith("-") and a not in ("sunday", "wednesday", "daily", "full", "mode")]
    meta = {"script_version": SCRIPT_VERSION, "argv": sys.argv[1:], "mode": mode}
    log.info("Downloading FUTURES prices (25-year history) ...")
    try:
        market_data = get_market_data()
        prices = market_data["Close"]
        log.info("Prices loaded for %d tickers, %d bars each", len(prices.columns), len(prices))
        meta["price_tickers"] = len(prices.columns)
        meta["price_bars"] = len(prices)
    except Exception as e:
        alert_failure("Price data unavailable (Yahoo/yfinance)", str(e))
        traceback.print_exc()
        sys.exit(1)
    _price_guard(prices, "sunday")
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
        telegram_send_message(f"⚠️ Market Scanner — positioning freshness\n{cot_msg}")
        if COT_STALE_ABORT and cot_status == "stale":
            alert_failure("Aborting: CFTC report too old", cot_msg)
            sys.exit(2)
    meta["cot_date"] = str(latest) if latest else None
    meta["cot_status"] = cot_status
    meta["cot_rows"] = len(cot)
    log.info("Building chart data (same futures OHLC as the analysis) ...")
    try:
        chart_data = get_chart_data(market_data)
        log.info("Charts prepared for %d markets", len(chart_data))
        meta["chart_markets"] = len(chart_data)
    except Exception as e:
        log.warning("Chart data failed (%s); Layer 2 pages will be skipped", e)
        traceback.print_exc()
        chart_data = {}
        meta["chart_markets"] = 0
    log.info("Running full scan ...")
    try:
        table = run_scan(prices, cot, seasonality=True)
    except Exception as e:
        alert_failure("Scan engine failed", str(e))
        traceback.print_exc()
        sys.exit(1)
    try:
        _failed = (table.loc[table["Fundamentals"] == "Error - review", "Market"].tolist()
                   if table is not None and "Fundamentals" in table.columns else [])
        if _failed:
            msg_f = f"{len(_failed)} market(s) could not be scanned: {', '.join(str(x) for x in _failed[:12])}"
            if len(_failed) >= SCAN_FAIL_ABORT:
                alert_failure("Too many markets failed to scan", msg_f + ". Nothing was reported.")
                sys.exit(1)
            telegram_send_message(f"⚠️ Market Scanner: {msg_f}. Their rows are blank in the report.")
    except SystemExit:
        raise
    except Exception as e:
        log.warning("Failed-market check could not run: %s", e)
    peer_states = {}
    try:
        result = show(table, latest, prices, cot, only, chart_data, prior_state,
                      cot_status=cot_status, cot_msg=cot_msg, mode=mode)
        if isinstance(result, tuple) and len(result) == 2:
            _, peer_states = result
    except Exception as e:
        alert_failure("Report generation failed", str(e))
        traceback.print_exc()
    meta["runtime_sec"] = round(time.time() - t0, 1)
    meta["markets_scanned"] = len(table) if table is not None else 0
    meta["peer_states_count"] = len(peer_states) if peer_states else 0
    try:
        _locks_now = _load_season_windows().get("locked", {})
        meta["season_windows_locked"] = len(_locks_now)
    except Exception:
        meta["season_windows_locked"] = 0
    save_state(table if table is not None else pd.DataFrame(), extra=meta, peer_states=peer_states)
    # bp28: guarantee the file exists so "missing file" never confuses with "no locks".
    _ensure_season_windows_file()
    log.info("Done in %.1f seconds. State saved.", meta["runtime_sec"])
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
    except SystemExit:
        raise
    except Exception as e:
        alert_failure("Unhandled exception", str(e))
        traceback.print_exc()
        sys.exit(1)
