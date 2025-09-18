# Capstone_Design/models/pipeline/economic_indicators.py
# (요약) KRX/FX/XAU/WTI + 기준금리(한국은행) 수집
# - bronze/ 이하 파케 저장
# - data_analyze/economic_indicator/krw_fed_rate_announcements.csv 생성(덮어쓰기)
#   컬럼: date, decision_rate, prev_rate
# - RAW CSV 업서트(누적)
#   ① 환율/지수: 'Date','종가'          ② 금/WTI: 'WTI_USD','XAU_USD','WTI_KRW','XAU_KRW'
# - DuckDB 테이블에도 MERGE 업서트 저장 (bronze 스키마)

import os
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone, timedelta, date as date_cls
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv

# =============================================================================
# 0) 경로/환경
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]  # .../Capstone_Design
DATA = ROOT / "data" / "lake" / "bronze"
DATA.mkdir(parents=True, exist_ok=True)

# 기준금리 공시 CSV (preprocess.py와 동일경로)
ANL = ROOT / "data_analyze" / "economic_indicator"
ANL.mkdir(parents=True, exist_ok=True)
ANNOUNCE_CSV = ANL / "krw_fed_rate_announcements.csv"

# RAW CSV (전처리 입력 호환)
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# DuckDB 위치
WAREHOUSE = ROOT / "data" / "warehouse"
WAREHOUSE.mkdir(parents=True, exist_ok=True)
DUCKDB_PATH = WAREHOUSE / "market.duckdb"

load_dotenv(ROOT / ".env")
ECOS_KEY = os.getenv("ECOS_KEY", "")
EIA_KEY  = os.getenv("EIA_KEY", "")
if not ECOS_KEY:
    raise SystemExit("ECOS_KEY가 필요합니다. Capstone_Design/.env 에 설정하세요.")

KST   = timezone(timedelta(hours=9))
TODAY = datetime.now(KST).date()
YMD   = TODAY.strftime("%Y%m%d")
print(f"=== {TODAY} (KST) ===")

# 공통 세션
def make_sess(allow_methods=("GET", "POST")):
    s = requests.Session()
    r = Retry(
        total=4, backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=set(allow_methods)
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://",  HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": "Mozilla/5.0 Chrome/124.0 Safari/537.36"})
    return s

SESS = make_sess()

# =============================================================================
# 공통: 파케 저장(원자적), CSV 업서트, DuckDB 업서트
# =============================================================================
def write_partition(df: pd.DataFrame, dataset: str, date_col="date", mode: str = "skip_if_exists"):
    if df is None or df.empty:
        print(f"[SKIP] {dataset}: empty"); return
    df = df.copy()

    def _scalarize(x):
        if isinstance(x, pd.Series):
            return x.iloc[0] if not x.empty else pd.NaT
        return x

    df[date_col] = pd.to_datetime(df[date_col].map(_scalarize), errors="coerce").dt.tz_localize(None)
    if df[date_col].dropna().empty:
        print(f"[SKIP] {dataset}: all {date_col} are NaT"); return

    part_date = df[date_col].dropna().max().date().isoformat()
    out_dir = DATA / dataset / f"date={part_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part.parquet"

    if mode == "skip_if_exists" and path.exists():
        print(f"[SKIP] {dataset}: {path.name} already exists"); return

    tmp = out_dir / ".__tmp_part.parquet"
    if tmp.exists(): tmp.unlink()
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"[OK] {dataset} → {path} rows={len(df)}")

def _upsert_price_csv(path: Path, date_value, price_value):
    """
    단일 시계열을 'Date','종가'로 업서트 저장.
    - 기존 컬럼은 보존
    - 동일 날짜가 있으면 '종가'만 갱신
    - 신규 날짜면 한 줄 append
    """
    if price_value is None or pd.isna(price_value) or date_value is None:
        return
    d = pd.to_datetime(date_value).strftime("%Y-%m-%d")

    # 새 레코드(필수 컬럼만)
    new_row = pd.DataFrame({"Date": [d], "종가": [float(price_value)]})

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        new_row.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[RAW] created → {path} rows=1")
        return

    old = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # Date 표준화
    if "Date" not in old.columns:
        for c in ["date", "날짜", "DATE"]:
            if c in old.columns:
                old = old.rename(columns={c: "Date"})
                break
    old["Date"] = pd.to_datetime(old["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 종가 컬럼 없으면 추가
    if "종가" not in old.columns:
        # 흔한 컬럼명 대응
        for c in ["Close", "Adj Close", "close", "adj close", "Price", "price"]:
            if c in old.columns:
                old = old.rename(columns={c: "종가"})
                break
        if "종가" not in old.columns:
            old["종가"] = pd.NA

    # 동일 날짜는 값만 갱신
    mask = (old["Date"] == d)
    if mask.any():
        old.loc[mask, "종가"] = float(price_value)
        old.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[RAW] updated (same day) → {path} rows={len(old)}")
        return

    # 신규 날짜는 append (헤더 없이)
    new_row.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    print(f"[RAW] appended → {path} +1 (Date={d})")

def _upsert_multi_cols_csv(path: Path, date_value, row: dict, columns: list):
    """
    다중 컬럼을 'Date'+columns 로 업서트 저장.
    - 기존 컬럼 보존(누락 컬럼은 추가)
    - 동일 날짜는 해당 컬럼만 갱신, 없으면 append
    """
    if date_value is None:
        return
    d = pd.to_datetime(date_value).strftime("%Y-%m-%d")

    if not path.exists():
        rec = {"Date": d}
        for c in columns:
            if c == "Date": continue
            v = row.get(c, None)
            rec[c] = (None if v is None or pd.isna(v) else float(v))
        df = pd.DataFrame([rec], columns=["Date"] + [c for c in columns if c != "Date"])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[RAW] created → {path} rows=1")
        return

    old = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "Date" not in old.columns:
        for c in ["date","날짜","DATE"]:
            if c in old.columns:
                old = old.rename(columns={c:"Date"})
                break
    if "Date" not in old.columns:
        old["Date"] = pd.NA

    # 누락 컬럼 보강
    for c in columns:
        if c not in old.columns:
            old[c] = pd.NA

    old["Date"] = pd.to_datetime(old["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mask = (old["Date"] == d)

    if mask.any():
        for c in columns:
            if c == "Date": continue
            v = row.get(c, None)
            if v is not None and not pd.isna(v):
                old.loc[mask, c] = float(v)
        old.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[RAW] updated (same day) → {path} rows={len(old)}")
        return

    # append
    rec = {"Date": d}
    for c in columns:
        if c == "Date": continue
        v = row.get(c, None)
        rec[c] = (None if v is None or pd.isna(v) else float(v))
    pd.DataFrame([rec], columns=["Date"] + [c for c in columns if c != "Date"])\
      .to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    print(f"[RAW] appended → {path} +1 (Date={d})")

def duck_upsert(table: str, df: pd.DataFrame, key_cols=("date",), schema: str = "core"):
    import duckdb
    WAREHOUSE = ROOT / "data" / "warehouse"
    WAREHOUSE.mkdir(parents=True, exist_ok=True)

    if df is None or df.empty:
        print(f"[DuckDB] skip {table}: empty")
        return

    con = duckdb.connect(str(WAREHOUSE / "market.duckdb"))
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")

    con.register("df_in", df)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}.{table} AS
        SELECT * FROM df_in WHERE 1=0
    """)
    cond = " AND ".join([f"t.{k} = s.{k}" for k in key_cols])
    con.execute(f"DELETE FROM {schema}.{table} t USING df_in s WHERE {cond};")
    con.execute(f"INSERT INTO {schema}.{table} BY NAME SELECT * FROM df_in;")

    con.unregister("df_in")
    con.close()
    print(f"[DuckDB] upserted → {schema}.{table} rows={len(df)}")

# =============================================================================
# 1) KRX 개장일 판정
# =============================================================================
def is_krx_trading_day(d: date_cls) -> bool:
    return d.weekday() < 5

# =============================================================================
# 2) XAUUSD (Stooq)
# =============================================================================
def fetch_xauusd_today_stooq(strict_today=True, backfill_days=3) -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    r = SESS.get(url, timeout=30); r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame(columns=["date","xau_usd"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    m = df[df["Date"].dt.date == TODAY]
    if not m.empty:
        return m.rename(columns={"Date":"date","Close":"xau_usd"})[["date","xau_usd"]]
    if strict_today:
        return pd.DataFrame(columns=["date","xau_usd"])
    # backfill (for parquet/duckdb internal only)
    for back in range(1, backfill_days+1):
        d = TODAY - timedelta(days=back)
        m = df[df["Date"].dt.date == d]
        if not m.empty:
            return m.rename(columns={"Date":"date","Close":"xau_usd"})[["date","xau_usd"]]
    return df.tail(1).rename(columns={"Date":"date","Close":"xau_usd"})[["date","xau_usd"]]

# =============================================================================
# 3) ECOS 공통
# =============================================================================
ECOS_BASE = "https://ecos.bok.or.kr/api"

def ecos_day_xml(stat_code: str, item_code1: str, ymd: str) -> pd.DataFrame:
    url = (f"{ECOS_BASE}/StatisticSearch/{ECOS_KEY}/xml/kr/1/10/"
           f"{stat_code}/D/{ymd}/{ymd}/{item_code1}/?/?/?")
    r = SESS.get(url, timeout=30); r.raise_for_status()
    root = ET.fromstring(r.content)
    if len(root) >= 2 and root[0].text in ("INFO","ERROR"):
        return pd.DataFrame(columns=["date","value"])
    rows = [{c.tag: c.text for c in row} for row in root.iter("row")]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date","value"])
    df["date"]  = pd.to_datetime(df["TIME"], errors="coerce")
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df.dropna(subset=["date","value"])[["date","value"]]

def ecos_range_xml(stat_code: str, item_code1: str, start_ymd: str, end_ymd: str, end_row: int = 200000) -> pd.DataFrame:
    url = (f"{ECOS_BASE}/StatisticSearch/{ECOS_KEY}/xml/kr/1/{end_row}/"
           f"{stat_code}/D/{start_ymd}/{end_ymd}/{item_code1}/?/?/?")
    r = SESS.get(url, timeout=30); r.raise_for_status()
    root = ET.fromstring(r.content)
    if len(root) >= 2 and root[0].text in ("INFO","ERROR"):
        return pd.DataFrame(columns=["date","value"])
    rows = [{c.tag: c.text for c in row} for row in root.iter("row")]
    df = pd.DataFrame(rows)
    if df.empty: return pd.DataFrame(columns=["date","value"])
    df["date"]  = pd.to_datetime(df["TIME"], errors="coerce")
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df.dropna(subset=["date","value"])[["date","value"]].sort_values("date")

# =============================================================================
# 4) ECOS 환율 — 최근 가용일까지 백필
# =============================================================================
def fetch_ecos_fx_T_or_Tminus1(ymd: str, max_back_days: int = 7) -> pd.DataFrame:
    base_dt = pd.to_datetime(ymd).date()
    for back in range(0, max_back_days + 1):
        d = (base_dt - timedelta(days=back)).strftime("%Y%m%d")
        usd = ecos_day_xml("731Y001","0000001", d)  # 원/달러
        eur = ecos_day_xml("731Y001","0000003", d)  # 원/유로
        jpy = ecos_day_xml("731Y001","0000002", d)  # 원/100엔
        cny = ecos_day_xml("731Y001","0000053", d)  # 원/위안
        xusdjpy = ecos_day_xml("731Y002","0000002", d)  # USD/JPY cross

        if any(not x.empty for x in [usd, eur, jpy, cny, xusdjpy]):
            out = {"date": pd.to_datetime(d)}
            if not usd.empty: out["usd_krw"] = float(usd["value"].iloc[-1])
            if not eur.empty: out["eur_krw"] = float(eur["value"].iloc[-1])
            if not cny.empty: out["cny_krw"] = float(cny["value"].iloc[-1])
            if not jpy.empty: out["jpy_krw"] = float(jpy["value"].iloc[-1]) / 100.0
            if not xusdjpy.empty: out["usd_jpy"] = float(xusdjpy["value"].iloc[-1])
            return pd.DataFrame([out])
    return pd.DataFrame(columns=["date","usd_krw","eur_krw","cny_krw","jpy_krw","usd_jpy"])

# =============================================================================
# 5) KOSPI — Yahoo Finance (^KS11) 최근가
# =============================================================================
def fetch_kospi_latest_yahoo_on_or_before(target_day: date_cls, days_window=14) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("[yfinance] 미설치 → pip install yfinance")
        return pd.DataFrame(columns=["date","kospi"])

    period = f"{max(1,int(days_window))}d"
    try:
        df = yf.download("^KS11", period=period, interval="1d", auto_adjust=False, progress=False)
    except TypeError:
        df = yf.download("^KS11", period=period, interval="1d", auto_adjust=False)

    if df is None or df.empty:
        return pd.DataFrame(columns=["date","kospi"])

    ser = None
    if "Close" in df.columns:
        ser = df["Close"]
    elif "Adj Close" in df.columns:
        ser = df["Adj Close"]
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:,0]
    if ser is None or ser.empty:
        return pd.DataFrame(columns=["date","kospi"])

    ser = pd.to_numeric(ser, errors="coerce").dropna()
    ser.index = pd.to_datetime(ser.index)
    ser = ser[ser.index.date <= target_day]
    if ser.empty:
        return pd.DataFrame(columns=["date","kospi"])

    last_dt = ser.index[-1]
    last_px = float(ser.iloc[-1])
    return pd.DataFrame([{"date": last_dt, "kospi": last_px}])

# =============================================================================
# 6) VKOSPI — KRX 데이터포털(getJsonData.cmd)로 수집
# =============================================================================
KRX_REF = "https://data.krx.co.kr/contents/MDC/STAT/standard/MDCSTAT01201.jspx"
KRX_JSON = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

def _krx_session():
    s = make_sess(allow_methods=("GET","POST"))
    # Referer 페이지를 먼저 열어 세션/쿠키 확보
    s.get(KRX_REF, timeout=30)
    return s

def fetch_vkospi_krx_range(start_ymd: str, end_ymd: str) -> pd.DataFrame:
    """MDCSTAT01201에서 VKOSPI 구간 데이터 조회 → date, vkospi"""
    sess = _krx_session()
    payload = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT01201",
        "locale": "ko_KR",
        "indTpCd": "1",
        "idxIndCd": "300",
        "idxCd": "1",
        "idxCd2": "300",
        "strtDd": start_ymd,
        "endDd": end_ymd,
        "csvxls_isNo": "false",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": KRX_REF,
        "Origin": "https://data.krx.co.kr",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    r = sess.post(KRX_JSON, data=payload, headers=headers, timeout=30)
    r.raise_for_status()
    j = r.json()
    rows = j.get("output", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date","vkospi"])

    date_col = "TRD_DD" if "TRD_DD" in df.columns else next((c for c in df.columns if "DD" in c), None)
    close_col = ("IDX_CLSPRC" if "IDX_CLSPRC" in df.columns
                 else "TDD_CLSPRC" if "TDD_CLSPRC" in df.columns
                 else next((c for c in df.columns if "CLSPRC" in c), None))
    if not date_col or not close_col:
        return pd.DataFrame(columns=["date","vkospi"])

    out = (
        df[[date_col, close_col]]
        .rename(columns={date_col:"date", close_col:"vkospi"})
        .assign(
            date=lambda x: pd.to_datetime(x["date"].astype(str).str.replace(r"[./-]", "", regex=True), errors="coerce"),
            vkospi=lambda x: pd.to_numeric(x["vkospi"].astype(str).str.replace(",", ""), errors="coerce"),
        )
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out

def fetch_vkospi_latest_on_or_before_krx(ymd_end: str, lookback_days: int = 14) -> pd.DataFrame:
    end_dt   = datetime.strptime(ymd_end, "%Y%m%d").date()
    start_dt = end_dt - timedelta(days=lookback_days)
    df = fetch_vkospi_krx_range(start_dt.strftime("%Y%m%d"), ymd_end)
    if df.empty:
        return pd.DataFrame(columns=["date","vkospi"])
    df = df[df["date"].dt.date <= end_dt]
    return df.tail(1)[["date","vkospi"]]

# =============================================================================
# 7) WTI — EOD만 사용
# =============================================================================
def fetch_wti_eia_latest_on_or_before(target_iso: str) -> pd.DataFrame:
    if not EIA_KEY:
        return pd.DataFrame(columns=["date","wti_usd","src"])
    url = "https://api.eia.gov/v2/seriesid/PET.RWTC.D"
    params = {"api_key": EIA_KEY, "end": target_iso,
              "sort[0][column]": "period", "sort[0][direction]": "desc",
              "length": 1}
    r = SESS.get(url, params=params, timeout=30); r.raise_for_status()
    rows = (r.json().get("response", {}) or {}).get("data", [])
    if not rows: return pd.DataFrame(columns=["date","wti_usd","src"])
    df = pd.DataFrame(rows).rename(columns={"period":"date","value":"wti_usd"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wti_usd"] = pd.to_numeric(df["wti_usd"], errors="coerce")
    df = df.dropna(subset=["date","wti_usd"])
    if df.empty: return pd.DataFrame(columns=["date","wti_usd","src"])
    df["src"] = "EIA"
    return df[["date","wti_usd","src"]]

def fetch_wti_fut_latest_yahoo_on_or_before(target_day: date_cls, days_window=5) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("[yfinance] 미설치 → pip install yfinance")
        return pd.DataFrame(columns=["date","wti_usd","src"])
    period = f"{max(1,int(days_window))}d"
    try:
        df = yf.download("CL=F", period=period, interval="1d", auto_adjust=False, progress=False)
    except TypeError:
        df = yf.download("CL=F", period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","wti_usd","src"])
    ser = None
    if "Close" in df.columns: ser = df["Close"]
    elif "Adj Close" in df.columns: ser = df["Adj Close"]
    if isinstance(ser, pd.DataFrame): ser = ser.iloc[:,0]
    if ser is None or ser.empty: return pd.DataFrame(columns=["date","wti_usd","src"])
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    ser.index = pd.to_datetime(ser.index)
    ser = ser[ser.index.date <= target_day]
    if ser.empty: return pd.DataFrame(columns=["date","wti_usd","src"])
    last_dt = ser.index[-1]; last_px = float(ser.iloc[-1])
    return pd.DataFrame([{"date": last_dt, "wti_usd": last_px, "src": "YF_CL=F"}])

def choose_wti_for_training(tday: date_cls, stale_threshold_days=2):
    """EOD만 반환."""
    target_iso = (tday - timedelta(days=1)).isoformat()
    eod = pd.DataFrame()

    eia = fetch_wti_eia_latest_on_or_before(target_iso)
    if not eia.empty:
        lag = (pd.to_datetime(target_iso).date() - eia["date"].iloc[0].date()).days
        print(f"[WTI EIA] got {eia['date'].iloc[0].date()} lag={lag}d")
        if lag <= stale_threshold_days:
            eia["lag_days"] = lag
            eod = eia

    if eod.empty:
        yf_t1 = fetch_wti_fut_latest_yahoo_on_or_before(tday - timedelta(days=1), days_window=5)
        if not yf_t1.empty:
            lag = (tday - timedelta(days=1) - yf_t1["date"].iloc[0].date()).days
            yf_t1["lag_days"] = lag
            eod = yf_t1

    return {"eod": eod}

# =============================================================================
# 8) 한국은행 기준금리
# =============================================================================
def fetch_base_rate_T_or_Tminus1(ymd: str, max_back_days: int = 7) -> pd.DataFrame:
    base_dt = pd.to_datetime(ymd).date()
    for back in range(0, max_back_days + 1):
        d = (base_dt - timedelta(days=back)).strftime("%Y%m%d")
        df = ecos_day_xml("722Y001", "0101000", d)  # 한국은행 기준금리(연%)
        if not df.empty:
            return df.rename(columns={"value":"base_rate"})[["date","base_rate"]]
    return pd.DataFrame(columns=["date","base_rate"])

def build_base_rate_announcements(start_ymd: str = "20000101", end_ymd: str = None) -> pd.DataFrame:
    if end_ymd is None:
        end_ymd = YMD
    series = ecos_range_xml("722Y001", "0101000", start_ymd, end_ymd)
    if series.empty:
        return pd.DataFrame(columns=["date","decision_rate","prev_rate"])
    series = series.rename(columns={"value":"rate"})[["date","rate"]].sort_values("date")
    series["prev_rate"] = series["rate"].shift(1)
    ann = series[series["rate"] != series["prev_rate"]].dropna(subset=["prev_rate"]).copy()
    ann = ann.rename(columns={"rate":"decision_rate"})[["date","decision_rate","prev_rate"]]
    ann["date"] = pd.to_datetime(ann["date"]).dt.strftime("%Y-%m-%d")
    for c in ["decision_rate","prev_rate"]:
        ann[c] = pd.to_numeric(ann[c], errors="coerce").round(2)
    return ann

def save_base_rate_announcements_csv(path: Path, start_ymd="20000101", end_ymd=None):
    df = build_base_rate_announcements(start_ymd, end_ymd)
    if df.empty:
        print("[policy] 기준금리 공시 없음(ECOS 응답 비어있음) → CSV 미생성/유지")
        return
    tmp = path.with_suffix(".tmp.csv")
    if tmp.exists(): tmp.unlink()
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    if path.exists(): path.unlink()
    tmp.replace(path)
    print(f"[policy] wrote announcements → {path} rows={len(df)}")

# =============================================================================
# 9) 실행
# =============================================================================
if __name__ == "__main__":
    # (a) KOSPI — 휴장일도 최근가 백필 (파케/DB)
    kospi = fetch_kospi_latest_yahoo_on_or_before(TODAY, days_window=14)
    print("\n[KOSPI latest]"); print(kospi if not kospi.empty else "KOSPI 없음")
    write_partition(kospi, "kospi", mode="skip_if_exists")
    duck_upsert("kospi",        kospi,        key_cols=("date",))

    # (b) VKOSPI — KRX 포털 (파케/DB)
    vkospi = fetch_vkospi_latest_on_or_before_krx(YMD, lookback_days=30)
    print("\n[V-KOSPI latest (KRX)]"); print(vkospi if not vkospi.empty else "V-KOSPI 없음")
    write_partition(vkospi, "vkospi", mode="skip_if_exists")
    duck_upsert("vkospi",       vkospi,       key_cols=("date",))

    # (c) FX (ECOS)
    # RAW용: "오늘만"
    fx_today = fetch_ecos_fx_T_or_Tminus1(YMD, max_back_days=0)
    # 파케/DB용: 최근 가용값
    fx_all   = fx_today if not fx_today.empty else fetch_ecos_fx_T_or_Tminus1(YMD, max_back_days=7)
    print("\n[ECOS FX (parquet/duckdb)]"); print(fx_all if not fx_all.empty else "FX 없음")
    write_partition(fx_all, "fx", mode="skip_if_exists")
    duck_upsert("fx",           fx_all,           key_cols=("date",))

    # (d) XAU
    # RAW용: "오늘만"
    xau_today = fetch_xauusd_today_stooq(strict_today=True, backfill_days=0)
    # 파케/DB용: 최근 가용값
    xau_all   = xau_today if not xau_today.empty else fetch_xauusd_today_stooq(strict_today=False, backfill_days=3)
    print("\n[XAUUSD (parquet/duckdb)]"); print(xau_all if not xau_all.empty else "XAU 없음")
    write_partition(xau_all, "xau", mode="skip_if_exists")
    duck_upsert("xau",          xau_all,          key_cols=("date",))

    # (e) WTI — ★EOD만 사용 (파케/DB)
    wti = choose_wti_for_training(TODAY, stale_threshold_days=2)
    eod = wti["eod"]
    print("\n[WTI EOD (≤ T-1)]"); print(eod if not eod.empty else "WTI EOD 없음")
    write_partition(eod,   "wti_eod",      mode="skip_if_exists")
    duck_upsert("wti_eod",      eod,          key_cols=("date",))

    # (f) 기준금리 (ECOS)
    base_today_or_recent = fetch_base_rate_T_or_Tminus1(YMD, max_back_days=7)
    print("\n[Base Rate latest ≤ T]"); print(base_today_or_recent if not base_today_or_recent.empty else "기준금리 없음")
    write_partition(base_today_or_recent.rename(columns={"base_rate":"decision_rate"}), "base_rate", mode="skip_if_exists")
    duck_upsert("base_rate",    base_today_or_recent,   key_cols=("date",))

    # (g) features_daily (인트라데이 관련 키 전면 제거)
    def _safe(df, col):
        return float(df[col].iloc[-1]) if (df is not None and not df.empty and col in df.columns) else None

    # 환율(usd_krw)은 fx_all 기준
    usd_krw_val = _safe(fx_all, "usd_krw")
    extras = {}
    if usd_krw_val is not None:
        if not xau_all.empty and "xau_usd" in xau_all.columns:
            extras["gold_krw"] = usd_krw_val * float(xau_all["xau_usd"].iloc[-1])
        if not eod.empty and "wti_usd" in eod.columns:
            extras["wti_krw_eod"] = usd_krw_val * float(eod["wti_usd"].iloc[-1])

    features = {
        "asof_date_kst": TODAY.isoformat(),
        "kospi_close": _safe(kospi, "kospi"),
        "vkospi": (float(vkospi["vkospi"].iloc[-1]) if not vkospi.empty else None),
        "usd_krw": _safe(fx_all, "usd_krw"),
        "eur_krw": _safe(fx_all, "eur_krw"),
        "cny_krw": _safe(fx_all, "cny_krw"),
        "jpy_krw": _safe(fx_all, "jpy_krw"),
        "usd_jpy": _safe(fx_all, "usd_jpy"),
        "xau_usd": _safe(xau_all, "xau_usd"),
        "wti_usd_tminus1_eod": _safe(eod, "wti_usd"),
        "wti_src_eod": (eod["src"].iloc[-1] if not eod.empty and "src" in eod.columns else None),
        "wti_asof_eod": (eod["date"].iloc[-1].isoformat() if not eod.empty else None),
        "policy_rate": _safe(base_today_or_recent, "base_rate"),
        **extras
    }
    feat_df = pd.DataFrame([features])
    print("\n[features_daily]"); print(feat_df.T)
    feat_df2 = feat_df.copy()
    feat_df2["date"] = pd.to_datetime(feat_df2["asof_date_kst"])
    write_partition(feat_df2, "features_daily", date_col="date", mode="skip_if_exists")
    duck_upsert("features_daily", feat_df2,   key_cols=("date",))

    # (h) 기준금리 공시 CSV (전체 스캔 → 변경일만)
    save_base_rate_announcements_csv(ANNOUNCE_CSV, start_ymd="20000101", end_ymd=YMD)

    # (i) RAW CSV 업서트 — ★오늘 데이터만 추가/갱신★
    # FX RAW (오늘만)
    if not fx_today.empty and fx_today["date"].iloc[-1].date() == TODAY:
        fx_dt = fx_today["date"].iloc[-1]
        for col, fname in [
            ("usd_krw", "USD_KRW Historical Data.csv"),
            ("eur_krw", "EUR_KRW Historical Data.csv"),
            ("cny_krw", "CNY_KRW Historical Data.csv"),
            ("jpy_krw", "JPY_KRW Historical Data.csv"),
            ("usd_jpy", "USD_JPY Historical Data.csv"),
        ]:
            if col in fx_today.columns and not pd.isna(fx_today[col].iloc[-1]):
                _upsert_price_csv(RAW / fname, fx_dt, fx_today[col].iloc[-1])
    else:
        print("[RAW] FX 오늘 값 없음 → RAW CSV 스킵")

    # XAU RAW (오늘만)
    if not xau_today.empty and xau_today["date"].iloc[-1].date() == TODAY:
        _upsert_price_csv(RAW / "XAUUSD Historical Data.csv",
                          xau_today["date"].iloc[-1], xau_today["xau_usd"].iloc[-1])
    else:
        print("[RAW] XAU 오늘 값 없음 → RAW CSV 스킵")

    # KOSPI RAW (오늘만)
    if not kospi.empty and pd.to_datetime(kospi["date"].iloc[-1]).date() == TODAY:
        _upsert_price_csv(RAW / "KOSPI Historical Data.csv",
                          kospi["date"].iloc[-1], kospi["kospi"].iloc[-1])
    else:
        print("[RAW] KOSPI 오늘 종가 없음/휴장 → RAW CSV 스킵")

    # VKOSPI RAW (오늘만)
    if not vkospi.empty and pd.to_datetime(vkospi["date"].iloc[-1]).date() == TODAY:
        _upsert_price_csv(RAW / "KOSPI Volatility Historical Data.csv",
                          vkospi["date"].iloc[-1], vkospi["vkospi"].iloc[-1])
    else:
        print("[RAW] VKOSPI 오늘 값 없음 → RAW CSV 스킵")

    # 금/WTI 복합 RAW (오늘만) — WTI는 T-1 EOD라 오늘 날짜 행은 의미 없으니 생략 권장
    # 필요하면 아래 주석을 해제하되, TODAY가 아닌 날짜 쓰기 주의
    usd_krw_for_raw = _safe(fx_today, "usd_krw")
    xau_usd_for_raw = _safe(xau_today, "xau_usd")
    if usd_krw_for_raw is not None and xau_usd_for_raw is not None:
        row = {
            "WTI_USD": None,  # EOD(T-1)이므로 오늘 행엔 쓰지 않음
            "XAU_USD": xau_usd_for_raw,
            "WTI_KRW": None,
            "XAU_KRW": xau_usd_for_raw * usd_krw_for_raw,
        }
        _upsert_multi_cols_csv(
            RAW / "gold_wti_prices_krw.csv",
            TODAY,  # 오늘 행
            row,
            ["Date","WTI_USD","XAU_USD","WTI_KRW","XAU_KRW"]
        )
    else:
        print("[RAW] gold_wti_prices_krw 오늘 계산 불가 → 스킵")

    print("\n✅ done")

# =============================================================================
# DuckDB 뷰(파케 읽는 뷰) — ★wti_intraday 제외
# =============================================================================
def register_duckdb_views():
    import duckdb
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    for ds in ["kospi", "vkospi", "fx", "xau", "wti_eod", "base_rate", "features_daily"]:
        pattern = str((DATA / ds / "date=*/part.parquet")).replace("\\", "/")
        con.execute(f"CREATE OR REPLACE VIEW bronze.{ds}_view AS SELECT * FROM read_parquet('{pattern}');")
    con.close()
    print("[OK] DuckDB views updated: data/warehouse/market.duckdb (schema=bronze)")

# 파일 끝의 main() 마지막에 호출
register_duckdb_views()
