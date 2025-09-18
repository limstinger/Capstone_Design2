#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess.py (date-aligned & robust; columns exactly match prior training header)

- 입력: data/raw/ (+ data_analyze/economic_indicator)
- 처리: 모든 전처리/파생을 메모리에서 수행 (KOSPI 거래일 달력 기준 좌조인)
- 출력(기본): data/processed/training_with_refined_features.csv
- --out 옵션으로 경로 변경 가능
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ───────────────────────── Paths
def get_base_dir() -> Path:
    # Capstone_Design 루트 추정
    cand = Path(__file__).resolve().parents[2]
    if (cand / "data" / "raw").exists():
        return cand
    alt = Path(__file__).resolve().parents[1]
    return alt if (alt / "data" / "raw").exists() else cand

BASE = get_base_dir()
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
ANL  = BASE / "data_analyze" / "economic_indicator"
NEWS = BASE / "data_analyze" / "news_data" / "output"
PROC.mkdir(parents=True, exist_ok=True)

# ───────────────────────── Utils
PREFERRED_PRICE = ["종가","Close","Adj Close","Adjusted Close","Price","마감","마감가","close","price","adj close"]

def to_num(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
         .str.replace("\u200b", "", regex=False)
         .str.replace("\u202f", "", regex=False)
         .str.replace(r"[^0-9\.\-]", "", regex=True)
         .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")

def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)

def find_date_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in ["date", "날짜", "timestamp", "Date"]:
        if c.lower() in lower:
            return lower[c.lower()]
    return df.columns[0]

def find_price_col_generic(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in PREFERRED_PRICE:
        if c.lower() in lower:
            return lower[c.lower()]
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return num_cols[-1] if num_cols else df.columns[-1]

def pick_kospi_close_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    candidates = []
    for c in PREFERRED_PRICE:
        if c.lower() in lower:
            candidates.append(lower[c.lower()])
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["vol", "volume", "거래량", "change", "%"]):
            continue
        if c not in candidates:
            candidates.append(c)
    best_col, best_score = None, -1.0
    for c in candidates:
        raw = df[c].astype(str)
        has_percent = raw.str.contains("%").mean() > 0.3
        s = to_num(raw)
        non_na_ratio = s.notna().mean()
        median_val = s.median(skipna=True)
        in_range = 100 <= (median_val if pd.notna(median_val) else -1) <= 10000
        score = non_na_ratio + (0.2 if in_range else 0.0) - (0.5 if has_percent else 0.0)
        if score > best_score:
            best_score = score
            best_col = c
    if best_col is None:
        raise RuntimeError("[KOSPI] 종가 컬럼 자동선택 실패")
    return best_col

def align_to_kospi_calendar(series_df: pd.DataFrame, series_name: str,
                            kospi_dates: pd.Series, do_ffill: bool = True) -> pd.DataFrame:
    cal = pd.DataFrame({"date": pd.to_datetime(kospi_dates).dt.floor("D")}).drop_duplicates().sort_values("date")
    out = cal.merge(series_df, on="date", how="left")
    if do_ffill:
        out[series_name] = out[series_name].ffill().bfill()
    return out

# ───────────────────────── Core builders (training_on_trading_days.py 준용)
def build_kospi_base() -> pd.DataFrame:
    path = RAW / "KOSPI Historical Data.csv"
    if not path.exists():
        raise FileNotFoundError(f"필수 파일 없음: {path}")
    df = read_csv_any(path)

    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date":"date"})
    elif "date" not in df.columns:
        df = df.rename(columns={df.columns[0]:"date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # close 선택
    pcol = pick_kospi_close_col(df)
    close_raw = to_num(df[pcol])
    if close_raw.notna().sum() == 0:
        df[pcol] = df[pcol].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        close_raw = to_num(df[pcol])
    if close_raw.notna().sum() == 0:
        raise RuntimeError(f"[KOSPI] 종가 파싱 실패: '{pcol}'")

    df["close"] = close_raw.ffill().bfill()

    # next/label/달력파생
    df["next_close"]  = df["close"].shift(-1)
    df["logret_next"] = np.log(df["next_close"] / df["close"].replace(0, np.nan))
    df["label_up"]    = (df["logret_next"] > 0).astype(int)

    df["prev_trade_date"] = df["date"].shift(1)
    df["next_trade_date"] = df["date"].shift(-1)
    df["days_since_last_trade"] = (df["date"] - df["prev_trade_date"]).dt.days.fillna(1).astype(int)
    df["upcoming_gap"]          = (df["next_trade_date"] - df["date"]).dt.days.fillna(1).astype(int)
    df["is_pre_holiday"]  = (df["upcoming_gap"] > 1).astype(int)
    df["is_post_holiday"] = (df["days_since_last_trade"] > 1).astype(int)

    dow = df["date"].dt.weekday.astype(float)
    df["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    df["dow_cos"] = np.cos(2*np.pi*dow/7.0)

    # 자체 로그수익
    df["logret_KOSPI_Close"] = np.log(df["close"] / df["close"].shift(1))
    return df

# ───────────────────────── Macro / FX / Vol
def load_one_series(path: Path, series_name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"⚠️ 없음(건너뜀): {path.name}")
        return pd.DataFrame(columns=["date", series_name])
    raw = read_csv_any(path)
    dcol = find_date_col(raw)
    pcol = find_price_col_generic(raw)
    df = raw[[dcol, pcol]].copy()
    df.columns = ["date", series_name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df[series_name] = to_num(df[series_name])
    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df

def build_macro_fx_vol(kospi_dates: pd.Series) -> pd.DataFrame:
    pieces = []
    def add(name: str, csv: str):
        df = load_one_series(RAW / csv, name)
        if df.empty:
            return
        pieces.append(align_to_kospi_calendar(df, name, kospi_dates, do_ffill=True))

    # 환율/변동성
    add("USD_KRW", "USD_KRW Historical Data.csv")
    add("USD_JPY", "USD_JPY Historical Data.csv")
    add("JPY_KRW", "JPY_KRW Historical Data.csv")
    add("CNY_KRW", "CNY_KRW Historical Data.csv")
    add("EUR_KRW", "EUR_KRW Historical Data.csv")
    add("KOSPI_Volatility", "KOSPI Volatility Historical Data.csv")

    # 금/WTI (선택)
    gw = RAW / "gold_wti_prices_krw.csv"
    if gw.exists():
        g = read_csv_any(gw)
        dcol = find_date_col(g)
        got = [c for c in ["gold_krw", "wti_krw"] if c in g.columns]
        if got:
            tmp = g[[dcol] + got].copy()
            tmp.columns = ["date"] + got
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.floor("D")
            for cname in got:
                tmp[cname] = to_num(tmp[cname])
                pieces.append(align_to_kospi_calendar(tmp[["date", cname]], cname, kospi_dates, do_ffill=True))

    out = None
    for p in pieces:
        out = p if out is None else out.merge(p, on="date", how="inner")
    if out is None:
        return pd.DataFrame()

    # 교차환율
    if {"USD_JPY","JPY_KRW"} <= set(out.columns):
        out["USD_KRW_derived"] = out["USD_JPY"] * out["JPY_KRW"]

    # 로그수익/모멘텀/변동성
    for c in [col for col in out.columns if col != "date"]:
        r = np.log(out[c] / out[c].replace(0, np.nan).shift(1))
        out[f"logret_{c.replace('-', '_')}"] = r
        out[f"logret_{c.replace('-', '_')}_momentum_3d"] = r.rolling(3, min_periods=1).sum()
        out[f"logret_{c.replace('-', '_')}_vol_5d"]      = r.rolling(5, min_periods=1).std()
    return out

# ───────────────────────── Sentiment
def build_sentiment(sentiment_csv: Path, kospi_dates: pd.Series) -> pd.DataFrame:
    if not sentiment_csv.exists():
        print(f"ℹ️ sentiment 파일 없음 → 감성 파생 생략: {sentiment_csv}")
        return pd.DataFrame()
    df = read_csv_any(sentiment_csv)
    dcol = None
    for c in ["date","Date","report_date","adjustedDate"]:
        if c in df.columns:
            dcol = c; break
    if dcol is None:
        dcol = df.columns[0]
    df = df.rename(columns={dcol: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

    if "sentiment_score" not in df.columns and {"pos_prob","neg_prob"} <= set(df.columns):
        df["pos_prob"] = pd.to_numeric(df["pos_prob"], errors="coerce")
        df["neg_prob"] = pd.to_numeric(df["neg_prob"], errors="coerce")
        df["sentiment_score"] = df["pos_prob"] - df["neg_prob"]

    keep = ["date"] + [c for c in ["pos_prob","neg_prob","sentiment_score"] if c in df.columns]
    df = df[keep].dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    cal = pd.DataFrame({"date": pd.to_datetime(kospi_dates).dt.floor("D")}).drop_duplicates().sort_values("date")
    out = cal.merge(df, on="date", how="left")
    for c in [col for col in out.columns if col != "date"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 감성 파생
    if "sentiment_score" in out.columns:
        out["sentiment_momentum_1d"] = out["sentiment_score"] - out["sentiment_score"].shift(1)
        out["sentiment_momentum_3d"] = out["sentiment_score"] - out["sentiment_score"].shift(3)
        out["sentiment_roll5_mean_prev"] = out["sentiment_score"].shift(1).rolling(5, min_periods=1).mean()
        out["sentiment_surprise_5d"] = out["sentiment_score"] - out["sentiment_roll5_mean_prev"]
        out["sentiment_vol_5d"] = out["sentiment_score"].rolling(5, min_periods=1).std()
        # z-score(30)
        m30 = out["sentiment_score"].rolling(30, min_periods=5).mean()
        s30 = out["sentiment_score"].rolling(30, min_periods=5).std()
        out["sentiment_z_30"] = (out["sentiment_score"] - m30) / s30.replace(0, np.nan)

    return out

# ───────────────────────── Topic-derived (뉴스)
def _entropy_arr(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))

def _kl(q: np.ndarray, p: np.ndarray, eps=1e-9) -> float:
    q = np.clip(q, eps, 1); p = np.clip(p, eps, 1)
    return float(np.sum(q * np.log(q / p)))

def build_topic_features(news_csv: Path, kospi_dates: pd.Series,
                         top_k=3, emergence_window=5, trend_window=5, emergence_th=1.5) -> pd.DataFrame:
    if not news_csv.exists():
        print(f"ℹ️ 뉴스 파일 없음 → 토픽 파생 생략: {news_csv}")
        return pd.DataFrame()
    df = read_csv_any(news_csv)
    dcol = None
    for c in ["adjustedDate","date","Date"]:
        if c in df.columns:
            dcol = c; break
    if dcol is None or "topic" not in df.columns or "topic_prob" not in df.columns:
        print("ℹ️ topic/date 컬럼 미존재 → 토픽 파생 생략")
        return pd.DataFrame()

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.floor("D")
    df = df.dropna(subset=[dcol])
    df["date"] = df[dcol].dt.date

    daily = (df.groupby(["date","topic"], as_index=False)["topic_prob"].sum()
               .rename(columns={"topic_prob":"soft_mass"}))
    total = (daily.groupby("date", as_index=False)["soft_mass"].sum()
               .rename(columns={"soft_mass":"total_soft"}))
    daily = daily.merge(total, on="date")
    daily["soft_prop"] = daily["soft_mass"] / daily["total_soft"]

    pivot = daily.pivot(index="date", columns="topic", values="soft_prop").fillna(0).sort_index()

    entropy_full = pivot.apply(lambda r: _entropy_arr(r.values), axis=1)
    pivot_ex = pivot.drop(columns=-1) if (-1 in pivot.columns) else pivot
    entropy_ex = pivot_ex.apply(lambda r: _entropy_arr(r.values), axis=1)
    hhi_full = (pivot**2).sum(axis=1)
    hhi_ex   = (pivot_ex**2).sum(axis=1)

    dates = pivot.index.to_list()
    kl_vals = [0.0]
    for i in range(1, len(dates)):
        kl_vals.append(_kl(pivot.loc[dates[i]].values, pivot.loc[dates[i-1]].values))
    kl_df = pd.DataFrame({"date": dates, "kl_topic_shift": kl_vals}).set_index("date")

    # Top-K(의미있는 것 우선, 부족시 전체로 채움)
    recs = []
    for d, g in daily.groupby("date"):
        meaningful = g[g["topic"] != -1].sort_values("soft_mass", ascending=False)
        fallback   = g.sort_values("soft_mass", ascending=False)
        topk = meaningful.head(top_k)
        if len(topk) < top_k:
            need = top_k - len(topk)
            extras = fallback.loc[~fallback.index.isin(topk.index)].head(need)
            topk = pd.concat([topk, extras], ignore_index=True)
        row = {"date": d}
        for i, r in enumerate(topk.itertuples(index=False), start=1):
            row[f"top{i}_topic"] = r.topic
            row[f"top{i}_soft_prop"] = r.soft_prop
        row["is_top1_unassigned"] = int(row.get("top1_topic", None) == -1)
        recs.append(row)
    topk_df = pd.DataFrame(recs).set_index("date").sort_index()

    # 변화/차이
    topk_df["top1_topic_change_flag"] = (topk_df["top1_topic"] != topk_df["top1_topic"].shift(1)).astype(int)
    if len(topk_df) > 0:
        topk_df.iloc[0, topk_df.columns.get_loc("top1_topic_change_flag")] = 0
    for i in range(1, top_k + 1):
        col = f"top{i}_soft_prop"
        topk_df[f"{col}_diff"] = topk_df[col] - topk_df[col].shift(1)

    # 트렌드/모멘텀
    entropy_full_series = entropy_full.sort_index()
    hhi_full_series     = hhi_full.sort_index()
    ent_roll_prev = entropy_full_series.rolling(trend_window, min_periods=1).mean().shift(1)
    hhi_roll_prev = hhi_full_series.rolling(trend_window, min_periods=1).mean().shift(1)
    entropy_trend = (entropy_full_series - ent_roll_prev)
    hhi_trend     = (hhi_full_series - hhi_roll_prev)
    concentration_momentum = hhi_full_series.pct_change().fillna(0)

    # New emergence
    rolling_topic_mean = pivot.rolling(emergence_window, min_periods=1).mean().shift(1)
    em_recs = []
    for d in topk_df.index:
        flags = {}
        for i in range(1, top_k + 1):
            t = topk_df.loc[d, f"top{i}_topic"] if f"top{i}_topic" in topk_df.columns else np.nan
            prop = topk_df.loc[d, f"top{i}_soft_prop"] if f"top{i}_soft_prop" in topk_df.columns else 0.0
            if pd.isna(t):
                flags[f"top{i}_new_emerge"] = 0
            else:
                prev_mean = 0.0
                if (t in rolling_topic_mean.columns) and (d in rolling_topic_mean.index):
                    prev_mean = rolling_topic_mean.loc[d, t]
                if prev_mean < 1e-6:
                    flags[f"top{i}_new_emerge"] = int(prop > 0.01)
                else:
                    flags[f"top{i}_new_emerge"] = int(prop > emergence_th * prev_mean)
        flags["date"] = d
        em_recs.append(flags)
    emergence_df = pd.DataFrame(em_recs).set_index("date")

    feat = pd.DataFrame(index=pivot.index)
    feat = feat.join(pd.DataFrame({
        "entropy_full": entropy_full,
        "entropy_excl_unassigned": entropy_ex.reindex(feat.index),
        "hhi_full": hhi_full,
        "hhi_excl_unassigned": hhi_ex.reindex(feat.index),
    }), how="left")
    feat = feat.join(kl_df, how="left").join(topk_df, how="left").join(emergence_df, how="left")
    feat["entropy_trend"] = entropy_trend
    feat["hhi_trend"] = hhi_trend
    feat["concentration_momentum"] = concentration_momentum
    feat = feat.fillna(0).reset_index().rename(columns={"index":"date"})
    feat["date"] = pd.to_datetime(feat["date"])

    # KOSPI 달력으로 좌조인
    cal = pd.DataFrame({"date": pd.to_datetime(kospi_dates)}).sort_values("date")
    out = cal.merge(feat, on="date", how="left").fillna(0)
    return out

# ───────────────────────── Policy flags
def build_policy_flags(kospi_dates: pd.Series, lag_days: int = 5) -> pd.DataFrame:
    ann_path = ANL / "krw_fed_rate_announcements.csv"
    if not ann_path.exists():
        print(f"ℹ️ 발표일 파일 없음 → 정책 플래그 생략: {ann_path}")
        return pd.DataFrame(columns=["date"])
    fr = read_csv_any(ann_path)
    if "date" not in fr.columns:
        fr = fr.rename(columns={find_date_col(fr): "date"})
    fr["date"] = pd.to_datetime(fr["date"], errors="coerce").dt.floor("D")
    fr = fr.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    cal = pd.DataFrame({"date": pd.to_datetime(kospi_dates).dt.floor("D")}).drop_duplicates().sort_values("date")
    out = cal.copy()
    out["rate_announce"] = (out["date"].isin(fr["date"])).astype(int)

    for i in range(1, lag_days + 1):
        out[f"rate_announce_lag_{i}d"] = out["rate_announce"].shift(i).fillna(0).astype(int)

    days, since = [], None
    for v in out["rate_announce"]:
        if v == 1:
            since = 0
        else:
            since = (since + 1) if since is not None else None
        days.append(0 if since is None else since)
    out["days_since_rate_announce"] = pd.Series(days, dtype=float)
    out["rate_announce_decay"] = np.exp(-out["days_since_rate_announce"].fillna(999) / 3.0)

    # 호환용
    out["policy_rate_change"] = out["rate_announce"]
    return out

# ───────────────────────── Technical features
def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_s = series.ewm(span=span_short, adjust=False).mean()
    ema_l = series.ewm(span=span_long,  adjust=False).mean()
    macd  = ema_s - ema_l
    sig   = macd.ewm(span=span_signal, adjust=False).mean()
    hist  = macd - sig
    return macd, sig, hist

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up    = delta.clip(lower=0.0)
    down  = -delta.clip(upper=0.0)
    roll_up   = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs  = roll_up / roll_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(method="ffill")

def bb_width(series: pd.Series, window: int = 20) -> pd.Series:
    ma = series.rolling(window, min_periods=1).mean()
    sd = series.rolling(window, min_periods=1).std()
    return (ma + 2*sd - (ma - 2*sd)) / (ma.abs() + 1e-9)

def add_refined(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    # 모든 logret_* 모멘텀/변동성
    for col in [c for c in out.columns if c.startswith("logret_")]:
        out[f"{col}_momentum_3d"] = out[col].rolling(3, min_periods=1).sum()
        out[f"{col}_vol_5d"]      = out[col].rolling(5, min_periods=1).std()

    # 가격 기반
    if "close" in out.columns:
        p = out["close"]
        out["ma_5"] = p.rolling(5, min_periods=1).mean()
        out["ma_20"] = p.rolling(20, min_periods=1).mean()
        out["ma_diff_5_20"]  = out["ma_5"] - out["ma_20"]
        out["ma_ratio_5_20"] = out["ma_5"] / out["ma_20"].replace(0, np.nan)
        macd_line, signal_line, macd_hist = compute_macd(p)
        out["macd_line"]      = macd_line
        out["signal_line"]    = signal_line
        out["macd_histogram"] = macd_hist

        # 가격 모멘텀
        if "logret_KOSPI_Close" in out.columns:
            out["price_momentum_3d"] = out["logret_KOSPI_Close"].rolling(3, min_periods=1).sum()
        elif "logret_next" in out.columns:
            out["price_momentum_3d"] = out["logret_next"].rolling(3, min_periods=1).sum()

        out["rsi_14"]      = rsi(p, 14)
        out["bb_width_20"] = bb_width(p, 20)

        # return_z_20
        mean20 = out["logret_KOSPI_Close"].rolling(20, min_periods=5).mean()
        std20  = out["logret_KOSPI_Close"].rolling(20, min_periods=5).std()
        out["return_z_20"] = (out["logret_KOSPI_Close"] - mean20) / std20.replace(0, np.nan)

        # 변동성 관련
        out["vol_5d"]  = out["logret_KOSPI_Close"].rolling(5,  min_periods=2).std()
        out["vol_20d"] = out["logret_KOSPI_Close"].rolling(20, min_periods=5).std()
        out["vol_ratio_5_20"] = out["vol_5d"] / out["vol_20d"].replace(0, np.nan)

    # 감성 × 가격/환율
    if "sentiment_score" in out.columns:
        pr = "logret_KOSPI_Close" if "logret_KOSPI_Close" in out.columns else None
        if pr:
            out["sentiment_x_price_logret"] = out["sentiment_score"] * out[pr]
        if "logret_USD_KRW" in out.columns:
            out["sentiment_x_usdkrw_logret"] = out["sentiment_score"] * out["logret_USD_KRW"]

    # 토픽 × 감성
    if "top1_topic_change_flag" in out.columns and "sentiment_surprise_5d" in out.columns:
        out["top1_topic_change_flag_x_sentiment_surprise"] = (
            out["top1_topic_change_flag"] * out["sentiment_surprise_5d"]
        )

    # 상대 모멘텀
    if "logret_USD_KRW_momentum_3d" in out.columns and "logret_CNY_KRW_momentum_3d" in out.columns:
        out["relative_currency_momentum"] = (
            out["logret_USD_KRW_momentum_3d"] - out["logret_CNY_KRW_momentum_3d"]
        )

    # 코스피 변동성 및 비율
    if "logret_KOSPI_Close" in out.columns:
        out["kospi_vol_5d"] = out["logret_KOSPI_Close"].rolling(5, min_periods=2).std()
        if "logret_USD_KRW_vol_5d" in out.columns:
            out["vol_ratio_kospi_usdkrw"] = out["kospi_vol_5d"] / out["logret_USD_KRW_vol_5d"].replace(0, np.nan)

    return out

# ───────────────────────── Column ordering (exactly as requested)
FINAL_COLS = [
    "date","close","next_close","logret_next","label_up",
    "is_pre_holiday","is_post_holiday","days_since_last_trade",
    "entropy_full","entropy_excl_unassigned","hhi_full","hhi_excl_unassigned","kl_topic_shift",
    "top1_topic","top1_soft_prop","top2_topic","top2_soft_prop","top3_topic","top3_soft_prop",
    "is_top1_unassigned","top1_topic_change_flag","top1_soft_prop_diff","top2_soft_prop_diff","top3_soft_prop_diff",
    "top1_new_emerge","top2_new_emerge","top3_new_emerge",
    "entropy_trend","hhi_trend","concentration_momentum",
    "pos_prob","neg_prob","sentiment_score",
    "rate_announce","rate_announce_lag_1d","rate_announce_lag_2d","rate_announce_lag_3d","rate_announce_lag_4d","rate_announce_lag_5d",
    "logret_KOSPI_Close",
    "KOSPI_Volatility","logret_KOSPI_Volatility",
    "USD_KRW","logret_USD_KRW","USD_JPY","logret_USD_JPY","JPY_KRW","logret_JPY_KRW","EUR_KRW","logret_EUR_KRW","CNY_KRW","logret_CNY_KRW",
    "USD_KRW_derived",
    "sentiment_momentum_1d","sentiment_momentum_3d","sentiment_roll5_mean_prev","sentiment_surprise_5d","sentiment_vol_5d",
    "logret_next_momentum_3d","logret_next_vol_5d",
    "logret_KOSPI_Close_momentum_3d","logret_KOSPI_Close_vol_5d",
    "logret_KOSPI_Volatility_momentum_3d","logret_KOSPI_Volatility_vol_5d",
    "logret_USD_KRW_momentum_3d","logret_USD_KRW_vol_5d",
    "logret_USD_JPY_momentum_3d","logret_USD_JPY_vol_5d",
    "logret_JPY_KRW_momentum_3d","logret_JPY_KRW_vol_5d",
    "logret_EUR_KRW_momentum_3d","logret_EUR_KRW_vol_5d",
    "logret_CNY_KRW_momentum_3d","logret_CNY_KRW_vol_5d",
    "ma_5","ma_20","ma_diff_5_20","ma_ratio_5_20","macd_line","signal_line","macd_histogram","price_momentum_3d",
    "days_since_rate_announce","rate_announce_decay",
    "sentiment_x_price_logret","sentiment_x_usdkrw_logret",
    "top1_topic_change_flag_x_sentiment_surprise","relative_currency_momentum",
    "kospi_vol_5d","vol_ratio_kospi_usdkrw",
    "dow_sin","dow_cos","rsi_14","bb_width_20",
    "return_z_20","vol_5d","vol_20d","vol_ratio_5_20","sentiment_z_30",
]

def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 누락된 타깃 컬럼은 채워 넣고, 초과 컬럼은 버림
    for c in FINAL_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[FINAL_COLS]
    return out

# ───────────────────────── Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(PROC / "training_with_refined_features.csv"))
    ap.add_argument("--news-path", type=str, default=str(NEWS / "news_with_sectors.csv"))
    ap.add_argument("--sentiment-path", type=str, default=str(RAW / "daily_sentiment_features_finbert_2021_2025.csv"))
    args = ap.parse_args()

    print("▶ 전처리 시작 (KOSPI 달력 좌조인 & robust parsing)")

    kospi = build_kospi_base()
    kospi_dates = kospi["date"]

    macro  = build_macro_fx_vol(kospi_dates)
    senti  = build_sentiment(Path(args.sentiment_path), kospi_dates)
    topics = build_topic_features(Path(args.news_path), kospi_dates)
    policy = build_policy_flags(kospi_dates, lag_days=5)

    merged = kospi.copy()
    for extra in [topics, senti, policy, macro]:
        if extra is not None and not extra.empty:
            merged = merged.merge(extra, on="date", how="left")

    # 전부 NaN인 컬럼 제거(중간)
    nan_cols = [c for c in merged.columns if c != "date" and merged[c].isna().all()]
    if nan_cols:
        print(f"⚠️ 전부 NaN 컬럼 제거: {nan_cols}")
        merged = merged.drop(columns=nan_cols)

    # 추가 파생
    refined = add_refined(merged)

    # 학습 타겟 유효 행만 유지
    refined = refined.dropna(subset=["label_up"]).sort_values("date").reset_index(drop=True)

    # 최종 칼럼 정렬/서브셋
    refined = finalize_columns(refined)

    out_path = Path(args.out)
    refined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"🎉 Saved: {out_path} (shape={refined.shape})")

if __name__ == "__main__":
    main()
