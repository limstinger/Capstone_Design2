# -*- coding: utf-8 -*-
# Capstone_Design/models/pipeline/news_indicators.py
# 오늘(KST) 뉴스(네이버) -> 감성분석(KR-FinBert) -> 섹터/토픽(BERTopic: 필수)
# 저장(Parquet partition + DuckDB BY NAME upsert) + RAW CSV 업서트(전처리 입력 호환)
# + 뷰 등록(bronze.*_view) + 간단 검증/메트릭

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_cls

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# 경로/환경
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # .../Capstone_Design
LAKE = ROOT / "data" / "lake" / "bronze"    # 파티션 Parquet
WARE = ROOT / "data" / "warehouse"          # DuckDB 웨어하우스
RAW  = ROOT / "data" / "raw"                # preprocess.py 호환용 미러 CSV
DBG  = ROOT / "models" / "pipeline" / "debug_html"
for p in [LAKE, WARE, RAW, DBG]:
    p.mkdir(parents=True, exist_ok=True)

# DuckDB 단일 파일(통일)
DUCK = WARE / "warehouse.duckdb"

# sector_mapping.py가 보통 있는 위치를 import 경로에 추가
SECTOR_DIR = ROOT / "models" / "pipeline" / "sector_mapping_model"
if SECTOR_DIR.exists():
    sys.path.insert(0, str(SECTOR_DIR))
    print(f"[import] added to sys.path: {SECTOR_DIR}")
else:
    print(f"[import][WARN] sector dir not found: {SECTOR_DIR}")

load_dotenv(ROOT / ".env")

KST = timezone(timedelta(hours=9))
TODAY = datetime.now(KST).date()
STAMP_UTC = lambda: datetime.utcnow().isoformat() + "Z"

# ─────────────────────────────────────────────────────────────
# 공통 유틸: Parquet 저장(원자적) / DuckDB 업서트(BY NAME) / RAW CSV 업서트 /
#            뷰 등록 / 간단 검증 / 운영 메트릭
# ─────────────────────────────────────────────────────────────
def write_parquet_partition(df: pd.DataFrame, dataset: str, date_col: str, mode: str = "overwrite"):
    """data/lake/bronze/<dataset>/date=YYYY-MM-DD/part.parquet 저장(원자적)."""
    if df is None or df.empty:
        print(f"[SKIP] {dataset}: empty"); return None
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].dropna().empty:
        print(f"[SKIP] {dataset}: all {date_col}=NaT"); return None
    d = df[date_col].dropna().max().date().isoformat()
    out_dir = LAKE / dataset / f"date={d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part.parquet"
    tmp = out_dir / ".__tmp.parquet"
    if mode == "skip_if_exists" and path.exists():
        print(f"[SKIP] {dataset}: already exists @ {path}"); return path
    if tmp.exists(): tmp.unlink()
    df.to_parquet(tmp, index=False)
    if path.exists(): path.unlink()
    tmp.replace(path)
    print(f"[OK] {dataset} → {path} rows={len(df)}")
    return path

def duck_upsert_by_name(table: str, df: pd.DataFrame, key_cols=("date",), schema: str = "bronze"):
    """단일 DuckDB 웨어하우스(DUCK) schema=bronze에 BY NAME 업서트."""
    import duckdb
    if df is None or df.empty:
        print(f"[DuckDB] skip {schema}.{table}: empty"); return
    con = duckdb.connect(str(DUCK))
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    con.register("df_in", df)
    # 최초 1회: df 스키마 그대로 테이블 생성
    con.execute(f"CREATE TABLE IF NOT EXISTS {schema}.{table} AS SELECT * FROM df_in WHERE 1=0;")
    cond = " AND ".join([f"t.{k} = s.{k}" for k in key_cols])
    if cond:
        con.execute(f"DELETE FROM {schema}.{table} t USING df_in s WHERE {cond};")
    con.execute(f"INSERT INTO {schema}.{table} BY NAME SELECT * FROM df_in;")
    con.unregister("df_in")
    con.close()
    print(f"[DuckDB] upserted → {schema}.{table} rows={len(df)}")

def _upsert_csv_by_keys(path: Path, df_new: pd.DataFrame, keys: list, sort_by: str):
    """RAW CSV 누적 + 키 중복 제거 + 정렬."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df_old = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        # 키 타입 정규화
        for k in keys:
            if k in df_old.columns and k in df_new.columns:
                df_old[k] = df_old[k].astype(str)
                df_new[k] = df_new[k].astype(str)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=keys, keep="last")
        after = len(df_all)
        print(f"[RAW] dedup by {keys}: {before} -> {after}")
    else:
        df_all = df_new.copy()

    if sort_by in df_all.columns:
        # adjustedDate는 YYYY.MM.DD → 안전하게 datetime으로 정렬
        if sort_by == "adjustedDate":
            sorter = pd.to_datetime(df_all[sort_by], errors="coerce", format="%Y.%m.%d")
        else:
            sorter = pd.to_datetime(df_all[sort_by], errors="coerce")
        df_all = df_all.assign(__sort=sorter).sort_values(["__sort", sort_by], na_position="last").drop(columns="__sort")

    df_all.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[RAW] upserted → {path} rows={len(df_all)}")

def register_duckdb_views_news():
    """Parquet 파티션을 읽는 bronze 뷰를 생성/갱신."""
    import duckdb
    con = duckdb.connect(str(DUCK))
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    # 저장하는 데이터셋 이름에 맞춰 추가
    for ds in ["news", "news_sentiment", "news_sentiment_daily"]:
        pattern = str((LAKE / ds / "date=*/part.parquet")).replace("\\", "/")
        con.execute(f"CREATE OR REPLACE VIEW bronze.{ds}_view AS SELECT * FROM read_parquet('{pattern}');")
    con.close()
    print("[OK] DuckDB views updated (news)")

def validate_news_daily(df: pd.DataFrame):
    """간단 데이터 계약(경고만): 스키마/범위/결측 체크."""
    errs = []
    if "date" not in df.columns: errs.append("date missing")
    for c in ["n","pos","neg","neu","sent","strong_ratio"]:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            errs.append(f"{c} not numeric")
    if "n" in df.columns and (df["n"] < 0).any(): errs.append("n < 0 exists")
    if errs:
        print("[VALIDATE][WARN] news_daily:", "; ".join(errs))

def log_ops(stage: str, rows: int, ok: bool, extra: dict = None):
    """간단 운영 메트릭 적재(duckdb: ops_metrics)."""
    import duckdb, datetime as dt
    con = duckdb.connect(str(DUCK))
    con.execute("""
        CREATE TABLE IF NOT EXISTS ops_metrics(
            ts TIMESTAMP, stage VARCHAR, rows BIGINT, ok BOOLEAN, extra VARCHAR
        );
    """)
    con.execute("INSERT INTO ops_metrics VALUES (?, ?, ?, ?, ?)",
                [dt.datetime.utcnow(), stage, int(rows), bool(ok), json.dumps(extra or {})])
    con.close()

# ─────────────────────────────────────────────────────────────
# 0) 피클 호환: __main__ 심볼을 sector_mapping의 심볼로 주입
# ─────────────────────────────────────────────────────────────
def ensure_sector_pickle_symbols():
    try:
        import sector_mapping as sm  # 사용자 제공 모듈
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            if not hasattr(main_mod, "okt_tokenizer") and hasattr(sm, "okt_tokenizer"):
                setattr(main_mod, "okt_tokenizer", sm.okt_tokenizer)
                print("[compat] injected __main__.okt_tokenizer")
            if not hasattr(main_mod, "korean_vectorizer") and hasattr(sm, "korean_vectorizer"):
                setattr(main_mod, "korean_vectorizer", sm.korean_vectorizer)
                print("[compat] injected __main__.korean_vectorizer")
    except Exception as e:
        print("[compat] sector_mapping import/inject failed:", e)

# ─────────────────────────────────────────────────────────────
# 1) 뉴스 수집
# ─────────────────────────────────────────────────────────────
def fetch_news_for_day(query="코스피", ds_str=None, ymd=None, debug_folder=DBG):
    try:
        import undetected_chromedriver as uc
        from selenium.webdriver.chrome.options import Options
    except Exception as e:
        print("[SELENIUM] 미설치:", e)
        return pd.DataFrame()

    debug_folder.mkdir(parents=True, exist_ok=True)
    opts = Options()
    try: opts.add_argument("--headless=new")
    except Exception: opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")

    try:
        driver = uc.Chrome(options=opts)
    except Exception:
        driver = uc.Chrome(options=opts, version_main=120)

    url = (
        f"https://search.naver.com/search.naver?where=news&query={query}"
        f"&sort=1&pd=3&ds={ds_str}&de={ds_str}&nso=so:dd,p:from{ymd}to{ymd},a:all"
    )
    print(f"[naver] GET {url}")
    driver.get(url)
    time.sleep(2)

    last_h = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.8)
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h: break
        last_h = new_h

    html = driver.page_source
    debug_path = debug_folder / f"debug_{ymd}.html"
    debug_path.write_text(html, encoding="utf-8")
    print(f"[debug] saved -> {debug_path}]")

    soup = BeautifulSoup(html, "html.parser")
    hits = []
    for span in soup.select("span[class*='headline']"):
        title = span.get_text(strip=True)
        a_tag = span.find_parent("a", href=True)
        if not title or not a_tag: continue
        link = a_tag["href"]
        press = ""
        body = span.find_next("span", class_=lambda v: v and "body2" in v)
        if body: press = body.get_text(strip=True)
        hits.append((title, link, press))
    driver.quit()

    print(f"[collect] hits={len(hits)}")
    if hits:
        print("  ↳ sample titles:")
        for i, (t, _, p) in enumerate(hits[:5], 1):
            print(f"    {i:>2}. {t[:90]} ({p})")

    rows = []
    for title, link, press in hits:
        pub = get_pub_date(link)
        rows.append({
            "url": link,
            "pubDate": pub,
            "adjustedDate": ds_str,   # 수집 기준일(YYYY.MM.DD)
            "headline": title,
            "press": press,
            "fetchedAt": STAMP_UTC()
        })
    df = pd.DataFrame(rows).drop_duplicates("url").reset_index(drop=True)
    print(f"[collect] unique urls={len(df)}")
    return df

def get_pub_date(url: str):
    try:
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "ko-KR,ko;q=0.9"
        }, timeout=6)
        soup = BeautifulSoup(r.text, "html.parser")
        meta = soup.find("meta", {"property": "og:article:published_time"})
        if meta and meta.get("content"): return meta["content"]
        meta2 = soup.find("meta", {"itemprop": "datePublished"})
        if meta2 and meta2.get("content"): return meta2["content"]
        return None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# 2) 감성분석 (KR-FinBert)
# ─────────────────────────────────────────────────────────────
def run_sentiment_krfinbert(headlines: pd.Series,
                            temperature: float = 0.5,
                            binary_only: bool = True,
                            batch_size: int = 64,
                            dead_zone: float = 0.05) -> pd.DataFrame:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

    model_id = os.getenv("KR_FINBERT_MODEL", "snunlp/KR-FinBert-SC")
    tok  = AutoTokenizer.from_pretrained(model_id)
    cfg  = AutoConfig.from_pretrained(model_id)
    mdl  = AutoModelForSequenceClassification.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()

    texts = headlines.fillna("").astype(str).tolist()
    print(f"[finbert] n_texts={len(texts)} device={device} model={model_id}")
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            all_logits.append(out.logits.detach().cpu().numpy())
            print(f"[finbert] batch {i//batch_size+1}: size={len(batch)}")

    if len(all_logits) == 0:
        print("[finbert] no logits")
        return pd.DataFrame(columns=["label","score_pos","score_neu","score_neg","confidence","sentiment_score"])

    logits = np.concatenate(all_logits, axis=0)
    z = logits / max(1e-6, float(temperature))
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)

    id2label = {i: cfg.id2label[i].lower() for i in range(mdl.config.num_labels)}
    label2idx = {v: k for k, v in id2label.items()}
    p_idx = label2idx.get("positive", 0)
    n_idx = label2idx.get("negative", 2)
    u_idx = label2idx.get("neutral",  1)

    pos = probs[:, p_idx]
    neu = probs[:, u_idx] if u_idx < probs.shape[1] else 0.0*pos
    neg = probs[:, n_idx]

    if binary_only:
        denom = pd.Series(pos + neg).clip(lower=1e-12).values
        pos = pos / denom
        neg = neg / denom
        neu = 0.0 * pos

    score = (pos - neg).astype(float)
    conf  = (abs(score)).astype(float)
    labels = pd.Series(score).apply(lambda x: "neutral" if abs(x) < dead_zone else ("positive" if x > 0 else "negative")).values

    df = pd.DataFrame({
        "label": labels.astype(str),
        "score_pos": pos.astype(float),
        "score_neu": neu.astype(float),
        "score_neg": neg.astype(float),
        "confidence": conf.astype(float),
        "sentiment_score": score.astype(float)
    })
    print(f"[finbert] done. mean_pos={df['score_pos'].mean():.4f} mean_neg={df['score_neg'].mean():.4f} mean_sent={df['sentiment_score'].mean():.4f}")
    return df

# ─────────────────────────────────────────────────────────────
# 3) 섹터/토픽 모델 로드 & 예측 — BERTopic (필수)
# ─────────────────────────────────────────────────────────────
def load_sector_model():
    """BERTopic 모델 로드: 학습 시 calculate_probabilities=True & prediction_data=True 권장."""
    ensure_sector_pickle_symbols()
    from bertopic import BERTopic
    from joblib import load as joblib_load

    candidates = [
        ROOT / "models" / "pipeline" / "sector_mapping_model" / "output" / "sector_model.pkl",
        ROOT / "models" / "pipeline" / "sector_mapping_model" / "sector_model.pkl",
        ROOT / "models" / "pipeline" / "sector_mapping_model" / "BERTopic_model.pkl",
        ROOT / "models" / "pipeline" / "sector_mapping_model" / "sector_model.joblib",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if not model_path:
        raise SystemExit("[ERROR] BERTopic 모델을 찾을 수 없습니다. sector_mapping_model/output/sector_model.pkl 확인하세요.")

    try:
        if model_path.suffix.lower() == ".pkl":
            mdl = BERTopic.load(str(model_path))
            print(f"[sector] BERTopic.load OK: {model_path}")
        else:
            mdl = joblib_load(model_path)
            if not hasattr(mdl, "transform"):
                raise RuntimeError("Loaded model does not support transform().")
            print(f"[sector] joblib.load OK: {model_path}")
        return mdl
    except Exception as e:
        raise SystemExit(f"[ERROR] BERTopic 모델 로드 실패: {e}")

def predict_topics_with_probs(topic_model, texts: pd.Series) -> pd.DataFrame:
    """
    BERTopic 전용:
    - topics, probs = model.transform(X)
    - topic_prob = P[i, topics[i]]  (배정된 토픽의 확률)
    - probs가 None이면 approximate_distribution(X) 보완
    """
    import numpy as np
    X = texts.fillna("").astype(str).tolist()
    topics, probs = topic_model.transform(X)

    # topics 정규화
    topics_arr = np.array([(-1 if t is None else int(t)) for t in topics], dtype=int)
    n = len(topics_arr)

    # 확률 행렬 확보
    P = None
    if probs is None:
        try:
            P = topic_model.approximate_distribution(X)
            print("[BERTopic] used approximate_distribution (transform probs=None)")
        except Exception as e:
            print("[BERTopic] approximate_distribution failed:", e)
    else:
        if isinstance(probs, list):
            try:
                P = np.vstack([np.asarray(pi).ravel() if pi is not None else np.zeros(1) for pi in probs])
            except Exception as e:
                print("[BERTopic] vstack probs(list) failed:", e)
                P = None
        else:
            P = np.asarray(probs)

    topic_prob = np.full(n, 0.0, dtype=float)
    if P is not None and P.ndim == 2:
        n_docs, n_topics = P.shape
        m = min(n_docs, n)
        rows = np.arange(m)
        valid = (topics_arr[:m] >= 0) & (topics_arr[:m] < n_topics)
        topic_prob[:m][valid] = P[rows[valid], topics_arr[:m][valid]]
        # 남은 NaN은 row-wise max로 보완
        rowmax = P.max(axis=1)
        for i in range(m):
            if topic_prob[i] == 0.0 and n_topics > 0:
                topic_prob[i] = float(rowmax[i])

    out = {"topic": topics_arr.tolist(), "topic_prob": topic_prob.astype(float).tolist()}
    # 토픽 이름 매핑(있으면)
    try:
        info = topic_model.get_topic_info()
        if isinstance(info, pd.DataFrame) and "Topic" in info.columns and "Name" in info.columns:
            id2name = {int(r["Topic"]): r["Name"] for _, r in info.iterrows()}
            out["topic_name"] = [id2name.get(int(t), None) if t >= 0 else None for t in topics_arr]
    except Exception:
        pass

    # (선택) sector_mapping.py 매핑 적용
    try:
        import sector_mapping as sm
        mapped = None
        if hasattr(sm, "map_topic_to_sector"):
            mapped = sm.map_topic_to_sector(out["topic"])
        elif hasattr(sm, "topic_to_sector") and isinstance(sm.topic_to_sector, dict):
            mapped = [sm.topic_to_sector.get(t, None) for t in out["topic"]]
        elif hasattr(sm, "TOPIC_TO_SECTOR") and isinstance(sm.TOPIC_TO_SECTOR, dict):
            mapped = [sm.TOPIC_TO_SECTOR.get(t, None) for t in out["topic"]]
        if mapped is not None:
            out["sector"] = [str(x) if x is not None else None for x in mapped]
            print(f"[sector-map] applied mapping (mapped={sum(x is not None for x in out['sector'])}/{len(out['topic'])})")
    except Exception as e:
        print("[sector-map] mapping not applied:", e)

    return pd.DataFrame(out)

# ─────────────────────────────────────────────────────────────
# 4) 하루 단위 파이프라인 실행 (오늘/백필)
# ─────────────────────────────────────────────────────────────
def run_for_date(target_day: date_cls, query="코스피", topic_model=None):
    if topic_model is None:
        raise SystemExit("[ERROR] BERTopic 모델이 필요합니다. --no-sector 옵션은 지원하지 않습니다.")

    ds_str = target_day.strftime("%Y.%m.%d")
    ymd = target_day.strftime("%Y%m%d")
    print(f"\n=== RUN {target_day.isoformat()} (KST) ===")

    # 1) 뉴스 수집
    raw = fetch_news_for_day(query=query, ds_str=ds_str, ymd=ymd)
    if raw.empty:
        print("[RUN] no news, skip the day.")
        log_ops("news_collect", 0, True, {"date": target_day.isoformat()})
        return
    print(f"[news] rows={len(raw)}  columns={list(raw.columns)}")
    print(raw[["headline","url"]].head(5).to_string(index=False))

    # 내부 처리용 날짜(정오 고정, tz 제거 전 Parquet 저장)
    raw["date"] = pd.to_datetime(datetime(target_day.year, target_day.month, target_day.day, 12, 0, 0, tzinfo=KST))
    write_parquet_partition(raw, "news", date_col="date", mode="overwrite")
    duck_upsert_by_name("raw_news", raw.drop(columns=["date"], errors="ignore"), key_cols=("url",), schema="bronze")
    log_ops("news_collect", len(raw), True, {"date": target_day.isoformat()})

    # 2) 감성분석
    sent = run_sentiment_krfinbert(raw["headline"])
    news_with_sent = pd.concat([raw.reset_index(drop=True), sent], axis=1)
    print(f"[sentiment] merged rows={len(news_with_sent)}  cols={list(news_with_sent.columns)}")
    write_parquet_partition(news_with_sent, "news_sentiment", date_col="date", mode="overwrite")
    duck_upsert_by_name("news_sentiment_headlines", news_with_sent.drop(columns=["date"], errors="ignore"),
                        key_cols=("url",), schema="bronze")
    log_ops("news_sentiment", len(news_with_sent), True, {"date": target_day.isoformat()})

    # 3) 일별 집계
    agg = (
        news_with_sent.assign(date_only=news_with_sent["date"].dt.date)
        .groupby("date_only", as_index=False)
        .agg(
            n=("label", "count"),
            pos=("score_pos", lambda s: float(pd.Series(s, dtype="float64").mean())),
            neu=("score_neu", lambda s: float(pd.Series(s, dtype="float64").mean())),
            neg=("score_neg", lambda s: float(pd.Series(s, dtype="float64").mean())),
            sent=("sentiment_score", lambda s: float(pd.Series(s, dtype="float64").mean())),
            strong_ratio=("sentiment_score", lambda s: float((pd.Series(s, dtype="float64").abs() >= 0.20).mean())),
        )
        .rename(columns={"date_only": "date"})
    )
    agg["date"] = pd.to_datetime(agg["date"])
    validate_news_daily(agg)
    print(f"[daily]\n{agg.to_string(index=False)}")
    write_parquet_partition(agg, "news_sentiment_daily", date_col="date", mode="overwrite")
    # DuckDB: 타입 보존 BY NAME 업서트
    duck_upsert_by_name("news_sentiment_daily", agg, key_cols=("date",), schema="bronze")
    log_ops("news_daily", len(agg), True, {"date": target_day.isoformat()})

    # RAW CSV 업서트(전처리 입력)
    daily_raw = agg.rename(columns={
        "pos": "pos_prob",
        "neg": "neg_prob",
        "sent": "sentiment_score",
    })[["date","pos_prob","neg_prob","sentiment_score"]].copy()
    daily_raw["date"] = pd.to_datetime(daily_raw["date"]).dt.strftime("%Y-%m-%d")
    _upsert_csv_by_keys(
        RAW / "daily_sentiment_features_finbert_2021_2025.csv",
        daily_raw,
        keys=["date"],
        sort_by="date"
    )
    print(f"[RAW][daily_sentiment] appended:\n{daily_raw.tail(1).to_string(index=False)}")

    # 4) 섹터/토픽 — 필수
    sect_df = predict_topics_with_probs(topic_model, news_with_sent["headline"])
    if sect_df.empty or not {"topic","topic_prob"}.issubset(sect_df.columns):
        raise SystemExit("[ERROR] 토픽/확률 추출 실패. BERTopic 모델 설정을 확인하세요.")

    news_with_sector = pd.concat([news_with_sent.reset_index(drop=True), sect_df], axis=1)

    # 최종 CSV 스키마(전처리 호환, adjustedDate 정렬/키)
    nw = news_with_sector.copy()
    if "adjustedDate" not in nw.columns:
        nw["adjustedDate"] = ds_str

    cols_out = ["adjustedDate", "url", "headline", "topic", "topic_prob"]
    for c in cols_out:
        if c not in nw.columns:
            nw[c] = None
    nw = (
        nw[cols_out]
        .dropna(subset=["url"])
        .drop_duplicates(subset=["adjustedDate","url"], keep="last")
    )

    print("[news_with_sectors] preview:")
    print(nw.head(5).to_string(index=False))

    _upsert_csv_by_keys(
        RAW / "news_with_sectors.csv",
        nw,
        keys=["adjustedDate", "url"],
        sort_by="adjustedDate"
    )
    print(f"[RAW][news_with_sectors] wrote rows={len(nw)}  adjustedDate_last={nw['adjustedDate'].iloc[-1] if len(nw)>0 else 'NA'}")
    log_ops("news_topics", len(nw), True, {"date": target_day.isoformat()})

# ─────────────────────────────────────────────────────────────
# 5) 메인 — 오늘 1일 or 범위 백필
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None, help="백필 시작일 YYYY-MM-DD")
    ap.add_argument("--end",   type=str, default=None, help="백필 종료일 YYYY-MM-DD (포함)")
    ap.add_argument("--query", type=str, default="코스피")
    args = ap.parse_args()

    # 섹터/토픽 모델은 필수
    topic_model = load_sector_model()

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end   = datetime.strptime(args.end,   "%Y-%m-%d").date()
        if end < start:
            raise SystemExit("end must be >= start")
        cur = start
        while cur <= end:
            run_for_date(cur, query=args.query, topic_model=topic_model)
            cur += timedelta(days=1)
    else:
        run_for_date(TODAY, query=args.query, topic_model=topic_model)

    # 마지막에 bronze 뷰 갱신(Parquet 기반 뷰)
    register_duckdb_views_news()
