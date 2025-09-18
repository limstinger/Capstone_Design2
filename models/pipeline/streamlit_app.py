# streamlit_app.py
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)
from scipy.stats import ks_2samp

# ==================== 경로 해석 ====================
def resolve_base_dir() -> Path:
    """
    Capstone_Design 루트를 parents[2]로 추정.
    (…/Capstone_Design/models/pipeline/streamlit_app.py 기준)
    폴백: parents[1] → 현재 디렉토리
    """
    here = Path(__file__).resolve()
    cands = [
        here.parents[2],  # .../Capstone_Design
        here.parents[1],  # .../Capstone_Design/models
        here.parent,      # .../Capstone_Design/models/pipeline
    ]
    for b in cands:
        if (b / "data").exists() or (b / "models").exists():
            return b
    return here.parents[2]

BASE = resolve_base_dir()
PIPELINE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE / "data"
PROC_DIR = DATA_DIR / "processed"
PRED_DIR = DATA_DIR / "predictions"
MODELS_ROOT = BASE / "models"

PROC_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = PROC_DIR / "training_with_refined_features.csv"
PRED_CSV = PRED_DIR / "next_day_predictions.csv"

# ==================== 유틸 함수 ====================
def find_latest_final_dir(models_root: Path) -> Path:
    """
    models 루트에서 가장 최신의 final_* 디렉터리를 찾는다.
    """
    patterns = ["final_wavelet_transformer_*", "final_*"]
    cands = []
    for pat in patterns:
        cands += list(models_root.glob(pat))
    if not cands:
        for pat in patterns:
            cands += list(models_root.rglob(pat))
    cands = [p for p in cands if p.is_dir()]
    if not cands:
        raise FileNotFoundError("models 폴더에서 final_* 디렉터리를 찾을 수 없습니다. 먼저 학습을 완료하세요.")
    return max(cands, key=lambda p: p.stat().st_mtime)

@st.cache_resource
def load_meta_and_names(final_dir: Path):
    """
    메타 모델(XGB)과 피처 이름, 전처리/보정 번들을 로드.
    """
    pkl = final_dir / "preproc_and_calibrators.pkl"
    xgbp = final_dir / "meta_xgb.pkl"
    bundle = joblib.load(pkl)
    meta_model: XGBClassifier = joblib.load(xgbp)

    feat_names = bundle.get("meta_feature_names", None)
    if feat_names is None:
        feat_names = getattr(meta_model, "feature_names_in_", None)
        feat_names = list(feat_names) if feat_names is not None else [f"f{i}" for i in range(meta_model.n_features_in_)]

    return meta_model, feat_names, bundle

def fmt_date_for_metric(val) -> str:
    """
    st.metric 값은 str/int/float/None만 허용 → 날짜는 문자열로 변환.
    """
    try:
        if hasattr(val, "strftime"):
            return val.strftime("%Y-%m-%d")
        return str(val)
    except Exception:
        return str(val)

# ==================== Streamlit UI ====================
st.set_page_config(page_title="KOSPI Next-Day Forecast", layout="wide")
st.sidebar.title("⚙️ 설정")

st.sidebar.caption(f"BASE: `{BASE}`")
st.sidebar.caption(f"PIPELINE_DIR: `{PIPELINE_DIR}`")

use_recalib = st.sidebar.checkbox("예측 직후 누적 라벨로 재보정(--recalibrate)", value=False)
date_override = st.sidebar.text_input("기준 날짜 YYYY-MM-DD (빈칸=CSV 마지막 날짜)", "")

colA, colB = st.sidebar.columns(2)
run_pred = colA.button("▶ 다음 영업일 예측 실행")
run_calib = colB.button("♻ 보정기 재학습(+CSV 반영)")

# -------- 예측 실행 버튼 --------
if run_pred:
    # 예측 스크립트 이름: 저장소에 맞게 사용
    script_name = "predict_next_day.py"
    if not (PIPELINE_DIR / script_name).exists():
        # predict_next_day.py가 없으면 predict_test.py로 폴백
        alt = "predict_test.py"
        if (PIPELINE_DIR / alt).exists():
            script_name = alt
        else:
            st.error(f"예측 스크립트를 찾을 수 없습니다: {script_name} / {alt}")
            st.stop()

    cmd = [
        sys.executable,
        script_name,
        "--models-root", str(MODELS_ROOT),
        "--data-csv", str(DATA_CSV),
        "--pred-out", str(PRED_CSV),
    ]
    if date_override.strip():
        cmd += ["--date", date_override.strip()]
    if use_recalib:
        cmd += ["--recalibrate"]

    with st.spinner("예측 실행 중..."):
        proc = subprocess.run(cmd, cwd=str(PIPELINE_DIR), capture_output=True, text=True)
    st.code("\n".join(["> " + " ".join(cmd), proc.stdout, proc.stderr]))
    if proc.returncode == 0:
        st.success(f"예측 완료 및 CSV 저장! → {PRED_CSV}")
    else:
        st.error("예측 실행 실패")

# -------- 보정기 재학습 버튼 --------
if run_calib:
    calib_script = "calibrate_predictions.py"
    if not (PIPELINE_DIR / calib_script).exists():
        st.error(f"보정 스크립트를 찾을 수 없습니다: {calib_script}")
        st.stop()

    cmd = [
        sys.executable, calib_script,
        "--pred-csv", str(PRED_CSV),
        "--calib-out", str(PRED_DIR / "post_calibrator.pkl"),
        "--min-n", "60",
        "--use-latest-only",
        "--write-back",
    ]
    with st.spinner("보정기 재학습 중..."):
        proc = subprocess.run(cmd, cwd=str(PIPELINE_DIR), capture_output=True, text=True)
    st.code("\n".join(["> " + " ".join(cmd), proc.stdout, proc.stderr]))
    if proc.returncode == 0:
        st.success("보정기 저장 및 CSV 갱신 완료!")
    else:
        st.error("보정기 재학습 실패")

st.title("📈 다음 영업일 KOSPI 등락 예측 대시보드")

# ==================== 데이터 로드 ====================
if not PRED_CSV.exists():
    st.info("아직 예측 결과 CSV가 없습니다. 왼쪽에서 '다음 영업일 예측 실행'을 먼저 눌러 생성하세요.")
    st.stop()

df = pd.read_csv(PRED_CSV, parse_dates=["asof_date", "pred_for"], low_memory=False)
# run_no가 없을 수도 있으니 보호
sort_cols = ["asof_date"] + (["run_no"] if "run_no" in df.columns else [])
df = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)

# 최신 실행 한 줄(각 asof_date의 최신)만 필터
if "is_latest" in df.columns:
    latest_mask = df["is_latest"].fillna(False).astype(bool)
    df_latest = df[latest_mask] if latest_mask.any() else df.groupby("asof_date", as_index=False).tail(1)
else:
    df_latest = df.groupby("asof_date", as_index=False).tail(1)

# ==================== 상단 KPI ====================
if not df_latest.empty:
    last = df_latest.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기준 날짜(asof)", fmt_date_for_metric(last["asof_date"]))
    c2.metric("예측 대상(pred_for)", fmt_date_for_metric(last["pred_for"]))
    c3.metric("최종 확률 proba_up", f"{float(last['proba_up']):.3f}" if pd.notna(last.get("proba_up", np.nan)) else "NaN")
    c4.metric("이진 예측 pred_up", "⬆️ 상승" if int(last.get("pred_up", 0)) == 1 else "⬇️ 하락")

# ==================== 시계열 플롯 ====================
st.subheader("최근 예측 확률(최종)과 라벨(있으면)")
chart_df = df_latest[["asof_date", "proba_up"]].copy()
chart_df = chart_df.rename(columns={"asof_date": "date"})
if "label_up" in df_latest.columns:
    chart_df["label_up"] = df_latest["label_up"]
st.line_chart(data=chart_df.set_index("date"))

# ==================== 운영 성능 모니터링 ====================
st.subheader("운영 성능 (롤링 AUC)")
if "label_up" in df_latest.columns:
    eval_df = df_latest.dropna(subset=["label_up"]).copy()

    def rolling_auc(s_proba, s_y, win=60):
        out = []
        for i in range(len(s_y)):
            L = max(0, i - win + 1)
            y = s_y.iloc[L:i+1]
            p = s_proba.iloc[L:i+1]
            if y.nunique() < 2:
                out.append(np.nan); continue
            out.append(roc_auc_score(y, p))
        return pd.Series(out, index=s_y.index)

    eval_df["AUC60"] = rolling_auc(eval_df["proba_up"], eval_df["label_up"], 60)
    eval_df["AUC20"] = rolling_auc(eval_df["proba_up"], eval_df["label_up"], 20)
    st.line_chart(eval_df.set_index("asof_date")[["AUC20", "AUC60"]])
else:
    st.info("라벨이 있는 데이터가 아직 부족합니다.")

# ==================== Threshold & 비용 탐색 ====================
st.subheader("임계값 & 비용 민감도 탐색")
if "label_up" in df_latest.columns:
    thr = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)
    FP_cost = st.number_input("FP 비용", 0.0, 10.0, 1.0)
    FN_cost = st.number_input("FN 비용", 0.0, 10.0, 1.0)

    y = df_latest["label_up"].dropna()
    p = df_latest.loc[y.index, "proba_up"]
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary")
    exp_cost = fp * FP_cost + fn * FN_cost

    st.write(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    st.write(f"Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, 기대비용={exp_cost:.2f}")
else:
    st.info("라벨이 있는 구간에서만 작동합니다.")

# ==================== 입력 분포 드리프트 감시 ====================
st.subheader("입력 분포 드리프트 감시")
cand_cols = ["sentiment_z_30", "logret_USD_KRW_vol_5d", "vol_ratio_5_20"]
sel = [c for c in cand_cols if c in df.columns]
if sel:
    hist = df_latest.tail(126)  # 약 6개월
    recent = df_latest.tail(21) # 약 1개월
    rows = []
    for c in sel:
        a = hist[c].dropna(); b = recent[c].dropna()
        if len(a) > 30 and len(b) > 10:
            ks = ks_2samp(a, b).pvalue
            rows.append((c, ks))
    if rows:
        drift_df = pd.DataFrame(rows, columns=["feature", "ks_pvalue"]).sort_values("ks_pvalue")
        st.dataframe(drift_df, use_container_width=True)
        if (drift_df["ks_pvalue"] < 0.05).any():
            st.warning("분포 변화 감지됨 (p<0.05). 재학습 권고.")
else:
    st.caption("드리프트 감시할 피처가 없음")

# ==================== 간단 백테스트 ====================
st.subheader("간단 백테스트 (현금↔지수, 비용 10bp)")
if "label_up" in df_latest.columns:
    cut = st.slider("백테스트 cut-off", 0.3, 0.7, 0.5, 0.01)
    fee = 0.001
    base = df_latest.dropna(subset=["label_up"]).copy()
    signal = (base["proba_up"] >= cut).astype(int)
    # 지수 수익률 컬럼이 없으면 간이 대체: 다음날 라벨 기준 ±0.5%
    if "ret_kospi" not in base.columns:
        base["ret_kospi"] = np.where(base["label_up"] == 1, 0.005, -0.005)
    pos = signal.shift(1).fillna(0)  # 다음날 진입
    trade = pos.diff().abs().fillna(0)
    pnl = pos * base["ret_kospi"] - trade * fee
    eq = (1 + pnl).cumprod()
    st.line_chart(eq.reset_index(drop=True))

# ==================== 메타 XGB 피처 중요도 ====================
st.subheader("메타 모델(XGBoost) 피처 중요도 (gain 기준)")
final_dir = None
try:
    final_dir = find_latest_final_dir(MODELS_ROOT)
    meta_model, feat_names, bundle = load_meta_and_names(final_dir)
    booster = meta_model.get_booster()
    score_map = booster.get_score(importance_type="gain")

    importances = []
    for i in range(meta_model.n_features_in_):
        key = f"f{i}"
        gain = score_map.get(key, 0.0)
        name = feat_names[i] if i < len(feat_names) else key
        importances.append((name, gain))
    imp_df = pd.DataFrame(importances, columns=["feature", "gain"]).sort_values("gain", ascending=False)
    st.dataframe(imp_df.head(30), use_container_width=True)
except Exception as e:
    st.warning(f"피처 중요도를 불러오지 못했습니다: {e}")

# ==================== 설명(이미지) 표시 ====================
st.subheader("설명 결과(있으면 자동 표시)")
if final_dir is not None:
    img_cols = st.columns(3)
    cand_imgs = [
        ("메타 SHAP summary", final_dir / "shap_meta_summary.png"),
        ("Transformer IG/SHAP", final_dir / "ig_low_summary.png"),
        ("CNN IG/SHAP", final_dir / "ig_high_summary.png"),
    ]
    for i, (title, p) in enumerate(cand_imgs):
        with img_cols[i % 3]:
            if p.exists():
                st.markdown(f"**{title}**")
                st.image(str(p))
            else:
                st.caption(f"{title}: 이미지 없음")
else:
    st.caption("설명 이미지를 표시할 모델 디렉터리가 아직 없습니다.")

# ==================== 히스토리 테이블 ====================
st.subheader("실행 히스토리 (최신 플래그 기준)")
st.dataframe(df_latest.sort_values("asof_date"), use_container_width=True)

st.caption("※ 이 앱은 `streamlit run models/pipeline/streamlit_app.py` 로 실행해야 정상 동작합니다.")
