#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
update_labels_in_predictions.py

- processed CSV에서 (asof_date == date) 행의 label_up을 찾아
  predictions CSV의 동일 asof_date 행들에 label_up / was_correct를 채움.
- 같은 asof_date에 여러 run이 있으면 전부 갱신 (is_latest는 유지).
- 추가:
  * used_sharpen / sharpen_T 결측치 보정(없으면 생성, 문자열/불리언 섞여도 숫자화)
  * 컬럼명 공백 제거, asof_date 문자열 정규화(YYYY-MM-DD)
  * pred_up 없고 pred_proba_up만 있으면 threshold로 0/1 생성
  * 저장 전 --backup 옵션 제공

Usage:
    python update_labels_in_predictions.py
    python update_labels_in_predictions.py --proc-csv data/processed/training_with_refined_features.csv \
                                           --pred-csv data/predictions/next_day_predictions.csv \
                                           --overwrite --threshold 0.5 --backup
"""
import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np


def _normalize_asof_date_series(s: pd.Series) -> pd.Series:
    """asof_date가 datetime/문자열 혼재여도 YYYY-MM-DD 문자열로 정규화."""
    # 이미 문자열인 값 보존을 위해 원본 복사
    orig = s.astype(str)
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    out = dt.dt.strftime("%Y-%m-%d")
    # 변환 실패(NaT)는 원본 문자열 유지
    out = out.where(dt.notna(), orig.str.strip())
    # 혹시 'YYYY-MM-DD HH:MM:SS' 같은 경우 자르기
    out = out.str.slice(0, 10)
    return out


def _ensure_and_fill_sharpen_cols(pred: pd.DataFrame) -> pd.DataFrame:
    """
    used_sharpen/sharpen_T 컬럼을 보정/생성하고 결측을 채운다.
    - used_sharpen: 없으면 0, 문자열/불리언 혼재 시 1/0으로 정규화 후 결측 0
    - sharpen_T   : 없으면 0.0, 숫자 변환 실패/결측 0.0
    """
    # 없으면 생성
    if "used_sharpen" not in pred.columns:
        pred["used_sharpen"] = 0
    if "sharpen_T" not in pred.columns:
        pred["sharpen_T"] = 0.0

    # used_sharpen: 문자열/불리언/숫자 혼재 → 1/0 정규화
    if pred["used_sharpen"].dtype == object:
        s = pred["used_sharpen"].astype(str).str.strip()
        mapping = {
            "True": 1, "true": 1, "1": 1, "yes": 1, "Y": 1, "y": 1, "t": 1, "T": 1,
            "False": 0, "false": 0, "0": 0, "no": 0, "N": 0, "n": 0, "f": 0, "F": 0,
            "": np.nan, "None": np.nan, "none": np.nan, "NaN": np.nan, "nan": np.nan
        }
        pred["used_sharpen"] = s.map(mapping)

    # 불리언 → int
    if pd.api.types.is_bool_dtype(pred["used_sharpen"]):
        pred["used_sharpen"] = pred["used_sharpen"].astype(int)
    else:
        # 그 외 → 숫자화 후 결측 0
        pred["used_sharpen"] = pd.to_numeric(pred["used_sharpen"], errors="coerce").fillna(0).astype(int)

    # sharpen_T: 숫자화 후 결측 0.0
    pred["sharpen_T"] = pd.to_numeric(pred["sharpen_T"], errors="coerce").fillna(0.0)

    return pred


def _ensure_pred_up(pred: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    pred_up이 없고 pred_proba_up만 있을 때 0/1로 생성.
    - threshold 이상이면 1, 미만 0.
    """
    if "pred_up" not in pred.columns and "pred_proba_up" in pred.columns:
        proba = pd.to_numeric(pred["pred_proba_up"], errors="coerce")
        pred["pred_up"] = (proba >= float(threshold)).astype("Int64").fillna(0).astype(int)
    return pred


def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[2]
    ap.add_argument("--proc-csv", type=str, default=str(base / "data" / "processed" / "training_with_refined_features.csv"))
    ap.add_argument("--pred-csv", type=str, default=str(base / "data" / "predictions" / "next_day_predictions.csv"))
    ap.add_argument("--overwrite", action="store_true", help="예측 CSV에 이미 있는 label_up도 덮어씀")
    ap.add_argument("--threshold", type=float, default=0.5, help="pred_proba_up → pred_up 변환 임계값(기본 0.5)")
    ap.add_argument("--backup", action="store_true", help="저장 전 predictions CSV 백업(.bak) 생성")
    args = ap.parse_args()

    proc_path = Path(args.proc_csv)
    pred_path = Path(args.pred_csv)

    if not proc_path.exists():
        raise FileNotFoundError(f"processed csv not found: {proc_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions csv not found: {pred_path}")

    # 읽기
    proc = pd.read_csv(proc_path, parse_dates=["date"])
    pred = pd.read_csv(pred_path)

    # 컬럼명 공백 제거 (엣지 케이스 방지: 'used_sharpen ', ' sharpen_T' 등)
    proc.columns = proc.columns.str.strip()
    pred.columns = pred.columns.str.strip()

    # 기본 컬럼 체크
    if "label_up" not in proc.columns:
        raise RuntimeError("processed csv에 'label_up' 컬럼이 필요합니다.")
    if "asof_date" not in pred.columns:
        raise RuntimeError("predictions csv에 'asof_date' 컬럼이 필요합니다.")

    # processed: date → 문자열 asof_date 키 생성
    proc["asof_date"] = proc["date"].dt.strftime("%Y-%m-%d")
    key = proc[["asof_date", "label_up"]].dropna(subset=["asof_date"]).copy()

    # predictions: asof_date 정규화(문자/날짜 혼재 대응)
    pred["asof_date"] = _normalize_asof_date_series(pred["asof_date"])

    # 라벨 병합
    pred = pred.merge(key, on="asof_date", how="left", suffixes=("", "_from_proc"))

    # 덮어쓰기/빈곳만 채우기
    if args.overwrite:
        pred["label_up"] = pred["label_up_from_proc"]
    else:
        pred["label_up"] = pred["label_up"].where(~pred["label_up"].isna(), pred["label_up_from_proc"])

    # 보조 컬럼 제거
    if "label_up_from_proc" in pred.columns:
        pred.drop(columns=["label_up_from_proc"], inplace=True)

    # pred_up 보장(없으면 pred_proba_up으로 생성)
    pred = _ensure_pred_up(pred, threshold=args.threshold)

    # was_correct 갱신 (pred_up & label_up 모두 존재 시)
    if {"pred_up", "label_up"}.issubset(pred.columns):
        mask = pred["label_up"].notna() & pred["pred_up"].notna()
        # 안전한 정수 비교(소수/문자열 들어와도 캐스팅)
        pred.loc[mask, "was_correct"] = (
            pd.to_numeric(pred.loc[mask, "pred_up"], errors="coerce").round().astype("Int64")
            ==
            pd.to_numeric(pred.loc[mask, "label_up"], errors="coerce").round().astype("Int64")
        ).astype("Int64").fillna(0).astype(int)

    # 🔧 used_sharpen / sharpen_T 결측 보정(없으면 생성 + 채움)
    pred = _ensure_and_fill_sharpen_cols(pred)

    # 저장 전 백업
    if args.backup:
        bak = pred_path.with_suffix(pred_path.suffix + ".bak")
        shutil.copy2(pred_path, bak)
        print(f"[OK] backup created → {bak}")

    # 저장
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 로그
    labeled = int(pred["label_up"].notna().sum()) if "label_up" in pred.columns else 0
    sharpen_info = f"used_sharpen(dtype={pred['used_sharpen'].dtype}), sharpen_T(dtype={pred['sharpen_T'].dtype})"
    correct_ct = int(pred["was_correct"].notna().sum()) if "was_correct" in pred.columns else 0
    print(f"[OK] updated labels in {pred_path} (labeled rows: {labeled})")
    print(f"[OK] sharpen columns filled → {sharpen_info}")
    print(f"[OK] was_correct updated rows: {correct_ct}")


if __name__ == "__main__":
    main()
