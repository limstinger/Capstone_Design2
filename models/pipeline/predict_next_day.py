#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_next_day.py

- 전처리 완료 CSV( data/processed/training_with_refined_features.csv )의
  최신 거래일까지를 사용해 다음 "거래일(주말 건너뜀)" KOSPI 등락(label_up)을 예측하여 저장.
- model_train.py 또는 test.py 가 저장한 최종 아티팩트(models/final_.../) 자동 탐색/사용.
- 같은 날 여러 번 실행 시: 모두 누적(append)하되, run_no, run_id를 달고,
  해당 asof_date의 최신 실행에 is_latest=True 표시(기존 동일 날짜 행은 False로 갱신).

Usage:
    python predict_next_day.py
    python predict_next_day.py --data-csv path/to/training_with_refined_features.csv \
                               --models-root models \
                               --pred-out data/predictions/next_day_predictions.csv \
                               --date 2025-03-28 \
                               --recalibrate \
                               --sharpen 0.85
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from datetime import datetime

# ───────────────────────── Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── ProbabilityCalibrator (언피클 호환용)
class ProbabilityCalibrator:
    """
    학습 시 피클된 보정기를 predict에서 재사용하기 위한 동일 시그니처 클래스.
    (model_train.py / test.py 모두 호환)
    """
    def __init__(self, dual_average: bool = False):
        self.method: Optional[str] = None          # "platt" | "isotonic" | "both" | None
        self.platt: Optional[LogisticRegression] = None
        self.iso: Optional[IsotonicRegression] = None
        self.trained: bool = False
        self.dual_average = dual_average

    def fit(self, p: np.ndarray, y: np.ndarray):
        y = y.astype(int)
        if len(np.unique(y)) < 2:
            self.method = None
            self.trained = False
            return self

        pl = LogisticRegression(solver="lbfgs", max_iter=1000)
        pl.fit(p.reshape(-1, 1), y)
        p_pl = pl.predict_proba(p.reshape(-1, 1))[:, 1]
        auc_pl = roc_auc_score(y, p_pl)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        p_iso = iso.transform(p)
        auc_iso = roc_auc_score(y, p_iso)

        if self.dual_average:
            self.method = "both"
            self.platt = pl
            self.iso = iso
            self.trained = True
            return self

        if auc_iso >= auc_pl:
            self.method = "isotonic"
            self.iso = iso
        else:
            self.method = "platt"
            self.platt = pl

        self.trained = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.trained or self.method is None:
            return p
        if self.method == "both" and (self.platt is not None and self.iso is not None):
            pp = self.platt.predict_proba(p.reshape(-1, 1))[:, 1]
            pi = self.iso.transform(p)
            return 0.5 * pp + 0.5 * pi
        if self.method == "platt" and self.platt is not None:
            return self.platt.predict_proba(p.reshape(-1, 1))[:, 1]
        if self.method == "isotonic" and self.iso is not None:
            return self.iso.transform(p)
        return p

# 피클에서 'model_train.ProbabilityCalibrator' 또는 'test.ProbabilityCalibrator'를
# 참조하는 경우를 위해 별칭 등록 (둘 다 매핑)
sys.modules.setdefault("model_train", sys.modules[__name__])
sys.modules.setdefault("test", sys.modules[__name__])

# ───────────────────────── Small helpers (훈련 시 정의와 구조 일치)
def make_meta_features(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    low = np.asarray(low).ravel()
    high = np.asarray(high).ravel()
    diff = high - low
    denom = (np.abs(low) + np.abs(high) + 1e-8)
    return np.column_stack([
        low, high, diff, np.abs(diff),
        (low + high) / 2.0, np.minimum(low, high), np.maximum(low, high),
        diff / denom, low + high
    ])

def _rolling_pred_feats(p: np.ndarray, wins=(3, 5, 10)) -> np.ndarray:
    # 예측 누적이 없으므로 안전하게 lag 1 기준으로 구성 (배치=1에도 동작)
    s = pd.Series(p)
    s_lag = s.shift(1)
    feats = []
    for w in wins:
        feats.append(s_lag.rolling(w, min_periods=1).mean())
        feats.append(s_lag.rolling(w, min_periods=1).std())
    return pd.concat(feats, axis=1).ffill().fillna(0.5).values

def _build_meta_matrix_from_probs(lpv_c: np.ndarray,
                                  hpv_c: np.ndarray,
                                  ctx_arr: Optional[np.ndarray],
                                  p_blend: Optional[np.ndarray] = None) -> np.ndarray:
    base = make_meta_features(lpv_c, hpv_c)
    roll_l = _rolling_pred_feats(lpv_c, wins=(3, 5, 10))
    roll_h = _rolling_pred_feats(hpv_c, wins=(3, 5, 10))
    cols = [base, roll_l, roll_h]
    if ctx_arr is not None and len(ctx_arr) >= len(base):
        cols.append(ctx_arr[:len(base)])
    if p_blend is not None:
        cols.append(p_blend.reshape(-1, 1))  # p_blend_regime
    return np.column_stack(cols)

def sharpen_probability(p: float, T: float = 0.85) -> float:
    """온도 샤프닝: T<1이면 더 과감한(극단) 확률로 변환"""
    if T is None or T <= 0:
        return float(p)
    eps = 1e-6
    p = float(np.clip(p, eps, 1 - eps))
    logit = np.log(p / (1 - p))
    return float(1 / (1 + np.exp(-logit / T)))

# ───────────────────────── Regime gate (test.py 호환)
def _regime_gate_from_row(row: Dict[str, float]) -> float:
    """입력 row(dict)에서 고주파 가중 w_high ∈ (0,1) 산출."""
    vol  = float(row.get("vol_20d", 0.0) or 0.0)
    surp = float(row.get("sentiment_surprise_5d", 0.0) or 0.0)
    rate = float(row.get("rate_announce_decay", 0.0) or 0.0)
    s = 0.0
    s += 3.0 * np.tanh((vol - 0.01) / 0.02)
    s += 1.5 * np.tanh(surp)
    s += 1.2 * (rate > 0.7)
    w_high = 1.0 / (1.0 + np.exp(-s))
    return float(np.clip(w_high, 0.0, 1.0))

def _make_regime_blend_single(p_low_c: float,
                              p_high_c: float,
                              ctx_row: Optional[np.ndarray],
                              ctx_cols: List[str]) -> float:
    """현재 시점(1행) 기준 레짐 블렌딩 확률 산출."""
    if ctx_row is None or not ctx_cols:
        return 0.5 * float(p_low_c) + 0.5 * float(p_high_c)
    row_dict = {c: float(v) for c, v in zip(ctx_cols, ctx_row.ravel())}
    w_high = _regime_gate_from_row(row_dict)
    return (1.0 - w_high) * float(p_low_c) + w_high * float(p_high_c)

# ───────────────────────── Tiny torch modules (추론 용; 구조만 일치)
class WaveAttTransformerClassifier(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, n_scales: int, dtw_gamma: float = 0.1):
        super().__init__()

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 500):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                pos = torch.arange(0, max_len).unsqueeze(1)
                div = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer('pe', pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, :x.size(1)].to(x.device)

        class PreNormEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward, dropout):
                super().__init__()
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.linear2 = nn.Linear(dim_feedforward, d_model)

            def forward(self, x):
                x_norm = self.norm1(x)
                attn_out, _ = self.attn(x_norm, x_norm, x_norm)
                x = x + self.dropout1(attn_out)
                x_norm = self.norm2(x)
                ff = self.linear2(self.dropout2(F.gelu(self.linear1(x_norm))))
                x = x + ff
                return x

        class WaveletAttention(nn.Module):
            def __init__(self, in_channels, d_model, nhead, n_scales=3, wavelet='db4'):
                super().__init__()
                import pywt
                self.n_scales = n_scales
                self.wavelet = wavelet
                self.to_q = nn.ModuleList([nn.Linear(in_channels, d_model) for _ in range(n_scales)])
                self.to_k = nn.ModuleList([nn.Linear(in_channels, d_model) for _ in range(n_scales)])
                self.to_v = nn.ModuleList([nn.Linear(in_channels, d_model) for _ in range(n_scales)])
                self.gate = nn.Sequential(nn.Linear(d_model, n_scales), nn.Softmax(dim=-1))
                self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

            def forward(self, x):
                import pywt
                B, T, C = x.shape
                maxlev_possible = pywt.swt_max_level(T)
                level = min(self.n_scales, maxlev_possible)
                padded, pad_len = False, 0
                if level > 0:
                    need = 2 ** level
                    pad_len = (-T) % need
                    if pad_len > 0:
                        last = x[:, -1:, :].repeat(1, pad_len, 1)
                        x = torch.cat([x, last], dim=1)
                        padded = True
                        T = x.size(1)
                if level < 1:
                    z = self.to_v[0](x)
                    return z[:, :-pad_len, :] if padded and pad_len > 0 else z

                arr = x.detach().cpu().numpy()
                coeffs = pywt.swt(arr, self.wavelet, level=level, axis=1)  # [(cA,cD)]*level
                details = [torch.from_numpy(cD).to(x.device, dtype=x.dtype) for (_, cD) in coeffs]

                Vs = []
                for s in range(level):
                    W = details[s]
                    Q = self.to_q[s](W); K = self.to_k[s](W); V = self.to_v[s](W)
                    out, _ = self.attn(Q, K, V)
                    Vs.append(out)

                V_stack = torch.stack(Vs, -1)                 # (B,T_pad,d_model,level)
                global_feat = V_stack.mean(1).mean(-1)        # (B,d_model)
                gate_full = self.gate(global_feat)            # (B,n_scales)
                gate = gate_full[:, :level]
                gate = gate / gate.sum(1, keepdim=True).clamp_min(1e-8)
                Z = (V_stack * gate.unsqueeze(1).unsqueeze(2)).sum(-1)
                if padded and pad_len > 0:
                    Z = Z[:, :-pad_len, :]
                return Z

        class DTWAttention(nn.Module):
            def __init__(self, gamma=0.1, bandwidth=None):
                super().__init__()
                from soft_dtw_cuda import SoftDTW
                self.gamma = gamma
                self.bandwidth = bandwidth
                self.soft_dtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=gamma, bandwidth=bandwidth)

            def _ensure_device(self, device):
                want_cuda = (device.type == 'cuda')
                curr_cuda = getattr(self.soft_dtw, 'use_cuda', None)
                if curr_cuda is None or curr_cuda != want_cuda:
                    from soft_dtw_cuda import SoftDTW
                    self.soft_dtw = SoftDTW(use_cuda=want_cuda, gamma=self.gamma, bandwidth=self.bandwidth)

            def forward(self, x, y):
                if y.device != x.device:
                    y = y.to(x.device)
                self._ensure_device(x.device)
                D = self.soft_dtw(x, y)        # (B,)
                w = torch.exp(-D).view(-1, 1, 1)
                return x * w

        self.input_proj = nn.Linear(input_size, d_model)
        self.wav_att = WaveletAttention(input_size, d_model, nhead, n_scales)
        self.dtw_att = DTWAttention(gamma=dtw_gamma)
        self.pos_enc = PositionalEncoding(d_model)
        self.enc_layers = nn.ModuleList([
            PreNormEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        z1 = self.wav_att(x)
        z2 = self.dtw_att(x_proj, z1)
        z3 = self.pos_enc(z2)
        h = z3
        for layer in self.enc_layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.classifier(h[:, -1, :])


class CNN1DClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x.mean(dim=-1)
        x = self.dropout(x)
        return self.head(x)

# ───────────────────────── IO helpers
def find_latest_final_dir(models_root: Path) -> Path:
    # model_train.py: "final_wavelet_transformer_*" (예)
    # test.py:       "final_wavelet_transformer_*" 또는 "final_*"
    patterns = ["final_wavelet_transformer_*", "final_*"]
    cands: List[Path] = []
    for pat in patterns:
        cands.extend(list(models_root.glob(pat)))
    if not cands:
        for pat in patterns:
            cands.extend(list(models_root.rglob(pat)))
    cands = [p for p in cands if p.is_dir()]
    if not cands:
        raise FileNotFoundError("models 폴더에서 final_* 디렉터리를 찾을 수 없습니다.")
    return max(cands, key=lambda p: p.stat().st_mtime)

def load_artifacts(final_dir: Path):
    pkl = final_dir / "preproc_and_calibrators.pkl"
    xgbp = final_dir / "meta_xgb.pkl"
    lrp  = final_dir / "meta_lr.pkl"
    wsp  = final_dir / "meta_blend_weights.pkl"
    tpt = final_dir / "transformer_low_final.pt"
    cpt = final_dir / "cnn_high_final.pt"

    miss = [str(x.name) for x in [pkl, xgbp, tpt, cpt] if not x.exists()]
    if miss:
        raise FileNotFoundError(f"예상 모델 파일 일부가 없습니다: {miss}. 학습 파이프라인을 먼저 실행하세요.")

    bundle = joblib.load(pkl)
    meta_model: XGBClassifier = joblib.load(xgbp)
    meta_lr = joblib.load(lrp) if lrp.exists() else None
    blend_w = joblib.load(wsp) if wsp.exists() else {"w_x": 1.0, "w_l": 0.0}

    low_scaler = bundle["low_scaler"]
    high_scaler = bundle["high_scaler"]
    low_feats = bundle["low_feats"]
    high_feats = bundle["high_feats"]
    cal_low = bundle["cal_low"]
    cal_high = bundle["cal_high"]
    params = bundle["params"]
    meta_ctx_cols = params.get("meta_ctx_cols", [])
    seq_len = int(params["seq_len"])
    d_model, nhead = params["d_model_nhead"]
    dim_feedforward = int(params["dim_feedforward"])
    num_layers = int(params.get("num_layers", 2))
    transf_dropout = float(params.get("transf_dropout", 0.2))
    n_scales = int(params.get("n_scales", 3))

    trans = WaveAttTransformerClassifier(
        input_size=len(low_feats), d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        dropout=transf_dropout, n_scales=n_scales
    ).to(DEVICE)
    trans.load_state_dict(torch.load(tpt, map_location=DEVICE))
    trans.eval()

    cnn = CNN1DClassifier(
        in_channels=len(high_feats),
        hidden=int(params.get("cnn_hidden", 128)),
        dropout=float(params.get("cnn_dropout", 0.2))
    ).to(DEVICE)
    cnn.load_state_dict(torch.load(cpt, map_location=DEVICE))
    cnn.eval()

    return {
        "trans": trans, "cnn": cnn, "meta_model": meta_model,
        "meta_lr": meta_lr, "blend_w": blend_w,
        "low_scaler": low_scaler, "high_scaler": high_scaler,
        "low_feats": low_feats, "high_feats": high_feats,
        "cal_low": cal_low, "cal_high": cal_high,
        "meta_ctx_cols": meta_ctx_cols, "seq_len": seq_len, "params": params
    }

# ───────────────────────── Build one-step-ahead slice
def build_latest_window(df: pd.DataFrame, feats: List[str], scaler, seq_len: int) -> np.ndarray:
    """
    df: 날짜 오름차순 정렬 DataFrame
    반환: X_seq (1, T, C)  # 마지막 T=seq_len 구간
    """
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing[:10]} ...")
    Xfull = df[feats].fillna(0).values.astype(float)
    if scaler is not None:
        Xfull = scaler.transform(Xfull)
    if len(Xfull) < seq_len:
        raise ValueError(f"데이터 길이({len(Xfull)})가 seq_len({seq_len})보다 짧습니다.")
    X_last = Xfull[-seq_len:, :]
    return X_last[None, ...]  # (1,T,C)

# ───────────────────────── Final post-hoc calibration (optional)
def fit_platt_on_labeled(pred_csv: Path, min_n: int = 50) -> Optional[LogisticRegression]:
    if not pred_csv.exists():
        return None
    df = pd.read_csv(pred_csv)
    if not {"proba_up", "label_up"}.issubset(df.columns):
        return None
    lab = df.dropna(subset=["label_up"])
    if len(lab) < min_n:
        return None
    X = lab["proba_up"].to_numpy().reshape(-1, 1)
    y = lab["label_up"].astype(int).to_numpy()
    try:
        lr = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, y)
        return lr
    except Exception:
        return None

# ───────────────────────── Date helpers (주말 건너뛰기)
def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    nd = d + pd.Timedelta(days=1)
    while nd.weekday() >= 5:  # 5=토, 6=일
        nd += pd.Timedelta(days=1)
    return nd

# ───────────────────────── Main predict
def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[2]
    ap.add_argument("--data-csv", type=str, default=str(base / "data" / "processed" / "training_with_refined_features.csv"))
    ap.add_argument("--models-root", type=str, default=str(base / "models"))
    ap.add_argument("--pred-out", type=str, default=str(base / "data" / "predictions" / "next_day_predictions.csv"))
    ap.add_argument("--date", type=str, default="", help="예측 기준 마지막 데이터 날짜(YYYY-MM-DD). 기본: CSV 마지막 행 날짜")
    ap.add_argument("--recalibrate", action="store_true", help="과거 라벨 축적분으로 최종 Platt 보정 재적용")
    ap.add_argument("--sharpen", type=float, default=0.0, help="T<1이면 확률 샤프닝 적용(예: 0.85). 0 또는 미지정 시 비활성")
    args = ap.parse_args()

    data_csv = Path(args.data_csv)
    models_root = Path(args.models_root)
    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    # 모델/전처리 아티팩트 로드
    final_dir = find_latest_final_dir(models_root)
    art = load_artifacts(final_dir)

    # 데이터 로드
    df = pd.read_csv(data_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("입력 CSV가 비어 있습니다.")

    # 기준 날짜
    if args.date:
        base_date = pd.to_datetime(args.date).floor("D")
        df_use = df[df["date"] <= base_date].copy()
        if df_use.empty:
            raise ValueError(f"{args.date} 이전(포함) 데이터가 없습니다.")
    else:
        base_date = pd.to_datetime(df.iloc[-1]["date"]).floor("D")
        df_use = df.copy()

    # 예측 대상 날짜: 다음 '영업일'(주말 건너뜀)
    pred_for_date = next_business_day(base_date)

    # 컨텍스트 (메타용 — 현재 시점 값만)
    ctx_cols = art["meta_ctx_cols"]
    if ctx_cols:
        ctx_row = df_use.iloc[-1:][ctx_cols].fillna(0).values  # (1, Ctx)
    else:
        ctx_row = None

    # 시퀀스 구성
    seq_len = art["seq_len"]
    X_low = build_latest_window(df_use, art["low_feats"], art["low_scaler"], seq_len)   # (1,T,C)
    X_high = build_latest_window(df_use, art["high_feats"], art["high_scaler"], seq_len)  # (1,T,C)

    # 신경망 추론 → 개별 보정
    trans: nn.Module = art["trans"]
    cnn: nn.Module = art["cnn"]
    with torch.no_grad():
        lp = torch.sigmoid(trans(torch.from_numpy(X_low).float().to(DEVICE))).view(-1).cpu().numpy()
        hp = torch.sigmoid(cnn(torch.from_numpy(X_high).float().to(DEVICE))).view(-1).cpu().numpy()
    p_low_raw = float(lp[-1])
    p_high_raw = float(hp[-1])

    cal_low = art["cal_low"]
    cal_high = art["cal_high"]
    # dual_average 여부는 훈련 시 저장된 객체 속성이 가짐
    p_low = float(cal_low.transform(np.array([p_low_raw]))[0]) if getattr(cal_low, "trained", False) else p_low_raw
    p_high = float(cal_high.transform(np.array([p_high_raw]))[0]) if getattr(cal_high, "trained", False) else p_high_raw

    # --- p_blend_regime 계산(현재 시점 컨텍스트 1행 사용)
    p_blend = np.array(
        [_make_regime_blend_single(p_low, p_high, ctx_row, ctx_cols)],
        dtype=float
    )

    # 메타 입력(배치=1 안전 형태) — p_blend 포함
    X_meta = _build_meta_matrix_from_probs(
        np.array([p_low], dtype=float),
        np.array([p_high], dtype=float),
        ctx_row.astype(float) if ctx_row is not None else None,
        p_blend=p_blend
    )

    # 메타 예측 (XGB 단독 또는 듀얼(XGB+LR) 가중합)
    meta_model: XGBClassifier = art["meta_model"]
    meta_lr = art.get("meta_lr", None)
    blend_w = art.get("blend_w", {"w_x": 1.0, "w_l": 0.0})
    px = float(meta_model.predict_proba(X_meta)[0, 1])
    if meta_lr is not None:
        pl = float(meta_lr.predict_proba(X_meta)[0, 1])
        proba_up_meta = blend_w.get("w_x", 1.0) * px + blend_w.get("w_l", 0.0) * pl
    else:
        proba_up_meta = px

    # (선택) 최종 Platt 재보정
    proba_up_cal = proba_up_meta
    if args.recalibrate:
        post_cal = fit_platt_on_labeled(pred_out)
        if post_cal is not None:
            proba_up_cal = float(post_cal.predict_proba(np.array([[proba_up_meta]], dtype=float))[0, 1])

    # (선택) 샤프닝(온도 T<1 이면 더 과감)
    used_sharpen = (args.sharpen is not None and args.sharpen > 0 and args.sharpen < 1.0)
    proba_up_final = sharpen_probability(proba_up_cal, T=args.sharpen) if used_sharpen else proba_up_cal

    pred_up = int(proba_up_final > 0.5)

    # 저장용 행
    model_sig = final_dir.name
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "asof_date": base_date.strftime("%Y-%m-%d"),
        "pred_for": pred_for_date.strftime("%Y-%m-%d"),
        "proba_up_meta": proba_up_meta,        # 메타(듀얼 가중합 포함) 직후
        "proba_up_cal": proba_up_cal,          # (재보정 후)
        "proba_up": proba_up_final,            # 최종(샤프닝 반영)
        "pred_up": pred_up,
        "p_low_raw": p_low_raw,
        "p_high_raw": p_high_raw,
        "p_low_cal": p_low,
        "p_high_cal": p_high,
        "p_blend_regime": float(p_blend[0]),
        "model_dir": model_sig,
        "seq_len": seq_len,
        "low_feats_n": len(art["low_feats"]),
        "high_feats_n": len(art["high_feats"]),
        "meta_ctx_cols_n": len(ctx_cols),
        "created_at": now_utc,         # 저장 시각(UTC)
        "run_id": f"{base_date.strftime('%Y%m%d')}-{datetime.utcnow().strftime('%H%M%S')}",
        "is_latest": True,             # 동일 asof_date 내 최신 실행 플래그
        "used_sharpen": int(used_sharpen),
        "sharpen_T": float(args.sharpen if args.sharpen else 0.0),
    }

    # 가능하면 정답 라벨/정오표 기록
    try:
        lab = df.loc[df["date"] == base_date, "label_up"]
        if not lab.empty and pd.notna(lab.iloc[0]):
            row["label_up"] = int(lab.iloc[0])
            row["was_correct"] = int(row["pred_up"] == row["label_up"])
    except Exception:
        pass

    # 파일 누적 저장:
    # - 같은 asof_date의 기존 행 수를 세서 run_no 부여
    # - 기존 동일 asof_date 행들의 is_latest=False 로 갱신
    if pred_out.exists():
        hist = pd.read_csv(pred_out)
        same = hist[hist["asof_date"] == row["asof_date"]]
        run_no = int(same["run_no"].max() + 1) if ("run_no" in same.columns and len(same) > 0) else (len(same) + 1)
        row["run_no"] = run_no

        if "is_latest" in hist.columns:
            hist.loc[hist["asof_date"] == row["asof_date"], "is_latest"] = False
        else:
            hist["is_latest"] = False
            hist.loc[hist["asof_date"] == row["asof_date"], "is_latest"] = False

        new_row_df = pd.DataFrame([row])
        for c in new_row_df.columns:
            if c not in hist.columns:
                hist[c] = np.nan
        for c in hist.columns:
            if c not in new_row_df.columns:
                new_row_df[c] = np.nan
        hist = pd.concat([hist, new_row_df[hist.columns]], ignore_index=True)
    else:
        row["run_no"] = 1
        hist = pd.DataFrame([row])

    hist.to_csv(pred_out, index=False, encoding="utf-8-sig")

    print(f"[OK] {row['asof_date']} → {row['pred_for']} "
          f"p(meta)={proba_up_meta:.4f} p(cal)={proba_up_cal:.4f} p(final)={proba_up_final:.4f} (pred_up={pred_up})")
    print(f"     model={model_sig}, run_no={row['run_no']}, sharpen={args.sharpen or 0}, saved: {pred_out}")

if __name__ == "__main__":
    main()
