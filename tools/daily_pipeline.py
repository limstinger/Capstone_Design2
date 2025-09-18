# tools/daily_pipeline.py
# -*- coding: utf-8 -*-
"""
매일 23:00에 일괄 실행하는 파이프라인 스케줄러
순서: econ -> news -> preprocess -> predict -> update_labels
- 표준 출력/에러는 logs/*-YYYYMMDD.log 에 적재
- 중간 실패 시 파이프라인 즉시 중단 (returncode != 0)
- predict_next_day.py가 없으면 predict_test.py로 폴백
- 필요시 --now/--once 옵션으로 즉시 실행 가능
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import schedule

# === 경로 설정 ===
ROOT = Path(__file__).resolve().parents[1]            # Capstone_Design/
PIPE = ROOT / "models" / "pipeline"
DATA = ROOT / "data"
PROC = DATA / "processed"
PRED = DATA / "predictions"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# 파이프라인 스크립트 경로들
ECON = PIPE / "economic_indicators.py"
NEWS = PIPE / "news_indicators.py"
PREP = PIPE / "preprocess.py"               # 위치 다르면 여기 수정
PREDICT_MAIN = PIPE / "predict_next_day.py" # 없으면 아래에서 폴백
PREDICT_FALLBACK = PIPE / "predict_test.py"
UPDATE_LABELS = PIPE / "update_labels_in_predictions.py"

# predict 인자 (Streamlit에서 쓰던 기본값과 일치하게)
MODELS_ROOT = ROOT / "models"                # --models-root
DATA_CSV    = PROC / "training_with_refined_features.csv"  # --data-csv
PRED_OUT    = PRED / "next_day_predictions.csv"            # --pred-out

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_append(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)

def run_cmd(cmd: list[str], cwd: Path | None, log_name: str) -> int:
    """하나의 커맨드를 실행하고 logs/<name>-YYYYMMDD.log 에 기록"""
    date_tag = datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"{log_name}-{date_tag}.log"

    header = f"\n\n=== [{ts()}] START: {' '.join(cmd)} (cwd={cwd or os.getcwd()}) ===\n"
    log_append(log_path, header)

    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)

    if proc.stdout:
        log_append(log_path, "[STDOUT]\n" + proc.stdout)
    if proc.stderr:
        log_append(log_path, "\n[STDERR]\n" + proc.stderr)

    footer = f"=== [{ts()}] END (returncode={proc.returncode}) ===\n"
    log_append(log_path, footer)

    print(f"[{ts()}] ran: {' '.join(cmd)} -> rc={proc.returncode}  log={log_path}")
    return proc.returncode

def step_or_die(cmd, cwd, name):
    rc = run_cmd(cmd, cwd, name)
    if rc != 0:
        # 요약 로그 남기고 즉시 중단
        log_append(LOG_DIR / "daily_summary.log", f"[{ts()}] FAIL at {name} rc={rc}\n")
        sys.exit(rc)

def job_daily(recalibrate: bool = False):
    """
    매일 실행할 잡: econ -> news -> preprocess -> predict -> update_labels
    recalibrate=True 이면 예측 직후 재보정 옵션을 전달(지원하는 스크립트가 있을 경우)
    """
    print(f"[{ts()}] === DAILY PIPELINE START ===")

    # 1) 경제지표
    if not ECON.exists():
        log_append(LOG_DIR / "daily_summary.log", f"[{ts()}] SKIP: {ECON} not found\n")
        sys.exit(1)
    step_or_die([PYTHON, str(ECON)], PIPE, "econ")

    # 2) 뉴스
    if not NEWS.exists():
        log_append(LOG_DIR / "daily_summary.log", f"[{ts()}] SKIP: {NEWS} not found\n")
        sys.exit(1)
    step_or_die([PYTHON, str(NEWS)], PIPE, "news")

    # 3) 전처리
    if not PREP.exists():
        log_append(LOG_DIR / "daily_summary.log", f"[{ts()}] SKIP: {PREP} not found\n")
        sys.exit(1)
    step_or_die([PYTHON, str(PREP)], PIPE, "preprocess")

    # 4) 예측 (predict_next_day or predict_test)
    predict_script = PREDICT_MAIN if PREDICT_MAIN.exists() else PREDICT_FALLBACK
    if not predict_script.exists():
        log_append(LOG_DIR / "daily_summary.log", f"[{ts()}] SKIP: no predict script\n")
        sys.exit(1)

    cmd_predict = [
        PYTHON, str(predict_script),
        "--models-root", str(MODELS_ROOT),
        "--data-csv",    str(DATA_CSV),
        "--pred-out",    str(PRED_OUT),
    ]
    if recalibrate:
        cmd_predict += ["--recalibrate"]

    step_or_die(cmd_predict, PIPE, "predict")

    # 5) 라벨 갱신
    if not UPDATE_LABELS.exists():
        log_append(LOG_DIR / "daily_summary.log", f"[{ts()}] SKIP: {UPDATE_LABELS} not found\n")
        sys.exit(1)
    # 필요하면 --overwrite / --backup 등 옵션 추가 가능
    step_or_die([PYTHON, str(UPDATE_LABELS), "--backup"], PIPE, "update_labels")

    # 요약
    log_append(LOG_DIR / "daily_summary.log", f"[{ts()}] DONE econ->news->prep->predict->update_labels\n")
    print(f"[{ts()}] === DAILY PIPELINE DONE ===")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="23:00", help="매일 실행 시각 (24h HH:MM) 기본 23:00")
    ap.add_argument("--once", action="store_true", help="지금 한 번만 실행하고 종료")
    ap.add_argument("--now",  action="store_true", help="지금 즉시 한 번 실행하고, 이후 스케줄 계속 대기")
    ap.add_argument("--recalibrate", action="store_true", help="예측 직후 --recalibrate 전달")
    args = ap.parse_args()

    if args.once:
        job_daily(recalibrate=args.recalibrate)
        return

    # 스케줄 등록
    schedule.clear()
    schedule.every().day.at(args.time).do(job_daily, recalibrate=args.recalibrate)
    print(f"[{ts()}] scheduler started. daily at {args.time}  (Ctrl+C to stop)")

    if args.now:
        print(f"[{ts()}] running once now (--now)")
        job_daily(recalibrate=args.recalibrate)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] scheduler stopped by user")

if __name__ == "__main__":
    main()
