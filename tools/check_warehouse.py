# tools/check_warehouse.py
# -*- coding: utf-8 -*-
"""
DuckDB 웨어하우스(warehouse.duckdb) 상태 점검 스크립트
- 뉴스 파이프라인: news_view / news_sentiment_view / news_sentiment_daily_view / raw_news
- 거시지표: kospi / fx / xau / wti_eod / base_rate / vkospi / features_daily
- ops_metrics: 최근 실행 기록
"""
from pathlib import Path
import argparse
import duckdb
import pandas as pd

WAREHOUSE = Path(__file__).resolve().parents[1] / "data" / "warehouse" / "warehouse.duckdb"

SQL_SETS = {
    "news": [
        # 최근 날짜 확인
        ( "최근 날짜(뉴스 뷰들)",
          """
          SELECT 'news' ds, max(date) AS max_date FROM bronze.news_view
          UNION ALL
          SELECT 'news_sent', max(date) FROM bronze.news_sentiment_view
          UNION ALL
          SELECT 'news_daily', max(date) FROM bronze.news_sentiment_daily_view
          """
        ),
        # 원시 뉴스 키 중복 점검
        ( "원시 뉴스 키 중복(url)",
          "SELECT COUNT(*) AS total, COUNT(DISTINCT url) AS uniq FROM bronze.raw_news"
        ),
        # 일별 감성 집계 행수(최근 7일)
        ( "news_sentiment_daily 최근 7일",
          """
          SELECT date::DATE AS d, n, pos, neg, neu, sent, strong_ratio
          FROM bronze.news_sentiment_daily_view
          WHERE date >= (SELECT max(date) FROM bronze.news_sentiment_daily_view) - INTERVAL 7 DAY
          ORDER BY d
          """
        ),
    ],
    "macro": [
        ( "거시지표 최근 날짜들",
          """
          SELECT 'kospi' ds, max(date) AS max_date FROM bronze.kospi_view
          UNION ALL SELECT 'fx',    max(date) FROM bronze.fx_view
          UNION ALL SELECT 'xau',   max(date) FROM bronze.xau_view
          UNION ALL SELECT 'wti',   max(date) FROM bronze.wti_eod_view
          UNION ALL SELECT 'vkospi',max(date) FROM bronze.vkospi_view
          UNION ALL SELECT 'base',  max(date) FROM bronze.base_rate_view
          UNION ALL SELECT 'feat',  max(date) FROM bronze.features_daily_view
          """
        ),
        ( "features_daily 최근 7행",
          """
          SELECT date::DATE AS d, kospi_close, usd_krw, xau_usd, wti_usd_tminus1_eod, policy_rate
          FROM bronze.features_daily_view
          ORDER BY date DESC
          LIMIT 7
          """
        ),
    ],
    "ops": [
        ( "최근 ops_metrics 20건",
          """
          SELECT ts, stage, rows, ok, extra
          FROM ops_metrics
          ORDER BY ts DESC
          LIMIT 20
          """
        ),
    ]
}

def show(con, title, sql):
    print(f"\n=== {title} ===")
    try:
        df = con.execute(sql).fetchdf()
        # pandas 출력 옵션 약간 정리
        with pd.option_context('display.max_columns', 20, 'display.width', 120):
            print(df if not df.empty else "(no rows)")
    except Exception as e:
        print(f"(query error) {e}")

def describe_table(con, full_name: str):
    print(f"\n--- DESCRIBE {full_name} ---")
    try:
        df = con.execute(f"DESCRIBE {full_name}").fetchdf()
        print(df if not df.empty else "(no columns)")
    except Exception as e:
        print(f"(describe error) {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default=str(WAREHOUSE), help="DuckDB 파일 경로")
    ap.add_argument("--section", choices=["all","news","macro","ops"], default="all",
                    help="점검 섹션 선택")
    ap.add_argument("--describe", action="store_true",
                    help="주요 테이블 스키마 DESCRIBE 출력")
    args = ap.parse_args()

    path = Path(args.db)
    if not path.exists():
        raise SystemExit(f"DuckDB 파일이 없습니다: {path}")

    con = duckdb.connect(str(path))

    # 섹션 실행
    sections = ["news","macro","ops"] if args.section == "all" else [args.section]
    for sec in sections:
        print(f"\n############ {sec.upper()} ############")
        for title, sql in SQL_SETS[sec]:
            show(con, title, sql)

    # 스키마 설명(선택)
    if args.describe:
        for t in [
            "bronze.news_sentiment_daily",
            "bronze.raw_news",
            "bronze.fx_view",
            "bronze.features_daily_view",
        ]:
            describe_table(con, t)

    con.close()
    print("\n✅ check done")

if __name__ == "__main__":
    main()
