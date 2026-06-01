#!/usr/bin/env python3
"""
cleanup_db_now.py
=================
Emergency one-shot script to purge stale rows from Supabase and bring the
database size back under the 0.5 GB free-plan limit.

Usage:
    python scripts/cleanup_db_now.py

The script reads DATABASE_URL from the environment (or from the project .env
file if present).  Set it explicitly if running outside the project root:

    DATABASE_URL=postgres://... python scripts/cleanup_db_now.py

What it does:
1.  Deletes live_traffic_records older than 12 hours
    (LSTM needs at most 24 x 15-min steps = 6 hours; 12 h is a safe margin).
2.  Deletes predictions older than 3 days.
3.  Runs VACUUM ANALYZE on both tables to actually reclaim disk pages.
4.  Prints before/after row counts and estimated sizes.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Try to load DATABASE_URL from project .env if not already in environment
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
_dotenv = _root / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print(
        "ERROR: DATABASE_URL is not set.\n"
        "Export it or add it to your .env file, then re-run:\n"
        "  DATABASE_URL=postgres://... python scripts/cleanup_db_now.py",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed.  Run: pip install psycopg2-binary", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config — match the defaults added to RuntimeConfig
# ---------------------------------------------------------------------------
RETAIN_LIVE_HOURS = int(os.environ.get("DB_LIVE_RECORDS_RETENTION_HOURS", "12"))
RETAIN_PRED_DAYS = int(os.environ.get("DB_PREDICTIONS_RETENTION_DAYS", "3"))

NOW_UTC = datetime.now(timezone.utc)
LIVE_CUTOFF = (NOW_UTC - timedelta(hours=RETAIN_LIVE_HOURS)).isoformat()
PRED_CUTOFF = (NOW_UTC - timedelta(days=RETAIN_PRED_DAYS)).isoformat()


def table_stats(cursor, table: str) -> tuple[int, float]:
    """Return (row_count, size_mb) for a table."""
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    rows = cursor.fetchone()[0]
    cursor.execute("SELECT pg_total_relation_size(%s)", (table,))
    size_bytes = cursor.fetchone()[0]
    return rows, round(size_bytes / (1024 * 1024), 3)


def main() -> None:
    print("=" * 60)
    print("  AWAI Supabase Emergency Cleanup")
    print(f"  Retain live records: last {RETAIN_LIVE_HOURS} hours")
    print(f"  Retain predictions:  last {RETAIN_PRED_DAYS} days")
    print(f"  Cutoff live:  {LIVE_CUTOFF}")
    print(f"  Cutoff preds: {PRED_CUTOFF}")
    print("=" * 60)

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # ---- Before stats ----------------------------------------
            print("\n[BEFORE]")
            for tbl in ("live_traffic_records", "predictions", "model_registry"):
                try:
                    rows, mb = table_stats(cur, tbl)
                    print(f"  {tbl}: {rows} rows / {mb} MB")
                except Exception as exc:
                    print(f"  {tbl}: could not read stats — {exc}")

            # ---- Prune live_traffic_records ---------------------------
            print(f"\nDeleting live_traffic_records older than {LIVE_CUTOFF} ...")
            cur.execute(
                "DELETE FROM live_traffic_records WHERE timestamp_wib < %s",
                (LIVE_CUTOFF,),
            )
            live_deleted = cur.rowcount
            print(f"  → {live_deleted} rows deleted")

            # ---- Prune predictions -----------------------------------
            print(f"\nDeleting predictions older than {PRED_CUTOFF} ...")
            cur.execute(
                "DELETE FROM predictions WHERE requested_at_wib < %s",
                (PRED_CUTOFF,),
            )
            pred_deleted = cur.rowcount
            print(f"  → {pred_deleted} rows deleted")

            conn.commit()
            print("\n✓ Commit successful")

        # VACUUM must run outside a transaction block
        conn.autocommit = True
        with conn.cursor() as cur:
            print("\nRunning VACUUM ANALYZE to reclaim disk space ...")
            cur.execute("VACUUM ANALYZE live_traffic_records")
            cur.execute("VACUUM ANALYZE predictions")
            print("✓ VACUUM ANALYZE done")

        conn.autocommit = False
        # ---- After stats -----------------------------------------
        with conn.cursor() as cur:
            print("\n[AFTER]")
            total_mb = 0.0
            for tbl in ("live_traffic_records", "predictions", "model_registry"):
                try:
                    rows, mb = table_stats(cur, tbl)
                    total_mb += mb
                    print(f"  {tbl}: {rows} rows / {mb} MB")
                except Exception as exc:
                    print(f"  {tbl}: could not read stats — {exc}")
            print(f"\n  Total estimated: {round(total_mb, 3)} MB / 512 MB limit")

        print("\n" + "=" * 60)
        print("  Cleanup complete!")
        print(
            "  The automated db_cleanup_job will now run every "
            f"{os.environ.get('DB_CLEANUP_INTERVAL_HOURS', '6')} hours to keep the DB clean."
        )
        print("=" * 60)

    except Exception as exc:
        conn.rollback()
        print(f"\nERROR during cleanup: {exc}", file=sys.stderr)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
