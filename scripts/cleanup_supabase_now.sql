-- ============================================================
-- AWAI Supabase Emergency Cleanup SQL
-- Run this in the Supabase SQL Editor to immediately reclaim
-- disk space and get below the 0.5 GB free-plan limit.
-- ============================================================

-- Step 1: Check current sizes BEFORE cleanup
SELECT
    relname AS table_name,
    n_live_tup AS live_rows,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_stat_user_tables
WHERE relname IN ('live_traffic_records', 'predictions', 'model_registry')
ORDER BY pg_total_relation_size(relid) DESC;

-- ============================================================
-- Step 2: Delete old live_traffic_records (keep last 12 hours)
-- The LSTM only needs 24 timesteps × 15 min = 6 hours of data.
-- 12 hours = 2× safety margin. Everything older is safe to delete.
-- ============================================================
DELETE FROM live_traffic_records
WHERE timestamp_wib < (NOW() AT TIME ZONE 'UTC' - INTERVAL '12 hours')::text;

-- If the above returns 0 rows deleted, try this alternative
-- (in case timestamps are stored in WIB/UTC+7):
-- DELETE FROM live_traffic_records
-- WHERE timestamp_wib < ((NOW() AT TIME ZONE 'Asia/Jakarta') - INTERVAL '12 hours')::text;

-- ============================================================
-- Step 3: Delete old predictions (keep last 3 days)
-- ============================================================
DELETE FROM predictions
WHERE requested_at_wib < (NOW() AT TIME ZONE 'UTC' - INTERVAL '3 days')::text;

-- ============================================================
-- Step 4: Verify sizes AFTER cleanup
-- ============================================================
SELECT
    relname AS table_name,
    n_live_tup AS live_rows,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_stat_user_tables
WHERE relname IN ('live_traffic_records', 'predictions', 'model_registry')
ORDER BY pg_total_relation_size(relid) DESC;

-- NOTE: Supabase runs autovacuum automatically, so storage reclaim
-- will happen within a few minutes after this query completes.
-- You can also check the actual billing size in the Supabase Dashboard
-- under Settings > Billing (may take up to 1 hour to refresh).
