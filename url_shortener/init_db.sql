-- URL Shortener Database Schema for QuestDB

-- Create url_mappings table
CREATE TABLE IF NOT EXISTS url_mappings (
    short_code SYMBOL INDEX CAPACITY 1024,
    original_url STRING,
    created_at TIMESTAMP,
    access_count LONG,
    last_accessed TIMESTAMP
) TIMESTAMP(created_at) PARTITION BY DAY WAL
DEDUP UPSERT KEYS(created_at, short_code);

-- Note: QuestDB automatically indexes the designated TIMESTAMP column (created_at)
-- The SYMBOL type for short_code provides fast lookups
-- DEDUP UPSERT KEYS ensures no duplicate short codes (must include timestamp column)



