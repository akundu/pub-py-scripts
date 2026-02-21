WITH Spikes AS (
    -- First, find all hourly bars in the last year where the high was >1% above the previous day's close
    SELECT
        h.datetime,
        h.high,
        d.close AS previous_day_close
    FROM
        hourly_prices h
    JOIN
        daily_prices d ON h.ticker = d.ticker AND d.date = date(h.datetime, '-1 day')
    WHERE
        h.ticker = 'QQQ'
        AND date(h.datetime) >= date('now', '-1 year')
        AND h.high > (d.close * 1.01)
),
FirstSpikePerDay AS (
    -- From that list of spikes, find the earliest one for each day
    SELECT
        MIN(datetime) AS first_spike_datetime
    FROM
        Spikes
    GROUP BY
        date(datetime)
)
-- Finally, select the details for only those first spikes
SELECT
    s.datetime AS event_datetime,
    s.previous_day_close,
    s.high AS price_at_first_spike,
    ROUND(((s.high - s.previous_day_close) / s.previous_day_close) * 100, 2) AS percent_increase
FROM
    Spikes s
JOIN
    FirstSpikePerDay fs ON s.datetime = fs.first_spike_datetime
ORDER BY
    s.datetime;
