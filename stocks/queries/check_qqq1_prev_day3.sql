WITH Dips AS (
    -- First, find all hourly bars in the last year where the low was >1% below the previous day's close
    SELECT
        h.datetime,
        h.low,
        d.close AS previous_day_close
    FROM
        hourly_prices h
    JOIN
        daily_prices d ON h.ticker = d.ticker AND d.date = date(h.datetime, '-1 day')
    WHERE
        h.ticker = 'QQQ'
        AND date(h.datetime) >= date('now', '-1 year')
        AND h.low < (d.close * 0.99) -- Condition for being down >1%
),
FirstDipPerDay AS (
    -- From that list of dips, find the earliest one for each day
    SELECT
        MIN(datetime) AS first_dip_datetime
    FROM
        Dips
    GROUP BY
        date(datetime)
)
-- Finally, select the details for only those first dips
SELECT
    d.datetime AS event_datetime,
    d.previous_day_close,
    d.low AS price_at_first_dip,
    ROUND(((d.low - d.previous_day_close) / d.previous_day_close) * 100, 2) AS percent_decrease
FROM
    Dips d
JOIN
    FirstDipPerDay fd ON d.datetime = fd.first_dip_datetime
ORDER BY
    d.datetime;
