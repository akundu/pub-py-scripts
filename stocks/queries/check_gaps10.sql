WITH ranked_dates AS (
    SELECT
        date,
        ROW_NUMBER() OVER (ORDER BY date) as rn
    FROM
        daily_prices
    WHERE
        ticker = 'QQQ'
        AND date >= date('now', '-1 year')
)
SELECT
    prev.date AS gap_start_date,
    CASE strftime('%w', prev.date)
        WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday' WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END AS gap_start_day,
    curr.date AS gap_end_date,
    CASE strftime('%w', curr.date)
        WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday' WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END AS gap_end_day,
    julianday(curr.date) - julianday(prev.date) AS days_between
FROM
    ranked_dates curr
JOIN
    ranked_dates prev ON curr.rn = prev.rn + 1
WHERE
    julianday(curr.date) - julianday(prev.date) > 1
ORDER BY
    gap_start_date;
