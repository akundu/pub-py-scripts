WITH last_year_dates AS (
    SELECT date
    FROM daily_prices
    WHERE ticker = 'QQQ' AND date >= date('now', '-1 year')
)
SELECT
    previous_date AS gap_start_date,
    CASE strftime('%w', previous_date)
        WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday' WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END AS gap_start_day,
    current_date AS gap_end_date,
    CASE strftime('%w', current_date)
        WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday' WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END AS gap_end_day,
    julianday(current_date) - julianday(previous_date) AS days_between
FROM (
    SELECT
        d1.date AS current_date,
        (SELECT MAX(d2.date) FROM last_year_dates d2 WHERE d2.date < d1.date) AS previous_date
    FROM
        last_year_dates d1
)
WHERE
    days_between > 1
ORDER BY
    gap_start_date;
