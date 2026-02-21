SELECT
    current_date,
    CASE strftime('%w', current_date)
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END AS current_day_of_week,
    previous_date,
    CASE strftime('%w', previous_date)
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday'
    END AS previous_day_of_week,
    days_between
FROM (
    SELECT
        date AS current_date,
        LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS previous_date,
        julianday(date) - julianday(LAG(date) OVER (PARTITION BY ticker ORDER BY date)) AS days_between
    FROM
        daily_prices
    WHERE
        ticker = 'QQQ'
)
WHERE
    days_between > 1;
