SELECT
    date AS current_date,
    LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS previous_date,
    julianday(date) - julianday(LAG(date) OVER (PARTITION BY ticker ORDER BY date)) AS days_between
FROM
    daily_prices
WHERE
    ticker = 'QQQ'
HAVING
    days_between > 1
ORDER BY
    date;
