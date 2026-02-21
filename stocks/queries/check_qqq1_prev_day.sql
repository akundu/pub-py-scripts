SELECT
    date(h.datetime) AS event_date,
    d.close AS previous_day_close,
    MAX(h.high) AS peak_hourly_high,
    ROUND(((MAX(h.high) - d.close) / d.close) * 100, 2) AS peak_percent_increase
FROM
    hourly_prices h
JOIN
    daily_prices d ON h.ticker = d.ticker AND d.date = date(h.datetime, '-1 day')
WHERE
    h.ticker = 'QQQ'
    AND date(h.datetime) >= date('now', '-1 year')
    AND h.high > (d.close * 1.01)
GROUP BY
    date(h.datetime)
ORDER BY
    event_date;
