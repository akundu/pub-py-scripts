WITH SignificantMoves AS (
    -- First, find all hourly bars that were either >1% up or >1% down
    SELECT
        h.datetime,
        d.close AS previous_day_close,
        -- Use a CASE statement to determine the type of move and the relevant price
        CASE
            WHEN h.high > (d.close * 1.01) THEN 'Spike'
            ELSE 'Dip'
        END AS move_type,
        CASE
            WHEN h.high > (d.close * 1.01) THEN h.high
            ELSE h.low
        END AS relevant_price
    FROM
        hourly_prices h
    JOIN
        daily_prices d ON h.ticker = d.ticker AND d.date = date(h.datetime, '-1 day')
    WHERE
        h.ticker = 'QQQ'
        AND date(h.datetime) >= date('now', '-1 year')
        -- The OR condition finds both types of events
        AND (h.high > (d.close * 1.01) OR h.low < (d.close * 0.99))
),
FirstMovePerDay AS (
    -- From that list, find the earliest move for each day
    SELECT
        MIN(datetime) AS first_move_datetime
    FROM
        SignificantMoves
    GROUP BY
        date(datetime)
)
-- Finally, select the full details for only those first moves
SELECT
    sm.datetime AS event_datetime,
    sm.move_type,
    sm.previous_day_close,
    sm.relevant_price,
    ROUND(((sm.relevant_price - sm.previous_day_close) / sm.previous_day_close) * 100, 2) AS percent_change
FROM
    SignificantMoves sm
JOIN
    FirstMovePerDay fm ON sm.datetime = fm.first_move_datetime
ORDER BY
    sm.datetime;
