# Test an hourly strategy for a single day
python test_llm_strategy.py --db-file your_db.db \
    --strategy "Buy AAPL when price drops below 150" \
    --investment-amount 10000 \
    --start-date "2024-03-20 09:30" \
    --end-date "2024-03-20 16:00" \
    --timeframe hourly \
    --plot

# Test an hourly strategy for multiple days
python test_llm_strategy.py --db-file your_db.db \
    --strategy "Buy AAPL when price drops below 150" \
    --investment-amount 10000 \
    --start-date "2024-03-18 09:30" \
    --end-date "2024-03-22 16:00" \
    --timeframe hourly \
    --plot
