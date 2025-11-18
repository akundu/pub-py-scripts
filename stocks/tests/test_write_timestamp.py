#!/usr/bin/env python3

import asyncio
from common.stock_db import get_stock_db

async def test_write_timestamp():
    db = get_stock_db('sqlite', 'data/stock_data.db')
    
    # Get realtime data for AAPL
    result = await db.get_realtime_data('AAPL', data_type='quote')
    
    print('Columns:', result.columns.tolist())
    print('\nData:')
    print(result.head())
    
    if not result.empty:
        print(f'\nFirst row write_timestamp: {result.iloc[0].get("write_timestamp", "NOT FOUND")}')
        print(f'First row timestamp: {result.index[0]}')

if __name__ == "__main__":
    asyncio.run(test_write_timestamp()) 