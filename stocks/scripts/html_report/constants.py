"""
Constants and configuration for HTML report generation.
"""

# Column name mappings for compact headers
COMPACT_HEADER_MAP = {
    'ticker': 'ticker',
    'price': 'price',
    'P/E': 'P/E',
    'MKT_CAP': 'MKT_CAP',
    'MKT_B': 'MKT_B',
    'STRK': 'STRK',
    'price_above_current': 'current_strike_diff',
    'option_premium': 'opt_premium',
    'bid_ask': 'bid:ask',
    'option_premium_percentage': 'opt_premium%',
    'premium_above_diff_percentage': 'DIFF%',
    'implied_volatility': 'IV',
    'delta': 'DEL',
    'theta': 'TH',
    'volume': 'VOL',
    'num_contracts': 'CNT',
    'potential_premium': 'POT_PREM',
    'daily_premium': 'DAILY_PREM',
    'expiration_date': 'EXP (UTC)',
    'days_to_expiry': 'DAYS',
    'last_quote_timestamp': 'LQUOTE_TS',
    'write_timestamp': 'WRITE_TS (EST)',
    'option_ticker': 'OPT_TKR',
    'long_strike_price': 'L_STRK',
    'long_option_premium': 'L_PREM',
    'long_bid_ask': 'l_bid:ask',
    'long_expiration_date': 'L_EXP',
    'long_days_to_expiry': 'L_DAYS',
    'long_option_ticker': 'L_OPT_TKR',
    'long_delta': 'L_DEL',
    'long_theta': 'L_TH',
    'long_implied_volatility': 'LIV',
    'long_volume': 'L_VOL',
    'long_contracts_available': 'L_CNT_AVL',
    'premium_diff': 'PREM_DIFF',
    'short_premium_total': 'S_PREM_TOT',
    'short_daily_premium': 'S_DAY_PREM',
    'long_premium_total': 'L_PREM_TOT',
    'NET_PREM': 'NET_PREM',
    'NET_DAY': 'NET_DAY'
}

# Columns to hide by default in tables
HIDDEN_COLUMNS = [
    'price_change_pct',
    'price_change_pc',
    'price_above_curr',
    'price_above_current',
    'premium_diff',
    'prem_diff',
    'days_to_expiry',
    'iv',
    'implied_volatility',
    'liv',
    'long_implied_volatility',
    'l_days_to_expiry',
    'long_days_to_expiry',
    'long_contracts_available',
    'option_ticker',
    'l_option_ticker',
    'buy_cost',
    'net_premium_after_spread',
    'spread_slippage',
    'net_daily_premium_after_spread',
    'spread_impact_pct',
    'liquidity_score',
    'assignment_risk',
]

# Columns to always hide (even when "Show hidden columns" is clicked)
ALWAYS_HIDDEN_COLUMNS = [
    'l_cnt_avl',
    'long_contracts_available',
]

# Column groups for visual grouping in table headers
# Format: group_name -> (short_column_variations, long_column_variations)
COLUMN_GROUPS = {
    'strike_price': (['strike_price'], ['l_strike']),
    'opt_prem': (['opt_prem.', 'opt_prem', 'option_premium'], ['l_opt_prem', 'l_prem', 'long_option_premium']),
    'expiration_date': (['expiration_date'], ['l_expiration_date', 'long_expiration_date']),
    'bid:ask': (['bid:ask', 'bid_ask'], ['l_bid:ask', 'l_bid_ask', 'long_bid_ask']),
    'delta': (['delta'], ['l_delta', 'long_delta']),
    'theta': (['theta'], ['l_theta', 'long_theta']),
    's_prem_tot': (['s_prem_tot', 'short_premium_total'], ['net_premium']),
    's_day_prem': (['s_day_prem', 'short_daily_premium'], ['net_daily_premium', 'net_daily_premi']),
}

# Group display names
GROUP_DISPLAY_NAMES = {
    'strike_price': 'Strike Price',
    'opt_prem': 'Option Premium',
    'expiration_date': 'Expiration Date',
    'bid:ask': 'Bid:Ask',
    'delta': 'Delta',
    'theta': 'Theta',
    's_prem_tot': 'Premium Total',
    's_day_prem': 'Daily Premium',
}

# Column order for grouping
GROUP_ORDER = ['strike_price', 'expiration_date', 'opt_prem', 'bid:ask', 'delta', 'theta', 's_prem_tot', 's_day_prem']

# Columns to place after daily premium columns
COLUMNS_AFTER_DAILY_PREMIUM = ['num_contracts', 'buy_cost', 'volume', 'trade_quality']

# Primary columns to show on mobile cards
PRIMARY_CARD_COLUMNS = [
    'strike_price',
    'option_premium',
    'expiration_date',
    'daily_premium',
    'net_daily_premium',
    'net_daily_premi',
    's_day_prem',
    'short_daily_premium'
]

# Primary column display labels for cards
PRIMARY_COLUMN_LABELS = {
    'ticker': 'Ticker',
    'current_price': 'Price',
    'curr_price': 'Price',
    'change_pct': 'Change',
    'price_with_change': 'Change',
    'strike_price': 'Strike',
    'option_premium': 'Premium',
    'opt_prem': 'Premium',
    'expiration_date': 'Expiry',
    'daily_premium': 'Daily',
    'net_daily_premium': 'Net Daily',
    'net_daily_premi': 'Net Daily',
    's_day_prem': 'Daily',
    'short_daily_premium': 'Daily',
}

# Desired column order for display
DESIRED_COLUMN_ORDER = [
    'ticker', 'option_type', 'pe_ratio', 'market_cap_b', 'current_price', 'price_with_change',
    'strike_price', 'option_premium', 'expiration_date', 'bid_ask', 'delta', 'theta',
    'short_premium_total', 'short_daily_premium',
    'long_strike_price', 'long_option_premium', 'long_expiration_date', 'long_bid_ask', 'long_delta', 'long_theta',
    'long_implied_volatility', 'long_days_to_expiry',
    'long_premium_total', 'long_contracts_available',
    'net_premium', 'net_daily_premium',
    'price_above_current', 'premium_above_diff_percentage',
    'implied_volatility', 'days_to_expiry',
    'potential_premium', 'daily_premium',
    'volume', 'num_contracts', 'option_ticker', 'long_option_ticker',
    'premium_diff',
    'spread_slippage', 'net_premium_after_spread', 'net_daily_premium_after_spread',
    'spread_impact_pct', 'liquidity_score', 'assignment_risk', 'trade_quality',
    'latest_option_writets'
]

# Column name variations for flexible matching
COLUMN_VARIATIONS = {
    'current_price': ['curr_price', 'cur_price', 'current_price'],
    'price_with_change': ['price_with_change', 'price_with_chan', 'price_with_ch'],
    'option_premium': ['opt_prem.', 'opt_prem', 'option_premium'],
    'bid_ask': ['bid:ask', 'bid_ask'],
    'short_premium_total': ['s_prem_tot', 's_prem_total', 'short_premium_total'],
    'short_daily_premium': ['s_day_prem', 's_daily_prem', 'short_daily_premium'],
    'long_strike_price': ['l_strike', 'l_strike_price', 'long_strike_price'],
    'long_option_premium': ['l_opt_prem', 'l_prem', 'l_option_premium', 'long_option_premium'],
    'long_bid_ask': ['l_bid:ask', 'l_bid_ask', 'long_bid_ask'],
    'long_implied_volatility': ['liv', 'l_iv', 'long_implied_volatility'],
    'long_delta': ['l_delta', 'long_delta'],
    'long_theta': ['l_theta', 'long_theta'],
    'long_expiration_date': ['l_expiration_date', 'long_expiration_date'],
    'long_days_to_expiry': ['l_days_to_expiry', 'long_days_to_expiry'],
    'long_premium_total': ['l_prem_tot', 'l_premium_total', 'long_premium_total'],
    'long_contracts_available': ['l_cnt_avl', 'l_contracts_available', 'long_contracts_available'],
    'net_daily_premium': ['net_daily_premi', 'net_daily_premium'],
}

