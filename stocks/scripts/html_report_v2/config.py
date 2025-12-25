"""
Configuration constants for HTML report generation.
"""

# Column name mappings for compact headers
COMPACT_HEADER_MAP = {
    'ticker': 'ticker',
    'pe_ratio': 'P/E',
    'market_cap_b': 'MKT_B',
    'current_price': 'price',
    'curr_price': 'price',
    'change_pct': 'change%',
    'price_with_change': 'change%',
    'strike_price': 'STRK',
    'option_premium': 'opt_prem',
    'opt_prem': 'opt_prem',
    'expiration_date': 'EXP',
    'bid:ask': 'bid:ask',
    'delta': 'DEL',
    'theta': 'TH',
    'volume': 'VOL',
    'num_contracts': 'CNT',
    'daily_premium': 'DAILY',
    'net_daily_premium': 'NET_DAILY',
    's_day_prem': 'DAILY',
    'short_daily_premium': 'DAILY',
    'premium_ratio_pct': 'PREM%',
}

# Columns to hide by default (can be shown with toggle)
HIDDEN_COLUMNS = [
    'price_change_pct',
    'price_change_pc',
    'price_above_curr',
    'price_above_current',
    'premium_diff',
    'prem_diff',
    'days_to_expiry',
    # IV columns are now visible by default
    # 'iv',
    # 'implied_volatility',
    # 'liv',
    # 'long_implied_volatility',
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
    'option_type',
    'l_prem_tot',
]

# Columns to always hide (never shown, used for sorting only)
ALWAYS_HIDDEN_COLUMNS = [
    'l_cnt_avl',
    'long_contracts_available',
]

# Column groups for visual grouping (Short/Long pairs)
COLUMN_GROUPS = {
    'strike_price': {
        'short': ['strike_price'],
        'long': ['l_strike', 'long_strike_price'],
        'display_name': 'Strike Price'
    },
    'option_premium': {
        'short': ['opt_prem.', 'opt_prem', 'option_premium'],
        'long': ['l_opt_prem', 'l_prem', 'long_option_premium'],
        'extras': ['premium_ratio_pct'],
        'display_name': 'Option Premium'
    },
    'expiration_date': {
        'short': ['expiration_date'],
        'long': ['l_expiration_date', 'long_expiration_date'],
        'display_name': 'Expiration Date'
    },
    'bid_ask': {
        'short': ['bid:ask', 'bid_ask'],
        'long': ['l_bid:ask', 'l_bid_ask', 'long_bid_ask'],
        'display_name': 'Bid:Ask'
    },
    'delta': {
        'short': ['delta'],
        'long': ['l_delta', 'long_delta'],
        'display_name': 'Delta'
    },
    'theta': {
        'short': ['theta'],
        'long': ['l_theta', 'long_theta'],
        'display_name': 'Theta'
    },
    'premium_total': {
        'short': ['s_prem_tot', 'short_premium_total'],
        'long': ['net_premium'],
        'display_name': 'Premium Total'
    },
    'daily_premium': {
        'short': ['s_day_prem', 'short_daily_premium'],
        'long': ['net_daily_premium', 'net_daily_premi'],
        'display_name': 'Daily Premium'
    },
}

# Order for column groups
GROUP_ORDER = [
    'strike_price',
    'expiration_date',
    'option_premium',
    'bid_ask',
    'delta',
    'theta',
    'premium_total',
    'daily_premium',
]

# Columns to place after daily premium
COLUMNS_AFTER_DAILY_PREMIUM = [
    'num_contracts',
    'buy_cost',
    'volume',
    'trade_quality',
]

# Primary columns for mobile cards (always visible)
PRIMARY_CARD_COLUMNS = [
    'strike_price',
    'option_premium',
    'expiration_date',
    'daily_premium',
    'net_daily_premium',
    'net_daily_premi',
    's_day_prem',
    'short_daily_premium',
]

# Primary column labels for cards
PRIMARY_COLUMN_LABELS = {
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

# Desired column order (before grouping)
DESIRED_COLUMN_ORDER = [
    'ticker',
    'pe_ratio',
    'market_cap_b',
    'current_price',
    'curr_price',
    'change_pct',
    'price_with_change',
]

# Column name variations to handle
COLUMN_VARIATIONS = {
    'current_price': ['current_price', 'curr_price', 'cur_price'],
    'price_with_change': ['price_with_change', 'price_with_chan', 'change_pct'],
    'price_change_pct': ['price_change_pct', 'price_change_pc'],
    'option_premium': ['option_premium', 'opt_prem.', 'opt_prem'],
    'long_option_premium': ['long_option_premium', 'l_opt_prem', 'l_prem'],
}

# Responsive breakpoint
MOBILE_BREAKPOINT = 768  # pixels

