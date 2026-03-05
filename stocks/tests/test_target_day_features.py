"""Tests for target_day_of_week and days_to_friday features."""

import pytest
from datetime import date

from scripts.close_predictor.multi_day_features import (
    MarketContext,
    _add_trading_days,
    _resolve_target_date,
    set_target_day_features,
    compute_feature_similarity,
)


class TestAddTradingDays:
    """Tests for _add_trading_days helper."""

    def test_zero_days(self):
        d = date(2025, 1, 6)  # Monday
        assert _add_trading_days(d, 0) == d

    def test_one_day_monday(self):
        # Monday + 1 trading day = Tuesday
        assert _add_trading_days(date(2025, 1, 6), 1) == date(2025, 1, 7)

    def test_one_day_friday(self):
        # Friday + 1 trading day = Monday (skip weekend)
        assert _add_trading_days(date(2025, 1, 10), 1) == date(2025, 1, 13)

    def test_five_days_monday(self):
        # Monday + 5 trading days = Monday next week
        assert _add_trading_days(date(2025, 1, 6), 5) == date(2025, 1, 13)

    def test_three_days_wednesday(self):
        # Wednesday + 3 trading days = Monday (skip weekend)
        assert _add_trading_days(date(2025, 1, 8), 3) == date(2025, 1, 13)

    def test_with_trading_calendar(self):
        cal = ['2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
               '2025-01-13', '2025-01-14']
        # Monday + 3 = Thursday
        assert _add_trading_days(date(2025, 1, 6), 3, cal) == date(2025, 1, 9)
        # Friday + 1 = Monday (using calendar)
        assert _add_trading_days(date(2025, 1, 10), 1, cal) == date(2025, 1, 13)

    def test_with_calendar_holiday(self):
        # Calendar skips Jan 9 (holiday)
        cal = ['2025-01-06', '2025-01-07', '2025-01-08', '2025-01-10',
               '2025-01-13', '2025-01-14']
        # Monday + 3 = Friday (since Thursday is a holiday)
        assert _add_trading_days(date(2025, 1, 6), 3, cal) == date(2025, 1, 10)

    def test_calendar_start_date_not_in_calendar(self):
        # Start date is a Saturday, calendar starts Monday
        cal = ['2025-01-06', '2025-01-07', '2025-01-08']
        result = _add_trading_days(date(2025, 1, 4), 2, cal)
        assert result == date(2025, 1, 8)  # Sat->Mon(idx 0) + 2 = Wed(idx 2)


class TestResolveTargetDate:
    """Tests for _resolve_target_date."""

    def test_basic(self):
        assert _resolve_target_date(date(2025, 1, 6), 1) == date(2025, 1, 7)

    def test_with_calendar(self):
        cal = ['2025-01-06', '2025-01-07', '2025-01-08']
        assert _resolve_target_date(date(2025, 1, 6), 2, cal) == date(2025, 1, 8)


class TestSetTargetDayFeatures:
    """Tests for set_target_day_features."""

    def test_monday_plus_1_is_tuesday(self):
        ctx = MarketContext()
        set_target_day_features(ctx, date(2025, 1, 6), 1)  # Mon + 1 = Tue
        assert ctx.target_day_of_week == 1  # Tuesday
        assert ctx.days_to_friday == 3  # Tue->Fri = 3 trading days

    def test_monday_plus_4_is_friday(self):
        ctx = MarketContext()
        set_target_day_features(ctx, date(2025, 1, 6), 4)  # Mon + 4 = Fri
        assert ctx.target_day_of_week == 4  # Friday
        assert ctx.days_to_friday == 0

    def test_wednesday_plus_3_is_monday(self):
        ctx = MarketContext()
        set_target_day_features(ctx, date(2025, 1, 8), 3)  # Wed + 3 = Mon
        assert ctx.target_day_of_week == 0  # Monday
        assert ctx.days_to_friday == 4

    def test_friday_plus_1_is_monday(self):
        ctx = MarketContext()
        set_target_day_features(ctx, date(2025, 1, 10), 1)  # Fri + 1 = Mon
        assert ctx.target_day_of_week == 0  # Monday
        assert ctx.days_to_friday == 4

    def test_with_trading_calendar(self):
        cal = ['2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
               '2025-01-13', '2025-01-14', '2025-01-15']
        ctx = MarketContext()
        set_target_day_features(ctx, date(2025, 1, 6), 5, cal)  # Mon + 5 = Mon
        assert ctx.target_day_of_week == 0  # Monday
        assert ctx.days_to_friday == 4

    def test_default_values_before_set(self):
        ctx = MarketContext()
        assert ctx.target_day_of_week == -1
        assert ctx.days_to_friday == -1

    def test_to_dict_includes_new_features(self):
        ctx = MarketContext()
        set_target_day_features(ctx, date(2025, 1, 6), 1)
        d = ctx.to_dict()
        assert 'target_day_of_week' in d
        assert 'days_to_friday' in d
        assert d['target_day_of_week'] == 1.0  # Tuesday
        assert d['days_to_friday'] == 3.0

    def test_to_dict_default_values(self):
        ctx = MarketContext()
        d = ctx.to_dict()
        assert d['target_day_of_week'] == -1.0
        assert d['days_to_friday'] == -1.0


class TestFeatureSimilarityWithTargetDay:
    """Tests that compute_feature_similarity uses target_day_of_week when available."""

    def test_same_target_day_higher_similarity(self):
        ctx1 = MarketContext(target_day_of_week=4, day_of_week=0)  # target=Fri, source=Mon
        ctx2_same = MarketContext(target_day_of_week=4, day_of_week=2)  # target=Fri, source=Wed
        ctx2_diff = MarketContext(target_day_of_week=1, day_of_week=0)  # target=Tue, source=Mon

        sim_same_target = compute_feature_similarity(ctx1, ctx2_same)
        sim_diff_target = compute_feature_similarity(ctx1, ctx2_diff)

        # Same target day should have higher similarity even with different source days
        assert sim_same_target > sim_diff_target

    def test_fallback_when_target_unknown(self):
        # When target_day_of_week == -1, should use old logic
        ctx1 = MarketContext(day_of_week=0)  # Mon, target=-1
        ctx2 = MarketContext(day_of_week=0)  # Mon, target=-1

        sim = compute_feature_similarity(ctx1, ctx2)
        # Should still compute a valid similarity
        assert 0.0 <= sim <= 1.0
