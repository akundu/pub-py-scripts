#!/usr/bin/env python3
"""
Day Simulation for Continuous Mode

Simulates a full trading day using historical data.
Time compression: 5 real minutes = 10 simulation seconds (30x speed).

Usage:
    python scripts/continuous/simulate_day.py --date 2026-02-20 --speed 30
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta, date as date_type
from typing import List, Optional
import pytz

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.continuous.market_data_v2 import get_current_market_context, create_default_provider
from scripts.continuous.opportunity_scanner import scan_opportunities, filter_actionable_opportunities
from scripts.continuous.alert_manager import ALERTS, AlertLevel
from scripts.continuous.config import CONFIG
from scripts.continuous.position_tracker import PositionTracker
from scripts.continuous.strike_resolver import find_option_chain_file, load_option_chain


class DaySimulator:
    """Simulates a trading day at accelerated speed."""

    def __init__(
        self,
        target_date: date_type,
        ticker: str = 'NDX',
        trend: str = 'sideways',
        speed_multiplier: int = 30,  # 5 min real = 10 sec sim (30x)
    ):
        """
        Initialize day simulator.

        Args:
            target_date: Date to simulate (e.g., 2026-02-20)
            ticker: Ticker symbol
            trend: Market trend
            speed_multiplier: Time compression (30 = 30x speed)
        """
        self.target_date = target_date
        self.ticker = ticker
        self.trend = trend
        self.speed_multiplier = speed_multiplier

        # Trading hours: 6:30 AM - 1:00 PM PST
        self.start_hour = 6
        self.start_minute = 30
        self.end_hour = 13
        self.end_minute = 0

        # Simulation time step (5 minutes)
        self.time_step_minutes = 5

        # Convert to simulation seconds
        self.sim_interval_seconds = (self.time_step_minutes * 60) / self.speed_multiplier

        # Data provider
        self.provider = None
        self.current_regime = None
        self.opportunities_found = []
        self.prev_price = None
        self.tracker = PositionTracker()

        # Historical VIX data (loaded from QuestDB for the target date)
        self._vix_timeline = None  # DataFrame: timestamp, vix, vix1d

        # Option chain for strike resolution
        self._option_chain = None

    def generate_time_points(self) -> List[datetime]:
        """Generate all time points for the simulation day."""
        pst = pytz.timezone('America/Los_Angeles')

        # Start time
        start = pst.localize(datetime.combine(
            self.target_date,
            datetime.min.time().replace(hour=self.start_hour, minute=self.start_minute)
        ))

        # End time
        end = pst.localize(datetime.combine(
            self.target_date,
            datetime.min.time().replace(hour=self.end_hour, minute=self.end_minute)
        ))

        time_points = []
        current = start
        while current <= end:
            time_points.append(current)
            current += timedelta(minutes=self.time_step_minutes)

        return time_points

    def load_historical_vix(self):
        """
        Load VIX data for the target date from QuestDB.

        Tries realtime table first, falls back to hourly_prices.
        Returns True if data was loaded, False if falling back to synthetic.
        """
        if self.provider is None:
            return False

        try:
            vix_df = self.provider.get_vix_for_date(self.target_date)
            if vix_df is not None and hasattr(vix_df, 'empty') and not vix_df.empty:
                self._vix_timeline = vix_df
                print(f"✓ Loaded {len(vix_df)} VIX readings for {self.target_date}")
                print(f"  VIX range: {vix_df['vix'].min():.2f} - {vix_df['vix'].max():.2f}")
                if 'vix1d' in vix_df.columns and vix_df['vix1d'].notna().any():
                    print(f"  VIX1D range: {vix_df['vix1d'].dropna().min():.2f} - {vix_df['vix1d'].dropna().max():.2f}")
                return True
            else:
                print(f"  No VIX data found for {self.target_date}")
                return False
        except Exception as e:
            print(f"  Error loading VIX data: {e}")
            return False

    def _lookup_vix_at_time(self, sim_time: datetime) -> dict:
        """
        Look up the closest VIX reading to the given simulation time.

        Uses the preloaded _vix_timeline from QuestDB.
        """
        import pandas as pd

        if self._vix_timeline is None or self._vix_timeline.empty:
            return None

        # Convert sim_time to UTC for comparison
        sim_utc = sim_time.astimezone(pytz.utc)
        sim_ts = pd.Timestamp(sim_utc)

        # Find closest timestamp
        timeline_ts = pd.to_datetime(self._vix_timeline['timestamp'])

        # Ensure timezone-aware comparison
        if timeline_ts.dt.tz is None:
            timeline_ts = timeline_ts.dt.tz_localize('UTC')
        elif str(timeline_ts.dt.tz) != 'UTC':
            timeline_ts = timeline_ts.dt.tz_convert('UTC')

        if sim_ts.tzinfo is None:
            sim_ts = sim_ts.tz_localize('UTC')

        diffs = abs(timeline_ts - sim_ts)
        closest_idx = diffs.idxmin()
        closest_row = self._vix_timeline.iloc[closest_idx]

        # Only use if within 30 minutes of target
        min_diff = diffs.min()
        if min_diff > pd.Timedelta('30min'):
            return None

        result = {'vix': float(closest_row['vix'])}
        if 'vix1d' in closest_row and pd.notna(closest_row.get('vix1d')):
            result['vix1d'] = float(closest_row['vix1d'])
        else:
            result['vix1d'] = None

        return result

    def simulate_market_data(self, sim_time: datetime):
        """
        Get market data for a specific simulation time.

        Uses real VIX data from QuestDB if available, falls back to synthetic.
        Price is always synthetic (randomized around base).
        """
        import random
        random.seed(int(sim_time.timestamp()))

        # Price simulation (synthetic — no intraday price history in QuestDB for indices)
        hour_progress = (sim_time.hour - self.start_hour) + (sim_time.minute / 60.0)
        total_hours = self.end_hour - self.start_hour

        base_price = 20000.0
        drift = random.uniform(-0.5, 0.5) * (hour_progress / total_hours)
        volatility = random.uniform(-0.2, 0.2)
        current_price = base_price * (1 + (drift + volatility) / 100)

        # VIX: use real data if available, else synthetic
        real_vix = self._lookup_vix_at_time(sim_time)

        if real_vix is not None:
            vix = real_vix['vix']
            vix1d = real_vix['vix1d']
            data_source = 'questdb'
        else:
            # Synthetic fallback
            base_vix = 14.5
            vix_pattern = abs(hour_progress - (total_hours / 2)) / (total_hours / 2)
            vix = base_vix + vix_pattern * 2 + random.uniform(-0.5, 0.5)
            vix1d = vix * 0.8
            data_source = 'synthetic'

        return {
            'price': current_price,
            'vix': vix,
            'vix1d': vix1d,
            'data_source': data_source,
        }

    def print_header(self, sim_time: datetime):
        """Print simulation time header."""
        print("\n" + "=" * 100)
        print(f"📅 SIMULATION TIME: {sim_time.strftime('%Y-%m-%d %I:%M %p PST')} "
              f"(Real speed: {self.speed_multiplier}x)")
        print("=" * 100)

    def print_market_context(self, context, sim_data):
        """Print current market context."""
        data_src = sim_data.get('data_source', 'synthetic')
        src_tag = '[DB]' if data_src == 'questdb' else '[SIM]'

        print(f"\n📊 MARKET CONTEXT: {src_tag}")
        print(f"   Price: ${context.current_price:,.2f} ({context.price_change_pct:+.2f}%)")
        vix_line = f"   VIX: {context.vix_level:.2f}"
        if context.vix1d:
            vix_line += f" | VIX1D: {context.vix1d:.2f}"
        if context.vix_term_spread is not None:
            spread_dir = 'stress' if context.vix_term_spread > 0 else 'calm'
            vix_line += f" | Spread: {context.vix_term_spread:+.2f} ({spread_dir})"
        print(vix_line)
        print(f"   Regime: {context.vix_regime.upper().replace('_', ' ')}")

        # VIX dynamics
        direction = getattr(context, 'vix_direction', 'stable')
        velocity = getattr(context, 'vix_velocity', 0.0)
        change_5m = getattr(context, 'vix_change_5m', None)
        if change_5m is not None:
            arrow = '↑' if direction == 'rising' else ('↓' if direction == 'falling' else '→')
            print(f"   VIX Direction: {arrow} {direction.upper()} | 5m: {change_5m:+.2f} | Velocity: {velocity:+.3f}/5m")

        print(f"   Volume: {context.volume_ratio:.2f}x average")

    def print_opportunities(self, opportunities, actionable):
        """Print discovered opportunities."""
        if not opportunities:
            print(f"\n   No qualifying opportunities at this time.")
            return

        print(f"\n🔍 OPPORTUNITIES DETECTED: {len(opportunities)} total, {len(actionable)} actionable")

        # Show top 3
        for i, opp in enumerate(opportunities[:3], 1):
            in_window = "✓ IN WINDOW" if opp.is_in_entry_window else ""
            quality = "✓ QUALITY" if opp.meets_quality_threshold else ""

            print(f"\n   #{i}: {opp.dte}DTE {opp.band} {opp.spread_type.upper()} ({opp.flow_mode}) @ {opp.entry_time_pst}")
            print(f"       Win: {opp.expected_win_pct:.1f}% | ROI: {opp.expected_roi_pct:.1f}% | Sharpe: {opp.sharpe:.2f}")
            print(f"       Credit: ${opp.estimated_credit:.0f} | Risk: ${opp.estimated_max_risk:.0f} | Score: {opp.trade_score:.1f}")
            if opp.trade_instruction:
                print(f"       >> {opp.trade_instruction}")
            print(f"       {in_window} {quality}")

        # Alert if actionable
        if actionable:
            print(f"\n   🚨 ALERT: {len(actionable)} ACTIONABLE OPPORTUNITIES NOW!")
            for opp in actionable[:2]:
                print(f"      → {opp.dte}DTE {opp.band} {opp.spread_type.upper()} @ {opp.entry_time_pst}")
                print(f"         Credit: ${opp.estimated_credit:.0f} | ROI: {opp.expected_roi_pct:.1f}%")

    def write_dashboard_data(self, context, opportunities, sim_time, step, total_steps):
        """Write current simulation state to dashboard JSON file."""
        # Build market dict matching what dashboard expects
        market = {
            'ticker': self.ticker,
            'current_price': context.current_price,
            'price_change_pct': context.price_change_pct,
            'vix_level': context.vix_level,
            'vix1d': context.vix1d,
            'vix_regime': context.vix_regime,
            'volume_ratio': context.volume_ratio,
            'is_market_hours': context.is_market_hours,
            'vix_direction': getattr(context, 'vix_direction', 'stable'),
            'vix_velocity': getattr(context, 'vix_velocity', 0.0),
            'vix_change_5m': getattr(context, 'vix_change_5m', None),
            'vix_term_spread': getattr(context, 'vix_term_spread', None),
        }

        # Opportunities as dicts
        opp_dicts = []
        for opp in opportunities[:10]:
            opp_dicts.append(opp.to_dict())

        # Positions
        open_positions = self.tracker.get_open_positions()
        summary = self.tracker.get_summary()

        # Recent alerts from log file
        alerts = []
        if CONFIG.alerts_log.exists():
            try:
                with open(CONFIG.alerts_log, 'r') as f:
                    lines = f.readlines()
                    alerts = [l.strip() for l in lines[-20:]]
            except Exception:
                pass

        data = {
            'timestamp': sim_time.isoformat(),
            'simulation': {
                'active': True,
                'date': str(self.target_date),
                'speed': self.speed_multiplier,
                'step': step,
                'total_steps': total_steps,
                'pct_complete': (step / total_steps) * 100,
                'sim_time': sim_time.strftime('%I:%M %p PST'),
            },
            'market': market,
            'opportunities': opp_dicts,
            'positions': {
                'open': [pos.to_dict() for pos in open_positions],
                'summary': summary,
            },
            'alerts': alerts,
        }

        try:
            CONFIG.dashboard_data.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG.dashboard_data, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"   Warning: Could not write dashboard data: {e}")

    def run(self, auto_start: bool = False):
        """Run the simulation."""
        print("\n" + "=" * 100)
        print(f"🎬 CONTINUOUS MODE DAY SIMULATION")
        print("=" * 100)
        print(f"Date: {self.target_date}")
        print(f"Ticker: {self.ticker}")
        print(f"Trend: {self.trend.upper()}")
        print(f"Speed: {self.speed_multiplier}x (5 real minutes = {self.sim_interval_seconds:.1f} sim seconds)")
        print(f"Trading Hours: {self.start_hour:02d}:{self.start_minute:02d} - {self.end_hour:02d}:{self.end_minute:02d} PST")
        print("=" * 100)

        time_points = self.generate_time_points()
        print(f"\nGenerated {len(time_points)} time points ({self.time_step_minutes}-minute intervals)")
        print(f"Total simulation time: ~{len(time_points) * self.sim_interval_seconds:.0f} seconds "
              f"({len(time_points) * self.sim_interval_seconds / 60:.1f} minutes)")

        if not auto_start:
            input("\nPress Enter to start simulation...")
        else:
            print("\nAuto-starting simulation in 2 seconds...")
            time.sleep(2)

        # Initialize provider
        print("\nInitializing data providers...")
        try:
            self.provider = create_default_provider()
        except Exception as e:
            print(f"Warning: Could not initialize providers: {e}")
            print("Using simulation data only")

        # Load historical VIX data for the target date
        print(f"\nLoading VIX data for {self.target_date}...")
        has_real_vix = self.load_historical_vix()
        if not has_real_vix:
            print("  Using synthetic VIX data (random)")

        # Load option chain for strike resolution
        print(f"\nLoading option chain for {self.target_date}...")
        chain_file = find_option_chain_file(self.ticker, self.target_date)
        if chain_file:
            self._option_chain = load_option_chain(chain_file)
            if self._option_chain:
                n_puts = len(self._option_chain.get('put', []))
                n_calls = len(self._option_chain.get('call', []))
                print(f"  Loaded chain: {n_puts} puts, {n_calls} calls")
            else:
                print("  Chain file found but no valid data")
        else:
            print("  No option chain file found (will use estimated strikes)")

        print("\n🚀 SIMULATION STARTING...\n")
        time.sleep(2)

        # Run simulation
        for i, sim_time in enumerate(time_points, 1):
            tick_start = time.time()

            self.print_header(sim_time)

            # Get market data for this time (real VIX from DB if available)
            sim_data = self.simulate_market_data(sim_time)

            # Record VIX reading in provider for dynamics tracking
            if self.provider is not None:
                self.provider.record_vix_reading(
                    sim_time, sim_data['vix'], sim_data.get('vix1d')
                )

            # Get VIX dynamics from provider's accumulated history
            vix_dynamics = {'vix_change_5m': None, 'vix_change_30m': None,
                           'vix_direction': 'stable', 'vix_velocity': 0.0,
                           'vix_term_spread': None}
            if self.provider is not None:
                vix_dynamics = self.provider.get_vix_dynamics()

            from scripts.regime_strategy_selector import detect_vix_regime

            class MockContext:
                def __init__(self, sim_data, sim_time, dynamics, trend):
                    self.timestamp = sim_time.isoformat()
                    self.ticker = 'NDX'
                    self.current_price = sim_data['price']
                    self.price_change_pct = 0.0
                    self.vix_level = sim_data['vix']
                    self.vix1d = sim_data['vix1d']
                    self.vix_regime = detect_vix_regime(self.vix_level)
                    self.volume_ratio = 1.0
                    self.is_market_hours = True
                    self.current_hour_pst = sim_time.hour
                    self.trend = trend
                    self.is_stale = False

                    # VIX dynamics
                    self.vix_change_5m = dynamics.get('vix_change_5m')
                    self.vix_change_30m = dynamics.get('vix_change_30m')
                    self.vix_direction = dynamics.get('vix_direction', 'stable')
                    self.vix_velocity = dynamics.get('vix_velocity', 0.0)
                    # Term spread: prefer from dynamics, else compute
                    self.vix_term_spread = dynamics.get('vix_term_spread')
                    if self.vix_term_spread is None and self.vix1d is not None:
                        self.vix_term_spread = self.vix_level - self.vix1d

            context = MockContext(sim_data, sim_time, vix_dynamics, self.trend)

            # Calculate price change from previous tick
            if self.prev_price is not None:
                context.price_change_pct = ((context.current_price - self.prev_price) / self.prev_price) * 100
            self.prev_price = context.current_price

            # Check for regime change
            if self.current_regime != context.vix_regime:
                if self.current_regime is not None:
                    print(f"\n⚠️  REGIME CHANGE: {self.current_regime.upper()} → {context.vix_regime.upper()} (VIX {context.vix_level:.2f})")
                    ALERTS.alert_regime_change(self.current_regime, context.vix_regime, context.vix_level)
                self.current_regime = context.vix_regime

            # Print market context
            self.print_market_context(context, sim_data)

            # Scan for opportunities
            try:
                opportunities = scan_opportunities(
                    context, top_n=CONFIG.regime_top_n_configs,
                    option_chain=self._option_chain,
                    target_date=self.target_date,
                )

                actionable = filter_actionable_opportunities(
                    opportunities,
                    require_entry_window=True,
                    require_quality=True,
                    top_n=5
                )

                self.print_opportunities(opportunities, actionable)

                # Track opportunities found
                if actionable:
                    self.opportunities_found.extend([
                        {
                            'time': sim_time,
                            'count': len(actionable),
                        }
                    ])

                # Write to dashboard JSON
                self.write_dashboard_data(context, opportunities, sim_time, i, len(time_points))

            except Exception as e:
                print(f"\n   Error scanning opportunities: {e}")
                # Still write dashboard data on error
                self.write_dashboard_data(context, [], sim_time, i, len(time_points))

            # Progress indicator
            pct_complete = (i / len(time_points)) * 100
            remaining = len(time_points) - i
            eta_seconds = remaining * self.sim_interval_seconds

            print(f"\n📈 Progress: {i}/{len(time_points)} ({pct_complete:.1f}%) | "
                  f"ETA: {eta_seconds/60:.1f} min | "
                  f"Opportunities found: {len(self.opportunities_found)} alerts")

            # Sleep for simulation interval
            elapsed = time.time() - tick_start
            sleep_time = max(0, self.sim_interval_seconds - elapsed)

            if i < len(time_points):  # Don't sleep after last tick
                time.sleep(sleep_time)

        # Simulation complete
        self.print_summary()

    def print_summary(self):
        """Print simulation summary."""
        print("\n" + "=" * 100)
        print("🏁 SIMULATION COMPLETE")
        print("=" * 100)

        print(f"\nDate Simulated: {self.target_date}")
        print(f"Total Opportunity Alerts: {len(self.opportunities_found)}")

        if self.opportunities_found:
            print(f"\n📊 Opportunity Times:")
            for opp in self.opportunities_found:
                print(f"   {opp['time'].strftime('%I:%M %p PST')}: {opp['count']} actionable opportunities")

        print(f"\n✅ Simulation ran at {self.speed_multiplier}x speed")
        print(f"Real trading day: ~6.5 hours → Simulated in: ~13 minutes")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Simulate a trading day')
    parser.add_argument('--date', type=str, default='2026-02-20',
                        help='Date to simulate (YYYY-MM-DD)')
    parser.add_argument('--ticker', type=str, default='NDX',
                        help='Ticker symbol')
    parser.add_argument('--trend', type=str, default='sideways',
                        choices=['up', 'down', 'sideways'],
                        help='Market trend')
    parser.add_argument('--speed', type=int, default=30,
                        help='Speed multiplier (default: 30x = 5 min in 10 sec)')
    parser.add_argument('--auto-start', action='store_true',
                        help='Auto-start without waiting for Enter key')
    args = parser.parse_args()

    # Parse date
    target_date = datetime.strptime(args.date, '%Y-%m-%d').date()

    # Create and run simulator
    simulator = DaySimulator(
        target_date=target_date,
        ticker=args.ticker,
        trend=args.trend,
        speed_multiplier=args.speed,
    )

    simulator.run(auto_start=args.auto_start)


if __name__ == '__main__':
    main()
