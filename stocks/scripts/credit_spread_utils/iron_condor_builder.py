"""
Iron condor builder using percentile-based strikes.

This module constructs 4-leg iron condors with short strikes positioned
at historical percentile boundaries instead of using combinatorial search.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys
import logging
import pandas as pd

# Add parent directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class IronCondorBuilder:
    """Build 4-leg iron condors using percentile-based strikes."""

    def __init__(
        self,
        min_credit: float = 0.50,
        min_rr_ratio: float = 0.15,
        min_wing_width: float = 10.0,
        max_wing_width: float = 100.0,
        use_mid_price: bool = False
    ):
        """
        Initialize iron condor builder.

        Args:
            min_credit: Minimum total credit to accept
            min_rr_ratio: Minimum risk/reward ratio
            min_wing_width: Minimum spread width per side
            max_wing_width: Maximum spread width per side
            use_mid_price: Use mid price instead of bid/ask
        """
        self.min_credit = min_credit
        self.min_rr_ratio = min_rr_ratio
        self.min_wing_width = min_wing_width
        self.max_wing_width = max_wing_width
        self.use_mid_price = use_mid_price

    def build_iron_condor(
        self,
        options_df: pd.DataFrame,
        call_target_strike: float,
        put_target_strike: float,
        call_spread_width: float,
        put_spread_width: float,
        prev_close: float
    ) -> List[Dict[str, Any]]:
        """
        Build iron condors with short strikes near percentile boundaries.

        Iron Condor Structure:
        - Put side: long_put < short_put (target) < prev_close
        - Call side: prev_close < short_call (target) < long_call
        - Validation: long_put < short_put < prev_close < short_call < long_call

        Args:
            options_df: DataFrame with option data (strike_price, bid, ask, option_type)
            call_target_strike: Target strike for short call (from percentile)
            put_target_strike: Target strike for short put (from percentile)
            call_spread_width: Width of call spread
            put_spread_width: Width of put spread
            prev_close: Previous close price (for validation)

        Returns:
            List of iron condor dicts, each with 4 legs and metrics
        """
        if options_df.empty:
            logger.warning("Empty options DataFrame")
            return []

        # Separate puts and calls
        # Try 'option_type' first, fall back to 'type'
        type_col = 'option_type' if 'option_type' in options_df.columns else 'type'
        puts = options_df[options_df[type_col].str.upper() == 'PUT'].copy()
        calls = options_df[options_df[type_col].str.upper() == 'CALL'].copy()

        if puts.empty or calls.empty:
            logger.warning(f"Missing puts ({len(puts)}) or calls ({len(calls)})")
            return []

        # Filter out invalid pricing
        puts = puts[(puts['bid'] > 0) & (puts['ask'] > 0) & (puts['bid'] < puts['ask'])]
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0) & (calls['bid'] < calls['ask'])]

        # Build call spread (short call near target, long call further out)
        call_spreads = self._build_call_spread(
            calls, call_target_strike, call_spread_width
        )

        # Build put spread (short put near target, long put further out)
        put_spreads = self._build_put_spread(
            puts, put_target_strike, put_spread_width
        )

        logger.debug(
            f"Built {len(call_spreads)} call spreads and {len(put_spreads)} put spreads"
        )

        # Combine into iron condors
        iron_condors = []

        for call_spread in call_spreads:
            for put_spread in put_spreads:
                # Validate structure
                if not self._validate_iron_condor_structure(
                    put_spread, call_spread, prev_close
                ):
                    continue

                # Calculate combined metrics
                ic = self._combine_spreads(put_spread, call_spread)

                # Filter by criteria
                if not self._passes_filters(ic):
                    continue

                iron_condors.append(ic)

        logger.info(
            f"Created {len(iron_condors)} valid iron condors from "
            f"{len(call_spreads)} call spreads × {len(put_spreads)} put spreads"
        )

        return iron_condors

    def _build_call_spread(
        self,
        calls: pd.DataFrame,
        target_strike: float,
        spread_width: float
    ) -> List[Dict[str, Any]]:
        """
        Build call spreads with short strike near target.

        Args:
            calls: DataFrame of call options
            target_strike: Target strike for short call
            spread_width: Spread width

        Returns:
            List of call spread dicts
        """
        spreads = []

        # Use correct column name (strike or strike_price)
        strike_col = 'strike_price' if 'strike_price' in calls.columns else 'strike'

        # Find calls near target strike (short call)
        tolerance = target_strike * 0.01  # 1% tolerance
        short_calls = calls[
            (calls[strike_col] >= target_strike - tolerance) &
            (calls[strike_col] <= target_strike + tolerance)
        ].sort_values(strike_col)

        for _, short_call in short_calls.iterrows():
            short_strike = short_call[strike_col]
            long_strike = short_strike + spread_width

            # Find long call at long_strike
            long_calls = calls[
                (calls[strike_col] >= long_strike - 0.5) &
                (calls[strike_col] <= long_strike + 0.5)
            ]

            if long_calls.empty:
                continue

            long_call = long_calls.iloc[0]

            # Calculate credit
            if self.use_mid_price:
                short_credit = (short_call['bid'] + short_call['ask']) / 2
                long_cost = (long_call['bid'] + long_call['ask']) / 2
            else:
                short_credit = short_call['bid']  # Sell at bid
                long_cost = long_call['ask']  # Buy at ask

            net_credit = short_credit - long_cost

            if net_credit <= 0:
                continue

            spreads.append({
                'short_call': short_call.to_dict(),
                'long_call': long_call.to_dict(),
                'short_strike': short_strike,
                'long_strike': long_strike,
                'width': spread_width,
                'credit': net_credit
            })

        return spreads

    def _build_put_spread(
        self,
        puts: pd.DataFrame,
        target_strike: float,
        spread_width: float
    ) -> List[Dict[str, Any]]:
        """
        Build put spreads with short strike near target.

        Args:
            puts: DataFrame of put options
            target_strike: Target strike for short put
            spread_width: Spread width

        Returns:
            List of put spread dicts
        """
        spreads = []

        # Use correct column name (strike or strike_price)
        strike_col = 'strike_price' if 'strike_price' in puts.columns else 'strike'

        # Find puts near target strike (short put)
        tolerance = target_strike * 0.01  # 1% tolerance
        short_puts = puts[
            (puts[strike_col] >= target_strike - tolerance) &
            (puts[strike_col] <= target_strike + tolerance)
        ].sort_values(strike_col)

        for _, short_put in short_puts.iterrows():
            short_strike = short_put[strike_col]
            long_strike = short_strike - spread_width

            # Find long put at long_strike
            long_puts = puts[
                (puts[strike_col] >= long_strike - 0.5) &
                (puts[strike_col] <= long_strike + 0.5)
            ]

            if long_puts.empty:
                continue

            long_put = long_puts.iloc[0]

            # Calculate credit
            if self.use_mid_price:
                short_credit = (short_put['bid'] + short_put['ask']) / 2
                long_cost = (long_put['bid'] + long_put['ask']) / 2
            else:
                short_credit = short_put['bid']  # Sell at bid
                long_cost = long_put['ask']  # Buy at ask

            net_credit = short_credit - long_cost

            if net_credit <= 0:
                continue

            spreads.append({
                'short_put': short_put.to_dict(),
                'long_put': long_put.to_dict(),
                'short_strike': short_strike,
                'long_strike': long_strike,
                'width': spread_width,
                'credit': net_credit
            })

        return spreads

    def _validate_iron_condor_structure(
        self,
        put_spread: Dict,
        call_spread: Dict,
        prev_close: float
    ) -> bool:
        """
        Validate iron condor structure.

        Required: long_put < short_put < prev_close < short_call < long_call

        Args:
            put_spread: Put spread dict
            call_spread: Call spread dict
            prev_close: Previous close price

        Returns:
            True if valid structure
        """
        long_put_strike = put_spread['long_strike']
        short_put_strike = put_spread['short_strike']
        short_call_strike = call_spread['short_strike']
        long_call_strike = call_spread['long_strike']

        # Check ordering
        valid = (
            long_put_strike < short_put_strike < prev_close <
            short_call_strike < long_call_strike
        )

        if not valid:
            logger.debug(
                f"Invalid structure: {long_put_strike:.2f} < {short_put_strike:.2f} < "
                f"{prev_close:.2f} < {short_call_strike:.2f} < {long_call_strike:.2f}"
            )

        return valid

    def _combine_spreads(
        self,
        put_spread: Dict,
        call_spread: Dict
    ) -> Dict[str, Any]:
        """
        Combine put and call spreads into iron condor.

        Args:
            put_spread: Put spread dict
            call_spread: Call spread dict

        Returns:
            Iron condor dict with combined metrics
        """
        total_credit = put_spread['credit'] + call_spread['credit']
        put_width = put_spread['width']
        call_width = call_spread['width']

        # Max loss is the wider wing minus total credit
        max_loss = max(put_width, call_width) - total_credit

        # Risk/reward ratio
        rr_ratio = total_credit / max_loss if max_loss > 0 else 0

        return {
            'put_spread': put_spread,
            'call_spread': call_spread,
            'total_credit': total_credit,
            'put_width': put_width,
            'call_width': call_width,
            'max_loss': max_loss,
            'rr_ratio': rr_ratio,
            'short_put_strike': put_spread['short_strike'],
            'long_put_strike': put_spread['long_strike'],
            'short_call_strike': call_spread['short_strike'],
            'long_call_strike': call_spread['long_strike']
        }

    def _passes_filters(self, iron_condor: Dict[str, Any]) -> bool:
        """
        Check if iron condor passes filter criteria.

        Args:
            iron_condor: Iron condor dict

        Returns:
            True if passes all filters
        """
        # Credit filter
        if iron_condor['total_credit'] < self.min_credit:
            logger.debug(
                f"Rejected: credit {iron_condor['total_credit']:.2f} < "
                f"min {self.min_credit:.2f}"
            )
            return False

        # Risk/reward filter
        if iron_condor['rr_ratio'] < self.min_rr_ratio:
            logger.debug(
                f"Rejected: R/R {iron_condor['rr_ratio']:.3f} < "
                f"min {self.min_rr_ratio:.3f}"
            )
            return False

        # Wing width filters
        if iron_condor['put_width'] < self.min_wing_width:
            logger.debug(
                f"Rejected: put width {iron_condor['put_width']:.2f} < "
                f"min {self.min_wing_width:.2f}"
            )
            return False

        if iron_condor['call_width'] < self.min_wing_width:
            logger.debug(
                f"Rejected: call width {iron_condor['call_width']:.2f} < "
                f"min {self.min_wing_width:.2f}"
            )
            return False

        if iron_condor['put_width'] > self.max_wing_width:
            logger.debug(
                f"Rejected: put width {iron_condor['put_width']:.2f} > "
                f"max {self.max_wing_width:.2f}"
            )
            return False

        if iron_condor['call_width'] > self.max_wing_width:
            logger.debug(
                f"Rejected: call width {iron_condor['call_width']:.2f} > "
                f"max {self.max_wing_width:.2f}"
            )
            return False

        return True

    def get_best_iron_condor(
        self,
        iron_condors: List[Dict[str, Any]],
        sort_by: str = 'total_credit'
    ) -> Optional[Dict[str, Any]]:
        """
        Get best iron condor from list.

        Args:
            iron_condors: List of iron condor dicts
            sort_by: 'total_credit', 'rr_ratio', or 'max_loss'

        Returns:
            Best iron condor dict, or None if list empty
        """
        if not iron_condors:
            return None

        if sort_by == 'total_credit':
            best = max(iron_condors, key=lambda x: x['total_credit'])
        elif sort_by == 'rr_ratio':
            best = max(iron_condors, key=lambda x: x['rr_ratio'])
        elif sort_by == 'max_loss':
            best = min(iron_condors, key=lambda x: x['max_loss'])
        else:
            raise ValueError(f"Invalid sort_by: {sort_by}")

        logger.info(
            f"Best iron condor by {sort_by}: "
            f"credit={best['total_credit']:.2f}, "
            f"R/R={best['rr_ratio']:.3f}, "
            f"max_loss={best['max_loss']:.2f}"
        )

        return best


# Convenience function
def build_iron_condors_from_percentiles(
    options_df: pd.DataFrame,
    call_percentile_strike: float,
    put_percentile_strike: float,
    call_spread_width: float,
    put_spread_width: float,
    prev_close: float,
    min_credit: float = 0.50
) -> List[Dict[str, Any]]:
    """
    Build iron condors using percentile-based strikes.

    Returns:
        List of valid iron condor dicts
    """
    builder = IronCondorBuilder(min_credit=min_credit)

    return builder.build_iron_condor(
        options_df,
        call_percentile_strike,
        put_percentile_strike,
        call_spread_width,
        put_spread_width,
        prev_close
    )
