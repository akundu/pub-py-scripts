"""
Scale-In on Breach Strategy utilities.

This module implements a layered entry strategy that enters at riskier levels for more premium,
then adds positions at progressively safer levels if breached. This reduces average losses
while capturing more premium.

Strategy Overview:
- L1 (Initial): Enter at market open at a moderately aggressive level
- L2 (First breach): Add position at safer level if L1 strike is hit
- L3 (Final): Add final position at safest level if L2 strike is hit

The strategy recovers 23-68% of losses depending on breach depth compared to single entry.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path


@dataclass
class LayerConfig:
    """Configuration for a single scale-in layer."""
    level: int  # 1, 2, or 3
    percent_beyond: float  # Distance from close (e.g., 0.025 = 2.5%)
    capital_pct: float  # Fraction of total capital (e.g., 0.40 = 40%)
    trigger: str  # "entry", "L1_breach", or "L2_breach"


@dataclass
class ScaleInConfig:
    """Configuration for the scale-in strategy."""
    enabled: bool = True
    total_capital: float = 100000.0
    spread_width: float = 30.0
    min_time_between_layers_minutes: int = 5
    put_layers: List[LayerConfig] = field(default_factory=list)
    call_layers: List[LayerConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: dict) -> 'ScaleInConfig':
        """Create config from dictionary (parsed JSON)."""
        put_layers = []
        call_layers = []

        layers_config = config.get('layers', {})

        for layer_dict in layers_config.get('put', []):
            put_layers.append(LayerConfig(
                level=layer_dict['level'],
                percent_beyond=layer_dict['percent_beyond'],
                capital_pct=layer_dict['capital_pct'],
                trigger=layer_dict['trigger']
            ))

        for layer_dict in layers_config.get('call', []):
            call_layers.append(LayerConfig(
                level=layer_dict['level'],
                percent_beyond=layer_dict['percent_beyond'],
                capital_pct=layer_dict['capital_pct'],
                trigger=layer_dict['trigger']
            ))

        return cls(
            enabled=config.get('enabled', True),
            total_capital=float(config.get('total_capital', 100000.0)),
            spread_width=float(config.get('spread_width', 30.0)),
            min_time_between_layers_minutes=int(config.get('min_time_between_layers_minutes', 5)),
            put_layers=put_layers,
            call_layers=call_layers,
        )

    @classmethod
    def from_file(cls, file_path: str) -> 'ScaleInConfig':
        """Load config from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Scale-in config file not found: {file_path}")

        with open(path, 'r') as f:
            config = json.load(f)

        return cls.from_dict(config)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'total_capital': self.total_capital,
            'spread_width': self.spread_width,
            'min_time_between_layers_minutes': self.min_time_between_layers_minutes,
            'layers': {
                'put': [
                    {
                        'level': layer.level,
                        'percent_beyond': layer.percent_beyond,
                        'capital_pct': layer.capital_pct,
                        'trigger': layer.trigger
                    }
                    for layer in self.put_layers
                ],
                'call': [
                    {
                        'level': layer.level,
                        'percent_beyond': layer.percent_beyond,
                        'capital_pct': layer.capital_pct,
                        'trigger': layer.trigger
                    }
                    for layer in self.call_layers
                ]
            }
        }

    def get_layers(self, option_type: str) -> List[LayerConfig]:
        """Get layers for the specified option type."""
        if option_type.lower() == 'put':
            return self.put_layers
        elif option_type.lower() == 'call':
            return self.call_layers
        else:
            return []


@dataclass
class LayerPosition:
    """Tracks a single layer position in the scale-in strategy."""
    layer_level: int
    option_type: str  # 'put' or 'call'
    entry_time: datetime
    short_strike: float
    long_strike: float
    spread_width: float
    capital_allocated: float
    num_contracts: int
    initial_credit_per_share: float
    initial_credit_total: float
    max_loss_per_share: float
    max_loss_total: float
    triggered: bool = False
    closed: bool = False
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    actual_pnl_per_share: Optional[float] = None
    actual_pnl_total: Optional[float] = None
    breach_detected: bool = False

    @property
    def is_profitable(self) -> Optional[bool]:
        """Check if the position was profitable."""
        if self.actual_pnl_per_share is None:
            return None
        return self.actual_pnl_per_share > 0


@dataclass
class ScaleInTradeState:
    """Tracks the state of a scale-in trade across all layers."""
    trading_date: datetime
    option_type: str
    prev_close: float
    layers: List[LayerPosition] = field(default_factory=list)
    current_layer: int = 0  # 0 = no layers active yet, 1 = L1 active, etc.
    all_closed: bool = False

    @property
    def total_capital_deployed(self) -> float:
        """Total capital deployed across all triggered layers."""
        return sum(layer.capital_allocated for layer in self.layers if layer.triggered)

    @property
    def total_initial_credit(self) -> float:
        """Total initial credit received across all triggered layers."""
        return sum(layer.initial_credit_total for layer in self.layers if layer.triggered)

    @property
    def total_max_loss(self) -> float:
        """Total max loss exposure across all triggered layers."""
        return sum(layer.max_loss_total for layer in self.layers if layer.triggered)

    @property
    def total_actual_pnl(self) -> Optional[float]:
        """Total actual P&L across all closed layers."""
        if not any(layer.closed for layer in self.layers):
            return None
        return sum(
            layer.actual_pnl_total or 0
            for layer in self.layers
            if layer.closed and layer.actual_pnl_total is not None
        )

    @property
    def num_triggered_layers(self) -> int:
        """Number of layers that were triggered."""
        return sum(1 for layer in self.layers if layer.triggered)

    @property
    def num_breached_layers(self) -> int:
        """Number of layers that experienced breach."""
        return sum(1 for layer in self.layers if layer.breach_detected)

    def get_layer(self, level: int) -> Optional[LayerPosition]:
        """Get layer by level number."""
        for layer in self.layers:
            if layer.layer_level == level:
                return layer
        return None


def calculate_layer_strikes(
    prev_close: float,
    layer: LayerConfig,
    option_type: str,
    spread_width: float
) -> Tuple[float, float]:
    """
    Calculate short and long strike prices for a layer.

    Args:
        prev_close: Previous day's closing price
        layer: Layer configuration
        option_type: 'put' or 'call'
        spread_width: Width of the spread in points

    Returns:
        Tuple of (short_strike, long_strike)
    """
    if option_type.lower() == 'put':
        # PUT spread: sell higher strike (closer to money), buy lower strike
        short_strike = prev_close * (1 - layer.percent_beyond)
        long_strike = short_strike - spread_width
    else:
        # CALL spread: sell lower strike (closer to money), buy higher strike
        short_strike = prev_close * (1 + layer.percent_beyond)
        long_strike = short_strike + spread_width

    return (round(short_strike, 2), round(long_strike, 2))


def calculate_layer_contracts(
    total_capital: float,
    capital_pct: float,
    max_loss_per_contract: float
) -> Tuple[int, float]:
    """
    Calculate number of contracts and actual capital for a layer.

    Args:
        total_capital: Total capital for the strategy
        capital_pct: Percentage of capital for this layer
        max_loss_per_contract: Maximum loss per contract

    Returns:
        Tuple of (num_contracts, actual_capital_allocated)
    """
    layer_capital = total_capital * capital_pct

    if max_loss_per_contract <= 0:
        return (0, 0.0)

    num_contracts = int(layer_capital / max_loss_per_contract)
    actual_capital = num_contracts * max_loss_per_contract

    return (num_contracts, actual_capital)


def check_breach(
    option_type: str,
    short_strike: float,
    price: float
) -> bool:
    """
    Check if a price breaches the short strike.

    Args:
        option_type: 'put' or 'call'
        short_strike: The short strike price
        price: Current underlying price

    Returns:
        True if breached, False otherwise
    """
    if option_type.lower() == 'put':
        # PUT spread breached when price falls below short strike
        return price < short_strike
    else:
        # CALL spread breached when price rises above short strike
        return price > short_strike


def calculate_layer_pnl(
    initial_credit: float,
    short_strike: float,
    long_strike: float,
    close_price: float,
    option_type: str
) -> float:
    """
    Calculate P&L for a layer based on closing price.

    Args:
        initial_credit: Credit received per share when opening
        short_strike: Short strike price
        long_strike: Long strike price
        close_price: Price at close/expiration
        option_type: 'put' or 'call'

    Returns:
        P&L per share (positive = profit, negative = loss)
    """
    # Calculate intrinsic value of the spread at close
    if option_type.lower() == 'put':
        # PUT spread: short strike > long strike
        if close_price >= short_strike:
            # Both options OTM, spread worthless
            spread_value = 0.0
        elif close_price <= long_strike:
            # Both options ITM, spread at max width
            spread_value = short_strike - long_strike
        else:
            # Price between strikes
            spread_value = short_strike - close_price
    else:
        # CALL spread: short strike < long strike
        if close_price <= short_strike:
            # Both options OTM, spread worthless
            spread_value = 0.0
        elif close_price >= long_strike:
            # Both options ITM, spread at max width
            spread_value = long_strike - short_strike
        else:
            # Price between strikes
            spread_value = close_price - short_strike

    # P&L = credit received - spread value
    return initial_credit - spread_value


def calculate_layered_pnl(
    trade_state: ScaleInTradeState,
    close_price: float,
    close_time: Optional[datetime] = None
) -> ScaleInTradeState:
    """
    Calculate P&L for all layers in a scale-in trade.

    Args:
        trade_state: The trade state with all layer positions
        close_price: The closing price of the underlying
        close_time: Time when positions closed

    Returns:
        Updated trade state with P&L calculated
    """
    for layer in trade_state.layers:
        if not layer.triggered:
            continue

        layer.close_price = close_price
        layer.close_time = close_time

        # Check for breach
        layer.breach_detected = check_breach(
            layer.option_type,
            layer.short_strike,
            close_price
        )

        # Calculate P&L per share
        layer.actual_pnl_per_share = calculate_layer_pnl(
            layer.initial_credit_per_share,
            layer.short_strike,
            layer.long_strike,
            close_price,
            layer.option_type
        )

        # Calculate total P&L
        layer.actual_pnl_total = layer.actual_pnl_per_share * layer.num_contracts * 100
        layer.closed = True

    trade_state.all_closed = True
    return trade_state


def initialize_scale_in_trade(
    trading_date: datetime,
    option_type: str,
    prev_close: float,
    config: ScaleInConfig,
    initial_credit_estimate: float = 3.50,  # Per-share credit estimate
    logger: Optional[logging.Logger] = None
) -> ScaleInTradeState:
    """
    Initialize a scale-in trade with all layer positions.

    Args:
        trading_date: The trading date
        option_type: 'put' or 'call'
        prev_close: Previous day's closing price
        config: Scale-in configuration
        initial_credit_estimate: Estimated credit per share (used for contract calc)
        logger: Optional logger

    Returns:
        Initialized trade state with all layer positions
    """
    trade_state = ScaleInTradeState(
        trading_date=trading_date,
        option_type=option_type,
        prev_close=prev_close,
        layers=[],
        current_layer=0
    )

    layers = config.get_layers(option_type)

    for layer_config in layers:
        # Calculate strikes
        short_strike, long_strike = calculate_layer_strikes(
            prev_close,
            layer_config,
            option_type,
            config.spread_width
        )

        # Calculate max loss per share and per contract
        max_loss_per_share = config.spread_width - initial_credit_estimate
        if max_loss_per_share < 0:
            max_loss_per_share = config.spread_width * 0.5  # Fallback

        max_loss_per_contract = max_loss_per_share * 100

        # Calculate number of contracts
        num_contracts, capital_allocated = calculate_layer_contracts(
            config.total_capital,
            layer_config.capital_pct,
            max_loss_per_contract
        )

        # Create layer position
        layer_position = LayerPosition(
            layer_level=layer_config.level,
            option_type=option_type,
            entry_time=trading_date,
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=config.spread_width,
            capital_allocated=capital_allocated,
            num_contracts=num_contracts,
            initial_credit_per_share=initial_credit_estimate,
            initial_credit_total=initial_credit_estimate * num_contracts * 100,
            max_loss_per_share=max_loss_per_share,
            max_loss_total=max_loss_per_contract * num_contracts,
            triggered=(layer_config.trigger == 'entry'),  # L1 triggers at entry
        )

        trade_state.layers.append(layer_position)

        if layer_config.trigger == 'entry':
            trade_state.current_layer = layer_config.level

        if logger:
            logger.debug(
                f"Layer {layer_config.level} ({option_type}): "
                f"Short={short_strike:.2f}, Long={long_strike:.2f}, "
                f"Contracts={num_contracts}, Capital=${capital_allocated:,.2f}, "
                f"Trigger={layer_config.trigger}"
            )

    return trade_state


def process_price_update(
    trade_state: ScaleInTradeState,
    current_price: float,
    current_time: datetime,
    config: ScaleInConfig,
    logger: Optional[logging.Logger] = None
) -> Tuple[ScaleInTradeState, bool]:
    """
    Process a price update and trigger new layers if needed.

    Args:
        trade_state: Current trade state
        current_price: Current underlying price
        current_time: Current timestamp
        config: Scale-in configuration
        logger: Optional logger

    Returns:
        Tuple of (updated_trade_state, new_layer_triggered)
    """
    new_layer_triggered = False
    layers = config.get_layers(trade_state.option_type)

    for layer_config in layers:
        layer_position = trade_state.get_layer(layer_config.level)
        if layer_position is None:
            continue

        # Skip already triggered layers
        if layer_position.triggered:
            # Check for breach on triggered layers
            if check_breach(trade_state.option_type, layer_position.short_strike, current_price):
                layer_position.breach_detected = True
            continue

        # Check if this layer should be triggered
        should_trigger = False

        if layer_config.trigger == 'L1_breach':
            l1 = trade_state.get_layer(1)
            if l1 and l1.triggered:
                should_trigger = check_breach(
                    trade_state.option_type,
                    l1.short_strike,
                    current_price
                )

        elif layer_config.trigger == 'L2_breach':
            l2 = trade_state.get_layer(2)
            if l2 and l2.triggered:
                should_trigger = check_breach(
                    trade_state.option_type,
                    l2.short_strike,
                    current_price
                )

        if should_trigger:
            # Check minimum time between layers
            prev_layer = trade_state.get_layer(layer_config.level - 1)
            if prev_layer and prev_layer.triggered:
                time_since_prev = (current_time - prev_layer.entry_time).total_seconds() / 60
                if time_since_prev < config.min_time_between_layers_minutes:
                    if logger:
                        logger.debug(
                            f"Layer {layer_config.level} trigger skipped: "
                            f"only {time_since_prev:.1f}min since L{layer_config.level - 1}"
                        )
                    continue

            layer_position.triggered = True
            layer_position.entry_time = current_time
            trade_state.current_layer = layer_config.level
            new_layer_triggered = True

            if logger:
                logger.info(
                    f"Layer {layer_config.level} TRIGGERED at price {current_price:.2f}: "
                    f"Short={layer_position.short_strike:.2f}, "
                    f"Contracts={layer_position.num_contracts}"
                )

    return trade_state, new_layer_triggered


def generate_scale_in_summary(
    trade_state: ScaleInTradeState,
    single_entry_pnl: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate a summary of the scale-in trade.

    Args:
        trade_state: The trade state with all layer positions
        single_entry_pnl: Optional P&L for comparison with single-entry strategy

    Returns:
        Summary dictionary with key metrics
    """
    summary = {
        'trading_date': trade_state.trading_date,
        'option_type': trade_state.option_type,
        'prev_close': trade_state.prev_close,
        'num_layers_triggered': trade_state.num_triggered_layers,
        'num_layers_breached': trade_state.num_breached_layers,
        'total_capital_deployed': trade_state.total_capital_deployed,
        'total_initial_credit': trade_state.total_initial_credit,
        'total_max_loss': trade_state.total_max_loss,
        'total_actual_pnl': trade_state.total_actual_pnl,
        'all_closed': trade_state.all_closed,
        'layers': []
    }

    for layer in trade_state.layers:
        layer_summary = {
            'level': layer.layer_level,
            'triggered': layer.triggered,
            'short_strike': layer.short_strike,
            'long_strike': layer.long_strike,
            'num_contracts': layer.num_contracts,
            'capital_allocated': layer.capital_allocated,
            'initial_credit': layer.initial_credit_total,
            'breach_detected': layer.breach_detected,
            'actual_pnl': layer.actual_pnl_total,
            'is_profitable': layer.is_profitable,
        }
        summary['layers'].append(layer_summary)

    # Calculate recovery if single entry P&L is provided
    if single_entry_pnl is not None and trade_state.total_actual_pnl is not None:
        if single_entry_pnl < 0 and trade_state.total_actual_pnl > single_entry_pnl:
            loss_single = abs(single_entry_pnl)
            loss_scale_in = abs(trade_state.total_actual_pnl) if trade_state.total_actual_pnl < 0 else 0
            recovery_amount = loss_single - loss_scale_in
            recovery_pct = (recovery_amount / loss_single) * 100 if loss_single > 0 else 0
            summary['recovery_vs_single'] = {
                'single_entry_pnl': single_entry_pnl,
                'recovery_amount': recovery_amount,
                'recovery_pct': recovery_pct
            }

    return summary


def format_scale_in_result(summary: Dict[str, Any]) -> str:
    """
    Format the scale-in summary for display.

    Args:
        summary: Summary dictionary from generate_scale_in_summary

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"SCALE-IN TRADE SUMMARY - {summary['option_type'].upper()}")
    lines.append(f"{'='*80}")
    lines.append(f"Trading Date: {summary['trading_date']}")
    lines.append(f"Previous Close: ${summary['prev_close']:,.2f}")
    lines.append(f"Layers Triggered: {summary['num_layers_triggered']}")
    lines.append(f"Layers Breached: {summary['num_layers_breached']}")
    lines.append(f"")
    lines.append(f"FINANCIAL SUMMARY:")
    lines.append(f"  Capital Deployed: ${summary['total_capital_deployed']:,.2f}")
    lines.append(f"  Initial Credit: ${summary['total_initial_credit']:,.2f}")
    lines.append(f"  Max Loss: ${summary['total_max_loss']:,.2f}")

    if summary['total_actual_pnl'] is not None:
        pnl = summary['total_actual_pnl']
        pnl_str = f"${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        lines.append(f"  Actual P&L: {pnl_str}")

    lines.append(f"")
    lines.append(f"LAYER DETAILS:")
    lines.append(f"  {'Layer':<8} {'Triggered':<10} {'Strike Range':<20} {'Contracts':<12} {'P&L':<15}")
    lines.append(f"  {'-'*8} {'-'*10} {'-'*20} {'-'*12} {'-'*15}")

    for layer in summary['layers']:
        triggered = "Yes" if layer['triggered'] else "No"
        strike_range = f"${layer['short_strike']:,.0f}/${layer['long_strike']:,.0f}"
        contracts = str(layer['num_contracts']) if layer['triggered'] else "-"
        if layer['actual_pnl'] is not None:
            pnl = layer['actual_pnl']
            pnl_str = f"${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        else:
            pnl_str = "-"

        lines.append(f"  L{layer['level']:<7} {triggered:<10} {strike_range:<20} {contracts:<12} {pnl_str:<15}")

    if 'recovery_vs_single' in summary:
        recovery = summary['recovery_vs_single']
        lines.append(f"")
        lines.append(f"RECOVERY ANALYSIS:")
        lines.append(f"  Single Entry P&L: ${recovery['single_entry_pnl']:,.2f}")
        lines.append(f"  Recovery Amount: ${recovery['recovery_amount']:,.2f}")
        lines.append(f"  Recovery %: {recovery['recovery_pct']:.1f}%")

    lines.append(f"{'='*80}")

    return "\n".join(lines)


def load_scale_in_config(config_path: Optional[str]) -> Optional[ScaleInConfig]:
    """
    Load scale-in configuration from file path.

    Args:
        config_path: Path to JSON config file

    Returns:
        ScaleInConfig or None if path is None/invalid
    """
    if not config_path:
        return None

    try:
        return ScaleInConfig.from_file(config_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load scale-in config: {e}")
