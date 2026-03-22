"""Playbook service — parse YAML instruction files, translate to trades, execute."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Optional, Union

import yaml

from app.models import (
    Broker,
    EquityOrder,
    InstructionResult,
    LedgerEventType,
    MultiLegOrder,
    OptionAction,
    OptionLeg,
    OptionType,
    OrderResult,
    OrderSide,
    OrderType,
    PlaybookDefinition,
    PlaybookInstruction,
    PlaybookResult,
    PositionSource,
    TradeRequest,
)

logger = logging.getLogger(__name__)


class PlaybookValidationError(Exception):
    """Raised when a playbook fails validation."""


class PlaybookService:
    """Parse YAML playbooks, translate instructions to TradeRequest, and execute."""

    def load(self, source: Union[str, Path, dict]) -> PlaybookDefinition:
        """Load a playbook from a file path, YAML string, or pre-parsed dict.

        Args:
            source: Path to YAML file, raw YAML string, or already-parsed dict.

        Returns:
            PlaybookDefinition with all instructions parsed.

        Raises:
            PlaybookValidationError: If the YAML structure is invalid.
        """
        if isinstance(source, dict):
            raw = source
        elif isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.endswith((".yaml", ".yml"))):
            path = Path(source)
            if not path.exists():
                raise PlaybookValidationError(f"Playbook file not found: {path}")
            with open(path) as f:
                raw = yaml.safe_load(f)
        else:
            raw = yaml.safe_load(source)

        if not isinstance(raw, dict):
            raise PlaybookValidationError("Playbook must be a YAML mapping")

        playbook_meta = raw.get("playbook", {})
        if not isinstance(playbook_meta, dict):
            raise PlaybookValidationError("'playbook' section must be a mapping")

        name = playbook_meta.get("name", "Unnamed Playbook")
        description = playbook_meta.get("description", "")
        broker_str = playbook_meta.get("broker", "ibkr")
        try:
            broker = Broker(broker_str)
        except ValueError:
            raise PlaybookValidationError(f"Unknown broker: {broker_str}")

        raw_instructions = raw.get("instructions", [])
        if not isinstance(raw_instructions, list):
            raise PlaybookValidationError("'instructions' must be a list")
        if not raw_instructions:
            raise PlaybookValidationError("Playbook has no instructions")

        instructions = []
        seen_ids = set()
        for i, item in enumerate(raw_instructions):
            if not isinstance(item, dict):
                raise PlaybookValidationError(f"Instruction {i} must be a mapping")

            instr_id = item.get("id")
            if not instr_id:
                raise PlaybookValidationError(f"Instruction {i} missing 'id'")
            if instr_id in seen_ids:
                raise PlaybookValidationError(f"Duplicate instruction id: {instr_id}")
            seen_ids.add(instr_id)

            instr_type = item.get("type")
            if not instr_type:
                raise PlaybookValidationError(f"Instruction '{instr_id}' missing 'type'")

            valid_types = {"equity", "single_option", "credit_spread", "debit_spread", "iron_condor",
                          "credit_spread_close", "debit_spread_close", "iron_condor_close"}
            if instr_type not in valid_types:
                raise PlaybookValidationError(
                    f"Instruction '{instr_id}' has invalid type '{instr_type}'. "
                    f"Valid types: {sorted(valid_types)}"
                )

            # All remaining keys become params
            params = {k: v for k, v in item.items() if k not in ("id", "type")}
            instructions.append(PlaybookInstruction(id=instr_id, type=instr_type, params=params))

        return PlaybookDefinition(
            name=name,
            description=description,
            broker=broker,
            instructions=instructions,
        )

    def instruction_to_trade_request(
        self, instr: PlaybookInstruction, broker: Broker
    ) -> TradeRequest:
        """Convert a playbook instruction to a TradeRequest.

        Args:
            instr: The parsed instruction.
            broker: Default broker from the playbook.

        Returns:
            A TradeRequest ready for execute_trade().

        Raises:
            PlaybookValidationError: If required fields are missing.
        """
        p = instr.params
        instr_broker = Broker(p["broker"]) if "broker" in p else broker

        if instr.type == "equity":
            return self._build_equity(instr, instr_broker, p)
        elif instr.type == "single_option":
            return self._build_single_option(instr, instr_broker, p)
        elif instr.type == "credit_spread":
            return self._build_credit_spread(instr, instr_broker, p)
        elif instr.type == "credit_spread_close":
            return self._build_credit_spread(instr, instr_broker, p, close=True)
        elif instr.type == "debit_spread":
            return self._build_debit_spread(instr, instr_broker, p)
        elif instr.type == "debit_spread_close":
            return self._build_debit_spread(instr, instr_broker, p, close=True)
        elif instr.type == "iron_condor":
            return self._build_iron_condor(instr, instr_broker, p)
        elif instr.type == "iron_condor_close":
            return self._build_iron_condor(instr, instr_broker, p, close=True)
        else:
            raise PlaybookValidationError(f"Unknown instruction type: {instr.type}")

    async def execute(
        self,
        playbook: PlaybookDefinition,
        dry_run: bool = False,
        post_submit_hook: Optional[
            Callable[[str, TradeRequest, OrderResult], Awaitable[OrderResult]]
        ] = None,
    ) -> PlaybookResult:
        """Execute all instructions in a playbook sequentially.

        Args:
            playbook: Parsed playbook definition.
            dry_run: If True, simulates without sending to broker.
            post_submit_hook: Optional async callback invoked after each live
                order submission. Receives (instruction_id, trade_request,
                initial_order_result) and should return the final OrderResult
                (e.g. after polling for fill). If None, the initial result is
                used as-is.

        Returns:
            PlaybookResult with per-instruction outcomes.
        """
        from app.services.ledger import get_ledger
        from app.services.trade_service import execute_trade

        result = PlaybookResult(
            playbook_name=playbook.name,
            total=len(playbook.instructions),
        )

        for instr in playbook.instructions:
            try:
                trade_request = self.instruction_to_trade_request(instr, playbook.broker)
                order_result = await execute_trade(trade_request, dry_run=dry_run)

                # If live and hook provided, wait for fill tracking
                if not dry_run and post_submit_hook is not None:
                    order_result = await post_submit_hook(
                        instr.id, trade_request, order_result,
                    )

                status = "dry_run" if dry_run else "success"
                instr_result = InstructionResult(
                    instruction_id=instr.id,
                    status=status,
                    order_result=order_result,
                )
                result.succeeded += 1

                # Tag ledger with playbook metadata
                ledger = get_ledger()
                if ledger:
                    from app.models import LedgerEntry
                    await ledger.append(LedgerEntry(
                        event_type=LedgerEventType.PLAYBOOK_EXECUTED,
                        broker=playbook.broker,
                        order_id=order_result.order_id,
                        source=PositionSource.PAPER if dry_run else PositionSource.LIVE_API,
                        dry_run=dry_run,
                        data={
                            "playbook_name": playbook.name,
                            "instruction_id": instr.id,
                            "instruction_type": instr.type,
                        },
                    ))

            except Exception as e:
                logger.error("Instruction '%s' failed: %s", instr.id, e)
                instr_result = InstructionResult(
                    instruction_id=instr.id,
                    status="failed",
                    error=str(e),
                )
                result.failed += 1

            result.results.append(instr_result)

        return result

    async def validate(
        self, playbook: PlaybookDefinition
    ) -> list[dict]:
        """Validate all instructions without executing.

        Returns a list of validation results per instruction.
        """
        validations = []
        for instr in playbook.instructions:
            try:
                self.instruction_to_trade_request(instr, playbook.broker)
                validations.append({
                    "instruction_id": instr.id,
                    "valid": True,
                    "error": None,
                })
            except Exception as e:
                validations.append({
                    "instruction_id": instr.id,
                    "valid": False,
                    "error": str(e),
                })
        return validations

    # ── Private builders ─────────────────────────────────────────────────────

    def _require(self, params: dict, key: str, instr_id: str):
        if key not in params:
            raise PlaybookValidationError(f"Instruction '{instr_id}' missing required field '{key}'")
        return params[key]

    def _build_equity(self, instr: PlaybookInstruction, broker: Broker, p: dict) -> TradeRequest:
        symbol = self._require(p, "symbol", instr.id)
        quantity = int(self._require(p, "quantity", instr.id))
        action = str(p.get("action", "BUY")).upper()
        order_type_str = str(p.get("order_type", "MARKET")).upper()

        return TradeRequest(
            equity_order=EquityOrder(
                broker=broker,
                symbol=symbol,
                side=OrderSide(action),
                quantity=quantity,
                order_type=OrderType(order_type_str),
                limit_price=p.get("limit_price"),
                time_in_force=p.get("time_in_force", "DAY"),
            )
        )

    def _build_single_option(self, instr: PlaybookInstruction, broker: Broker, p: dict) -> TradeRequest:
        symbol = self._require(p, "symbol", instr.id)
        expiration = str(self._require(p, "expiration", instr.id))
        strike = float(self._require(p, "strike", instr.id))
        option_type_str = str(self._require(p, "option_type", instr.id)).upper()
        action_str = str(p.get("action", "BUY_TO_OPEN")).upper()
        quantity = int(p.get("quantity", 1))
        net_price = p.get("limit_price") or p.get("net_price")
        order_type_str = str(p.get("order_type", "LIMIT" if net_price else "MARKET")).upper()

        return TradeRequest(
            multi_leg_order=MultiLegOrder(
                broker=broker,
                legs=[
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=strike,
                        option_type=OptionType(option_type_str),
                        action=OptionAction(action_str),
                        quantity=quantity,
                    ),
                ],
                order_type=OrderType(order_type_str),
                net_price=net_price,
                quantity=quantity,
                time_in_force=p.get("time_in_force", "DAY"),
            )
        )

    def _build_credit_spread(self, instr: PlaybookInstruction, broker: Broker, p: dict, close: bool = False) -> TradeRequest:
        symbol = self._require(p, "symbol", instr.id)
        expiration = str(self._require(p, "expiration", instr.id))
        short_strike = float(self._require(p, "short_strike", instr.id))
        long_strike = float(self._require(p, "long_strike", instr.id))
        option_type_str = str(self._require(p, "option_type", instr.id)).upper()
        quantity = int(p.get("quantity", 1))

        option_type = OptionType(option_type_str)

        if close:
            # Close: buy back short, sell back long
            short_action = OptionAction.BUY_TO_CLOSE
            long_action = OptionAction.SELL_TO_CLOSE
        else:
            short_action = OptionAction.SELL_TO_OPEN
            long_action = OptionAction.BUY_TO_OPEN

        return TradeRequest(
            multi_leg_order=MultiLegOrder(
                broker=broker,
                legs=[
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=short_strike,
                        option_type=option_type,
                        action=short_action,
                        quantity=1,
                    ),
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=long_strike,
                        option_type=option_type,
                        action=long_action,
                        quantity=1,
                    ),
                ],
                order_type=OrderType.LIMIT if p.get("net_price") else OrderType.MARKET,
                net_price=p.get("net_price"),
                quantity=quantity,
                time_in_force=p.get("time_in_force", "DAY"),
            )
        )

    def _build_debit_spread(self, instr: PlaybookInstruction, broker: Broker, p: dict, close: bool = False) -> TradeRequest:
        symbol = self._require(p, "symbol", instr.id)
        expiration = str(self._require(p, "expiration", instr.id))
        long_strike = float(self._require(p, "long_strike", instr.id))
        short_strike = float(self._require(p, "short_strike", instr.id))
        option_type_str = str(self._require(p, "option_type", instr.id)).upper()
        quantity = int(p.get("quantity", 1))

        option_type = OptionType(option_type_str)

        if close:
            long_action = OptionAction.SELL_TO_CLOSE
            short_action = OptionAction.BUY_TO_CLOSE
        else:
            long_action = OptionAction.BUY_TO_OPEN
            short_action = OptionAction.SELL_TO_OPEN

        return TradeRequest(
            multi_leg_order=MultiLegOrder(
                broker=broker,
                legs=[
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=long_strike,
                        option_type=option_type,
                        action=long_action,
                        quantity=1,
                    ),
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=short_strike,
                        option_type=option_type,
                        action=short_action,
                        quantity=1,
                    ),
                ],
                order_type=OrderType.LIMIT if p.get("net_price") else OrderType.MARKET,
                net_price=p.get("net_price"),
                quantity=quantity,
                time_in_force=p.get("time_in_force", "DAY"),
            )
        )

    def _build_iron_condor(self, instr: PlaybookInstruction, broker: Broker, p: dict, close: bool = False) -> TradeRequest:
        symbol = self._require(p, "symbol", instr.id)
        expiration = str(self._require(p, "expiration", instr.id))
        put_short = float(self._require(p, "put_short", instr.id))
        put_long = float(self._require(p, "put_long", instr.id))
        call_short = float(self._require(p, "call_short", instr.id))
        call_long = float(self._require(p, "call_long", instr.id))
        quantity = int(p.get("quantity", 1))

        if close:
            sell_action = OptionAction.BUY_TO_CLOSE
            buy_action = OptionAction.SELL_TO_CLOSE
        else:
            sell_action = OptionAction.SELL_TO_OPEN
            buy_action = OptionAction.BUY_TO_OPEN

        return TradeRequest(
            multi_leg_order=MultiLegOrder(
                broker=broker,
                legs=[
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=put_short,
                        option_type=OptionType.PUT,
                        action=sell_action,
                        quantity=1,
                    ),
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=put_long,
                        option_type=OptionType.PUT,
                        action=buy_action,
                        quantity=1,
                    ),
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=call_short,
                        option_type=OptionType.CALL,
                        action=sell_action,
                        quantity=1,
                    ),
                    OptionLeg(
                        symbol=symbol,
                        expiration=expiration,
                        strike=call_long,
                        option_type=OptionType.CALL,
                        action=buy_action,
                        quantity=1,
                    ),
                ],
                order_type=OrderType.LIMIT if p.get("net_price") else OrderType.MARKET,
                net_price=p.get("net_price"),
                quantity=quantity,
                time_in_force=p.get("time_in_force", "DAY"),
            )
        )
