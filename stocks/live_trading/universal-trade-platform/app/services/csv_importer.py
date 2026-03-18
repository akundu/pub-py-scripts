"""CSV transaction importer for Robinhood and E*TRADE exports."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.models import (
    Broker,
    CSVImportResult,
    LedgerEntry,
    LedgerEventType,
    PositionSource,
)
from app.services.ledger import TransactionLedger
from app.services.position_store import PlatformPositionStore

logger = logging.getLogger(__name__)


class CSVTransactionImporter:
    """Parses and imports CSV transaction exports from brokerages."""

    def __init__(
        self,
        position_store: PlatformPositionStore,
        ledger: TransactionLedger,
    ) -> None:
        self._store = position_store
        self._ledger = ledger

    async def import_file(
        self,
        file_path: Path,
        broker: Broker,
        format: str = "auto",
    ) -> CSVImportResult:
        """Import a CSV file. Auto-detects format or uses explicit format."""
        result = CSVImportResult(file_name=file_path.name, broker=broker)

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except OSError as e:
            result.errors.append(f"Failed to read file: {e}")
            return result

        return await self._import_content(content, broker, format, result)

    async def import_content(
        self,
        content: str,
        broker: Broker,
        filename: str = "upload.csv",
        format: str = "auto",
    ) -> CSVImportResult:
        """Import from CSV content string (for file uploads)."""
        result = CSVImportResult(file_name=filename, broker=broker)
        return await self._import_content(content, broker, format, result)

    async def _import_content(
        self,
        content: str,
        broker: Broker,
        format: str,
        result: CSVImportResult,
    ) -> CSVImportResult:
        """Core import logic."""
        if format == "auto":
            format = self._detect_format(content, broker)

        if broker == Broker.ROBINHOOD:
            raw = self.parse_robinhood_csv(content)
        elif broker == Broker.ETRADE:
            raw = self.parse_etrade_csv(content)
        else:
            result.errors.append(f"Unsupported broker for CSV import: {broker.value}")
            return result

        transactions = self._normalize_transactions(raw, broker)
        transactions = self._deduplicate(transactions)

        for txn in transactions:
            try:
                pos_id = self._store.add_position_from_csv(
                    broker=broker,
                    symbol=txn["symbol"],
                    side=txn["side"],
                    quantity=txn["quantity"],
                    price=txn["price"],
                    trade_date=txn["date"],
                    status=txn.get("status", "closed"),
                )

                await self._ledger.append(LedgerEntry(
                    event_type=LedgerEventType.CSV_IMPORTED,
                    broker=broker,
                    position_id=pos_id,
                    source=PositionSource.CSV_IMPORT,
                    data={
                        "symbol": txn["symbol"],
                        "side": txn["side"],
                        "quantity": txn["quantity"],
                        "price": txn["price"],
                        "original_description": txn.get("description", ""),
                    },
                ))
                result.records_imported += 1
            except Exception as e:
                result.errors.append(f"Row error: {e}")
                result.records_skipped += 1

        return result

    def parse_robinhood_csv(self, content: str) -> list[dict]:
        """Parse Robinhood transaction history CSV.

        Expected columns: Activity Date, Process Date, Settle Date,
        Instrument, Description, Trans Code, Quantity, Price, Amount
        """
        reader = csv.DictReader(io.StringIO(content))
        results = []

        for row in reader:
            try:
                activity_date = row.get("Activity Date", "").strip()
                instrument = row.get("Instrument", "").strip()
                description = row.get("Description", "").strip()
                trans_code = row.get("Trans Code", "").strip()
                quantity_str = row.get("Quantity", "0").strip()
                price_str = row.get("Price", "0").strip().replace("$", "").replace(",", "")
                amount_str = row.get("Amount", "0").strip().replace("$", "").replace(",", "")

                if not instrument or not trans_code:
                    continue

                quantity = abs(float(quantity_str)) if quantity_str else 0
                price = abs(float(price_str)) if price_str else 0
                amount = float(amount_str) if amount_str else 0

                # Parse date
                trade_date = None
                for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
                    try:
                        trade_date = datetime.strptime(activity_date, fmt)
                        break
                    except ValueError:
                        continue

                if not trade_date:
                    continue

                results.append({
                    "date": trade_date,
                    "symbol": instrument,
                    "description": description,
                    "trans_code": trans_code,
                    "quantity": quantity,
                    "price": price,
                    "amount": amount,
                })
            except (ValueError, KeyError):
                continue

        return results

    def parse_etrade_csv(self, content: str) -> list[dict]:
        """Parse E*TRADE transaction history CSV.

        Expected columns: TransactionDate, TransactionType, SecurityType,
        Symbol, Quantity, Price, Commission, Amount
        """
        reader = csv.DictReader(io.StringIO(content))
        results = []

        for row in reader:
            try:
                txn_date_str = row.get("TransactionDate", "").strip()
                txn_type = row.get("TransactionType", "").strip()
                symbol = row.get("Symbol", "").strip()
                quantity_str = row.get("Quantity", "0").strip()
                price_str = row.get("Price", "0").strip().replace("$", "").replace(",", "")
                amount_str = row.get("Amount", "0").strip().replace("$", "").replace(",", "")

                if not symbol or not txn_type:
                    continue

                quantity = abs(float(quantity_str)) if quantity_str else 0
                price = abs(float(price_str)) if price_str else 0
                amount = float(amount_str) if amount_str else 0

                trade_date = None
                for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
                    try:
                        trade_date = datetime.strptime(txn_date_str, fmt)
                        break
                    except ValueError:
                        continue

                if not trade_date:
                    continue

                results.append({
                    "date": trade_date,
                    "symbol": symbol,
                    "description": txn_type,
                    "trans_code": txn_type,
                    "quantity": quantity,
                    "price": price,
                    "amount": amount,
                })
            except (ValueError, KeyError):
                continue

        return results

    def _normalize_transactions(self, raw: list[dict], broker: Broker) -> list[dict]:
        """Convert parsed rows to a unified schema."""
        normalized = []
        buy_codes = {"Buy", "Bought", "BUY", "BTC"}
        sell_codes = {"Sell", "Sold", "SELL", "STO", "STC"}

        for row in raw:
            code = row.get("trans_code", "")
            if code in buy_codes:
                side = "BUY"
            elif code in sell_codes:
                side = "SELL"
            else:
                continue  # skip dividends, interest, etc.

            normalized.append({
                "symbol": row["symbol"],
                "side": side,
                "quantity": row["quantity"],
                "price": row["price"],
                "date": row["date"],
                "description": row.get("description", ""),
                "status": "closed",  # imported transactions are historical
            })

        return normalized

    def _deduplicate(self, transactions: list[dict]) -> list[dict]:
        """Skip transactions that would duplicate existing ledger entries."""
        seen = set()
        unique = []
        for txn in transactions:
            key = (
                txn["symbol"],
                txn["side"],
                str(txn["quantity"]),
                str(txn["price"]),
                str(txn["date"]),
            )
            if key not in seen:
                seen.add(key)
                unique.append(txn)
        return unique

    def _detect_format(self, content: str, broker: Broker) -> str:
        """Auto-detect CSV format based on headers."""
        first_line = content.split("\n")[0] if content else ""
        if "Activity Date" in first_line:
            return "robinhood"
        if "TransactionDate" in first_line:
            return "etrade"
        return broker.value

    def preview(self, content: str, broker: Broker, max_rows: int = 10) -> list[dict]:
        """Parse and return first N rows without importing."""
        if broker == Broker.ROBINHOOD:
            raw = self.parse_robinhood_csv(content)
        elif broker == Broker.ETRADE:
            raw = self.parse_etrade_csv(content)
        else:
            return []

        normalized = self._normalize_transactions(raw, broker)
        preview_rows = []
        for txn in normalized[:max_rows]:
            preview_rows.append({
                "symbol": txn["symbol"],
                "side": txn["side"],
                "quantity": txn["quantity"],
                "price": txn["price"],
                "date": txn["date"].isoformat() if hasattr(txn["date"], "isoformat") else str(txn["date"]),
            })
        return preview_rows

    @staticmethod
    def supported_formats() -> dict:
        """Return supported CSV formats and their expected columns."""
        return {
            "robinhood": {
                "columns": [
                    "Activity Date", "Process Date", "Settle Date",
                    "Instrument", "Description", "Trans Code",
                    "Quantity", "Price", "Amount",
                ],
                "example_row": "03/15/2026,,03/17/2026,AAPL,Buy,Buy,10,$175.50,$1755.00",
            },
            "etrade": {
                "columns": [
                    "TransactionDate", "TransactionType", "SecurityType",
                    "Symbol", "Quantity", "Price", "Commission", "Amount",
                ],
                "example_row": "03/15/2026,Bought,Equity,AAPL,10,$175.50,$0.00,$1755.00",
            },
        }
