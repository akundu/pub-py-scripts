"""Tests for UTP Voice — Natural Language Mobile Trading Interface."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure common/ package is importable (for market_hours, etc.)
_stocks_root = str(Path(__file__).resolve().parents[3])
if _stocks_root not in sys.path:
    sys.path.insert(0, _stocks_root)

import httpx
import pytest
from fastapi.testclient import TestClient

# Set required env vars before importing
os.environ.setdefault("UTP_VOICE_JWT_SECRET", "test-secret-key-for-tests")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-api-key")

import utp_voice


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_state(tmp_path):
    """Reset module-level state between tests."""
    # Reset daemon client singleton
    utp_voice._daemon_client = None
    # Reset pending confirmations
    utp_voice._pending_confirmations.clear()
    # Reset options caches
    utp_voice._options_cache.clear()
    utp_voice._expirations_cache.clear()
    utp_voice._prefetch_in_progress.clear()
    utp_voice._quote_cache.clear()
    utp_voice.PUBLIC_MODE = False  # Default: require auth
    # Override credentials file to tmp
    utp_voice.CREDENTIALS_FILE = str(tmp_path / "credentials.json")
    yield
    utp_voice._daemon_client = None
    utp_voice._pending_confirmations.clear()
    utp_voice._options_cache.clear()
    utp_voice._expirations_cache.clear()
    utp_voice._prefetch_in_progress.clear()
    utp_voice._quote_cache.clear()
    utp_voice.PUBLIC_MODE = False


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(utp_voice.app)


@pytest.fixture
def auth_cookie(tmp_path):
    """Create a test user and return an auth cookie dict."""
    utp_voice.CREDENTIALS_FILE = str(tmp_path / "credentials.json")
    utp_voice.add_user("testuser", "testpass123")
    token = utp_voice.create_token("testuser")
    return {utp_voice.COOKIE_NAME: token}


# ── Credential Management Tests ──────────────────────────────────────────────


class TestCredentials:
    def test_add_user_creates_file(self, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "creds.json")
        utp_voice.add_user("alice", "password123")
        creds = utp_voice._load_credentials()
        assert len(creds) == 1
        assert creds[0]["username"] == "alice"
        assert creds[0]["password_hash"].startswith("$2b$")

    def test_verify_user_correct_password(self, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "creds.json")
        utp_voice.add_user("alice", "password123")
        assert utp_voice.verify_user("alice", "password123") is True

    def test_verify_user_wrong_password(self, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "creds.json")
        utp_voice.add_user("alice", "password123")
        assert utp_voice.verify_user("alice", "wrongpass") is False

    def test_verify_user_nonexistent(self, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "creds.json")
        utp_voice.add_user("alice", "password123")
        assert utp_voice.verify_user("bob", "password123") is False

    def test_update_existing_user(self, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "creds.json")
        utp_voice.add_user("alice", "oldpass")
        utp_voice.add_user("alice", "newpass")
        creds = utp_voice._load_credentials()
        assert len(creds) == 1
        assert utp_voice.verify_user("alice", "newpass") is True
        assert utp_voice.verify_user("alice", "oldpass") is False

    def test_list_users(self, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "creds.json")
        utp_voice.add_user("alice", "pass1")
        utp_voice.add_user("bob", "pass2")
        users = utp_voice.list_users()
        assert set(users) == {"alice", "bob"}

    def test_load_empty_credentials(self, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "nonexistent.json")
        assert utp_voice._load_credentials() == []


# ── JWT Tests ─────────────────────────────────────────────────────────────────


class TestJWT:
    def test_create_and_decode_token(self):
        token = utp_voice.create_token("alice")
        assert utp_voice.decode_token(token) == "alice"

    def test_invalid_token_returns_none(self):
        assert utp_voice.decode_token("garbage.token.here") is None

    def test_expired_token_returns_none(self):
        # Create token with very short expiry
        original = utp_voice.JWT_EXPIRE_MINUTES
        utp_voice.JWT_EXPIRE_MINUTES = -1  # Already expired
        token = utp_voice.create_token("alice")
        utp_voice.JWT_EXPIRE_MINUTES = original
        assert utp_voice.decode_token(token) is None


# ── Auth API Tests ────────────────────────────────────────────────────────────


class TestAuthAPI:
    def test_login_success(self, client, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "creds.json")
        utp_voice.add_user("testuser", "testpass")
        resp = client.post("/api/login", json={"username": "testuser", "password": "testpass"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert utp_voice.COOKIE_NAME in resp.cookies

    def test_login_failure(self, client, tmp_path):
        utp_voice.CREDENTIALS_FILE = str(tmp_path / "creds.json")
        utp_voice.add_user("testuser", "testpass")
        resp = client.post("/api/login", json={"username": "testuser", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_missing_fields(self, client):
        resp = client.post("/api/login", json={"username": ""})
        assert resp.status_code == 400

    def test_logout(self, client, auth_cookie):
        resp = client.post("/api/logout", cookies=auth_cookie)
        assert resp.status_code == 200

    def test_protected_endpoint_no_auth(self, client):
        resp = client.post("/api/chat", json={"message": "test"})
        assert resp.status_code == 401

    def test_protected_endpoint_with_auth(self, client, auth_cookie):
        # Should reach the endpoint (may fail on anthropic call, but not on auth)
        with patch.object(utp_voice, "run_agent", new_callable=AsyncMock, return_value=[
            utp_voice.AgentResponse(type="text", content="hello")
        ]):
            resp = client.post(
                "/api/chat",
                json={"message": "test", "history": []},
                cookies=auth_cookie,
            )
            assert resp.status_code == 200


# ── Trade Payload Builder Tests ───────────────────────────────────────────────


class TestBuildTradePayload:
    def test_credit_spread(self):
        payload = utp_voice.build_trade_payload({
            "trade_type": "credit-spread",
            "symbol": "SPX",
            "option_type": "PUT",
            "expiration": "2026-04-02",
            "short_strike": 6400,
            "long_strike": 6380,
            "quantity": 25,
        })
        assert "multi_leg_order" in payload
        mlo = payload["multi_leg_order"]
        assert mlo["broker"] == "ibkr"
        assert len(mlo["legs"]) == 2
        assert mlo["legs"][0]["action"] == "SELL_TO_OPEN"
        assert mlo["legs"][0]["strike"] == 6400
        assert mlo["legs"][1]["action"] == "BUY_TO_OPEN"
        assert mlo["legs"][1]["strike"] == 6380
        assert mlo["order_type"] == "MARKET"
        assert mlo["quantity"] == 25

    def test_credit_spread_with_limit(self):
        payload = utp_voice.build_trade_payload({
            "trade_type": "credit-spread",
            "symbol": "RUT",
            "option_type": "PUT",
            "expiration": "2026-04-02",
            "short_strike": 2420,
            "long_strike": 2400,
            "quantity": 45,
            "net_price": 3.50,
        })
        mlo = payload["multi_leg_order"]
        assert mlo["order_type"] == "LIMIT"
        assert mlo["net_price"] == 3.50

    def test_equity_buy(self):
        payload = utp_voice.build_trade_payload({
            "trade_type": "equity",
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 100,
        })
        assert "equity_order" in payload
        eq = payload["equity_order"]
        assert eq["symbol"] == "SPY"
        assert eq["side"] == "BUY"
        assert eq["quantity"] == 100

    def test_iron_condor(self):
        payload = utp_voice.build_trade_payload({
            "trade_type": "iron-condor",
            "symbol": "SPX",
            "expiration": "2026-04-02",
            "put_short": 6300,
            "put_long": 6275,
            "call_short": 6600,
            "call_long": 6625,
            "quantity": 10,
        })
        mlo = payload["multi_leg_order"]
        assert len(mlo["legs"]) == 4
        put_legs = [l for l in mlo["legs"] if l["option_type"] == "PUT"]
        call_legs = [l for l in mlo["legs"] if l["option_type"] == "CALL"]
        assert len(put_legs) == 2
        assert len(call_legs) == 2

    def test_debit_spread(self):
        payload = utp_voice.build_trade_payload({
            "trade_type": "debit-spread",
            "symbol": "QQQ",
            "option_type": "CALL",
            "expiration": "2026-04-02",
            "long_strike": 480,
            "short_strike": 490,
            "quantity": 5,
            "net_price": 4.00,
        })
        mlo = payload["multi_leg_order"]
        assert mlo["legs"][0]["action"] == "BUY_TO_OPEN"
        assert mlo["legs"][1]["action"] == "SELL_TO_OPEN"

    def test_single_option(self):
        payload = utp_voice.build_trade_payload({
            "trade_type": "single-option",
            "symbol": "SPX",
            "option_type": "PUT",
            "expiration": "2026-04-02",
            "short_strike": 6300,
            "quantity": 1,
            "side": "BUY",
        })
        mlo = payload["multi_leg_order"]
        assert len(mlo["legs"]) == 1
        assert mlo["legs"][0]["action"] == "BUY_TO_OPEN"

    def test_unknown_trade_type_raises(self):
        with pytest.raises(ValueError, match="Unknown trade_type"):
            utp_voice.build_trade_payload({"trade_type": "magic", "symbol": "X"})


# ── Trade Description Tests ───────────────────────────────────────────────────


class TestDescribeTrade:
    def test_credit_spread_description(self):
        desc = utp_voice.describe_trade({
            "trade_type": "credit-spread",
            "symbol": "RUT",
            "option_type": "PUT",
            "short_strike": 2420,
            "long_strike": 2400,
            "quantity": 25,
            "expiration": "2026-04-02",
        })
        assert "CREDIT SPREAD" in desc
        assert "RUT" in desc
        assert "2420" in desc
        assert "2400" in desc
        assert "x25" in desc

    def test_equity_description(self):
        desc = utp_voice.describe_trade({
            "trade_type": "equity",
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 100,
        })
        assert "EQUITY" in desc
        assert "BUY" in desc
        assert "SPY" in desc

    def test_describe_write_action_close(self):
        desc = utp_voice.describe_write_action("close_position", {
            "position_id": "abc123",
            "net_price": 0.10,
        })
        assert "CLOSE" in desc
        assert "abc123" in desc
        assert "$0.10" in desc

    def test_describe_write_action_cancel(self):
        desc = utp_voice.describe_write_action("cancel_order", {"order_id": "ord456"})
        assert "CANCEL" in desc
        assert "ord456" in desc

    def test_describe_write_action_reconcile(self):
        desc = utp_voice.describe_write_action("reconcile_flush", {})
        assert "FLUSH" in desc


# ── Pending Confirmation Tests ────────────────────────────────────────────────


class TestPendingConfirmations:
    def test_store_and_retrieve(self):
        action = utp_voice.PendingAction(
            tool_name="execute_trade",
            tool_input={"trade_type": "credit-spread", "symbol": "SPX"},
            description="test trade",
        )
        cid = utp_voice.store_pending(action)
        retrieved = utp_voice.get_pending(cid)
        assert retrieved is not None
        assert retrieved.tool_name == "execute_trade"

    def test_remove_pending(self):
        action = utp_voice.PendingAction(
            tool_name="execute_trade",
            tool_input={},
            description="test",
        )
        cid = utp_voice.store_pending(action)
        utp_voice.remove_pending(cid)
        assert utp_voice.get_pending(cid) is None

    def test_expired_confirmation_not_found(self):
        action = utp_voice.PendingAction(
            tool_name="execute_trade",
            tool_input={},
            description="test",
            created_at=time.time() - 600,  # 10 minutes ago
        )
        utp_voice._pending_confirmations[action.confirmation_id] = action
        assert utp_voice.get_pending(action.confirmation_id) is None

    def test_nonexistent_confirmation(self):
        assert utp_voice.get_pending("nonexistent-id") is None


# ── Daemon Client Tests ───────────────────────────────────────────────────────


class TestDaemonClient:
    @pytest.mark.asyncio
    async def test_get_portfolio(self):
        client = utp_voice.UTPDaemonClient("http://fake:8000")
        mock_response = MagicMock()
        mock_response.json.return_value = {"positions": [], "balances": {"cash": 100000}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response) as mock_get:
            result = await client.get_portfolio()
            assert result == {"positions": [], "balances": {"cash": 100000}}
            mock_get.assert_called_once_with("/dashboard/portfolio", params=None)

    @pytest.mark.asyncio
    async def test_get_quote(self):
        client = utp_voice.UTPDaemonClient("http://fake:8000")
        mock_response = MagicMock()
        mock_response.json.return_value = {"symbol": "SPX", "last": 6500.0}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response) as mock_get:
            result = await client.get_quote("spx")
            assert result["symbol"] == "SPX"
            mock_get.assert_called_once_with("/market/quote/SPX", params=None)

    @pytest.mark.asyncio
    async def test_execute_trade(self):
        client = utp_voice.UTPDaemonClient("http://fake:8000")
        mock_response = MagicMock()
        mock_response.json.return_value = {"order_id": "123", "status": "SUBMITTED"}
        mock_response.raise_for_status = MagicMock()

        payload = {"multi_leg_order": {"broker": "ibkr", "legs": []}}
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await client.execute_trade(payload)
            assert result["order_id"] == "123"
            mock_post.assert_called_once_with("/trade/execute", json=payload)

    @pytest.mark.asyncio
    async def test_close_position(self):
        client = utp_voice.UTPDaemonClient("http://fake:8000")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await client.close_position("abc123", net_price=0.10)
            mock_post.assert_called_once_with(
                "/trade/close",
                json={"position_id": "abc123", "net_price": 0.10},
            )

    @pytest.mark.asyncio
    async def test_get_options_with_strike_range(self):
        client = utp_voice.UTPDaemonClient("http://fake:8000")
        quote_resp = MagicMock()
        quote_resp.json.return_value = {"last": 2500.0}
        quote_resp.raise_for_status = MagicMock()

        options_resp = MagicMock()
        options_resp.json.return_value = {"quotes": {"put": []}}
        options_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=[quote_resp, options_resp]):
            result = await client.get_options("RUT", option_type="PUT", strike_range_pct=5)
            assert "quotes" in result


# ── Agent Read Tool Tests ─────────────────────────────────────────────────────


class TestReadTools:
    @pytest.mark.asyncio
    async def test_execute_read_tool_portfolio(self):
        mock_client = AsyncMock()
        mock_client.get_portfolio.return_value = {"positions": []}

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            result = await utp_voice.execute_read_tool("get_portfolio", {})
            assert result == {"positions": []}

    @pytest.mark.asyncio
    async def test_execute_read_tool_quote(self):
        mock_client = AsyncMock()
        mock_client.get_quote.return_value = {"symbol": "SPX", "last": 6500}

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            result = await utp_voice.execute_read_tool("get_quote", {"symbol": "SPX"})
            assert result["symbol"] == "SPX"

    @pytest.mark.asyncio
    async def test_execute_read_tool_connection_error(self):
        mock_client = AsyncMock()
        mock_client.get_portfolio.side_effect = httpx.ConnectError("Connection refused")

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            result = await utp_voice.execute_read_tool("get_portfolio", {})
            assert "error" in result
            assert "Cannot connect" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_read_tool_unknown(self):
        with patch.object(utp_voice, "get_daemon_client", return_value=AsyncMock()):
            result = await utp_voice.execute_read_tool("unknown_tool", {})
            assert "error" in result


# ── Agent Write Tool Tests ────────────────────────────────────────────────────


class TestWriteTools:
    @pytest.mark.asyncio
    async def test_execute_write_trade(self):
        mock_client = AsyncMock()
        mock_client.execute_trade.return_value = {"order_id": "123", "status": "FILLED"}

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            result = await utp_voice.execute_write_tool("execute_trade", {
                "trade_type": "credit-spread",
                "symbol": "SPX",
                "option_type": "PUT",
                "expiration": "2026-04-02",
                "short_strike": 6400,
                "long_strike": 6380,
                "quantity": 25,
            })
            assert result["order_id"] == "123"
            # Verify the payload was built correctly
            call_args = mock_client.execute_trade.call_args
            payload = call_args[0][0]
            assert "multi_leg_order" in payload

    @pytest.mark.asyncio
    async def test_execute_write_close(self):
        mock_client = AsyncMock()
        mock_client.close_position.return_value = {"status": "ok"}

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            result = await utp_voice.execute_write_tool("close_position", {
                "position_id": "abc123",
            })
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_write_connection_error(self):
        mock_client = AsyncMock()
        mock_client.execute_trade.side_effect = httpx.ConnectError("refused")

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            result = await utp_voice.execute_write_tool("execute_trade", {
                "trade_type": "equity", "symbol": "SPY", "side": "BUY", "quantity": 1,
            })
            assert "error" in result


# ── Agent Loop Tests ──────────────────────────────────────────────────────────


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_agent_missing_api_key(self):
        original = utp_voice.ANTHROPIC_API_KEY
        utp_voice.ANTHROPIC_API_KEY = ""
        try:
            responses = await utp_voice.run_agent("test", [])
            assert len(responses) == 1
            assert responses[0].type == "error"
            assert "ANTHROPIC_API_KEY" in responses[0].content
        finally:
            utp_voice.ANTHROPIC_API_KEY = original

    @pytest.mark.asyncio
    async def test_agent_text_response(self):
        """Test that a simple text response from Claude is returned."""
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Here is your portfolio summary."

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            responses = await utp_voice.run_agent("show portfolio", [])
            assert len(responses) == 1
            assert responses[0].type == "text"
            assert "portfolio" in responses[0].content.lower()

    @pytest.mark.asyncio
    async def test_agent_read_tool_call(self):
        """Test that read tools are executed and results fed back."""
        # First response: Claude calls get_quote
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "get_quote"
        tool_block.input = {"symbol": "SPX"}
        tool_block.id = "tool_1"
        tool_block.model_dump.return_value = {
            "type": "tool_use", "id": "tool_1",
            "name": "get_quote", "input": {"symbol": "SPX"},
        }

        first_response = MagicMock()
        first_response.content = [tool_block]
        first_response.stop_reason = "tool_use"

        # Second response: Claude gives text answer
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "SPX is trading at 6500."

        second_response = MagicMock()
        second_response.content = [text_block]
        second_response.stop_reason = "end_turn"

        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(side_effect=[first_response, second_response])

        mock_daemon = AsyncMock()
        mock_daemon.get_quote.return_value = {"symbol": "SPX", "last": 6500.0}

        with patch("anthropic.AsyncAnthropic", return_value=mock_claude), \
             patch.object(utp_voice, "get_daemon_client", return_value=mock_daemon):
            responses = await utp_voice.run_agent("what's SPX at?", [])
            # Should have tool_result + text
            types = [r.type for r in responses]
            assert "tool_result" in types
            assert "text" in types

    @pytest.mark.asyncio
    async def test_agent_write_tool_returns_confirmation(self):
        """Test that write tools return pending confirmation."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "execute_trade"
        tool_block.input = {
            "trade_type": "credit-spread",
            "symbol": "RUT",
            "option_type": "PUT",
            "short_strike": 2420,
            "long_strike": 2400,
            "quantity": 25,
            "expiration": "2026-04-02",
        }
        tool_block.id = "tool_2"
        tool_block.model_dump.return_value = {
            "type": "tool_use", "id": "tool_2",
            "name": "execute_trade", "input": tool_block.input,
        }

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.stop_reason = "tool_use"

        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_claude):
            responses = await utp_voice.run_agent("sell 25 RUT puts 2420/2400", [])
            confirmations = [r for r in responses if r.type == "pending_confirmation"]
            assert len(confirmations) == 1
            assert confirmations[0].confirmation_id is not None
            assert "RUT" in confirmations[0].content
            # Verify it's stored
            assert utp_voice.get_pending(confirmations[0].confirmation_id) is not None


# ── API Endpoint Tests ────────────────────────────────────────────────────────


class TestChatAPI:
    def test_chat_requires_auth(self, client):
        resp = client.post("/api/chat", json={"message": "test"})
        assert resp.status_code == 401

    def test_chat_empty_message(self, client, auth_cookie):
        resp = client.post("/api/chat", json={"message": "", "history": []}, cookies=auth_cookie)
        assert resp.status_code == 400

    def test_chat_returns_responses(self, client, auth_cookie):
        with patch.object(utp_voice, "run_agent", new_callable=AsyncMock, return_value=[
            utp_voice.AgentResponse(type="text", content="Your portfolio is empty."),
        ]):
            resp = client.post(
                "/api/chat",
                json={"message": "show portfolio", "history": []},
                cookies=auth_cookie,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "responses" in data
            assert len(data["responses"]) == 1
            assert data["responses"][0]["type"] == "text"


class TestConfirmAPI:
    def test_confirm_requires_auth(self, client):
        resp = client.post("/api/confirm/some-id")
        assert resp.status_code == 401

    def test_confirm_not_found(self, client, auth_cookie):
        resp = client.post("/api/confirm/nonexistent", cookies=auth_cookie)
        assert resp.status_code == 404

    def test_confirm_success(self, client, auth_cookie):
        # Store a pending action
        action = utp_voice.PendingAction(
            tool_name="close_position",
            tool_input={"position_id": "abc123"},
            description="CLOSE abc123",
        )
        cid = utp_voice.store_pending(action)

        with patch.object(utp_voice, "execute_write_tool", new_callable=AsyncMock, return_value={
            "status": "ok", "message": "Position closed",
        }):
            resp = client.post(f"/api/confirm/{cid}", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "executed"
            # Should be removed after execution
            assert utp_voice.get_pending(cid) is None


class TestExecuteRawAPI:
    def test_raw_requires_auth(self, client):
        resp = client.post("/api/execute-raw", json={"endpoint": "/health"})
        assert resp.status_code == 401

    def test_raw_get(self, client, auth_cookie):
        mock_client = AsyncMock()
        mock_client._get = AsyncMock(return_value={"status": "ok"})

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            resp = client.post(
                "/api/execute-raw",
                json={"endpoint": "/health", "method": "GET"},
                cookies=auth_cookie,
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_raw_missing_endpoint(self, client, auth_cookie):
        resp = client.post("/api/execute-raw", json={}, cookies=auth_cookie)
        assert resp.status_code == 400


class TestHealthAPI:
    def test_health_no_auth_required(self, client):
        with patch.object(utp_voice, "get_daemon_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.health.return_value = {"status": "ok", "ibkr_connected": True}
            mock_get.return_value = mock_client
            resp = client.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_health_daemon_down(self, client):
        with patch.object(utp_voice, "get_daemon_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.health.side_effect = Exception("Connection refused")
            mock_get.return_value = mock_client
            resp = client.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "degraded"


# ── Page Serving Tests ────────────────────────────────────────────────────────


class TestPageServing:
    def test_index_redirects_without_auth_by_default(self, client):
        """In default mode, index redirects to login."""
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 302

    def test_index_serves_with_auth(self, client, auth_cookie):
        resp = client.get("/", cookies=auth_cookie)
        assert resp.status_code == 200
        assert "UTP Voice" in resp.text

    def test_login_page_serves(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200
        assert "UTP Voice" in resp.text


# ── Tool Classification Tests ─────────────────────────────────────────────────


class TestToolClassification:
    def test_write_tools_set(self):
        assert "execute_trade" in utp_voice.WRITE_TOOLS
        assert "close_position" in utp_voice.WRITE_TOOLS
        assert "cancel_order" in utp_voice.WRITE_TOOLS
        assert "reconcile_flush" in utp_voice.WRITE_TOOLS

    def test_read_tools_set(self):
        assert "get_portfolio" in utp_voice.READ_TOOLS
        assert "get_quote" in utp_voice.READ_TOOLS
        assert "get_options" in utp_voice.READ_TOOLS
        assert "get_trades" in utp_voice.READ_TOOLS
        assert "get_orders" in utp_voice.READ_TOOLS
        assert "get_performance" in utp_voice.READ_TOOLS

    def test_all_tools_classified(self):
        all_tool_names = {t["name"] for t in utp_voice.TOOLS}
        classified = utp_voice.WRITE_TOOLS | utp_voice.READ_TOOLS
        assert all_tool_names == classified, f"Unclassified tools: {all_tool_names - classified}"


# ── Breach Status Tests ───────────────────────────────────────────────────────


class TestBreachStatus:
    """Test _calc_breach_status in daemon's live_data_service."""

    def test_put_safe(self):
        from app.services.live_data_service import _calc_breach_status
        pos = {"legs": [{"strike": 2400, "option_type": "PUT", "action": "SELL_TO_OPEN"}]}
        result = _calc_breach_status(2500, pos)
        assert result is not None
        assert result["severity"] == "safe"
        assert not result["is_itm"]

    def test_put_breached(self):
        from app.services.live_data_service import _calc_breach_status
        pos = {"legs": [{"strike": 2400, "option_type": "PUT", "action": "SELL_TO_OPEN"}]}
        result = _calc_breach_status(2390, pos)
        assert result["severity"] == "breached"
        assert result["is_itm"]

    def test_call_warning(self):
        from app.services.live_data_service import _calc_breach_status
        pos = {"legs": [{"strike": 6500, "option_type": "CALL", "action": "SELL_TO_OPEN"}]}
        result = _calc_breach_status(6460, pos)
        assert result is not None
        assert result["severity"] in ("watch", "warning")

    def test_no_price(self):
        from app.services.live_data_service import _calc_breach_status
        pos = {"legs": [{"strike": 2400, "option_type": "PUT", "action": "SELL_TO_OPEN"}]}
        result = _calc_breach_status(0, pos)
        assert result is None

    def test_no_legs(self):
        from app.services.live_data_service import _calc_breach_status
        result = _calc_breach_status(2500, {"legs": []})
        assert result is None

    def test_single_leg_position(self):
        from app.services.live_data_service import _calc_breach_status
        pos = {"legs": [], "strike": 2400, "right": "P"}
        result = _calc_breach_status(2500, pos)
        assert result is not None
        assert result["severity"] == "safe"


# ── Pre-Built View API Tests ─────────────────────────────────────────────────


class TestPortfolioAPI:
    def test_portfolio_requires_auth(self, client):
        resp = client.get("/api/portfolio")
        assert resp.status_code == 401

    def test_portfolio_returns_daemon_data(self, client, auth_cookie):
        """Voice app now passes through daemon response (breach + metrics computed there)."""
        mock_client = AsyncMock()
        mock_client.get_portfolio.return_value = {
            "positions": [
                {"symbol": "SPX", "quantity": 25, "position_id": "abc123",
                 "avg_cost": 3.50, "market_value": 8750,
                 "current_price": 6500.0,
                 "breach_status": {"severity": "safe", "distance_pct": 1.5},
                 "spread_metrics": {"spread_width": 20, "derived_credit": 5000}}
            ],
            "balances": {"net_liquidation": 500000, "buying_power": 200000},
            "unrealized_pnl": 1500,
            "realized_pnl": 3000,
        }

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            resp = client.get("/api/portfolio", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            assert "positions" in data
            assert "balances" in data
            pos = data["positions"][0]
            assert pos["current_price"] == 6500.0
            assert pos["breach_status"]["severity"] == "safe"
            assert pos["spread_metrics"]["spread_width"] == 20

    def test_portfolio_voice_meta_ibkr(self, client, auth_cookie):
        """Portfolio response includes _voice_meta with source and fetched_at."""
        mock_client = AsyncMock()
        mock_client.get_portfolio.return_value = {
            "positions": [],
            "balances": {"net_liquidation": 500000},
        }
        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            resp = client.get("/api/portfolio", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            meta = data["_voice_meta"]
            assert meta["source"] == "ibkr_daemon"
            assert "fetched_at" in meta

    def test_portfolio_voice_meta_local(self, client, auth_cookie):
        """When no net_liquidation, source is 'local'."""
        mock_client = AsyncMock()
        mock_client.get_portfolio.return_value = {
            "positions": [],
            "balances": {},
        }
        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            resp = client.get("/api/portfolio", cookies=auth_cookie)
            data = resp.json()
            assert data["_voice_meta"]["source"] == "local"


class TestTradesListAPI:
    def test_trades_list_voice_meta(self, client, auth_cookie, tmp_path):
        """Trades list response includes _voice_meta with CSV mtime."""
        csv_path = tmp_path / "trades.csv"
        csv_path.write_text("timestamp,symbol\n2026-04-10,SPX\n")
        with patch.object(utp_voice, "TRADES_CSV_PATH", csv_path):
            resp = client.get("/api/trades/list", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            meta = data["_voice_meta"]
            assert meta["source"] == "trades_csv"
            assert meta["csv_modified_at"] is not None
            assert meta["fetched_at"] is not None

    def test_trades_list_no_csv(self, client, auth_cookie, tmp_path):
        """When CSV doesn't exist, csv_modified_at is null."""
        csv_path = tmp_path / "nonexistent.csv"
        with patch.object(utp_voice, "TRADES_CSV_PATH", csv_path):
            resp = client.get("/api/trades/list", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            meta = data["_voice_meta"]
            assert meta["csv_modified_at"] is None


class TestQuotesAPI:
    def test_quotes_requires_auth_by_default(self, client):
        """In default mode (PUBLIC_MODE=False), quotes require auth."""
        resp = client.get("/api/quotes")
        assert resp.status_code == 401

    def test_quotes_returns_multiple(self, client, auth_cookie):
        mock_client = AsyncMock()
        mock_client.get_quote.side_effect = [
            {"symbol": "SPX", "last": 6500},
            {"symbol": "NDX", "last": 21000},
        ]
        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            resp = client.get("/api/quotes?symbols=SPX,NDX", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            assert "SPX" in data
            assert "NDX" in data

    def test_quotes_uses_cache(self, client, auth_cookie):
        """Second call should serve from cache, not hit daemon again."""
        mock_client = AsyncMock()
        mock_client.get_quote.side_effect = [
            {"symbol": "SPX", "last": 6500},
        ]
        # Populate cache
        utp_voice._put_cached_quote("SPX", {"symbol": "SPX", "last": 6500})

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            resp = client.get("/api/quotes?symbols=SPX", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            assert data["SPX"]["last"] == 6500
            # Daemon should NOT have been called since cache was populated
            mock_client.get_quote.assert_not_called()

    def test_quotes_concurrent_fetch(self, client, auth_cookie):
        """Quotes for multiple symbols should be fetched concurrently."""
        mock_client = AsyncMock()
        mock_client.get_quote.side_effect = [
            {"symbol": "SPX", "last": 6500},
            {"symbol": "NDX", "last": 21000},
            {"symbol": "RUT", "last": 2500},
        ]
        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            resp = client.get("/api/quotes?symbols=SPX,NDX,RUT", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 3
            assert data["SPX"]["last"] == 6500
            assert data["RUT"]["last"] == 2500


class TestOptionsGridAPI:
    def test_options_grid_requires_auth_by_default(self, client):
        """In default mode, options grid requires auth."""
        resp = client.get("/api/options-grid")
        assert resp.status_code == 401

    def test_options_grid_returns_chain(self, client, auth_cookie):
        mock_client = AsyncMock()
        mock_client.get_quote.return_value = {"last": 2500.0}
        mock_client.get_options.side_effect = [
            {"expirations": ["2099-04-06"]},  # list_expirations call (IBKR)
            {"quotes": {"put": [{"strike": 2490, "bid": 3.0, "ask": 3.5}]}},  # PUT chain
            {"quotes": {"call": [{"strike": 2510, "bid": 2.5, "ask": 3.0}]}},  # CALL chain
        ]

        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client), \
             patch.object(utp_voice, "_get_csv_expirations", return_value=[]):
            resp = client.get("/api/options-grid?symbol=RUT&strike_range_pct=3", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            assert data["symbol"] == "RUT"
            assert data["current_price"] == 2500.0
            assert "chain" in data
            assert "source" in data


class TestQuickTradeAPI:
    def test_quick_trade_requires_auth(self, client):
        resp = client.post("/api/quick-trade", json={})
        assert resp.status_code == 401

    def test_quick_trade_creates_confirmation(self, client, auth_cookie):
        resp = client.post("/api/quick-trade", json={
            "trade_type": "credit-spread",
            "symbol": "SPX",
            "option_type": "PUT",
            "short_strike": 6400,
            "long_strike": 6380,
            "quantity": 25,
            "expiration": "2026-04-03",
        }, cookies=auth_cookie)
        assert resp.status_code == 200
        data = resp.json()
        assert "confirmation_id" in data
        assert "description" in data
        assert "SPX" in data["description"]
        # Verify the confirmation is stored
        assert utp_voice.get_pending(data["confirmation_id"]) is not None

    def test_quick_trade_missing_fields(self, client, auth_cookie):
        resp = client.post("/api/quick-trade", json={
            "trade_type": "credit-spread",
            "symbol": "SPX",
        }, cookies=auth_cookie)
        assert resp.status_code == 400

    def test_quick_trade_default_quantity(self, client, auth_cookie):
        resp = client.post("/api/quick-trade", json={
            "trade_type": "credit-spread",
            "symbol": "RUT",
            "option_type": "PUT",
            "short_strike": 2420,
            "long_strike": 2400,
            "expiration": "2026-04-03",
        }, cookies=auth_cookie)
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool_input"]["quantity"] == 1


class TestPerformanceAPI:
    def test_performance_requires_auth_by_default(self, client):
        """In default mode, performance requires auth."""
        resp = client.get("/api/performance-summary")
        assert resp.status_code == 401

    def test_performance_returns_metrics(self, client, auth_cookie):
        mock_client = AsyncMock()
        mock_client.get_performance.return_value = {
            "total_trades": 50, "win_rate": 0.92, "net_pnl": 15000,
        }
        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            resp = client.get("/api/performance-summary", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_trades"] == 50


# ── CSV Fallback Tests ────────────────────────────────────────────────────────


class TestCSVExpirations:
    def test_csv_expirations_from_directory(self, tmp_path):
        """Test reading expiration dates from CSV filenames."""
        csv_dir = tmp_path / "SPX"
        csv_dir.mkdir()
        # Create some CSV files
        (csv_dir / "2099-04-06.csv").write_text("header\n")
        (csv_dir / "2099-04-07.csv").write_text("header\n")
        (csv_dir / "2099-04-08.csv").write_text("header\n")
        (csv_dir / "2020-03-01.csv").write_text("header\n")  # Past date
        (csv_dir / "not-a-date.csv").write_text("header\n")  # Invalid

        original = utp_voice.CSV_EXPORTS_DIR
        utp_voice.CSV_EXPORTS_DIR = str(tmp_path)
        try:
            exps = utp_voice._get_csv_expirations("SPX")
            assert "2099-04-06" in exps
            assert "2099-04-07" in exps
            assert "2099-04-08" in exps
            assert "not-a-date" not in exps
            assert exps == sorted(exps)
        finally:
            utp_voice.CSV_EXPORTS_DIR = original

    def test_csv_expirations_missing_dir(self, tmp_path):
        original = utp_voice.CSV_EXPORTS_DIR
        utp_voice.CSV_EXPORTS_DIR = str(tmp_path)
        try:
            assert utp_voice._get_csv_expirations("NOSYMBOL") == []
        finally:
            utp_voice.CSV_EXPORTS_DIR = original


class TestCSVReader:
    def test_load_options_from_csv(self, tmp_path):
        """Test reading option quotes from a CSV file."""
        csv_dir = tmp_path / "SPX"
        csv_dir.mkdir()
        csv_file = csv_dir / "2026-04-06.csv"
        csv_file.write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,delta,gamma,theta,vega,implied_volatility,volume\n"
            "2026-04-03T07:50:00,O:SPX260406P06400,put,6400,2026-04-06,10.5,11.0,10.8,,0.35,,,,,100\n"
            "2026-04-03T07:50:00,O:SPX260406C06500,call,6500,2026-04-06,8.0,8.5,8.2,,,,,,,200\n"
            "2026-04-03T07:59:00,O:SPX260406P06400,put,6400,2026-04-06,10.8,11.2,10.9,,0.36,,,,,150\n"
            "2026-04-03T07:59:00,O:SPX260406C06500,call,6500,2026-04-06,7.8,8.3,8.0,,,,,,,250\n"
        )

        original = utp_voice.CSV_EXPORTS_DIR
        utp_voice.CSV_EXPORTS_DIR = str(tmp_path)
        try:
            quotes, ts = utp_voice._load_options_from_csv("SPX", "2026-04-06")
            assert len(quotes) == 2  # Latest timestamp only
            assert ts == "2026-04-03T07:59:00"
            # Check that we got the latest snapshot
            put_q = [q for q in quotes if q["_option_type"] == "put"][0]
            assert put_q["bid"] == 10.8
            assert put_q["strike"] == 6400
            assert put_q["greeks"]["delta"] == 0.36
        finally:
            utp_voice.CSV_EXPORTS_DIR = original

    def test_load_csv_with_strike_filter(self, tmp_path):
        csv_dir = tmp_path / "SPX"
        csv_dir.mkdir()
        csv_file = csv_dir / "2026-04-06.csv"
        csv_file.write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,delta,gamma,theta,vega,implied_volatility,volume\n"
            "2026-04-03T08:00:00,O:X,put,6000,2026-04-06,50,51,,,,,,,,,\n"
            "2026-04-03T08:00:00,O:X,put,6400,2026-04-06,10,11,,,,,,,,,\n"
            "2026-04-03T08:00:00,O:X,put,6800,2026-04-06,1,2,,,,,,,,,\n"
        )

        original = utp_voice.CSV_EXPORTS_DIR
        utp_voice.CSV_EXPORTS_DIR = str(tmp_path)
        try:
            quotes, _ = utp_voice._load_options_from_csv("SPX", "2026-04-06", 6300, 6500)
            assert len(quotes) == 1
            assert quotes[0]["strike"] == 6400
        finally:
            utp_voice.CSV_EXPORTS_DIR = original

    def test_load_csv_missing_file(self, tmp_path):
        original = utp_voice.CSV_EXPORTS_DIR
        utp_voice.CSV_EXPORTS_DIR = str(tmp_path)
        try:
            quotes, ts = utp_voice._load_options_from_csv("SPX", "2099-01-01")
            assert quotes == []
            assert ts == ""
        finally:
            utp_voice.CSV_EXPORTS_DIR = original


class TestSplitCSVQuotes:
    def test_split_by_type(self):
        quotes = [
            {"strike": 6400, "_option_type": "put", "bid": 10},
            {"strike": 6500, "_option_type": "call", "bid": 8},
            {"strike": 6350, "_option_type": "put", "bid": 5},
        ]
        result = utp_voice._split_csv_quotes_by_type(quotes)
        assert len(result["put"]) == 2
        assert len(result["call"]) == 1
        # _option_type should be removed
        assert "_option_type" not in result["put"][0]


class TestMergeExpirations:
    def test_merge_dedup_sort(self):
        csv = ["2099-04-06", "2099-04-07", "2099-04-08"]
        ibkr = ["2099-04-10", "2099-04-07", "2099-04-17"]
        result = utp_voice._merge_expirations(csv, ibkr)
        assert result == ["2099-04-06", "2099-04-07", "2099-04-08", "2099-04-10", "2099-04-17"]

    def test_merge_normalizes_yyyymmdd(self):
        csv = ["2099-04-06"]
        ibkr = ["20990410", "20990417"]  # IBKR format
        result = utp_voice._merge_expirations(csv, ibkr)
        assert result == ["2099-04-06", "2099-04-10", "2099-04-17"]

    def test_merge_dedup_across_formats(self):
        csv = ["2099-04-07"]
        ibkr = ["20990407"]  # Same date, different format
        result = utp_voice._merge_expirations(csv, ibkr)
        assert result == ["2099-04-07"]

    def test_merge_filters_past_dates(self):
        """Past expirations should be excluded."""
        past = ["2020-01-01", "2020-06-15"]
        future = ["2099-12-31"]
        result = utp_voice._merge_expirations(past, future)
        assert result == ["2099-12-31"]

    def test_merge_filters_past_ibkr_format(self):
        """IBKR YYYYMMDD past dates should also be excluded."""
        ibkr = ["20200101", "20990410"]
        result = utp_voice._merge_expirations(ibkr)
        assert result == ["2099-04-10"]

    def test_merge_empty_lists(self):
        assert utp_voice._merge_expirations([], []) == []
        assert utp_voice._merge_expirations(["2099-04-06"], []) == ["2099-04-06"]

    def test_merge_filters_non_trading_days(self):
        """Weekends and holidays should be excluded from expirations."""
        # 2026-04-11 is Saturday, 2026-04-12 is Sunday, 2026-04-13 is Monday
        exps = ["2026-04-11", "2026-04-12", "2026-04-13"]
        result = utp_voice._merge_expirations(exps)
        assert "2026-04-11" not in result  # Saturday
        assert "2026-04-12" not in result  # Sunday
        assert "2026-04-13" in result      # Monday (trading day)


class TestSpreadMetrics:
    """Test _compute_spread_metrics in daemon's live_data_service."""

    def test_computes_metrics_for_multi_leg(self):
        from app.services.live_data_service import _compute_spread_metrics
        positions = [{
            "order_type": "multi_leg",
            "quantity": 25,
            "legs": [
                {"strike": 6400, "option_type": "PUT", "action": "SELL"},
                {"strike": 6380, "option_type": "PUT", "action": "BUY"},
            ],
            "broker_unrealized_pnl": 2000,
            "market_value": -1000,
        }]
        _compute_spread_metrics(positions)
        sm = positions[0].get("spread_metrics")
        assert sm is not None
        assert sm["spread_width"] == 20
        assert sm["gross_risk"] == 50000  # 20 * 25 * 100
        assert sm["derived_credit"] == 3000  # 2000 + abs(-1000)
        assert sm["max_loss"] == 47000
        assert sm["roi_pct"] > 0

    def test_skips_non_multi_leg(self):
        from app.services.live_data_service import _compute_spread_metrics
        positions = [{"order_type": "equity", "quantity": 100}]
        _compute_spread_metrics(positions)
        assert "spread_metrics" not in positions[0]


class TestCacheSourcePriority:
    def test_ibkr_not_overwritten_by_csv(self):
        """IBKR data should not be overwritten by CSV data."""
        with patch.object(utp_voice, "_is_market_hours", return_value=False):
            utp_voice._put_cached_options("SPX", "2026-04-06", "PUT", [{"strike": 6400}], source="ibkr")
            utp_voice._put_cached_options("SPX", "2026-04-06", "PUT", [{"strike": 6400, "old": True}], source="csv_exports")
            entry = utp_voice._get_cached_options("SPX", "2026-04-06", "PUT")
            assert entry is not None
            assert entry.source == "ibkr"

    def test_csv_can_be_overwritten_by_ibkr(self):
        """CSV data should be overwritable by IBKR data."""
        with patch.object(utp_voice, "_is_market_hours", return_value=False):
            utp_voice._put_cached_options("SPX", "2026-04-06", "PUT", [{"old": True}], source="csv_exports")
            utp_voice._put_cached_options("SPX", "2026-04-06", "PUT", [{"new": True}], source="ibkr")
            entry = utp_voice._get_cached_options("SPX", "2026-04-06", "PUT")
            assert entry.source == "ibkr"
            assert entry.data == [{"new": True}]


# ── Server-Side Spread Computation Tests ──────────────────────────────────────


class TestComputeSpreadsServer:
    def test_basic_credit_spread(self):
        chain = {
            "put": [
                {"strike": 6400, "bid": 5.0, "ask": 5.5, "greeks": {"delta": -0.15}},
                {"strike": 6380, "bid": 2.5, "ask": 3.0, "greeks": {"delta": -0.08}},
            ],
            "call": [],
        }
        spreads = utp_voice.compute_spreads_server(chain, "SPX", 6500, 20)
        assert len(spreads) == 1
        s = spreads[0]
        assert s["option_type"] == "PUT"
        assert s["short_strike"] == 6400
        assert s["long_strike"] == 6380
        assert s["credit"] == 2.0  # 5.0 - 3.0
        assert s["credit_per_contract"] == 200
        assert s["max_loss"] == 1800  # 20*100 - 200
        assert s["short_delta"] == -0.15

    def test_filters_applied(self):
        chain = {
            "put": [
                {"strike": 6490, "bid": 10.0, "ask": 10.5},  # 0.15% OTM
                {"strike": 6470, "bid": 5.0, "ask": 5.5},
                {"strike": 6400, "bid": 3.0, "ask": 3.5},  # 1.54% OTM
                {"strike": 6380, "bid": 1.5, "ask": 2.0},
            ],
            "call": [],
        }
        # With min_otm 1%: should exclude 6490 (too close)
        spreads = utp_voice.compute_spreads_server(chain, "SPX", 6500, 20, {"min_otm_pct": 1.0})
        strikes = [s["short_strike"] for s in spreads]
        assert 6490 not in strikes
        assert 6400 in strikes

    def test_itm_filtered(self):
        chain = {
            "put": [
                {"strike": 6600, "bid": 50.0, "ask": 51.0},  # ITM put
                {"strike": 6580, "bid": 30.0, "ask": 31.0},
            ],
            "call": [],
        }
        spreads = utp_voice.compute_spreads_server(chain, "SPX", 6500, 20)
        assert len(spreads) == 0  # All ITM


class TestAutoTradeState:
    def test_save_and_load(self, tmp_path):
        utp_voice.AUTO_TRADE_STATE_PATH = tmp_path / "state.json"
        state = {"active": True, "trading_day": "2026-04-07", "executed_today": []}
        utp_voice._save_auto_trade_state(state)
        loaded = utp_voice._load_auto_trade_state()
        assert loaded["active"] is True
        assert loaded["trading_day"] == "2026-04-07"

    def test_load_missing(self, tmp_path):
        utp_voice.AUTO_TRADE_STATE_PATH = tmp_path / "nonexistent.json"
        loaded = utp_voice._load_auto_trade_state()
        assert loaded["active"] is False


class TestTradeCSVLogger:
    def test_log_creates_csv(self, tmp_path):
        utp_voice.TRADES_CSV_PATH = tmp_path / "trades.csv"
        utp_voice.log_trade_to_csv(
            {"trade_type": "credit-spread", "symbol": "SPX", "option_type": "PUT",
             "short_strike": 6400, "long_strike": 6380, "quantity": 25, "expiration": "2026-04-07"},
            {"order_id": "123", "status": "FILLED"},
            source="manual",
        )
        assert utp_voice.TRADES_CSV_PATH.exists()
        import csv
        with open(utp_voice.TRADES_CSV_PATH) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "SPX"
        assert rows[0]["source"] == "manual"
        assert rows[0]["order_id"] == "123"


class TestProfitTargets:
    """Test ProfitTargetService in the daemon."""

    def test_save_and_load(self, tmp_path):
        from app.services.profit_target_service import ProfitTargetService
        store = MagicMock()
        ledger = MagicMock()
        svc = ProfitTargetService(tmp_path, store, ledger)
        svc.set_target("pos123", 1.50, 50, "SPX", 6400, 6380, 25)
        targets = svc.get_targets()
        assert "pos123" in targets
        assert targets["pos123"]["profit_target_pct"] == 50
        assert targets["pos123"]["entry_credit"] == 1.50

    def test_remove_target(self, tmp_path):
        from app.services.profit_target_service import ProfitTargetService
        store = MagicMock()
        ledger = MagicMock()
        svc = ProfitTargetService(tmp_path, store, ledger)
        svc.set_target("pos123", 1.50, 50, "SPX")
        assert svc.remove_target("pos123")
        assert "pos123" not in svc.get_targets()

    def test_load_empty(self, tmp_path):
        from app.services.profit_target_service import ProfitTargetService
        store = MagicMock()
        ledger = MagicMock()
        svc = ProfitTargetService(tmp_path, store, ledger)
        assert svc.get_targets() == {}


class TestPositionLimits:
    @pytest.mark.asyncio
    async def test_check_limits_allowed(self):
        mock_client = AsyncMock()
        mock_client.get_portfolio.return_value = {"positions": [{"quantity": 25}, {"quantity": 25}]}
        utp_voice.MAX_OPEN_POSITIONS = 10
        utp_voice.MAX_DAILY_TRADES = 20
        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            result = await utp_voice._check_position_limits()
            assert result["allowed"] is True
            assert result["open_count"] == 2

    @pytest.mark.asyncio
    async def test_check_limits_exceeded(self):
        mock_client = AsyncMock()
        mock_client.get_portfolio.return_value = {"positions": [{"quantity": 1}] * 10}
        utp_voice.MAX_OPEN_POSITIONS = 10
        with patch.object(utp_voice, "get_daemon_client", return_value=mock_client):
            result = await utp_voice._check_position_limits()
            assert result["allowed"] is False
            assert "Max open positions" in result["reason"]


class TestPrevCloses:
    """Tests for /api/prev-closes response format and market hours logic."""

    def test_prev_closes_returns_both_closes(self, client, auth_cookie):
        """API returns last_close + prev_close with dates."""
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [
            {"date": "2026-04-10T00:00:00Z", "close": 6816.89},
            {"date": "2026-04-09T00:00:00Z", "close": 6824.66},
        ]}

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            with patch.object(utp_voice, "_is_market_hours", return_value=False):
                resp = client.get("/api/prev-closes?symbols=SPX", cookies=auth_cookie)
                assert resp.status_code == 200
                data = resp.json()
                assert "SPX" in data
                spx = data["SPX"]
                assert spx["last_close"] == 6816.89
                assert spx["prev_close"] == 6824.66
                assert spx["last_close_date"] == "2026-04-10"

    def test_prev_closes_includes_meta(self, client, auth_cookie):
        """Response includes _meta with is_trading_day and is_session_active."""
        utp_voice._prev_close_cache = {
            "SPX": {"last_close": 6816.89, "last_close_date": "2026-04-10",
                    "prev_close": 6824.66, "prev_close_date": "2026-04-09"},
        }
        utp_voice._prev_close_cache_at = time.time()
        with patch.object(utp_voice, "_is_market_hours", return_value=False), \
             patch.dict("sys.modules", {"common": MagicMock(), "common.market_hours": MagicMock(
                 is_trading_day=MagicMock(return_value=False),
                 is_trading_session_active=MagicMock(return_value=False),
             )}):
            resp = client.get("/api/prev-closes?symbols=SPX", cookies=auth_cookie)
            assert resp.status_code == 200
            data = resp.json()
            assert "_meta" in data
            assert "is_trading_day" in data["_meta"]


class TestMarketHoursHoliday:
    """Tests for holiday-aware market hours."""

    @classmethod
    def setup_class(cls):
        import sys
        stocks_root = str(Path(__file__).resolve().parents[3])  # stocks/
        if stocks_root not in sys.path:
            sys.path.insert(0, stocks_root)

    def test_is_market_hours_weekday(self):
        """Regular weekday during market hours returns True."""
        from common.market_hours import is_market_hours
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        # Wed Apr 8 2026, 10:30 AM ET
        dt = datetime(2026, 4, 8, 14, 30, tzinfo=timezone.utc)  # 10:30 AM ET
        assert is_market_hours(dt) is True

    def test_is_market_hours_weekend(self):
        """Saturday returns False."""
        from common.market_hours import is_market_hours
        from datetime import datetime, timezone
        # Sat Apr 11 2026
        dt = datetime(2026, 4, 11, 14, 30, tzinfo=timezone.utc)
        assert is_market_hours(dt) is False

    def test_is_trading_day_regular(self):
        """Regular weekday is a trading day."""
        from common.market_hours import is_trading_day
        from datetime import date
        assert is_trading_day(date(2026, 4, 10)) is True  # Friday

    def test_is_trading_day_weekend(self):
        """Saturday is not a trading day."""
        from common.market_hours import is_trading_day
        from datetime import date
        assert is_trading_day(date(2026, 4, 11)) is False  # Saturday

    def test_is_trading_day_good_friday(self):
        """Good Friday 2026 (Apr 3) is a market holiday."""
        from common.market_hours import is_trading_day
        from datetime import date
        assert is_trading_day(date(2026, 4, 3)) is False

    def test_previous_trading_day_monday(self):
        """Previous trading day before Monday is Friday."""
        from common.market_hours import previous_trading_day
        from datetime import date
        prev = previous_trading_day(date(2026, 4, 13))  # Monday
        assert prev == date(2026, 4, 10)  # Friday

    def test_previous_trading_day_after_holiday(self):
        """Previous trading day before day after Good Friday is Thursday."""
        from common.market_hours import previous_trading_day
        from datetime import date
        # Apr 4 is Saturday, Apr 3 is Good Friday (holiday)
        # Previous trading day before Apr 6 (Monday) should skip Sat+Sun+Good Friday = Thursday Apr 2
        prev = previous_trading_day(date(2026, 4, 6))
        assert prev == date(2026, 4, 2)

    def test_is_session_active_premarket(self):
        """Pre-market (7 AM ET) on a trading day is session active."""
        from common.market_hours import is_trading_session_active
        from datetime import datetime, timezone
        # Wed Apr 8 2026, 7:00 AM ET = 11:00 UTC
        dt = datetime(2026, 4, 8, 11, 0, tzinfo=timezone.utc)
        assert is_trading_session_active(dt) is True

    def test_is_session_active_weekend(self):
        """Saturday is not session active."""
        from common.market_hours import is_trading_session_active
        from datetime import datetime, timezone
        dt = datetime(2026, 4, 11, 14, 0, tzinfo=timezone.utc)
        assert is_trading_session_active(dt) is False


class TestPercentileTradingDays:
    """Tests for trading day calendar in percentiles response."""

    def test_percentiles_include_trading_days(self, client, auth_cookie):
        """Percentiles response includes _trading_days from exchange_calendars."""
        import sys
        stocks_root = str(Path(__file__).resolve().parents[3])
        if stocks_root not in sys.path:
            sys.path.insert(0, stocks_root)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"tickers": [{"ticker": "SPX", "metadata": {}, "windows": {}}]}

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            with patch.object(utp_voice, "_is_market_hours", return_value=False):
                resp = client.get("/api/percentiles", cookies=auth_cookie)
                assert resp.status_code == 200
                data = resp.json()
                assert "_trading_days" in data
                # Should have trading days (not weekends/holidays)
                for td in data["_trading_days"]:
                    from datetime import date as _d
                    d = _d.fromisoformat(td)
                    assert d.weekday() < 5  # No weekends

    def test_trading_days_no_weekends(self):
        """Trading day calendar from exchange_calendars excludes weekends."""
        import sys
        stocks_root = str(Path(__file__).resolve().parents[3])
        if stocks_root not in sys.path:
            sys.path.insert(0, stocks_root)
        from common.market_hours import is_trading_day
        from datetime import date, timedelta
        today = date.today()
        for i in range(30):
            d = today + timedelta(days=i)
            if is_trading_day(d):
                assert d.weekday() < 5, f"{d} is a weekend but marked as trading day"

    def test_good_friday_not_trading_day(self):
        """Good Friday 2026 (Apr 3) is not a trading day."""
        import sys
        stocks_root = str(Path(__file__).resolve().parents[3])
        if stocks_root not in sys.path:
            sys.path.insert(0, stocks_root)
        from common.market_hours import is_trading_day
        from datetime import date
        assert is_trading_day(date(2026, 4, 3)) is False
