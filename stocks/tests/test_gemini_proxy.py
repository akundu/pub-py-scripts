"""Tests for common/gemini_proxy.py — proxy with per-client topic gating."""

import json
import os
import time
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from common import gemini_proxy
from common.gemini_proxy import (
    ClientConfig,
    DailyBudget,
    GeminiResult,
    RateLimiter,
    Registry,
    REFUSAL_TEXT,
    build_system_instruction,
    extract_client_key,
    handle_gemini_ask,
    handle_gemini_ping,
    is_lan_admin,
    is_refusal,
    reset_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_KEY = "k" * 48  # >= MIN_CLIENT_KEY_LEN
ANOTHER_KEY = "z" * 48


def _registry_payload(**overrides):
    """One-client registry payload for tests."""
    base = {
        "client_key": VALID_KEY,
        "name": "guitar-app",
        "topic_label": "guitar and music",
        "system_prompt": "You are a friendly guitar tutor.",
        "model": "gemini-flash-latest",
        "strict_mode": False,
        "rate_limit_per_min": 60,
        "daily_token_budget": 200_000,
        "max_prompt_chars": 4000,
        "enabled": True,
    }
    base.update(overrides)
    return {"clients": [base]}


def _write_registry(tmp_path, payload):
    p = tmp_path / "clients.json"
    p.write_text(json.dumps(payload))
    return p


def _make_request(
    *,
    ip: str = "203.0.113.5",  # public, not LAN
    xff: str = "",
    headers=None,
    body=None,
    method: str = "POST",
    app: web.Application | None = None,
):
    transport = MagicMock()
    transport.get_extra_info.return_value = (ip, 12345)
    req = MagicMock(spec=web.Request)
    req.transport = transport
    h = dict(headers or {})
    if xff:
        h["X-Forwarded-For"] = xff
    req.headers = h
    req.method = method
    if body is not None:
        async def _json():
            return body
        req.json = _json
    else:
        async def _json_err():
            raise ValueError("no body")
        req.json = _json_err
    req.app = app if app is not None else web.Application()
    return req


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Each test gets a clean env, fresh registry/budget/audit paths."""
    # Clear env vars that influence the proxy
    for k in [
        "GEMINI_API_KEY",
        "GEMINI_PROXY_REGISTRY",
        "GEMINI_PROXY_ADMIN_KEY",
        "GEMINI_PROXY_DISABLED",
        "GEMINI_PROXY_BUDGET_PATH",
        "GEMINI_PROXY_AUDIT_PATH",
    ]:
        monkeypatch.delenv(k, raising=False)
    # Default: a working API key + isolated paths
    monkeypatch.setenv("GEMINI_API_KEY", "test-upstream-key")
    monkeypatch.setenv("GEMINI_PROXY_BUDGET_PATH", str(tmp_path / "budget.json"))
    monkeypatch.setenv("GEMINI_PROXY_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    yield


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_load_valid(self, tmp_path):
        p = _write_registry(tmp_path, _registry_payload())
        reg = Registry(p)
        reg.load()
        cfg = reg.lookup(VALID_KEY)
        assert cfg is not None
        assert cfg.name == "guitar-app"
        assert cfg.topic_label == "guitar and music"
        assert cfg.enabled is True

    def test_lookup_unknown_returns_none(self, tmp_path):
        p = _write_registry(tmp_path, _registry_payload())
        reg = Registry(p)
        reg.load()
        assert reg.lookup("nope" * 20) is None
        assert reg.lookup("") is None
        assert reg.lookup(None) is None  # type: ignore[arg-type]

    def test_lookup_constant_time_path(self, tmp_path):
        # Just exercise both code paths (match + no-match) — value comparison
        # uses hmac.compare_digest internally.
        p = _write_registry(tmp_path, _registry_payload())
        reg = Registry(p)
        reg.load()
        assert reg.lookup(VALID_KEY).name == "guitar-app"
        assert reg.lookup(VALID_KEY[:-1] + "x") is None

    def test_missing_required_field_rejected(self, tmp_path):
        bad = _registry_payload()
        del bad["clients"][0]["topic_label"]
        p = _write_registry(tmp_path, bad)
        reg = Registry(p)
        with pytest.raises(ValueError, match="missing required fields"):
            reg.load()

    def test_short_key_rejected(self, tmp_path):
        bad = _registry_payload(client_key="short")
        p = _write_registry(tmp_path, bad)
        reg = Registry(p)
        with pytest.raises(ValueError, match="too short"):
            reg.load()

    def test_duplicate_key_rejected(self, tmp_path):
        payload = _registry_payload()
        payload["clients"].append(dict(payload["clients"][0]))  # same key
        p = _write_registry(tmp_path, payload)
        reg = Registry(p)
        with pytest.raises(ValueError, match="duplicate client_key"):
            reg.load()

    def test_missing_file_logs_and_empties(self, tmp_path):
        reg = Registry(tmp_path / "nonexistent.json")
        reg.load()
        assert reg.lookup(VALID_KEY) is None
        assert reg.all_clients() == []

    def test_hot_reload_on_mtime_change(self, tmp_path):
        p = _write_registry(tmp_path, _registry_payload())
        reg = Registry(p)
        reg.load()
        assert reg.lookup(VALID_KEY).name == "guitar-app"

        # Replace contents with a different client (different name + key)
        new_payload = _registry_payload(client_key=ANOTHER_KEY, name="cooking-app", topic_label="cooking")
        # Force a different mtime even on fast filesystems
        time.sleep(0.01)
        p.write_text(json.dumps(new_payload))
        os.utime(p, None)

        reg.maybe_reload()
        assert reg.lookup(VALID_KEY) is None
        assert reg.lookup(ANOTHER_KEY).name == "cooking-app"

    def test_unknown_field_ignored(self, tmp_path):
        # Forward-compat: unknown keys in JSON shouldn't crash the parser.
        payload = _registry_payload(extra_future_field="ignore me")
        p = _write_registry(tmp_path, payload)
        reg = Registry(p)
        reg.load()
        assert reg.lookup(VALID_KEY).name == "guitar-app"


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_up_to_capacity(self):
        rl = RateLimiter()
        # capacity = 3, all consumed within the same instant
        for i in range(3):
            assert rl.allow("c", 3, now=100.0) is True, f"call {i}"
        assert rl.allow("c", 3, now=100.0) is False

    def test_refills_over_time(self):
        rl = RateLimiter()
        # Drain capacity
        for _ in range(60):
            assert rl.allow("c", 60, now=100.0) is True
        assert rl.allow("c", 60, now=100.0) is False
        # 30 seconds later, half a minute's worth refilled (=30 tokens)
        assert rl.allow("c", 60, now=130.0) is True

    def test_independent_keys(self):
        rl = RateLimiter()
        for _ in range(3):
            assert rl.allow("a", 3, now=0.0) is True
        # Key "a" exhausted, but "b" still has full capacity
        assert rl.allow("a", 3, now=0.0) is False
        assert rl.allow("b", 3, now=0.0) is True

    def test_zero_per_min_means_no_limit(self):
        rl = RateLimiter()
        for _ in range(1000):
            assert rl.allow("c", 0, now=0.0) is True


# ---------------------------------------------------------------------------
# DailyBudget
# ---------------------------------------------------------------------------

class TestDailyBudget:
    def test_remaining_budget_decreases_with_consumption(self, tmp_path):
        b = DailyBudget(tmp_path / "budget.json")
        assert b.remaining("guitar-app", 1000) == 1000
        b.consume("guitar-app", 250)
        assert b.remaining("guitar-app", 1000) == 750
        assert b.has_budget("guitar-app", 1000) is True

    def test_exhausted_returns_zero(self, tmp_path):
        b = DailyBudget(tmp_path / "budget.json")
        b.consume("guitar-app", 1500)
        assert b.remaining("guitar-app", 1000) == 0
        assert b.has_budget("guitar-app", 1000) is False

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "budget.json"
        b1 = DailyBudget(path)
        b1.consume("c", 333)
        b2 = DailyBudget(path)
        assert b2.remaining("c", 1000) == 667

    def test_resets_when_date_changes(self, tmp_path):
        path = tmp_path / "budget.json"
        b = DailyBudget(path)
        # Manually plant yesterday's usage
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        b._state["c"] = {"date": yesterday, "tokens": 999_999}
        b._save()

        # New instance reads file but should ignore yesterday's bucket today
        b2 = DailyBudget(path)
        assert b2.remaining("c", 1000) == 1000

    def test_zero_budget(self, tmp_path):
        b = DailyBudget(tmp_path / "budget.json")
        assert b.remaining("c", 0) == 0
        assert b.has_budget("c", 0) is False


# ---------------------------------------------------------------------------
# build_system_instruction & is_refusal
# ---------------------------------------------------------------------------

class TestSystemInstruction:
    def test_includes_user_prompt_and_topic_and_refusal(self):
        cfg = ClientConfig(
            client_key=VALID_KEY,
            name="guitar-app",
            topic_label="guitar and music",
            system_prompt="You are a friendly guitar tutor.",
        )
        sys_inst = build_system_instruction(cfg)
        assert "You are a friendly guitar tutor." in sys_inst
        assert "guitar and music" in sys_inst
        assert REFUSAL_TEXT in sys_inst
        assert "off-topic" in sys_inst

    def test_is_refusal_matches_canonical(self):
        assert is_refusal(REFUSAL_TEXT) is True
        assert is_refusal(REFUSAL_TEXT + ".") is True  # trailing punctuation tolerated
        assert is_refusal("  i can't talk about that  ") is True
        assert is_refusal("I can answer that!") is False
        assert is_refusal("") is False


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

class TestExtractClientKey:
    def test_bearer_header(self):
        req = _make_request(headers={"Authorization": f"Bearer {VALID_KEY}"})
        assert extract_client_key(req) == VALID_KEY

    def test_x_client_key_fallback(self):
        req = _make_request(headers={"X-Client-Key": VALID_KEY})
        assert extract_client_key(req) == VALID_KEY

    def test_missing(self):
        req = _make_request(headers={})
        assert extract_client_key(req) is None

    def test_bearer_takes_precedence(self):
        req = _make_request(
            headers={"Authorization": f"Bearer {VALID_KEY}", "X-Client-Key": ANOTHER_KEY}
        )
        assert extract_client_key(req) == VALID_KEY


class TestIsLanAdmin:
    def test_private_ip_with_correct_admin_key(self):
        req = _make_request(ip="192.168.1.5", headers={"X-Admin-Key": "secret"})
        assert is_lan_admin(req, "secret") is True

    def test_private_ip_without_key(self):
        req = _make_request(ip="192.168.1.5", headers={})
        assert is_lan_admin(req, "secret") is False

    def test_public_ip_with_correct_key(self):
        req = _make_request(ip="8.8.8.8", headers={"X-Admin-Key": "secret"})
        assert is_lan_admin(req, "secret") is False

    def test_no_admin_key_configured(self):
        req = _make_request(ip="192.168.1.5", headers={"X-Admin-Key": "anything"})
        assert is_lan_admin(req, None) is False

    def test_wrong_admin_key(self):
        req = _make_request(ip="192.168.1.5", headers={"X-Admin-Key": "wrong"})
        assert is_lan_admin(req, "secret") is False


# ---------------------------------------------------------------------------
# handle_gemini_ask — auth & gating paths
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_registry(tmp_path, monkeypatch):
    """Provide a configured app + registry path that handlers will use."""
    p = _write_registry(tmp_path, _registry_payload())
    monkeypatch.setenv("GEMINI_PROXY_REGISTRY", str(p))
    app = web.Application()
    yield app, tmp_path
    reset_state(app)


def _make_post(body, headers=None, ip="203.0.113.5", app=None):
    return _make_request(method="POST", body=body, headers=headers, ip=ip, app=app)


def _resp_json(resp: web.Response):
    return json.loads(resp.body.decode())


@pytest.mark.asyncio
class TestHandleAskAuth:
    async def test_missing_key_returns_401(self, app_with_registry):
        app, _ = app_with_registry
        req = _make_post({"prompt": "hi"}, headers={}, app=app)
        with patch("common.gemini_proxy.call_gemini", new=AsyncMock()) as mock_call:
            resp = await handle_gemini_ask(req)
        assert resp.status == 401
        assert "missing client key" in _resp_json(resp)["error"]
        mock_call.assert_not_called()

    async def test_unknown_key_returns_401(self, app_with_registry):
        app, _ = app_with_registry
        req = _make_post(
            {"prompt": "hi"},
            headers={"Authorization": f"Bearer {ANOTHER_KEY}"},
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=AsyncMock()) as mock_call:
            resp = await handle_gemini_ask(req)
        assert resp.status == 401
        mock_call.assert_not_called()

    async def test_disabled_client_returns_403(self, tmp_path, monkeypatch):
        p = _write_registry(tmp_path, _registry_payload(enabled=False))
        monkeypatch.setenv("GEMINI_PROXY_REGISTRY", str(p))
        app = web.Application()
        req = _make_post(
            {"prompt": "hi"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=AsyncMock()) as mock_call:
            resp = await handle_gemini_ask(req)
        assert resp.status == 403
        mock_call.assert_not_called()

    async def test_missing_gemini_api_key_returns_503(self, app_with_registry, monkeypatch):
        app, _ = app_with_registry
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        req = _make_post(
            {"prompt": "hi"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        resp = await handle_gemini_ask(req)
        assert resp.status == 503

    async def test_disabled_endpoint_returns_503(self, app_with_registry, monkeypatch):
        app, _ = app_with_registry
        monkeypatch.setenv("GEMINI_PROXY_DISABLED", "1")
        req = _make_post(
            {"prompt": "hi"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        resp = await handle_gemini_ask(req)
        assert resp.status == 503

    async def test_invalid_json_body_returns_400(self, app_with_registry):
        app, _ = app_with_registry
        # _make_request with body=None → request.json() raises
        req = _make_request(
            method="POST",
            body=None,
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        resp = await handle_gemini_ask(req)
        assert resp.status == 400

    async def test_empty_prompt_returns_400(self, app_with_registry):
        app, _ = app_with_registry
        req = _make_post(
            {"prompt": "   "},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        resp = await handle_gemini_ask(req)
        assert resp.status == 400

    async def test_prompt_too_long_returns_413(self, tmp_path, monkeypatch):
        p = _write_registry(tmp_path, _registry_payload(max_prompt_chars=10))
        monkeypatch.setenv("GEMINI_PROXY_REGISTRY", str(p))
        app = web.Application()
        req = _make_post(
            {"prompt": "x" * 100},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        resp = await handle_gemini_ask(req)
        assert resp.status == 413


@pytest.mark.asyncio
class TestHandleAskHappyPath:
    async def test_on_topic_returns_answer_and_consumes_budget(self, app_with_registry):
        app, _ = app_with_registry
        fake = AsyncMock(
            return_value=GeminiResult(
                text="A Telecaster has a bright single-coil tone...",
                finish_reason="STOP",
                input_tokens=12,
                output_tokens=80,
            )
        )
        req = _make_post(
            {"prompt": "What's a Telecaster?"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=fake):
            resp = await handle_gemini_ask(req)
        assert resp.status == 200
        body = _resp_json(resp)
        assert body["client"] == "guitar-app"
        assert body["on_topic"] is True
        assert body["finish_reason"] == "STOP"
        assert body["tokens"] == {"input": 12, "output": 80}
        # call_gemini should have received the system instruction
        kwargs = fake.call_args.kwargs
        assert kwargs["model"] == "gemini-flash-latest"
        assert "guitar and music" in kwargs["system_instruction"]
        assert REFUSAL_TEXT in kwargs["system_instruction"]
        # Budget consumed
        state = gemini_proxy._get_state(app)
        budget: DailyBudget = state["budget"]
        assert budget.remaining("guitar-app", 200_000) == 200_000 - (12 + 80)

    async def test_off_topic_default_mode_returns_refusal_text(self, app_with_registry):
        # Default mode: Gemini's system_instruction makes it return the refusal.
        app, _ = app_with_registry
        fake = AsyncMock(
            return_value=GeminiResult(
                text=REFUSAL_TEXT,
                finish_reason="STOP",
                input_tokens=8,
                output_tokens=6,
            )
        )
        req = _make_post(
            {"prompt": "What's the weather in Tokyo?"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=fake):
            resp = await handle_gemini_ask(req)
        assert resp.status == 200
        body = _resp_json(resp)
        assert body["answer"] == REFUSAL_TEXT
        assert body["on_topic"] is False

    async def test_strict_mode_short_circuits_off_topic(self, tmp_path, monkeypatch):
        p = _write_registry(tmp_path, _registry_payload(strict_mode=True))
        monkeypatch.setenv("GEMINI_PROXY_REGISTRY", str(p))
        app = web.Application()

        # Classifier says NO, main call must NOT be invoked.
        fake_classifier = AsyncMock(return_value=False)
        fake_main = AsyncMock(side_effect=AssertionError("main call should not run"))

        req = _make_post(
            {"prompt": "Tell me about the stock market."},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        with patch("common.gemini_proxy.classify_on_topic", new=fake_classifier), \
             patch("common.gemini_proxy.call_gemini", new=fake_main):
            resp = await handle_gemini_ask(req)
        assert resp.status == 200
        body = _resp_json(resp)
        assert body["answer"] == REFUSAL_TEXT
        assert body["on_topic"] is False
        assert body["finish_reason"] == "BLOCKED_OFF_TOPIC"
        assert body["tokens"] == {"input": 0, "output": 0}
        fake_main.assert_not_called()

    async def test_strict_mode_passes_through_when_classifier_yes(self, tmp_path, monkeypatch):
        p = _write_registry(tmp_path, _registry_payload(strict_mode=True))
        monkeypatch.setenv("GEMINI_PROXY_REGISTRY", str(p))
        app = web.Application()

        fake_classifier = AsyncMock(return_value=True)
        fake_main = AsyncMock(
            return_value=GeminiResult(text="Drop-D is...", finish_reason="STOP",
                                      input_tokens=10, output_tokens=20)
        )
        req = _make_post(
            {"prompt": "What is drop-D tuning?"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        with patch("common.gemini_proxy.classify_on_topic", new=fake_classifier), \
             patch("common.gemini_proxy.call_gemini", new=fake_main):
            resp = await handle_gemini_ask(req)
        assert resp.status == 200
        body = _resp_json(resp)
        assert body["on_topic"] is True
        assert body["answer"] == "Drop-D is..."

    async def test_upstream_error_returns_502(self, app_with_registry):
        app, _ = app_with_registry
        fake = AsyncMock(side_effect=RuntimeError("gemini exploded"))
        req = _make_post(
            {"prompt": "anything"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=fake):
            resp = await handle_gemini_ask(req)
        assert resp.status == 502


@pytest.mark.asyncio
class TestRateAndBudget:
    async def test_rate_limit_exceeded_returns_429(self, tmp_path, monkeypatch):
        p = _write_registry(tmp_path, _registry_payload(rate_limit_per_min=2))
        monkeypatch.setenv("GEMINI_PROXY_REGISTRY", str(p))
        app = web.Application()
        fake = AsyncMock(
            return_value=GeminiResult(text="ok", finish_reason="STOP",
                                      input_tokens=1, output_tokens=1)
        )
        with patch("common.gemini_proxy.call_gemini", new=fake):
            for _ in range(2):
                req = _make_post(
                    {"prompt": "hi"},
                    headers={"Authorization": f"Bearer {VALID_KEY}"},
                    app=app,
                )
                resp = await handle_gemini_ask(req)
                assert resp.status == 200, _resp_json(resp)
            # Third within the same minute → 429
            req = _make_post(
                {"prompt": "hi"},
                headers={"Authorization": f"Bearer {VALID_KEY}"},
                app=app,
            )
            resp = await handle_gemini_ask(req)
        assert resp.status == 429

    async def test_budget_exhausted_returns_429(self, tmp_path, monkeypatch):
        p = _write_registry(tmp_path, _registry_payload(daily_token_budget=10))
        monkeypatch.setenv("GEMINI_PROXY_REGISTRY", str(p))
        app = web.Application()

        # First call burns more than the daily budget.
        fake = AsyncMock(
            return_value=GeminiResult(text="ok", finish_reason="STOP",
                                      input_tokens=20, output_tokens=20)
        )
        with patch("common.gemini_proxy.call_gemini", new=fake):
            req = _make_post(
                {"prompt": "hi"},
                headers={"Authorization": f"Bearer {VALID_KEY}"},
                app=app,
            )
            resp = await handle_gemini_ask(req)
            assert resp.status == 200
            # Second call: budget already exhausted by previous consumption
            req = _make_post(
                {"prompt": "hi again"},
                headers={"Authorization": f"Bearer {VALID_KEY}"},
                app=app,
            )
            resp = await handle_gemini_ask(req)
        assert resp.status == 429
        assert "budget" in _resp_json(resp)["error"].lower()


# ---------------------------------------------------------------------------
# LAN admin bypass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLanAdmin:
    async def test_lan_admin_skips_topic_and_auth(self, app_with_registry, monkeypatch):
        app, _ = app_with_registry
        monkeypatch.setenv("GEMINI_PROXY_ADMIN_KEY", "adminsecret")
        fake = AsyncMock(
            return_value=GeminiResult(text="Paris.", finish_reason="STOP",
                                      input_tokens=5, output_tokens=2)
        )
        req = _make_post(
            {"prompt": "Capital of France?"},
            headers={"X-Admin-Key": "adminsecret"},
            ip="192.168.1.10",  # LAN
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=fake):
            resp = await handle_gemini_ask(req)
        assert resp.status == 200
        body = _resp_json(resp)
        assert body["client"] == "lan-admin"
        # No system_instruction passed
        kwargs = fake.call_args.kwargs
        assert kwargs["system_instruction"] is None

    async def test_lan_without_admin_key_still_requires_client_key(self, app_with_registry, monkeypatch):
        app, _ = app_with_registry
        monkeypatch.setenv("GEMINI_PROXY_ADMIN_KEY", "adminsecret")
        req = _make_post(
            {"prompt": "anything"},
            headers={},  # no X-Admin-Key, no client key
            ip="192.168.1.10",
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=AsyncMock()) as mock_call:
            resp = await handle_gemini_ask(req)
        assert resp.status == 401
        mock_call.assert_not_called()

    async def test_public_ip_admin_key_ignored(self, app_with_registry, monkeypatch):
        app, _ = app_with_registry
        monkeypatch.setenv("GEMINI_PROXY_ADMIN_KEY", "adminsecret")
        # Public IP, admin key presented — should NOT bypass; missing client key → 401
        req = _make_post(
            {"prompt": "anything"},
            headers={"X-Admin-Key": "adminsecret"},
            ip="8.8.8.8",
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=AsyncMock()) as mock_call:
            resp = await handle_gemini_ask(req)
        assert resp.status == 401
        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPing:
    async def test_ping_unknown_key_401(self, app_with_registry):
        app, _ = app_with_registry
        req = _make_request(
            method="GET",
            headers={"Authorization": f"Bearer {ANOTHER_KEY}"},
            app=app,
        )
        resp = await handle_gemini_ping(req)
        assert resp.status == 401

    async def test_ping_returns_remaining_budget(self, app_with_registry):
        app, _ = app_with_registry
        req = _make_request(
            method="GET",
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        resp = await handle_gemini_ping(req)
        assert resp.status == 200
        body = _resp_json(resp)
        assert body["ok"] is True
        assert body["client"] == "guitar-app"
        assert body["remaining_tokens_today"] == 200_000

    async def test_ping_lan_admin(self, app_with_registry, monkeypatch):
        app, _ = app_with_registry
        monkeypatch.setenv("GEMINI_PROXY_ADMIN_KEY", "adminsecret")
        req = _make_request(
            method="GET",
            ip="192.168.0.5",
            headers={"X-Admin-Key": "adminsecret"},
            app=app,
        )
        resp = await handle_gemini_ping(req)
        assert resp.status == 200
        body = _resp_json(resp)
        assert body["client"] == "lan-admin"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAuditLog:
    async def test_audit_record_written_for_success(self, app_with_registry, tmp_path):
        app, _ = app_with_registry
        audit_path = os.environ["GEMINI_PROXY_AUDIT_PATH"]
        fake = AsyncMock(
            return_value=GeminiResult(text="A Strat...", finish_reason="STOP",
                                      input_tokens=4, output_tokens=4)
        )
        req = _make_post(
            {"prompt": "Tell me about Strats"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            app=app,
        )
        with patch("common.gemini_proxy.call_gemini", new=fake):
            resp = await handle_gemini_ask(req)
        assert resp.status == 200
        lines = open(audit_path).read().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["status"] == "ok"
        assert rec["client"] == "guitar-app"
        assert rec["on_topic"] is True
        assert "prompt_preview" in rec
        assert rec["prompt_len"] > 0

    async def test_audit_record_written_for_unknown_key(self, app_with_registry):
        app, _ = app_with_registry
        audit_path = os.environ["GEMINI_PROXY_AUDIT_PATH"]
        req = _make_post(
            {"prompt": "hi"},
            headers={"Authorization": f"Bearer {ANOTHER_KEY}"},
            app=app,
        )
        resp = await handle_gemini_ask(req)
        assert resp.status == 401
        rec = json.loads(open(audit_path).read().strip())
        assert rec["status"] == "unknown_key"
