"""Tests for common/notify.py — notification endpoint (LAN-only, SMS/email)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from common.notify import (
    _phone_to_digits,
    _send_sms_gateway_sync,
    _send_sms_twilio_sync,
    _send_email_sync,
    handle_notify,
    is_private_ip,
    send_notification,
)


# ---------------------------------------------------------------------------
# is_private_ip tests
# ---------------------------------------------------------------------------

class TestIsPrivateIP:
    """Test IP restriction logic."""

    def _make_request(self, ip: str, xff: str = ""):
        """Build a mock aiohttp request with the given peername and XFF header."""
        transport = MagicMock()
        transport.get_extra_info.return_value = (ip, 12345)
        request = MagicMock(spec=web.Request)
        request.transport = transport
        headers = {"X-Forwarded-For": xff} if xff else {}
        request.headers = headers
        return request

    def test_localhost_allowed(self):
        assert is_private_ip(self._make_request("127.0.0.1")) is True

    def test_ipv6_loopback_allowed(self):
        assert is_private_ip(self._make_request("::1")) is True

    def test_192_168_allowed(self):
        assert is_private_ip(self._make_request("192.168.1.50")) is True

    def test_10_x_allowed(self):
        assert is_private_ip(self._make_request("10.0.0.5")) is True

    def test_172_16_allowed(self):
        assert is_private_ip(self._make_request("172.16.0.1")) is True

    def test_172_31_allowed(self):
        assert is_private_ip(self._make_request("172.31.255.254")) is True

    def test_public_ip_rejected(self):
        assert is_private_ip(self._make_request("8.8.8.8")) is False

    def test_public_ip_rejected_2(self):
        assert is_private_ip(self._make_request("1.2.3.4")) is False

    def test_xff_public_rejects_even_if_direct_is_local(self):
        """Envoy proxied request from external client: direct IP is 127.0.0.1 but XFF is public."""
        assert is_private_ip(self._make_request("127.0.0.1", xff="1.2.3.4")) is False

    def test_xff_private_allowed(self):
        """Request from another LAN host via proxy."""
        assert is_private_ip(self._make_request("127.0.0.1", xff="192.168.1.100")) is True

    def test_xff_multiple_entries_checks_first(self):
        """Multiple XFF entries — first is the real client."""
        assert is_private_ip(self._make_request("127.0.0.1", xff="8.8.8.8, 192.168.1.1")) is False

    def test_no_peername(self):
        transport = MagicMock()
        transport.get_extra_info.return_value = None
        request = MagicMock(spec=web.Request)
        request.transport = transport
        request.headers = {}
        assert is_private_ip(request) is False

    def test_invalid_ip_rejected(self):
        assert is_private_ip(self._make_request("not-an-ip")) is False


# ---------------------------------------------------------------------------
# Phone number helpers
# ---------------------------------------------------------------------------

class TestPhoneToDigits:
    def test_plus_prefix(self):
        assert _phone_to_digits("+14085551234") == "14085551234"

    def test_dashes(self):
        assert _phone_to_digits("408-555-1234") == "4085551234"

    def test_parens_and_spaces(self):
        assert _phone_to_digits("(408) 555-1234") == "4085551234"

    def test_plain_digits(self):
        assert _phone_to_digits("14085551234") == "14085551234"


# ---------------------------------------------------------------------------
# send_notification tests
# ---------------------------------------------------------------------------

class TestSendNotification:
    """Test the unified send_notification function."""

    @pytest.mark.asyncio
    async def test_sms_with_default_recipient(self):
        with patch.dict("os.environ", {"NOTIFY_DEFAULT_SMS": "+15551234567"}), \
             patch("common.notify.send_sms", new_callable=AsyncMock, return_value={"status": "sent", "sid": "SM123"}):
            result = await send_notification(channel="sms", message="test")
            assert result["status"] == "sent"
            assert result["channels"]["sms"]["status"] == "sent"

    @pytest.mark.asyncio
    async def test_email_with_default_recipient(self):
        with patch.dict("os.environ", {"NOTIFY_DEFAULT_EMAIL": "test@example.com"}), \
             patch("common.notify.send_email", new_callable=AsyncMock, return_value={"status": "sent"}):
            result = await send_notification(channel="email", message="test", subject="Alert")
            assert result["status"] == "sent"
            assert result["channels"]["email"]["status"] == "sent"

    @pytest.mark.asyncio
    async def test_both_channels(self):
        with patch.dict("os.environ", {
            "NOTIFY_DEFAULT_SMS": "+15551234567",
            "NOTIFY_DEFAULT_EMAIL": "test@example.com",
        }), \
             patch("common.notify.send_sms", new_callable=AsyncMock, return_value={"status": "sent", "sid": "SM1"}), \
             patch("common.notify.send_email", new_callable=AsyncMock, return_value={"status": "sent"}):
            result = await send_notification(channel="both", message="test")
            assert result["status"] == "sent"
            assert result["channels"]["sms"]["status"] == "sent"
            assert result["channels"]["email"]["status"] == "sent"

    @pytest.mark.asyncio
    async def test_sms_no_recipient_no_default(self):
        env = {k: v for k, v in __import__("os").environ.items() if k != "NOTIFY_DEFAULT_SMS"}
        with patch.dict("os.environ", env, clear=True):
            result = await send_notification(channel="sms", message="test")
            assert result["status"] == "error"
            assert "No recipient" in result["channels"]["sms"]["error"]

    @pytest.mark.asyncio
    async def test_explicit_recipient_overrides_default(self):
        with patch.dict("os.environ", {"NOTIFY_DEFAULT_SMS": "+10000000000"}), \
             patch("common.notify.send_sms", new_callable=AsyncMock, return_value={"status": "sent", "sid": "SM2"}) as mock_sms:
            await send_notification(channel="sms", message="test", to="+19999999999")
            mock_sms.assert_called_once_with("+19999999999", "test", via="twilio")

    @pytest.mark.asyncio
    async def test_sms_via_gateway(self):
        """sms_via='gateway' passes through to send_sms."""
        with patch.dict("os.environ", {"NOTIFY_DEFAULT_SMS": "+15551234567"}), \
             patch("common.notify.send_sms", new_callable=AsyncMock, return_value={"status": "sent", "provider": "gateway"}) as mock_sms:
            result = await send_notification(channel="sms", message="test", sms_via="gateway")
            mock_sms.assert_called_once_with("+15551234567", "test", via="gateway")
            assert result["status"] == "sent"

    @pytest.mark.asyncio
    async def test_sms_via_env_default(self):
        """NOTIFY_SMS_PROVIDER env var sets the default provider."""
        with patch.dict("os.environ", {"NOTIFY_DEFAULT_SMS": "+15551234567", "NOTIFY_SMS_PROVIDER": "gateway"}), \
             patch("common.notify.send_sms", new_callable=AsyncMock, return_value={"status": "sent", "provider": "gateway"}) as mock_sms:
            result = await send_notification(channel="sms", message="test")
            mock_sms.assert_called_once_with("+15551234567", "test", via="gateway")

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        with patch.dict("os.environ", {
            "NOTIFY_DEFAULT_SMS": "+15551234567",
            "NOTIFY_DEFAULT_EMAIL": "test@example.com",
        }), \
             patch("common.notify.send_sms", new_callable=AsyncMock, return_value={"status": "sent", "sid": "SM1"}), \
             patch("common.notify.send_email", new_callable=AsyncMock, return_value={"status": "error", "error": "SMTP down"}):
            result = await send_notification(channel="both", message="test")
            assert result["status"] == "partial"

    @pytest.mark.asyncio
    async def test_unknown_channel(self):
        result = await send_notification(channel="pigeon", message="coo")
        assert result["status"] == "error"
        assert "Unknown channel" in result["channels"]["pigeon"]["error"]

    @pytest.mark.asyncio
    async def test_timestamp_present(self):
        with patch.dict("os.environ", {"NOTIFY_DEFAULT_SMS": "+15551234567"}), \
             patch("common.notify.send_sms", new_callable=AsyncMock, return_value={"status": "sent", "sid": "SM1"}):
            result = await send_notification(channel="sms", message="test")
            assert "timestamp" in result


# ---------------------------------------------------------------------------
# SMS Twilio internals
# ---------------------------------------------------------------------------

class TestSMSTwilioSync:
    """Test _send_sms_twilio_sync with mocked Twilio."""

    def test_missing_env_vars(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _send_sms_twilio_sync("+1555", "hi")
            assert result["status"] == "error"
            assert "TWILIO_ACCOUNT_SID" in result["error"]

    def test_twilio_not_installed(self):
        with patch.dict("os.environ", {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_FROM_NUMBER": "+1000",
        }):
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "twilio.rest":
                    raise ImportError("No module named 'twilio'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = _send_sms_twilio_sync("+1555", "hi")
                assert result["status"] == "error"
                assert "twilio" in result["error"].lower()

    def test_twilio_success(self):
        mock_msg = MagicMock()
        mock_msg.sid = "SM_TEST_123"
        mock_client_cls = MagicMock()
        mock_client_cls.return_value.messages.create.return_value = mock_msg

        with patch.dict("os.environ", {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_FROM_NUMBER": "+1000",
        }):
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "twilio.rest":
                    mod = MagicMock()
                    mod.Client = mock_client_cls
                    return mod
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = _send_sms_twilio_sync("+15559999999", "hello")
                assert result["status"] == "sent"
                assert result["sid"] == "SM_TEST_123"
                assert result["provider"] == "twilio"


# ---------------------------------------------------------------------------
# SMS Gateway internals
# ---------------------------------------------------------------------------

class TestSMSGatewaySync:
    """Test _send_sms_gateway_sync with mocked SMTP."""

    def test_missing_env_vars(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _send_sms_gateway_sync("+14085551234", "hi")
            assert result["status"] == "error"
            assert "SMTP_USER" in result["error"]

    def test_gateway_success_default_domain(self):
        with patch.dict("os.environ", {
            "SMTP_USER": "me@gmail.com",
            "SMTP_PASSWORD": "app-password",
        }), patch("common.notify.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            result = _send_sms_gateway_sync("+14085551234", "trade filled")
            assert result["status"] == "sent"
            assert result["provider"] == "gateway"
            assert result["gateway"] == "14085551234@tmomail.net"
            mock_server.starttls.assert_called_once()
            mock_server.send_message.assert_called_once()

    def test_gateway_custom_domain(self):
        with patch.dict("os.environ", {
            "SMTP_USER": "me@gmail.com",
            "SMTP_PASSWORD": "app-password",
            "NOTIFY_SMS_GATEWAY": "vtext.com",
        }), patch("common.notify.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            result = _send_sms_gateway_sync("+14085551234", "hi")
            assert result["gateway"] == "14085551234@vtext.com"

    def test_gateway_strips_plus_and_dashes(self):
        with patch.dict("os.environ", {
            "SMTP_USER": "me@gmail.com",
            "SMTP_PASSWORD": "app-password",
        }), patch("common.notify.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            result = _send_sms_gateway_sync("+1-408-555-1234", "hi")
            assert result["gateway"] == "14085551234@tmomail.net"

    def test_gateway_smtp_failure(self):
        with patch.dict("os.environ", {
            "SMTP_USER": "me@gmail.com",
            "SMTP_PASSWORD": "app-password",
        }), patch("common.notify.smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.__enter__ = MagicMock(side_effect=Exception("Connection refused"))
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            result = _send_sms_gateway_sync("+14085551234", "msg")
            assert result["status"] == "error"
            assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# Email internals
# ---------------------------------------------------------------------------

class TestEmailSync:
    """Test _send_email_sync with mocked SMTP."""

    def test_missing_env_vars(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _send_email_sync("test@example.com", "Sub", "Body")
            assert result["status"] == "error"
            assert "SMTP_USER" in result["error"]

    def test_smtp_success(self):
        with patch.dict("os.environ", {
            "SMTP_USER": "me@gmail.com",
            "SMTP_PASSWORD": "app-password",
        }), patch("common.notify.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            result = _send_email_sync("to@example.com", "Alert", "Trade filled")
            assert result["status"] == "sent"
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("me@gmail.com", "app-password")
            mock_server.send_message.assert_called_once()

    def test_smtp_failure(self):
        with patch.dict("os.environ", {
            "SMTP_USER": "me@gmail.com",
            "SMTP_PASSWORD": "app-password",
        }), patch("common.notify.smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.__enter__ = MagicMock(side_effect=Exception("Connection refused"))
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            result = _send_email_sync("to@example.com", "Alert", "msg")
            assert result["status"] == "error"
            assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# Handler tests (aiohttp integration)
# ---------------------------------------------------------------------------

class TestHandleNotify:
    """Test the aiohttp handler via mock requests."""

    def _make_request(self, body: dict, ip: str = "127.0.0.1", xff: str = ""):
        transport = MagicMock()
        transport.get_extra_info.return_value = (ip, 12345)
        request = MagicMock(spec=web.Request)
        request.transport = transport
        headers = {"X-Forwarded-For": xff} if xff else {}
        request.headers = headers

        async def json_coro():
            return body
        request.json = json_coro
        return request

    @pytest.mark.asyncio
    async def test_forbidden_from_public_ip(self):
        req = self._make_request({"message": "hi"}, ip="8.8.8.8")
        resp = await handle_notify(req)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_forbidden_via_xff(self):
        req = self._make_request({"message": "hi"}, ip="127.0.0.1", xff="1.2.3.4")
        resp = await handle_notify(req)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_missing_message(self):
        req = self._make_request({"channel": "sms"})
        resp = await handle_notify(req)
        assert resp.status == 400
        data = json.loads(resp.body)
        assert "message" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_channel(self):
        req = self._make_request({"channel": "telegram", "message": "hi"})
        resp = await handle_notify(req)
        assert resp.status == 400
        data = json.loads(resp.body)
        assert "Invalid channel" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_sms_via(self):
        req = self._make_request({"message": "hi", "sms_via": "pigeonpost"})
        resp = await handle_notify(req)
        assert resp.status == 400
        data = json.loads(resp.body)
        assert "Invalid sms_via" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        transport = MagicMock()
        transport.get_extra_info.return_value = ("127.0.0.1", 12345)
        request = MagicMock(spec=web.Request)
        request.transport = transport
        request.headers = {}

        async def bad_json():
            raise ValueError("bad json")
        request.json = bad_json
        resp = await handle_notify(request)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_success_sms(self):
        req = self._make_request({"message": "Trade filled", "to": "+15551234567"})
        with patch("common.notify.send_notification", new_callable=AsyncMock, return_value={
            "status": "sent",
            "channels": {"sms": {"status": "sent", "sid": "SM1"}},
            "timestamp": "2026-04-20T10:30:00+00:00",
        }):
            resp = await handle_notify(req)
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_success_sms_via_gateway(self):
        req = self._make_request({"message": "Trade filled", "to": "+15551234567", "sms_via": "gateway"})
        with patch("common.notify.send_notification", new_callable=AsyncMock, return_value={
            "status": "sent",
            "channels": {"sms": {"status": "sent", "provider": "gateway"}},
            "timestamp": "2026-04-20T10:30:00+00:00",
        }) as mock_send:
            resp = await handle_notify(req)
            assert resp.status == 200
            mock_send.assert_called_once_with(
                channel="sms", message="Trade filled", to="+15551234567",
                subject="Trade Alert", sms_via="gateway",
            )

    @pytest.mark.asyncio
    async def test_partial_returns_207(self):
        req = self._make_request({"channel": "both", "message": "test", "to": "+1555"})
        with patch("common.notify.send_notification", new_callable=AsyncMock, return_value={
            "status": "partial",
            "channels": {"sms": {"status": "sent"}, "email": {"status": "error", "error": "no creds"}},
            "timestamp": "2026-04-20T10:30:00+00:00",
        }):
            resp = await handle_notify(req)
            assert resp.status == 207

    @pytest.mark.asyncio
    async def test_all_fail_returns_503(self):
        req = self._make_request({"message": "test", "to": "+1555"})
        with patch("common.notify.send_notification", new_callable=AsyncMock, return_value={
            "status": "error",
            "channels": {"sms": {"status": "error", "error": "no creds"}},
            "timestamp": "2026-04-20T10:30:00+00:00",
        }):
            resp = await handle_notify(req)
            assert resp.status == 503
