"""
Notification service for trade alerts — SMS (Twilio or carrier gateway) and Email (SMTP).

LAN-only: requests must originate from private networks (192.168.x.x, 10.x.x.x,
172.16-31.x.x, 127.0.0.1, ::1). External requests proxied through envoy are
detected via X-Forwarded-For and rejected.

SMS has two providers:
    twilio   — Twilio API (requires purchased Twilio number, ~$1.15/mo)
    gateway  — Email-to-SMS via carrier gateway (free, uses SMTP).
               Sends email to {phone}@{carrier_gateway} which arrives as SMS.
               Default gateway: tmomail.net (T-Mobile / Google Fi).

Environment variables:
    SMS (Twilio):
        TWILIO_ACCOUNT_SID   — Twilio account SID
        TWILIO_AUTH_TOKEN     — Twilio auth token
        TWILIO_FROM_NUMBER    — Twilio phone number (e.g. +1234567890)

    SMS (Gateway) / Email (SMTP / Gmail):
        SMTP_HOST             — SMTP server (default: smtp.gmail.com)
        SMTP_PORT             — SMTP port (default: 587)
        SMTP_USER             — SMTP username / email
        SMTP_PASSWORD         — SMTP password / Gmail App Password

    Gateway:
        NOTIFY_SMS_GATEWAY    — Carrier email-to-SMS domain (default: tmomail.net)

    Defaults:
        NOTIFY_DEFAULT_SMS    — Default SMS recipient phone number
        NOTIFY_DEFAULT_EMAIL  — Default email recipient address
        NOTIFY_SMS_PROVIDER   — Default SMS provider: "twilio" or "gateway" (default: twilio)

Usage from other modules:
    from common.notify import send_notification, is_private_ip

    # SMS via Twilio (default)
    result = await send_notification(channel="sms", message="Trade filled", to="+1...")

    # SMS via carrier email gateway (free, no Twilio number needed)
    result = await send_notification(channel="sms", message="Trade filled", sms_via="gateway")

    # In an aiohttp handler
    if not is_private_ip(request):
        return web.json_response({"error": "forbidden"}, status=403)
"""

import asyncio
import ipaddress
import logging
import os
import re
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

# Common carrier email-to-SMS gateways
CARRIER_GATEWAYS = {
    "tmomail.net": "T-Mobile / Google Fi",
    "vtext.com": "Verizon",
    "txt.att.net": "AT&T",
    "messaging.sprintpcs.com": "Sprint",
    "myboostmobile.com": "Boost Mobile",
    "sms.cricketwireless.net": "Cricket",
}

# Prefix tag on all email subjects — use this to create a Gmail filter
# that forces priority/notification on your phone.
SUBJECT_TAG = "[UTP-ALERT]"

# ---------------------------------------------------------------------------
# IP restriction
# ---------------------------------------------------------------------------

def is_private_ip(request: web.Request) -> bool:
    """Check that the request originates from a private/local network.

    If X-Forwarded-For is present (envoy proxy), the *first* entry is the
    real client IP — that must also be private.  This prevents external
    requests that arrive via the reverse proxy from bypassing the check.
    """
    peername = request.transport.get_extra_info('peername')
    direct_ip = peername[0] if peername else None
    if not direct_ip:
        return False

    # If proxied, check the original client IP
    xff = request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
    check_ip = xff if xff else direct_ip

    try:
        addr = ipaddress.ip_address(check_ip)
        return addr.is_private or addr.is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Phone number helpers
# ---------------------------------------------------------------------------

def _phone_to_digits(phone: str) -> str:
    """Strip a phone number to just digits (e.g. '+14085551234' -> '14085551234')."""
    return re.sub(r'\D', '', phone)


# ---------------------------------------------------------------------------
# SMS via Twilio
# ---------------------------------------------------------------------------

def _send_sms_twilio_sync(to: str, message: str) -> dict:
    """Send an SMS via Twilio (blocking).  Called in an executor."""
    sid = os.environ.get('TWILIO_ACCOUNT_SID')
    token = os.environ.get('TWILIO_AUTH_TOKEN')
    from_number = os.environ.get('TWILIO_FROM_NUMBER')

    if not all([sid, token, from_number]):
        missing = [v for v in ('TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_FROM_NUMBER')
                   if not os.environ.get(v)]
        return {"status": "error", "error": f"Missing env vars: {', '.join(missing)}"}

    try:
        from twilio.rest import Client  # lazy import
    except ImportError:
        return {"status": "error", "error": "twilio package not installed (pip install twilio)"}

    try:
        client = Client(sid, token)
        msg = client.messages.create(body=message, from_=from_number, to=to)
        logger.info("SMS (twilio) sent to %s  sid=%s", to, msg.sid)
        return {"status": "sent", "sid": msg.sid, "provider": "twilio"}
    except Exception as exc:
        logger.exception("SMS (twilio) send failed")
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# SMS via carrier email-to-SMS gateway
# ---------------------------------------------------------------------------

def _send_sms_gateway_sync(to: str, message: str) -> dict:
    """Send SMS via email-to-SMS carrier gateway (blocking).  Called in an executor.

    Sends an email to {digits}@{gateway} using SMTP.  The carrier delivers
    it as a text message.  Default gateway: tmomail.net (T-Mobile / Google Fi).
    """
    host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    port = int(os.environ.get('SMTP_PORT', '587'))
    user = os.environ.get('SMTP_USER')
    password = os.environ.get('SMTP_PASSWORD')

    if not all([user, password]):
        missing = [v for v in ('SMTP_USER', 'SMTP_PASSWORD') if not os.environ.get(v)]
        return {"status": "error", "error": f"Missing env vars: {', '.join(missing)}"}

    gateway = os.environ.get('NOTIFY_SMS_GATEWAY', 'tmomail.net')
    digits = _phone_to_digits(to)
    gateway_email = f"{digits}@{gateway}"

    msg = MIMEText(message, 'plain')
    msg['From'] = user
    msg['To'] = gateway_email
    msg['Subject'] = SUBJECT_TAG

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        logger.info("SMS (gateway) sent to %s via %s", to, gateway_email)
        return {"status": "sent", "provider": "gateway", "gateway": gateway_email}
    except Exception as exc:
        logger.exception("SMS (gateway) send failed")
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Unified SMS dispatcher
# ---------------------------------------------------------------------------

async def send_sms(to: str, message: str, via: str = "twilio") -> dict:
    """Send SMS via the chosen provider.

    Args:
        to:  Recipient phone number (e.g. +14085551234)
        message: Text body
        via: "twilio" or "gateway"
    """
    loop = asyncio.get_running_loop()
    if via == "gateway":
        return await loop.run_in_executor(None, _send_sms_gateway_sync, to, message)
    return await loop.run_in_executor(None, _send_sms_twilio_sync, to, message)


# ---------------------------------------------------------------------------
# Email via SMTP
# ---------------------------------------------------------------------------

def _send_email_sync(to: str, subject: str, message: str) -> dict:
    """Send an email via SMTP (blocking).  Called in an executor."""
    host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    port = int(os.environ.get('SMTP_PORT', '587'))
    user = os.environ.get('SMTP_USER')
    password = os.environ.get('SMTP_PASSWORD')

    if not all([user, password]):
        missing = [v for v in ('SMTP_USER', 'SMTP_PASSWORD') if not os.environ.get(v)]
        return {"status": "error", "error": f"Missing env vars: {', '.join(missing)}"}

    tagged_subject = f"{SUBJECT_TAG} {subject}" if not subject.startswith(SUBJECT_TAG) else subject

    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = to
    msg['Subject'] = tagged_subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        logger.info("Email sent to %s  subject=%r", to, subject)
        return {"status": "sent"}
    except Exception as exc:
        logger.exception("Email send failed")
        return {"status": "error", "error": str(exc)}


async def send_email(to: str, subject: str, message: str) -> dict:
    """Async wrapper — runs SMTP call in the default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _send_email_sync, to, subject, message)


# ---------------------------------------------------------------------------
# Unified send
# ---------------------------------------------------------------------------

async def send_notification(
    *,
    channel: str = "sms",
    message: str,
    to: Optional[str] = None,
    subject: str = "Trade Alert",
    sms_via: Optional[str] = None,
) -> dict:
    """Send a notification via the specified channel(s).

    Args:
        channel:  "sms", "email", or "both"
        message:  Notification body text
        to:       Recipient (phone for SMS, email for email).
                  Falls back to NOTIFY_DEFAULT_SMS / NOTIFY_DEFAULT_EMAIL env vars.
        subject:  Email subject line (ignored for SMS-only)
        sms_via:  SMS provider — "twilio" or "gateway".
                  Falls back to NOTIFY_SMS_PROVIDER env var, then "twilio".

    Returns:
        dict with per-channel results and overall status.
    """
    results: dict = {}
    channels = ["sms", "email"] if channel == "both" else [channel]
    provider = sms_via or os.environ.get('NOTIFY_SMS_PROVIDER', 'twilio')

    for ch in channels:
        if ch == "sms":
            recipient = to or os.environ.get('NOTIFY_DEFAULT_SMS')
            if not recipient:
                results["sms"] = {"status": "error", "error": "No recipient — set 'to' or NOTIFY_DEFAULT_SMS"}
                continue
            results["sms"] = await send_sms(recipient, message, via=provider)

        elif ch == "email":
            recipient = to or os.environ.get('NOTIFY_DEFAULT_EMAIL')
            if not recipient:
                results["email"] = {"status": "error", "error": "No recipient — set 'to' or NOTIFY_DEFAULT_EMAIL"}
                continue
            results["email"] = await send_email(recipient, subject, message)

        else:
            results[ch] = {"status": "error", "error": f"Unknown channel: {ch}"}

    all_ok = all(r.get("status") == "sent" for r in results.values())
    return {
        "status": "sent" if all_ok else "partial" if any(r.get("status") == "sent" for r in results.values()) else "error",
        "channels": results,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# aiohttp handler
# ---------------------------------------------------------------------------

async def handle_notify(request: web.Request) -> web.Response:
    """POST /api/notify — send SMS/email notification.  LAN-only.

    Request body (JSON):
        {
            "channel":  "sms" | "email" | "both",   // default: "sms"
            "sms_via":  "twilio" | "gateway",        // default: "twilio" (or NOTIFY_SMS_PROVIDER)
            "to":       "+1234567890" | "a@b.com",   // optional if env default set
            "subject":  "Trade Alert",               // optional, email only
            "message":  "Sold 5x SPX 5500P ..."      // required
        }
    """
    if not is_private_ip(request):
        return web.json_response({"error": "Forbidden — LAN access only"}, status=403)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    message = body.get("message")
    if not message:
        return web.json_response({"error": "Missing required field: message"}, status=400)

    channel = body.get("channel", "sms")
    if channel not in ("sms", "email", "both"):
        return web.json_response({"error": f"Invalid channel: {channel}. Use sms, email, or both"}, status=400)

    sms_via = body.get("sms_via")
    if sms_via and sms_via not in ("twilio", "gateway"):
        return web.json_response({"error": f"Invalid sms_via: {sms_via}. Use twilio or gateway"}, status=400)

    result = await send_notification(
        channel=channel,
        message=message,
        to=body.get("to"),
        subject=body.get("subject", "Trade Alert"),
        sms_via=sms_via,
    )

    status_code = 200 if result["status"] == "sent" else 207 if result["status"] == "partial" else 503
    return web.json_response(result, status=status_code)
