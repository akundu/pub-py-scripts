"""
Gemini proxy with per-client topic gating.

Mobile (and other shipped) apps call this proxy instead of holding the
GEMINI_API_KEY directly.  Each client identifies itself with a long random
key; the server looks up the client's allowed topic and prepends a
system instruction that forces Gemini to refuse off-topic questions.

Endpoints (registered in scripts/routes.py and db_server.py):
    POST /api/gemini/ask    body {prompt, model?, max_output_tokens?}
    GET  /api/gemini/ping

Environment:
    GEMINI_API_KEY              upstream key, never sent to clients
    GEMINI_PROXY_REGISTRY       path to client registry JSON
                                (default ~/.config/stocks/gemini_proxy_clients.json)
    GEMINI_PROXY_ADMIN_KEY      LAN admin bypass key (header X-Admin-Key)
    GEMINI_PROXY_DISABLED       "1" -> both endpoints return 503
    GEMINI_PROXY_BUDGET_PATH    daily budget checkpoint path
    GEMINI_PROXY_AUDIT_PATH     audit log path

LAN admin bypass: a request from a private IP that also presents
X-Admin-Key matching GEMINI_PROXY_ADMIN_KEY skips the client-key check
and the topic restriction.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hmac
import json
import logging
import os
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aiohttp import web

from common.notify import is_private_ip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY_PATH = "~/.config/stocks/gemini_proxy_clients.json"
DEFAULT_BUDGET_PATH = "~/.config/stocks/gemini_proxy_budget.json"
DEFAULT_AUDIT_PATH = "~/.config/stocks/gemini_proxy_audit.jsonl"

REFUSAL_TEXT = "I can't talk about that."

# Suffix appended to every per-client system_prompt to force a uniform refusal
# on off-topic queries.  Keep this terse and exact — variations let Gemini
# editorialise the refusal.
REFUSAL_SUFFIX_TEMPLATE = (
    "\n\nIMPORTANT: You may only answer questions about {topic_label}. "
    "If the user asks about anything else, respond with exactly: "
    "{refusal!r}. Do not answer the off-topic question, do not explain why, "
    "do not apologise further, do not add anything after that sentence."
)

CLASSIFIER_MODEL_DEFAULT = "gemini-2.5-flash-lite"

MIN_CLIENT_KEY_LEN = 32  # bytes after base64-decode would be ~24, but we keep the raw string length check loose


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ClientConfig:
    client_key: str
    name: str
    topic_label: str
    system_prompt: str
    model: str = "gemini-flash-latest"
    strict_mode: bool = False
    rate_limit_per_min: int = 60
    daily_token_budget: int = 200_000
    max_prompt_chars: int = 4000
    enabled: bool = True


@dataclasses.dataclass
class GeminiResult:
    text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class Registry:
    """Loads client config from a JSON file. Auto-reloads on mtime change."""

    def __init__(self, path: str | os.PathLike):
        self._path = Path(os.path.expanduser(str(path)))
        self._mtime: float = 0.0
        self._by_key: dict[str, ClientConfig] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        if not self._path.exists():
            self._by_key = {}
            self._mtime = 0.0
            self._loaded = True
            logger.warning("gemini_proxy: registry file not found at %s", self._path)
            return

        st = self._path.stat()
        # World-readable warning
        mode = st.st_mode & 0o777
        if mode & 0o044:
            logger.warning(
                "gemini_proxy: registry %s is world/group-readable (mode %o). "
                "Restrict with `chmod 600`.",
                self._path,
                mode,
            )

        with self._path.open("r") as fh:
            data = json.load(fh)

        clients = data.get("clients", [])
        by_key: dict[str, ClientConfig] = {}
        for entry in clients:
            cfg = self._parse_entry(entry)
            if cfg.client_key in by_key:
                raise ValueError(
                    f"gemini_proxy: duplicate client_key in {self._path} for client {cfg.name!r}"
                )
            by_key[cfg.client_key] = cfg

        self._by_key = by_key
        self._mtime = st.st_mtime
        self._loaded = True
        logger.info("gemini_proxy: loaded %d client(s) from %s", len(by_key), self._path)

    @staticmethod
    def _parse_entry(entry: dict) -> ClientConfig:
        required = ("client_key", "name", "topic_label", "system_prompt")
        missing = [k for k in required if not entry.get(k)]
        if missing:
            raise ValueError(f"gemini_proxy: client entry missing required fields: {missing}")
        client_key = entry["client_key"]
        if len(client_key) < MIN_CLIENT_KEY_LEN:
            raise ValueError(
                f"gemini_proxy: client_key for {entry.get('name')!r} is too short "
                f"(len={len(client_key)}, min={MIN_CLIENT_KEY_LEN})"
            )

        # Allow only the documented fields.
        allowed = {f.name for f in dataclasses.fields(ClientConfig)}
        kwargs = {k: v for k, v in entry.items() if k in allowed}
        return ClientConfig(**kwargs)

    def maybe_reload(self) -> None:
        """Reload if the file has changed since last read. Cheap mtime check."""
        try:
            if not self._loaded:
                self.load()
                return
            if not self._path.exists():
                if self._by_key:
                    logger.warning("gemini_proxy: registry file disappeared at %s", self._path)
                    self._by_key = {}
                self._mtime = 0.0
                return
            mtime = self._path.stat().st_mtime
            if mtime != self._mtime:
                self.load()
        except Exception:
            logger.exception("gemini_proxy: registry reload failed; keeping previous in-memory copy")

    def lookup(self, client_key: str) -> Optional[ClientConfig]:
        """Constant-time match against every registered key."""
        if not client_key:
            return None
        for k, cfg in self._by_key.items():
            if hmac.compare_digest(k, client_key):
                return cfg
        return None

    def all_clients(self) -> list[ClientConfig]:
        return list(self._by_key.values())


# ---------------------------------------------------------------------------
# Rate limiter (per-minute token bucket, per client)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-key per-minute token bucket. Refills linearly."""

    def __init__(self) -> None:
        # key -> (tokens, last_refill_ts)
        self._state: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, per_min: int, now: Optional[float] = None) -> bool:
        if per_min <= 0:
            return True
        now = now if now is not None else time.monotonic()
        capacity = float(per_min)
        tokens, last = self._state.get(key, (capacity, now))
        elapsed = max(0.0, now - last)
        # refill rate: per_min tokens / 60s
        tokens = min(capacity, tokens + elapsed * (capacity / 60.0))
        if tokens >= 1.0:
            tokens -= 1.0
            self._state[key] = (tokens, now)
            return True
        self._state[key] = (tokens, now)
        return False


# ---------------------------------------------------------------------------
# Daily token budget (persisted across restarts)
# ---------------------------------------------------------------------------

class DailyBudget:
    def __init__(self, persist_path: str | os.PathLike):
        self._path = Path(os.path.expanduser(str(persist_path)))
        # name -> {"date": "YYYY-MM-DD", "tokens": int}
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._state = {}
            return
        try:
            with self._path.open("r") as fh:
                self._state = json.load(fh) or {}
        except Exception:
            logger.exception("gemini_proxy: budget checkpoint unreadable; starting fresh")
            self._state = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w") as fh:
                json.dump(self._state, fh)
            os.replace(tmp, self._path)
        except Exception:
            logger.exception("gemini_proxy: failed to persist budget checkpoint")

    @staticmethod
    def _today_str() -> str:
        return date.today().isoformat()

    def _get_today_used(self, name: str) -> int:
        today = self._today_str()
        entry = self._state.get(name)
        if not entry or entry.get("date") != today:
            return 0
        return int(entry.get("tokens", 0))

    def remaining(self, name: str, daily_budget: int) -> int:
        if daily_budget <= 0:
            return 0
        return max(0, daily_budget - self._get_today_used(name))

    def consume(self, name: str, tokens: int) -> None:
        if tokens <= 0:
            return
        today = self._today_str()
        entry = self._state.get(name)
        if not entry or entry.get("date") != today:
            entry = {"date": today, "tokens": 0}
        entry["tokens"] = int(entry.get("tokens", 0)) + int(tokens)
        self._state[name] = entry
        self._save()

    def has_budget(self, name: str, daily_budget: int) -> bool:
        return self.remaining(name, daily_budget) > 0


# ---------------------------------------------------------------------------
# System instruction builder
# ---------------------------------------------------------------------------

def build_system_instruction(cfg: ClientConfig) -> str:
    """Compose the per-client system_prompt + the fixed refusal suffix."""
    suffix = REFUSAL_SUFFIX_TEMPLATE.format(topic_label=cfg.topic_label, refusal=REFUSAL_TEXT)
    return f"{cfg.system_prompt.rstrip()}{suffix}"


def is_refusal(text: str) -> bool:
    """Heuristic: did Gemini's reply look like the canonical refusal?"""
    norm = (text or "").strip().rstrip(".").lower()
    target = REFUSAL_TEXT.rstrip(".").lower()
    return norm == target


# ---------------------------------------------------------------------------
# Gemini call wrappers
# ---------------------------------------------------------------------------

async def call_gemini(
    *,
    system_instruction: Optional[str],
    prompt: str,
    model: str,
    max_output_tokens: Optional[int] = None,
) -> GeminiResult:
    """Async wrapper around the synchronous google.genai SDK."""
    return await asyncio.to_thread(
        _call_gemini_sync,
        system_instruction,
        prompt,
        model,
        max_output_tokens,
    )


def _call_gemini_sync(
    system_instruction: Optional[str],
    prompt: str,
    model: str,
    max_output_tokens: Optional[int],
) -> GeminiResult:
    # Imported lazily so the module loads cleanly even if google-genai isn't
    # installed in the test environment (tests mock call_gemini directly).
    import google.genai as genai
    from google.genai import types as genai_types  # type: ignore

    client = genai.Client()

    config_kwargs: dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if max_output_tokens:
        config_kwargs["max_output_tokens"] = int(max_output_tokens)

    config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=config,
    )

    text = (response.text or "").strip() if hasattr(response, "text") else ""
    finish_reason = "UNKNOWN"
    if getattr(response, "candidates", None):
        fr = getattr(response.candidates[0], "finish_reason", None)
        if fr is not None:
            finish_reason = getattr(fr, "name", str(fr))

    usage = getattr(response, "usage_metadata", None)
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

    return GeminiResult(
        text=text,
        finish_reason=finish_reason,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


async def classify_on_topic(prompt: str, topic_label: str, model: str) -> bool:
    """Cheap pre-classifier for strict_mode. Returns True if on topic."""
    instruction = (
        f"You are a strict topic classifier. The allowed topic is: {topic_label}. "
        f"Reply with exactly one word: YES if the user message is about that topic, "
        f"NO otherwise. No punctuation, no explanation."
    )
    try:
        result = await call_gemini(
            system_instruction=instruction,
            prompt=prompt,
            model=model,
            max_output_tokens=4,
        )
    except Exception:
        # If classifier fails, fall through to the main call (which still
        # has the system-prompt refusal). Better than blocking legitimate users.
        logger.exception("gemini_proxy: classifier call failed; defaulting to on-topic")
        return True

    answer = (result.text or "").strip().upper()
    return answer.startswith("Y")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def audit_log(record: dict, path: str | os.PathLike) -> None:
    """Append one JSONL line. Best-effort; never raises into the request path."""
    try:
        p = Path(os.path.expanduser(str(path)))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        logger.exception("gemini_proxy: audit_log write failed (record=%s)", record.get("status"))


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def extract_client_key(request: web.Request) -> Optional[str]:
    """Read the bearer token from Authorization, or X-Client-Key fallback."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[len("Bearer "):].strip()
        if token:
            return token
    fallback = request.headers.get("X-Client-Key", "").strip()
    return fallback or None


def is_lan_admin(request: web.Request, admin_key_env: Optional[str]) -> bool:
    """LAN admin bypass: private IP AND matching X-Admin-Key."""
    if not admin_key_env:
        return False
    if not is_private_ip(request):
        return False
    presented = request.headers.get("X-Admin-Key", "")
    if not presented:
        return False
    return hmac.compare_digest(presented, admin_key_env)


# ---------------------------------------------------------------------------
# Singleton state on the aiohttp app
# ---------------------------------------------------------------------------

_STATE_KEY = "gemini_proxy_state"


def _get_state(app: web.Application) -> dict:
    """Lazy-init the proxy's state dict on the app."""
    state = app.get(_STATE_KEY)
    if state is None:
        registry_path = os.environ.get("GEMINI_PROXY_REGISTRY") or DEFAULT_REGISTRY_PATH
        budget_path = os.environ.get("GEMINI_PROXY_BUDGET_PATH") or DEFAULT_BUDGET_PATH
        audit_path = os.environ.get("GEMINI_PROXY_AUDIT_PATH") or DEFAULT_AUDIT_PATH
        registry = Registry(registry_path)
        try:
            registry.load()
        except Exception:
            logger.exception("gemini_proxy: initial registry load failed; will retry on next request")
        state = {
            "registry": registry,
            "rate_limiter": RateLimiter(),
            "budget": DailyBudget(budget_path),
            "audit_path": os.path.expanduser(audit_path),
        }
        app[_STATE_KEY] = state
    return state


def reset_state(app: web.Application) -> None:
    """Test helper: drop the cached state so the next request reloads from env."""
    app.pop(_STATE_KEY, None)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _client_ip(request: web.Request) -> str:
    xff = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if xff:
        return xff
    peer = request.transport.get_extra_info("peername") if request.transport else None
    return peer[0] if peer else "?"


async def handle_gemini_ask(request: web.Request) -> web.Response:
    """POST /api/gemini/ask"""
    if os.environ.get("GEMINI_PROXY_DISABLED") == "1":
        return web.json_response({"error": "endpoint disabled"}, status=503)
    if not os.environ.get("GEMINI_API_KEY"):
        return web.json_response({"error": "server missing GEMINI_API_KEY"}, status=503)

    state = _get_state(request.app)
    registry: Registry = state["registry"]
    rate: RateLimiter = state["rate_limiter"]
    budget: DailyBudget = state["budget"]
    audit_path: str = state["audit_path"]

    registry.maybe_reload()

    # Parse body
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "prompt is required"}, status=400)
    requested_model = body.get("model")
    requested_max_tokens = body.get("max_output_tokens")

    started = time.monotonic()
    client_ip = _client_ip(request)

    admin = is_lan_admin(request, os.environ.get("GEMINI_PROXY_ADMIN_KEY"))

    cfg: Optional[ClientConfig] = None
    client_key_for_log = ""

    if admin:
        # Unrestricted LAN admin path
        client_name = "lan-admin"
        model = requested_model or "gemini-flash-latest"
        if requested_max_tokens is not None:
            try:
                requested_max_tokens = int(requested_max_tokens)
            except (TypeError, ValueError):
                return web.json_response({"error": "max_output_tokens must be int"}, status=400)
        try:
            result = await call_gemini(
                system_instruction=None,
                prompt=prompt,
                model=model,
                max_output_tokens=requested_max_tokens,
            )
        except Exception as exc:
            logger.exception("gemini_proxy: admin upstream error")
            audit_log(
                _audit_record(
                    client_name=client_name,
                    client_ip=client_ip,
                    prompt=prompt,
                    on_topic=None,
                    status="upstream_error",
                    started=started,
                    extra={"error": str(exc)},
                ),
                audit_path,
            )
            return web.json_response({"error": "upstream gemini error"}, status=502)

        audit_log(
            _audit_record(
                client_name=client_name,
                client_ip=client_ip,
                prompt=prompt,
                on_topic=None,
                status="ok",
                started=started,
                extra={
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "finish_reason": result.finish_reason,
                    "admin": True,
                },
            ),
            audit_path,
        )
        return web.json_response(
            {
                "answer": result.text,
                "on_topic": True,
                "finish_reason": result.finish_reason,
                "tokens": {"input": result.input_tokens, "output": result.output_tokens},
                "client": client_name,
            }
        )

    # Regular client path
    client_key = extract_client_key(request)
    client_key_for_log = (client_key or "")[:8]
    if not client_key:
        audit_log(
            _audit_record(
                client_name="?",
                client_ip=client_ip,
                prompt=prompt,
                on_topic=None,
                status="missing_key",
                started=started,
                extra={"key_prefix": client_key_for_log},
            ),
            audit_path,
        )
        return web.json_response({"error": "missing client key"}, status=401)

    cfg = registry.lookup(client_key)
    if cfg is None:
        audit_log(
            _audit_record(
                client_name="?",
                client_ip=client_ip,
                prompt=prompt,
                on_topic=None,
                status="unknown_key",
                started=started,
                extra={"key_prefix": client_key_for_log},
            ),
            audit_path,
        )
        return web.json_response({"error": "unknown client key"}, status=401)

    if not cfg.enabled:
        audit_log(
            _audit_record(
                client_name=cfg.name,
                client_ip=client_ip,
                prompt=prompt,
                on_topic=None,
                status="client_disabled",
                started=started,
            ),
            audit_path,
        )
        return web.json_response({"error": "client disabled"}, status=403)

    if len(prompt) > cfg.max_prompt_chars:
        audit_log(
            _audit_record(
                client_name=cfg.name,
                client_ip=client_ip,
                prompt=prompt,
                on_topic=None,
                status="prompt_too_long",
                started=started,
                extra={"prompt_len": len(prompt), "max": cfg.max_prompt_chars},
            ),
            audit_path,
        )
        return web.json_response({"error": "prompt too long"}, status=413)

    if not rate.allow(cfg.name, cfg.rate_limit_per_min):
        audit_log(
            _audit_record(
                client_name=cfg.name,
                client_ip=client_ip,
                prompt=prompt,
                on_topic=None,
                status="rate_limited",
                started=started,
            ),
            audit_path,
        )
        return web.json_response({"error": "rate limit exceeded"}, status=429)

    if not budget.has_budget(cfg.name, cfg.daily_token_budget):
        audit_log(
            _audit_record(
                client_name=cfg.name,
                client_ip=client_ip,
                prompt=prompt,
                on_topic=None,
                status="budget_exhausted",
                started=started,
            ),
            audit_path,
        )
        return web.json_response({"error": "daily token budget exhausted"}, status=429)

    model = requested_model or cfg.model

    # Strict mode: cheap pre-classifier
    if cfg.strict_mode:
        on_topic = await classify_on_topic(prompt, cfg.topic_label, CLASSIFIER_MODEL_DEFAULT)
        if not on_topic:
            audit_log(
                _audit_record(
                    client_name=cfg.name,
                    client_ip=client_ip,
                    prompt=prompt,
                    on_topic=False,
                    status="off_topic_blocked",
                    started=started,
                ),
                audit_path,
            )
            return web.json_response(
                {
                    "answer": REFUSAL_TEXT,
                    "on_topic": False,
                    "finish_reason": "BLOCKED_OFF_TOPIC",
                    "tokens": {"input": 0, "output": 0},
                    "client": cfg.name,
                }
            )

    # Main call
    if requested_max_tokens is not None:
        try:
            requested_max_tokens = int(requested_max_tokens)
        except (TypeError, ValueError):
            return web.json_response({"error": "max_output_tokens must be int"}, status=400)

    try:
        result = await call_gemini(
            system_instruction=build_system_instruction(cfg),
            prompt=prompt,
            model=model,
            max_output_tokens=requested_max_tokens,
        )
    except Exception as exc:
        logger.exception("gemini_proxy: upstream error for client=%s", cfg.name)
        audit_log(
            _audit_record(
                client_name=cfg.name,
                client_ip=client_ip,
                prompt=prompt,
                on_topic=None,
                status="upstream_error",
                started=started,
                extra={"error": str(exc)},
            ),
            audit_path,
        )
        return web.json_response({"error": "upstream gemini error"}, status=502)

    on_topic = not is_refusal(result.text)
    budget.consume(cfg.name, result.input_tokens + result.output_tokens)

    audit_log(
        _audit_record(
            client_name=cfg.name,
            client_ip=client_ip,
            prompt=prompt,
            on_topic=on_topic,
            status="ok",
            started=started,
            extra={
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "finish_reason": result.finish_reason,
                "model": model,
                "strict_mode": cfg.strict_mode,
            },
        ),
        audit_path,
    )

    return web.json_response(
        {
            "answer": result.text,
            "on_topic": on_topic,
            "finish_reason": result.finish_reason,
            "tokens": {"input": result.input_tokens, "output": result.output_tokens},
            "client": cfg.name,
        }
    )


async def handle_gemini_ping(request: web.Request) -> web.Response:
    """GET /api/gemini/ping — quick auth + budget check, no Gemini call."""
    if os.environ.get("GEMINI_PROXY_DISABLED") == "1":
        return web.json_response({"error": "endpoint disabled"}, status=503)
    if not os.environ.get("GEMINI_API_KEY"):
        return web.json_response({"error": "server missing GEMINI_API_KEY"}, status=503)

    state = _get_state(request.app)
    registry: Registry = state["registry"]
    budget: DailyBudget = state["budget"]
    registry.maybe_reload()

    if is_lan_admin(request, os.environ.get("GEMINI_PROXY_ADMIN_KEY")):
        return web.json_response({"ok": True, "client": "lan-admin", "remaining_tokens_today": -1})

    client_key = extract_client_key(request)
    if not client_key:
        return web.json_response({"error": "missing client key"}, status=401)
    cfg = registry.lookup(client_key)
    if cfg is None:
        return web.json_response({"error": "unknown client key"}, status=401)
    if not cfg.enabled:
        return web.json_response({"error": "client disabled"}, status=403)
    return web.json_response(
        {
            "ok": True,
            "client": cfg.name,
            "topic": cfg.topic_label,
            "remaining_tokens_today": budget.remaining(cfg.name, cfg.daily_token_budget),
        }
    )


# ---------------------------------------------------------------------------
# Audit record helper
# ---------------------------------------------------------------------------

def _audit_record(
    *,
    client_name: str,
    client_ip: str,
    prompt: str,
    on_topic: Optional[bool],
    status: str,
    started: float,
    extra: Optional[dict] = None,
) -> dict:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "client": client_name,
        "ip": client_ip,
        "status": status,
        "on_topic": on_topic,
        "prompt_len": len(prompt),
        "prompt_preview": prompt[:80],
        "latency_ms": int((time.monotonic() - started) * 1000),
    }
    if extra:
        rec.update(extra)
    return rec
