"""API Key authentication and OAuth2 with scopes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import (
    APIKeyHeader,
    OAuth2PasswordBearer,
    SecurityScopes,
)
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import settings

# ── API Key Auth ───────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Annotated[str | None, Depends(api_key_header)],
) -> str:
    if api_key and api_key == settings.api_key_secret:
        return api_key
    return ""


# ── OAuth2 / JWT Auth ─────────────────────────────────────────────────────────

SCOPES = {
    "trades:read": "Read order history",
    "trades:write": "Submit and cancel orders",
    "market:read": "Read quotes and market data",
    "account:read": "Read positions and balances",
}

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    scopes=SCOPES,
    auto_error=False,
)


class TokenData(BaseModel):
    sub: str
    scopes: list[str] = []


def create_access_token(subject: str, scopes: list[str]) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "scopes": scopes, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> TokenData:
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    return TokenData(sub=payload.get("sub", ""), scopes=payload.get("scopes", []))


# ── Unified dependency: API key OR OAuth2 token ───────────────────────────────

async def require_auth(
    request: Request,
    security_scopes: SecurityScopes,
    api_key: Annotated[str, Depends(verify_api_key)],
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> TokenData:
    """Accept LAN trust, API key, or JWT."""

    # LAN trusted clients skip auth
    if getattr(request.state, "lan_trusted", False):
        return TokenData(sub="lan-user", scopes=list(SCOPES.keys()))

    # API key grants full access
    if api_key:
        return TokenData(sub="api-key-user", scopes=list(SCOPES.keys()))

    # Otherwise require a valid JWT
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key or Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        token_data = decode_access_token(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {scope}",
            )

    return token_data
