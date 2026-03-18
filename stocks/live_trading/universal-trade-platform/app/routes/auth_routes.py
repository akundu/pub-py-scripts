"""POST /auth/token — OAuth2 token endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.auth import SCOPES, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    username: str
    password: str
    scopes: list[str] = []


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    scopes: list[str]


@router.post("/token", response_model=TokenResponse)
async def get_token(request: TokenRequest) -> TokenResponse:
    """Issue a JWT token. In production, validate against a user store.

    For demo purposes, any non-empty username/password is accepted.
    """
    if not request.username or not request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # Validate requested scopes
    granted = [s for s in request.scopes if s in SCOPES] or list(SCOPES.keys())

    token = create_access_token(subject=request.username, scopes=granted)
    return TokenResponse(access_token=token, scopes=granted)
