"""
resolver_ui/auth.py
--------------------
JWT authentication for LAW.ai.

Simple hardcoded users to start — replace with database later.
Uses python-jose for JWT, passlib for password hashing.

Usage:
    from resolver_ui.auth import require_auth, router as auth_router

    # Add router to app
    app.include_router(auth_router)

    # Protect an endpoint
    @app.get("/cases")
    async def list_cases(user=Depends(require_auth)):
        ...
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────

# Secret key — set JWT_SECRET in environment, never hardcode in production
JWT_SECRET    = os.environ.get("JWT_SECRET", "verdictra-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

# ── Hardcoded users — replace with database when scaling ──────────────────────
# Passwords are bcrypt hashed
# To generate a hash: python -c "from passlib.context import CryptContext; print(CryptContext(['bcrypt']).hash('yourpassword'))"

pwd_context = None # Not used - plain comparison below

# Add your lawyer users here
# Format: "username": {"password_hash": "...", "name": "Display Name", "role": "lawyer"}
USERS = {
    "admin": {
        "password":  os.environ.get("ADMIN_PASSWORD", "verdictra2024"),
        "name":      "Administrator",
        "role":      "admin",
    },
    "lawyer1": {
        "password":  os.environ.get("LAWYER1_PASSWORD", "lawyer2024"),
        "name":      "Lawyer 1",
        "role":      "lawyer",
    },
    "Shlok":{
        "password":  os.environ.get("LAWYER2_PASSWORD", "lawyer2024"),
        "name":      "Shlok",
        "role":      "lawyer",
    },
    "Tapasya":{
        "password":  os.environ.get("LAWYER3_PASSWORD", "lawyer2024"),
        "name":      "Tapasya",
        "role":      "lawyer",
    },
    "Aditya":{
        "password":  os.environ.get("LAWYER4_PASSWORD", "lawyer2024"),
        "name":      "Aditya",
        "role":      "lawyer",
    }
}


# ── Models ─────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    name:         str
    role:         str


class UserInfo(BaseModel):
    username: str
    name:     str
    role:     str


# ── JWT helpers ────────────────────────────────────────────────────────────────

def create_token(username: str, name: str, role: str) -> str:
    """Create a JWT token for a user."""
    expire  = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub":  username,
        "name": name,
        "role": role,
        "exp":  expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[UserInfo]:
    """Verify JWT token and return user info. Returns None if invalid."""
    try:
        payload  = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        name     = payload.get("name")
        role     = payload.get("role")
        if not username:
            return None
        return UserInfo(username=username, name=name, role=role)
    except JWTError:
        return None


# ── Auth dependency ────────────────────────────────────────────────────────────

security = HTTPBearer()


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInfo:
    """
    FastAPI dependency — use with Depends(require_auth) on any endpoint.
    Returns UserInfo if valid, raises 401 if not.
    """
    user = verify_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ── Auth router ────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Login endpoint. Returns JWT token on success.
    Frontend stores token in localStorage and sends as Bearer header.
    """
    user = USERS.get(request.username)

    if not user or user["password"] != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_token(
        username=request.username,
        name=user["name"],
        role=user["role"],
    )

    return TokenResponse(
        access_token=token,
        name=user["name"],
        role=user["role"],
    )


@router.get("/me", response_model=UserInfo)
async def me(user: UserInfo = Depends(require_auth)):
    """Return current user info. Used by frontend to verify token on load."""
    return user


@router.post("/logout")
async def logout():
    """
    Logout — frontend just deletes the token from localStorage.
    No server-side session to invalidate with stateless JWT.
    """
    return {"status": "logged out"}