"""
resolver_ui/auth.py
--------------------
JWT authentication for Verdictra.

Drop-in replacement for the hardcoded USERS dict version.
Keeps identical public interface — require_auth, router, UserInfo,
/auth/login with JSON body, HTTPBearer — so app.py needs zero changes
beyond the two lines added in on_startup (startup_auth) and the upload
endpoint (check_case_limit).

New endpoints added:
  POST /auth/signup
  GET  /auth/verify
  POST /auth/forgot-password
  POST /auth/reset-password
  POST /auth/welcome-dismissed
  GET  /auth/me  (extended response)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────────

JWT_SECRET       = os.environ.get("JWT_SECRET", "verdictra-dev-secret-change-in-production")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = 24

# ── Legacy USERS dict — kept as fallback and for existing pilot users ───────────

USERS = {
    "admin": {
        "password": os.environ.get("ADMIN_PASSWORD", "verdictra2024"),
        "name":     "Administrator",
        "role":     "admin",
    },
    "lawyer1": {
        "password": os.environ.get("LAWYER1_PASSWORD", "lawyer2024"),
        "name":     "Lawyer 1",
        "role":     "lawyer",
    },
    "Shlok": {
        "password": os.environ.get("LAWYER2_PASSWORD", "lawyer2024"),
        "name":     "Shlok",
        "role":     "lawyer",
    },
    "Tapasya": {
        "password": os.environ.get("LAWYER3_PASSWORD", "lawyer2024"),
        "name":     "Tapasya",
        "role":     "lawyer",
    },
    "Aditya": {
        "password": os.environ.get("LAWYER4_PASSWORD", "lawyer2024"),
        "name":     "Aditya",
        "role":     "lawyer",
    },
}

# ── Models ──────────────────────────────────────────────────────────────────────

class UserInfo(BaseModel):
    """
    Returned by require_auth. Extended with optional fields —
    all existing code using .username / .name / .role works unchanged.
    """
    username:    str
    name:        str
    role:        str
    plan:        str = "free"
    user_id:     Optional[int] = None
    first_login: bool = False


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    name:         str
    role:         str
    plan:         str = "free"
    first_login:  bool = False


class SignupRequest(BaseModel):
    username:     str
    email:        str
    name:         str
    password:     str
    firm_name:    Optional[str] = None
    account_type: str = "individual"


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token:        str
    new_password: str


# ── JWT helpers ─────────────────────────────────────────────────────────────────

def create_token(
    username: str,
    name: str,
    role: str,
    plan: str = "free",
    user_id: Optional[int] = None,
    first_login: bool = False,
) -> str:
    expire  = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub":         username,
        "name":        name,
        "role":        role,
        "plan":        plan,
        "user_id":     user_id,
        "first_login": first_login,
        "exp":         expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[UserInfo]:
    try:
        payload  = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        name     = payload.get("name")
        role     = payload.get("role")
        if not username:
            return None
        return UserInfo(
            username    = username,
            name        = name or "",
            role        = role or "lawyer",
            plan        = payload.get("plan", "free"),
            user_id     = payload.get("user_id"),
            first_login = payload.get("first_login", False),
        )
    except JWTError:
        return None


# ── Auth dependency ─────────────────────────────────────────────────────────────

security = HTTPBearer()


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInfo:
    if os.environ.get("ENVIRONMENT") == "development":
        return UserInfo(username="dev", name="Developer", role="admin")
    user = verify_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ── Plan limit enforcement ──────────────────────────────────────────────────────

PLAN_CASE_LIMITS: dict[str, Optional[int]] = {
    "free":         2,
    "pro":          10,
    "max":          50,
    "firm_small":   50,
    "firm_mid":     100,
    "firm_large":   None,
}


def check_case_limit(user: UserInfo, current_case_count: int) -> None:
    """
    Raises HTTP 403 with upgrade message if user is at their case limit.
    Call inside the upload endpoint before creating a new case:

        if user.role != "admin":
            user_cases = [c for c in all_cases() if c.get("created_by") == user.username]
            check_case_limit(user, len(user_cases))
    """
    limit = PLAN_CASE_LIMITS.get(user.plan, 2)
    if limit is None:
        return  # unlimited plan
    if current_case_count >= limit:
        plan_label = {
            "free": "Free (2 cases)",
            "pro":  "Pro (10 cases)",
            "max":  "Max (50 cases)",
        }.get(user.plan, user.plan)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Case limit reached for your {plan_label} plan "
                f"({current_case_count}/{limit}). "
                "To upload more cases, upgrade your plan — "
                "email support@verdictra.ai."
            ),
        )


# ── Startup ─────────────────────────────────────────────────────────────────────

def startup_auth() -> None:
    """
    Call once in app.py on_startup.
    Initialises SQLite DB and imports existing env-var users.
    Safe to call multiple times — skips existing usernames.
    """
    try:
        from resolver_ui.user_db import init_db, migrate_env_users
        init_db()
        migrate_env_users()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            f"[auth] startup_auth: user_db not available ({exc}), "
            "falling back to USERS dict only."
        )


# ── Login helper: SQLite first, USERS dict fallback ────────────────────────────

def _authenticate(username: str, password: str) -> Optional[dict]:
    """
    Returns a plain dict with: username, name, role, plan, id, first_login.
    Tries SQLite first, falls back to hardcoded USERS dict.
    """
    # Try SQLite
    try:
        from resolver_ui.user_db import get_user_by_username, verify_password, update_last_login
        db_user = get_user_by_username(username)
        if db_user:
            if not verify_password(password, db_user["password_hash"]):
                return None
            if not db_user["is_verified"]:
                raise HTTPException(
                    status_code=403,
                    detail="Please verify your email address before logging in. "
                           "Check your inbox for the verification link.",
                )
            if not db_user["is_active"]:
                raise HTTPException(
                    status_code=403,
                    detail="Your account has been suspended. Contact support@verdictra.ai.",
                )
            update_last_login(db_user["id"])
            return {
                "username":    db_user["username"],
                "name":        db_user["name"],
                "role":        db_user["role"],
                "plan":        db_user.get("plan", "free"),
                "id":          db_user["id"],
                "first_login": bool(db_user.get("first_login", 0)),
            }
    except HTTPException:
        raise
    except Exception:
        pass

    # Fall back to USERS dict (legacy pilot users / dev)
    legacy = USERS.get(username)
    if legacy and legacy["password"] == password:
        return {
            "username":    username,
            "name":        legacy["name"],
            "role":        legacy["role"],
            "plan":        "free",
            "id":          None,
            "first_login": False,
        }
    return None


# ── Auth router ─────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Login endpoint. Returns JWT token on success.
    Frontend stores token in localStorage and sends as Bearer header.
    """
    user = _authenticate(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_token(
        username    = user["username"],
        name        = user["name"],
        role        = user["role"],
        plan        = user.get("plan", "free"),
        user_id     = user.get("id"),
        first_login = user.get("first_login", False),
    )
    return TokenResponse(
        access_token = token,
        name         = user["name"],
        role         = user["role"],
        plan         = user.get("plan", "free"),
        first_login  = user.get("first_login", False),
    )


@router.get("/me")
async def me(user: UserInfo = Depends(require_auth)):
    """Return current user info. Extended with profile data if available."""
    profile = {
        "username":    user.username,
        "name":        user.name,
        "role":        user.role,
        "plan":        user.plan,
        "first_login": user.first_login,
    }
    if user.user_id:
        try:
            from resolver_ui.user_db import get_user_by_id
            db = get_user_by_id(user.user_id)
            if db:
                profile.update({
                    "email":        db.get("email", ""),
                    "firm_name":    db.get("firm_name"),
                    "account_type": db.get("account_type", "individual"),
                    "created_at":   db.get("created_at", ""),
                })
        except Exception:
            pass
    return profile


@router.post("/logout")
async def logout():
    """Frontend deletes the token from localStorage — no server session to clear."""
    return {"status": "logged out"}


# ── Self-service signup ─────────────────────────────────────────────────────────

@router.post("/signup", status_code=201)
async def signup(req: SignupRequest):
    try:
        from resolver_ui.user_db import create_user, create_email_token
        from resolver_ui.email_service import (
            send_verification_email,
            send_admin_new_user_notification,
        )
    except ImportError:
        raise HTTPException(503, "Signup unavailable — database not initialised.")

    try:
        user = create_user(
            username     = req.username.strip().lower(),
            email        = req.email.strip().lower(),
            name         = req.name.strip(),
            password     = req.password,
            firm_name    = req.firm_name,
            account_type = req.account_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    token = create_email_token(user["id"], "verify")
    send_verification_email(req.email, req.name, token)

    admin_email = os.environ.get("ADMIN_EMAIL", "")
    if admin_email:
        send_admin_new_user_notification(
            admin_email    = admin_email,
            new_user_name  = req.name,
            new_user_email = req.email,
            account_type   = req.account_type,
            firm_name      = req.firm_name,
        )
    return {"message": "Account created. Check your email to verify before logging in."}


@router.get("/verify")
async def verify_email(token: str):
    try:
        from resolver_ui.user_db import consume_email_token
        from resolver_ui.email_service import send_welcome_email
    except ImportError:
        raise HTTPException(503, "Verification unavailable.")

    user = consume_email_token(token, "verify")
    if not user:
        raise HTTPException(
            400,
            "Verification link is invalid or has expired. "
            "Please sign up again or contact support@verdictra.ai.",
        )
    send_welcome_email(user["email"], user["name"])
    return {"message": "Email verified. You can now log in.", "verified": True}


@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    # Always return 200 — never reveal whether an email exists
    try:
        from resolver_ui.user_db import get_user_by_email, create_email_token
        from resolver_ui.email_service import send_password_reset_email
        user = get_user_by_email(req.email.strip().lower())
        if user and user["is_active"]:
            token = create_email_token(user["id"], "reset")
            send_password_reset_email(req.email, user["name"], token)
    except Exception:
        pass
    return {"message": "If an account with that email exists, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest):
    if len(req.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    try:
        from resolver_ui.user_db import consume_email_token, set_password
    except ImportError:
        raise HTTPException(503, "Password reset unavailable.")

    user = consume_email_token(req.token, "reset")
    if not user:
        raise HTTPException(
            400,
            "Reset link is invalid or has expired. Please request a new one.",
        )
    set_password(user["id"], req.new_password)
    return {"message": "Password updated. You can now log in with your new password."}


@router.post("/welcome-dismissed")
async def welcome_dismissed(user: UserInfo = Depends(require_auth)):
    if user.user_id:
        try:
            from resolver_ui.user_db import clear_first_login
            clear_first_login(user.user_id)
        except Exception:
            pass
    return {"ok": True}