"""
resolver_ui/user_db.py
----------------------
SQLite-backed user management. Replaces the hardcoded USERS dict in auth.py.

Schema
------
users
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  username        TEXT UNIQUE NOT NULL
  email           TEXT UNIQUE NOT NULL
  name            TEXT NOT NULL
  firm_name       TEXT
  account_type    TEXT NOT NULL DEFAULT 'individual'   -- 'individual' | 'firm'
  password_hash   TEXT NOT NULL                        -- argon2 hash (fallback: plain for dev)
  role            TEXT NOT NULL DEFAULT 'lawyer'        -- 'lawyer' | 'admin'
  plan            TEXT NOT NULL DEFAULT 'free'          -- 'free' | 'starter' | 'professional' | 'firm_small' | 'firm_mid' | 'firm_large'
  is_verified     INTEGER NOT NULL DEFAULT 0            -- 0 = unverified, 1 = verified
  is_active       INTEGER NOT NULL DEFAULT 1            -- 0 = suspended by admin
  first_login     INTEGER NOT NULL DEFAULT 1            -- 1 = show welcome screen
  created_at      TEXT NOT NULL
  last_login      TEXT

email_tokens
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
  token           TEXT UNIQUE NOT NULL
  token_type      TEXT NOT NULL   -- 'verify' | 'reset'
  expires_at      TEXT NOT NULL
  used            INTEGER NOT NULL DEFAULT 0
"""

import os
import sqlite3
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get("USER_DB_PATH", "/opt/verdictra/data/users.db")

PLAN_CASE_LIMITS = {
    "free":         2,
    "starter":      10,
    "professional": 50,
    "firm_small":   50,
    "firm_mid":     100,
    "firm_large":   None,   # unlimited
}

TOKEN_EXPIRY_HOURS = {
    "verify": 48,
    "reset":  1,
}


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------
def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT UNIQUE NOT NULL,
                email        TEXT UNIQUE NOT NULL,
                name         TEXT NOT NULL,
                firm_name    TEXT,
                account_type TEXT NOT NULL DEFAULT 'individual',
                password_hash TEXT NOT NULL,
                role         TEXT NOT NULL DEFAULT 'lawyer',
                plan         TEXT NOT NULL DEFAULT 'free',
                is_verified  INTEGER NOT NULL DEFAULT 0,
                is_active    INTEGER NOT NULL DEFAULT 1,
                first_login  INTEGER NOT NULL DEFAULT 1,
                created_at   TEXT NOT NULL,
                last_login   TEXT
            );

            CREATE TABLE IF NOT EXISTS email_tokens (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token      TEXT UNIQUE NOT NULL,
                token_type TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0
            );
        """)


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------
def _hash_password(password: str) -> str:
    """
    Use argon2-cffi when available (production).
    Falls back to SHA-256 + salt for dev environments without argon2.
    Replace this entirely with argon2 before launch.
    """
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        return ph.hash(password)
    except ImportError:
        # Dev fallback — NOT for production
        salt = secrets.token_hex(16)
        digest = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
        return f"sha256:{salt}:{digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    if stored_hash.startswith("sha256:"):
        _, salt, digest = stored_hash.split(":", 2)
        return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest() == digest
    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError
        ph = PasswordHasher()
        try:
            return ph.verify(stored_hash, password)
        except VerifyMismatchError:
            return False
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------
def create_user(
    username: str,
    email: str,
    name: str,
    password: str,
    firm_name: Optional[str] = None,
    account_type: str = "individual",
    role: str = "lawyer",
    plan: str = "free",
) -> dict:
    """
    Creates an unverified user. Returns the new user row as a dict.
    Raises ValueError if username or email already exists.
    """
    init_db()
    password_hash = _hash_password(password)
    now = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        try:
            conn.execute(
                """INSERT INTO users
                   (username, email, name, firm_name, account_type,
                    password_hash, role, plan, is_verified, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (username, email, name, firm_name, account_type,
                 password_hash, role, plan, now),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            if "username" in str(exc):
                raise ValueError("Username already taken.")
            if "email" in str(exc):
                raise ValueError("An account with this email already exists.")
            raise

    return get_user_by_email(email)


def get_user_by_username(username: str) -> Optional[dict]:
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def update_last_login(user_id: int) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), user_id),
        )
        conn.commit()


def clear_first_login(user_id: int) -> None:
    """Call after the welcome screen is dismissed."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET first_login = 0 WHERE id = ?", (user_id,)
        )
        conn.commit()


def set_password(user_id: int, new_password: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (_hash_password(new_password), user_id),
        )
        conn.commit()


def set_plan(user_id: int, plan: str) -> None:
    """Admin-only: manually set a user's plan."""
    if plan not in PLAN_CASE_LIMITS:
        raise ValueError(f"Unknown plan: {plan}")
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET plan = ? WHERE id = ?", (plan, user_id)
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Email tokens
# ---------------------------------------------------------------------------
def create_email_token(user_id: int, token_type: str) -> str:
    """
    Creates a secure random token, stores it, returns the raw token string.
    Any previous unused tokens of the same type for this user are invalidated.
    """
    assert token_type in TOKEN_EXPIRY_HOURS, f"Unknown token type: {token_type}"
    token = secrets.token_urlsafe(32)
    expiry_hours = TOKEN_EXPIRY_HOURS[token_type]
    expires_at = (
        datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
    ).isoformat()

    with _get_conn() as conn:
        # Invalidate old unused tokens of same type for this user
        conn.execute(
            "UPDATE email_tokens SET used = 1 "
            "WHERE user_id = ? AND token_type = ? AND used = 0",
            (user_id, token_type),
        )
        conn.execute(
            "INSERT INTO email_tokens (user_id, token, token_type, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, token, token_type, expires_at),
        )
        conn.commit()

    return token


def consume_email_token(token: str, token_type: str) -> Optional[dict]:
    """
    Validates and consumes a token. Returns the associated user dict on success.
    Returns None if invalid, expired, already used, or wrong type.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        row = conn.execute(
            """SELECT et.*, u.id as user_id
               FROM email_tokens et
               JOIN users u ON u.id = et.user_id
               WHERE et.token = ?
                 AND et.token_type = ?
                 AND et.used = 0
                 AND et.expires_at > ?""",
            (token, token_type, now),
        ).fetchone()

        if not row:
            return None

        # Mark as used
        conn.execute(
            "UPDATE email_tokens SET used = 1 WHERE token = ?", (token,)
        )

        # If it's a verify token, mark the user as verified
        if token_type == "verify":
            conn.execute(
                "UPDATE users SET is_verified = 1 WHERE id = ?", (row["user_id"],)
            )

        conn.commit()

    return get_user_by_id(row["user_id"])


# ---------------------------------------------------------------------------
# Plan enforcement helper
# ---------------------------------------------------------------------------
def get_case_limit(plan: str) -> Optional[int]:
    """Returns None for unlimited, int for capped plans."""
    return PLAN_CASE_LIMITS.get(plan, 2)


# ---------------------------------------------------------------------------
# Migration helper: import hardcoded USERS from env
# ---------------------------------------------------------------------------
def migrate_env_users() -> None:
    """
    One-time migration: reads the old env-var-backed USERS dict and inserts
    them as verified admin/lawyer users if they don't already exist.
    Safe to run multiple times (skips existing usernames).

    Old env vars: ADMIN_PASSWORD, LAWYER1_PASSWORD, LAWYER2_PASSWORD,
                  LAWYER3_PASSWORD, LAWYER4_PASSWORD
    """
    init_db()
    legacy = [
        ("admin",   os.environ.get("ADMIN_PASSWORD", ""),   "admin",  "Admin User",     "admin"),
        ("lawyer1", os.environ.get("LAWYER1_PASSWORD", ""), "lawyer", "Lawyer 1",       "max"),
        ("shlok",   os.environ.get("LAWYER2_PASSWORD", ""), "lawyer", "Shlok",          "max"),
        ("tapasya", os.environ.get("LAWYER3_PASSWORD", ""), "lawyer", "Tapasya",        "max"),
        ("aditya",  os.environ.get("LAWYER4_PASSWORD", ""), "lawyer", "Aditya",         "max"),
    ]

    for username, password, role, name, plan in legacy:
        if not password:
            continue
        existing = get_user_by_username(username)
        if existing:
            continue
        try:
            user = create_user(
                username=username,
                email=f"{username}@verdictra.internal",
                name=name,
                password=password,
                role=role,
                plan=plan,
            )
            # Mark migrated users as verified + not first_login
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE users SET is_verified = 1, first_login = 0 WHERE id = ?",
                    (user["id"],),
                )
                conn.commit()
            print(f"[user_db] Migrated legacy user: {username}")
        except ValueError as e:
            print(f"[user_db] Skipped {username}: {e}")