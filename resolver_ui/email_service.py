"""
resolver_ui/email_service.py
----------------------------
Transactional email via SendGrid free tier (100 emails/day).

Required env vars:
  SENDGRID_API_KEY   — from sendgrid.com/settings/api_keys
  FROM_EMAIL         — verified sender address (e.g. noreply@verdictra.ai)
  APP_BASE_URL       — public URL (e.g. https://verdictra.ai)
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
FROM_EMAIL       = os.environ.get("FROM_EMAIL", "noreply@verdictra.ai")
FROM_NAME        = "Verdictra"
APP_BASE_URL     = os.environ.get("APP_BASE_URL", "https://verdictra.ai").rstrip("/")


# ---------------------------------------------------------------------------
# Core send helper
# ---------------------------------------------------------------------------
def _send(to_email: str, subject: str, html_body: str) -> bool:
    """
    Sends an email via SendGrid API.
    Returns True on success, False on failure (logs the error).
    Falls back to console logging if SENDGRID_API_KEY is not set (dev mode).
    """
    if not SENDGRID_API_KEY:
        logger.warning(
            "[email_service DEV MODE] Would send to %s — %s\n%s",
            to_email, subject, html_body
        )
        return True

    try:
        import httpx
        payload = {
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {"email": FROM_EMAIL, "name": FROM_NAME},
            "subject": subject,
            "content": [{"type": "text/html", "value": html_body}],
        }
        resp = httpx.post(
            "https://api.sendgrid.com/v3/mail/send",
            json=payload,
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        if resp.status_code in (200, 202):
            return True
        logger.error("[email_service] SendGrid error %s: %s", resp.status_code, resp.text)
        return False
    except Exception as exc:
        logger.error("[email_service] Failed to send email: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------------
def _base_template(body_html: str) -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #fcf9f5; font-family: 'Georgia', serif; }}
    .wrapper {{ max-width: 560px; margin: 40px auto; background: #fff; border: 1px solid #e5e2de; border-radius: 8px; overflow: hidden; }}
    .header {{ background: #17191b; padding: 24px 32px; }}
    .header-logo {{ color: #fcf9f5; font-size: 20px; font-weight: 700; letter-spacing: -0.01em; }}
    .header-sub {{ color: #75777a; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; font-family: sans-serif; }}
    .body {{ padding: 40px 32px; }}
    .body p {{ font-family: sans-serif; font-size: 15px; line-height: 1.6; color: #44474a; margin-bottom: 16px; }}
    .btn {{ display: inline-block; background: #17191b; color: #fff !important; font-family: sans-serif; font-size: 14px; font-weight: 600; padding: 12px 28px; border-radius: 4px; text-decoration: none; margin: 8px 0 24px; }}
    .note {{ font-size: 13px !important; color: #75777a !important; }}
    .footer {{ padding: 20px 32px; border-top: 1px solid #e5e2de; font-family: sans-serif; font-size: 12px; color: #75777a; }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="header">
      <div class="header-logo">Verdictra</div>
      <div class="header-sub">Legal Intelligence</div>
    </div>
    <div class="body">
      {body_html}
    </div>
    <div class="footer">
      Built for Indian law. Grounded in evidence. Verified by you.<br>
      Verdictra · <a href="{APP_BASE_URL}/privacy.html" style="color:#75777a;">Privacy</a> · <a href="{APP_BASE_URL}/terms.html" style="color:#75777a;">Terms</a>
    </div>
  </div>
</body>
</html>
"""


def send_verification_email(to_email: str, name: str, token: str) -> bool:
    verify_url = f"{APP_BASE_URL}/app/verify?token={token}"
    body = f"""
      <p>Hi {name},</p>
      <p>Thanks for signing up for Verdictra. Click the button below to verify your email address and activate your account.</p>
      <a href="{verify_url}" class="btn">Verify my email</a>
      <p class="note">This link expires in 48 hours. If you didn't create a Verdictra account, you can safely ignore this email.</p>
      <p class="note">Or copy this URL into your browser:<br>{verify_url}</p>
    """
    return _send(
        to_email=to_email,
        subject="Verify your Verdictra account",
        html_body=_base_template(body),
    )


def send_password_reset_email(to_email: str, name: str, token: str) -> bool:
    reset_url = f"{APP_BASE_URL}/app/reset-password?token={token}"
    body = f"""
      <p>Hi {name},</p>
      <p>We received a request to reset the password for your Verdictra account. Click the button below to choose a new password.</p>
      <a href="{reset_url}" class="btn">Reset my password</a>
      <p class="note">This link expires in 1 hour. If you didn't request a password reset, your account is safe — you can ignore this email.</p>
      <p class="note">Or copy this URL into your browser:<br>{reset_url}</p>
    """
    return _send(
        to_email=to_email,
        subject="Reset your Verdictra password",
        html_body=_base_template(body),
    )


def send_admin_new_user_notification(
    admin_email: str,
    new_user_name: str,
    new_user_email: str,
    account_type: str,
    firm_name: Optional[str] = None,
) -> bool:
    firm_line = f"<p><strong>Firm:</strong> {firm_name}</p>" if firm_name else ""
    body = f"""
      <p>A new user has signed up and is pending activation:</p>
      <p><strong>Name:</strong> {new_user_name}</p>
      <p><strong>Email:</strong> {new_user_email}</p>
      <p><strong>Account type:</strong> {account_type}</p>
      {firm_line}
      <p>Log in to the admin panel to activate their account.</p>
      <a href="{APP_BASE_URL}/app/admin" class="btn">Open admin panel</a>
    """
    return _send(
        to_email=admin_email,
        subject=f"New Verdictra signup: {new_user_name}",
        html_body=_base_template(body),
    )


def send_welcome_email(to_email: str, name: str) -> bool:
    body = f"""
      <p>Hi {name},</p>
      <p>Your Verdictra account is active. You're ready to upload your first case.</p>
      <a href="{APP_BASE_URL}/app" class="btn">Open Verdictra</a>
      <p>A few things to know:</p>
      <p>· Upload any PDF, DOCX, or scanned document — Verdictra handles the rest.</p>
      <p>· Review the extracted entities before your first query.</p>
      <p>· Every answer includes the exact page number in your document.</p>
      <p class="note">Questions? Reply to this email or reach us on WhatsApp — we typically respond within the hour.</p>
    """
    return _send(
        to_email=to_email,
        subject="Your Verdictra account is ready",
        html_body=_base_template(body),
    )