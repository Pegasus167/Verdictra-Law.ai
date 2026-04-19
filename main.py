"""
main.py
--------
Uvicorn entry point for LAW.ai FastAPI backend.

Run with:
    poetry run uvicorn main:app --reload
    poetry run uvicorn main:app --reload --port 8000

The app lives in resolver_ui/app.py.
This file just re-exports it so uvicorn can find it.
"""

from resolver_ui.app import app  # noqa: F401

# That's it — uvicorn main:app --reload will find the FastAPI app here.