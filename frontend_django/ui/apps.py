from __future__ import annotations

from django.apps import AppConfig


class UiConfig(AppConfig):
    """Configuration for the UI app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "ui"
