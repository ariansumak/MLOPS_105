#!/usr/bin/env python
from __future__ import annotations

import os
import sys


def main() -> None:
    """Run Django administrative tasks."""

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dtu_frontend.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Django is not installed. Install dependencies before running.") from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
