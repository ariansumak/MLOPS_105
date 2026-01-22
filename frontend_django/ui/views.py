from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def index(request: HttpRequest) -> HttpResponse:
    """Render the inference tester UI.

    Args:
        request: Incoming HTTP request.

    Returns:
        Rendered HTML response.
    """

    return render(request, "ui/index.html")
