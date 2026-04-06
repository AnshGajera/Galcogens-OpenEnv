# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Emailtriage Environment.

Endpoints:
    - POST /reset: Reset the environment (accepts optional task_id)
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

import argparse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import APIRouter

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import EmailtriageAction, EmailtriageObservation
    from .EmailTriage_environment import EmailtriageEnvironment
except ImportError:
    from models import EmailtriageAction, EmailtriageObservation
    from server.EmailTriage_environment import EmailtriageEnvironment


# Create the app with web interface and README integration
app = create_app(
    EmailtriageEnvironment,
    EmailtriageAction,
    EmailtriageObservation,
    env_name="EmailTriage",
    max_concurrent_envs=1,
)

# Keep docs metadata explicit to avoid generic duplicate naming in Swagger UI.
app.title = "EmailTriage Environment API"
app.description = (
    "HTTP API for the dynamic EmailTriage OpenEnv environment "
    "with 3 difficulty-graded tasks (easy, medium, hard)."
)


def _load_readme_content() -> str:
    """Load environment README markdown for metadata endpoint."""
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    try:
        return readme_path.read_text(encoding="utf-8")
    except Exception:
        return "EmailTriage environment documentation is unavailable."


def _replace_route(
    path: str,
    method: str,
    endpoint,
    *,
    summary: str,
) -> None:
    """Replace an existing route path+method with a custom handler."""
    method_upper = method.upper()
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) == path
            and method_upper in getattr(route, "methods", set())
        )
    ]

    router = APIRouter()
    router.add_api_route(
        path,
        endpoint,
        methods=[method_upper],
        summary=summary,
    )
    app.include_router(router)


def _metadata_payload() -> dict[str, Any]:
    """Build non-null metadata for /metadata."""
    return {
        "name": "EmailTriage",
        "description": (
            "Dynamic multi-turn email triage environment for OpenEnv "
            "post-training and evaluation with 3 difficulty-graded tasks"
        ),
        "readme_content": _load_readme_content(),
        "version": "1.0.0",
        "author": "Galcogens",
        "documentation_url": "/docs",
        "tasks": [
            {
                "id": "easy",
                "name": "Quick Sort",
                "description": "Archive 3 spam/newsletter emails",
                "difficulty": "easy",
            },
            {
                "id": "medium",
                "name": "Priority Triage",
                "description": (
                    "Triage 5 mixed-priority emails with "
                    "calendar scheduling"
                ),
                "difficulty": "medium",
            },
            {
                "id": "hard",
                "name": "Dynamic Crisis",
                "description": (
                    "Handle 7-10 emails with dynamic events "
                    "and escalations"
                ),
                "difficulty": "hard",
            },
        ],
    }


def _root_payload() -> dict[str, Any]:
    """Return a simple JSON index for localhost:8000."""
    return {
        "service": "EmailTriage OpenEnv API",
        "status": "ok",
        "docs": "/docs",
        "metadata": "/metadata",
        "endpoints": [
            "/reset",
            "/step",
            "/state",
            "/schema",
            "/metadata",
            "/health",
            "/web",
            "/ws",
        ],
    }


_replace_route(
    "/metadata",
    "GET",
    _metadata_payload,
    summary="Get environment metadata",
)
_replace_route(
    "/",
    "GET",
    _root_payload,
    summary="Get API index",
)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution via uv run or python -m."""
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """CLI-compatible entrypoint expected by OpenEnv validator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
