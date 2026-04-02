ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app
COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh ; \
    mv /root/.local/bin/uv /usr/local/bin/uv ; \
    mv /root/.local/bin/uvx /usr/local/bin/uvx ; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f EmailTriage/uv.lock ]; then \
    cd EmailTriage && uv sync --frozen --no-editable ; \
    else \
    cd EmailTriage && uv sync --no-editable ; \
    fi

FROM ${BASE_IMAGE}

WORKDIR /app
COPY --from=builder /app/env /app/env
COPY --from=builder /app/env/EmailTriage/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV env_enable_web_interface=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn EmailTriage.server.app:app --host 0.0.0.0 --port 8000"]
