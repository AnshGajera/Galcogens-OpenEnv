---
title: EmailTriage Environment Server
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - email-triage
---

# EmailTriage Environment

A dynamic, multi-turn OpenEnv environment for realistic email triage tasks.

This environment is designed for post-training and RL evaluation workflows where an agent must adapt to changing inbox and scheduling conditions over multiple steps.

## What This Environment Tests

- Prioritization of high-value vs low-value emails
- Correct archiving decisions
- Professional draft generation for client/escalation threads
- Calendar-aware scheduling behavior
- Adaptation to mid-episode state changes

## Action Schema

EmailtriageAction fields:

- action_type: one of read, archive, query_calendar, draft_email
- target_email_id: target email for read/archive/draft_email
- draft_content: response text for draft_email
- proposed_slot: calendar slot for scheduling drafts

## Observation Schema

EmailtriageObservation includes:

- inbox_preview
- returned_emails
- calendar_slots
- last_action_result
- conversation_history
- inbox_remaining
- reward
- done
- metadata

## Reward Philosophy

The grader returns diverse, continuous values in [0, 1] (not simple binary pass/fail):

- Partial reward for useful intermediate steps
- Higher reward for correct business decisions
- Graded quality signals for drafted replies
- Extra reward for measurable progress and episode completion

## Dynamic Episode Behavior

The environment is non-static:

- New urgent email can arrive mid-episode
- Calendar availability can change while task is in progress

Agents must adapt and recover over multiple steps.

## Quick Start

### Use from Python via Docker image

```python
from EmailTriage import EmailtriageAction, EmailtriageEnv

with EmailtriageEnv.from_docker_image("emailtriage-env:latest") as env:
    result = env.reset()

    action = EmailtriageAction(
        action_type="query_calendar",
        target_email_id=-1,
        draft_content="",
        proposed_slot="",
    )
    result = env.step(action)
    print(result.reward, result.done)
```

### Connect to a running server

```python
from EmailTriage import EmailtriageAction, EmailtriageEnv

with EmailtriageEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    result = env.step(
        EmailtriageAction(
            action_type="read",
            target_email_id=101,
            draft_content="",
            proposed_slot="",
        )
    )
```

## Local Run

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Validate

```bash
openenv validate
```

## Push to Hugging Face Spaces

```bash
openenv push --repo-id OMCHOKSI108/EmailTriage
```

After deployment, endpoints are available for:

- Web UI: /web
- API Docs: /docs
- Health: /health
- WebSocket: /ws

## Project Layout

```text
EmailTriage/
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
└── server/
    ├── app.py
    ├── EmailTriage_environment.py
    └── Dockerfile
```
