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

A dynamic, multi-turn OpenEnv environment for realistic email triage tasks with **3 difficulty-graded scenarios**.

## Tasks

| ID | Name | Emails | Max Steps | Dynamic Events |
|----|------|--------|-----------|----------------|
| `easy` | Quick Sort | 3 | 6 | ❌ |
| `medium` | Priority Triage | 5 | 10 | ❌ |
| `hard` | Dynamic Crisis | 7–10 | 12 | ✅ |

## What This Environment Tests

- Prioritization of high-value vs low-value emails
- Correct archiving decisions (spam, newsletters, notifications)
- Professional draft generation for client/escalation threads
- Calendar-aware scheduling behavior
- Adaptation to mid-episode state changes (hard mode)

## Action Schema

EmailtriageAction fields:

- `action_type`: one of `read`, `archive`, `query_calendar`, `draft_email`
- `target_email_id`: target email for read/archive/draft_email
- `draft_content`: response text for draft_email
- `proposed_slot`: calendar slot for scheduling drafts

## Observation Schema

EmailtriageObservation includes:

- `inbox_preview` — up to 5 unread email summaries
- `returned_emails` — full text from read actions
- `calendar_slots` — available scheduling slots
- `last_action_result` — grader feedback
- `conversation_history` — recent action trace
- `inbox_remaining` — unread count
- `reward` — step reward in [0, 1]
- `done` — episode completion flag
- `metadata` — episode_id, step, task_id, progress stats

## Reward Philosophy

Continuous [0, 1] rewards with partial credit:

- Correct archiving: 0.62–0.80
- Reading (priority-weighted): 0.09–0.25
- Calendar queries: 0.10–0.46
- Drafts: multi-factor (appropriateness + quality + calendar + slot + urgency)
- Progress bonus: +0.12 per processed email
- Completion bonus: +0.10 for full inbox clearance

## Dynamic Episode Behavior (Hard Mode)

- New urgent email arrives at step 3
- Calendar slot removed at step 4+
- Forces adaptation and multi-step recovery

## Quick Start

### Use from Python via Docker image

```python
from EmailTriage import EmailtriageAction, EmailtriageEnv

with EmailtriageEnv.from_docker_image("emailtriage-env:latest") as env:
    # Easy task
    result = env.reset(options={"task_id": "easy"})
    action = EmailtriageAction(
        action_type="archive",
        target_email_id=101,
    )
    result = env.step(action)
    print(result.reward, result.done)
```

### Connect to a running server

```python
from EmailTriage import EmailtriageAction, EmailtriageEnv

with EmailtriageEnv(base_url="http://localhost:8000") as env:
    result = env.reset(options={"task_id": "medium"})
    result = env.step(
        EmailtriageAction(
            action_type="read",
            target_email_id=101,
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
openenv push --repo-id YOUR_USERNAME/EmailTriage
```

## Endpoints

- Web UI: `/web`
- API Docs: `/docs`
- Health: `/health`
- WebSocket: `/ws`
- Metadata: `/metadata`

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
