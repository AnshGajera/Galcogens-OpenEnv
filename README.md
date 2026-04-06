# EmailTriage OpenEnv — Hackathon Submission

A production-grade OpenEnv environment that simulates **real-world email triage** — the daily task of processing, prioritizing, and responding to a mixed work inbox. Built for the OpenEnv Hackathon with 3 difficulty-graded tasks, continuous partial rewards, and dynamic mid-episode events.

## Why Email Triage?

Email triage is a task professionals perform daily: scanning an inbox, deciding what to archive, what needs a reply, coordinating calendar availability, and handling urgent escalations. This makes it an ideal testbed for evaluating agent decision-making, prioritization, and multi-step planning under changing conditions.

## Tasks

The environment defines **3 tasks** with increasing difficulty:

| Task ID | Name | Emails | Max Steps | Dynamic Events | Description |
|---------|------|--------|-----------|----------------|-------------|
| `easy` | Quick Sort | 3 | 6 | ❌ | Archive 3 spam/newsletter emails. Tests basic categorization. |
| `medium` | Priority Triage | 5 | 10 | ❌ | Triage 5 mixed-priority emails with calendar scheduling. Tests reading, drafting, and archiving decisions. |
| `hard` | Dynamic Crisis | 7–10 | 12 | ✅ | Handle a full inbox with mid-episode urgent emails and calendar changes. Tests adaptation and escalation handling. |

Each task has a **programmatic grader** that scores agent performance on a continuous 0.0–1.0 scale based on action appropriateness, draft quality, calendar awareness, and progress.

## Action Space

The agent sends an `EmailtriageAction` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"read" \| "archive" \| "query_calendar" \| "draft_email"` | The tool/action to execute |
| `target_email_id` | `int` | Email ID to act on (-1 for query_calendar) |
| `draft_content` | `str` | Reply text for draft_email actions |
| `proposed_slot` | `str` | Calendar slot for scheduling drafts |

## Observation Space

After each step the agent receives an `EmailtriageObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `inbox_preview` | `List[Dict]` | Metadata for up to 5 unread emails (id, sender, subject, priority, status) |
| `returned_emails` | `List[str]` | Full email text from read actions |
| `calendar_slots` | `List[str]` | Available calendar slots |
| `last_action_result` | `str` | Grader feedback for the most recent action |
| `inbox_remaining` | `int` | Count of unread emails |
| `conversation_history` | `List[str]` | Recent action/feedback trace |
| `reward` | `float` | Step reward in [0, 1] |
| `done` | `bool` | Whether the episode has ended |

## Reward Function

Rewards are **continuous and partially informative** (not binary pass/fail):

- **Archive spam/newsletters**: 0.62–0.80 per correct archive
- **Read emails**: 0.09–0.25 depending on priority (higher for critical emails)
- **Query calendar**: 0.10–0.46 based on pending scheduling workload
- **Draft replies**: Multi-factor scoring based on:
  - Task appropriateness (is this email worth drafting?)
  - Draft quality (length, professionalism, keyword relevance)
  - Calendar awareness (did you check availability first?)
  - Valid proposed slot
  - Urgency handling for escalations
- **Progress bonus**: +0.12 for each email successfully processed
- **Completion bonus**: +0.10 when all inbox items are triaged
- **Penalties**: Archiving important emails scores 0.03–0.08 (not zero)

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- `openenv-core` and `uv` installed

### Install Dependencies

```bash
# Root-level (for inference script)
pip install -r requirements.txt

# Environment (using uv)
cd EmailTriage
uv sync
```

### Run Locally

```bash
# Start the environment server
cd EmailTriage
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run Inference

```bash
# Set required environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"
export LOCAL_IMAGE_NAME="emailtriage-env:latest"

# Run all 3 tasks
python inference.py
```

### Docker Build

```bash
docker build -t emailtriage-env:latest .
```

### Validate

```bash
cd EmailTriage
openenv validate
```

### Deploy to Hugging Face Spaces

```bash
cd EmailTriage
openenv push --repo-id YOUR_USERNAME/EmailTriage
```

## Project Structure

```text
Galcogens-OpenEnv/
├── inference.py              # Hackathon inference script (runs 3 tasks)
├── openenv.yaml              # Root OpenEnv manifest with task metadata
├── Dockerfile                # Root container definition
├── requirements.txt          # Inference-only dependencies
├── README.md                 # This file
└── EmailTriage/
    ├── __init__.py            # Package exports
    ├── client.py              # EnvClient implementation
    ├── models.py              # Pydantic Action/Observation/State models
    ├── openenv.yaml           # Inner OpenEnv manifest
    ├── pyproject.toml         # Package configuration
    ├── README.md              # HF Space README
    └── server/
        ├── app.py             # FastAPI server
        ├── EmailTriage_environment.py  # Core environment + 3 task graders
        └── Dockerfile         # Server container definition
```

## Hackathon Checklist

- [x] Real-world task simulation (email triage)
- [x] Full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml
- [x] 3 tasks with agent graders (easy → medium → hard, scores 0.0–1.0)
- [x] Meaningful reward function with partial progress signals
- [x] Baseline inference script with reproducible scores
- [x] Dockerfile builds
- [x] README with environment description, action/observation spaces, setup instructions

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | Hugging Face API key |
| `LOCAL_IMAGE_NAME` | No | `emailtriage-env:latest` | Docker image name |
