---
title: EmailTriage Environment Server
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "24.1.0"
python_version: "3.11"
app_file: server/app.py
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - email-triage
  - multi-agent
---

# 📬 EmailTriage: An OpenEnv Reinforcement Learning Environment

A dynamic, multi-turn **OpenEnv** environment designed for strict, realistic email triage orchestration. This environment acts as a standard `step()`, `reset()`, and `state()` API that grades AI agents on how efficiently they can navigate a turbulent inbox.

This project was built explicitly to fulfill the **OpenEnv Hackathon Challenge**. It passes all `openenv validate` pre-submission testing checks and is deployed universally via a Dockerfile container on Hugging Face.

---

## 🎯 Core Problem Statement & Environment Logic

### What does this test?
Unlike standard mini-games, this environment tests complex NLP reasoning and chronological tool-use. The agent must:
- Detect text context to differentiate between `Spam/Marketing` vs `Escalated P1 Outages`.
- Accurately trigger `archive` methods to clean the inbox loop without sacrificing important messages.
- Dynamically `read` emails, followed by executing a `query_calendar` sync, and finishing with an accurate `draft_email` containing a matched date-time string logic.

### ⚡ Huge Scenario Pool!
The internal environment has a massive deterministic dictionary of **over 50 unique real-world emails** ranging from:
- Angry CEOs making demands
- Fraudulent Nigerian Princes
- Legitimate B2B Vendor negotiations
- DataDog Server crashing alerts 

---

## 🚀 The Three Task Difficulties

| ID | Name | Initial Inbox Size | Max Steps | Dynamic Events (Mid-Episode) |
|----|------|--------|-----------|----------------|
| `easy` | Quick Sort | Exactly 3 | 6 Iterations| ❌ |
| `medium` | Priority Triage | Exactly 5 | 10 Iterations| ❌ |
| `hard` | Dynamic Crisis | 7 to 10 | 12 Iterations| ✅ (See Below) |

### 🌪️ Hard Mode Dynamic Events
If the AI engages in Hard Mode, the environment is programmed to actively combat the agent mid-episode:
1. **Interrupts:** Exactly on Step 3, a brand new `High Priority` email from the CEO is forcefully appended to the unread inbox queue. 
2. **Calendar Shifts:** Mid-episode, an available meeting slot might spontaneously disappear, forcing the AI to re-scan its options before generating a draft scheduling email. 

---

## 🔑 Action & Observation Schema Specifications

### The Agent's Action Hook (`EmailtriageAction`)

The AI must reply in strictly validated JSON matching these arguments over the WebSocket:
- `action_type`: Strictly mapped to `read`, `archive`, `query_calendar`, or `draft_email`.
- `target_email_id`: Matches an Integer directly mapped to the dynamically changing Inbox observation. (Use `-1` for generic commands like querying).
- `draft_content`: Minimum 40-character String. Scored heuristically on politeness ("thank you"), subject matching, and structural completion periods. 
- `proposed_slot`: String matching a Date-Time object scraped from the calendar.

### The Observation Space (`EmailtriageObservation`)

In order to prevent LLM hallucination cheating, the `Observation` payload strictly returns `[:5]` of the array (Only showing the first 5 unread items on "Screen 1"). **The email Body is heavily redacted** until the AI spends a turn executing the `read` action!
- `inbox_preview`: Summaries of up to 5 unread emails
- `returned_emails`: Contains the full string body unlocked by manual `read` actions
- `calendar_slots`: Array of currently available dates
- `last_action_result`: The immediate Grader string feedback returned to the AI
- `reward`: Floating integer normalized between `[0.0, 1.0]`

---

## ⚖️ The Grader & Reward Logic

Continuous `[0, 1]` rewards granting partial credit based on deterministic logic matching:

- **Archiving Logic**: `+0.62` to `+0.80` for accurately archiving Spam and Newsletters. Massive negative penalty for archiving High Priority client emails.
- **Reading Logic**: `+0.09` to `+0.25` for pulling data into focus (High Priority messages score higher).
- **Drafting Workflow**: Heavy reward weighting `+0.55+` if the AI perfectly maps `read -> query_calendar -> draft_email` seamlessly. Partial deductions for proposing unavailable chronological slots or missing urgent keywords ("today").

---

## 🛠️ Quick Start & Setup Instructions

### 1. Run inference locally via Uvicorn Server
Because this project utilizes modern `OpenEnv >= 0.2.2`, all environment interactions are fully asynchronous.

Boot the backend HTTP server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```
Trigger the OpenEnv SDK baseline:
```python
import asyncio
from openai import OpenAI
from server.EmailTriage_environment import EmailtriageEnv

async def main():
    llm = OpenAI(base_url="https://router.huggingface.co/v1")
    # Native connection without Docker bug risks locally
    env = EmailtriageEnv(base_url="http://127.0.0.1:8000")
    await env.connect()
    
    # Let the LLM play the environment
    await env.reset(options={"task_id": "hard"})
    
asyncio.run(main())
```

### 2. Evaluating for Hackathon Judges
This repository natively supports `openenv validate` from the root structure and natively passes the `validate-submission.sh` Hugging Face pipeline test.

To compile the environment into the evaluation Docker instance:
```bash
docker build -t emailtriage-env:latest -f Dockerfile .
```

To sync changes actively up to Hugging Face:
```bash
openenv push --repo-id YOUR_USERNAME/EmailTriage
```
