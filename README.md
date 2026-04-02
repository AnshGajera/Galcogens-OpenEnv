# EmailTriage OpenEnv Hackathon Submission

This repository contains a production-grade OpenEnv environment for dynamic email triage, built for the OpenEnv Hackathon.

## Submission Status

- Environment status: Complete and deployed
- Hugging Face Space: Running
- Inference script status: Complete (HF Router + HF token)
- Core objective coverage: Complete for hackathon criteria

## Objective Checklist (Hackathon)

1. Rich, diverse, continuous rewards in [0, 1]: DONE
2. Multi-step, long-running dynamic task: DONE
3. Inference script as evaluator (not over-engineered agent): DONE
4. Real-world task utility (email triage workflow): DONE
5. OpenEnv/HF deployment compatibility: DONE

## What Was Implemented

### 1) Environment task design

- Built a realistic email triage environment with mixed inbox traffic:
  - spam/newsletter/notification
  - client requests
  - meeting scheduling
  - high-priority escalations
- Episode inbox size is randomized per reset (5 to 10 emails).
- State tracks processed emails, calendar query status, remaining inbox workload, and conversation history.

### 2) Reward system (partial progress, continuous)

Reward is always clamped to [0, 1].

- Querying calendar gives partial reward based on pending scheduling workload.
- Reading emails gives graded reward by priority and usefulness.
- Archiving low-value emails gives high reward with kind-based diversity.
- Incorrect archiving still gives a small low score (not hard zero), with harsher outcome for high-priority mistakes.
- Drafting replies is graded on:
  - task appropriateness
  - draft quality heuristics
  - calendar behavior for scheduling cases
  - valid proposed slot
  - urgency handling for escalation threads
- Extra reward bonus for concrete progress (processing an email).
- Completion bonus when all inbox items are triaged.

This creates a rich learning landscape and avoids repetitive binary scoring.

### 3) Dynamic, long-running interactions

The environment changes during an episode:

- A new urgent email can arrive mid-episode.
- Calendar availability can change mid-episode (slot removed).

This forces adaptation and multi-step correction rather than static one-shot behavior.

### 4) Inference script purpose and implementation

- inference.py is intentionally lightweight and evaluation-focused.
- Uses Hugging Face Router endpoint.
- Uses HF_TOKEN authentication only.
- Runs reset/step loop, formats prompt, sends actions, logs rewards.
- Avoids unnecessary reflection/memory-agent complexity.

## Key Files

- EmailTriage/server/EmailTriage_environment.py: Dynamic task logic and reward grader
- EmailTriage/models.py: Action, Observation, State contracts
- EmailTriage/server/app.py: OpenEnv server app and callable main entrypoint
- inference.py: Hackathon evaluation loop
- openenv.yaml: Root deployment manifest
- Dockerfile: Root deployment container

## How To Validate Locally

```bash
cd EmailTriage
openenv validate
```

## How To Push

```bash
cd EmailTriage
openenv push --repo-id OMCHOKSI108/EmailTriage
```

## Notes

- The project focus is the environment quality (task, rules, rewards), aligned with hackathon judging.
- The environment is built to be trainable for RL due to diverse partial reward signals.
