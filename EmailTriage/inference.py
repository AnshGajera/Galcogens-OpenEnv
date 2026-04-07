"""Hackathon inference loop for the EmailTriage OpenEnv environment.

Runs all 3 tasks (easy, medium, hard) sequentially using the OpenAI client.
Emits structured [START]/[STEP]/[END] logs per the hackathon spec.
"""

import os
import sys
import asyncio
import json
from typing import List, Optional, Dict, Any

import requests
from openai import OpenAI

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("IMAGE_NAME")
BENCHMARK_NAME = "openenv-emailtriage"

# Space URL - can be set via env or use default
# Use local server: http://localhost:8000
# Or HuggingFace Space: https://omchoksi108-emailopenenvrl.hf.space
SPACE_URL = os.getenv("SPACE_URL", "http://127.0.0.1:8000")

# Benchmark config
BENCHMARK_NAME = "openenv-emailtriage"
TASK_IDS = ["easy", "medium", "hard"]

# Per-task max steps
TASK_MAX_STEPS = {
    "easy": 6,
    "medium": 10,
    "hard": 12,
}

# Success threshold
SUCCESS_SCORE_THRESHOLD = 0.1


# ========================================
# Structured logging (EXACT format)
# ========================================


def log_start(task: str, env: str, model: str) -> None:
    """[START] task=<task> env=<benchmark> model=<model>"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    """[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>"""
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ========================================
# Prompt construction
# ========================================

SYSTEM_PROMPT = (
    "You are an elite, proactive email triage assistant. "
    "Your goal is to process the entire inbox efficiently.\n\n"
    "CRITICAL RULES:\n"
    "1. AVOID LOOPS: Don't read the same email twice.\n"
    "2. ARCHIVE SPAM: If subject looks like spam/marketing, use archive.\n"
    "3. READ IMPORTANT: For client/meeting emails, use read first.\n"
    "4. RESPOND: Use draft_email for replies with professional content.\n"
    "5. SCHEDULE: First query_calendar, then draft_email with slot.\n"
    "6. JSON ONLY: Respond with valid JSON containing: action_type, target_email_id, draft_content, proposed_slot."
)


def build_user_prompt(
    task_id: str,
    inbox_preview: List[Dict],
    returned_emails: List[str],
    calendar_slots: List[str],
    last_action_result: str,
) -> str:
    slots = ", ".join(calendar_slots) if calendar_slots else "none"

    inbox_lines = [
        f"id={item.get('id')} sender={item.get('sender')} priority={item.get('priority')} subject={item.get('subject')}"
        for item in inbox_preview
    ]
    inbox_block = " | ".join(inbox_lines) if inbox_lines else "no unread emails"
    reads_block = " | ".join(returned_emails) if returned_emails else "none"

    return (
        f"Task: {task_id}. Inbox: {inbox_block}. "
        f"Read: {reads_block}. "
        f"Calendar: {slots}. "
        f"Last result: {last_action_result}"
    )


# ========================================
# LLM action selection
# ========================================


def choose_action_with_llm(client: OpenAI, task_id: str, prompt: str) -> Dict[str, Any]:
    """Get action from LLM with safe defaults."""
    default_action = {
        "action_type": "query_calendar",
        "target_email_id": -1,
        "draft_content": "",
        "proposed_slot": "",
    }

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200,
            stream=False,
        )
        raw_content = (completion.choices[0].message.content or "").strip()
        if not raw_content:
            return default_action

        # Strip markdown fences
        if raw_content.startswith("```"):
            lines = raw_content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_content = "\n".join(lines)

        data = json.loads(raw_content)
        return {
            "action_type": data.get("action_type", "query_calendar"),
            "target_email_id": int(data.get("target_email_id", -1)),
            "draft_content": data.get("draft_content", ""),
            "proposed_slot": data.get("proposed_slot", ""),
        }
    except Exception as e:
        print(f"[WARN] LLM error: {e}", flush=True)
        return default_action


# ========================================
# HTTP API calls
# ========================================


def call_reset(api_url: str, task_id: str) -> Dict:
    """Call reset endpoint."""
    try:
        r = requests.post(
            f"{api_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Reset: {e}", flush=True)
        return {"observation": {}, "reward": 0, "done": True}


def call_step(api_url: str, action: Dict) -> Dict:
    """Call step endpoint."""
    try:
        r = requests.post(
            f"{api_url}/step",
            json={"action": action},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Step: {e}", flush=True)
        return {"observation": {}, "reward": 0, "done": True}


# ========================================
# Single task runner
# ========================================


async def run_task_http(
    api_url: str,
    llm_client: OpenAI,
    task_id: str,
) -> None:
    """Run a single task using HTTP API."""
    max_steps = TASK_MAX_STEPS[task_id]
    task_name = f"email-triage-{task_id}"
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        # Reset environment
        result = call_reset(api_url, task_id)
        obs = result.get("observation", {})

        for step in range(1, max_steps + 1):
            # Check if done
            inbox_remaining = obs.get("inbox_remaining", 0)
            if result.get("done") or inbox_remaining <= 0:
                break

            # Build prompt
            prompt = build_user_prompt(
                task_id=task_id,
                inbox_preview=obs.get("inbox_preview", []),
                returned_emails=obs.get("returned_emails", []),
                calendar_slots=obs.get("calendar_slots", []),
                last_action_result=obs.get("last_action_result", ""),
            )

            # Get action from LLM
            action_dict = choose_action_with_llm(llm_client, task_id, prompt)
            action = {"action": action_dict}

            # Execute step
            result = call_step(api_url, action)

            reward = float(result.get("reward", 0.0))
            rewards.append(reward)
            steps_taken = step

            # Format action string
            action_str = (
                f"{action_dict['action_type']}("
                f"target_email_id={action_dict['target_email_id']},"
                f"proposed_slot={action_dict['proposed_slot']})"
            )

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=bool(result.get("done")),
                error=None,
            )

            obs = result.get("observation", {})

            if result.get("done"):
                break

        # Calculate score (normalized)
        if rewards:
            # Total max possible reward varies by task
            max_total_reward = max_steps * 1.0  # Assuming max 1.0 per step
            score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
            score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Task failed: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ========================================
# Main entry point
# ========================================


async def main() -> None:
    """Main async function."""
    global SPACE_URL

    # Validate API key
    if HF_TOKEN:
        # Use with HF_TOKEN
        SPACE_URL = os.getenv(
            "SPACE_URL", "https://omchoksi108-emailopenenvrl.hf.space"
        )
    else:
        # Use SPACE_URL or fail
        if not SPACE_URL:
            print("[ERROR] HF_TOKEN or SPACE_URL must be set", flush=True)
            sys.exit(1)

    print(f"[INFO] Using API: {SPACE_URL}", flush=True)

    # Initialize LLM client
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    try:
        # Verify connectivity
        try:
            r = requests.get(f"{SPACE_URL}/health", timeout=10)
            if r.status_code != 200:
                print(f"[ERROR] Health check: {r.status_code}", flush=True)
                sys.exit(1)
            print("[INFO] API health: OK", flush=True)
        except Exception as e:
            print(f"[ERROR] Cannot reach API: {e}", flush=True)
            sys.exit(1)

        # Run all tasks
        for task_id in TASK_IDS:
            await run_task_http(SPACE_URL, llm_client, task_id)

    except KeyboardInterrupt:
        print("[INFO] Interrupted", flush=True)
    except Exception as e:
        print(f"[ERROR] Main: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
