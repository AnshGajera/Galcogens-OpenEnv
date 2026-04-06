"""Hackathon inference loop for the EmailTriage OpenEnv environment.

Runs all 3 tasks (easy, medium, hard) sequentially using the OpenAI client.
Emits structured [START]/[STEP]/[END] logs per the hackathon spec.
"""

import os
import json
from typing import List, Optional

from openai import OpenAI

from EmailTriage import EmailtriageAction, EmailtriageEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "emailtriage-env:latest")
BENCHMARK_NAME = "openenv-emailtriage"

TASK_IDS = ["easy", "medium", "hard"]

# Per-task step budgets (must fit within 20min total runtime)
TASK_MAX_STEPS = {
    "easy": 6,
    "medium": 10,
    "hard": 12,
}


# ---------------------------------------------------------------------------
# Structured stdout logging (hackathon spec)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an elite, proactive email triage assistant operating in a strictly structured environment. "
    "Your goal is to process the entire inbox efficiently, maximizing your rewards.\n"
    "CRITICAL RULES FOR STATE ADVANCEMENT:\n"
    "1. AVOID LOOPS: Check the 'Last action result' and 'Recently read emails'. If you just read an email, DO NOT read it again. You must take the next logical step (archive or draft_email).\n"
    "2. SPAM/NEWSLETTERS: If an unread email subject from the 'Inbox preview' clearly looks like spam, marketing, or a low-priority notification, immediately use action_type='archive'.\n"
    "3. IMPORTANT EMAILS: If an unread email is a client request, meeting, or escalation, use action_type='read' first to get the full text.\n"
    "4. RESPONDING: If 'Recently read emails' contains a client email that needs a reply, immediately use action_type='draft_email'. "
    "Your draft_content MUST be professional, mention 'thank', reference specific details from the subject, end firmly with a period, and be over 40 characters.\n"
    "5. SCHEDULING CALENDAR: If a read email asks for a meeting, first use action_type='query_calendar' (target_email_id=-1) to load availability. "
    "In your VERY NEXT turn, use action_type='draft_email' and provide one of the listed slots exactly as shown in the 'proposed_slot' field.\n"
    "6. JSON FORMAT: Respond ONLY with valid JSON. Keys required: action_type, target_email_id, draft_content, proposed_slot. No markdown, no conversational text."
)


def build_user_prompt(
    task_id: str,
    inbox_preview: List[dict],
    returned_emails: List[str],
    calendar_slots: List[str],
    last_action_result: str,
) -> str:
    slots = ", ".join(calendar_slots) if calendar_slots else "none"
    inbox_lines = [
        f"id={item.get('id')} sender={item.get('sender')} "
        f"priority={item.get('priority')} subject={item.get('subject')}"
        for item in inbox_preview
    ]
    inbox_block = (
        " | ".join(inbox_lines) if inbox_lines else "no unread emails"
    )
    reads_block = " | ".join(returned_emails) if returned_emails else "none"

    return (
        f"Task difficulty: {task_id}. "
        f"Inbox preview: {inbox_block}. "
        f"Recently read emails: {reads_block}. "
        f"Calendar slots: {slots}. "
        f"Last action result: {last_action_result}."
    )


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------


def choose_action_with_llm(
    client: OpenAI,
    task_id: str,
    prompt: str,
) -> EmailtriageAction:
    default_action = EmailtriageAction(
        action_type="query_calendar",
        target_email_id=-1,
        draft_content="",
        proposed_slot="",
    )

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

        # Strip markdown fences if the model wraps JSON
        if raw_content.startswith("```"):
            lines = raw_content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_content = "\n".join(lines)

        data = json.loads(raw_content)
        return EmailtriageAction(
            action_type=data.get("action_type", "query_calendar"),
            target_email_id=int(data.get("target_email_id", -1)),
            draft_content=data.get("draft_content", ""),
            proposed_slot=data.get("proposed_slot", ""),
        )
    except Exception:
        return default_action


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------


async def run_task(
    llm_client: OpenAI,
    env: EmailtriageEnv,
    task_id: str,
) -> None:
    """Run a single task (easy/medium/hard) and emit structured logs."""
    max_steps = TASK_MAX_STEPS[task_id]
    task_name = f"email-triage-{task_id}"
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        result = await env.reset(options={"task_id": task_id})

        for step in range(1, max_steps + 1):
            obs = result.observation
            if result.done or obs.inbox_remaining <= 0:
                break

            prompt = build_user_prompt(
                task_id=task_id,
                inbox_preview=obs.inbox_preview,
                returned_emails=obs.returned_emails,
                calendar_slots=obs.calendar_slots,
                last_action_result=obs.last_action_result,
            )
            action = choose_action_with_llm(llm_client, task_id, prompt)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            action_str = (
                f"{action.action_type}("
                f"target_email_id={action.target_email_id},"
                f"proposed_slot={action.proposed_slot})"
            )
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=bool(result.done),
                error=None,
            )

            if result.done:
                break

        if rewards:
            avg = sum(rewards) / len(rewards)
            success = avg >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "HF_TOKEN must be set in environment variables."
        )

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Bypassing Docker networking issues on Windows by connecting to your already-running container on port 8000
    env = EmailtriageEnv(base_url="http://127.0.0.1:8000")
    await env.connect()
    
    # WARNING: When submitting for the hackathon, switch back to:
    # env = await EmailtriageEnv.from_docker_image(LOCAL_IMAGE_NAME)


    try:
        for task_id in TASK_IDS:
            await run_task(llm_client, env, task_id)
    finally:
        await env.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

