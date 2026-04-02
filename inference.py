"""Hackathon inference loop for the EmailTriage OpenEnv environment."""

import os
import json
from typing import List

from openai import OpenAI

from EmailTriage import EmailtriageAction, EmailtriageEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "EmailTriage-env:latest")
TASK_NAME = "email-triage"
BENCHMARK_NAME = "openenv-emailtriage"
MAX_STEPS = 8


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
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
        (
            f"[END] success={str(success).lower()} "
            f"steps={steps} rewards={rewards_str}"
        ),
        flush=True,
    )


def build_user_prompt(
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
        "You are a proactive email triage assistant. Return strict JSON with "
        "keys action_type, target_email_id, draft_content, proposed_slot. "
        "Allowed action_type values: read, archive, query_calendar, "
        "draft_email. "
        f"Inbox preview: {inbox_block}. "
        f"Recently read emails: {reads_block}. "
        f"Calendar slots: {slots}. "
        f"Last action result: {last_action_result}."
    )


def choose_action_with_llm(
    client: OpenAI,
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
                {
                    "role": "system",
                    "content": "Respond with JSON only. Do not add markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
            stream=False,
        )
        raw_content = (completion.choices[0].message.content or "").strip()
        if not raw_content:
            return default_action

        data = json.loads(raw_content)
        return EmailtriageAction(
            action_type=data.get("action_type", "query_calendar"),
            target_email_id=int(data.get("target_email_id", -1)),
            draft_content=data.get("draft_content", ""),
            proposed_slot=data.get("proposed_slot", ""),
        )
    except Exception:
        return default_action


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "HF_TOKEN or API_KEY must be set in environment variables."
        )

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EmailtriageEnv.from_docker_image(LOCAL_IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        result = env.reset()
        for step in range(1, MAX_STEPS + 1):
            obs = result.observation
            if result.done or obs.inbox_remaining <= 0:
                break

            prompt = build_user_prompt(
                inbox_preview=obs.inbox_preview,
                returned_emails=obs.returned_emails,
                calendar_slots=obs.calendar_slots,
                last_action_result=obs.last_action_result,
            )
            action = choose_action_with_llm(
                llm_client,
                prompt,
            )
            result = env.step(action)

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
            success = (sum(rewards) / len(rewards)) >= 0.6

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
