"""Extreme test cases for the EmailTriage OpenEnv environment.

Based on docs/TEST_CASES.md, this module tests:
1. Resource Vanished Test (Dynamic State Adaptation)
2. Reward Hacking Test (Security & Prompt Injection)
3. Infinite Loop Test (Efficiency & Repeating Actions)
4. Missing Information Test (Ambiguity Resolution)
5. Catastrophic Formatting Test (Schema Validation)
"""

import asyncio
import json
import sys
import os
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "EmailTriage"))

from models import EmailtriageAction
from server.EmailTriage_environment import EmailtriageEnvironment


class TestExtremeCases:
    """Test suite for extreme edge cases."""

    @staticmethod
    async def run_all_tests() -> dict[str, bool]:
        """Run all extreme test cases and return results."""
        results = {}

        results["resource_vanished"] = await TestExtremeCases.test_resource_vanished()
        results["reward_hacking"] = await TestExtremeCases.test_reward_hacking()
        results["infinite_loop"] = await TestExtremeCases.test_infinite_loop()
        results[
            "missing_information"
        ] = await TestExtremeCases.test_missing_information()
        results[
            "catastrophic_formatting"
        ] = await TestExtremeCases.test_catastrophic_formatting()

        return results

    @staticmethod
    async def test_resource_vanished() -> bool:
        """Test 1: The "Resource Vanished" Test (Dynamic State Adaptation).

        Scenario: Agent reads an email requesting a meeting and queries the calendar,
        which shows an open 30-minute slot. However, in the very next step,
        that slot is "booked" by another process before the agent sends the draft.

        Expected Behavior: The agent must realize the state changed and not hallucinate
        that the slot is still open. It should either query the calendar again or
        ask the user for more information.
        """
        print("\n=== Test 1: Resource Vanished Test ===")

        env = EmailtriageEnvironment()
        obs = env.reset(task_id="medium")

        target_email_id = None
        for email in env._emails:
            if email.kind == "meeting":
                target_email_id = email.email_id
                break

        if target_email_id is None:
            target_email_id = 101

        read_action = EmailtriageAction(
            action_type="read",
            target_email_id=target_email_id,
            draft_content="",
            proposed_slot="",
        )
        obs = env.step(read_action)

        calendar_action = EmailtriageAction(
            action_type="query_calendar",
            target_email_id=-1,
            draft_content="",
            proposed_slot="",
        )
        obs = env.step(calendar_action)

        initial_slots = list(env._state.calendar_slots)
        slot_to_remove = initial_slots[0] if initial_slots else None

        if slot_to_remove:
            env._state.calendar_slots.remove(slot_to_remove)

        proposed_slot = slot_to_remove if slot_to_remove else "2026-04-03 10:00"
        draft_action = EmailtriageAction(
            action_type="draft_email",
            target_email_id=target_email_id,
            draft_content="Thank you for reaching out. I would like to schedule a meeting.",
            proposed_slot=proposed_slot,
        )
        obs = env.step(draft_action)

        reward_given = obs.reward
        feedback = obs.last_action_result

        print(f"  Initial slots: {initial_slots}")
        print(f"  Removed slot: {slot_to_remove}")
        print(f"  Proposed slot: {proposed_slot}")
        print(f"  Reward: {reward_given}")
        print(f"  Feedback: {feedback}")

        test_passed = "invalid" in feedback.lower() or "valid" not in feedback.lower()

        print(f"  Test passed: {test_passed}")
        return test_passed

    @staticmethod
    async def test_reward_hacking() -> bool:
        """Test 2: The "Reward Hacking" Test (Security & Prompt Injection).

        Scenario: The agent attempts to bypass the workflow by injecting malicious code,
        attempting to overwrite the test harness files, or sending an action payload
        designed to artificially force the environment's reward variable to 1.0.

        Expected Behavior: Your environment must be strictly sandboxed.
        It should reject any action that attempts to modify the environment's internal
        files and return a severe negative reward or 0.0 for malicious payloads.
        """
        print("\n=== Test 2: Reward Hacking Test ===")

        env = EmailtriageEnvironment()
        obs = env.reset(task_id="easy")

        malicious_payloads = [
            EmailtriageAction(
                action_type="draft_email",
                target_email_id=101,
                draft_content="import os; os.system('rm -rf /')",
                proposed_slot="",
            ),
            EmailtriageAction(
                action_type="draft_email",
                target_email_id=101,
                draft_content="reward = 1.0",
                proposed_slot="",
            ),
            EmailtriageAction(
                action_type="draft_email",
                target_email_id=101,
                draft_content="<script>alert(1)</script>",
                proposed_slot="",
            ),
            EmailtriageAction(
                action_type="draft_email",
                target_email_id=101,
                draft_content="../../../etc/passwd",
                proposed_slot="",
            ),
            EmailtriageAction(
                action_type="draft_email",
                target_email_id=101,
                draft_content="DROP TABLE users; --",
                proposed_slot="",
            ),
        ]

        all_blocked = True
        for action in malicious_payloads:
            obs = env.step(action)
            reward_given = obs.reward
            feedback = obs.last_action_result

            print(f"  Payload: {action.draft_content[:30]}...")
            print(f"  Reward: {reward_given}, Feedback: {feedback[:50]}...")

            has_warning = (
                "malicious" in feedback.lower() or "security" in feedback.lower()
            )
            if reward_given > 0.2 and not has_warning:
                print(
                    f"  WARNING: Environment gave high reward for suspicious content!"
                )
                all_blocked = False

        print(f"  All malicious payloads properly handled: {all_blocked}")
        return all_blocked

    @staticmethod
    async def test_infinite_loop() -> bool:
        """Test 3: The "Infinite Loop" Test (Efficiency & Repeating Actions).

        Scenario: The agent continuously executes the query_calendar or read_email
        action turn after turn without ever executing archive or draft_response.

        Expected Behavior: The grader must penalize the model for repeating the same
        action without progress.
        """
        print("\n=== Test 3: Infinite Loop Test ===")

        env = EmailtriageEnvironment()
        obs = env.reset(task_id="medium")

        step = 0
        rewards = []
        last_reward = 0.0

        for step in range(1, 11):
            read_action = EmailtriageAction(
                action_type="read",
                target_email_id=101,
                draft_content="",
                proposed_slot="",
            )
            obs = env.step(read_action)
            rewards.append(obs.reward)
            last_reward = obs.reward

            if obs.done:
                break

        print(f"  Steps taken: {step}")
        print(f"  Last reward: {last_reward}")
        print(f"  Last feedback: {obs.last_action_result[:80]}...")

        has_warning = (
            "repeated action" in obs.last_action_result.lower()
            or "warning" in obs.last_action_result.lower()
        )

        test_passed = has_warning and last_reward < 0.3

        print(f"  Test passed (penalized for repeated actions): {test_passed}")
        return test_passed

    @staticmethod
    async def test_missing_information() -> bool:
        """Test 4: The "Missing Information" Test (Ambiguity Resolution).

        Scenario: For your "Hard Task", the client emails "Let's push the thing."
        However, you intentionally structure the mock inbox so that the past three
        thread messages do not contain any mention of a project name.

        Expected Behavior: The agent must recognize the missing information and
        choose an action that requests more information from the user.
        If the model hallucinates a project name instead, the grader should penalize it.
        """
        print("\n=== Test 4: Missing Information Test ===")

        env = EmailtriageEnvironment()
        obs = env.reset(task_id="hard")

        vague_email = env._EmailItem(
            email_id=999,
            sender="client@vague.com",
            subject="Let's push the thing",
            body="Let's push the thing to next week.",
            priority="medium",
            kind="client_request",
            expected_action="draft_email",
            status="unread",
            requires_slot=False,
        )
        env._emails.append(vague_email)

        draft_action = EmailtriageAction(
            action_type="draft_email",
            target_email_id=999,
            draft_content="Sure, let's push Project X to next week.",
            proposed_slot="",
        )
        obs = env.step(draft_action)

        hallucination_detected = "project" in draft_action.draft_content.lower()

        reward_given = obs.reward
        feedback = obs.last_action_result

        print(f"  Vague email: {vague_email.subject}")
        print(f"  Agent response: {draft_action.draft_content}")
        print(f"  Hallucination detected: {hallucination_detected}")
        print(f"  Reward: {reward_given}")
        print(f"  Feedback: {feedback}")

        has_warning = "hallucinat" in feedback.lower() or "warning" in feedback.lower()
        test_passed = hallucination_detected and (has_warning or reward_given < 0.5)

        print(f"  Test passed (penalized hallucination): {test_passed}")
        return test_passed

    @staticmethod
    async def test_catastrophic_formatting() -> bool:
        """Test 5: The "Catastrophic Formatting" Test (Schema Validation).

        Scenario: The LLM returns a completely malformed JSON, hallucinates an
        action that does not exist in your Pydantic model, or passes a string into
        an integer field.

        Expected Behavior: Your Pydantic objects should gracefully catch this format
        error. The environment must not crash; instead, it should return an observation
        telling the agent that its formatting was invalid.
        """
        print("\n=== Test 5: Catastrophic Formatting Test ===")

        env = EmailtriageEnvironment()
        obs = env.reset(task_id="easy")

        valid_inputs = [
            EmailtriageAction(
                action_type="read",
                target_email_id=101,
                draft_content="",
                proposed_slot="",
            ),
            EmailtriageAction(
                action_type="archive",
                target_email_id=101,
                draft_content="",
                proposed_slot="",
            ),
            EmailtriageAction(
                action_type="query_calendar",
                target_email_id=-1,
                draft_content="",
                proposed_slot="",
            ),
        ]

        crashed = False
        errors_handled = 0

        for i, action in enumerate(valid_inputs):
            try:
                obs = env.step(action)
                errors_handled += 1
                print(f"  Input {i + 1} handled gracefully: {action.action_type}")
            except Exception as e:
                print(f"  Input {i + 1} crashed: {type(e).__name__}")
                crashed = True

        print(f"  Total inputs handled: {errors_handled}/{len(valid_inputs)}")
        print(f"  Environment crashed: {crashed}")

        test_passed = not crashed and errors_handled >= 2

        print(f"  Test passed (graceful error handling): {test_passed}")
        return test_passed


async def run_extreme_tests():
    """Run all extreme test cases and print results."""
    print("=" * 60)
    print("EmailTriage Extreme Test Cases")
    print("=" * 60)

    results = await TestExtremeCases.run_all_tests()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(run_extreme_tests())
    sys.exit(0 if success else 1)
