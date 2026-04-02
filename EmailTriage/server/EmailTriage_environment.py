# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EmailTriage environment implementation."""

from dataclasses import dataclass
import random
from typing import List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        EmailtriageAction,
        EmailtriageObservation,
        EmailtriageState,
    )
except ImportError:
    from models import (
        EmailtriageAction,
        EmailtriageObservation,
        EmailtriageState,
    )


class EmailtriageEnvironment(Environment):
    """Multi-turn environment for email triage and workflow orchestration."""

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance when using factory mode in app.py.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    @dataclass
    class _EmailItem:
        email_id: int
        sender: str
        subject: str
        body: str
        priority: str
        kind: str
        expected_action: str
        status: str = "unread"
        requires_slot: bool = False

    def __init__(self):
        """Initialize the EmailTriage environment."""
        self._state = EmailtriageState(
            episode_id=str(uuid4()),
            step_count=0,
            inbox=[],
            calendar_slots=[],
            queried_calendar=False,
            processed_email_ids=[],
        )
        self._history: List[str] = []
        self._last_action_result = ""
        self._max_steps = 12
        self._default_calendar_slots = [
            "2026-04-03 10:00",
            "2026-04-03 14:00",
            "2026-04-04 09:30",
            "2026-04-04 15:30",
        ]
        self._emails: List[EmailtriageEnvironment._EmailItem] = []
        self._last_read_payload: List[str] = []

    def reset(self) -> EmailtriageObservation:
        """Reset the environment and return the first inbox observation."""
        self._state = EmailtriageState(
            episode_id=str(uuid4()),
            step_count=0,
            inbox=[],
            calendar_slots=list(self._default_calendar_slots),
            queried_calendar=False,
            processed_email_ids=[],
        )
        self._history = []
        self._last_read_payload = []
        self._last_action_result = (
            "Environment reset. Start triaging the inbox."
        )
        self._emails = self._build_episode_emails()
        self._sync_state_inbox()

        return self._build_observation(reward=0.0, done=False)

    def step(
        self, action: EmailtriageAction
    ) -> EmailtriageObservation:  # type: ignore[override]
        """Execute one triage step and return graded feedback."""
        self._state.step_count += 1
        self._last_read_payload = []

        if self._is_all_processed():
            self._last_action_result = (
                "Inbox is already complete. Call reset()."
            )
            return self._build_observation(reward=0.0, done=True)

        reward, feedback = self._route_action(action)
        self._last_action_result = feedback
        self._sync_state_inbox()

        self._history.append(
            f"step={self._state.step_count} "
            f"action={action.action_type} "
            f"reward={reward:.2f} "
            f"feedback={feedback}"
        )

        done = (
            self._is_all_processed()
            or self._state.step_count >= self._max_steps
        )

        if done and self._is_all_processed():
            self._last_action_result = "Inbox triage complete."

        return self._build_observation(reward=reward, done=done)

    def _build_episode_emails(self) -> List[_EmailItem]:
        """Build a randomized 5-10 email inbox for each episode."""
        required = [
            self._EmailItem(
                email_id=0,
                sender="no-reply@promo.shop",
                subject="Flash sale: 70% off accessories",
                body="Marketing content. No user follow-up needed.",
                priority="low",
                kind="spam",
                expected_action="archive",
            ),
            self._EmailItem(
                email_id=0,
                sender="alex@clientco.com",
                subject="Request: project kickoff meeting",
                body="Please suggest a 30-minute slot next week.",
                priority="medium",
                kind="meeting",
                expected_action="draft_email",
                requires_slot=True,
            ),
        ]

        pool = [
            self._EmailItem(
                email_id=0,
                sender="alerts@monitoring.io",
                subject="CPU usage spike detected",
                body="Please review incident notes and notify on-call.",
                priority="high",
                kind="escalation",
                expected_action="draft_email",
            ),
            self._EmailItem(
                email_id=0,
                sender="digest@newsletters.ai",
                subject="Weekly AI digest",
                body="Curated newsletter with general updates.",
                priority="low",
                kind="newsletter",
                expected_action="archive",
            ),
            self._EmailItem(
                email_id=0,
                sender="sam@partner.io",
                subject="Need integration timeline",
                body="Could you share a realistic delivery window?",
                priority="medium",
                kind="client_request",
                expected_action="draft_email",
            ),
            self._EmailItem(
                email_id=0,
                sender="hr@company.com",
                subject="Policy reminder",
                body="Quarterly policy acknowledgment reminder.",
                priority="low",
                kind="notification",
                expected_action="archive",
            ),
            self._EmailItem(
                email_id=0,
                sender="mira@vendor.net",
                subject="Schedule security review",
                body="Need a slot this week for security walkthrough.",
                priority="medium",
                kind="meeting",
                expected_action="draft_email",
                requires_slot=True,
            ),
            self._EmailItem(
                email_id=0,
                sender="ops@platform.org",
                subject="Customer complaint escalation",
                body="Escalated account issue waiting on owner response.",
                priority="high",
                kind="escalation",
                expected_action="draft_email",
            ),
            self._EmailItem(
                email_id=0,
                sender="billing@services.com",
                subject="Invoice copy",
                body="Please confirm invoice receipt and processing ETA.",
                priority="medium",
                kind="client_request",
                expected_action="draft_email",
            ),
            self._EmailItem(
                email_id=0,
                sender="offers@store.example",
                subject="Weekend coupons",
                body="Promotional coupons and shopping suggestions.",
                priority="low",
                kind="spam",
                expected_action="archive",
            ),
        ]

        target_count = random.randint(5, 10)
        additional_count = target_count - len(required)
        selected = required + random.sample(pool, k=additional_count)
        random.shuffle(selected)

        emails: List[EmailtriageEnvironment._EmailItem] = []
        for index, item in enumerate(selected, start=1):
            emails.append(
                self._EmailItem(
                    email_id=100 + index,
                    sender=item.sender,
                    subject=item.subject,
                    body=item.body,
                    priority=item.priority,
                    kind=item.kind,
                    status="unread",
                    requires_slot=item.requires_slot,
                    expected_action=item.expected_action,
                )
            )
        return emails

    def _build_observation(
        self, reward: float, done: bool
    ) -> EmailtriageObservation:
        """Build observation for the current inbox state."""
        preview = [
            {
                "id": str(email.email_id),
                "sender": email.sender,
                "subject": email.subject,
                "priority": email.priority,
                "status": email.status,
            }
            for email in self._emails
            if email.status == "unread"
        ][:5]

        unread_count = len(
            [email for email in self._emails if email.status == "unread"]
        )

        return EmailtriageObservation(
            inbox_preview=preview,
            returned_emails=self._last_read_payload,
            calendar_slots=list(self._state.calendar_slots),
            last_action_result=self._last_action_result,
            conversation_history=self._history[-8:],
            inbox_remaining=unread_count,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "emails_total": len(self._emails),
                "emails_processed": len(self._state.processed_email_ids),
                "queried_calendar": self._state.queried_calendar,
            },
        )

    def _route_action(self, action: EmailtriageAction) -> tuple[float, str]:
        """Route action to task logic and compute reward in [0, 1]."""
        if action.action_type == "query_calendar":
            self._state.queried_calendar = True
            pending_scheduling = any(
                email.status == "unread" and email.requires_slot
                for email in self._emails
            )
            reward = 0.25 + (0.2 if pending_scheduling else 0.0)
            return min(1.0, reward), "Calendar queried successfully."

        target = self._find_email(action.target_email_id)
        if target is None:
            return 0.0, "Invalid or missing target_email_id."

        if action.action_type == "read":
            self._last_read_payload = [
                (
                    f"Email {target.email_id} from {target.sender}: "
                    f"{target.subject} | {target.body}"
                )
            ]
            reward = 0.15 if target.status == "unread" else 0.05
            return reward, "Email content returned to the agent."

        if action.action_type == "archive":
            if target.status != "unread":
                return 0.05, "Email already processed."

            if target.kind in {"spam", "newsletter", "notification"}:
                target.status = "archived"
                self._mark_processed(target.email_id)
                return 0.8, "Correctly archived low-value email."

            return 0.1, "Archived an email that likely needed a response."

        if action.action_type == "draft_email":
            return self._grade_draft_action(target, action)

        return 0.0, "Unsupported action type."

    def _grade_draft_action(
        self, target: _EmailItem, action: EmailtriageAction
    ) -> tuple[float, str]:
        """Grade draft_email actions with partial rewards."""
        if target.status != "unread":
            return 0.05, "Email already processed."

        reward = 0.2
        feedback_parts: List[str] = []

        if target.kind in {"meeting", "client_request", "escalation"}:
            reward += 0.2
            feedback_parts.append(
                "Draft action is appropriate for this email."
            )
        else:
            feedback_parts.append(
                "Drafting may be unnecessary for this email."
            )

        draft_quality = self._draft_quality_score(action.draft_content)
        reward += 0.35 * draft_quality

        if target.requires_slot:
            if self._state.queried_calendar:
                reward += 0.15
                feedback_parts.append(
                    "Checked calendar before proposing a slot."
                )
            else:
                feedback_parts.append(
                    "Calendar should be queried before scheduling."
                )

            if action.proposed_slot in self._state.calendar_slots:
                reward += 0.25
                feedback_parts.append("Proposed a valid available slot.")
            else:
                feedback_parts.append("Missing or invalid proposed slot.")

        reward = max(0.0, min(1.0, reward))
        if reward >= 0.55:
            target.status = "drafted"
            self._mark_processed(target.email_id)
        return reward, " ".join(feedback_parts)

    @staticmethod
    def _draft_quality_score(draft_content: str) -> float:
        """Score draft quality with deterministic heuristics."""
        clean_text = draft_content.strip().lower()
        if not clean_text:
            return 0.0

        score = 0.0
        if len(clean_text) >= 40:
            score += 0.45
        if "thank" in clean_text:
            score += 0.2
        if (
            "meeting" in clean_text
            or "schedule" in clean_text
            or "slot" in clean_text
        ):
            score += 0.2
        if clean_text.endswith(".") or clean_text.endswith("!"):
            score += 0.15
        return max(0.0, min(1.0, score))

    def _find_email(self, email_id: int) -> _EmailItem | None:
        """Find an email by ID."""
        for email in self._emails:
            if email.email_id == email_id:
                return email
        return None

    def _mark_processed(self, email_id: int) -> None:
        """Record processed email IDs once."""
        if email_id not in self._state.processed_email_ids:
            self._state.processed_email_ids.append(email_id)

    def _is_all_processed(self) -> bool:
        """Return True when inbox has no unread emails."""
        return all(email.status != "unread" for email in self._emails)

    def _sync_state_inbox(self) -> None:
        """Mirror internal inbox into serializable state payload."""
        self._state.inbox = [
            {
                "id": str(email.email_id),
                "sender": email.sender,
                "subject": email.subject,
                "body": email.body,
                "priority": email.priority,
                "kind": email.kind,
                "status": email.status,
            }
            for email in self._emails
        ]

    @property
    def state(self) -> EmailtriageState:
        """Get current environment state."""
        return self._state
