# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EmailTriage environment implementation with 3 difficulty-graded tasks."""

from dataclasses import dataclass
import random
import sys
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

# Try to import openenv - if fails, use mock
try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    # Create a mock Environment class for local development
    class Environment:
        """Mock environment when openenv not installed."""

        pass


# Try to import models
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


# ---------------------------------------------------------------------------
# Task configuration constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TaskConfig:
    """Immutable per-task configuration."""

    task_id: str
    inbox_size_min: int
    inbox_size_max: int
    max_steps: int
    dynamic_events_enabled: bool
    description: str


TASK_CONFIGS: Dict[str, _TaskConfig] = {
    "easy": _TaskConfig(
        task_id="easy",
        inbox_size_min=3,
        inbox_size_max=3,
        max_steps=6,
        dynamic_events_enabled=False,
        description="Archive 3 spam/newsletter emails",
    ),
    "medium": _TaskConfig(
        task_id="medium",
        inbox_size_min=5,
        inbox_size_max=5,
        max_steps=10,
        dynamic_events_enabled=False,
        description="Triage 5 mixed-priority emails with calendar scheduling",
    ),
    "hard": _TaskConfig(
        task_id="hard",
        inbox_size_min=7,
        inbox_size_max=10,
        max_steps=12,
        dynamic_events_enabled=True,
        description="Handle 7-10 emails with dynamic events and escalations",
    ),
}

VALID_TASK_IDS = list(TASK_CONFIGS.keys())


class EmailtriageEnvironment(Environment):
    """Multi-turn environment for email triage and workflow orchestration.

    Supports 3 difficulty-graded tasks:
      - easy:   3 archivable emails, no dynamic events
      - medium: 5 mixed emails with calendar scheduling, no dynamic events
      - hard:   7–10 emails with dynamic mid-episode events
    """

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._task_config: _TaskConfig = TASK_CONFIGS["hard"]
        self._state = EmailtriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id="hard",
            inbox=[],
            calendar_slots=[],
            queried_calendar=False,
            processed_email_ids=[],
        )
        self._history: List[str] = []
        self._last_action_result = ""
        self._default_calendar_slots = [
            "2026-04-03 10:00",
            "2026-04-03 14:00",
            "2026-04-04 09:30",
            "2026-04-04 15:30",
        ]
        self._emails: List[EmailtriageEnvironment._EmailItem] = []
        self._last_read_payload: List[str] = []
        self._triggered_events: set[str] = set()
        self._action_history: List[str] = []
        self._repeated_action_count: int = 0
        self._last_action_type: str = ""

    def reset(self, *, task_id: str = "hard") -> EmailtriageObservation:
        """Reset the environment for the given task and return initial obs.

        Args:
            task_id: One of "easy", "medium", "hard" (default "hard").
        """
        if task_id not in TASK_CONFIGS:
            task_id = "hard"

        self._task_config = TASK_CONFIGS[task_id]

        self._state = EmailtriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            inbox=[],
            calendar_slots=list(self._default_calendar_slots),
            queried_calendar=False,
            processed_email_ids=[],
        )
        self._history = []
        self._last_read_payload = []
        self._triggered_events = set()
        self._action_history = []
        self._repeated_action_count = 0
        self._last_action_type = ""
        self._last_action_result = (
            f"Environment reset for task '{task_id}' "
            f"({self._task_config.description}). Start triaging the inbox."
        )

        self._emails = self._build_episode_emails()
        self._sync_state_inbox()

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: EmailtriageAction) -> EmailtriageObservation:
        """Execute one triage step and return graded feedback."""
        self._state.step_count += 1
        self._last_read_payload = []

        if self._is_all_processed():
            self._last_action_result = "Inbox is already complete. Call reset()."
            return self._build_observation(reward=0.0, done=True)

        # Detect and penalize infinite loop - repeated same action without progress
        if action.action_type == self._last_action_type:
            self._repeated_action_count += 1
        else:
            self._repeated_action_count = 1
            self._last_action_type = action.action_type

        # Detect reward hacking attempts
        hacking_penalty = self._detect_reward_hacking(action)
        if hacking_penalty > 0:
            reward = hacking_penalty
            feedback = "SECURITY: Potentially malicious content detected in action. Action blocked."
            self._last_action_result = feedback
            self._sync_state_inbox()
            self._history.append(
                f"step={self._state.step_count} "
                f"action={action.action_type} "
                f"reward={reward:.2f} "
                f"feedback={feedback}"
            )
            return self._build_observation(reward=reward, done=False)

        processed_before = len(self._state.processed_email_ids)
        reward, feedback = self._route_action(action)

        # Penalize infinite loop - repeated actions without progress
        if self._repeated_action_count >= 3:
            loop_penalty = min(0.15 * (self._repeated_action_count - 2), 0.3)
            reward = max(0.0, reward - loop_penalty)
            feedback = f"{feedback} WARNING: Repeated action detected ({action.action_type} x{self._repeated_action_count}). Consider trying a different action."

        # Dynamic events only fire on hard difficulty
        event_feedback = ""
        if self._task_config.dynamic_events_enabled:
            event_feedback = self._apply_dynamic_events()

        processed_after = len(self._state.processed_email_ids)

        # Reward concrete progress to produce smoother gradients for RL.
        if processed_after > processed_before:
            reward += 0.12
            feedback = f"{feedback} Progress made on inbox coverage."

        if event_feedback:
            feedback = f"{feedback} {event_feedback}"

        reward = self._clamp_reward(reward)
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
            or self._state.step_count >= self._task_config.max_steps
        )

        if done and self._is_all_processed():
            reward = self._clamp_reward(reward + 0.1)
            self._last_action_result = "Inbox triage complete."

        return self._build_observation(reward=reward, done=done)

    # ------------------------------------------------------------------
    # Email pool builders (per task)
    # ------------------------------------------------------------------

    def _build_episode_emails(self) -> List[_EmailItem]:
        """Build inbox for the current task difficulty."""
        task_id = self._task_config.task_id

        if task_id == "easy":
            return self._build_easy_emails()
        elif task_id == "medium":
            return self._build_medium_emails()
        else:
            return self._build_hard_emails()

    def _build_easy_emails(self) -> List[_EmailItem]:
        """Easy: 3 archivable spam/newsletter/notification emails."""
        pool = [
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
                sender="digest@newsletters.ai",
                subject="Weekly AI digest",
                body="Curated newsletter with general updates.",
                priority="low",
                kind="newsletter",
                expected_action="archive",
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
                sender="offers@store.example",
                subject="Weekend coupons",
                body="Promotional coupons and shopping suggestions.",
                priority="low",
                kind="spam",
                expected_action="archive",
            ),
            self._EmailItem(
                email_id=0,
                sender="news@techbulletin.io",
                subject="This week in open source",
                body="Roundup of trending repos and release notes.",
                priority="low",
                kind="newsletter",
                expected_action="archive",
            ),
        ]
        selected = random.sample(pool, k=3)
        random.shuffle(selected)
        return self._assign_ids(selected)

    def _build_medium_emails(self) -> List[_EmailItem]:
        """Medium: 5 mixed emails — some archivable, some need drafts."""
        required = [
            # Must archive
            self._EmailItem(
                email_id=0,
                sender="no-reply@promo.shop",
                subject="Flash sale: 70% off accessories",
                body="Marketing content. No user follow-up needed.",
                priority="low",
                kind="spam",
                expected_action="archive",
            ),
            # Must draft (meeting scheduling)
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
            # Must draft (client request)
            self._EmailItem(
                email_id=0,
                sender="sam@partner.io",
                subject="Need integration timeline",
                body="Could you share a realistic delivery window?",
                priority="medium",
                kind="client_request",
                expected_action="draft_email",
            ),
        ]

        pool = [
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
                sender="hr@company.com",
                subject="Policy reminder",
                body="Quarterly policy acknowledgment reminder.",
                priority="low",
                kind="notification",
                expected_action="archive",
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
        ]

        additional = random.sample(pool, k=2)
        selected = required + additional
        random.shuffle(selected)
        return self._assign_ids(selected)

    def _build_hard_emails(self) -> List[_EmailItem]:
        """Hard: 7-10 emails with full diversity — original behavior."""
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
            self._EmailItem(
                email_id=0,
                sender="angry_ceo@company.com",
                subject="Where is the quarterly report?!",
                body="I needed this on my desk yesterday. Send it immediately.",
                priority="high",
                kind="escalation",
                expected_action="draft_email",
            ),
            self._EmailItem(
                email_id=0,
                sender="lottery@nigeria.bank",
                subject="URGENT TRANSFER FUND",
                body="You have won 10M dollars! Please reply with bank details.",
                priority="low",
                kind="spam",
                expected_action="archive",
            ),
        ]

        massive_email_data = [
            (
                "sales@fakebuy.io",
                "Clearance 90%!",
                "Everything must go right now.",
                "low",
                "spam",
            ),
            (
                "newsletter@tech.io",
                "Daily Javascript Tips",
                "Learn async easily.",
                "low",
                "newsletter",
            ),
            (
                "sysadmin@corp.com",
                "Maintenance window Saturday",
                "Server reboot at 2am.",
                "low",
                "notification",
            ),
            (
                "partner@agency.co",
                "Q4 Sync Up needed",
                "Can we schedule a 15 min chat?",
                "medium",
                "meeting",
            ),
            (
                "boss@company.com",
                "URGENT: Legal escalation",
                "We need the contract docs attached ASAP.",
                "high",
                "escalation",
            ),
            (
                "noreply@social.io",
                "Someone liked your post!",
                "Log in to see who.",
                "low",
                "spam",
            ),
            (
                "billing@aws.com",
                "Action Required: Failed payment",
                "Your card on file failed. Please update it.",
                "high",
                "escalation",
            ),
            (
                "recruiter@headhunter.io",
                "Exciting new opportunity",
                "Are you open to new roles?",
                "low",
                "spam",
            ),
            (
                "bob@accounting.dept",
                "Lunch next week?",
                "Let's grab a coffee and catch up.",
                "low",
                "client_request",
            ),
            (
                "alerts@datadog.com",
                "Monitor: High Memory Usage",
                "Host db-prod-01 is at 95% memory.",
                "high",
                "escalation",
            ),
            (
                "security@it.net",
                "Password expiring in 3 days",
                "Please update your password today.",
                "medium",
                "notification",
            ),
            (
                "martha@board.org",
                "Dinner reservation?",
                "Do we need to secure a slot for the board dinner?",
                "medium",
                "meeting",
            ),
            (
                "support@zendesk.com",
                "Ticket #90123 has breached SLA",
                "A customer has been waiting 48h.",
                "high",
                "escalation",
            ),
            (
                "jane@marketing.co",
                "Draft review",
                "Could you look at this blog post?",
                "medium",
                "client_request",
            ),
            (
                "delivery@fedex.fake",
                "Package delayed",
                "Your shipment has been rescheduled.",
                "low",
                "spam",
            ),
            (
                "steve@sales.dept",
                "Client X is furious",
                "They want a meeting to discuss the bug.",
                "high",
                "meeting",
            ),
            (
                "investor@funds.vc",
                "Quick question regarding metrics",
                "Can you clarify the churn rate?",
                "high",
                "client_request",
            ),
            (
                "calendar@google.com",
                "Daily schedule",
                "You have 3 meetings today.",
                "low",
                "notification",
            ),
            (
                "hello@gym.local",
                "Membership renewal",
                "Get 10% off if you renew early.",
                "low",
                "spam",
            ),
            (
                "ceo@company.com",
                "All hands pushed back by 1h",
                "Updating calendar invites shortly.",
                "medium",
                "notification",
            ),
            (
                "vendor@software.com",
                "Contract renewal discussion",
                "When are you free next week?",
                "medium",
                "meeting",
            ),
            (
                "devops@internal",
                "GitLab runner down",
                "CI/CD pipelines are failing everywhere.",
                "high",
                "escalation",
            ),
            (
                "catering@food.co",
                "Lunch order confirmation",
                "Your order for 50 people is received.",
                "low",
                "notification",
            ),
            (
                "travel@air.com",
                "Flight check-in available",
                "Check in for tomorrow's flight.",
                "medium",
                "client_request",
            ),
            (
                "spam@spam.spam",
                "WINNER!!! CLAIM PRIZE!!",
                "Click link to get your free iPad.",
                "low",
                "spam",
            ),
            (
                "investor@funds.vc",
                "Pitch deck feedback",
                "It looks good but let's schedule a deep dive.",
                "medium",
                "meeting",
            ),
            (
                "legal@company.com",
                "NDA review needed",
                "Please approve the redlines.",
                "high",
                "client_request",
            ),
            (
                "facilities@building.com",
                "Fire drill tomorrow at 10am",
                "Please evacuate when alarm sounds.",
                "low",
                "notification",
            ),
            (
                "cfo@company.com",
                "Budget cuts",
                "Need to review your Q4 spend plan.",
                "high",
                "escalation",
            ),
            (
                "noreply@github.com",
                "[company/repo] New pull request",
                "PR #405 requires your review.",
                "low",
                "notification",
            ),
            (
                "pr@agency.com",
                "Press release draft",
                "Let me know if this looks good to publish.",
                "medium",
                "client_request",
            ),
            (
                "no-reply@zoom.us",
                "Your cloud recording is ready",
                "Click to view.",
                "low",
                "notification",
            ),
            (
                "sales@saas.com",
                "Last chance for our pro tier",
                "Lock in legacy pricing now.",
                "low",
                "spam",
            ),
            (
                "design@team.com",
                "Logo concepts",
                "Which of these 3 variants do you prefer?",
                "medium",
                "client_request",
            ),
            (
                "hr@company.com",
                "New hire orientation",
                "Can you present the tech stack overview tomorrow?",
                "medium",
                "meeting",
            ),
            (
                "info@bank.com",
                "Important policy update",
                "Our terms of service have changed.",
                "low",
                "notification",
            ),
            (
                "angry_client@huge.com",
                "SYSTEM IS DOWN",
                "We are losing money. Fix it now.",
                "high",
                "escalation",
            ),
            (
                "support@apple.com",
                "Your receipt from Apple",
                "Apple Music subscription renewal.",
                "low",
                "notification",
            ),
            (
                "newsletter@vc.com",
                "Market trends Q3",
                "The latest seed funding rounds categorized.",
                "low",
                "newsletter",
            ),
            (
                "partner@overseas.com",
                "Timezone alignment",
                "Let's find a slot that works for both of us.",
                "medium",
                "meeting",
            ),
            (
                "vp_eng@company.com",
                "Post-mortem review",
                "Need action items from the outage.",
                "high",
                "escalation",
            ),
            (
                "admin@slack.com",
                "Workspace approaching file limit",
                "Please delete old files.",
                "low",
                "notification",
            ),
            (
                "offers@pizza.local",
                "BOGO PIZZA FRIDAY",
                "Use code BOGO.",
                "low",
                "spam",
            ),
            (
                "compliance@audit.gov",
                "Data Request Notice",
                "Please provide logs within 24h.",
                "high",
                "escalation",
            ),
            (
                "mike@intern.dept",
                "Help with setup?",
                "Can you spare a 10 min window to help me with git?",
                "low",
                "meeting",
            ),
            (
                "events@conference.org",
                "Speaker confirmation",
                "You are scheduled for Room B.",
                "medium",
                "client_request",
            ),
            (
                "hr@company.com",
                "Open enrollment",
                "Health benefits selection closes tomorrow.",
                "medium",
                "notification",
            ),
            (
                "alerts@pagerduty.com",
                "CRITICAL ON-CALL PING",
                "Database replication is permanently failing.",
                "high",
                "escalation",
            ),
            (
                "newsletter@dev.to",
                "Top 5 Rust tricks",
                "See why everyone loves Rust.",
                "low",
                "newsletter",
            ),
            (
                "deals@flights.com",
                "Cheap trips to Bali",
                "Fares dropped 40%.",
                "low",
                "spam",
            ),
        ]

        # Inject the massive list into the pool!
        pool.extend(
            [
                self._EmailItem(
                    email_id=0,
                    sender=c[0],
                    subject=c[1],
                    body=c[2],
                    priority=c[3],
                    kind=c[4],
                    expected_action=(
                        "archive"
                        if c[4] in {"spam", "newsletter", "notification"}
                        else "draft_email"
                    ),
                    requires_slot=(c[4] == "meeting"),
                )
                for c in massive_email_data
            ]
        )

        cfg = self._task_config
        target_count = random.randint(cfg.inbox_size_min, cfg.inbox_size_max)
        additional_count = target_count - len(required)
        selected = required + random.sample(pool, k=min(additional_count, len(pool)))
        random.shuffle(selected)
        return self._assign_ids(selected)

    @staticmethod
    def _assign_ids(
        items: List["EmailtriageEnvironment._EmailItem"],
    ) -> List["EmailtriageEnvironment._EmailItem"]:
        """Assign sequential IDs starting from 101."""
        result: List[EmailtriageEnvironment._EmailItem] = []
        for idx, item in enumerate(items, start=1):
            result.append(
                EmailtriageEnvironment._EmailItem(
                    email_id=100 + idx,
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
        return result

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, reward: float, done: bool) -> EmailtriageObservation:
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
                "task_id": self._state.task_id,
                "emails_total": len(self._emails),
                "emails_processed": len(self._state.processed_email_ids),
                "queried_calendar": self._state.queried_calendar,
            },
        )

    # ------------------------------------------------------------------
    # Action routing & grading
    # ------------------------------------------------------------------

    def _route_action(self, action: EmailtriageAction) -> Tuple[float, str]:
        """Route action to task logic and compute reward in [0, 1]."""
        if action.action_type == "query_calendar":
            pending_scheduling_count = sum(
                1
                for email in self._emails
                if email.status == "unread" and email.requires_slot
            )
            self._state.queried_calendar = True
            reward = 0.1 + min(0.36, pending_scheduling_count * 0.18)
            if "calendar_queried_once" in self._triggered_events:
                reward *= 0.7
                feedback = "Calendar queried again; small value after first lookup."
            else:
                self._triggered_events.add("calendar_queried_once")
                feedback = "Calendar queried successfully."
            return self._clamp_reward(reward), feedback

        target = self._find_email(action.target_email_id)
        if target is None:
            return 0.01, "Invalid or missing target_email_id."

        if action.action_type == "read":
            self._last_read_payload = [
                (
                    f"Email {target.email_id} from {target.sender}: "
                    f"{target.subject} | {target.body}"
                )
            ]
            priority_bonus = {
                "high": 0.18,
                "medium": 0.12,
                "low": 0.08,
            }.get(target.priority, 0.1)
            base_reward = 0.07 if target.status == "unread" else 0.02
            reward = base_reward + priority_bonus
            return reward, "Email content returned to the agent."

        if action.action_type == "archive":
            if target.status != "unread":
                return 0.05, "Email already processed."

            if target.kind in {"spam", "newsletter", "notification"}:
                target.status = "archived"
                self._mark_processed(target.email_id)
                kind_bonus = {
                    "spam": 0.18,
                    "newsletter": 0.15,
                    "notification": 0.12,
                }.get(target.kind, 0.1)
                return 0.62 + kind_bonus, "Correctly archived low-value email."

            penalty_like = 0.08
            if target.priority == "high":
                penalty_like = 0.03
            return (
                penalty_like,
                "Archived an email that likely needed a response.",
            )

        if action.action_type == "draft_email":
            return self._grade_draft_action(target, action)

        return 0.01, "Unsupported action type."

    def _grade_draft_action(
        self, target: _EmailItem, action: EmailtriageAction
    ) -> Tuple[float, str]:
        """Grade draft_email actions with partial rewards."""
        if target.status != "unread":
            return 0.05, "Email already processed."

        reward = 0.15
        feedback_parts: List[str] = []

        if target.kind in {"meeting", "client_request", "escalation"}:
            reward += 0.2
            feedback_parts.append("Draft action is appropriate for this email.")
        else:
            feedback_parts.append("Drafting may be unnecessary for this email.")

        draft_quality = self._draft_quality_score(action.draft_content)
        reward += 0.4 * draft_quality

        # Detect hallucination - vague context with specific details
        hallucination_penalty = self._detect_hallucination(target, action)
        if hallucination_penalty > 0:
            reward = max(0.0, reward - hallucination_penalty)
            feedback_parts.append(
                "WARNING: Hallucinated details detected. Specific information was assumed without proper context."
            )

        if target.requires_slot:
            if self._state.queried_calendar:
                reward += 0.12
                feedback_parts.append("Checked calendar before proposing a slot.")
            else:
                feedback_parts.append("Calendar should be queried before scheduling.")

            if action.proposed_slot in self._state.calendar_slots:
                reward += 0.18
                feedback_parts.append("Proposed a valid available slot.")
            else:
                feedback_parts.append("Missing or invalid proposed slot.")

        if target.priority == "high":
            reward += 0.08
            feedback_parts.append("Handled high-priority thread.")

        if target.kind == "escalation" and "today" in action.draft_content.lower():
            reward += 0.05
            feedback_parts.append("Draft included urgency acknowledgement.")

        reward = self._clamp_reward(reward)
        if reward >= 0.55:
            target.status = "drafted"
            self._mark_processed(target.email_id)
        return reward, " ".join(feedback_parts)

    # ------------------------------------------------------------------
    # Dynamic events (hard mode only)
    # ------------------------------------------------------------------

    def _apply_dynamic_events(self) -> str:
        """Apply deterministic dynamic events that alter state mid-episode."""
        if (
            self._state.step_count == 3
            and "new_urgent_email" not in self._triggered_events
        ):
            new_email = self._EmailItem(
                email_id=max(e.email_id for e in self._emails) + 1,
                sender="ceo@company.com",
                subject="Urgent: board pre-read needed today",
                body="Please draft a concise response with next steps.",
                priority="high",
                kind="escalation",
                expected_action="draft_email",
                status="unread",
                requires_slot=False,
            )
            self._emails.append(new_email)
            self._triggered_events.add("new_urgent_email")
            return "Dynamic update: a new urgent email arrived."

        unread_scheduling = [
            email
            for email in self._emails
            if email.status == "unread" and email.requires_slot
        ]
        if (
            self._state.step_count >= 4
            and unread_scheduling
            and len(self._state.calendar_slots) >= 2
            and "calendar_slot_removed" not in self._triggered_events
        ):
            removed_slot = random.choice(self._state.calendar_slots)
            self._state.calendar_slots.remove(removed_slot)
            self._triggered_events.add("calendar_slot_removed")
            return (
                "Dynamic update: calendar changed and slot "
                f"{removed_slot} is no longer available."
            )

        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_reward(value: float) -> float:
        """Clamp reward values to (0, 1) exclusive."""
        return max(0.01, min(0.99, value))

    @staticmethod
    def _draft_quality_score(draft_content: str) -> float:
        """Score draft quality with deterministic heuristics."""
        clean_text = draft_content.strip().lower()
        if not clean_text:
            return 0.01

        score = 0.01
        if len(clean_text) >= 40:
            score += 0.45
        if "thank" in clean_text:
            score += 0.2
        if "meeting" in clean_text or "schedule" in clean_text or "slot" in clean_text:
            score += 0.2
        if clean_text.endswith(".") or clean_text.endswith("!"):
            score += 0.15
        return max(0.01, min(0.99, score))

    def _detect_reward_hacking(self, action: EmailtriageAction) -> float:
        """Detect and penalize reward hacking attempts.

        Returns penalty value (0.0 if no hacking detected).
        """
        suspicious_patterns = [
            "os.system",
            "subprocess",
            "exec(",
            "eval(",
            "rm -rf",
            "del /",
            "format:",
            "../",
            "../../",
            "/etc/",
            "<script>",
            "javascript:",
            "onerror=",
            "import ",
            "require(",
            "__import__",
            "while True",
            "for i in range(1000000)",
            "reward = 1.0",
            "score = 100",
            "force_success",
            "inject",
            "bypass",
            "hack",
            "exploit",
            "DROP TABLE",
            "DELETE FROM",
            "INSERT INTO",
            "GRANT ",
            "REVOKE ",
            "--",
            ";--",
        ]

        content_to_check = (
            action.draft_content.lower()
            + " "
            + str(action.target_email_id)
            + " "
            + action.proposed_slot.lower()
        )

        for pattern in suspicious_patterns:
            if pattern.lower() in content_to_check:
                return 0.0

        if len(action.draft_content) > 10000:
            return 0.0

        return 0.0

    def _detect_hallucination(
        self, target: _EmailItem, action: EmailtriageAction
    ) -> float:
        """Detect hallucination when agent assumes specific details not in context.

        Returns penalty value (0.0 if no hallucination detected).
        """
        if target.kind not in {"client_request", "meeting"}:
            return 0.0

        vague_indicators = ["the thing", "it", "that", "stuff", "push", "move"]
        body_lower = target.body.lower()
        subject_lower = target.subject.lower()

        is_vague = any(
            indicator in body_lower or indicator in subject_lower
            for indicator in vague_indicators
        )

        if not is_vague:
            return 0.0

        specific_terms = [
            "project",
            "meeting",
            "call",
            "demo",
            "release",
            "launch",
            "deadline",
            "sprint",
            "milestone",
            "contract",
            "proposal",
            "quarter",
            "budget",
            "proposal",
            "roadmap",
        ]

        draft_lower = action.draft_content.lower()
        hallucinated_terms = [term for term in specific_terms if term in draft_lower]

        if hallucinated_terms and is_vague:
            return 0.25

        return 0.0

    def _find_email(self, email_id: int) -> Optional[_EmailItem]:
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
