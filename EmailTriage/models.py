# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the EmailTriage environment."""

from typing import Dict, List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class EmailtriageAction(Action):
    """Action schema for proactive email triage tools."""

    action_type: Literal[
        "read", "archive", "query_calendar", "draft_email"
    ] = Field(
        ...,
        description="Tool/action selected by the agent",
    )
    target_email_id: int = Field(
        default=-1,
        description="Target email identifier for read/archive/draft_email",
    )
    draft_content: str = Field(
        default="",
        description="Drafted response text when action_type is draft_email",
    )
    proposed_slot: str = Field(
        default="",
        description=(
            "Proposed calendar slot for scheduling drafts. "
            "Should be one of the available slots when applicable"
        ),
    )


class EmailtriageState(State):
    """Environment state stored between steps."""

    inbox: List[Dict[str, str]] = Field(
        default_factory=list,
        description="In-memory inbox records with mutable status",
    )
    calendar_slots: List[str] = Field(
        default_factory=list,
        description="Current calendar availability",
    )
    queried_calendar: bool = Field(
        default=False,
        description="Whether the calendar has been queried this episode",
    )
    processed_email_ids: List[int] = Field(
        default_factory=list,
        description="Emails that are finished via archive or draft",
    )


class EmailtriageObservation(Observation):
    """Observation schema returned after each step."""

    inbox_preview: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Visible inbox metadata for unread items",
    )
    returned_emails: List[str] = Field(
        default_factory=list,
        description="Expanded email text returned by read action",
    )
    calendar_slots: List[str] = Field(
        default_factory=list,
        description="Calendar availability in the current episode",
    )
    last_action_result: str = Field(
        ...,
        description="Evaluator feedback for the most recent action",
    )
    inbox_remaining: int = Field(
        default=0,
        description="Number of unread emails left",
    )
    conversation_history: List[str] = Field(
        default_factory=list,
        description="Recent action trace and feedback history",
    )
