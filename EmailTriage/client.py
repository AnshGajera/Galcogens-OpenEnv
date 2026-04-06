# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Emailtriage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import EmailtriageAction, EmailtriageObservation, EmailtriageState


class EmailtriageEnv(
    EnvClient[EmailtriageAction, EmailtriageObservation, EmailtriageState]
):
    """
    Client for the Emailtriage Environment.

    This client maintains a persistent WebSocket
    connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with EmailtriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(options={"task_id": "easy"})
        ...     print(result.observation.inbox_remaining)
        ...
        ...     result = client.step(EmailtriageAction(
        ...         action_type="archive", target_email_id=101))
        ...     print(result.reward)

    Example with Docker:
        >>> client = EmailtriageEnv.from_docker_image("emailtriage-env:latest")
        >>> try:
        ...     result = client.reset(options={"task_id": "medium"})
        ...     result = client.step(EmailtriageAction(
        ...         action_type="read", target_email_id=101))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: EmailtriageAction) -> Dict:
        """Convert EmailtriageAction to JSON payload for step message."""
        return {
            "action_type": action.action_type,
            "target_email_id": action.target_email_id,
            "draft_content": action.draft_content,
            "proposed_slot": action.proposed_slot,
        }

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[EmailtriageObservation]:
        """Parse server response into StepResult[EmailtriageObservation]."""
        obs_data = payload.get("observation", {})
        observation = EmailtriageObservation(
            inbox_preview=obs_data.get("inbox_preview", []),
            returned_emails=obs_data.get("returned_emails", []),
            calendar_slots=obs_data.get("calendar_slots", []),
            last_action_result=obs_data.get("last_action_result", ""),
            conversation_history=obs_data.get("conversation_history", []),
            inbox_remaining=obs_data.get("inbox_remaining", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EmailtriageState:
        """Parse server response into EmailtriageState object."""
        return EmailtriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "hard"),
            inbox=payload.get("inbox", []),
            calendar_slots=payload.get("calendar_slots", []),
            queried_calendar=payload.get("queried_calendar", False),
            processed_email_ids=payload.get("processed_email_ids", []),
        )
