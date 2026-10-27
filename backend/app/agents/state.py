"""Shared PatientState passed between LangGraph agents."""
from __future__ import annotations

from typing import Any, TypedDict


class ReasoningStep(TypedDict, total=False):
    agent: str
    observed: dict[str, Any]
    inferred: dict[str, Any]
    decided: str
    tools_called: list[str]


class PatientState(TypedDict, total=False):
    patient_id: str
    raw_discharge_text: str
    sdoh_responses: dict[str, Any]
    clinical_extracted: dict[str, Any]
    sdoh_profile: dict[str, Any]
    knowledge_graph: Any  # NetworkX DiGraph (in-memory)
    knowledge_graph_json: dict[str, Any]
    care_plan: dict[str, Any]
    current_message: str
    classification: dict[str, Any]
    decision: str
    tools_called: list[str]
    reasoning_steps: list[ReasoningStep]
    risk_score: float
    language: str
    channel: str
    # Patient context loaded from DB
    patient_record: dict[str, Any]
    # Check-in frequency set at onboarding
    check_in_times_per_day: int  # default 3
    # Inbound trigger metadata (engagement)
    triggered_by: str  # "cron" | "inbound" | "onboarding"
    # Outgoing transcript collectors populated by engagement_node.
    outgoing_messages: list[dict[str, Any]]
    outgoing_emails: list[dict[str, Any]]
    retry_care_plan:bool




def empty_state() -> PatientState:
    return PatientState(
        tools_called=[],
        retry_care_plan=False,
        reasoning_steps=[],
        risk_score=0.0,
        language="en",
        channel="whatsapp_text",
        outgoing_messages=[],
        outgoing_emails=[],
    )
