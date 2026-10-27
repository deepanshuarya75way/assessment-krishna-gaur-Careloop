"""Agent 2: Care Plan generator + initial outreach."""
from __future__ import annotations

import logging
from typing import Any

from app.agents.state import PatientState
from app.db.client import safe_insert, safe_select
from app.tools import kg as kg_tools
from app.tools.email_tool import (
    build_caregiver_email_html,
    build_patient_welcome_html,
    send_email,
)
from app.tools.llm import chat_json
from app.tools.voice import public_audio_url, synthesize_sync
from app.tools.whatsapp import format_whatsapp_message, send_whatsapp

log = logging.getLogger(__name__)


def _persist_reasoning(patient_id: str, observed: dict, inferred: dict, decided: str, tools: list[str]):
    safe_insert(
        "reasoning_traces",
        {
            "patient_id": patient_id,
            "agent_name": "care_plan",
            "observed": observed,
            "inferred": inferred,
            "decided": decided,
            "tools_called": tools,
        },
    )


def _patient_record(patient_id: str) -> dict[str, Any]:
    rows = safe_select("patients", match={"id": patient_id}, limit=1)
    return rows[0] if rows else {}


def care_plan_node(state: PatientState) -> PatientState:
    patient_id = state["patient_id"]
    clinical = state.get("clinical_extracted", {}) or {}
    sdoh_profile = state.get("sdoh_profile", {}) or {}
    g = state.get("knowledge_graph")
    kg_high = kg_tools.kg_highlights(g) if g is not None else {}
    patient = state.get("patient_record") or _patient_record(patient_id)

    tools_called: list[str] = []

    # Step 1: generate plan via Groq with chain-of-thought prompt
    plan = (
        chat_json(
            "care_plan_generator",
            diagnosis=clinical.get("diagnosis", ""),
            medications=clinical.get("medications", []),
            comorbidities=clinical.get("comorbidities", []),
            sdoh_profile=sdoh_profile,
            risk_score=state.get("risk_score", 0.0),
            kg_highlights=kg_high,
        )
        or {}
    )
    tools_called.append("llm:care_plan_generator")
    if not isinstance(plan,dict) or not plan:
        raise ValueError("Care plan generation returned an invalid plan")

    # Defensive defaults
    plan.setdefault("channel", state.get("channel", "whatsapp_text"))
    plan.setdefault("check_in_times_per_day", int(state.get("check_in_times_per_day") or 3))
    # Language posture: prefer the SDOH-derived language carried on state; only
    # fall back to English if nothing upstream produced a code. This is what lets
    # the gTTS voice path actually speak Hindi / Tamil / Bengali / etc.
    plan.setdefault("language", state.get("language") or sdoh_profile.get("language") or "en")
    plan.setdefault("simplification_level", "high" if sdoh_profile.get("literacy_level") == "low" else "medium")
    plan.setdefault("check_in_cadence", "daily")
    plan.setdefault("check_in_time", "09:00")
    plan.setdefault("caregiver_loop_enabled", sdoh_profile.get("caregiver_risk") in ("medium", "high"))
    plan.setdefault("red_flag_symptoms", kg_high.get("red_flags", []))

    # Persist plan
    inserted = safe_insert(
        "care_plans",
        {
            "patient_id": patient_id,
            "plan_json": plan,
            "reasoning_trace": plan.get("reasoning", ""),
        },
    )
    if inserted is None:
        raise ValueError("Failed to persist care plan")
        
    tools_called.append("db:insert(care_plans)")

    retry_only= bool(state.get("retry_care_plan".False))

    # Step 2: WhatsApp welcome to patient
    name = patient.get("name") or "Patient"
    if not state.get("retry_care_plan",False):
        welcome = _build_welcome(name, plan)
        media_url = None
        if plan.get("channel") == "whatsapp_voice":
            audio_path = synthesize_sync(welcome, language=plan.get("language", "en"))
            if audio_path:
                tools_called.append("voice:gtts")
                media_url = public_audio_url(audio_path) or None
                
                
            if patient.get("phone"):
                result = send_whatsapp(patient["phone"], welcome, media_url=media_url)
                tools_called.append(f"whatsapp:welcome({'mock' if result.get('mock') else 'live'})")
                safe_insert(
                "interactions",
                {
                    "patient_id": patient_id,
                    "channel": "whatsapp",
                    "direction": "outbound",
                    "content": welcome,
                    "classification": "welcome",
                    "agent_decision": "send_welcome",
                },
            )
        
        # Step 3a: Welcome email to patient
    patient_email = patient.get("email")
    if patient_email:
        p_subj, p_html = build_patient_welcome_html(name, plan)
        p_body = (
            f"Welcome to CareLoop, {name}!\n\n"
            f"You are enrolled in CareLoop post-discharge monitoring.\n"
            f"Check-in schedule: {plan.get('check_in_times_per_day', 3)}x daily "
            f"via {plan.get('channel', 'whatsapp_text')}.\n\n"
            f"— The CareLoop Team"
        )
        result = send_email(patient_email, p_subj, p_body, html=p_html)
        tools_called.append(f"email:patient_welcome({'mock' if result.get('mock') else 'live'})")

    # Step 3b: Email caregiver summary (send whenever caregiver_email is present)
    caregiver_email = patient.get("caregiver_email")
    if caregiver_email:
        cg_subj, cg_html = build_caregiver_email_html(name, plan, clinical, sdoh_profile)
        cg_body = _build_caregiver_email(name, plan, clinical, sdoh_profile)
        result = send_email(caregiver_email, cg_subj, cg_body, html=cg_html)
        tools_called.append(f"email:caregiver_summary({'mock' if result.get('mock') else 'live'})")

    # Step 4: schedule check-ins (3× daily by default)
    try:
        from app.scheduler.jobs import schedule_daily_checkin

        times_per_day = int(plan.get("check_in_times_per_day") or state.get("check_in_times_per_day") or 3)
        schedule_daily_checkin(patient_id, plan.get("check_in_time", "09:00"), times_per_day=times_per_day)
        tools_called.append(f"scheduler:daily_checkin×{times_per_day}")
    except Exception as e:
        log.warning("Scheduler unavailable: %s", e)

    decided = (
        f"Care plan: channel={plan.get('channel')}, cadence={plan.get('check_in_cadence')} @ "
        f"{plan.get('check_in_time')} IST, simplification={plan.get('simplification_level')}, "
        f"caregiver_loop={plan.get('caregiver_loop_enabled')}."
    )
    _persist_reasoning(
        patient_id,
        observed={"sdoh_profile": sdoh_profile, "clinical_summary": {"diagnosis": clinical.get("diagnosis")}},
        inferred={"plan": plan},
        decided=decided,
        tools=tools_called,
    )

    state["care_plan"] = plan
    state["channel"] = plan.get("channel")
    state["language"] = plan.get("language")
    state.setdefault("tools_called", []).extend(tools_called)
    state.setdefault("reasoning_steps", []).append(
        {
            "agent": "care_plan",
            "observed": {"risk_score": state.get("risk_score")},
            "inferred": {"plan_keys": list(plan.keys())},
            "decided": decided,
            "tools_called": tools_called,
        }
    )
    return state


def _checkin_cadence_description(plan: dict) -> str:
    """Human-readable check-in schedule for patient/caregiver messages."""
    n = int(plan.get("check_in_times_per_day") or 1)
    time_ = plan.get("check_in_time", "09:00")
    if n == 1:
        return f"once a day at {time_}"
    if n == 2:
        return "twice a day (morning and evening)"
    if n == 3:
        return "3 times a day (morning, afternoon, and evening)"
    if n == 4:
        return "4 times a day (morning, midday, afternoon, and evening)"
    if n == 6:
        return "6 times a day (every 2 hours from morning to evening)"
    return f"{n} times a day"


def _build_welcome(name: str, plan: dict) -> str:
    cadence_desc = _checkin_cadence_description(plan)
    return format_whatsapp_message(
        title="Welcome",
        body=(
            f"Hi {name}, this is CareLoop - your post-discharge care companion.\n"
            f"I'll check in with you {cadence_desc} for the next 30 days.\n"
            "Reply any time if something feels off - I'll loop in your doctor right away."
        ),
    )
    return (
        f"🩺 CareLoop\n"
        f"Hi {name}, this is CareLoop — your post-discharge care companion.\n"
        f"I'll check in with you {cadence_desc} for the next 30 days.\n"
        f"Reply any time if something feels off — I'll loop in your doctor right away."
    )


def _build_caregiver_email(name: str, plan: dict, clinical: dict, sdoh: dict) -> str:
    sched = plan.get("medication_schedule", [])
    sched_lines = "\n".join(
        f"  • {m.get('med')} at {m.get('time')} — {m.get('instruction')}" for m in sched
    ) or "  (none)"
    flags = ", ".join(plan.get("red_flag_symptoms", [])) or "(none listed)"
    cadence_desc = _checkin_cadence_description(plan)
    return (
        f"Hi,\n\nCareLoop is now monitoring {name} after discharge.\n\n"
        f"Check-in schedule: {cadence_desc} via {plan.get('channel')}\n\n"
        f"Medication schedule:\n{sched_lines}\n\n"
        f"Watch for these red-flag symptoms — message us immediately if you see them:\n  {flags}\n\n"
        f"You'll receive an email if anything escalates.\n\n— CareLoop"
    )
