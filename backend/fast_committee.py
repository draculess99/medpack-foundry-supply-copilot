"""Fast local MedPack committee endpoint.

This module exists because the main demo button must never hang.  It avoids:
- remote LLM providers
- streaming responses
- joblib / XGBoost model loading
- memory-event writes

It still uses Stage 2 usable-stock logic, shortage rules, and packing priority.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from backend.shortage_rules import calculate_shortage_risk
from backend.packing_optimizer import calculate_priority_and_pack_quantity
from backend.traceability import ensure_traceability_fields, load_inventory_state
from backend.task_manager import list_tasks
from backend.usable_stock import build_usable_stock_analysis
from backend.stage3_action_plan import build_stage3_action_plan
from backend.stage4_roi import build_stage4_roi_analysis
from backend.stage5_command_center import build_stage5_command_center

try:
    from backend.rag_manager import rag_manager
except ImportError:
    rag_manager = None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fast_local_forecast(telemetry: Mapping[str, Any]) -> float:
    """Small deterministic forecast that cannot block on model loading.

    The formula mirrors the app's FallbackModel so the committee remains useful
    even when the full ML route is unavailable or slow.
    """
    recent_usage = _as_float(telemetry.get("recent_usage_rate"), 5.0)
    volume = _as_float(telemetry.get("patient_volume"), 10.0)
    acuity = _as_float(telemetry.get("acuity_level"), 2.0)
    procedures = _as_float(telemetry.get("procedure_count"), 4.0)
    delay = _as_float(telemetry.get("supplier_delay_days"), 2.0)

    pred = (recent_usage * 0.8) + (volume * acuity * 0.15) + (procedures * 0.5)
    if delay > 4.0:
        pred += 2.0
    return max(0.0, float(pred))


def build_fast_committee_payload(telemetry: Mapping[str, Any]) -> Dict[str, Any]:
    telemetry = dict(telemetry or {})
    item = telemetry.get("item_name", "Unknown Item")
    dept = telemetry.get("department", "Unknown Department")

    predicted = fast_local_forecast(telemetry)

    try:
        inventory_records = ensure_traceability_fields() or load_inventory_state(enrich=True)
    except Exception:
        inventory_records = []

    try:
        tasks = list_tasks(limit=500)
    except Exception:
        tasks = []

    try:
        usable_analysis = build_usable_stock_analysis(
            telemetry,
            predicted_24h_demand=predicted,
            inventory_records=inventory_records,
            tasks=tasks,
        )
        shortage_result = usable_analysis.get("shortage_risk_using_usable_stock", {})
    except Exception as exc:
        current_stock = int(_as_float(telemetry.get("current_stock"), 0.0))
        shortage_result = calculate_shortage_risk(predicted, current_stock)
        shortage_result["stock_basis"] = "current_stock_fallback"
        shortage_result["reasoning"] += f" Fast committee fallback used because usable-stock analysis failed: {exc}"
        usable_analysis = {
            "stage": "Fast Local Committee Fallback",
            "item_name": item,
            "department": dept,
            "predicted_24h_demand": round(predicted, 2),
            "stock_basis": "current_stock_fallback",
            "total_stock": current_stock,
            "usable_stock": current_stock,
            "unsafe_stock": 0,
            "active_task_reserved_stock": 0,
            "true_shortage_gap": round(predicted - current_stock, 2),
            "recommended_stage2_action": "Use current stock fallback and retry full Stage 2 panel after backend restart.",
            "safety_notes": ["Fast fallback used; no remote LLM or streaming call was made."],
            "shortage_risk_using_usable_stock": shortage_result,
        }

    priority_result = calculate_priority_and_pack_quantity(item, dept, shortage_result, telemetry)

    try:
        stage3_plan = build_stage3_action_plan(
            telemetry,
            predicted_24h_demand=predicted,
            inventory_records=inventory_records,
            tasks=tasks,
        )
    except Exception as exc:
        stage3_plan = {
            "stage": "Stage 3 - Supplier + Transfer Intelligence",
            "status": "fallback",
            "error": str(exc),
            "best_action": "Stage 2 action only",
            "final_recommendation": "Stage 3 action-plan builder failed, so MedPack kept the Stage 2 packing recommendation.",
            "recommended_sequence": ["Use Stage 2 packing recommendation and retry Stage 3 after restart."],
        }

    try:
        stage4_roi = build_stage4_roi_analysis(
            telemetry,
            predicted_24h_demand=predicted,
            inventory_records=inventory_records,
            tasks=tasks,
            stage3_plan=stage3_plan,
        )
    except Exception as exc:
        stage4_roi = {
            "stage": "Stage 4 - Cost, Waste & ROI Executive Dashboard",
            "status": "fallback",
            "error": str(exc),
            "executive_recommendation": "Stage 4 ROI view failed, so MedPack kept Stage 3 operational recommendation.",
            "net_value_estimate": 0.0,
        }

    try:
        stage5_command = build_stage5_command_center(
            telemetry,
            predicted_24h_demand=predicted,
            inventory_records=inventory_records,
            tasks=tasks,
            stage3_plan=stage3_plan,
            stage4_roi=stage4_roi,
        )
    except Exception as exc:
        stage5_command = {
            "stage": "Stage 5 - Agentic Command Center",
            "status": "fallback",
            "error": str(exc),
            "priority_code": "P3",
            "command_status": "GREEN - Monitor",
            "commander_decision": "Stage 5 command-center builder failed, so MedPack kept the Stage 4 recommendation.",
            "control_tower_summary": "Stage 5 command center unavailable; use Stage 4/Stage 3 output.",
            "action_cards": [],
        }

    risk = shortage_result.get("risk_level", "Low")
    forecast = round(predicted, 2)
    usable = usable_analysis.get("usable_stock", shortage_result.get("current_stock", telemetry.get("current_stock", 0)))
    total = usable_analysis.get("total_stock", telemetry.get("current_stock", 0))
    unsafe = usable_analysis.get("unsafe_stock", 0)
    reserved = usable_analysis.get("active_task_reserved_stock", 0)
    gap = usable_analysis.get("true_shortage_gap", shortage_result.get("shortage_gap", 0))
    action = usable_analysis.get("recommended_stage2_action") or priority_result.get("recommended_action")

    committee = {
        "demand_forecast_agent": (
            f"Fast local forecast estimates {forecast} units of {item} will be needed in the next 24 hours for {dept}. "
            "This button uses a deterministic no-token safety path so it cannot hang on a remote LLM."
        ),
        "inventory_risk_agent": (
            f"Inventory is judged using Stage 2 usable stock: total stock {total}, usable stock {usable}, "
            f"unsafe stock {unsafe}, and active task-reserved stock {reserved}. True shortage gap is {gap}."
        ),
        "packing_priority_agent": priority_result.get("reasoning", "Packing priority calculated locally."),
        "clinical_safety_agent": (
            f"Clinical safety risk is {risk}. "
            "Expired, recalled, and task-reserved stock are not counted as available bedside supply."
        ),
        "final_recommendation_agent": stage3_plan.get("final_recommendation", priority_result.get("recommended_action", action)),
        "stage3_control_tower_agent": stage3_plan.get("control_tower_summary", stage3_plan.get("final_recommendation", "Stage 3 action plan unavailable.")),
        "stage4_financial_impact_agent": stage4_roi.get("control_tower_summary", stage4_roi.get("executive_recommendation", "Stage 4 financial view unavailable.")),
        "stage5_command_center_agent": stage5_command.get("control_tower_summary", stage5_command.get("commander_decision", "Stage 5 command center unavailable.")),
        "committee_summary": (
            f"MedPack Stage 5 command center: {item} in {dept} is {stage5_command.get('priority_code', 'P3')} "
            f"/{stage5_command.get('command_status', 'GREEN - Monitor')}. "
            f"Commander decision: {stage5_command.get('commander_decision', stage3_plan.get('final_recommendation', action))}"
        ),
        "requested_agent_mode": str(telemetry.get("agent_mode", "local")),
        "actual_agent_mode": "fast_local_no_remote",
        "remote_llm_enabled": False,
        "remote_llm_key_present": False,
        "remote_model": None,
        "tokens_used": 0,
        "fallback_mode": True,
        "mode_note": "Freeze Fix v3: fast local committee endpoint used. No Groq, Gemini, streaming, joblib model load, or remote LLM call was made.",
        "rag_knowledge": rag_manager.query_rag(f"{item} in {dept}") if (rag_manager and getattr(rag_manager, "_initialized", False)) else "(RAG optional for fast path)",
    }

    return {
        "telemetry": telemetry,
        "prediction": {
            "predicted_24h_demand": forecast,
            "model_type": "fast_local_fallback_formula",
        },
        "shortage_risk": shortage_result,
        "packing_priority": priority_result,
        "usable_stock_analysis": usable_analysis,
        "stage3_action_plan": stage3_plan,
        "stage4_roi_analysis": stage4_roi,
        "stage5_command_center": stage5_command,
        "memory_state": {
            "mode": "fast_local_no_write",
            "note": "Fast committee does not write memory events, preventing file I/O from blocking the demo button.",
        },
        "committee": committee,
    }
