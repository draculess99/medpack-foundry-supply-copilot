import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.model import load_model_and_predict
from backend.shortage_rules import calculate_shortage_risk
from backend.packing_optimizer import calculate_priority_and_pack_quantity
from backend.supply_memory import (
    load_supply_memory,
    update_supply_memory_after_actual,
    append_memory_event
)
from backend.agents.adk_agents import run_committee, run_committee_stream
from backend.packing_queue import build_packing_queue
from backend.traceability import ensure_traceability_fields, enrich_inventory_records, load_inventory_state
from backend.compliance_rules import build_compliance_alerts
from backend.scan_events import append_scan_event, list_scan_events, VALID_SCAN_EVENTS
from backend.par_recommendation import recommend_par
from backend.task_manager import create_task, update_task, list_tasks, TASK_STATUSES
from backend.usable_stock import build_usable_stock_analysis
from backend.fast_committee import build_fast_committee_payload
from backend.transfer_optimizer import build_transfer_recommendation
from backend.supplier_risk import build_supplier_risk
from backend.substitution_engine import build_substitute_options
from backend.stage3_action_plan import build_stage3_action_plan
from backend.stage4_roi import build_stage4_roi_analysis, load_stage4_assumptions
from backend.stage5_command_center import build_stage5_command_center, load_stage5_playbooks
from backend.stage6_whatif_simulator import build_stage6_whatif_simulation, build_stage6_scenario_benchmark, load_stage6_scenarios
from flask import Response

app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get("MEDPACK_BACKEND_PORT", 5001))


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_committee_agent_mode(telemetry):
    """Return the safest committee mode.

    The main demo should never freeze because of remote LLM or streaming calls.
    By default the committee runs locally unless MEDPACK_FORCE_LOCAL_COMMITTEE=false
    and the payload explicitly asks for remote mode.
    """
    force_local_payload = _truthy(telemetry.get("force_local_committee", False))
    force_local_env = _truthy(os.environ.get("MEDPACK_FORCE_LOCAL_COMMITTEE", "true"))
    if force_local_payload or force_local_env:
        return "local"
    requested = str(telemetry.get("agent_mode", os.environ.get("DEFAULT_AGENT_MODE", "local"))).strip().lower()
    return "remote" if requested == "remote" else "local"

_ML_READY = False

@app.route("/health", methods=["GET"])
def health():
    remote_enabled = str(os.environ.get("USE_LLM_AGENTS", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
    foundry_enabled = str(os.environ.get("USE_FOUNDRY_AGENT", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
    return jsonify({
        "status": "ok",
        "service": "MedPack AI backend",
        "default_agent_mode": os.environ.get("DEFAULT_AGENT_MODE", "local"),
        "remote_llm_enabled": remote_enabled or foundry_enabled,
        "groq_key_present": bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "gemini_key_present": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "safe_default": "local_zero_token",
        "stage1_control_tower": True,
        "stage2_usable_stock_integration": True,
        "stage3_supplier_transfer_intelligence": True,
        "stage4_cost_waste_roi": True,
        "stage5_agentic_command_center": True,
        "stage6_whatif_surge_simulator": True
    })

@app.route("/ready", methods=["GET"])
def ready():
    """Returns 200 only when the deterministic prediction engine is ready."""
    if _ML_READY:
        return jsonify({"ready": True, "status": "ok"})
    return jsonify({"ready": False, "status": "starting"}), 503

@app.route("/api/data-sources", methods=["GET"])
def data_sources():
    path = "database/data_sources.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Data sources schema not found"}), 404

@app.route("/api/inventory", methods=["GET"])
def inventory():
    path = "database/inventory_state.json"
    if os.path.exists(path):
        records = ensure_traceability_fields()
        return jsonify(records)
    
    # Fallback to loading some from processed dataset
    processed_path = "database/processed/medpack_training_data.csv"
    if os.path.exists(processed_path):
        import pandas as pd
        df = pd.read_csv(processed_path)
        # return top 50 unique items/departments
        df_unique = df.drop_duplicates(subset=["item_name", "department"]).head(50)
        return jsonify(enrich_inventory_records(df_unique.to_dict(orient="records")))
        
    return jsonify([]), 200


@app.route("/api/compliance-alerts", methods=["GET", "POST"])
def compliance_alerts():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.args.to_dict()
    department = payload.get("department") or None
    expiration_window_days = int(payload.get("expiration_window_days", 30))
    records = ensure_traceability_fields() or load_inventory_state(enrich=True)
    if not records:
        records = inventory().get_json() or []
    return jsonify(build_compliance_alerts(records, department=department, expiration_window_days=expiration_window_days))


@app.route("/api/scan-events", methods=["GET"])
def scan_events():
    limit = int(request.args.get("limit", 25))
    return jsonify(list_scan_events(limit=limit))


@app.route("/api/scan-event", methods=["POST"])
def scan_event():
    try:
        event = append_scan_event(request.get_json(silent=True) or {})
        return jsonify({"status": "success", "event": event})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "valid_event_types": VALID_SCAN_EVENTS}), 400


@app.route("/api/par-recommendation", methods=["POST"])
def par_recommendation():
    data = request.get_json(silent=True) or {}
    return jsonify(recommend_par(data))


@app.route("/api/packing-tasks", methods=["GET", "POST", "PATCH"])
def packing_tasks():
    try:
        if request.method == "GET":
            limit = int(request.args.get("limit", 25))
            return jsonify({"tasks": list_tasks(limit=limit), "valid_statuses": TASK_STATUSES})
        payload = request.get_json(silent=True) or {}
        if request.method == "POST":
            return jsonify({"status": "success", "task": create_task(payload), "valid_statuses": TASK_STATUSES})
        return jsonify({"status": "success", "task": update_task(payload), "valid_statuses": TASK_STATUSES})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "valid_statuses": TASK_STATUSES}), 400



@app.route("/api/stage3-reference-data", methods=["GET"])
def stage3_reference_data():
    """Return Stage 3 seed data for vendors and substitutions."""
    result = {}
    for key, path in {
        "vendor_state": "database/vendor_state.json",
        "substitution_rules": "database/substitution_rules.json",
    }.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                result[key] = json.load(f)
        else:
            result[key] = {}
    return jsonify(result)


@app.route("/api/transfer-recommendation", methods=["POST"])
def transfer_recommendation():
    data = request.get_json(silent=True) or {}
    try:
        predicted = data.get("predicted_24h_demand")
        if predicted is None:
            predicted = load_model_and_predict(data)
        records = ensure_traceability_fields() or load_inventory_state(enrich=True)
        usable = build_usable_stock_analysis(
            data,
            predicted_24h_demand=float(predicted),
            inventory_records=records,
            tasks=list_tasks(limit=500),
        )
        return jsonify(build_transfer_recommendation(
            data,
            predicted_24h_demand=float(predicted),
            usable_analysis=usable,
            inventory_records=records,
            tasks=list_tasks(limit=500),
        ))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/supplier-risk", methods=["POST"])
def supplier_risk():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(build_supplier_risk(
            data,
            shortage_gap=data.get("true_shortage_gap"),
            post_transfer_gap=data.get("post_transfer_gap"),
        ))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/substitute-options", methods=["POST"])
def substitute_options():
    data = request.get_json(silent=True) or {}
    try:
        records = ensure_traceability_fields() or load_inventory_state(enrich=True)
        return jsonify(build_substitute_options(
            data,
            remaining_gap=data.get("remaining_gap", data.get("post_transfer_gap", data.get("true_shortage_gap", 0))),
            inventory_records=records,
            tasks=list_tasks(limit=500),
        ))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/stage3-action-plan", methods=["POST"])
def stage3_action_plan():
    data = request.get_json(silent=True) or {}
    try:
        predicted = data.get("predicted_24h_demand")
        if predicted is None:
            predicted = load_model_and_predict(data)
        records = ensure_traceability_fields() or load_inventory_state(enrich=True)
        return jsonify(build_stage3_action_plan(
            data,
            predicted_24h_demand=float(predicted),
            inventory_records=records,
            tasks=list_tasks(limit=500),
        ))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500




@app.route("/api/stage4-reference-data", methods=["GET"])
def stage4_reference_data():
    """Return Stage 4 cost/ROI assumptions for transparent demo math."""
    return jsonify(load_stage4_assumptions())


@app.route("/api/stage4-roi-analysis", methods=["POST"])
def stage4_roi_analysis():
    data = request.get_json(silent=True) or {}
    try:
        predicted = data.get("predicted_24h_demand")
        if predicted is None:
            predicted = load_model_and_predict(data)
        records = ensure_traceability_fields() or load_inventory_state(enrich=True)
        tasks = list_tasks(limit=500)
        stage3 = build_stage3_action_plan(
            data,
            predicted_24h_demand=float(predicted),
            inventory_records=records,
            tasks=tasks,
        )
        return jsonify(build_stage4_roi_analysis(
            data,
            predicted_24h_demand=float(predicted),
            inventory_records=records,
            tasks=tasks,
            stage3_plan=stage3,
        ))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500




@app.route("/api/stage5-reference-data", methods=["GET"])
def stage5_reference_data():
    """Return Stage 5 command-center playbooks for transparent demo workflow."""
    return jsonify(load_stage5_playbooks())


@app.route("/api/stage5-command-center", methods=["POST"])
def stage5_command_center():
    data = request.get_json(silent=True) or {}
    try:
        predicted = data.get("predicted_24h_demand")
        if predicted is None:
            predicted = load_model_and_predict(data)
        predicted = float(predicted)
        records = ensure_traceability_fields() or load_inventory_state(enrich=True)
        tasks = list_tasks(limit=500)
        stage3 = build_stage3_action_plan(
            data,
            predicted_24h_demand=predicted,
            inventory_records=records,
            tasks=tasks,
        )
        stage4 = build_stage4_roi_analysis(
            data,
            predicted_24h_demand=predicted,
            inventory_records=records,
            tasks=tasks,
            stage3_plan=stage3,
        )
        return jsonify(build_stage5_command_center(
            data,
            predicted_24h_demand=predicted,
            inventory_records=records,
            tasks=tasks,
            stage3_plan=stage3,
            stage4_roi=stage4,
        ))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/stage6-scenarios", methods=["GET"])
def stage6_scenarios():
    """Return Stage 6 what-if scenario playbooks."""
    return jsonify(load_stage6_scenarios())


@app.route("/api/stage6-whatif-simulator", methods=["POST"])
def stage6_whatif_simulator():
    """Run the selected scenario or compare all scenarios.

    This endpoint is deterministic and local. It does not call remote LLMs, does
    not stream, and uses the Stage 2-5 logic as the scenario impact engine.
    """
    data = request.get_json(silent=True) or {}
    try:
        telemetry = data.get("telemetry") if isinstance(data.get("telemetry"), dict) else data
        scenario_id = data.get("scenario_id", telemetry.get("scenario_id", "ED_SURGE_40"))
        custom_modifiers = data.get("custom_modifiers", {})
        records = ensure_traceability_fields() or load_inventory_state(enrich=True)
        tasks = list_tasks(limit=500)
        if data.get("compare_all") or scenario_id == "COMPARE_ALL":
            return jsonify(build_stage6_scenario_benchmark(
                telemetry,
                custom_modifiers=custom_modifiers,
                inventory_records=records,
                tasks=tasks,
            ))
        return jsonify(build_stage6_whatif_simulation(
            telemetry,
            scenario_id=scenario_id,
            custom_modifiers=custom_modifiers,
            inventory_records=records,
            tasks=tasks,
        ))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500



@app.route("/api/usable-stock-analysis", methods=["POST"])
def usable_stock_analysis():
    """Stage 2 endpoint: forecast demand vs true usable stock."""
    data = request.get_json(silent=True) or {}
    try:
        predicted = data.get("predicted_24h_demand")
        if predicted is None:
            predicted = load_model_and_predict(data)
        records = ensure_traceability_fields() or load_inventory_state(enrich=True)
        analysis = build_usable_stock_analysis(
            data,
            predicted_24h_demand=float(predicted),
            inventory_records=records,
            tasks=list_tasks(limit=500),
        )
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/packing-queue", methods=["GET", "POST"])
def packing_queue():
    """Return the top supplies to pack first using the live sidebar scenario.

    GET is kept for simple browser checks. POST is used by the Streamlit dashboard
    so department, sliders, hour/day, and season all affect the Top 5 queue.
    """
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.args.to_dict()

    try:
        limit = int(payload.get("limit", 5))
        max_records = int(payload.get("max_records", 50))
        department = payload.get("department", None)

        path = "database/inventory_state.json"
        if os.path.exists(path):
            inventory_all_records = ensure_traceability_fields()
        else:
            inventory_all_records = inventory().get_json() or []

        inventory_records = inventory_all_records
        if department:
            inventory_records = [rec for rec in inventory_all_records if rec.get("department") == department]

        queue = build_packing_queue(
            inventory_records,
            limit=limit,
            max_records=max_records,
            scenario=payload,
            transfer_inventory_records=inventory_all_records,
        )
        return jsonify({
            "department": department,
            "scenario_used": payload,
            "queue": queue,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/reset", methods=["POST"])
def reset_system():
    if os.path.exists("database/supply_memory_state.json"):
        os.remove("database/supply_memory_state.json")
    if os.path.exists("database/supply_memory_events.jsonl"):
        os.remove("database/supply_memory_events.jsonl")
    return jsonify({"status": "reset"})

@app.route("/api/supply-memory", methods=["GET"])
def supply_memory():
    return jsonify(load_supply_memory())

@app.route("/api/supply-memory-events", methods=["GET"])
def supply_memory_events():
    path = "database/supply_memory_events.jsonl"
    if not os.path.exists(path):
        return jsonify([])
    
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception as e:
                    pass
    
    # Return last 10 events
    return jsonify(events[-10:])

@app.route("/api/predict-supply-demand", methods=["POST"])
def predict_supply_demand():
    telemetry = request.get_json(silent=True) or {}
    try:
        predicted_demand = load_model_and_predict(telemetry)
        records = ensure_traceability_fields() or load_inventory_state(enrich=True)
        usable_analysis = build_usable_stock_analysis(
            telemetry,
            predicted_24h_demand=predicted_demand,
            inventory_records=records,
            tasks=list_tasks(limit=500),
        )
        
        # Load memory state
        memory_before = load_supply_memory()
        
        # Append forecast event
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "forecast",
            "item_name": telemetry.get("item_name", "Unknown"),
            "department": telemetry.get("department", "Unknown"),
            "telemetry_used": telemetry,
            "memory_before": memory_before,
            "forecast_result": {
                "predicted_24h_demand": round(predicted_demand, 2),
                "stock_basis": "usable_stock",
                "usable_stock": usable_analysis.get("usable_stock"),
                "true_shortage_gap": usable_analysis.get("true_shortage_gap"),
            }
        }
        append_memory_event(event)
        
        return jsonify({
            "predicted_24h_demand": round(predicted_demand, 2),
            "usable_stock_analysis": usable_analysis,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/shortage-risk", methods=["POST"])
def shortage_risk():
    data = request.get_json(silent=True) or {}
    predicted_demand = data.get("predicted_24h_demand", 0.0)
    current_stock = data.get("current_stock", 0)

    if data.get("use_usable_stock") or data.get("stock_basis") == "usable_stock":
        analysis = build_usable_stock_analysis(
            data,
            predicted_24h_demand=float(predicted_demand),
            inventory_records=ensure_traceability_fields() or load_inventory_state(enrich=True),
            tasks=list_tasks(limit=500),
        )
        result = analysis.get("shortage_risk_using_usable_stock", {})
        result["usable_stock_analysis"] = analysis
        return jsonify(result)
    
    result = calculate_shortage_risk(predicted_demand, current_stock)
    return jsonify(result)

@app.route("/api/packing-priority", methods=["POST"])
def packing_priority():
    data = request.get_json(silent=True) or {}
    item_name = data.get("item_name", "")
    department = data.get("department", "")
    shortage_result = data.get("shortage_result", {})
    telemetry = data.get("telemetry", {})
    
    result = calculate_priority_and_pack_quantity(
        item_name,
        department,
        shortage_result,
        telemetry
    )
    return jsonify(result)

@app.route("/api/update-supply-memory", methods=["POST"])
def update_supply_memory():
    data = request.get_json(silent=True) or {}
    item_name = data.get("item_name", "Unknown")
    department = data.get("department", "Unknown")
    predicted_demand = float(data.get("predicted_demand", 0.0))
    actual_usage = float(data.get("actual_usage", 0.0))
    
    new_memory = update_supply_memory_after_actual(
        predicted_demand,
        actual_usage,
        item_name,
        department
    )
    return jsonify(new_memory)

@app.route("/api/run-medpack-committee-fast", methods=["POST"])
def run_medpack_committee_fast():
    """Guaranteed-fast local committee path for the Streamlit demo button.

    This route deliberately avoids remote LLM calls, streaming, model loading, and
    memory writes. It is the safe path used by the main button when the old
    committee route is suspected of hanging.
    """
    telemetry = request.get_json(silent=True) or {}
    telemetry["agent_mode"] = "local"
    telemetry["force_local_committee"] = True
    return jsonify(build_fast_committee_payload(telemetry))


@app.route("/api/foundry-explanation", methods=["POST"])
def foundry_explanation():
    """Optional post-hoc explanation from Microsoft Foundry.

    Accepts the locked deterministic result fields and asks the Foundry agent
    to explain them in plain language. Foundry is strictly read-only: it cannot
    change predicted demand, usable stock, risk level, shortage gap, packing
    priority, recommended action, or the human-approval requirement.

    Returns a safe JSON payload regardless of Foundry availability, so that
    a Foundry failure can never break the deterministic dashboard result.
    """
    from backend.foundry_client import invoke_foundry_agent, is_foundry_configured

    if not is_foundry_configured():
        return jsonify({
            "available": False,
            "source": "Microsoft Foundry",
            "explanation": "",
            "tokens_used": 0,
            "model": "azure-foundry-agent",
            "reason": "Foundry is disabled or not configured on this server."
        })

    data = request.get_json(silent=True) or {}
    telemetry         = data.get("telemetry", {})
    prediction        = data.get("prediction", {})
    shortage_risk     = data.get("shortage_risk", {})
    packing_priority  = data.get("packing_priority", {})
    memory_state      = data.get("memory_state", {})

    try:
        foundry_result = invoke_foundry_agent(
            telemetry=telemetry,
            prediction_result=prediction,
            shortage_result=shortage_risk,
            priority_result=packing_priority,
            memory_state=memory_state,
        )
        return jsonify({
            "available": True,
            "source": "Microsoft Foundry",
            "explanation": foundry_result.get("text", ""),
            "tokens_used": foundry_result.get("tokens_used", 0),
            "model": foundry_result.get("model", "azure-foundry-agent"),
        })
    except Exception as e:
        import traceback
        print(f"FOUNDRY ERROR: {e}", flush=True)
        traceback.print_exc()
        # Never expose credentials, endpoint URLs, Azure tenant IDs, or stack traces
        # to the frontend. Return a safe, generic failure message only.
        return jsonify({
            "available": False,
            "source": "Microsoft Foundry",
            "explanation": "",
            "tokens_used": 0,
            "model": "azure-foundry-agent",
            "reason": "Foundry agent call failed. Deterministic result is unaffected."
        })
@app.route("/api/run-medpack-committee", methods=["POST"])
def run_medpack_committee():
    telemetry = request.get_json(silent=True) or {}

    # Freeze Fix v3: by default, the committee endpoint itself delegates to the
    # guaranteed fast local path. Set MEDPACK_ALLOW_FULL_COMMITTEE_ROUTE=true
    # only if you intentionally want to test the heavier legacy committee route.
    if not _truthy(os.environ.get("MEDPACK_ALLOW_FULL_COMMITTEE_ROUTE", "false")):
        telemetry["agent_mode"] = "local"
        telemetry["force_local_committee"] = True
        return jsonify(build_fast_committee_payload(telemetry))
    
    # Run pipeline steps. Stage 2 uses usable stock, not just total stock.
    predicted_demand = load_model_and_predict(telemetry)
    inventory_records = ensure_traceability_fields() or load_inventory_state(enrich=True)
    usable_analysis = build_usable_stock_analysis(
        telemetry,
        predicted_24h_demand=predicted_demand,
        inventory_records=inventory_records,
        tasks=list_tasks(limit=500),
    )
    shortage_result = usable_analysis.get("shortage_risk_using_usable_stock", {})
    
    priority_result = calculate_priority_and_pack_quantity(
        telemetry.get("item_name", ""),
        telemetry.get("department", ""),
        shortage_result,
        telemetry
    )
    
    memory_state = load_supply_memory()
    
    # Run ADK agent committee
    agent_mode = _resolve_committee_agent_mode(telemetry)

    committee_result = run_committee(
        telemetry,
        {"predicted_24h_demand": predicted_demand},
        shortage_result,
        priority_result,
        memory_state,
        agent_mode=agent_mode
    )
    
    # Append forecast event
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "forecast",
        "item_name": telemetry.get("item_name", "Unknown"),
        "department": telemetry.get("department", "Unknown"),
        "telemetry_used": telemetry,
        "memory_before": memory_state,
        "forecast_result": {
            "predicted_24h_demand": round(predicted_demand, 2),
            "stock_basis": "usable_stock",
            "usable_stock": usable_analysis.get("usable_stock"),
            "true_shortage_gap": usable_analysis.get("true_shortage_gap"),
        }
    }
    append_memory_event(event)

    stage3_plan = build_stage3_action_plan(
        telemetry,
        predicted_24h_demand=float(predicted_demand),
        inventory_records=inventory_records,
        tasks=list_tasks(limit=500),
    )
    stage4_roi = build_stage4_roi_analysis(
        telemetry,
        predicted_24h_demand=float(predicted_demand),
        inventory_records=inventory_records,
        tasks=list_tasks(limit=500),
        stage3_plan=stage3_plan,
    )
    
    return jsonify({
        "telemetry": telemetry,
        "prediction": {"predicted_24h_demand": round(predicted_demand, 2)},
        "shortage_risk": shortage_result,
        "packing_priority": priority_result,
        "usable_stock_analysis": usable_analysis,
        "stage3_action_plan": stage3_plan,
        "stage4_roi_analysis": stage4_roi,
        "memory_state": memory_state,
        "committee": committee_result
    })

@app.route("/api/run-medpack-committee-stream", methods=["POST"])
def run_medpack_committee_stream():
    telemetry = request.get_json(silent=True) or {}

    if not _truthy(os.environ.get("MEDPACK_ALLOW_COMMITTEE_STREAM", "false")):
        payload = build_fast_committee_payload({**telemetry, "agent_mode": "local", "force_local_committee": True})
        def fast_generate():
            yield json.dumps({"event": "complete", "data": payload.get("committee", {})}) + "\n"
            yield json.dumps({"event": "final_payload", **payload}) + "\n"
        return Response(fast_generate(), mimetype="application/x-ndjson")
    
    def generate():
        try:
            predicted_demand = load_model_and_predict(telemetry)
            inventory_records = ensure_traceability_fields() or load_inventory_state(enrich=True)
            usable_analysis = build_usable_stock_analysis(
                telemetry,
                predicted_24h_demand=predicted_demand,
                inventory_records=inventory_records,
                tasks=list_tasks(limit=500),
            )
            shortage_result = usable_analysis.get("shortage_risk_using_usable_stock", {})
            priority_result = calculate_priority_and_pack_quantity(
                telemetry.get("item_name", ""), telemetry.get("department", ""), shortage_result, telemetry
            )
            memory_state = load_supply_memory()
            
            yield json.dumps({"event": "init_done", "data": "Initialization complete"}) + "\n"
            
            agent_mode = _resolve_committee_agent_mode(telemetry)
            
            final_data = None
            for chunk in run_committee_stream(telemetry, {"predicted_24h_demand": predicted_demand}, shortage_result, priority_result, memory_state, agent_mode=agent_mode):
                yield json.dumps(chunk) + "\n"
                if chunk.get("event") == "complete":
                    final_data = chunk.get("data")
                    
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "forecast",
                "item_name": telemetry.get("item_name", "Unknown"),
                "department": telemetry.get("department", "Unknown"),
                "telemetry_used": telemetry,
                "memory_before": memory_state,
                "forecast_result": {
                    "predicted_24h_demand": round(predicted_demand, 2),
                    "stock_basis": "usable_stock",
                    "usable_stock": usable_analysis.get("usable_stock"),
                    "true_shortage_gap": usable_analysis.get("true_shortage_gap"),
                }
            }
            append_memory_event(event)
            
            if final_data:
                yield json.dumps({
                    "event": "final_payload",
                    "telemetry": telemetry,
                    "prediction": {"predicted_24h_demand": round(predicted_demand, 2)},
                    "shortage_risk": shortage_result,
                    "packing_priority": priority_result,
                    "usable_stock_analysis": usable_analysis,
                    "memory_state": memory_state,
                    "committee": final_data
                }) + "\n"
                
        except Exception as e:
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"
            
    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/api/rag", methods=["POST"])
def rag_query():
    """Query the RAG knowledge base.

    Accepts JSON body with ``query`` (required) and optional ``top_k``
    (default 3).  Returns the most relevant policy/SOP document chunks.
    """
    from backend.rag_manager import rag_manager as _rag
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400
    top_k = int(data.get("top_k", 3))
    results = _rag.retrieve(query, top_k=top_k)
    return jsonify({"query": query, "results": results})


@app.route("/api/rag/stats", methods=["GET"])
def rag_stats():
    """Return RAG index statistics (backend type, chunk count, sources)."""
    from backend.rag_manager import rag_manager as _rag
    return jsonify(_rag.get_stats())


def pre_warm_ml_model():
    """Pre-warm the ML model so the first user request is served instantly.
    Without this, joblib + XGBoost cold-load on the first /api/predict call
    can take 30+ seconds, exceeding the dashboard's request timeout.
    """
    global _ML_READY
    try:
        print("[startup] ML prewarm started", flush=True)
        import time
        t0 = time.time()
        from backend.model import load_model_and_predict
        _dummy = {
            "department": "Emergency Department",
            "item_name": "IV Start Kit",
            "item_category": "IV Supplies",
            "season": "Winter",
            "current_stock": 10,
            "patient_volume": 20,
            "acuity_level": 2,
            "procedure_count": 5,
            "recent_usage_rate": 5,
            "supplier_delay_days": 2,
            "day_of_week": 1,
            "hour": 8,
            "reorder_point": 25,
            "supplier_reliability_score": 0.9,
            "clinical_criticality": 3,
            "pack_time_minutes": 4,
        }
        load_model_and_predict(_dummy)
        
        t1 = time.time()
        print(f"[startup] ML prewarm completed in {t1 - t0:.2f} sec", flush=True)
        
        _ML_READY = True
    except Exception as _e:
        print(f"[startup] ML model pre-warm skipped: {_e}", flush=True)

# Run pre-warming unconditionally during app import/startup (e.g., Gunicorn workers)
pre_warm_ml_model()

if __name__ == "__main__":
    print(f"[startup] Flask binding port {PORT}", flush=True)
    print("[startup] backend ready", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=os.environ.get("FLASK_DEBUG", "0") == "1", threaded=True)