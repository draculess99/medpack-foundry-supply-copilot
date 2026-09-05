import os

from backend.agents.adk_agents import run_committee


def _sample_inputs():
    telemetry = {
        "item_name": "Oxygen Mask",
        "department": "Emergency Department",
        "current_stock": 5,
        "patient_volume": 30,
        "acuity_level": 3.0,
        "pack_time_minutes": 4.0,
        "clinical_criticality": 4,
    }
    prediction = {"predicted_24h_demand": 42.0}
    shortage = {"shortage_gap": 37.0, "coverage_ratio": 0.12, "risk_level": "Critical"}
    priority = {
        "priority_score": 96.0,
        "recommended_pack_quantity": 40,
        "recommended_action": "Pack immediately and escalate",
        "escalation_required": True,
    }
    memory = {"trend_direction": "increasing", "rolling_usage_avg": 30.0}
    return telemetry, prediction, shortage, priority, memory


def test_local_mode_is_zero_token(monkeypatch):
    monkeypatch.setenv("USE_LLM_AGENTS", "false")
    monkeypatch.setenv("USE_FOUNDRY_AGENT", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = run_committee(*_sample_inputs(), agent_mode="local")
    assert result["actual_agent_mode"] == "local"
    assert result["tokens_used"] == 0
    assert result["fallback_mode"] is True


def test_remote_mode_without_env_falls_back_zero_token(monkeypatch):
    monkeypatch.setenv("USE_LLM_AGENTS", "false")
    monkeypatch.setenv("USE_FOUNDRY_AGENT", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = run_committee(*_sample_inputs(), agent_mode="remote")
    assert result["requested_agent_mode"] == "remote"
    assert result["actual_agent_mode"] == "local"
    assert result["tokens_used"] == 0
    assert "USE_LLM_AGENTS" in result["mode_note"]
