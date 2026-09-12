"""
test_foundry_endpoint.py
------------------------
Focused tests for the /api/foundry-explanation backend endpoint.

These tests verify:
1. Successful explanation response when Foundry is configured and available.
2. Safe "available: false" response when Foundry is disabled/unconfigured.
3. Safe "available: false" response when invoke_foundry_agent raises an exception.
4. Deterministic fields passed in are reflected in the invoke call; the endpoint
   never mutates them.

All Foundry SDK calls are mocked so the test suite runs without Azure credentials
and without network access.
"""
import json
from unittest import mock

import pytest

from backend.server import app


SAMPLE_PAYLOAD = {
    "telemetry": {"item_name": "IV-Start Kit", "department": "ICU"},
    "prediction": {"predicted_24h_demand": 18.0},
    "shortage_risk": {
        "risk_level": "High",
        "shortage_gap": 4.5,
        "usable_stock": 13,
        "transfer_candidate_stock": 2,
        "recommended_action": "Expedite order",
    },
    "packing_priority": {
        "priority_score": 87.0,
        "recommended_pack_quantity": 20,
        "recommended_action": "Pack immediately",
        "escalation_required": True,
        "reasoning": "Critical demand gap detected.",
    },
    "memory_state": {"rolling_avg_demand": 15.2},
}


# ---------------------------------------------------------------------------
# Test 1: Successful Foundry explanation response
# ---------------------------------------------------------------------------
@mock.patch("backend.foundry_client.is_foundry_configured", return_value=True)
@mock.patch(
    "backend.foundry_client.invoke_foundry_agent",
    return_value={
        "text": "The ICU faces a High-risk shortage of IV-Start Kits...",
        "tokens_used": 312,
        "model": "azure-foundry-agent",
    },
)
def test_foundry_endpoint_success(mock_invoke, mock_configured):
    client = app.test_client()
    response = client.post(
        "/api/foundry-explanation",
        json=SAMPLE_PAYLOAD,
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    assert data["available"] is True
    assert data["source"] == "Microsoft Foundry"
    assert "IV-Start Kit" in data["explanation"] or len(data["explanation"]) > 0
    assert data["tokens_used"] == 312
    assert data["model"] == "azure-foundry-agent"

    # Verify invoke was called with the correct deterministic fields
    call_kwargs = mock_invoke.call_args.kwargs
    assert call_kwargs["telemetry"] == SAMPLE_PAYLOAD["telemetry"]
    assert call_kwargs["prediction_result"] == SAMPLE_PAYLOAD["prediction"]
    assert call_kwargs["shortage_result"] == SAMPLE_PAYLOAD["shortage_risk"]
    assert call_kwargs["priority_result"] == SAMPLE_PAYLOAD["packing_priority"]
    assert call_kwargs["memory_state"] == SAMPLE_PAYLOAD["memory_state"]


# ---------------------------------------------------------------------------
# Test 2: Foundry disabled / not configured → safe "available: false" response
# ---------------------------------------------------------------------------
@mock.patch("backend.foundry_client.is_foundry_configured", return_value=False)
def test_foundry_endpoint_disabled(mock_configured):
    client = app.test_client()
    response = client.post(
        "/api/foundry-explanation",
        json=SAMPLE_PAYLOAD,
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    assert data["available"] is False
    assert data["source"] == "Microsoft Foundry"
    assert data["explanation"] == ""
    assert data["tokens_used"] == 0
    assert "reason" in data
    # Reason must be a short, non-secret message
    assert len(data["reason"]) < 200
    # Must NOT contain secrets, URLs, or tenant IDs
    reason_lower = data["reason"].lower()
    assert "azure.com" not in reason_lower
    assert "tenant" not in reason_lower
    assert "key" not in reason_lower


# ---------------------------------------------------------------------------
# Test 3: invoke_foundry_agent raises → safe failure payload, no exception leak
# ---------------------------------------------------------------------------
@mock.patch("backend.foundry_client.is_foundry_configured", return_value=True)
@mock.patch(
    "backend.foundry_client.invoke_foundry_agent",
    side_effect=RuntimeError("FOUNDRY_PROJECT_ENDPOINT is missing."),
)
def test_foundry_endpoint_exception_returns_safe_payload(mock_invoke, mock_configured):
    client = app.test_client()
    response = client.post(
        "/api/foundry-explanation",
        json=SAMPLE_PAYLOAD,
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    assert data["available"] is False
    assert data["source"] == "Microsoft Foundry"
    assert data["explanation"] == ""
    assert data["tokens_used"] == 0
    # The raw exception message must NOT appear in the response
    assert "FOUNDRY_PROJECT_ENDPOINT" not in json.dumps(data)
    assert "RuntimeError" not in json.dumps(data)
    assert "reason" in data


# ---------------------------------------------------------------------------
# Test 4: Deterministic fields are passed through unchanged to invoke
# ---------------------------------------------------------------------------
@mock.patch("backend.foundry_client.is_foundry_configured", return_value=True)
@mock.patch(
    "backend.foundry_client.invoke_foundry_agent",
    return_value={"text": "Explanation text", "tokens_used": 50, "model": "azure-foundry-agent"},
)
def test_foundry_endpoint_deterministic_fields_unchanged(mock_invoke, mock_configured):
    """Verify that the endpoint passes deterministic fields to Foundry unchanged.

    The endpoint must never mutate risk_level, shortage_gap, predicted_24h_demand,
    priority_score, recommended_action, or any other deterministic field.
    """
    client = app.test_client()
    response = client.post(
        "/api/foundry-explanation",
        json=SAMPLE_PAYLOAD,
        content_type="application/json",
    )
    assert response.status_code == 200

    # The call must have received the exact same objects
    call_kwargs = mock_invoke.call_args.kwargs

    # Risk level unchanged
    assert call_kwargs["shortage_result"]["risk_level"] == "High"
    # Shortage gap unchanged
    assert call_kwargs["shortage_result"]["shortage_gap"] == 4.5
    # Predicted demand unchanged
    assert call_kwargs["prediction_result"]["predicted_24h_demand"] == 18.0
    # Priority score unchanged
    assert call_kwargs["priority_result"]["priority_score"] == 87.0
    # Recommended action unchanged
    assert call_kwargs["priority_result"]["recommended_action"] == "Pack immediately"
    # human_approval_required is enforced inside foundry_client, not stripped here
    # (the endpoint is a thin relay; it never removes or adds deterministic fields)