import os
from unittest import mock
import pytest

from backend.foundry_client import invoke_foundry_agent, is_foundry_configured

@pytest.fixture
def mock_env():
    with mock.patch.dict(os.environ, {
        "USE_FOUNDRY_AGENT": "true",
        "FOUNDRY_PROJECT_ENDPOINT": "https://test.services.ai.azure.com/api/projects/test",
        "FOUNDRY_AGENT_NAME": "test-agent"
    }):
        yield

def test_is_foundry_configured(mock_env):
    assert is_foundry_configured() is True
    
    with mock.patch.dict(os.environ, {"USE_FOUNDRY_AGENT": "false"}):
        assert is_foundry_configured() is False

@mock.patch("backend.foundry_client.AZURE_SDK_AVAILABLE", True)
@mock.patch("backend.foundry_client.AIProjectClient")
@mock.patch("backend.foundry_client.DefaultAzureCredential")
def test_invoke_foundry_agent(mock_credential, mock_ai_client, mock_env):
    # Setup mock
    mock_client_instance = mock.MagicMock()
    mock_ai_client.return_value = mock_client_instance
    
    # Mock context manager
    mock_client_instance.__enter__.return_value = mock_client_instance
    
    # Mock get_openai_client
    mock_openai_client = mock.MagicMock()
    mock_client_instance.get_openai_client.return_value = mock_openai_client
    
    # Mock responses.create
    mock_response = mock.MagicMock()
    mock_response.output_text = "Mocked Agent Recommendation"
    mock_response.usage.total_tokens = 42
    
    mock_openai_client.responses.create.return_value = mock_response
    
    # Invoke
    telemetry = {"item_name": "Test Item"}
    prediction_result = {"predicted_24h_demand": 10.0}
    shortage_result = {"shortage_gap": 5.0}
    priority_result = {"priority_score": 90.0}
    memory_state = {}
    
    result = invoke_foundry_agent(
        telemetry, prediction_result, shortage_result, priority_result, memory_state, agent_name="test-agent"
    )
    
    # Verify
    assert result["text"] == "Mocked Agent Recommendation"
    assert result["tokens_used"] == 42
    assert result["model"] == "azure-foundry-agent"
    
    # Verify correct SDK initialization parameters
    mock_ai_client.assert_called_once()
    kwargs = mock_ai_client.call_args.kwargs
    assert kwargs.get("endpoint") == "https://test.services.ai.azure.com/api/projects/test"
    assert "credential" in kwargs
    assert kwargs.get("allow_preview") is True
    
    # Verify prompt text includes authoritative snapshot
    mock_openai_client.responses.create.assert_called_once()
    create_kwargs = mock_openai_client.responses.create.call_args.kwargs
    prompt_text = create_kwargs.get("input", "")
    
    assert "AUTHORITATIVE and IMMUTABLE facts" in prompt_text
    assert "10.0" in prompt_text  # prediction_result["predicted_24h_demand"]
    assert "5.0" in prompt_text   # shortage_result["shortage_gap"]
    assert "90.0" in prompt_text  # priority_result["priority_score"] (not explicitly snapshot but in payload)
