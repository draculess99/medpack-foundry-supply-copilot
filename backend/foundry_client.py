import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Safely load the root .env file for direct script execution
load_dotenv()

try:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False


def is_foundry_configured() -> bool:
    """Check if Foundry is enabled and configured."""
    enabled = str(os.environ.get("USE_FOUNDRY_AGENT", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    return enabled and bool(endpoint)


def invoke_foundry_agent(
    telemetry: Dict[str, Any],
    prediction_result: Dict[str, Any],
    shortage_result: Dict[str, Any],
    priority_result: Dict[str, Any],
    memory_state: Dict[str, Any],
    agent_name: str = None
) -> Dict[str, Any]:
    """
    Invokes the deployed Microsoft Foundry agent.
    Returns a summarized dictionary of the agent's output.
    """
    if not AZURE_SDK_AVAILABLE:
        raise RuntimeError("azure-ai-projects>=2.0.0 and azure-identity are not installed. Cannot invoke Foundry Agent.")

    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError("FOUNDRY_PROJECT_ENDPOINT is missing.")

    if not agent_name:
        agent_name = os.environ.get("FOUNDRY_AGENT_NAME", "medpack-supply-copilot").strip()

    # Base context to send to the agent
    base_data = {
        "telemetry": telemetry,
        "prediction": prediction_result,
        "shortage_risk": shortage_result,
        "packing_priority": priority_result,
        "memory_state": memory_state,
    }
    
    # Compute the authoritative decision snapshot with existing MedPack logic
    usable_stock = shortage_result.get("usable_stock", 0)
    transfer_stock = shortage_result.get("transfer_candidate_stock", 0)
    authoritative_snapshot = {
        "risk_level": shortage_result.get("risk_level", "Unknown"),
        "next_24_hour_demand": prediction_result.get("predicted_24h_demand", 0.0),
        "usable_stock_now": usable_stock,
        "confirmed_transfer": transfer_stock,
        "expected_available_after_transfer": usable_stock + transfer_stock,
        "shortage_gap": shortage_result.get("shortage_gap", 0.0),
        "recommended_action": priority_result.get("recommended_action", "Unknown"),
        "human_approval_required": True
    }
    
    # Initialize the project client using the Azure AI Projects SDK endpoint pattern
    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )
    
    with project_client:
        openai_client = project_client.get_openai_client(agent_name=agent_name)
        
        prompt_text = (
            "You are the MedPack AI Committee Summarizer. "
            "The following MedPack decision snapshot contains AUTHORITATIVE and IMMUTABLE facts.\n"
            f"SNAPSHOT:\n{json.dumps(authoritative_snapshot, indent=2)}\n\n"
            "You MUST explain these provided facts. DO NOT recalculate, relabel, or override the risk level, gap, or recommended action.\n"
            "Your job is limited to providing a clear operational explanation, stating assumptions, issuing missing-data warnings, and reminding about human-approval.\n\n"
            f"Full Context:\n{json.dumps(base_data, indent=2)}"
        )
        
        try:
            response = openai_client.responses.create(
                input=prompt_text
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI responses call failed: {e}")
            
        reply_text = response.output_text
        
        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = getattr(response.usage, "total_tokens", 0)
        
        return {
            "text": reply_text,
            "tokens_used": tokens_used,
            "model": "azure-foundry-agent"
        }
