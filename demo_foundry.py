import os
import json
import logging
from dotenv import load_dotenv

# Load root .env before checking configuration
load_dotenv()

from backend.foundry_client import is_foundry_configured, invoke_foundry_agent
from backend.agents.adk_agents import run_committee

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_synthetic_demo():
    print("====================================================")
    print("   MedPack AI Microsoft Foundry Synthetic Demo      ")
    print("====================================================")
    
    # 1. Check if Foundry is configured
    foundry_enabled = is_foundry_configured()
    agent_name = os.environ.get("FOUNDRY_AGENT_NAME", "medpack-supply-copilot").strip()
    
    print(f"[*] Foundry enabled: {foundry_enabled}")
    if foundry_enabled:
        print(f"[*] Configured agent name: {agent_name}")
    else:
        print("[*] Configured agent name: N/A (Local Fallback)")
    
    # 2. Define the synthetic scenario
    print("\n[Scenario]")
    print("- Item: IV-Start Kits")
    print("- Department: Emergency")
    print("- Predicted 24h Demand: 80 kits")
    print("- Current Usable Stock: 45 kits")
    print("- Incoming Transfer: 25 kits (in 12 hours)")
    
    telemetry = {
        "item_name": "IV-Start Kits",
        "department": "Emergency",
        "patient_volume": 120,
        "acuity_level": "High",
        "clinical_criticality": 4, # High impact
        "pack_time_minutes": 5.0,
        "selected_provider": "Microsoft Foundry" if foundry_enabled else "Local Engine",
        "agent_mode": "remote" if foundry_enabled else "local"
    }
    
    prediction_result = {
        "predicted_24h_demand": 80.0
    }
    
    shortage_result = {
        "usable_stock": 45,
        "total_stock": 50,
        "unsafe_stock": 5,
        "active_task_reserved_stock": 0,
        "transfer_candidate_stock": 25,
        "coverage_ratio": 70 / 80.0,
        "true_shortage_gap": 10.0,
        "shortage_gap": 10.0,
        "risk_level": "High",
    }
    
    priority_result = {
        "priority_score": 95.0,
        "recommended_pack_quantity": 45,
        "recommended_action": "Expedite Order Recommended. Human approval required."
    }
    
    memory_state = {
        "trend_direction": "increasing",
        "rolling_usage_avg": 75.0
    }
    
    print("\n[Expected Result]")
    print("- High shortage risk")
    print("- 10-kit gap")
    print("- Expedited order recommendation")
    print("- Explicit human approval required")
    
    if foundry_enabled:
        print("\n[*] Invoking Microsoft AI Foundry Agent directly...")
        try:
            result = invoke_foundry_agent(
                telemetry=telemetry,
                prediction_result=prediction_result,
                shortage_result=shortage_result,
                priority_result=priority_result,
                memory_state=memory_state,
                agent_name=agent_name
            )
            
            print("\n====================================================")
            print("                 FOUNDRY AGENT SUMMARY              ")
            print("====================================================")
            print(f"Model: {result.get('model', 'Unknown')}")
            print(f"Tokens Used: {result.get('tokens_used', 0)}")
            print("\n[Recommendation]")
            print(result.get("text", ""))
            print("====================================================\n")
            return
            
        except Exception as e:
            logging.error(f"Foundry invocation failed: {e}")
            print("\n[*] Falling back to MedPack AI Local Committee...")
    
    # Fallback / Local mode
    print("\n[*] Invoking MedPack AI Local Committee...")
    telemetry["selected_provider"] = "Local Engine"
    telemetry["agent_mode"] = "local"
    
    try:
        result = run_committee(
            telemetry=telemetry,
            prediction_result=prediction_result,
            shortage_result=shortage_result,
            priority_result=priority_result,
            memory_state=memory_state,
            agent_mode="local"
        )
        
        print("\n====================================================")
        print("                 COMMITTEE SUMMARY                  ")
        print("====================================================")
        print(f"Mode Used: {result.get('mode_note', 'Unknown')}")
        print("\n[Final Recommendation Agent]")
        print(result.get("final_recommendation_agent", ""))
        print("\n[Committee Summarizer]")
        print(result.get("committee_summary", ""))
        print("====================================================\n")
        
    except Exception as e:
        logging.error(f"Demo failed: {e}")

if __name__ == "__main__":
    run_synthetic_demo()
