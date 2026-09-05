# Microsoft AI Foundry Integration Guide

This document provides a concise overview of the MedPack AI integration with Microsoft AI Foundry for hackathon judges and evaluators.

## 1. Agent Instructions (`medpack-supply-copilot`)

The deployed Azure agent (`medpack-supply-copilot`) is instructed to act as a **Human-Governed Supply Risk Copilot**.

**Core System Prompt / Instructions:**
> "You are the MedPack AI Supply Risk Copilot. You receive operational context including ML demand predictions, usable stock analysis, and hospital telemetry. Your job is to concisely summarize the supply chain risk, propose mitigation steps (like expedited ordering or inter-department transfers), and explicitly cite any clinical safety boundaries provided in the context. You must never diagnose patients or prescribe medication. You must explicitly state that all recommendations require human approval."

*(Note: We use the `gpt-5.4-nano` model for this agent to maintain high processing speed and strict cost control.)*

## 2. Required Environment Variables

To activate the Foundry integration, the following environment variables must be present in your `.env` file or environment:

```env
USE_FOUNDRY_AGENT=true
FOUNDRY_PROJECT_ENDPOINT=https://your-ai-services-account.services.ai.azure.com/api/projects/your-project-name
FOUNDRY_AGENT_NAME=medpack-supply-copilot
```

**Authentication Note:** 
The application relies on the official `azure-ai-projects>=2.0.0` and `azure-identity` Python SDKs. It uses `DefaultAzureCredential` to authenticate. Do **not** inject raw API keys; you must be logged into Azure CLI, VS Code, or have a managed identity/service principal configured locally.

## 3. Call Flow

1. **Trigger**: A request is made to `/api/run-medpack-committee` with `agent_mode=remote` and `selected_provider="Microsoft Foundry"`.
2. **Local Pre-processing**: MedPack calculates the demand forecast (via XGBoost), usable stock (ignoring expired stock), and the true shortage gap.
3. **Foundry Invocation**: `backend/foundry_client.py` uses `AIProjectClient` to:
   - Connect to the Foundry project via the configured endpoint.
   - Locate the configured agent (e.g. `medpack-supply-copilot`).
   - Create a Thread and send a Message containing the synthesized local context.
   - Create a Run and poll for completion.
4. **Response**: The agent's concise summary is returned to the frontend.
5. **Human Approval**: The Streamlit dashboard displays the recommendation, explicitly blocking automated execution until a human administrator clicks "Approve".

## 4. How a Judge Can Reproduce the Demo

We have built a safe, offline-capable synthetic demo that strictly enforces the required scenario:
- **Scenario**: 80 IV-start kits required in 24 hours, 45 usable now, transfer of 25 in 12 hours.
- **Expected Outcome**: High shortage risk, 10-kit gap, expedited order recommendation, explicit human approval required.

**To run the demo:**

1. **Ensure Dependencies are Installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the Scenario (Local/Safe Mode):**
   ```bash
   python demo_foundry.py
   ```
   *(This will run locally by default, demonstrating the deterministic core logic without requiring any Azure credentials.)*

## 5. Manual Smoke Test (Azure Required)

**Important**: This manual smoke test must **never** run automatically in CI/CD tests. It requires an active Azure sign-in session and will contact the Microsoft AI Foundry backend.

1. Ensure `USE_FOUNDRY_AGENT=true`, `FOUNDRY_PROJECT_ENDPOINT`, and `FOUNDRY_AGENT_NAME` are set in `.env`.
2. Ensure you are logged into Azure via `az login` with access to the Foundry project.
3. Run the synthetic demo with Foundry enabled:
   ```bash
   python demo_foundry.py
   ```
   The script will detect the Foundry configuration, connect to your endpoint, and output the remote agent's response instead of the local engine.
