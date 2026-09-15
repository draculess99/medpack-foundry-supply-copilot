import os
import requests
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import uuid
import time
from pathlib import Path


def _load_dotenv_for_streamlit_frontend():
    """Load .env when the dashboard is started directly with Streamlit.

    The Groq rewrite runs in the Streamlit process, not the Flask backend.
    If the app is launched with `streamlit run frontend/dashboard.py`, the
    backend/app.py never injects GROQ_API_KEY into this process. Loading .env
    here makes Groq mode and the token meter behave consistently.
    """
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass


_load_dotenv_for_streamlit_frontend()

# Token meter state. These are owned by the Streamlit frontend because the
# freeze-safe Groq rewrite happens in the frontend process.
if "current_tokens" not in st.session_state:
    st.session_state["current_tokens"] = 0
if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = 0
if "last_llm_usage" not in st.session_state:
    st.session_state["last_llm_usage"] = {}
if "counted_llm_call_ids" not in st.session_state:
    st.session_state["counted_llm_call_ids"] = set()

if "medpack_deterministic_result" not in st.session_state:
    st.session_state["medpack_deterministic_result"] = None
if "foundry_result" not in st.session_state:
    st.session_state["foundry_result"] = None


# Configure page
st.set_page_config(
    page_title="MedPack AI / MedAIM Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ports & Configs
MEDPACK_API_BASE_URL = os.environ.get("MEDPACK_LOCAL_API_BASE_URL") or os.environ.get("MEDPACK_API_BASE_URL", "http://127.0.0.1:5001")

APP_ROOT = Path(__file__).resolve().parents[1]
LLM_USAGE_STATE_PATH = APP_ROOT / "database" / "llm_usage_state.json"


def _load_persistent_llm_usage():
    """Load the last recorded LLM token usage from disk.

    Streamlit redraws the sidebar before the button callback finishes.  Keeping
    the latest token usage in a tiny JSON file makes the meter survive reruns,
    fallback parsing, and browser refreshes.
    """
    try:
        if not LLM_USAGE_STATE_PATH.exists():
            return {}
        data = json.loads(LLM_USAGE_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _persist_llm_usage(usage_record):
    """Persist the latest token usage so the sidebar cannot lose it."""
    try:
        LLM_USAGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        safe_record = dict(usage_record or {})
        # Keep this file metadata-only. Never store API keys or full prompts.
        for key in ("prompt_text", "completion_text", "api_key"):
            safe_record.pop(key, None)
        LLM_USAGE_STATE_PATH.write_text(json.dumps(safe_record, indent=2), encoding="utf-8")
    except Exception:
        pass


def _sync_llm_usage_from_disk_if_needed():
    """Recover token state from disk if Streamlit session state is empty."""
    if int(st.session_state.get("current_tokens", 0) or 0) > 0:
        return
    usage = _load_persistent_llm_usage()
    tokens = int(usage.get("total_tokens", usage.get("tokens_used", 0)) or 0)
    if tokens > 0:
        st.session_state["current_tokens"] = tokens
        st.session_state["last_llm_usage"] = usage
        if int(st.session_state.get("total_tokens", 0) or 0) <= 0:
            st.session_state["total_tokens"] = tokens


_sync_llm_usage_from_disk_if_needed()


# Apply modern premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography Override */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Sleek Backgrounds and Cards */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 60%);
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(128, 128, 128, 0.25) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Premium Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
        border-color: rgba(255, 255, 255, 0.4);
        color: white !important;
    }
    
    /* Beautiful Metric Values */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #60a5fa 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Soften Metric Labels */
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Modern Headers */
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Add subtle glassmorphism to info/success/warning boxes */
    div[data-testid="stAlert"] {
        background-color: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
    
    /* Clean up the dataframe borders */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# App Titles
st.title("\U0001F3E5 MedPack AI: Hospital Supply Shortage & Packing Priority System")
st.subheader("Forecasting hospital supply demand and converting shortage risk into packing and replenishment action.")

st.markdown("""
**Problem Statement:**
Hospital supply shortages are not just inventory problems  -  they are patient-flow and clinical operations problems. When demand rises faster than supplies can be packed, staged, and replenished, nurses lose time searching for critical items and patient care slows down. MedPack AI forecasts supply demand, identifies shortage risk, and prioritizes packing actions before the shortage reaches the bedside.
""")

# Sidebar Controls
def reset_state():
    st.session_state["current_tokens"] = 0
    st.session_state["total_tokens"] = 0
    st.session_state["last_llm_usage"] = {}
    st.session_state["counted_llm_call_ids"] = set()
    # Reset the visible meter for a new mode/model selection, but the next
    # successful Groq call will immediately repopulate both session state and
    # database/llm_usage_state.json.
    _persist_llm_usage({
        "provider": "none",
        "model": "none",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tokens_used": 0,
        "token_usage_source": "reset",
        "last_call_status": "reset",
        "timestamp": time.time(),
    })
    try:
        requests.post(f"{MEDPACK_API_BASE_URL}/api/reset", timeout=90)
    except:
        pass

with st.sidebar.container(border=True):
    st.subheader("Runtime Mode")

    agent_mode_label = st.radio(
        "Agent execution",
        ["Local AI Mode  -  zero tokens", "Remote LLM Mode  -  may use tokens"],
        index=0,
        on_change=reset_state,
        help="Local mode is deterministic and free. Groq remote mode uses one short, non-streaming Groq call to rewrite the committee response in a more LLM-like style if GROQ_API_KEY is available."
    )
    agent_mode = "remote" if agent_mode_label.startswith("Remote") else "local"

    if agent_mode == "local":
        st.session_state["current_tokens"] = 0
        st.session_state["total_tokens"] = 0
        st.session_state["last_llm_usage"] = {}
        st.success("Local AI Mode: no LLM/API tokens used.")
        selected_provider = None
        selected_model = None
    else:
        selected_provider = st.selectbox(
            "LLM Provider",
            ["Groq", "Google Gemini"],
            index=0,
            on_change=reset_state
        )
        if selected_provider == "Groq":
            st.warning("Groq selected: the committee keeps the freeze-safe local calculation, then uses one short Groq call to rewrite the response if GROQ_API_KEY is available.")
            
            selected_model = st.selectbox(
                "Select Remote Model",
                ["openai/gpt-oss-120b", "qwen/qwen3.6-27b"],
                index=0,
                on_change=reset_state
            )
            
            usage = st.session_state.get("last_llm_usage", {}) or {}
            last_provider = usage.get("provider", "").lower()
            last_status = usage.get("last_call_status", "").lower()

            if last_provider == "groq" and "success" in last_status:
                tokens_used = usage.get("total_tokens", 0)
                used_model = usage.get("model", selected_model)
                st.success(
                    f"**Groq Committee active**  \n"
                    f"Remote Groq call succeeded.  \n\n"
                    f"Provider: Groq  \n"
                    f"Model: {used_model}  \n"
                    f"Status: success  \n"
                    f"Tokens: {tokens_used:,}"
                )
            elif last_provider == "groq" and "fail" in last_status:
                st.error("Groq remote call failed. Groq fell back to local mode and the token meter correctly stayed at 0.")
            else:
                st.info("Groq selected — awaiting first committee execution.")
        else:
            st.warning("Remote LLM Mode selected. Tokens are only used if the backend is explicitly configured with USE_LLM_AGENTS=true and GEMINI_API_KEY.")
            selected_model = st.selectbox(
                "Select Remote Model",
                ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-3.1-flash-lite", "gemini-2.0-flash-exp"],
                index=0,
                on_change=reset_state
            )

    # Render Token Meter
    token_gauge_placeholder = st.empty()
    token_detail_placeholder = st.empty()

def render_token_gauge(tokens_val):
    if agent_mode != "local":
        _sync_llm_usage_from_disk_if_needed()
    tokens_val = int(tokens_val or st.session_state.get("current_tokens", 0) or 0)
    
    if agent_mode == "local":
        tokens_val = 0
        usage = {}
    else:
        usage = st.session_state.get("last_llm_usage", {}) or {}

    gauge_max = max(1000, int(tokens_val * 1.25) if tokens_val else 1000)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = tokens_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Last LLM Tokens", 'font': {'size': 14, 'color': '#f8fafc'}},
        gauge = {
            'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "rgba(255,255,255,0.02)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, gauge_max * 0.25], 'color': "rgba(59, 130, 246, 0.1)"},
                {'range': [gauge_max * 0.25, gauge_max * 0.75], 'color': "rgba(59, 130, 246, 0.3)"},
                {'range': [gauge_max * 0.75, gauge_max], 'color': "rgba(59, 130, 246, 0.5)"}],
        }
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
    token_gauge_placeholder.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
    try:
        provider = usage.get("provider", "Groq" if tokens_val else "None")
        model = usage.get("model", "")
        status = usage.get("last_call_status", "success" if tokens_val else "no successful call yet")
        prompt_t = int(usage.get("prompt_tokens", 0) or 0)
        completion_t = int(usage.get("completion_tokens", 0) or 0)
        src = usage.get("token_usage_source", "none")
        token_detail_placeholder.caption(
            f"Last call: {tokens_val:,} tokens  |  Session total: {int(st.session_state.get('total_tokens', 0)):,}\n\n"
            f"Provider: {provider}  |  Model: {model}  |  Status: {status}\n\n"
            f"Prompt: {prompt_t:,}  |  Completion: {completion_t:,}  |  Source: {src}"
        )
    except Exception:
        pass

render_token_gauge(st.session_state["current_tokens"])

st.sidebar.caption(f"Backend API: `{MEDPACK_API_BASE_URL}`")

with st.sidebar.container(border=True):
    st.subheader("\U0001F4E6 Supply Selection")
    # Options
    DEPARTMENTS = [
        "Emergency Department",
        "ICU",
        "Surgery",
        "Med-Surg",
        "Labor and Delivery",
        "Radiology",
        "Outpatient Clinic"
    ]

    ITEMS = {
        "IV Start Kit": "IV Supplies",
        "Syringe 10ml": "IV Supplies",
        "Nitrile Gloves Medium": "PPE",
        "Nitrile Gloves Large": "PPE",
        "Oxygen Mask": "Respiratory",
        "Wound Care Pack": "Wound Care",
        "Foley Catheter Kit": "Catheterization",
        "Blood Draw Kit": "Lab Supplies",
        "PPE Gown": "PPE",
        "Saline Flush": "IV Supplies",
        "Sterile Gauze": "Wound Care",
        "Surgical Tray": "Surgical Supplies",
        "Nasal Cannula": "Respiratory",
        "Patient Monitoring Leads": "Monitoring"
    }

    dept = st.selectbox("Department", DEPARTMENTS, index=0)
    item = st.selectbox("Supply Item", list(ITEMS.keys()), index=0)
    item_cat = ITEMS[item]
    st.markdown(f"**Category:** `{item_cat}`")

with st.sidebar.container(border=True):
    st.subheader("\U0001F4C8 Operational Metrics")
    current_stock = st.number_input("Current Stock", min_value=0, max_value=500, value=30)
    patient_volume = st.slider("Patient Volume", 1, 100, 15)
    acuity_level = st.slider("Acuity Level (1=Low, 4=Critical)", 1.0, 4.0, 2.5, step=0.1)
    procedure_count = st.slider("Procedure Count", 0, 50, 6)
    recent_usage_rate = st.slider("Recent Usage Rate (units/hr)", 0.0, 50.0, 8.5, step=0.5)

with st.sidebar.container(border=True):
    st.subheader("\U0001F69A Supply & Logistics")
    supplier_delay = st.slider("Supplier Delay (Days)", 0.0, 14.0, 2.5, step=0.5)
    reorder_point = st.number_input("Reorder Point", min_value=0, max_value=200, value=25)
    supplier_reliability = st.slider("Supplier Reliability Score", 0.0, 1.0, 0.9, step=0.05)
    unit_cost_input = st.number_input("Estimated Unit Cost ($)", min_value=0.01, max_value=500.0, value=15.0, step=0.50)
    pack_time = st.number_input("Pack Time (Minutes)", min_value=1.0, max_value=30.0, value=4.5, step=0.5)
    clinical_criticality = st.slider("Clinical Criticality (1-4)", 1, 4, 3)

with st.sidebar.container(border=True):
    st.subheader("\U0001F552 Time Context")
    hour = st.slider("Hour of Day", 0, 23, 12)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"], index=1)

# Build Telemetry Dict
telemetry = {
    "department": dept,
    "item_name": item,
    "item_category": item_cat,
    "current_stock": int(current_stock),
    "patient_volume": int(patient_volume),
    "acuity_level": float(acuity_level),
    "procedure_count": int(procedure_count),
    "recent_usage_rate": float(recent_usage_rate),
    "supplier_delay_days": float(supplier_delay),
    "day_of_week": int(day_of_week),
    "hour": int(hour),
    "season": season,
    "reorder_point": int(reorder_point),
    "unit_cost": float(unit_cost_input), # Stage 4 cost/ROI estimate
    "supplier_reliability_score": float(supplier_reliability),
    "pack_time_minutes": float(pack_time),
    "clinical_criticality": int(clinical_criticality),
    "agent_mode": agent_mode,
    "selected_provider": selected_provider,
    "selected_model": selected_model
}


class _MockResponse:
    def __init__(self):
        self.status_code = 503
        self.text = "MedPack prediction engine is starting. Please wait..."
    def json(self):
        return {"error": self.text}

def api_get(path, params=None):
    try:
        return requests.get(f"{MEDPACK_API_BASE_URL}{path}", params=params or {}, timeout=60)
    except requests.exceptions.ConnectionError:
        return _MockResponse()


def api_post(path, payload=None):
    try:
        return requests.post(f"{MEDPACK_API_BASE_URL}{path}", json=payload or {}, timeout=90)
    except requests.exceptions.ConnectionError:
        return _MockResponse()



def _normalise_groq_model_for_ui(model_name):
    """Keep old/decommissioned Groq model names from breaking the demo."""
    model_name = (model_name or "").strip()
    deprecated = {"llama3-8b-8192", "llama3-70b-8192"}
    if not model_name or model_name in deprecated:
        return "openai/gpt-oss-120b"
    return model_name


def _extract_json_object(text):
    """Parse JSON returned by the LLM, even if it wrapped the object in markdown."""
    if not text:
        raise ValueError("Empty Groq response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)



def _compact_for_groq(value, max_chars=520, max_items=4, depth=0):
    """Return a small JSON-safe version of app state for Groq.

    Groq was returning HTTP 413 because the previous prompt sometimes included
    the full Stage 3/4/5/6 nested result objects. This helper keeps the LLM
    rewrite useful while guaranteeing the request stays small.
    """
    if depth > 4:
        return "..."
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        s = value.strip()
        return s if len(s) <= max_chars else s[:max_chars].rstrip() + "..."
    if isinstance(value, (list, tuple)):
        return [_compact_for_groq(v, max_chars=max_chars, max_items=max_items, depth=depth + 1) for v in list(value)[:max_items]]
    if isinstance(value, dict):
        compact = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 24:
                compact["_truncated_keys"] = len(value) - i
                break
            compact[str(k)] = _compact_for_groq(v, max_chars=max_chars, max_items=max_items, depth=depth + 1)
        return compact
    return _compact_for_groq(str(value), max_chars=max_chars, max_items=max_items, depth=depth + 1)


def _pick_existing(source, *keys, default=None):
    """Pick the first present key from a dict without assuming exact schema."""
    if not isinstance(source, dict):
        return default
    for key in keys:
        if key in source and source.get(key) is not None:
            return source.get(key)
    return default


def _build_tiny_groq_context(telemetry_payload, local_result):
    """Build a tiny, stable context object for the Groq narrative rewrite.

    The LLM should rewrite the decision, not consume the whole dashboard state.
    This keeps the request under Groq/OpenAI-compatible payload limits and fixes
    the 413 Payload Too Large failure that made the token meter stay at zero.
    """
    prediction = local_result.get("prediction", {}) if isinstance(local_result, dict) else {}
    shortage = local_result.get("shortage_risk", {}) if isinstance(local_result, dict) else {}
    packing = local_result.get("packing_priority", {}) if isinstance(local_result, dict) else {}
    usable = local_result.get("usable_stock_analysis", {}) if isinstance(local_result, dict) else {}
    stage3 = local_result.get("stage3_action_plan", {}) if isinstance(local_result, dict) else {}
    stage4 = local_result.get("stage4_roi_analysis", {}) if isinstance(local_result, dict) else {}
    stage5 = local_result.get("stage5_command_center", {}) if isinstance(local_result, dict) else {}
    committee = local_result.get("committee", {}) if isinstance(local_result, dict) else {}

    # Rely purely on the backend's provided RAG knowledge to avoid blocking the UI.
    rag_text = committee.get("rag_knowledge", "")
    if not rag_text or rag_text == "(RAG optional for fast path)":
        rag_text = "RAG knowledge not loaded in fast mode. Run a RAG-enabled workflow to retrieve supporting knowledge."
    committee["rag_knowledge"] = rag_text

    tiny = {
        "selected_case": {
            "department": telemetry_payload.get("department"),
            "item_name": telemetry_payload.get("item_name"),
            "item_category": telemetry_payload.get("item_category"),
            "patient_volume": telemetry_payload.get("patient_volume"),
            "acuity_level": telemetry_payload.get("acuity_level"),
            "procedure_count": telemetry_payload.get("procedure_count"),
            "recent_usage_rate": telemetry_payload.get("recent_usage_rate"),
            "supplier_delay_days": telemetry_payload.get("supplier_delay_days"),
            "supplier_reliability_score": telemetry_payload.get("supplier_reliability_score"),
            "clinical_criticality": telemetry_payload.get("clinical_criticality"),
        },
        "forecast_and_stock": {
            "forecasted_24h_demand": _pick_existing(prediction, "forecasted_24h_demand", "predicted_24h_demand", "prediction"),
            "risk_level": _pick_existing(shortage, "risk_level", "risk"),
            "total_stock": _pick_existing(usable, "total_stock", "current_stock", default=telemetry_payload.get("current_stock")),
            "usable_stock": _pick_existing(usable, "usable_stock"),
            "unsafe_stock": _pick_existing(usable, "unsafe_stock", "expired_recalled_stock"),
            "reserved_stock": _pick_existing(usable, "active_task_reserved_stock", "reserved_stock"),
            "true_shortage_gap": _pick_existing(usable, "true_shortage_gap", "shortage_gap", default=_pick_existing(shortage, "true_shortage_gap", "shortage_gap")),
            "coverage_ratio": _pick_existing(usable, "coverage_ratio", default=_pick_existing(shortage, "coverage_ratio")),
        },
        "packing_decision": {
            "recommended_pack_quantity": _pick_existing(packing, "recommended_pack_quantity", "packing_quantity", "pack_quantity"),
            "priority_score": _pick_existing(packing, "priority_score"),
            "reasoning": _compact_for_groq(_pick_existing(packing, "reasoning", "recommendation", default=""), max_chars=360),
        },
        "stage3_response_options": {
            "best_action": _compact_for_groq(_pick_existing(stage3, "best_action", "recommended_action", "action", default=""), max_chars=260),
            "transfer_quantity": _pick_existing(stage3, "transfer_quantity", "internal_transfer", "recommended_transfer_qty"),
            "post_transfer_gap": _pick_existing(stage3, "post_transfer_gap", "remaining_gap"),
            "supplier_action": _compact_for_groq(_pick_existing(stage3, "supplier_action", "supplier_recommendation", "vendor_recommendation", default=""), max_chars=300),
            "substitute_action": _compact_for_groq(_pick_existing(stage3, "substitute_action", "substitute_recommendation", default=""), max_chars=260),
        },
        "stage4_finance": {
            "shortage_dollars_at_risk": _pick_existing(stage4, "shortage_dollars_at_risk", "shortage_cost_at_risk"),
            "waste_risk_value": _pick_existing(stage4, "waste_risk_value", "expiry_waste_risk_value"),
            "net_estimated_value": _pick_existing(stage4, "net_estimated_value", "net_value"),
            "roi_ratio": _pick_existing(stage4, "roi_ratio", "roi"),
            "summary": _compact_for_groq(_pick_existing(stage4, "summary", "finance_summary", default=""), max_chars=260),
        },
        "rag_knowledge": committee.get("rag_knowledge", ""),
        "stage5_command": {
            "priority_code": _pick_existing(stage5, "priority_code"),
            "command_status": _pick_existing(stage5, "command_status"),
            "response_window_minutes": _pick_existing(stage5, "response_window_minutes"),
            "primary_owner": _pick_existing(stage5, "primary_owner"),
            "escalation_owner": _pick_existing(stage5, "escalation_owner"),
            "commander_decision": _compact_for_groq(_pick_existing(stage5, "commander_decision", "control_tower_summary", default=""), max_chars=420),
        },
        "local_baseline_text": {
            "demand_forecast_agent": _compact_for_groq(_pick_existing(committee, "demand_forecast_agent", default=""), max_chars=280),
            "inventory_risk_agent": _compact_for_groq(_pick_existing(committee, "inventory_risk_agent", default=""), max_chars=280),
            "packing_priority_agent": _compact_for_groq(_pick_existing(committee, "packing_priority_agent", default=""), max_chars=280),
            "clinical_safety_agent": _compact_for_groq(_pick_existing(committee, "clinical_safety_agent", default=""), max_chars=280),
            "final_recommendation_agent": _compact_for_groq(_pick_existing(committee, "final_recommendation_agent", default="") + f" BACKUP POLICY: {rag_text}", max_chars=500),
            "committee_summary": _compact_for_groq(_pick_existing(committee, "committee_summary", default=""), max_chars=360),
        },
    }
    return _compact_for_groq(tiny, max_chars=420, max_items=3)


def _json_compact_limited(obj, max_chars=6500):
    """Compact JSON string with a hard character ceiling for Groq prompts."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    # Last-resort shrink: include the most important top-level facts only.
    fallback = {
        "selected_case": obj.get("selected_case", {}),
        "forecast_and_stock": obj.get("forecast_and_stock", {}),
        "packing_decision": obj.get("packing_decision", {}),
        "stage3_response_options": obj.get("stage3_response_options", {}),
        "stage5_command": obj.get("stage5_command", {}),
        "note": "Context was compacted to avoid Groq 413 Payload Too Large.",
    }
    text = json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
    return text[:max_chars]

def _rough_token_estimate(*texts):
    """Fallback estimate for Groq-compatible responses that omit usage metadata."""
    combined = " ".join(str(t or "") for t in texts)
    # OpenAI/Groq-style tokens average roughly 3-4 chars in English. This is only
    # used when the API response does not return usage.
    return max(1, int(len(combined) / 4))


def _normalise_groq_usage(data, prompt_text, completion_text):
    """Return reliable token accounting for Groq UI calls.

    Groq usually returns usage.total_tokens, but some compatible responses may
    omit it or return null. The sidebar should still show that an LLM call
    happened, so we estimate only when real usage is unavailable.
    """
    usage = data.get("usage", {}) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    estimated = False

    if total_tokens <= 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    if total_tokens <= 0:
        prompt_tokens = _rough_token_estimate(prompt_text)
        completion_tokens = _rough_token_estimate(completion_text)
        total_tokens = prompt_tokens + completion_tokens
        estimated = True

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated": estimated,
    }




def _record_llm_token_usage(provider, model, token_usage, prompt_text="", completion_text="", source_note=""):
    """Store token usage immediately and exactly once per Groq/API call.

    This updates the sidebar meter even when the LLM text cannot be parsed and
    the committee falls back to the local explanation. That was the main reason
    the Groq meter could stay at zero after a real API call.
    """
    token_usage = token_usage or {}
    prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(token_usage.get("total_tokens", 0) or 0)
    estimated = bool(token_usage.get("estimated", False))

    if total_tokens <= 0:
        prompt_tokens = _rough_token_estimate(prompt_text)
        completion_tokens = _rough_token_estimate(completion_text)
        total_tokens = prompt_tokens + completion_tokens
        estimated = True

    call_id = f"{provider}:{model}:{time.time_ns()}:{total_tokens}"
    counted = st.session_state.setdefault("counted_llm_call_ids", set())
    if call_id not in counted:
        counted.add(call_id)
        st.session_state["total_tokens"] = int(st.session_state.get("total_tokens", 0) or 0) + total_tokens

    usage_record = {
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_used": total_tokens,
        "estimated": estimated,
        "token_usage_source": "estimated" if estimated else "groq_usage",
        "source_note": source_note,
        "last_call_status": "success",
        "call_id": call_id,
        "timestamp": time.time(),
    }
    st.session_state["current_tokens"] = total_tokens
    st.session_state["last_llm_usage"] = usage_record
    _persist_llm_usage(usage_record)
    return usage_record




def _record_llm_failure(provider, model, error_message):
    """Record failed remote attempts separately from token usage.

    A failed request has no billable completion token metadata, so the meter
    should remain 0, but the sidebar should explain why.
    """
    usage_record = {
        "provider": provider,
        "model": model or "unknown",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tokens_used": 0,
        "estimated": False,
        "token_usage_source": "no_response",
        "last_call_status": f"failed before token usage: {str(error_message)[:180]}",
        "timestamp": time.time(),
    }
    st.session_state["last_llm_usage"] = usage_record
    st.session_state["current_tokens"] = 0
    _persist_llm_usage(usage_record)
    return usage_record

def _apply_last_llm_usage_to_committee(committee, fallback_mode=False):
    """Copy the last frontend Groq usage into a committee result."""
    usage = st.session_state.get("last_llm_usage", {}) or {}
    if int(usage.get("total_tokens", 0) or 0) <= 0:
        return committee
    committee["tokens_used"] = int(usage.get("total_tokens", 0) or 0)
    committee["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
    committee["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
    committee["token_usage_source"] = usage.get("token_usage_source", "groq_usage")
    committee["remote_model"] = usage.get("model", committee.get("remote_model", "Groq"))
    committee["actual_agent_mode"] = (
        "remote_groq_llm_like_frontend_parse_fallback" if fallback_mode else "remote_groq_llm_like_frontend_safe"
    )
    return committee


def _refresh_sidebar_token_meter_from_state():
    """Redraw the sidebar gauge using authoritative frontend session/disk usage."""
    _sync_llm_usage_from_disk_if_needed()
    render_token_gauge(int(st.session_state.get("current_tokens", 0) or 0))

def _groq_llm_committee_rewrite(telemetry_payload, local_result):
    """Use one short, non-streaming Groq call to make the local committee sound LLM-like.

    This does not call the Flask backend or the old streaming committee path. If Groq
    fails or times out, the caller keeps the no-freeze local response.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in your .env/environment.")

    model_name = _normalise_groq_model_for_ui(
        telemetry_payload.get("selected_model") or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    timeout_seconds = int(os.environ.get("MEDPACK_GROQ_UI_TIMEOUT_SECONDS", "10") or 10)
    timeout_seconds = max(4, min(timeout_seconds, 20))

    compact_context = _build_tiny_groq_context(telemetry_payload, local_result)
    compact_context_json = _json_compact_limited(compact_context, max_chars=int(os.environ.get("MEDPACK_GROQ_CONTEXT_MAX_CHARS", "6500") or 6500))

    system_prompt = (
        "You are MedPack AI's hospital supply-chain control-tower committee. "
        "Make the response sound like a professional LLM advisor, not a dry rule engine. "
        "Use only the supplied facts. Do not invent vendors, patients, or diagnoses. "
        "Be decisive, practical, and concise. Return ONLY valid JSON with the exact keys requested."
    )
    user_prompt = f"""
Rewrite the local rule-based committee output into a natural Groq-powered advisory response.

Tone: confident hospital operations control tower, clear and human, but not overly long.
Style: 2-4 sentences per agent, with specific action language.
Important: weave in Stage 3 supplier/transfer/substitute intelligence, Stage 4 cost/waste/ROI evidence, and Stage 5 command-center priority/action-card logic when present.
CRITICAL: Do NOT summarize the RAG policy away. You MUST explicitly state the exact substitution rule (e.g., "Substitute with Pediatric IV Kits") in the Final Recommendation!

Return valid JSON only with these exact string fields:
- demand_forecast_agent
- inventory_risk_agent
- packing_priority_agent
- clinical_safety_agent
- final_recommendation_agent
- committee_summary

Context JSON, compacted to avoid provider payload limits:
{compact_context_json}
"""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.45,
        "max_tokens": int(os.environ.get("MEDPACK_GROQ_MAX_TOKENS", "650") or 650),
        "response_format": {"type": "json_object"},
        "stream": False,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=(3, timeout_seconds),
    )
    # Some Groq/OpenAI-compatible models may not accept response_format. Retry once
    # without it before falling back to the local no-freeze committee. Do not retry
    # 413 with the same payload; that means the request is still too large.
    if response.status_code == 413:
        raise RuntimeError(
            f"Groq rejected the prompt as too large even after compaction "
            f"({len(system_prompt) + len(user_prompt):,} characters). "
            "Lower MEDPACK_GROQ_CONTEXT_MAX_CHARS or use the compact patch zip."
        )
    if response.status_code >= 400 and "response_format" in payload:
        payload.pop("response_format", None)
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=(3, timeout_seconds),
        )
        if response.status_code == 413:
            raise RuntimeError(
                f"Groq rejected the prompt as too large even after compaction "
                f"({len(system_prompt) + len(user_prompt):,} characters). "
                "Lower MEDPACK_GROQ_CONTEXT_MAX_CHARS or use the compact patch zip."
            )
    response.raise_for_status()
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # Record tokens BEFORE JSON parsing/validation. If Groq returned text but
    # the parser rejects it, the app may still fall back to local wording, but
    # the token meter must still show the real LLM usage.
    token_usage = _normalise_groq_usage(data, system_prompt + "\n" + user_prompt, content)
    usage_record = _record_llm_token_usage(
        provider="Groq",
        model=model_name,
        token_usage=token_usage,
        prompt_text=system_prompt + "\n" + user_prompt,
        completion_text=content,
        source_note="groq_committee_rewrite_response_received",
    )

    rewritten = _extract_json_object(content)

    required = [
        "demand_forecast_agent",
        "inventory_risk_agent",
        "packing_priority_agent",
        "clinical_safety_agent",
        "final_recommendation_agent",
        "committee_summary",
    ]
    for key in required:
        if not str(rewritten.get(key, "")).strip():
            raise RuntimeError(f"Groq response missing required field: {key}")

    rewritten["actual_agent_mode"] = "remote_groq_llm_like_frontend_safe"
    rewritten["tokens_used"] = int(usage_record["total_tokens"])
    rewritten["prompt_tokens"] = int(usage_record["prompt_tokens"])
    rewritten["completion_tokens"] = int(usage_record["completion_tokens"])
    rewritten["token_usage_source"] = usage_record.get("token_usage_source", "groq_usage")
    rewritten["remote_model"] = model_name
    rewritten["mode_note"] = (
        f"Groq LLM-like rewrite used {model_name}. Token source: {rewritten['token_usage_source']} "
        f"(prompt={rewritten['prompt_tokens']:,}, completion={rewritten['completion_tokens']:,}, total={rewritten['tokens_used']:,}). "
        "The freeze-safe Streamlit committee still calculates the facts locally, then Groq rewrites the explanation in one short non-streaming call. "
        "If Groq fails, MedPack falls back to local text."
    )
    rewritten["fallback_mode"] = False
    return rewritten


# Main Layout Columns
st.sidebar.markdown("---")
show_sys_panels = st.sidebar.checkbox("Show Right-Hand Side Panels", value=False)

if show_sys_panels:
    col1, col2 = st.columns([2, 1])
else:
    col1 = st.container()
    col2 = None

with col1:
    st.header("\U0001F4CA Prediction & Logistics Actions")
    
    # Buttons
    is_predicting = st.session_state.get("is_predicting", False)
    run_forecast = st.button("\U0001F52E Predict 24-Hour Supply Demand", disabled=is_predicting)
    run_committee_btn = st.button("\U0001F9E0 Run MedPack Committee Decision", disabled=is_predicting)
    
    if run_forecast or run_committee_btn:
        st.session_state["is_predicting"] = True
        # Call Backend. Stage 2 fix: every committee path has a timeout and a local fallback.
        try:
            result = None
            request_payload = dict(telemetry)

            def run_local_committee_fallback(reason=None):
                """Guarantee the button returns a decision instead of hanging on remote/stream mode."""
                fallback_payload = dict(request_payload)
                fallback_payload["agent_mode"] = "local"
                if reason:
                    st.warning(f"Remote/stream committee did not finish cleanly, so MedPack used the local zero-token committee. Reason: {reason}")
                fallback_res = requests.post(
                    f"{MEDPACK_API_BASE_URL}/api/run-medpack-committee",
                    json=fallback_payload,
                    timeout=25,
                )
                if fallback_res.status_code == 200:
                    return fallback_res.json()
                st.error(f"Local fallback also failed: {fallback_res.status_code}: {fallback_res.text}")
                return None

            if run_committee_btn:
                st.markdown("---")
                st.header("Agentic Committee Panel")
                st.caption("Freeze Guard v4 + Stage 5/6 is active: the core decision runs locally first, then Groq can rewrite the wording if selected.")
                lights_ph = st.empty()

                agents = ["Demand Forecast Agent", "Inventory Risk Agent", "Packing Priority Agent", "Clinical Safety Agent", "Supplier & Transfer Agent", "Cost/Waste/ROI Agent", "Command Center Agent", "Final Recommendation Agent", "Committee Summarizer"]
                status_map = {a: "pending" for a in agents}

                def render_lights():
                    """Render a polished command-center flow map with directional cues.

                    The tracker is intentionally compact: three-card rows, subtle arrows,
                    row handoff indicators, active-step highlighting, and a legend.
                    It stays inside the Streamlit page width and avoids raw HTML output.
                    """
                    import html as _html

                    done_count = sum(1 for status in status_map.values() if status == "done")
                    running_count = sum(1 for status in status_map.values() if status == "running")
                    active_label = next((name for name, status in status_map.items() if status == "running"), "Standing by")
                    progress_pct = int(((done_count + (0.5 if running_count else 0)) / max(len(agents), 1)) * 100)
                    progress_pct = max(0, min(100, progress_pct))

                    def _card(idx, agent_name):
                        status = status_map.get(agent_name, "pending")
                        if status == "done":
                            icon, label, css = "", "Complete", "done"
                        elif status == "running":
                            icon, label, css = "", "Running", "running"
                        else:
                            icon, label, css = "", "Queued", "queued"
                        return (
                            "<div class='mp-flow-card {css}'>"
                            "  <div class='mp-flow-card-top'>"
                            "    <span class='mp-flow-step'>STEP {idx}</span>"
                            "    <span class='mp-flow-pill {css}'>{icon} {label}</span>"
                            "  </div>"
                            "  <div class='mp-flow-name'>{agent}</div>"
                            "</div>"
                        ).format(css=css, idx=idx, icon=icon, label=label, agent=_html.escape(agent_name))

                    def _row(start_idx, row_agents):
                        parts = []
                        for offset, name in enumerate(row_agents):
                            step = start_idx + offset
                            parts.append(_card(step, name))
                            if offset < len(row_agents) - 1:
                                parts.append("<div class='mp-flow-arrow' aria-label='next'>-></div>")
                        return "<div class='mp-flow-row'>" + "".join(parts) + "</div>"

                    rows_html = []
                    row_groups = [agents[0:3], agents[3:6], agents[6:9]]
                    for row_index, group in enumerate(row_groups):
                        rows_html.append(_row(row_index * 3 + 1, group))
                        if row_index < len(row_groups) - 1:
                            from_step = (row_index + 1) * 3
                            to_step = from_step + 1
                            rows_html.append(
                                "<div class='mp-row-handoff'>"
                                "  <span class='mp-row-handoff-line'></span>"
                                "  <span class='mp-row-handoff-badge'>Step {from_step} -> Step {to_step}</span>"
                                "  <span class='mp-row-handoff-arrow'>v</span>"
                                "  <span class='mp-row-handoff-line'></span>"
                                "</div>".format(from_step=from_step, to_step=to_step)
                            )

                    tracker_html = """
                    <style>
                        .mp-flow-shell {{
                            width: 100%;
                            border: 1px solid rgba(148, 163, 184, 0.22);
                            border-radius: 20px;
                            padding: 16px 18px 18px 18px;
                            background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.10), transparent 34%),
                                        linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.70));
                            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.24);
                            margin: 8px 0 24px 0;
                            box-sizing: border-box;
                            overflow: hidden;
                        }}
                        .mp-flow-header {{
                            display: flex;
                            justify-content: space-between;
                            align-items: flex-start;
                            gap: 16px;
                            margin-bottom: 10px;
                        }}
                        .mp-flow-title {{
                            color: #f8fafc;
                            font-weight: 900;
                            letter-spacing: 0.2px;
                            font-size: 1.05rem;
                            margin-bottom: 4px;
                        }}
                        .mp-flow-subtitle {{
                            color: #cbd5e1;
                            font-size: 0.84rem;
                            line-height: 1.35;
                        }}
                        .mp-flow-counter {{
                            color: #dbeafe;
                            background: rgba(59, 130, 246, 0.16);
                            border: 1px solid rgba(96, 165, 250, 0.34);
                            border-radius: 999px;
                            padding: 6px 12px;
                            font-size: 0.80rem;
                            font-weight: 900;
                            white-space: nowrap;
                        }}
                        .mp-progress-track {{
                            height: 8px;
                            background: rgba(51, 65, 85, 0.84);
                            border-radius: 999px;
                            overflow: hidden;
                            margin: 10px 0 12px 0;
                        }}
                        .mp-progress-fill {{
                            height: 100%;
                            width: {progress_pct}%;
                            background: linear-gradient(90deg, #38bdf8, #22c55e);
                            border-radius: 999px;
                            box-shadow: 0 0 18px rgba(34, 197, 94, 0.38);
                            transition: width 240ms ease;
                        }}
                        .mp-flow-legend {{
                            display: flex;
                            flex-wrap: wrap;
                            gap: 8px;
                            margin: 0 0 12px 0;
                            color: #94a3b8;
                            font-size: 0.74rem;
                        }}
                        .mp-legend-chip {{
                            border: 1px solid rgba(148, 163, 184, 0.18);
                            background: rgba(15, 23, 42, 0.52);
                            border-radius: 999px;
                            padding: 4px 9px;
                        }}
                        .mp-flow-map {{
                            display: flex;
                            flex-direction: column;
                            gap: 7px;
                            width: 100%;
                        }}
                        .mp-flow-row {{
                            display: grid;
                            grid-template-columns: minmax(0, 1fr) 34px minmax(0, 1fr) 34px minmax(0, 1fr);
                            gap: 8px;
                            align-items: center;
                            width: 100%;
                        }}
                        .mp-flow-card {{
                            min-height: 68px;
                            border-radius: 14px;
                            padding: 10px 12px;
                            box-sizing: border-box;
                            border: 1px solid rgba(148, 163, 184, 0.18);
                            background: rgba(15, 23, 42, 0.74);
                        }}
                        .mp-flow-card.done {{
                            background: rgba(16, 185, 129, 0.13);
                            border-color: rgba(16, 185, 129, 0.36);
                        }}
                        .mp-flow-card.running {{
                            background: rgba(37, 99, 235, 0.18);
                            border-color: rgba(56, 189, 248, 0.78);
                            box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.20), 0 0 24px rgba(56, 189, 248, 0.22);
                        }}
                        .mp-flow-card.queued {{
                            background: rgba(30, 41, 59, 0.48);
                            border-color: rgba(148, 163, 184, 0.14);
                            opacity: 0.78;
                        }}
                        .mp-flow-card-top {{
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            gap: 8px;
                            margin-bottom: 9px;
                        }}
                        .mp-flow-step {{
                            color: #bae6fd;
                            font-size: 0.70rem;
                            font-weight: 900;
                            text-transform: uppercase;
                            letter-spacing: 0.10em;
                        }}
                        .mp-flow-pill {{
                            border-radius: 999px;
                            padding: 3px 8px;
                            font-size: 0.67rem;
                            font-weight: 900;
                            white-space: nowrap;
                        }}
                        .mp-flow-pill.done {{
                            color: #bbf7d0;
                            background: rgba(34, 197, 94, 0.14);
                            border: 1px solid rgba(34, 197, 94, 0.32);
                        }}
                        .mp-flow-pill.running {{
                            color: #e0f2fe;
                            background: rgba(14, 165, 233, 0.18);
                            border: 1px solid rgba(56, 189, 248, 0.45);
                        }}
                        .mp-flow-pill.queued {{
                            color: #cbd5e1;
                            background: rgba(148, 163, 184, 0.10);
                            border: 1px solid rgba(148, 163, 184, 0.18);
                        }}
                        .mp-flow-name {{
                            color: #f8fafc;
                            font-size: 0.88rem;
                            font-weight: 800;
                            line-height: 1.25;
                        }}
                        .mp-flow-arrow {{
                            height: 100%;
                            min-height: 44px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: #38bdf8;
                            font-size: 1.15rem;
                            font-weight: 900;
                            opacity: 0.92;
                        }}
                        .mp-row-handoff {{
                            display: grid;
                            grid-template-columns: 1fr auto auto 1fr;
                            align-items: center;
                            gap: 8px;
                            padding: 0 8px;
                            color: #93c5fd;
                        }}
                        .mp-row-handoff-line {{
                            height: 1px;
                            background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.32), transparent);
                        }}
                        .mp-row-handoff-badge {{
                            font-size: 0.68rem;
                            font-weight: 900;
                            letter-spacing: 0.06em;
                            text-transform: uppercase;
                            color: #c4b5fd;
                            border: 1px solid rgba(139, 92, 246, 0.26);
                            background: rgba(88, 28, 135, 0.16);
                            border-radius: 999px;
                            padding: 3px 8px;
                            white-space: nowrap;
                        }}
                        .mp-row-handoff-arrow {{
                            color: #38bdf8;
                            font-size: 1.05rem;
                            font-weight: 900;
                        }}
                        @media (max-width: 900px) {{
                            .mp-flow-row {{
                                grid-template-columns: 1fr;
                            }}
                            .mp-flow-arrow {{
                                min-height: 18px;
                                height: 18px;
                                transform: rotate(90deg);
                            }}
                            .mp-row-handoff {{
                                grid-template-columns: 1fr auto 1fr;
                            }}
                            .mp-row-handoff-badge {{
                                display: none;
                            }}
                        }}
                    </style>
                    <div class='mp-flow-shell'>
                        <div class='mp-flow-header'>
                            <div>
                                <div class='mp-flow-title'>Agent Execution Flow</div>
                                <div class='mp-flow-subtitle'>Current step: <b>{active_label}</b></div>
                            </div>
                            <div class='mp-flow-counter'>{done_count}/{total_count} complete</div>
                        </div>
                        <div class='mp-progress-track'><div class='mp-progress-fill'></div></div>
                        <div class='mp-flow-legend'>
                            <span class='mp-legend-chip'> Complete</span>
                            <span class='mp-legend-chip'> Running</span>
                            <span class='mp-legend-chip'> Queued</span>
                            <span class='mp-legend-chip'>Flow: left -> right, then down to next row</span>
                        </div>
                        <div class='mp-flow-map'>{rows}</div>
                    </div>
                    """.format(
                        progress_pct=progress_pct,
                        active_label=_html.escape(active_label),
                        done_count=done_count,
                        total_count=len(agents),
                        rows="".join(rows_html),
                    )

                    lights_ph.empty()
                    with lights_ph.container():
                        st.markdown(tracker_html, unsafe_allow_html=True)
                render_lights()

                # Freeze Guard v4: the committee button is now fully client-side.
                # This avoids ALL possible backend/request/stream/LLM/model-lock hangs during the demo.
                # The backend still exists for prediction and control-tower panels, but this button
                # returns a Stage 2-style committee decision immediately inside Streamlit.
                for a in agents:
                    status_map[a] = "running"
                    render_lights()
                    time.sleep(0.08)
                    status_map[a] = "done"
                    render_lights()
                    time.sleep(0.04)
                render_lights()

                forecast = max(
                    0.0,
                    (float(recent_usage_rate) * 0.8)
                    + (float(patient_volume) * float(acuity_level) * 0.15)
                    + (float(procedure_count) * 0.5)
                    + (2.0 if float(supplier_delay) > 4 else 0.0),
                )

                # Local Stage 2 usable-stock approximation.  It intentionally avoids API calls.
                # The slider remains the source of truth for the selected scenario's total stock.
                total_stock = int(current_stock)
                unsafe_stock = 0
                transfer_candidate_stock = 0
                reserved_stock = 0
                try:
                    inv_path = os.path.join("database", "inventory_state.json")
                    if os.path.exists(inv_path):
                        with open(inv_path, "r", encoding="utf-8") as f:
                            inv_records = json.load(f)
                        if isinstance(inv_records, dict):
                            inv_records = inv_records.get("inventory", inv_records.get("records", []))
                        for rec in inv_records or []:
                            if str(rec.get("item_name", "")).strip() != str(item).strip():
                                continue
                            rec_stock = int(float(rec.get("current_stock", 0) or 0))
                            rec_dept = str(rec.get("department", "")).strip()
                            recall_status = str(rec.get("recall_status", "Clear")).strip().lower()
                            exp_date = str(rec.get("expiration_date", "")).strip()
                            is_recalled = recall_status not in {"", "clear", "none", "ok", "safe"}
                            is_expired = False
                            try:
                                from datetime import date, datetime
                                if exp_date:
                                    is_expired = datetime.fromisoformat(exp_date[:10]).date() < date.today()
                            except Exception:
                                is_expired = False
                            if rec_dept == str(dept).strip() and (is_recalled or is_expired):
                                unsafe_stock += rec_stock
                            elif rec_dept != str(dept).strip() and not (is_recalled or is_expired):
                                transfer_candidate_stock += rec_stock
                except Exception:
                    unsafe_stock = 0
                    transfer_candidate_stock = 0

                try:
                    task_path = os.path.join("database", "packing_tasks.jsonl")
                    if os.path.exists(task_path):
                        with open(task_path, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    task = json.loads(line)
                                except Exception:
                                    continue
                                if str(task.get("item_name", "")).strip() != str(item).strip():
                                    continue
                                if str(task.get("department", "")).strip() != str(dept).strip():
                                    continue
                                if str(task.get("status", "NEW")).upper() in {"NEW", "ASSIGNED", "PICKING", "PACKED"}:
                                    reserved_stock += int(float(task.get("quantity", 0) or 0))
                except Exception:
                    reserved_stock = 0

                usable_stock = max(0, total_stock - unsafe_stock - reserved_stock)
                true_gap = round(forecast - usable_stock, 2)
                post_transfer_gap = round(forecast - usable_stock - transfer_candidate_stock, 2)
                coverage_ratio = usable_stock / max(forecast, 1.0)
                risk_level = (
                    "Critical" if true_gap >= 30 or coverage_ratio < 0.50 else
                    "High" if true_gap >= 15 or coverage_ratio < 0.75 else
                    "Medium" if true_gap >= 5 or coverage_ratio < 1.0 else
                    "Low"
                )
                safety_buffer = 20 if risk_level == "Critical" else 10 if risk_level == "High" else 5 if risk_level == "Medium" else 0
                pack_qty = max(0, int(round(true_gap + safety_buffer)))
                if true_gap > 0 and transfer_candidate_stock > 0:
                    action = f"Pack {pack_qty} units of {item} for {dept}; also review {transfer_candidate_stock} transferable units in other departments."
                elif true_gap > 0:
                    action = f"Pack {pack_qty} units of {item} for {dept} and consider replenishment because usable stock is short."
                else:
                    action = f"No urgent pack required for {item}; usable stock covers the local forecast."

                result = {
                    "prediction": {"predicted_24h_demand": round(forecast, 2), "model_type": "streamlit_v4_no_backend_committee"},
                    "shortage_risk": {
                        "predicted_24h_demand": round(forecast, 2),
                        "current_stock": total_stock,
                        "usable_stock": usable_stock,
                        "shortage_gap": true_gap,
                        "coverage_ratio": round(coverage_ratio, 4),
                        "risk_level": risk_level,
                        "reasoning": (
                            f"Freeze Fix v4: committee ran fully inside Streamlit with no backend request. "
                            f"Risk uses usable stock ({usable_stock}) instead of total stock ({total_stock})."
                        ),
                    },
                    "packing_priority": {
                        "priority_score": 95 if risk_level == "Critical" else 80 if risk_level == "High" else 55 if risk_level == "Medium" else 20,
                        "recommended_pack_quantity": pack_qty,
                        "recommended_action": action,
                        "escalation_required": risk_level in ["Critical", "High"],
                        "reasoning": "No-freeze local committee path. It does not call Groq, Gemini, Flask, streaming, or the XGBoost model.",
                    },
                    "usable_stock_analysis": {
                        "predicted_24h_demand": round(forecast, 2),
                        "total_stock": total_stock,
                        "usable_stock": usable_stock,
                        "unsafe_stock": unsafe_stock,
                        "active_task_reserved_stock": reserved_stock,
                        "transfer_candidate_stock": transfer_candidate_stock,
                        "true_shortage_gap": true_gap,
                        "post_transfer_gap": post_transfer_gap,
                        "recommended_stage2_action": action,
                        "safety_notes": [
                            "Committee button is now guaranteed local/no-backend to prevent UI freezing.",
                            "Use the Predict button and Stage 2 panel for full backend/model validation."
                        ],
                    },
                    "committee": {
                        "demand_forecast_agent": f"Local no-freeze forecast estimates {forecast:.1f} units needed over 24 hours for {item} in {dept}.",
                        "inventory_risk_agent": f"Stage 2 usable stock check: total={total_stock}, unsafe={unsafe_stock}, reserved={reserved_stock}, usable={usable_stock}, true gap={true_gap}.",
                        "packing_priority_agent": f"Packing quantity recommendation is {pack_qty} units. {action}",
                        "clinical_safety_agent": f"Clinical risk is {risk_level}. Expired/recalled and already-reserved stock are not treated as bedside-available supply.",
                        "final_recommendation_agent": action,
                        "committee_summary": f"No-freeze committee result: {item} in {dept} is {risk_level} risk using usable-stock logic. {action}",
                        "actual_agent_mode": "streamlit_v4_no_backend_no_llm",
                        "tokens_used": 0,
                        "mode_note": "Freeze Fix v4: this committee button does not make a backend request at all, so it cannot hang on Flask, Groq, Gemini, streaming, model loading, or file writes.",
                    },
                }

                # Stage 3: Supplier + Transfer Intelligence, still no backend request.
                # This adds transfer, vendor, substitute, and action-plan logic to the no-freeze committee path.
                try:
                    from backend.stage3_action_plan import build_stage3_action_plan
                    stage3_plan = build_stage3_action_plan(request_payload, predicted_24h_demand=forecast)
                    result["stage3_action_plan"] = stage3_plan
                    result["committee"]["stage3_control_tower_agent"] = stage3_plan.get("control_tower_summary", "Stage 3 action plan generated.")
                    result["committee"]["final_recommendation_agent"] = stage3_plan.get("final_recommendation", result["committee"].get("final_recommendation_agent", action))
                    result["committee"]["committee_summary"] = (
                        f"No-freeze Stage 3 committee result: {item} in {dept} is {risk_level} risk. "
                        f"Best action: {stage3_plan.get('best_action', 'Stage 2 action only')}. "
                        f"{stage3_plan.get('final_recommendation', action)}"
                    )
                except Exception as stage3_exc:
                    result["stage3_action_plan"] = {
                        "stage": "Stage 3 - Supplier + Transfer Intelligence",
                        "status": "fallback",
                        "best_action": "Stage 2 action only",
                        "final_recommendation": f"Stage 3 local action plan failed, so MedPack kept Stage 2 output. Reason: {stage3_exc}",
                    }

                # Stage 4: Cost, Waste, ROI and Executive Value, still no backend request.
                try:
                    from backend.stage4_roi import build_stage4_roi_analysis
                    stage4_roi = build_stage4_roi_analysis(
                        request_payload,
                        predicted_24h_demand=forecast,
                        stage3_plan=result.get("stage3_action_plan"),
                    )
                    result["stage4_roi_analysis"] = stage4_roi
                    result["committee"]["stage4_financial_impact_agent"] = stage4_roi.get(
                        "control_tower_summary",
                        stage4_roi.get("executive_recommendation", "Stage 4 financial view generated.")
                    )
                    result["committee"]["committee_summary"] = (
                        result["committee"].get("committee_summary", "")
                        + f" Stage 4 estimate: net value ${stage4_roi.get('net_value_estimate', 0):,.0f}; "
                        + stage4_roi.get("executive_recommendation", "")
                    )
                except Exception as stage4_exc:
                    result["stage4_roi_analysis"] = {
                        "stage": "Stage 4 - Cost, Waste & ROI Executive Dashboard",
                        "status": "fallback",
                        "executive_recommendation": f"Stage 4 local ROI view failed, so MedPack kept Stage 3 output. Reason: {stage4_exc}",
                        "net_value_estimate": 0,
                    }


                # Stage 5: Agentic Command Center, still no backend request.
                # This wraps Stages 1-4 into owner-based action cards, priority status, escalation, and a final commander decision.
                try:
                    from backend.stage5_command_center import build_stage5_command_center
                    stage5_command = build_stage5_command_center(
                        request_payload,
                        predicted_24h_demand=forecast,
                        stage3_plan=result.get("stage3_action_plan"),
                        stage4_roi=result.get("stage4_roi_analysis"),
                    )
                    result["stage5_command_center"] = stage5_command
                    result["committee"]["stage5_command_center_agent"] = stage5_command.get(
                        "control_tower_summary",
                        stage5_command.get("commander_decision", "Stage 5 command center generated.")
                    )
                    result["committee"]["final_recommendation_agent"] = stage5_command.get(
                        "commander_decision",
                        result["committee"].get("final_recommendation_agent", action)
                    )
                    result["committee"]["committee_summary"] = (
                        f"Stage 5 Command Center: {stage5_command.get('priority_code', 'P3')} / "
                        f"{stage5_command.get('command_status', 'GREEN - Monitor')} for {item} in {dept}. "
                        f"{stage5_command.get('commander_decision', result['committee'].get('committee_summary', action))}"
                    )
                except Exception as stage5_exc:
                    result["stage5_command_center"] = {
                        "stage": "Stage 5 - Agentic Command Center",
                        "status": "fallback",
                        "priority_code": "P3",
                        "command_status": "GREEN - Monitor",
                        "commander_decision": f"Stage 5 local command center failed, so MedPack kept Stage 4 output. Reason: {stage5_exc}",
                        "action_cards": [],
                    }


                # Groq LLM-like mode: keep the freeze-safe local calculation, then ask
                # Groq to rewrite the committee language in one short non-streaming call.
                if agent_mode == "remote" and selected_provider == "Groq":
                    try:
                        with st.spinner("Groq is rewriting the committee into an LLM-style control-tower response..."):
                            groq_committee = _groq_llm_committee_rewrite(request_payload, result)
                        result["committee"].update(groq_committee)
                        result["prediction"]["model_type"] = "streamlit_v4_safe_local_calc_plus_groq_rewrite"
                        result["shortage_risk"]["reasoning"] += " Groq rewrote the committee explanation, but did not change the numeric usable-stock calculation."
                        result["packing_priority"]["reasoning"] += " Groq was used only for language/narrative quality, not for the underlying calculation."
                    except Exception as groq_exc:
                        # If Groq returned an HTTP response but the JSON rewrite could not be
                        # parsed/validated, keep the local wording but preserve the token usage.
                        result["committee"] = _apply_last_llm_usage_to_committee(result["committee"], fallback_mode=True)
                        if int(result["committee"].get("tokens_used", 0) or 0) <= 0:
                            failure_usage = _record_llm_failure("Groq", selected_model or "Groq", groq_exc)
                            result["committee"]["actual_agent_mode"] = "streamlit_v4_local_fallback_after_groq_error"
                            result["committee"]["tokens_used"] = 0
                            result["committee"]["token_usage_source"] = failure_usage.get("token_usage_source", "no_response")
                        result["committee"]["fallback_mode"] = True
                        result["committee"]["mode_note"] = (
                            "Groq was selected. If Groq returned a response, its token usage is preserved in the meter; "
                            "if it failed before any response arrived, there are no Groq usage tokens to count and the sidebar explains the failure status. "
                            f"MedPack kept the no-freeze local committee response. Reason: {groq_exc}"
                        )
                # Release lock for Groq path
                st.session_state["is_predicting"] = False
            else:
                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    try:
                        msg = "Running forecast pipeline..." if attempt == 1 else f"MedPack prediction engine is starting... (Attempt {attempt}/{max_attempts})"
                        with st.spinner(msg):
                            res = requests.post(
                                f"{MEDPACK_API_BASE_URL}/api/run-medpack-committee-fast",
                                json=request_payload,
                                timeout=(5, 20),
                            )
                        if res.status_code == 200:
                            result = res.json()
                            break
                        else:
                            st.error(f"Backend API returned error code {res.status_code}: {res.text}")
                            break
                    except requests.exceptions.RequestException as e:
                        if attempt < max_attempts:
                            time.sleep(attempt)
                        else:
                            st.info("MedPack prediction engine is starting. Please wait...")
                            break
                
                # Release lock after API call is done
                st.session_state["is_predicting"] = False

            if result:
                # Persist the locked deterministic result so the Foundry panel
                # survives Streamlit reruns triggered by its own button.
                st.session_state["medpack_deterministic_result"] = result
                st.session_state["foundry_result"] = None  # reset on new run
                prediction = result["prediction"]
                shortage = result["shortage_risk"]
                priority = result["packing_priority"]
                committee = result["committee"]
                usable_analysis = result.get("usable_stock_analysis", {})
                
                # Render Metrics
                st.markdown("### Live Analytics Output")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                with m_col1:
                    st.metric(
                        label="Forecasted 24h Demand",
                        value=f"{prediction['predicted_24h_demand']:.1f} units"
                    )
                with m_col2:
                    st.metric(
                        label="True Shortage Gap",
                        value=f"{shortage['shortage_gap']:.1f} units",
                        delta=f"Usable Stock: {shortage.get('usable_stock', shortage.get('current_stock', current_stock))}"
                    )
                with m_col3:
                    st.metric(
                        label="Coverage Ratio",
                        value=f"{shortage['coverage_ratio'] * 100:.1f}%"
                    )
                with m_col4:
                    risk_color = "red" if shortage["risk_level"] == "Critical" else "orange" if shortage["risk_level"] == "High" else "yellow" if shortage["risk_level"] == "Medium" else "green"
                    st.markdown(f"<div style='text-align: center;'><span class='metric-label'>Risk Level</span><br><span style='color: {risk_color}; font-size: 1.8rem; font-weight: bold;'>{shortage['risk_level']}</span></div>", unsafe_allow_html=True)
                
                # Shortage & Packing Panels
                st.info(f"**Shortage Logic Explanation:** {shortage['reasoning']}")
                
                # Stage 2 Forecast + Usable Stock Panel
                if usable_analysis:
                    st.markdown("---")
                    st.markdown("### \U0001F9EE Demand Forecast vs Usable Stock")
                    st.caption("The forecast is now compared against safe, usable stock instead of raw total stock.")
                    u1, u2, u3, u4, u5, u6 = st.columns(6)
                    u1.metric("Forecast Demand", f"{usable_analysis.get('predicted_24h_demand', 0):.1f}")
                    u2.metric("Total Stock", usable_analysis.get("total_stock", 0))
                    u3.metric("Usable Stock", usable_analysis.get("usable_stock", 0))
                    u4.metric("Unsafe Stock", usable_analysis.get("unsafe_stock", 0))
                    u5.metric("Task-Reserved", usable_analysis.get("active_task_reserved_stock", 0))
                    u6.metric("True Gap", f"{usable_analysis.get('true_shortage_gap', 0):.1f}")
                    st.warning(usable_analysis.get("recommended_stage2_action", "No usable-stock action returned."))
                    notes = usable_analysis.get("safety_notes", [])
                    if notes:
                        st.caption("| ".join(notes))
                    transfer_options = usable_analysis.get("top_transfer_options", [])
                    if transfer_options:
                        with st.expander("Transfer candidates from other departments", expanded=False):
                            st.dataframe(pd.DataFrame(transfer_options), use_container_width=True)
                
                # Stage 3 Supplier + Transfer Intelligence Panel
                stage3_plan = result.get("stage3_action_plan", {})
                if stage3_plan:
                    st.markdown("---")
                    st.markdown("### Supply Chain Response Plan")
                    st.caption("After finding the true shortage, the app decides whether to transfer, order, substitute, or escalate.")
                    transfer = stage3_plan.get("transfer_recommendation", {}) or {}
                    supplier = stage3_plan.get("supplier_risk", {}) or {}
                    substitute = stage3_plan.get("substitute_options", {}) or {}
                    s3a, s3b, s3c, s3d = st.columns(4)
                    s3a.metric("Best Action", stage3_plan.get("best_action", "N/A"))
                    s3b.metric("Transfer Qty", transfer.get("recommended_transfer_qty", 0))
                    s3c.metric("Post-Transfer Gap", f"{stage3_plan.get('post_transfer_gap', 0):.1f}")
                    s3d.metric("Supplier Status", supplier.get("supplier_status", "N/A"))
                    st.success(stage3_plan.get("final_recommendation", "No supply-chain recommendation returned."))
                    sequence = stage3_plan.get("recommended_sequence", [])
                    if sequence:
                        st.markdown("#### Action Sequence")
                        for step in sequence:
                            st.write(f"- {step}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        with st.expander("Transfer options", expanded=False):
                            options = transfer.get("transfer_options", [])
                            if options:
                                st.dataframe(pd.DataFrame(options), use_container_width=True)
                            else:
                                st.write(transfer.get("recommendation", "No transfer options found."))
                    with c2:
                        with st.expander("Supplier ranking", expanded=False):
                            vendors = supplier.get("ranked_vendors", [])
                            if vendors:
                                display_vendor_cols = ["vendor_name", "role", "adjusted_delay_days", "reliability_score", "backorder_probability", "unit_cost_multiplier", "recommended_order_qty", "supplier_score"]
                                df_v = pd.DataFrame(vendors)
                                st.dataframe(df_v[[c for c in display_vendor_cols if c in df_v.columns]], use_container_width=True)
                            else:
                                st.write(supplier.get("recommendation", "No vendor data returned."))
                    with c3:
                        with st.expander("Substitute options", expanded=False):
                            subs = substitute.get("substitute_options", [])
                            if subs:
                                display_sub_cols = ["substitute_item", "clinical_fit", "suitability_score", "total_available_substitute_stock", "recommended_substitute_qty", "acceptable"]
                                df_s = pd.DataFrame(subs)
                                st.dataframe(df_s[[c for c in display_sub_cols if c in df_s.columns]], use_container_width=True)
                            else:
                                st.write(substitute.get("recommendation", "No substitute options found."))


                # Stage 4 Cost, Waste, ROI Executive Dashboard
                stage4_roi = result.get("stage4_roi_analysis", {})
                if stage4_roi:
                    st.markdown("---")
                    st.markdown("### Financial Impact & ROI Dashboard")
                    st.caption("The operational recommendation is translated into estimated dollars at risk, waste exposure, action cost, and value protected.")
                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("Shortage Risk $", f"${stage4_roi.get('shortage_risk_value', 0):,.0f}")
                    r2.metric("Waste Risk $", f"${(stage4_roi.get('waste_risk') or {}).get('total_waste_risk_value', 0):,.0f}")
                    r3.metric("Action Cost", f"${stage4_roi.get('estimated_action_cost', 0):,.0f}")
                    r4.metric("Net Value", f"${stage4_roi.get('net_value_estimate', 0):,.0f}")
                    r5.metric("ROI Ratio", f"{stage4_roi.get('roi_ratio', 0):,.2f}x")
                    st.success(stage4_roi.get("executive_recommendation", "No financial recommendation returned."))
                    with st.expander("Financial details: assumptions and value breakdown", expanded=False):
                        b1, b2 = st.columns(2)
                        with b1:
                            st.markdown("**Value Protected**")
                            st.json({
                                "stockout_risk_value_protected": stage4_roi.get("stockout_risk_value_protected"),
                                "transfer_savings_vs_emergency": stage4_roi.get("transfer_savings_vs_emergency"),
                                "labor_time_value": stage4_roi.get("labor_time_value"),
                                "gross_value_protected": stage4_roi.get("gross_value_protected"),
                            })
                        with b2:
                            st.markdown("**Action Costs / Exposure**")
                            st.json({
                                "emergency_order_cost": stage4_roi.get("emergency_order_cost"),
                                "emergency_premium_cost": stage4_roi.get("emergency_premium_cost"),
                                "transfer_labor_cost": stage4_roi.get("transfer_labor_cost"),
                                "monthly_carrying_cost": stage4_roi.get("monthly_carrying_cost"),
                                "supplier_delay_penalty": stage4_roi.get("supplier_delay_penalty"),
                            })
                        waste_rows = (stage4_roi.get("waste_risk") or {}).get("expiration_rows", [])
                        if waste_rows:
                            st.markdown("**Expiring-soon lots**")
                            st.dataframe(pd.DataFrame(waste_rows), use_container_width=True)
                        st.caption((stage4_roi.get("assumptions_used") or {}).get("note", "Demo estimates only."))

                # Stage 5 Agentic Command Center
                stage5_command = result.get("stage5_command_center", {})
                if stage5_command:
                    st.markdown("---")
                    st.markdown("### \U0001F9ED Command Center")
                    st.caption("The final command layer: converts all analysis into priority status, owners, response windows, action cards, escalation, and an executive handoff.")
                    z1, z2, z3, z4, z5 = st.columns(5)
                    z1.metric("Priority", stage5_command.get("priority_code", "P3"))
                    z2.metric("Command Status", stage5_command.get("command_status", "Monitor"))
                    z3.metric("Response Window", f"{stage5_command.get('response_window_minutes', 0)} min")
                    z4.metric("Open Actions", stage5_command.get("open_action_count", 0))
                    z5.metric("Net Value", f"${stage5_command.get('net_value_estimate', 0):,.0f}")
                    st.success(stage5_command.get("commander_decision", "No command-center decision returned."))
                    h1, h2 = st.columns(2)
                    with h1:
                        st.markdown("**Primary Owner**")
                        st.write(stage5_command.get("primary_owner", "N/A"))
                    with h2:
                        st.markdown("**Escalation Owner**")
                        st.write(stage5_command.get("escalation_owner", "N/A"))
                    action_cards = stage5_command.get("action_cards", [])
                    if action_cards:
                        with st.expander("Action cards / handoff queue", expanded=True):
                            display_card_cols = ["action_id", "owner", "status", "due_minutes", "action", "success_metric"]
                            df_cards = pd.DataFrame(action_cards)
                            st.dataframe(df_cards[[c for c in display_card_cols if c in df_cards.columns]], use_container_width=True)
                    with st.expander("Audit checklist and command briefing", expanded=False):
                        st.markdown("**Audit checklist**")
                        checklist = stage5_command.get("audit_checklist", [])
                        if checklist:
                            st.dataframe(pd.DataFrame(checklist), use_container_width=True)
                        st.markdown("**Agent briefing**")
                        st.json(stage5_command.get("agent_briefing", {}))


                # Packing Priority Panel
                st.markdown("---")
                st.markdown("### \U0001F4E6 Warehouse Packing Instructions")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.metric("Priority Score", f"{priority['priority_score']:.1f} / 100")
                with p_col2:
                    st.metric("Recommended Pack Quantity", f"{priority['recommended_pack_quantity']} units")
                
                if priority["escalation_required"]:
                    st.error(f" **Escalation Triggered!** {priority['recommended_action']}")
                else:
                    st.success(f" **Action Ready:** {priority['recommended_action']}")
                st.caption(priority["reasoning"])
                
                # Committee Panel
                if run_committee_btn:
                    with st.expander("Demand Forecast Agent Insights", expanded=True):
                        st.write(committee["demand_forecast_agent"])
                    with st.expander("Inventory Risk Agent Insights", expanded=True):
                        st.write(committee["inventory_risk_agent"])
                    with st.expander("Packing Priority Agent Insights", expanded=True):
                        st.write(committee["packing_priority_agent"])
                    with st.expander("Clinical Safety Agent Insights", expanded=True):
                        st.write(committee["clinical_safety_agent"])
                    if committee.get("stage3_control_tower_agent") or result.get("stage3_action_plan"):
                        with st.expander("Supplier & Transfer Agent Insights", expanded=True):
                            st.write(committee.get("stage3_control_tower_agent", result.get("stage3_action_plan", {}).get("control_tower_summary", "Stage 3 action plan unavailable.")))
                    if committee.get("stage4_financial_impact_agent") or result.get("stage4_roi_analysis"):
                        with st.expander("Cost / Waste / ROI Agent Insights", expanded=True):
                            st.write(committee.get("stage4_financial_impact_agent", result.get("stage4_roi_analysis", {}).get("control_tower_summary", "Stage 4 financial view unavailable.")))
                    if committee.get("stage5_command_center_agent") or result.get("stage5_command_center"):
                        with st.expander("Command Center Agent Insights", expanded=True):
                            st.write(committee.get("stage5_command_center_agent", result.get("stage5_command_center", {}).get("control_tower_summary", "Command center unavailable.")))
                    with st.expander("Final Recommendation Agent Insights", expanded=True):
                        st.write(committee["final_recommendation_agent"])
                        
                    st.markdown("#### \U0001F4DD Committee Consensus Summary")
                    st.success(committee["committee_summary"])
                    with st.expander("\U0001F4DA DEBUG: RAG Knowledge Received", expanded=True):
                        # Display the RAG knowledge received from the backend (if any)
                        debug_rag_text = committee.get("rag_knowledge", "")
                        if not debug_rag_text or debug_rag_text == "(RAG optional for fast path)":
                            debug_rag_text = "RAG knowledge not loaded in fast mode. Run a RAG-enabled workflow to retrieve supporting knowledge."
                        
                        st.write(f"RAG Payload: {debug_rag_text}")
                        with open("rag_debug_log.txt", "w") as f:
                            f.write(str(debug_rag_text))
                    
                    mode_actual = str(committee.get("actual_agent_mode", "local"))
                    tokens_used = int(committee.get("tokens_used", 0) or 0)
                    # Groq usage is recorded at the moment the HTTP response arrives, so do not
                    # add it again here. This block only syncs the sidebar to the latest known
                    # usage. For local mode, it intentionally stays at zero.
                    if tokens_used > 0:
                        st.session_state["current_tokens"] = tokens_used
                        latest_usage = st.session_state.get("last_llm_usage", {}) or {}
                        if int(latest_usage.get("total_tokens", 0) or 0) <= 0:
                            st.session_state["last_llm_usage"] = {
                                "provider": "Groq",
                                "model": committee.get("remote_model", selected_model or "Groq"),
                                "prompt_tokens": int(committee.get("prompt_tokens", 0) or 0),
                                "completion_tokens": int(committee.get("completion_tokens", 0) or 0),
                                "total_tokens": tokens_used,
                                "tokens_used": tokens_used,
                                "token_usage_source": committee.get("token_usage_source", "committee_result"),
                            }
                    else:
                        st.session_state["current_tokens"] = 0
                    _refresh_sidebar_token_meter_from_state()
                    
                    mode_note = committee.get("mode_note", "")
                    is_remote_llm = ("groq" in mode_actual.lower()) or mode_actual.lower().startswith("remote")
                    if is_remote_llm and tokens_used > 0:
                        st.warning(f"Groq LLM mode used. Tokens reported: {tokens_used:,}")
                        st.caption(
                            f"Prompt: {int(committee.get('prompt_tokens', 0) or 0):,}  |  "
                            f"Completion: {int(committee.get('completion_tokens', 0) or 0):,}  |  "
                            f"Source: {committee.get('token_usage_source', 'groq_usage')}"
                        )
                    elif is_remote_llm:
                        st.warning("Groq was selected, but no token usage was reported because the app fell back to the local response.")
                    else:
                        st.info(f"Local AI mode used. Tokens reported: {tokens_used:,}")
                    st.caption(mode_note)
        except Exception as e:
            st.error(f"Failed to connect to backend server at {MEDPACK_API_BASE_URL}. Ensure server is running. Error: {e}")


# -----------------------------------------------------------------------
# Microsoft Foundry Explanation (Optional)
# Module-level: runs on every rerun so it survives the Foundry button rerun.
# Reads the locked deterministic result from session_state.
# Foundry is read-only: it cannot change any deterministic field.
# -----------------------------------------------------------------------
if st.session_state.get("medpack_deterministic_result") is not None:
    _locked_result = st.session_state["medpack_deterministic_result"]
    st.markdown("---")
    st.markdown("### \U0001F916 Microsoft Foundry Explanation (Optional)")
    st.caption(
        "Foundry explains the locked MedPack result. "
        "It cannot change the decision."
    )

    foundry_btn = st.button(
        "Explain this deterministic result with Foundry",
        key="foundry_explain_btn",
    )
    if foundry_btn:
        foundry_payload = {
            "telemetry": _locked_result.get("telemetry", {}),
            "prediction": _locked_result.get("prediction", {}),
            "shortage_risk": _locked_result.get("shortage_risk", {}),
            "packing_priority": _locked_result.get("packing_priority", {}),
            "memory_state": _locked_result.get("memory_state", {}),
        }
        try:
            with st.spinner("Calling Microsoft Foundry agent..."):
                foundry_res = requests.post(
                    f"{MEDPACK_API_BASE_URL}/api/foundry-explanation",
                    json=foundry_payload,
                    timeout=(5, 60),
                )
            if foundry_res.status_code == 200:
                st.session_state["foundry_result"] = foundry_res.json()
            else:
                st.session_state["foundry_result"] = {
                    "available": False,
                    "reason": f"Backend returned HTTP {foundry_res.status_code}.",
                }
        except Exception as _foundry_exc:
            st.session_state["foundry_result"] = {
                "available": False,
                "reason": "Could not reach the backend for Foundry explanation.",
            }

    if st.session_state.get("foundry_result") is not None:
        _fr = st.session_state["foundry_result"]
        if _fr.get("available"):
            st.markdown(
                """
                <style>
                /* Scope the styles to only affect elements following this specific marker within the same container */
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] {
                    color: #F8FAFC !important;
                }
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] p,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] li,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] span,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] div,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] strong,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] em {
                    color: #F8FAFC !important;
                }
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] h1,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] h2,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] h3,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] h4,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] h5,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stAlert"] h6,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container h4 {
                    color: #60A5FA !important;
                }
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stCaptionContainer"],
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stCaptionContainer"] p,
                div.element-container:has(#foundry-narrative-css-marker) ~ div.element-container div[data-testid="stCaptionContainer"] span {
                    color: #F8FAFC !important;
                }
                </style>
                <div id="foundry-narrative-css-marker"></div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### \U0001F4AC Foundry Narrative")
            st.info(_fr.get("explanation", ""))
            _tokens = int(_fr.get("tokens_used", 0) or 0)
            _model  = _fr.get("model", "azure-foundry-agent")
            if _tokens > 0:
                st.caption(f"Model: {_model} | Tokens used: {_tokens:,}")
            else:
                st.caption(f"Model: {_model}")
        else:
            st.warning(
                "Warning: Foundry explanation unavailable - "
                f"{_fr.get('reason', 'unknown error')}. "
                "All deterministic results above remain unchanged."
            )


st.markdown("---")
with st.expander("🧪 What-If Scenario Simulator — HYPOTHETICAL", expanded=False):
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"]:has(#whatif-simulator-marker) {
            border: 2px solid #0d9488 !important;
            border-radius: 12px;
            overflow: hidden;
        }
        div[data-testid="stExpanderDetails"]:has(#whatif-simulator-marker) {
            background-color: rgba(13, 148, 136, 0.1) !important;
            border-top: 1px solid rgba(13, 148, 136, 0.3) !important;
        }
        </style>
        <div id="whatif-simulator-marker"></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 🧪 What-If Scenario Simulator <span style='font-size: 0.6em; background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 8px; border-radius: 10px; vertical-align: middle; margin-left: 10px;'>SIMULATION</span>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.9em; margin-bottom: 20px;'>Hypothetical analysis — does not modify the active MedPack prediction or committee decision.</p>", unsafe_allow_html=True)
    try:
        stage6_ref_res = api_get("/api/stage6-scenarios")
        if stage6_ref_res.status_code == 200:
            stage6_ref = stage6_ref_res.json()
            scenario_list = stage6_ref.get("scenarios", [])
        else:
            scenario_list = []
        if not scenario_list:
            scenario_list = [
                {"scenario_id": "ED_SURGE_40", "scenario_name": "ED Surge +40%"},
                {"scenario_id": "ICU_RESPIRATORY_SPIKE", "scenario_name": "ICU Respiratory Spike"},
                {"scenario_id": "FLU_SEASON_DEMAND", "scenario_name": "Flu Season Demand"},
                {"scenario_id": "SUPPLIER_DELAY_5D", "scenario_name": "Supplier Delay +5 Days"},
                {"scenario_id": "MASS_CASUALTY_MODE", "scenario_name": "Mass Casualty Mode"},
                {"scenario_id": "WEEKEND_STAFFING_CONSTRAINT", "scenario_name": "Weekend Staffing Constraint"},
                {"scenario_id": "SURGERY_SCHEDULE_SPIKE", "scenario_name": "Surgery Schedule Spike"},
            ]

        scenario_names = [s.get("scenario_name", s.get("scenario_id")) for s in scenario_list]
        scenario_name_to_id = {s.get("scenario_name", s.get("scenario_id")): s.get("scenario_id") for s in scenario_list}
        w1, w2, w3 = st.columns([1.5, 1, 1])
        with w1:
            selected_scenario_name = st.selectbox("Scenario", scenario_names, index=0, key="stage6_scenario_select")
            selected_scenario_id = scenario_name_to_id.get(selected_scenario_name, "ED_SURGE_40")
        with w2:
            compare_all_scenarios = st.checkbox("Compare all scenarios", value=False, key="stage6_compare_all")
        with w3:
            show_stage6_json = st.checkbox("Show full JSON", value=False, key="stage6_show_json")

        with st.expander("Optional custom shock controls", expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                custom_demand_multiplier = st.number_input("Demand multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.05, key="stage6_demand_mult")
            with c2:
                custom_supplier_delay = st.number_input("Add supplier delay days", min_value=0.0, max_value=14.0, value=0.0, step=0.5, key="stage6_delay_add")
            with c3:
                custom_stock_loss = st.number_input("Stock loss units", min_value=0, max_value=500, value=0, step=1, key="stage6_stock_loss")
            with c4:
                custom_acuity_delta = st.number_input("Acuity delta", min_value=0.0, max_value=2.0, value=0.0, step=0.1, key="stage6_acuity_delta")
            with c5:
                custom_pack_multiplier = st.number_input("Pack time multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.05, key="stage6_pack_mult")

        if st.button("↺ Return to Current MedPack Baseline", key="reset_stage6"):
            keys_to_clear = ["stage6_scenario_select", "stage6_compare_all", "stage6_show_json", "stage6_demand_mult", "stage6_delay_add", "stage6_stock_loss", "stage6_acuity_delta", "stage6_pack_mult"]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

        if st.button("Run What-If Simulator", key="run_stage6_whatif"):
            stage6_payload = {
                "telemetry": telemetry,
                "scenario_id": selected_scenario_id,
                "compare_all": bool(compare_all_scenarios),
                "custom_modifiers": {
                    "demand_multiplier": float(custom_demand_multiplier),
                    "supplier_delay_add_days": float(custom_supplier_delay),
                    "stock_reduction_units": int(custom_stock_loss),
                    "acuity_delta": float(custom_acuity_delta),
                    "pack_time_multiplier": float(custom_pack_multiplier),
                },
            }
            with st.spinner("Running scenario through Stage 2-5 control tower..."):
                stage6_res = api_post("/api/stage6-whatif-simulator", stage6_payload)

            if stage6_res.status_code != 200:
                st.error(f"Stage 6 simulator API error: {stage6_res.status_code} - {stage6_res.text}")
            else:
                stage6 = stage6_res.json()
                if stage6.get("benchmark_rows"):
                    st.success(stage6.get("control_tower_summary", "Scenario benchmark complete."))
                    st.markdown("###### 🧪 SIMULATED RESULT")
                    df_stage6 = pd.DataFrame(stage6.get("benchmark_rows", []))
                    show_cols = [
                        "scenario_name", "severity", "scenario_forecast", "demand_delta",
                        "scenario_gap", "gap_delta", "scenario_priority", "scenario_score",
                    ]
                    st.dataframe(df_stage6[[c for c in show_cols if c in df_stage6.columns]], use_container_width=True)
                    top = stage6.get("highest_risk_scenario", {})
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Highest Risk Simulated Scenario", top.get("scenario_name", "N/A"))
                    b2.metric("Simulated Priority", top.get("scenario_priority", "N/A"))
                    b3.metric("Simulated Shortage Gap", top.get("scenario_gap", 0))
                    b4.metric("Demand (Change)", top.get("demand_delta", 0))
                else:
                    s1, s2, s3, s4, s5, s6 = st.columns(6)
                    s1.metric("Current Baseline Demand", f"{stage6.get('baseline_forecast', 0):.1f}")
                    s2.metric("Simulated Demand", f"{stage6.get('scenario_forecast', 0):.1f}", delta=f"{stage6.get('demand_delta', 0):+.1f}")
                    s3.metric("Current Shortage Gap", stage6.get("baseline_true_shortage_gap", 0))
                    s4.metric("Simulated Shortage Gap", stage6.get("scenario_true_shortage_gap", 0), delta=f"{stage6.get('true_shortage_gap_delta', 0):+.1f}")
                    s5.metric("Priority Shift", f"{stage6.get('baseline_priority_code')} -> {stage6.get('scenario_priority_code')}")
                    s6.metric("Net Value (Change)", f"${stage6.get('net_value_delta', 0):,.0f}")
                    st.warning(stage6.get("simulator_summary", "No scenario summary returned."))
                    st.markdown("###### 🧪 SIMULATED RESULT")
                    st.success(stage6.get("recommended_scenario_action", "No scenario action returned."))
                    scenario_stage5 = stage6.get("scenario_stage5_command_center", {})
                    if scenario_stage5:
                        with st.expander("Scenario Stage 5 command cards", expanded=True):
                            cards = scenario_stage5.get("action_cards", [])
                            if cards:
                                df_cards = pd.DataFrame(cards)
                                cols = ["action_id", "owner", "status", "due_minutes", "action", "success_metric"]
                                st.dataframe(df_cards[[c for c in cols if c in df_cards.columns]], use_container_width=True)
                            else:
                                st.write("No action cards generated.")
                    with st.expander("Applied scenario modifiers", expanded=False):
                        mods = stage6.get("applied_modifiers", [])
                        if mods:
                            st.dataframe(pd.DataFrame(mods), use_container_width=True)
                        else:
                            st.write("No modifiers returned.")
                if show_stage6_json:
                    st.json(stage6)
        
        st.markdown("<div style='margin-top: 20px; padding: 12px; border-radius: 8px; background: rgba(0,0,0,0.25); border-left: 4px solid #0d9488;'><span style='color: #cbd5e1; font-size: 0.95em;'>Simulation results are for planning and exploration only. They do not modify the current MedPack prediction, committee decision, or any saved data.</span></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Stage 6 simulator panel unavailable: {e}")

if st.session_state.get("medpack_deterministic_result") is not None:
    st.stop()  # Freeze guard: prevent lower panels from refreshing after Foundry renders.

# Top 5 Supplies Table
st.markdown("---")
st.header("\U0001F4CB Top 5 Supplies to Pack First")
st.caption(
    "This queue uses the live sidebar scenario. Department changes the supply universe; "
    "the sliders, hour/day, and season change risk, pack quantity, priority score, and ranking."
)
try:
    queue_payload = dict(telemetry)
    queue_payload.update({"limit": 5, "max_records": 75})
    try:
        queue_res = requests.post(
            f"{MEDPACK_API_BASE_URL}/api/packing-queue",
            json=queue_payload, timeout=90,
        )
    except requests.exceptions.ConnectionError:
        queue_res = _MockResponse()
    if queue_res.status_code == 200:
        queue_response = queue_res.json()
        queue_data = queue_response.get("queue", queue_response if isinstance(queue_response, list) else [])
        if queue_data:
            df_queue = pd.DataFrame(queue_data)
            display_cols = [
                "item_name",
                "item_category",
                "department",
                "total_stock",
                "usable_stock",
                "unsafe_stock",
                "active_task_reserved_stock",
                "transfer_candidate_stock",
                "predicted_24h_demand",
                "true_shortage_gap",
                "post_transfer_gap",
                "risk_level",
                "recommended_pack_quantity",
                "priority_score",
                "scenario_pressure_score",
                "escalation_required",
            ]
            available_cols = [c for c in display_cols if c in df_queue.columns]
            st.dataframe(df_queue[available_cols], use_container_width=True)
            with st.expander("Why did the Top 5 change?", expanded=False):
                st.write(
                    "The table was recalculated using the current sidebar inputs: "
                    f"department={dept}, patient_volume={patient_volume}, acuity={acuity_level}, "
                    f"procedure_count={procedure_count}, recent_usage_rate={recent_usage_rate}, "
                    f"supplier_delay={supplier_delay}, season={season}, hour={hour}, day={day_of_week}."
                )
                if "pressure_note" in df_queue.columns:
                    st.dataframe(df_queue[["item_name", "pressure_note"]], use_container_width=True)
        else:
            st.write("No inventory records found for this department.")
    else:
        st.error(f"Failed to load live packing queue from API: {queue_res.status_code} - {queue_res.text}")
except Exception as e:
    st.write(f"Cannot load live queue: {e}")

# Stage 1 + Stage 2 + Stage 3 + Stage 4 + Stage 5 Control Tower Panels
st.markdown("---")
st.header("Full Pipeline Control Tower")
st.caption("Traceability -> Usable Stock -> Supply Chain Response -> Financial Impact -> Command Center. Each panel below runs independently against the backend.")

st.markdown("#### Usable-Stock Check")
st.caption("Shows the difference between total stock on paper and what is actually usable for the selected item.")
try:
    stage2_res = api_post("/api/usable-stock-analysis", telemetry)
    if stage2_res.status_code == 200:
        stage2 = stage2_res.json()
        s2a, s2b, s2c, s2d, s2e, s2f = st.columns(6)
        s2a.metric("Forecast", f"{stage2.get('predicted_24h_demand', 0):.1f}")
        s2b.metric("Total", stage2.get("total_stock", 0))
        s2c.metric("Usable", stage2.get("usable_stock", 0))
        s2d.metric("Unsafe", stage2.get("unsafe_stock", 0))
        s2e.metric("Transferable", stage2.get("transfer_candidate_stock", 0))
        s2f.metric("True Gap", f"{stage2.get('true_shortage_gap', 0):.1f}")
        st.info(stage2.get("explanation", ""))
        st.warning(stage2.get("recommended_stage2_action", ""))
        if stage2.get("top_transfer_options"):
            with st.expander("Transfer options", expanded=False):
                st.dataframe(pd.DataFrame(stage2.get("top_transfer_options", [])), use_container_width=True)
    else:
        st.error(f"Stage 2 usable-stock API error: {stage2_res.status_code} - {stage2_res.text}")
except Exception as e:
    st.error(f"Stage 2 usable-stock panel unavailable: {e}")

st.markdown("#### \U0001F69A Supply Chain Action Plan")
st.caption("From shortage detection to fix: transfer internally, order from a vendor, use a substitute, or escalate.")
try:
    stage3_res = api_post("/api/stage3-action-plan", telemetry)
    if stage3_res.status_code == 200:
        stage3 = stage3_res.json()
        transfer = stage3.get("transfer_recommendation", {}) or {}
        supplier = stage3.get("supplier_risk", {}) or {}
        substitute = stage3.get("substitute_options", {}) or {}
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Best Action", stage3.get("best_action", "N/A"))
        g2.metric("True Gap", f"{stage3.get('true_shortage_gap', 0):.1f}")
        g3.metric("Transfer Qty", transfer.get("recommended_transfer_qty", 0))
        g4.metric("Post-Transfer Gap", f"{stage3.get('post_transfer_gap', 0):.1f}")
        g5.metric("Supplier", (supplier.get("recommended_vendor") or {}).get("vendor_name", "N/A"))
        st.success(stage3.get("final_recommendation", "No supply-chain recommendation returned."))
        with st.expander("Details: transfer, suppliers, substitutes", expanded=False):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Transfer**")
                if transfer.get("transfer_options"):
                    st.dataframe(pd.DataFrame(transfer.get("transfer_options", [])), use_container_width=True)
                else:
                    st.write(transfer.get("recommendation", "No transfer data."))
            with d2:
                st.markdown("**Suppliers**")
                if supplier.get("ranked_vendors"):
                    st.dataframe(pd.DataFrame(supplier.get("ranked_vendors", [])), use_container_width=True)
                else:
                    st.write(supplier.get("recommendation", "No supplier data."))
            with d3:
                st.markdown("**Substitutes**")
                if substitute.get("substitute_options"):
                    st.dataframe(pd.DataFrame(substitute.get("substitute_options", [])), use_container_width=True)
                else:
                    st.write(substitute.get("recommendation", "No substitute data."))
    else:
        st.error(f"Stage 3 API error: {stage3_res.status_code} - {stage3_res.text}")
except Exception as e:
    st.error(f"Stage 3 action-plan panel unavailable: {e}")


st.markdown("#### \U0001F4B0 Cost, Waste & ROI Executive View")
st.caption("Business value at a glance: shortage dollars at risk, waste exposure, emergency premium, action cost, and net estimated value.")
try:
    stage4_res = api_post("/api/stage4-roi-analysis", telemetry)
    if stage4_res.status_code == 200:
        stage4 = stage4_res.json()
        q1, q2, q3, q4, q5, q6 = st.columns(6)
        q1.metric("Unit Cost", f"${stage4.get('unit_cost_used', 0):,.2f}")
        q2.metric("Shortage Risk $", f"${stage4.get('shortage_risk_value', 0):,.0f}")
        q3.metric("Waste Risk $", f"${(stage4.get('waste_risk') or {}).get('total_waste_risk_value', 0):,.0f}")
        q4.metric("Action Cost", f"${stage4.get('estimated_action_cost', 0):,.0f}")
        q5.metric("Net Value", f"${stage4.get('net_value_estimate', 0):,.0f}")
        q6.metric("ROI", f"{stage4.get('roi_ratio', 0):,.2f}x")
        st.success(stage4.get("executive_recommendation", "No executive recommendation returned."))
        with st.expander("Full financial breakdown", expanded=False):
            st.json(stage4)
    else:
        st.error(f"Stage 4 API error: {stage4_res.status_code} - {stage4_res.text}")
except Exception as e:
    st.error(f"Stage 4 ROI panel unavailable: {e}")




st.markdown("#### \U0001F9ED Command Center")
st.caption("The final command layer: priority code, owner, response window, action cards, escalation owner, audit checklist, and handoff packet.")
try:
    stage5_res = api_post("/api/stage5-command-center", telemetry)
    if stage5_res.status_code == 200:
        stage5 = stage5_res.json()
        v1, v2, v3, v4, v5, v6 = st.columns(6)
        v1.metric("Priority", stage5.get("priority_code", "P3"))
        v2.metric("Status", stage5.get("command_status", "Monitor"))
        v3.metric("Risk", stage5.get("risk_level", "Low"))
        v4.metric("Window", f"{stage5.get('response_window_minutes', 0)} min")
        v5.metric("Open Actions", stage5.get("open_action_count", 0))
        v6.metric("Net Value", f"${stage5.get('net_value_estimate', 0):,.0f}")
        st.success(stage5.get("commander_decision", "No command decision returned."))
        cards = stage5.get("action_cards", [])
        if cards:
            with st.expander("Action-card queue", expanded=True):
                df_cards = pd.DataFrame(cards)
                cols = ["action_id", "owner", "status", "due_minutes", "action", "success_metric"]
                st.dataframe(df_cards[[c for c in cols if c in df_cards.columns]], use_container_width=True)
        with st.expander("Full command packet", expanded=False):
            st.json(stage5)
    else:
        st.error(f"Stage 5 API error: {stage5_res.status_code} - {stage5_res.text}")
except Exception as e:
    st.error(f"Stage 5 command-center panel unavailable: {e}")




stage_tabs = st.tabs([
    " Traceability",
    " Compliance Alerts",
    " Scan Simulator",
    " Dynamic PAR",
    " Packing Tasks",
    " Stage 3 Data",
    " Stage 4 Finance",
    " Stage 5 Playbook",
    " Stage 6 Scenarios",
])

with stage_tabs[0]:
    try:
        inv_res = api_get("/api/inventory")
        if inv_res.status_code == 200:
            inventory_records = inv_res.json()
            df_inv = pd.DataFrame(inventory_records)
            if not df_inv.empty:
                df_scope = df_inv[df_inv["department"] == dept].copy() if "department" in df_inv.columns else df_inv.copy()
                st.markdown("#### Item Traceability Snapshot")
                trace_cols = [
                    "item_name", "department", "current_stock", "par_level", "max_stock",
                    "lot_number", "udi_code", "barcode", "expiration_date", "recall_status",
                    "location", "vendor_name", "storage_type", "last_scan_event", "last_scan_at",
                ]
                available = [c for c in trace_cols if c in df_scope.columns]
                st.dataframe(df_scope[available].head(25), use_container_width=True)
                st.info("This is the first visible upgrade: every supply row now carries lot, UDI, barcode, expiration, PAR, location, vendor, recall, and scan-state fields.")
            else:
                st.write("No inventory records available.")
        else:
            st.error(f"Inventory API error: {inv_res.status_code}")
    except Exception as e:
        st.error(f"Traceability panel unavailable: {e}")

with stage_tabs[1]:
    try:
        alert_res = api_post("/api/compliance-alerts", {"department": dept, "expiration_window_days": 30})
        if alert_res.status_code == 200:
            alerts = alert_res.json()
            counts = alerts.get("counts", {})
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Below PAR", counts.get("below_par", 0))
            a2.metric("Expiring < 30d", counts.get("expiring_soon", 0))
            a3.metric("Expired", counts.get("expired", 0))
            a4.metric("Recalled", counts.get("recalled", 0))
            a5.metric("Cold Chain", counts.get("temperature_sensitive", 0))
            st.warning(alerts.get("control_tower_summary", "No summary returned."))

            alert_buckets = alerts.get("alerts", {})
            for title, key in [
                ("Below PAR", "below_par"),
                ("Expiring Soon", "expiring_soon"),
                ("Expired", "expired"),
                ("Recalled", "recalled"),
                ("Temperature Sensitive", "temperature_sensitive"),
            ]:
                bucket = alert_buckets.get(key, [])
                with st.expander(f"{title} ({len(bucket)})", expanded=key in {"recalled", "expired", "below_par"} and len(bucket) > 0):
                    if bucket:
                        show_cols = [
                            "item_name", "department", "current_stock", "par_level", "lot_number",
                            "expiration_date", "days_until_expiration", "recall_status", "location", "reason", "severity",
                        ]
                        df_bucket = pd.DataFrame(bucket)
                        st.dataframe(df_bucket[[c for c in show_cols if c in df_bucket.columns]], use_container_width=True)
                    else:
                        st.write("No records in this bucket.")
        else:
            st.error(f"Compliance API error: {alert_res.status_code} - {alert_res.text}")
    except Exception as e:
        st.error(f"Compliance panel unavailable: {e}")

with stage_tabs[2]:
    st.markdown("#### Barcode / UDI Scan Event Simulator")
    st.caption("Use this to simulate real supply movement. RECEIVED/STOCKED adds stock; PICKED/CONSUMED/WASTED/RECALLED removes stock; PACKED/DELIVERED logs workflow without changing stock.")
    scan_col1, scan_col2, scan_col3 = st.columns([1.2, 1, 1])
    with scan_col1:
        scan_barcode = st.text_input("Barcode optional", value="", placeholder="Paste barcode from Traceability tab or leave blank")
        scan_operator = st.text_input("Operator", value="Warehouse Team")
    with scan_col2:
        scan_event_type = st.selectbox("Scan Event", ["RECEIVED", "STOCKED", "PICKED", "PACKED", "DELIVERED", "CONSUMED", "WASTED_EXPIRED", "RECALLED_REMOVED"], index=2)
        scan_qty = st.number_input("Quantity", min_value=1, max_value=500, value=1, step=1)
    with scan_col3:
        scan_note = st.text_area("Note", value="Stage 1 demo scan", height=110)

    if st.button("Submit Scan Event"):
        payload = {
            "barcode": scan_barcode.strip(),
            "item_name": item,
            "department": dept,
            "event_type": scan_event_type,
            "quantity": int(scan_qty),
            "operator": scan_operator,
            "note": scan_note,
        }
        try:
            scan_res = api_post("/api/scan-event", payload)
            if scan_res.status_code == 200:
                st.success("Scan event saved and inventory updated.")
                st.json(scan_res.json().get("event", {}))
            else:
                st.error(f"Scan failed: {scan_res.text}")
        except Exception as e:
            st.error(f"Scan event failed: {e}")

    st.markdown("#### Recent Scan Events")
    try:
        events_res = api_get("/api/scan-events", params={"limit": 15})
        if events_res.status_code == 200:
            events = events_res.json()
            if events:
                df_events = pd.DataFrame(events)
                display = ["timestamp", "event_type", "quantity", "item_name", "department", "barcode", "lot_number", "stock_before", "stock_after", "operator"]
                st.dataframe(df_events[[c for c in display if c in df_events.columns]].sort_values("timestamp", ascending=False), use_container_width=True)
            else:
                st.write("No scan events yet.")
    except Exception as e:
        st.write(f"Recent scans unavailable: {e}")

with stage_tabs[3]:
    st.markdown("#### Dynamic PAR Recommendation")
    st.caption("This converts forecast + supplier delay + clinical criticality into a recommended PAR and max-stock level.")
    try:
        # First get a fresh prediction so the PAR recommendation reflects the current sidebar scenario.
        pred_res = api_post("/api/predict-supply-demand", telemetry)
        predicted = None
        if pred_res.status_code == 200:
            predicted = pred_res.json().get("predicted_24h_demand")
        par_payload = dict(telemetry)
        par_payload["predicted_24h_demand"] = predicted or recent_usage_rate * 24
        par_payload["par_level"] = reorder_point
        par_res = api_post("/api/par-recommendation", par_payload)
        if par_res.status_code == 200:
            par = par_res.json()
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Current PAR", par.get("current_par"))
            p2.metric("Recommended PAR", par.get("recommended_par"), delta=par.get("par_delta"))
            p3.metric("Max Stock", par.get("recommended_max_stock"))
            p4.metric("Safety Buffer", f"{par.get('safety_buffer_pct')}%")
            st.success(par.get("recommended_action"))
            st.caption(par.get("reasoning"))
            st.json(par)
        else:
            st.error(f"PAR API error: {par_res.status_code} - {par_res.text}")
    except Exception as e:
        st.error(f"Dynamic PAR panel unavailable: {e}")

with stage_tabs[4]:
    st.markdown("#### Packing Task Lifecycle")
    st.caption("Turn a recommendation into an operational task: NEW -> ASSIGNED -> PICKING -> PACKED -> DELIVERED or ESCALATED.")
    task_col1, task_col2 = st.columns([1, 1])
    with task_col1:
        task_assignee = st.text_input("Assign To", value="Warehouse Team")
        task_qty = st.number_input("Task Quantity", min_value=1, max_value=500, value=max(1, int(current_stock if current_stock < 10 else 10)), step=1)
    with task_col2:
        task_risk = st.selectbox("Risk Level", ["Low", "Medium", "High", "Critical"], index=2)
        task_note = st.text_area("Task Note", value="Stage 1 packing task created from dashboard.", height=100)

    if st.button("Create Packing Task"):
        payload = {
            "item_name": item,
            "department": dept,
            "quantity": int(task_qty),
            "assigned_to": task_assignee,
            "risk_level": task_risk,
            "priority_score": 75 if task_risk == "High" else 95 if task_risk == "Critical" else 50,
            "recommended_action": f"Pack {int(task_qty)} units of {item} for {dept}.",
            "note": task_note,
        }
        try:
            create_res = api_post("/api/packing-tasks", payload)
            if create_res.status_code == 200:
                st.success("Packing task created.")
                st.json(create_res.json().get("task", {}))
            else:
                st.error(f"Task create failed: {create_res.text}")
        except Exception as e:
            st.error(f"Task create failed: {e}")

    try:
        tasks_res = api_get("/api/packing-tasks", params={"limit": 25})
        if tasks_res.status_code == 200:
            tasks_payload = tasks_res.json()
            tasks = tasks_payload.get("tasks", [])
            if tasks:
                df_tasks = pd.DataFrame(tasks)
                st.markdown("#### Current Tasks")
                task_cols = ["task_id", "status", "item_name", "department", "quantity", "assigned_to", "risk_level", "priority_score", "updated_at", "recommended_action"]
                st.dataframe(df_tasks[[c for c in task_cols if c in df_tasks.columns]], use_container_width=True)
                selected_task = st.selectbox("Update Task", df_tasks["task_id"].tolist())
                new_status = st.selectbox("New Status", tasks_payload.get("valid_statuses", ["NEW", "ASSIGNED", "PICKING", "PACKED", "DELIVERED", "ESCALATED", "CANCELLED"]), index=1)
                if st.button("Update Selected Task Status"):
                    upd_res = requests.patch(f"{MEDPACK_API_BASE_URL}/api/packing-tasks", json={"task_id": selected_task, "status": new_status, "assigned_to": task_assignee, "note": "Updated from Streamlit task lifecycle panel."}, timeout=90)
                    if upd_res.status_code == 200:
                        st.success("Task status updated.")
                        st.json(upd_res.json().get("task", {}))
                    else:
                        st.error(f"Task update failed: {upd_res.text}")
            else:
                st.write("No packing tasks yet.")
    except Exception as e:
        st.write(f"Task list unavailable: {e}")

with stage_tabs[5]:
    st.markdown("#### Stage 3 Reference Data")
    st.caption("This is the seed data Stage 3 uses for supplier delay risk and substitute-item decisions.")
    try:
        ref_res = api_get("/api/stage3-reference-data")
        if ref_res.status_code == 200:
            ref = ref_res.json()
            vendors = (ref.get("vendor_state") or {}).get("vendors", [])
            rules = (ref.get("substitution_rules") or {}).get("rules", {})
            if vendors:
                st.markdown("##### Vendor Intelligence")
                df_vendors = pd.DataFrame(vendors)
                show_vendor_cols = ["vendor_id", "vendor_name", "role", "category_focus", "normal_lead_time_days", "current_delay_days", "reliability_score", "backorder_probability", "unit_cost_multiplier", "emergency_order_available"]
                st.dataframe(df_vendors[[c for c in show_vendor_cols if c in df_vendors.columns]], use_container_width=True)
            st.markdown("##### Substitute Rules")
            rows = []
            for primary, options in rules.items():
                for opt in options:
                    rows.append({"primary_item": primary, **opt})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.write("No substitution rules found.")
        else:
            st.error(f"Stage 3 reference API error: {ref_res.status_code} - {ref_res.text}")
    except Exception as e:
        st.error(f"Stage 3 reference data unavailable: {e}")

with stage_tabs[6]:
    st.markdown("#### Stage 4 Financial Assumptions")
    st.caption("Transparent assumptions used to estimate shortage risk, waste risk, emergency-order premium, labor value, and ROI.")
    try:
        cost_ref_res = api_get("/api/stage4-reference-data")
        if cost_ref_res.status_code == 200:
            st.json(cost_ref_res.json())
        else:
            st.error(f"Stage 4 reference API error: {cost_ref_res.status_code} - {cost_ref_res.text}")
    except Exception as e:
        st.error(f"Stage 4 reference data unavailable: {e}")


with stage_tabs[7]:
    st.markdown("#### Stage 5 Command Playbook")
    st.caption("Editable owners, priority codes, response windows, and escalation cadence used by the final command center.")
    try:
        playbook_res = api_get("/api/stage5-reference-data")
        if playbook_res.status_code == 200:
            st.json(playbook_res.json())
        else:
            st.error(f"Stage 5 reference API error: {playbook_res.status_code} - {playbook_res.text}")
    except Exception as e:
        st.error(f"Stage 5 reference data unavailable: {e}")


with stage_tabs[8]:
    st.markdown("#### Stage 6 Scenario Playbooks")
    st.caption("Editable what-if scenarios used by the surge simulator. These modify telemetry and then run the adjusted case through Stages 2-5.")
    try:
        scenario_res = api_get("/api/stage6-scenarios")
        if scenario_res.status_code == 200:
            scenario_ref = scenario_res.json()
            scenarios = scenario_ref.get("scenarios", [])
            if scenarios:
                df_scen = pd.DataFrame(scenarios)
                cols = ["scenario_id", "scenario_name", "severity", "affected_departments", "affected_categories", "recommended_play"]
                st.dataframe(df_scen[[c for c in cols if c in df_scen.columns]], use_container_width=True)
            with st.expander("Full scenario playbook JSON", expanded=False):
                st.json(scenario_ref)
        else:
            st.error(f"Stage 6 reference API error: {scenario_res.status_code} - {scenario_res.text}")
    except Exception as e:
        st.error(f"Stage 6 reference data unavailable: {e}")



if col2 is not None:
    with col2:
        st.header("Runtime Status")
        try:
            health_res = requests.get(f"{MEDPACK_API_BASE_URL}/health", timeout=90)
            if health_res.status_code == 200:
                health = health_res.json()
                st.json({
                    "api_base_url": MEDPACK_API_BASE_URL,
                    "selected_agent_mode": agent_mode,
                    "remote_llm_enabled_on_backend": health.get("remote_llm_enabled"),
                    "groq_key_present_on_backend": health.get("groq_key_present"),
                    "gemini_key_present_on_backend": health.get("gemini_key_present"),
                    "safe_default": health.get("safe_default"),
                    "stage3_supplier_transfer_intelligence": health.get("stage3_supplier_transfer_intelligence"),
                    "stage4_cost_waste_roi": health.get("stage4_cost_waste_roi"),
                    "stage5_agentic_command_center": health.get("stage5_agentic_command_center"),
                    "stage6_whatif_surge_simulator": health.get("stage6_whatif_surge_simulator"),
                })
            else:
                st.warning("Backend health check did not return OK.")
        except Exception as e:
            st.warning(f"Backend health check unavailable: {e}")

        st.header("Transparent Memory")
        try:
            mem_res = requests.get(f"{MEDPACK_API_BASE_URL}/api/supply-memory", timeout=90)
            if mem_res.status_code == 200:
                memory_state = mem_res.json()
                st.markdown("### Current Rolling State")
                st.json(memory_state)
            else:
                st.write("Error loading memory state.")
        except Exception as e:
            st.write(f"Memory unavailable: {e}")
            
        st.markdown("### Log Feedback & Update Memory")
        feedback_usage = st.number_input("Actual Usage next 24 Hours", min_value=0.0, value=12.0, step=1.0)
        update_mem_btn = st.button("Update Memory with Actual Usage")
        
        if update_mem_btn:
            try:
                update_res = requests.post(f"{MEDPACK_API_BASE_URL}/api/update-supply-memory", json={
                    "item_name": item,
                    "department": dept,
                    "predicted_demand": recent_usage_rate * 1.2,
                    "actual_usage": feedback_usage
                })
                if update_res.status_code == 200:
                    st.success("Memory updated successfully! Page will reflect new rolling averages and delta metrics.")
                    st.json(update_res.json())
                else:
                    st.error("Failed to update memory.")
            except Exception as e:
                st.error(f"Error updating memory: {e}")
                
        st.markdown("### Recent Event Logs (supply_memory_events.jsonl)")
        try:
            events_res = requests.get(f"{MEDPACK_API_BASE_URL}/api/supply-memory-events", timeout=90)
            if events_res.status_code == 200:
                st.json(events_res.json())
        except Exception as e:
            st.write(f"Logs unavailable: {e}")

# Data sources section
st.markdown("---")
st.header("Data Source Transparency")
try:
    sources_res = requests.get(f"{MEDPACK_API_BASE_URL}/api/data-sources", timeout=90)
    if sources_res.status_code == 200:
        data_sources = sources_res.json()
        st.json(data_sources)
        
        # Check Kaggle vs Synthetic
        raw_exists = os.path.exists("database/raw/kaggle_hospital_supply_chain.csv")
        if raw_exists:
            st.success("Primary Kaggle source detected at `database/raw/kaggle_hospital_supply_chain.csv`")
        else:
            st.warning("Kaggle source not found. Relying on synthesized, PHI-free operational simulation records.")
except Exception as e:
    st.write(f"Data sources descriptor unavailable: {e}")

# Architecture Section
st.markdown("---")
st.header("System Architecture & Workflow")
st.markdown("""
1. **Machine Learning Pipeline:** An XGBoost Regressor model predicts `actual_usage_next_24h` based on real-time operational telemetry (volumetrics, recent run rates, acuity).
2. **Deterministic Rules Engine:** Translates predicted demand & current stock levels into structured safety thresholds (`Low`, `Medium`, `High`, `Critical` risk levels).
3. **Logistics Packing Optimizer:** Integrates risk status with clinical significance, department priorities, and packaging speed constraints to compute a prioritized scoring vector.
4. **Agentic Advisory Committee:** A 5-agent advisory committee reviews the metrics. The left-sidebar switch controls local zero-token mode vs optional remote LLM mode.
5. **Stateful Feedback Loop:** Transparently updates standard memory structures and writes JSONL telemetry logs for auditing and dashboard validation.
6. **Stage 1 Control Tower Layer:** Adds item traceability, recall/expiration/PAR alerts, barcode scan events, dynamic PAR recommendations, and packing task lifecycle.
7. **Stage 2 Usable-Stock Integration:** Connects the forecast to traceability by calculating shortage risk against usable stock after expired, recalled, and task-reserved units are removed.
8. **Stage 3 Supplier + Transfer Intelligence:** Recommends internal transfers, backup vendors, substitute items, and escalation when a true shortage remains.
9. **Stage 4 Cost/Waste/ROI Dashboard:** Translates the action plan into shortage dollars at risk, waste exposure, emergency-order cost, labor value, and executive ROI metrics.
""")
