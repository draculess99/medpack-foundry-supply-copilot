import os
import json
from typing import Dict, Any, Generator

import requests

try:
    from backend.rag_manager import rag_manager
except ImportError:
    rag_manager = None

try:
    from backend.foundry_client import invoke_foundry_agent, is_foundry_configured
except ImportError:
    # If the module isn't loaded properly or dependencies are missing
    def is_foundry_configured(): return False
    def invoke_foundry_agent(*args, **kwargs): raise NotImplementedError()


AGENT_NAMES = [
    "Demand Forecast Agent",
    "Inventory Risk Agent",
    "Packing Priority Agent",
    "Clinical Safety Agent",
    "Final Recommendation Agent",
    "Committee Summarizer",
]


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _remote_timeout() -> int:
    """Short timeout so the Streamlit button never appears frozen for minutes."""
    try:
        return max(3, min(int(os.environ.get("REMOTE_LLM_TIMEOUT_SECONDS", "8")), 30))
    except Exception:
        return 8


def _normalise_groq_model(model_name: str) -> str:
    """Avoid known decommissioned Groq defaults from older project versions."""
    model_name = (model_name or "").strip()
    deprecated = {"llama3-8b-8192", "llama3-70b-8192"}
    if not model_name or model_name in deprecated:
        return "openai/gpt-oss-120b"
    return model_name


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _build_local_committee_texts(
    telemetry: Dict[str, Any],
    prediction_result: Dict[str, Any],
    shortage_result: Dict[str, Any],
    priority_result: Dict[str, Any],
    memory_state: Dict[str, Any],
) -> Dict[str, str]:
    """Build the deterministic, zero-token committee response."""
    item_name = telemetry.get("item_name", "Supply Item")
    department = telemetry.get("department", "Hospital Department")
    rag_context = ""
    if rag_manager:
        rag_context = rag_manager.query_rag(f"{item_name} in {department}", n_results=2)


    usable_stock = shortage_result.get(
        "usable_stock",
        shortage_result.get("current_stock", telemetry.get("current_stock", 0)),
    )
    total_stock = shortage_result.get("total_stock", telemetry.get("current_stock", usable_stock))
    unsafe_stock = shortage_result.get("unsafe_stock", 0)
    reserved_stock = shortage_result.get("active_task_reserved_stock", 0)
    transfer_candidate_stock = shortage_result.get("transfer_candidate_stock", 0)

    clinical_criticality = _safe_int(telemetry.get("clinical_criticality", 3), 3)
    acuity_level = telemetry.get("acuity_level", "Medium")
    predicted_demand = _safe_float(prediction_result.get("predicted_24h_demand", 0.0))

    shortage_gap = _safe_float(shortage_result.get("shortage_gap", shortage_result.get("true_shortage_gap", 0.0)))
    coverage_ratio = _safe_float(shortage_result.get("coverage_ratio", 1.0), 1.0)
    risk_level = shortage_result.get("risk_level", "Low")

    priority_score = _safe_float(priority_result.get("priority_score", 0.0))
    recommended_pack_quantity = _safe_int(priority_result.get("recommended_pack_quantity", 0))
    recommended_action = priority_result.get("recommended_action", "Review stock and replenish if needed.")

    trend = memory_state.get("trend_direction", "stable") if isinstance(memory_state, dict) else "stable"
    rolling_avg = _safe_float(memory_state.get("rolling_usage_avg", 0.0) if isinstance(memory_state, dict) else 0.0)

    demand_agent_text = (
        f"[Demand Forecast Agent] I analyzed the 24-hour demand projection for {item_name} in {department}. "
        f"The ML model predicts {predicted_demand:.1f} units of demand. "
        f"Historical memory shows a '{trend}' trend with a rolling usage average of {rolling_avg:.1f} units. "
        f"Patient volume ({telemetry.get('patient_volume', 10)}) and acuity ({acuity_level}) support this demand estimate."
    )

    hours_remaining = _safe_float(usable_stock) / max(predicted_demand / 24.0, 0.1)
    risk_agent_text = (
        f"[Inventory Risk Agent] Stage 2 uses usable stock rather than raw total stock. "
        f"{item_name} in {department} has {usable_stock} usable units out of {total_stock} total units. "
        f"Unsafe/unavailable stock is {unsafe_stock}, active task-reserved stock is {reserved_stock}, "
        f"and possible transfer stock is {transfer_candidate_stock}. "
        f"Coverage is {coverage_ratio * 100:.1f}%, giving a true shortage gap of {shortage_gap:.1f} units. "
        f"At current projected demand, usable stock covers about {hours_remaining:.1f} hours. Risk level: {risk_level}."
    )

    pack_time = _safe_float(telemetry.get("pack_time_minutes", 5.0), 5.0)
    priority_agent_text = (
        f"[Packing Priority Agent] The packing optimizer calculated a priority score of {priority_score:.1f}. "
        f"Recommended pack quantity is {recommended_pack_quantity} units, requiring about "
        f"{recommended_pack_quantity * pack_time:.1f} warehouse minutes. "
        f"Action: {recommended_action}"
    )

    clinical_impacts = {
        1: "Low clinical impact. Minor delays in non-urgent supply access.",
        2: "Moderate clinical impact. Potential delays in diagnostic or treatment steps.",
        3: "High clinical impact. Nurses may lose care time searching for supplies.",
        4: "Critical clinical impact. Direct patient-safety concern if supplies support emergency, surgery, or life-support workflows.",
    }
    impact_text = clinical_impacts.get(clinical_criticality, "Standard operational workflow impact.")
    clinical_agent_text = (
        f"[Clinical Safety Agent] Clinical criticality is {clinical_criticality}/4 for {item_name}. "
        f"A shortage in {department} with acuity '{acuity_level}' creates this impact: {impact_text} "
        f"Preventive packing keeps staff at the bedside instead of hunting for supplies."
    )

    final_agent_text = (
        f"[Final Recommendation Agent] Control Tower Action Plan for {item_name} in {department}: "
        f"1. {recommended_action} "
        f"2. Pack or stage {recommended_pack_quantity} units now if the gap remains positive. "
        f"3. Do not count expired, recalled, or task-reserved stock as available. "
        f"4. Consider internal transfer before waiting on a supplier delay of {telemetry.get('supplier_delay_days', 2.0)} days. "
        f"5. Log the decision in the packing workflow."
    )

    summary = (
        f"MedPack AI Committee consensus: {item_name} in {department} is at {risk_level} shortage risk using Stage 2 usable-stock logic. "
        f"True gap is {shortage_gap:.1f} units; recommended pack quantity is {recommended_pack_quantity} units."
    )

    return {
        "demand_forecast_agent": demand_agent_text,
        "inventory_risk_agent": risk_agent_text,
        "packing_priority_agent": priority_agent_text,
        "clinical_safety_agent": clinical_agent_text,
        "final_recommendation_agent": final_agent_text,
        "committee_summary": summary,
        "rag_knowledge": rag_context,
    }


def _try_remote_groq_summary(
    telemetry: Dict[str, Any],
    prediction_result: Dict[str, Any],
    shortage_result: Dict[str, Any],
    priority_result: Dict[str, Any],
    memory_state: Dict[str, Any],
) -> Dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    model_name = _normalise_groq_model(telemetry.get("selected_model") or os.environ.get("GROQ_MODEL", ""))
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout_seconds = _remote_timeout()

    def _call_groq(sys_prompt: str, user_prompt: str):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        res = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        res.raise_for_status()
        data = res.json()
        txt = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not txt:
            raise RuntimeError("Remote LLM returned an empty response.")
        usage = data.get("usage", {}) or {}
        return txt, int(usage.get("total_tokens", 0) or 0)

    total_tokens = 0
    parsed_json: Dict[str, str] = {}
    base_data = json.dumps(
        {
            "telemetry": telemetry,
            "prediction": prediction_result,
            "shortage_risk": shortage_result,
            "packing_priority": priority_result,
            "memory_state": memory_state,
            "rag_knowledge": rag_manager.query_rag(f"{telemetry.get('item_name')} in {telemetry.get('department')}") if rag_manager else ""
        },
        indent=2,
    )

    prompts = [
        ("demand_forecast_agent", "You are the Demand Forecast Agent. Analyze demand in under 3 sentences.", f"Context:\n{base_data}"),
        ("inventory_risk_agent", "You are the Inventory Risk Agent. Assess shortage risk using usable stock in under 3 sentences.", None),
        ("packing_priority_agent", "You are the Packing Priority Agent. Recommend packing action in under 3 sentences.", None),
        ("clinical_safety_agent", "You are the Clinical Safety Agent. Evaluate patient-safety impact in under 3 sentences. You MUST explicitly quote and apply any RAG policies provided in the context.", None),
        ("final_recommendation_agent", "You are the Final Recommendation Agent. Issue a concise action plan in under 4 sentences. You MUST explicitly quote and apply any RAG knowledge/policies provided, even as a backup plan.", None),
        ("committee_summary", "You are the Committee Summarizer. Provide a concise 2-sentence summary. You MUST incorporate and quote the RAG policies in your summary.", None),
    ]

    history = ""
    for key, sys_prompt, user_prompt in prompts:
        user_prompt = user_prompt or f"Context:\n{base_data}\n\nPrior committee discussion:\n{history}"
        txt, tokens = _call_groq(sys_prompt, user_prompt)
        total_tokens += tokens
        parsed_json[key] = txt
        history += f"\n{key}: {txt}"

    return {"agent_texts": parsed_json, "tokens_used": total_tokens, "model": model_name}


def _try_remote_gemini_summary(
    telemetry: Dict[str, Any],
    prediction_result: Dict[str, Any],
    shortage_result: Dict[str, Any],
    priority_result: Dict[str, Any],
    memory_state: Dict[str, Any],
) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model_name = telemetry.get("selected_model") or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    api_version = "v1alpha" if "exp" in model_name or "2.0" in model_name else "v1beta"
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"
    prompt = {
        "role": "MedPack AI remote LLM committee summarizer",
        "instruction": "Create a concise hospital supply-chain decision summary. Do not invent facts. Use the provided ML prediction, shortage risk, packing priority, and memory state. You MUST explicitly quote and incorporate the RAG knowledge/policies provided in the context.",
        "telemetry": telemetry,
        "prediction": prediction_result,
        "shortage_risk": shortage_result,
        "packing_priority": priority_result,
        "memory_state": memory_state,
        "rag_knowledge": rag_manager.query_rag(f"{telemetry.get('item_name')} in {telemetry.get('department')}") if rag_manager else ""
    }
    payload = {
        "contents": [{"parts": [{"text": json.dumps(prompt, indent=2)}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }

    response = requests.post(url, json=payload, timeout=_remote_timeout())
    response.raise_for_status()
    data = response.json()
    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
    if not text:
        raise RuntimeError("Remote LLM returned an empty response.")
    usage = data.get("usageMetadata", {}) or {}
    token_count = usage.get("totalTokenCount") or usage.get("total_tokens") or 0
    return {"text": text, "tokens_used": token_count, "model": model_name}


def _decorate_result(
    local_texts: Dict[str, str],
    requested_mode: str,
    actual_mode: str,
    remote_enabled: bool,
    provider: str,
    has_key: bool,
    remote_model: str | None,
    tokens_used: int,
    fallback_mode: bool,
    mode_note: str,
) -> Dict[str, Any]:
    return {
        **local_texts,
        "requested_agent_mode": requested_mode,
        "actual_agent_mode": actual_mode,
        "remote_llm_enabled": remote_enabled,
        "remote_llm_key_present": has_key,
        "remote_model": remote_model,
        "tokens_used": int(tokens_used or 0),
        "fallback_mode": bool(fallback_mode),
        "mode_note": mode_note,
    }


def run_committee(
    telemetry,
    prediction_result,
    shortage_result,
    priority_result,
    memory_state,
    agent_mode="local",
):
    """Execute the MedPack committee with safe local fallback."""
    requested_mode = str(agent_mode or telemetry.get("agent_mode", "local")).strip().lower()
    if requested_mode not in {"local", "remote"}:
        requested_mode = "local"

    local_texts = _build_local_committee_texts(telemetry, prediction_result, shortage_result, priority_result, memory_state)
    remote_enabled = _truthy(os.environ.get("USE_LLM_AGENTS", "false")) or is_foundry_configured()
    provider = telemetry.get("selected_provider") or "Groq"
    has_groq_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    
    if provider == "Google Gemini":
        has_key = has_gemini_key
    elif provider == "Microsoft Foundry":
        has_key = is_foundry_configured()
    else:
        has_key = has_groq_key

    mode_note = "Local deterministic mode selected. No remote LLM call was made. Zero LLM tokens used."
    actual_mode = "local"
    fallback_mode = True
    remote_model = None
    tokens_used = 0

    if requested_mode == "remote":
        if not remote_enabled:
            mode_note = "Remote mode was selected, but USE_LLM_AGENTS is not true. MedPack used local zero-token agents."
        elif not has_key:
            mode_note = f"Remote mode was selected for {provider}, but the API key is missing. MedPack used local zero-token agents."
        else:
            try:
                if provider == "Google Gemini":
                    remote = _try_remote_gemini_summary(telemetry, prediction_result, shortage_result, priority_result, memory_state)
                    local_texts["final_recommendation_agent"] = "[Remote LLM Committee Summary] " + remote["text"]
                    local_texts["committee_summary"] = remote["text"]
                elif provider == "Microsoft Foundry":
                    remote = invoke_foundry_agent(telemetry, prediction_result, shortage_result, priority_result, memory_state)
                    local_texts["final_recommendation_agent"] = "[Foundry Agent Summary] " + remote["text"]
                    local_texts["committee_summary"] = remote["text"]
                else:
                    remote = _try_remote_groq_summary(telemetry, prediction_result, shortage_result, priority_result, memory_state)
                    agent_texts = remote.get("agent_texts", {})
                    for key in [
                        "demand_forecast_agent",
                        "inventory_risk_agent",
                        "packing_priority_agent",
                        "clinical_safety_agent",
                        "final_recommendation_agent",
                    ]:
                        if agent_texts.get(key):
                            local_texts[key] = "[Remote LLM] " + agent_texts[key]
                    if agent_texts.get("committee_summary"):
                        local_texts["committee_summary"] = agent_texts["committee_summary"]

                actual_mode = "remote"
                fallback_mode = False
                tokens_used = int(remote.get("tokens_used") or 0)
                remote_model = remote.get("model")
                mode_note = f"Remote LLM Mode used {provider} ({remote_model}). Token usage reported: {tokens_used}."
            except Exception as exc:
                mode_note = (
                    f"Remote LLM Mode ({provider}) did not finish cleanly, so MedPack fell back to the local zero-token committee. "
                    f"Reason: {exc}"
                )

    return _decorate_result(
        local_texts,
        requested_mode,
        actual_mode,
        remote_enabled,
        provider,
        has_key,
        remote_model,
        tokens_used,
        fallback_mode,
        mode_note,
    )


def _yield_local_stream(local_texts: Dict[str, str], requested_mode: str, mode_note: str) -> Generator[Dict[str, Any], None, None]:
    key_by_agent = {
        "Demand Forecast Agent": "demand_forecast_agent",
        "Inventory Risk Agent": "inventory_risk_agent",
        "Packing Priority Agent": "packing_priority_agent",
        "Clinical Safety Agent": "clinical_safety_agent",
        "Final Recommendation Agent": "final_recommendation_agent",
        "Committee Summarizer": "committee_summary",
    }
    for agent in AGENT_NAMES:
        yield {"event": "agent_start", "agent_name": agent}
        yield {"event": "agent_done", "agent_name": agent}

    local_texts["requested_agent_mode"] = requested_mode
    local_texts["actual_agent_mode"] = "local"
    local_texts["remote_llm_enabled"] = _truthy(os.environ.get("USE_LLM_AGENTS", "false"))
    local_texts["tokens_used"] = 0
    local_texts["fallback_mode"] = requested_mode == "remote"
    local_texts["mode_note"] = mode_note
    yield {"event": "complete", "data": local_texts}


def run_committee_stream(
    telemetry,
    prediction_result,
    shortage_result,
    priority_result,
    memory_state,
    agent_mode="local",
):
    """Stream committee status while guaranteeing a complete local fallback."""
    requested_mode = str(agent_mode or telemetry.get("agent_mode", "local")).strip().lower()
    if requested_mode not in {"local", "remote"}:
        requested_mode = "local"

    provider = telemetry.get("selected_provider") or "Groq"
    remote_enabled = _truthy(os.environ.get("USE_LLM_AGENTS", "false")) or is_foundry_configured()
    
    if provider == "Google Gemini":
        has_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    elif provider == "Microsoft Foundry":
        has_key = is_foundry_configured()
    else:
        has_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
        
    local_texts = _build_local_committee_texts(telemetry, prediction_result, shortage_result, priority_result, memory_state)

    if requested_mode != "remote":
        yield {"event": "start", "message": "Running local zero-token committee."}
        yield from _yield_local_stream(local_texts, requested_mode, "Local Mode used. Zero tokens.")
        return

    if not remote_enabled or not has_key:
        reason = "USE_LLM_AGENTS is not true" if not remote_enabled else f"{provider} API key or configuration is missing"
        yield {"event": "warning", "message": f"Remote mode unavailable ({reason}). Falling back to local zero-token committee."}
        yield from _yield_local_stream(local_texts, requested_mode, f"Remote mode unavailable ({reason}). Local zero-token committee used.")
        return

    try:
        if provider == "Google Gemini":
            yield {"event": "agent_start", "agent_name": "Committee Summarizer"}
            remote = _try_remote_gemini_summary(telemetry, prediction_result, shortage_result, priority_result, memory_state)
            yield {"event": "agent_done", "agent_name": "Committee Summarizer"}
            local_texts["final_recommendation_agent"] = "[Remote LLM] " + remote["text"]
            local_texts["committee_summary"] = remote["text"]
            local_texts.update({
                "requested_agent_mode": requested_mode,
                "actual_agent_mode": "remote",
                "remote_llm_enabled": True,
                "remote_llm_key_present": True,
                "remote_model": remote.get("model"),
                "tokens_used": int(remote.get("tokens_used") or 0),
                "fallback_mode": False,
                "mode_note": f"Remote LLM Mode used Gemini ({remote.get('model')}).",
            })
            yield {"event": "complete", "data": local_texts}
            return
            
        if provider == "Microsoft Foundry":
            yield {"event": "agent_start", "agent_name": "Committee Summarizer"}
            remote = invoke_foundry_agent(telemetry, prediction_result, shortage_result, priority_result, memory_state)
            yield {"event": "agent_done", "agent_name": "Committee Summarizer"}
            local_texts["final_recommendation_agent"] = "[Foundry Agent Summary] " + remote["text"]
            local_texts["committee_summary"] = remote["text"]
            local_texts.update({
                "requested_agent_mode": requested_mode,
                "actual_agent_mode": "remote",
                "remote_llm_enabled": True,
                "remote_llm_key_present": True,
                "remote_model": remote.get("model"),
                "tokens_used": int(remote.get("tokens_used") or 0),
                "fallback_mode": False,
                "mode_note": f"Remote LLM Mode used Microsoft Foundry ({remote.get('model')}).",
            })
            yield {"event": "complete", "data": local_texts}
            return

        # Groq streaming-style sequence. Each network request has a short timeout.
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        model_name = _normalise_groq_model(telemetry.get("selected_model") or os.environ.get("GROQ_MODEL", ""))
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        timeout_seconds = _remote_timeout()
        base_data = json.dumps(
            {
                "telemetry": telemetry, 
                "prediction": prediction_result, 
                "shortage_risk": shortage_result, 
                "packing_priority": priority_result, 
                "memory_state": memory_state,
                "rag_knowledge": rag_manager.query_rag(f"{telemetry.get('item_name')} in {telemetry.get('department')}") if rag_manager else ""
            },
            indent=2,
        )

        def _call_groq(sys_prompt: str, user_prompt: str):
            payload = {
                "model": model_name,
                "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": 0.2,
                "max_tokens": 512,
            }
            res = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            res.raise_for_status()
            data = res.json()
            txt = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not txt:
                raise RuntimeError("Remote LLM returned an empty response.")
            return txt, int((data.get("usage", {}) or {}).get("total_tokens", 0) or 0)

        steps = [
            ("Demand Forecast Agent", "demand_forecast_agent", "You are the Demand Forecast Agent. Analyze demand in under 3 sentences."),
            ("Inventory Risk Agent", "inventory_risk_agent", "You are the Inventory Risk Agent. Assess shortage risk using usable stock in under 3 sentences."),
            ("Packing Priority Agent", "packing_priority_agent", "You are the Packing Priority Agent. Recommend packing action in under 3 sentences."),
            ("Clinical Safety Agent", "clinical_safety_agent", "You are the Clinical Safety Agent. Evaluate patient-safety impact in under 3 sentences."),
            ("Final Recommendation Agent", "final_recommendation_agent", "You are the Final Recommendation Agent. Issue a concise action plan in under 4 sentences."),
            ("Committee Summarizer", "committee_summary", "You are the Committee Summarizer. Provide a concise 2-sentence summary."),
        ]
        total_tokens = 0
        history = ""
        for agent_name, key, prompt in steps:
            yield {"event": "agent_start", "agent_name": agent_name}
            txt, tokens = _call_groq(prompt, f"Context:\n{base_data}\n\nPrior committee discussion:\n{history}")
            total_tokens += tokens
            local_texts[key] = "[Remote LLM] " + txt if key != "committee_summary" else txt
            history += f"\n{agent_name}: {txt}"
            yield {"event": "agent_done", "agent_name": agent_name}

        local_texts.update({
            "requested_agent_mode": requested_mode,
            "actual_agent_mode": "remote",
            "remote_llm_enabled": True,
            "remote_llm_key_present": True,
            "remote_model": model_name,
            "tokens_used": total_tokens,
            "fallback_mode": False,
            "mode_note": f"Remote LLM Mode used Groq ({model_name}). Token usage reported: {total_tokens}.",
        })
        yield {"event": "complete", "data": local_texts}

    except Exception as exc:
        yield {"event": "warning", "message": f"Remote committee did not finish cleanly: {exc}. Falling back to local zero-token committee."}
        yield from _yield_local_stream(
            local_texts,
            requested_mode,
            f"Remote committee failed or timed out. Local zero-token committee used instead. Reason: {exc}",
        )
