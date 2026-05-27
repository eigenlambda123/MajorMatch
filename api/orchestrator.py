from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from app_logic import recommend_track, suggest_career_context
from course_index import search_courses_with_projection
from api.ollama import chat_completion, resolve_chat_model


ToolCallable = Callable[[List[Dict[str, str]], Optional[str]], Dict[str, Any]]


def _clean_assistant_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    markers = [
        r"<\|start_header_id\|>assistant<\|end_header_id\|>",
        r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*",
        r"<\|start_header_id\|>",
        r"<\|end_header_id\|>",
    ]
    for marker in markers:
        cleaned = re.sub(marker, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"^assistant\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _is_normal_chat_question(user_message: str) -> bool:
    lowered = (user_message or "").strip().lower()
    if not lowered:
        return True

    greeting_patterns = [
        r"^(hi|hello|hey|yo|sup)[\W_]*$",
        r"^(hi|hello|hey|yo|sup)\b",
        r"^what can (majormatch|you) help me with\??$",
        r"^how does (this app|majormatch|it) work\??$",
        r"^what do you do\??$",
        r"what are you\??$",
        r"who are you\??$",
        r"tell me about yourself\??$",
        r"introduce yourself\??$",
        r"^(thanks|thank you|thx|appreciate it|that's all|that is all)[\W_]*$",
        r"\bthank(s| you| you very much|s a lot)?\b",
    ]
    for pattern in greeting_patterns:
        if re.search(pattern, lowered):
            return True
    return False


def _friendly_identity_reply() -> str:
    return (
        "I am MajorMatch, an AI assistant that helps you choose courses and career paths. "
        "I can answer questions directly, and I’ll use tools only when they add value."
    )


def _friendly_gratitude_reply() -> str:
    return "You're welcome. If you want to explore another major or career path, I can help with that too."


@dataclass(frozen=True)
class ToolTrace:
    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]


@dataclass(frozen=True)
class OrchestratorResult:
    reply: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    tool_trace: List[ToolTrace] = field(default_factory=list)
    raw: str = ""


def build_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "predict_track",
                "description": "Predict the most likely career track. Call with `selected_features` (array of feature names) to run the model, or set `open_ui=true` to request the front-end open the interactive prediction UI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_features": {"type": "array", "items": {"type": "string"}, "description": "A list of selected feature names from the model's feature set."},
                        "features": {"type": "array", "items": {"type": "string"}, "description": "Alias for `selected_features`."},
                        "open_ui": {"type": "boolean", "description": "If true, request the front-end to open the predict UI for interactive input."},
                    }
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_career_context",
                "description": "Fetch live job-market context for a recommended career track.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "track": {"type": "string"},
                        "location": {"type": "string"},
                    },
                    "required": ["track"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_semantic_search",
                "description": "Search the course corpus semantically and generate a projection for visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {"type": "string", "description": "The user's course exploration or learning question."},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                        "projection_method": {
                            "type": "string",
                            "enum": ["pca", "umap", "tsne"],
                            "default": "pca",
                        },
                    },
                    "required": ["user_query"],
                },
            },
        },
    ]


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    # Normalize dict-like inputs and try to recover from nested/serialized payloads
    if isinstance(arguments, dict):
        parsed: Dict[str, Any] = dict(arguments)
    elif not arguments:
        return {}
    elif isinstance(arguments, str):
        try:
            loaded = json.loads(arguments)
            parsed = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    else:
        return {}

    # Some models/agents return parameters nested inside arrays or as JSON-encoded strings.
    # Attempt to unwrap common wrappers so callers can find keys like 'track', 'user_query', etc.
    # Example malformed shapes handled:
    # - {'object': "[{'track': 'Healthcare', 'location': 'United States'}]"}
    # - {'parameters': [{'object': "{...}"}]}
    # - {'0': {'track': 'X'}}

    # If single key 'object' contains a JSON string or list, try to decode it.
    if "object" in parsed and isinstance(parsed["object"], str):
        try:
            inner = json.loads(parsed["object"])
            if isinstance(inner, list) and inner:
                # if list of dicts, merge first dict
                if isinstance(inner[0], dict):
                    parsed.update(inner[0])
            elif isinstance(inner, dict):
                parsed.update(inner)
        except json.JSONDecodeError:
            pass

    # If any values are JSON strings, try to decode them too.
    for k, v in list(parsed.items()):
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                decoded = json.loads(v)
                if isinstance(decoded, dict):
                    parsed[k] = decoded
                    parsed.update(decoded)
            except json.JSONDecodeError:
                pass
        if isinstance(v, list) and v and isinstance(v[0], dict):
            # flatten single-item parameter lists
            parsed.update(v[0])

    # Also handle 'parameters' wrappers
    if "parameters" in parsed and isinstance(parsed["parameters"], list) and parsed["parameters"]:
        first = parsed["parameters"][0]
        if isinstance(first, dict):
            parsed.update(first)

    return {k: v for k, v in parsed.items()}


def _extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = call.get("name") or function.get("name")
        if not name:
            continue
        normalized.append(
            {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "name": name,
                "arguments": _parse_tool_arguments(call.get("arguments") or function.get("arguments")),
            }
        )
    return normalized


def _try_extract_tool_call_from_content(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Some models return a tool invocation as raw JSON in the assistant
    content (e.g. {"name":"get_career_context","parameters":[{...}]}).
    Try to parse and normalize that into a tool call so the orchestrator
    executes it rather than surfacing raw JSON to the user.
    """
    content = (message.get("content") or "").strip()
    if not content or not content.lstrip().startswith("{"):
        return []
    try:
        parsed = json.loads(content)
    except Exception:
        return []

    calls: List[Dict[str, Any]] = []
    # Support either a single dict or a list of dicts
    items = [parsed] if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        # arguments may be under 'parameters' or 'arguments'
        arguments = item.get("parameters") or item.get("arguments") or {}
        # If parameters were returned as a list (common pattern: [ { ... } ]),
        # unwrap the first dict so _parse_tool_arguments can normalize it.
        if isinstance(arguments, list) and arguments and isinstance(arguments[0], dict):
            arguments = arguments[0]
        calls.append({"id": None, "type": "function", "name": name, "arguments": _parse_tool_arguments(arguments)})
    return calls


def _execute_tool(name: str, arguments: Dict[str, Any], location: str) -> Dict[str, Any]:
    if name == "predict_track":
        # Disallow implicit inference of user skills for predictions. The
        # prediction tool must be called with explicit selected features (a
        # sequence) or the front-end UI should be opened via `open_ui` so the
        # user can select interests. If the model supplies numeric
        # coding/math/design keys, do not infer a user profile from them —
        # instead, ask the UI to open.
        provided_numeric = any(k in arguments for k in ("coding", "math", "design"))
        if provided_numeric or arguments.get("open_ui") is True:
            return {
                "action": "open_ui",
                "message": "open_predict_ui",
            }

        # If the model provided explicit feature selections, forward them to
        # the recommendation function which calls into the predictor.
        features = arguments.get("features") or arguments.get("selected_features") or []
        recommendation = recommend_track(features)
        return {
            "prediction": recommendation,
        }

    if name == "get_career_context":
        track = str(arguments.get("track") or "").strip()
        effective_location = str(arguments.get("location") or location or "United States")
        context = suggest_career_context(track, location=effective_location)
        return context.to_dict()

    if name == "execute_semantic_search":
        query = str(arguments.get("user_query") or "").strip()
        top_k = max(1, min(int(arguments.get("top_k") or 5), 5))
        projection_method = str(arguments.get("projection_method") or "pca").strip().lower() or "pca"
        if projection_method == "tnse":
            projection_method = "tsne"
        results, selected_projection = search_courses_with_projection(query, top_k=top_k, method=projection_method)

        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "projection": {
                **selected_projection,
            },
        }

    raise ValueError(f"Unsupported tool: {name}")


def _build_final_response_prompt(tool_trace: List[ToolTrace], artifacts: Dict[str, Any]) -> str:
    # Global instruction: short one-line summary + up to 2 concise bullets.
    sections: List[str] = [
        "Write one short, user-facing reply grounded in the tool results below.",
        "Format: 1) single-sentence summary, 2) up to two short bullet points (concise).",
        "Do not include suggested next steps or follow-ups; keep the reply strictly descriptive.",
    ]

    for trace in tool_trace:
        if trace.name == "predict_track":
            prediction = artifacts.get("prediction") or {}
            sections.append(
                "Career recommendation result: "
                f"track={prediction.get('track', 'unknown')}, "
                f"confidence={float(prediction.get('confidence', 0.0)):.2f}. "
                "Tone: concise, prescriptive — state the top recommendation in one sentence and two bullets of supporting signals."
            )
        elif trace.name == "get_career_context":
            career_context = artifacts.get("career_context") or {}
            sections.append(
                "Career market context result: "
                f"available={career_context.get('available', False)}, "
                f"track={career_context.get('track', 'unknown')}, "
                f"job_count={career_context.get('job_count')}, "
                f"salary_min={career_context.get('salary_min')}, "
                f"salary_max={career_context.get('salary_max')}. "
                "Use a strict database-summary style: start with 'Based on the career context tool, the results are:' and then report only the exact job_count, salary_min, salary_max, top_job_titles, and top_companies from the tool output. Do not invent example jobs, do not change the role names, and do not add interpretation beyond a brief note that these are tool results."
            )
        elif trace.name == "execute_semantic_search":
            semantic_search = artifacts.get("semantic_search") or {}
            results = semantic_search.get("results") or []
            top_titles = [item.get("title", "Untitled") for item in results[:5]]
            projection = semantic_search.get("projection") or {}
            sections.append(
                "Semantic search result: "
                f"query='{semantic_search.get('query', '')}', "
                f"top_matches={top_titles}, "
                f"projection_available={projection.get('available', False)}. "
                "Use a strict database-summary style: start with 'Based on what is in my database using the semantic search tool, the results are:' and then list the top matches exactly as returned. Do not add interpretation, ranking claims, or extra recommendations. If a map/projection is available, mention only that the visualization is shown below."
            )

    return "\n\n".join(sections)


def run_orchestrated_assistant(
    user_message: str,
    *,
    location: str = "United States",
    model: Optional[str] = None,
    allow_tool_calls: bool = True,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_steps: int = 4,
    chat_fn: Callable[..., Dict[str, Any]] = chat_completion,
    # Optional streaming hook: a stream-capable chat function and a chunk callback.
    stream_chat_fn: Optional[Callable[..., Iterable[str]]] = None,
    on_stream_chunk: Optional[Callable[[str], None]] = None,
) -> OrchestratorResult:
    resolved_model = resolve_chat_model(model)

    if _is_normal_chat_question(user_message):
        lowered = user_message.lower()
        return OrchestratorResult(
            reply=(
                _friendly_identity_reply()
                if re.search(r"\b(what are you|who are you|tell me about yourself|introduce yourself)\b", lowered)
                else _friendly_gratitude_reply()
                if re.search(r"\b(thanks|thank you|thx|appreciate it|that's all|that is all)\b", lowered)
                else "Hello. I am MajorMatch, an AI assistant that helps with courses and careers."
            ),
            artifacts={},
            tool_trace=[],
            raw="",
        )

    system_prompt = (
        "You are MajorMatch's orchestrator. Use tools only when explicitly required to produce structured results (predictions, market data, or semantic search). "
        "Do not call tools for greetings, introductions, identity questions, or other normal chat. "
        "Do not call tools for gratitude, acknowledgements, or wrap-up messages. "
        "If the user asks what you are or says hello, answer directly with a friendly plain-language introduction. "
        "Call predict_track when the user's skill fit or preferences need a recommendation. "
        "Call get_career_context when the user asks about market demand, salary, or career outlook. "
        "Call execute_semantic_search when the user asks for courses, learning resources, course planning, or a visual map of relevant options. "
        "When no tool is needed, respond naturally and user-friendly without mentioning tools. "
        "Keep replies concise, grounded in tool results when used, and user-facing."
    )

    # Do not include any structured profile values in system prompts. The
    # assistant should not make assumptions about user skills unless a
    # prediction tool is explicitly invoked. Only include non-sensitive
    # contextual info such as preferred location.
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    if location:
        messages.append({"role": "system", "content": f"Preferred location: {location}."})
    if conversation_history:
        messages.extend(conversation_history)
    else:
        messages.append({"role": "user", "content": user_message})

    tool_schemas = build_tool_schemas() if allow_tool_calls else None
    trace: List[ToolTrace] = []
    artifacts: Dict[str, Any] = {}
    last_content = ""
    raw = ""

    for _ in range(max_steps):
        response = chat_fn(messages, model=resolved_model, tools=tool_schemas, options={"temperature": 0.2})
        raw = json.dumps(response)
        message = response.get("message", {}) if isinstance(response, dict) else {}
        last_content = str(message.get("content", "") or "").strip()
        tool_calls = _extract_tool_calls(message)
        # If the model printed a JSON tool invocation into the assistant
        # content instead of using the `tool_calls` structure, try to recover
        # it and treat it as a proper tool call.
        if not tool_calls:
            tool_calls = _try_extract_tool_call_from_content(message)

        if not tool_calls:
            if trace:
                final_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": _build_final_response_prompt(trace, artifacts),
                    },
                ]
                if stream_chat_fn and on_stream_chunk:
                    final_content = ""
                    try:
                        for chunk in stream_chat_fn(
                            final_messages,
                            model=resolved_model,
                            tools=tool_schemas,
                            options={"temperature": 0.2},
                        ):
                            chunk_text = str(chunk or "")
                            if not chunk_text:
                                continue
                            on_stream_chunk(chunk_text)
                            final_content += chunk_text
                        if final_content:
                            raw = "<streamed>"
                            return OrchestratorResult(
                                reply=final_content,
                                artifacts=artifacts,
                                tool_trace=trace,
                                raw=raw,
                            )
                    except Exception:
                        # Fall back to the synchronous final-response path if streaming fails.
                        pass

                final_response = chat_fn(
                    final_messages,
                    model=resolved_model,
                    tools=tool_schemas,
                    options={"temperature": 0.2},
                )
                final_message = final_response.get("message", {}) if isinstance(final_response, dict) else {}
                final_content = _clean_assistant_text(str(final_message.get("content", "") or ""))
                if final_content:
                    raw = json.dumps(final_response)
                    return OrchestratorResult(
                        reply=final_content,
                        artifacts=artifacts,
                        tool_trace=trace,
                        raw=raw,
                    )
            return OrchestratorResult(
                reply=_clean_assistant_text(last_content) or "I could not generate a response.",
                artifacts=artifacts,
                tool_trace=trace,
                raw=raw,
            )

        assistant_message: Dict[str, Any] = {"role": "assistant", "content": last_content, "tool_calls": tool_calls}
        messages.append(assistant_message)

        for call in tool_calls:
            name = str(call.get("name") or "")
            arguments = call.get("arguments") or {}
            result = _execute_tool(name, arguments, location)
            trace.append(ToolTrace(name=name, arguments=arguments, result=result))

            if name == "predict_track":
                artifacts["prediction"] = result.get("prediction")
            elif name == "get_career_context":
                artifacts["career_context"] = result
            elif name == "execute_semantic_search":
                artifacts["semantic_search"] = result

            tool_message: Dict[str, Any] = {
                "role": "tool",
                "content": json.dumps(result),
                "name": name,
            }
            if call.get("id"):
                tool_message["tool_call_id"] = call["id"]
            messages.append(tool_message)

    return OrchestratorResult(
        reply=last_content or "I reached the tool limit before producing a final response.",
        artifacts=artifacts,
        tool_trace=trace,
        raw=raw,
    )
