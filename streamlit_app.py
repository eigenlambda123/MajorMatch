import plotly.express as px
import streamlit as st

from app_logic import (
    summarize_matches,
    should_open_prediction_tool,
)
from api.ollama import chat_completion, ollama_is_available, resolve_chat_model, chat_completion_stream
from api.orchestrator import OrchestratorResult, run_orchestrated_assistant
from api.predict import get_prediction_feature_columns, predict_track
from api.search import CourseIndexError, rebuild_index


def _render_tool_prediction(prediction):
    st.markdown("### Career Track Prediction")
    st.success(f"Predicted major: {prediction.get('track') or prediction.get('label') or 'Unknown'}")
    if prediction.get("category"):
        st.caption(f"Career family: {prediction['category']}")
    st.progress(min(float(prediction["confidence"]), 1.0))
    st.caption(f"Confidence score: {prediction['confidence']:.2f}")

    top_predictions = prediction.get("top_predictions") or []
    if top_predictions:
        st.markdown("Top 3 predictions")
        for index, item in enumerate(top_predictions[:3], start=1):
            label = item.get("label") or "Unknown"
            confidence = item.get("confidence")
            category = item.get("category")
            details = f"{index}. {label} ({float(confidence):.2f})" if confidence is not None else f"{index}. {label}"
            if category:
                details += f" - {category}"
            st.write(details)


def _render_prediction_tool():
    st.markdown("### Prediction Tool")
    st.caption("Choose the interests that fit you best, then run the model to get the raw major label.")

    try:
        feature_columns = get_prediction_feature_columns()
    except Exception as error:
        st.error(f"Could not load prediction features: {error}")
        return

    selected_interests = st.multiselect(
        "What are you good at?",
        options=feature_columns,
        key="prediction_selected_interests",
        placeholder="Search and select your strongest interests...",
    )

    if st.button("Analyze My Track", key="prediction_analyze_button"):
        if not selected_interests:
            st.warning("Please select at least one interest before running the prediction.")
        else:
            prediction = predict_track(selected_interests)
            if isinstance(prediction, dict):
                _render_tool_prediction(prediction)
            else:
                label, confidence = prediction
                _render_tool_prediction({"label": label, "confidence": confidence, "category": None})


def _render_tool_career_context(career_context):
    st.markdown("### Career Context")
    available = bool(career_context.get("available"))
    if available:
        job_count = career_context.get("job_count")
        salary_min = career_context.get("salary_min")
        salary_max = career_context.get("salary_max")
        salary_currency = career_context.get("salary_currency") or "USD"
        left_metric, right_metric = st.columns(2)
        with left_metric:
            st.metric("Job count", f"{job_count:,}" if job_count is not None else "N/A")
        with right_metric:
            if salary_min is not None and salary_max is not None:
                st.metric(
                    f"Salary range ({salary_currency})",
                    f"{salary_min:,} - {salary_max:,}",
                )
            else:
                st.metric(f"Salary range ({salary_currency})", "N/A")

        if career_context.get("top_job_titles"):
            st.caption("Top job titles: " + ", ".join(career_context.get("top_job_titles", [])))
        if career_context.get("top_companies"):
            st.caption("Top companies: " + ", ".join(career_context.get("top_companies", [])))
    else:
        st.info(career_context.get("note") or "Job-market context is not available yet.")


def _render_semantic_search(search_artifact):
    query = search_artifact.get("query") or ""
    results = search_artifact.get("results") or []
    projection = search_artifact.get("projection") or {}

    st.markdown("### Semantic Search")
    if query:
        st.caption(f"Query: {query}")
    st.info(summarize_matches(results))

    projection_methods = projection.get("methods") or {}
    selected_method = str(projection.get("method") or "pca").lower()
    method_labels = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}

    available_methods = [method for method in ("pca", "umap", "tsne") if projection_methods.get(method, {}).get("available")]
    if projection.get("available") and available_methods:
        default_index = available_methods.index(selected_method) if selected_method in available_methods else 0
        chosen_method = st.selectbox(
            "Projection method",
            available_methods,
            index=default_index,
            format_func=lambda method: method_labels.get(method, method.upper()),
            key="semantic_projection_method",
        )
        chosen_projection = projection_methods.get(chosen_method) or projection
        course_points = chosen_projection.get("courses") or []
        query_point = chosen_projection.get("query_point")
        if course_points and query_point:
            top_title = results[0].get("title") if results else None
            frame = {
                "title": [point["title"] for point in course_points] + [query_point["title"]],
                "description": [point["description"] for point in course_points] + [query_point["description"]],
                "x": [point["x"] for point in course_points] + [query_point["x"]],
                "y": [point["y"] for point in course_points] + [query_point["y"]],
                "kind": [
                    "top_match" if top_title and point["title"] == top_title else "course"
                    for point in course_points
                ]
                + ["query"],
            }

            figure = px.scatter(
                frame,
                x="x",
                y="y",
                color="kind",
                color_discrete_map={"course": "blue", "top_match": "red", "query": "green"},
                hover_name="title",
                hover_data={"description": True, "x": False, "y": False, "kind": True},
                title=f"{method_labels.get(chosen_method, chosen_method.upper())} projection of semantic search results",
            )
            figure.update_traces(marker=dict(size=11, opacity=0.85))
            figure.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Semantic search returned results, but the projection map is not available yet.")
    elif projection.get("available"):
        method = str(projection.get("method") or "pca").upper()
        course_points = projection.get("courses") or []
        query_point = projection.get("query_point")
        if course_points and query_point:
            top_title = results[0].get("title") if results else None
            frame = {
                "title": [point["title"] for point in course_points] + [query_point["title"]],
                "description": [point["description"] for point in course_points] + [query_point["description"]],
                "x": [point["x"] for point in course_points] + [query_point["x"]],
                "y": [point["y"] for point in course_points] + [query_point["y"]],
                "kind": [
                    "top_match" if top_title and point["title"] == top_title else "course"
                    for point in course_points
                ]
                + ["query"],
            }

            figure = px.scatter(
                frame,
                x="x",
                y="y",
                color="kind",
                color_discrete_map={"course": "blue", "top_match": "red", "query": "green"},
                hover_name="title",
                hover_data={"description": True, "x": False, "y": False, "kind": True},
                title=f"{method} projection of semantic search results",
            )
            figure.update_traces(marker=dict(size=11, opacity=0.85))
            figure.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Semantic search returned results, but the projection map is not available yet.")
    elif projection.get("error"):
        st.info(f"Visualization unavailable: {projection['error']}")

    for course in results[:5]:
        with st.container(border=True):
            st.markdown(f"**{course.get('title', 'Untitled')}**")
            st.write(course.get("description", "No description available."))


def _render_hero_section(ollama_ready: bool, resolved_model: str) -> None:
    st.markdown(
        """
        <style>
            .mm-hero {
                padding: 1.4rem 1.5rem;
                border-radius: 1.1rem;
                background: linear-gradient(135deg, rgba(18, 24, 38, 0.92), rgba(15, 23, 42, 0.78));
                border: 1px solid rgba(148, 163, 184, 0.16);
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
            }
            .mm-hero-kicker {
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                color: rgba(148, 163, 184, 0.9);
                margin-bottom: 0.6rem;
            }
            .mm-hero-title {
                font-size: 2.2rem;
                line-height: 1.05;
                font-weight: 800;
                margin-bottom: 0.55rem;
            }
            .mm-hero-copy {
                color: rgba(226, 232, 240, 0.9);
                font-size: 1rem;
                line-height: 1.6;
                max-width: 58ch;
                margin-bottom: 1rem;
            }
            .mm-pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }
            .mm-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.45rem 0.75rem;
                border-radius: 999px;
                background: rgba(30, 41, 59, 0.9);
                color: rgba(241, 245, 249, 0.95);
                font-size: 0.88rem;
                border: 1px solid rgba(148, 163, 184, 0.16);
            }
            .mm-status-card {
                padding: 1.1rem 1.15rem;
                border-radius: 1rem;
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.16);
                height: 100%;
            }
            .mm-status-label {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: rgba(148, 163, 184, 0.9);
                margin-bottom: 0.45rem;
            }
            .mm-status-value {
                font-size: 1.02rem;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }
            .mm-status-note {
                color: rgba(226, 232, 240, 0.85);
                font-size: 0.9rem;
                line-height: 1.5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    hero_left, hero_right = st.columns([1.7, 1])
    with hero_left:
        st.markdown(
            """
            <div class="mm-hero">
                <div class="mm-hero-kicker">AI course and career explorer</div>
                <div class="mm-hero-title">Helping students pick a path with less guesswork.</div>
                <div class="mm-hero-copy">
                    Ask about majors, careers, salaries, or courses. The assistant responds normally when it can,
                    and uses tools only when a grounded answer will help.
                </div>
                <div class="mm-pill-row">
                    <span class="mm-pill">Chat-first</span>
                    <span class="mm-pill">Tool-grounded replies</span>
                    <span class="mm-pill">Semantic course search</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        status_text = "Connected locally" if ollama_ready else "Not connected"
        status_badge = "Ready" if ollama_ready else "Offline"
        st.markdown(
            f"""
            <div class="mm-status-card">
                <div class="mm-status-label">Live status</div>
                <div class="mm-status-value">{status_text}</div>
                <div class="mm-status-note">
                    <strong>Model:</strong> {resolved_model}<br/>
                    <strong>Mode:</strong> {status_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_tool_output(result: OrchestratorResult):
    latest_tool_name = result.artifacts.get("latest_tool_name")
    if latest_tool_name == "predict_track" and result.artifacts.get("prediction"):
        _render_tool_prediction(result.artifacts["prediction"])
        return

    if latest_tool_name == "get_career_context" and result.artifacts.get("career_context"):
        _render_tool_career_context(result.artifacts["career_context"])
        return

    if latest_tool_name == "execute_semantic_search" and result.artifacts.get("semantic_search"):
        _render_semantic_search(result.artifacts["semantic_search"])
        return

    if result.artifacts.get("semantic_search"):
        _render_semantic_search(result.artifacts["semantic_search"])
    elif result.artifacts.get("career_context"):
        _render_tool_career_context(result.artifacts["career_context"])
    elif result.artifacts.get("prediction"):
        _render_tool_prediction(result.artifacts["prediction"])


def main():
    st.set_page_config(page_title="MajorMatch", layout="wide")
    st.title("MajorMatch")
    st.caption("A semantic course and career pathfinder for students, advisors, and freshmen.")

    ollama_ready = ollama_is_available()
    resolved_model = resolve_chat_model()
    _render_hero_section(ollama_ready, resolved_model)

    st.subheader("Chat Assistant")
    st.caption("Ask naturally. Normal questions get a direct friendly reply; tools are used only when needed.")

    allow_tools = st.checkbox(
        "Allow assistant to call tools (recommendations, market data, search)",
        value=True,
        help="When off, the assistant will reply without invoking prediction, career-context, or semantic-search tools.",
    )

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "content": "I can help you decide what course in college or career to take. Ask me a question and I will answer directly unless a tool is useful.",
            }
        ]
    if "assistant_tools_state" not in st.session_state:
        st.session_state["assistant_tools_state"] = {}
    if "assistant_latest_tool_name" not in st.session_state:
        st.session_state["assistant_latest_tool_name"] = ""
    if "prediction_tool_open" not in st.session_state:
        st.session_state["prediction_tool_open"] = False

    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.get("prediction_tool_open"):
        st.divider()
        _render_prediction_tool()

    user_message = st.chat_input("Ask a question about majors, careers, courses, or maps...")
    if user_message:
        st.session_state["assistant_messages"].append({"role": "user", "content": user_message})
        # Clear previous artifacts for the new turn.
        st.session_state["assistant_tools_state"] = {}
        st.session_state["assistant_latest_tool_name"] = ""
        st.session_state["prediction_tool_open"] = False

        if should_open_prediction_tool(user_message) and allow_tools:
            st.session_state["prediction_tool_open"] = True
            st.session_state["assistant_messages"].append(
                {
                    "role": "assistant",
                    "content": "I opened the prediction tool below. Pick the interests that match you best, then run the analyzer.",
                }
            )
            st.rerun()

        try:
            # Prepare a streaming placeholder so assistant output appears chunked
            # in the chat UI as it is generated.
            stream_key = "_streaming_text"
            st.session_state[stream_key] = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()

            def _on_chunk(chunk: str) -> None:
                # Append the incoming chunk and update the placeholder
                current = st.session_state.get(stream_key, "") or ""
                current += str(chunk)
                st.session_state[stream_key] = current
                placeholder.markdown(current)

            result = run_orchestrated_assistant(
                user_message,
                location="United States",
                model=resolved_model,
                allow_tool_calls=allow_tools,
                conversation_history=st.session_state["assistant_messages"],
                stream_chat_fn=chat_completion_stream,
                on_stream_chunk=_on_chunk,
            )

            # Finalize UI state once streaming completes
            # No structured profile to persist anymore.
            st.session_state["assistant_tools_state"] = result.artifacts
            latest_tool_name = ""
            for trace in reversed(result.tool_trace):
                if trace.name == "execute_semantic_search" and result.artifacts.get("semantic_search"):
                    latest_tool_name = "execute_semantic_search"
                    break
                if trace.name == "get_career_context" and result.artifacts.get("career_context"):
                    latest_tool_name = "get_career_context"
                    break
                if trace.name == "predict_track" and result.artifacts.get("prediction"):
                    latest_tool_name = "predict_track"
                    break
            st.session_state["assistant_latest_tool_name"] = latest_tool_name
            open_predict_ui_requested = any(
                trace.name == "predict_track"
                and isinstance(trace.result, dict)
                and (
                    trace.result.get("action") == "open_ui"
                    or trace.result.get("message") == "open_predict_ui"
                )
                for trace in result.tool_trace
            )
            st.session_state["prediction_tool_open"] = open_predict_ui_requested

            assistant_reply = result.reply or "I am MajorMatch, an AI assistant that can help you decide what course in college or career to take."
            st.session_state["assistant_messages"].append({"role": "assistant", "content": assistant_reply})
            # Clear streaming buffer
            st.session_state.pop(stream_key, None)
        except Exception as error:
            st.session_state["assistant_tools_state"] = {}
            st.session_state["assistant_latest_tool_name"] = ""
            st.session_state["prediction_tool_open"] = False
            st.session_state["assistant_messages"].append(
                {
                    "role": "assistant",
                    "content": f"I hit an issue while processing that request: {error}",
                }
            )
        st.rerun()

    tools_state = st.session_state.get("assistant_tools_state", {})
    if tools_state:
        st.divider()
        st.caption("Latest tool output")
        result = OrchestratorResult(
            reply="",
            artifacts={
                **tools_state,
                "latest_tool_name": st.session_state.get("assistant_latest_tool_name", ""),
            },
            tool_trace=[],
            raw="",
        )
        _render_tool_output(result)


if __name__ == "__main__":
    main()
