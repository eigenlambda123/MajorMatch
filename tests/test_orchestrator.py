import api.orchestrator as orchestrator


def test_orchestrated_assistant_executes_tools_in_sequence(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "recommend_track",
        lambda profile: {
            "track": "B.Tech.-Computer Science and Engineering",
            "confidence": 0.84,
            "category": "Software Engineer",
            "source": "model",
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "suggest_career_context",
        lambda track, location="United States": type(
            "FakeContext",
            (),
            {
                "to_dict": lambda self: {
                    "track": track,
                    "location": location,
                    "source": "Adzuna",
                    "available": True,
                    "job_count": 101,
                    "salary_min": 60000,
                    "salary_max": 95000,
                    "salary_currency": "USD",
                    "top_job_titles": ["Junior Developer"],
                    "top_companies": ["Example Co"],
                    "note": None,
                    "query_url": None,
                }
            },
        )(),
    )
    monkeypatch.setattr(
        orchestrator,
        "search_courses_with_projection",
        lambda query, top_k=5, method="pca": (
            [
                {
                    "id": 1,
                    "title": "Web Development with Flask",
                    "description": "Build web apps with Flask",
                    "score": 0.9,
                    "score_normalized": 0.95,
                    "source": "db",
                }
            ],
            {
                "available": True,
                "method": method,
                "courses": [
                    {
                        "id": 1,
                        "title": "Web Development with Flask",
                        "description": "Build web apps with Flask",
                        "x": 1.0,
                        "y": 2.0,
                    }
                ],
                "query_point": {
                    "id": -1,
                    "title": "(query)",
                    "description": query,
                    "x": 0.0,
                    "y": 0.0,
                },
                "methods": {
                    method: {
                        "available": True,
                        "method": method,
                        "courses": [
                            {
                                "id": 1,
                                "title": "Web Development with Flask",
                                "description": "Build web apps with Flask",
                                "x": 1.0,
                                "y": 2.0,
                            }
                        ],
                        "query_point": {
                            "id": -1,
                            "title": "(query)",
                            "description": query,
                            "x": 0.0,
                            "y": 0.0,
                        },
                    }
                },
            },
        ),
    )

    responses = iter(
        [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "predict_track",
                                "arguments": "{\"coding\": 9, \"math\": 4, \"design\": 2}",
                            },
                        }
                    ],
                }
            },
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "get_career_context",
                                "arguments": "{\"track\": \"Software Engineer\", \"location\": \"United States\"}",
                            },
                        }
                    ],
                }
            },
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-3",
                            "type": "function",
                            "function": {
                                "name": "execute_semantic_search",
                                "arguments": "{\"user_query\": \"web development\", \"top_k\": 2, \"projection_method\": \"pca\"}",
                            },
                        }
                    ],
                }
            },
            {"message": {"content": "Here is a compact plan for you.", "tool_calls": []}},
            {"message": {"content": "Software Engineer is the strongest fit. Here are the best job and course signals.", "tool_calls": []}},
        ]
    )

    def fake_chat_fn(messages, model=None, tools=None, options=None):
        return next(responses)

    result = orchestrator.run_orchestrated_assistant(
        "I like coding and want a practical path.",
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert result.reply == "Software Engineer is the strongest fit. Here are the best job and course signals."
    # With structured profile removed, the orchestrator will request the
    # front-end prediction UI when numeric skill keys are present, rather
    # than inferring a recommendation automatically.
    assert result.tool_trace[0].name == "predict_track"
    assert result.tool_trace[0].result.get("action") == "open_ui"
    assert result.artifacts["career_context"]["job_count"] == 101
    assert result.artifacts["semantic_search"]["results"][0]["title"] == "Web Development with Flask"
    assert [trace.name for trace in result.tool_trace] == [
        "predict_track",
        "get_career_context",
        "execute_semantic_search",
    ]


def test_career_context_prompt_requests_exact_tool_fields():
    prompt = orchestrator._build_final_response_prompt(
        [orchestrator.ToolTrace(name="get_career_context", arguments={}, result={})],
        {
            "career_context": {
                "available": True,
                "track": "Software Engineer",
                "job_count": 101,
                "salary_min": 60000,
                "salary_max": 95000,
                "top_job_titles": ["Junior Developer"],
                "top_companies": ["Example Co"],
            }
        },
    )

    assert "Based on the career context tool, the results are:" in prompt
    assert "job_count=101" in prompt
    assert "salary_min=60000" in prompt
    assert "salary_max=95000" in prompt
    assert "top_job_titles" in prompt
    assert "top_companies" in prompt


def test_normal_question_does_not_need_tools():
    calls = []

    def fake_chat_fn(messages, model=None, tools=None, options=None):
        calls.append({"messages": messages, "tools": tools})
        return {"message": {"content": orchestrator._friendly_identity_reply(), "tool_calls": []}}

    result = orchestrator.run_orchestrated_assistant(
        "hello, what are you?",
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert calls, "chat_fn should have been called for a normal greeting"
    assert result.reply == orchestrator._friendly_identity_reply()
    assert result.tool_trace == []


def test_assistant_header_tokens_are_stripped_from_final_reply():
    def fake_chat_fn(messages, model=None, tools=None, options=None):
        return {"message": {"content": "<|start_header_id|>assistant<|end_header_id|> Hi there.", "tool_calls": []}}

    result = orchestrator.run_orchestrated_assistant(
        "recommend a course path",
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert result.reply == "Hi there."


def test_identity_question_skips_tools_entirely():
    calls = []

    def fake_chat_fn(messages, model=None, tools=None, options=None):
        calls.append({"messages": messages, "tools": tools})
        return {"message": {"content": orchestrator._friendly_identity_reply(), "tool_calls": []}}

    result = orchestrator.run_orchestrated_assistant(
        "what are you?",
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert result.tool_trace == []
    assert calls, "chat_fn should be invoked for identity questions"
    assert result.reply == orchestrator._friendly_identity_reply()


def test_app_intro_question_skips_tools_entirely():
    calls = []

    def fake_chat_fn(messages, model=None, tools=None, options=None):
        calls.append({"messages": messages, "tools": tools})
        return {"message": {"content": "Hello. I am MajorMatch, an AI assistant that helps with courses and careers.", "tool_calls": []}}

    result = orchestrator.run_orchestrated_assistant(
        "What can MajorMatch help me with?",
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert result.tool_trace == []
    assert calls, "chat_fn should be invoked for app intro questions"
    assert result.reply == "Hello. I am MajorMatch, an AI assistant that helps with courses and careers."


def test_gratitude_message_skips_tools_entirely():
    calls = []

    def fake_chat_fn(messages, model=None, tools=None, options=None):
        calls.append({"messages": messages, "tools": tools})
        return {"message": {"content": orchestrator._friendly_gratitude_reply(), "tool_calls": []}}

    result = orchestrator.run_orchestrated_assistant(
        "Thank you for the help.",
        model="test-model",
        chat_fn=fake_chat_fn,
    )

    assert result.tool_trace == []
    assert calls, "chat_fn should be invoked for gratitude messages"
    assert result.reply == orchestrator._friendly_gratitude_reply()


def test_streamed_final_reply_preserves_spaces(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "suggest_career_context",
        lambda track, location="United States": type(
            "FakeContext",
            (),
            {
                "to_dict": lambda self: {
                    "track": track,
                    "location": location,
                    "source": "Adzuna",
                    "available": True,
                    "job_count": 33,
                    "salary_min": 92300,
                    "salary_max": 237715,
                    "salary_currency": "USD",
                    "top_job_titles": ["Data Scientist"],
                    "top_companies": ["Google"],
                    "note": None,
                    "query_url": None,
                }
            },
        )(),
    )

    responses = iter(
        [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "get_career_context",
                                "arguments": "{\"track\": \"Data Scientist\", \"location\": \"United States\"}",
                            },
                        }
                    ],
                }
            },
            {"message": {"content": "", "tool_calls": []}},
        ]
    )

    def fake_chat_fn(messages, model=None, tools=None, options=None):
        return next(responses)

    streamed_chunks = []

    def fake_stream_chat_fn(messages, model=None, options=None):
        yield "Based on the career context tool, the results are:"
        yield "\n\nJob count: 33"
        yield "\nSalary range: 92,300 - 237,715"

    result = orchestrator.run_orchestrated_assistant(
        "How many jobs are there for data scientist?",
        model="test-model",
        chat_fn=fake_chat_fn,
        stream_chat_fn=fake_stream_chat_fn,
        on_stream_chunk=streamed_chunks.append,
    )

    assert result.reply == "Based on the career context tool, the results are:\n\nJob count: 33\nSalary range: 92,300 - 237,715"
    assert streamed_chunks == [
        "Based on the career context tool, the results are:",
        "\n\nJob count: 33",
        "\nSalary range: 92,300 - 237,715",
    ]
