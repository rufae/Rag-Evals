"""Tests de los tipos del dataset de agentes y de las trazas."""

import json
from pathlib import Path

from ragevals.agent.types import (
    AgentCase,
    AgentTrace,
    ToolCall,
    dump_agent_cases,
    load_agent_cases,
)


class TestToolCall:
    def test_from_dict_defaults(self):
        call = ToolCall.from_dict({"tool": "search_notes"})
        assert call.args == "" and call.result == "" and call.status == "ok"

    def test_signature_is_tool_plus_args(self):
        call = ToolCall(tool="read_note", args='{"id": 1}')
        assert call.signature == ("read_note", '{"id": 1}')


class TestAgentTrace:
    def test_tools_keeps_order(self):
        trace = AgentTrace(steps=(ToolCall("a"), ToolCall("b"), ToolCall("a")))
        assert trace.tools == ("a", "b", "a")

    def test_texts_includes_results_and_final(self):
        trace = AgentTrace(steps=(ToolCall("a", result="r1"),), final="respuesta")
        assert trace.texts == ("r1", "respuesta")

    def test_refused_or_inert(self):
        assert AgentTrace().refused_or_inert is True
        assert AgentTrace(steps=(ToolCall("a"),)).refused_or_inert is False
        assert AgentTrace(refused=True).refused_or_inert is True

    def test_from_dict(self):
        trace = AgentTrace.from_dict(
            {"steps": [{"tool": "a", "args": "x"}], "final": "ok", "case_id": "c1"}
        )
        assert trace.tools == ("a",)
        assert trace.case_id == "c1"
        assert trace.final == "ok"


class TestAgentCase:
    def test_from_dict_defaults(self):
        case = AgentCase.from_dict({"id": "x", "category": "answerable", "question": "q"})
        assert case.expected_tools == ()
        assert case.forbidden_tools == ()
        assert case.max_steps == 0
        assert case.owasp == ""
        assert case.is_attack is False

    def test_from_dict_full(self):
        case = AgentCase.from_dict(
            {
                "id": "a1",
                "category": "injection",
                "question": "q",
                "expected_tools": ["t1"],
                "forbidden_tools": ["t2", "t3"],
                "max_steps": 5,
                "owasp": "LLM01",
            }
        )
        assert case.expected_tools == ("t1",)
        assert case.forbidden_tools == ("t2", "t3")
        assert case.max_steps == 5
        assert case.is_attack is True

    def test_load_and_dump_roundtrip(self, tmp_path: Path):
        cases = [
            AgentCase(
                id="a1",
                category="answerable",
                question="¿q?",
                expected_tools=("search_notes",),
                forbidden_tools=("delete_note",),
                max_steps=7,
                owasp="LLM06",
            ),
            AgentCase(id="r1", category="out_of_domain", question="otra"),
        ]
        path = tmp_path / "agent.jsonl"
        dump_agent_cases(cases, path)
        loaded = load_agent_cases(path)
        assert loaded == cases

    def test_load_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps({"id": "a", "category": "answerable", "question": "q"}) + "\n\n",
            encoding="utf-8",
        )
        assert len(load_agent_cases(path)) == 1
