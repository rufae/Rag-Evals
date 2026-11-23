"""Tests de la ingesta de trazas (Langfuse y OpenTelemetry)."""

import json
from pathlib import Path

import pytest

from ragevals.agent.traces import (
    _otel_attrs,
    canonical,
    detect_format,
    from_langfuse,
    from_otel,
    load_traces,
)


class TestCanonical:
    def test_none_is_empty(self):
        assert canonical(None) == ""

    def test_string_passthrough(self):
        assert canonical("hola") == "hola"

    def test_dict_is_stable_and_sorted(self):
        assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})

    def test_nested_dict_is_json(self):
        assert json.loads(canonical({"x": [1, 2]})) == {"x": [1, 2]}


LANGFUSE = {
    "data": [
        {
            "id": "t1",
            "input": "pregunta",
            "output": "respuesta final",
            "metadata": {"case_id": "c1", "refused": False},
            "observations": [
                {
                    "id": "o2",
                    "type": "TOOL",
                    "name": "read_note",
                    "startTime": "2026-01-01T00:00:02Z",
                    "input": {"id": 1},
                    "output": "nota",
                },
                {
                    "id": "o1",
                    "type": "GENERATION",
                    "name": "plan",
                    "startTime": "2026-01-01T00:00:01Z",
                },
                {
                    "id": "o3",
                    "type": "TOOL",
                    "name": "search_notes",
                    "startTime": "2026-01-01T00:00:00Z",
                    "input": {"query": "x"},
                    "level": "ERROR",
                },
            ],
        }
    ]
}


class TestFromLangfuse:
    def test_tool_observations_only(self):
        (trace,) = from_langfuse(LANGFUSE)
        assert trace.tools == ("search_notes", "read_note")  # ordenado por startTime

    def test_metadata_carried_over(self):
        (trace,) = from_langfuse(LANGFUSE)
        assert trace.case_id == "c1"
        assert trace.final == "respuesta final"
        assert trace.refused is False

    def test_error_status_detected(self):
        (trace,) = from_langfuse(LANGFUSE)
        assert trace.steps[0].status == "error"

    def test_args_serialized_as_json(self):
        (trace,) = from_langfuse(LANGFUSE)
        assert json.loads(trace.steps[1].args) == {"id": 1}

    def test_tool_names_match_by_name(self):
        payload = {
            "observations": [
                {"type": "SPAN", "name": "get_calendar", "input": {"range": "hoy"}, "output": []}
            ]
        }
        trace = from_langfuse(payload, tool_names=("get_calendar",))[0]
        assert trace.tools == ("get_calendar",)

    def test_single_trace_without_wrapper(self):
        assert len(from_langfuse(LANGFUSE["data"])) == 1

    def test_refused_from_metadata(self):
        payload = {"observations": [], "metadata": {"refused": "true"}}
        assert from_langfuse(payload)[0].refused is True

    def test_tool_name_without_type(self):
        payload = {"observations": [{"toolName": "delete_note", "input": {"id": 1}}]}
        assert from_langfuse(payload)[0].tools == ("delete_note",)

    def test_tool_observation_without_name_is_skipped(self):
        payload = {"observations": [{"type": "TOOL"}, {"type": "TOOL", "name": ""}]}
        assert from_langfuse(payload)[0].tools == ()

    def test_non_dict_payload_raises(self):
        with pytest.raises(ValueError):
            from_langfuse(42)


OTEL = {
    "resourceSpans": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        {
                            "traceId": "aa",
                            "spanId": "a1",
                            "name": "invoke_agent",
                            "startTimeUnixNano": "1000",
                            "attributes": [
                                {"key": "gen_ai.response.text", "value": {"stringValue": "listo"}},
                                {"key": "eval.case_id", "value": {"stringValue": "c1"}},
                            ],
                        },
                        {
                            "traceId": "aa",
                            "spanId": "a2",
                            "name": "read_note",
                            "startTimeUnixNano": "3000",
                            "attributes": [
                                {"key": "gen_ai.tool.name", "value": {"stringValue": "read_note"}},
                                {
                                    "key": "gen_ai.tool.call.arguments",
                                    "value": {"stringValue": '{"id": 1}'},
                                },
                                {
                                    "key": "gen_ai.tool.call.result",
                                    "value": {"stringValue": "la nota"},
                                },
                            ],
                        },
                        {
                            "traceId": "aa",
                            "spanId": "a3",
                            "name": "search_notes",
                            "startTimeUnixNano": "2000",
                            "attributes": [
                                {
                                    "key": "gen_ai.tool.name",
                                    "value": {"stringValue": "search_notes"},
                                },
                            ],
                        },
                        {
                            "traceId": "bb",
                            "spanId": "b1",
                            "name": "invoke_agent",
                            "startTimeUnixNano": "4000",
                            "attributes": [
                                {"key": "eval.refused", "value": {"boolValue": True}},
                            ],
                        },
                    ]
                }
            ]
        }
    ]
}


class TestFromOtel:
    def test_grouped_by_trace_and_sorted_by_time(self):
        traces = from_otel(OTEL)
        assert len(traces) == 2
        assert traces[0].tools == ("search_notes", "read_note")
        assert traces[1].tools == ()

    def test_final_case_id_and_refused(self):
        first, second = from_otel(OTEL)
        assert first.final == "listo"
        assert first.case_id == "c1"
        assert second.refused is True

    def test_span_name_used_when_in_catalog(self):
        payload = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "cc",
                                    "spanId": "c1",
                                    "name": "get_calendar",
                                    "startTimeUnixNano": "1",
                                    "attributes": [],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        assert from_otel(payload, tool_names=("get_calendar",))[0].tools == ("get_calendar",)

    def test_flat_span_list_accepted(self):
        spans = [OTEL["resourceSpans"][0]["scopeSpans"][0]["spans"][0]]
        assert from_otel(spans)[0].case_id == "c1"

    def test_unknown_payload_raises(self):
        with pytest.raises(ValueError):
            from_otel({"nada": True})

    def test_single_span_dict_accepted(self):
        span = OTEL["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert from_otel(span)[0].case_id == "c1"

    def test_resource_spans_must_be_a_list(self):
        with pytest.raises(ValueError):
            from_otel({"resourceSpans": "no-soy-una-lista"})

    def test_non_dict_resource_is_skipped(self):
        payload = {"resourceSpans": [42, OTEL["resourceSpans"][0]]}
        assert len(from_otel(payload)) == 2


class TestOtelAttributeValues:
    def _attrs(self, value: object) -> dict[str, str]:
        return _otel_attrs({"attributes": [{"key": "a", "value": value}]})

    def test_missing_value_is_empty(self):
        assert _otel_attrs({"attributes": [{"key": "a"}]})["a"] == ""

    def test_scalar_types(self):
        assert _otel_attrs({"attributes": [{"key": "a", "value": {"intValue": 7}}]})["a"] == "7"
        assert (
            _otel_attrs({"attributes": [{"key": "a", "value": {"boolValue": True}}]})["a"] == "True"
        )

    def test_array_value(self):
        value = {"arrayValue": {"values": [{"stringValue": "x"}, {"intValue": 3}]}}
        assert json.loads(self._attrs(value)["a"]) == ["x", "3"]

    def test_kvlist_value(self):
        value = {"kvlistValue": {"values": [{"key": "k", "value": {"stringValue": "v"}}]}}
        assert json.loads(self._attrs(value)["a"]) == {"k": "v"}

    def test_unknown_shape_falls_back_to_json(self):
        assert json.loads(self._attrs({"weird": 1})["a"]) == {"weird": 1}

    def test_non_dict_value_is_canonicalized(self):
        assert self._attrs(None)["a"] == ""


class TestDetectFormat:
    def test_langfuse(self):
        assert detect_format(LANGFUSE) == "langfuse"

    def test_otel(self):
        assert detect_format(OTEL) == "otel"

    def test_bare_span_is_otel(self):
        assert detect_format({"traceId": "aa", "spanId": "a1"}) == "otel"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            detect_format({"x": 1})


class TestLoadTraces:
    def test_auto_detect_langfuse(self, tmp_path: Path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps(LANGFUSE), encoding="utf-8")
        traces = load_traces(path)
        assert traces[0].tools == ("search_notes", "read_note")

    def test_auto_detect_otel(self, tmp_path: Path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps(OTEL), encoding="utf-8")
        traces = load_traces(path)
        assert len(traces) == 2

    def test_explicit_format_overrides(self, tmp_path: Path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps(OTEL), encoding="utf-8")
        assert len(load_traces(path, "otel")) == 2

    def test_bad_format_raises(self, tmp_path: Path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps(LANGFUSE), encoding="utf-8")
        with pytest.raises(ValueError):
            load_traces(path, "xml")
