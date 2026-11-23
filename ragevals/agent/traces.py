"""Ingesta de trazas: export JSON de Langfuse y OTLP JSON de OpenTelemetry.

Sin dependencias: los SDK de ambos ya exportan JSON, así que `ragevals`
sólo lo parsea y lo normaliza a ``AgentTrace``. La idea es la misma que con
el retriever: tú aportas la traza (viva o grabada), aquí se mide.

Formatos aceptados por ``load_traces``:

- ``langfuse``: un trace (``{"id": ..., "observations": [...]}``), una lista de
  traces o un paginado ``{"data": [...]}``.
- ``otel``: un export OTLP/JSON (``{"resourceSpans": [{"scopeSpans": [{"spans":
  [...]}]}]}``), también ``resourceLogs`` o una lista plana de spans.

En ambos casos, para reconstruir la trayectoria hacen falta observaciones que
sean llamadas a herramienta: en Langfuse por ``type: "TOOL"``, por llevar
``toolName``/``toolCallId`` o por coincidir con ``tool_names``; en OTel por los
atributos de la convención semántica GenAI (``gen_ai.tool.name``,
``tool.name``...) o por coincidir con ``tool_names``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragevals.agent.types import STATUS_ERROR, STATUS_OK, AgentTrace, ToolCall

FORMAT_LANGFUSE = "langfuse"
FORMAT_OTEL = "otel"
FORMAT_AUTO = "auto"

# Langfuse: observaciones que son llamadas a herramienta.
_LANGFUSE_TOOL_TYPES = frozenset({"TOOL", "TOOL_CALL"})

# OpenTelemetry: convenciones semánticas GenAI / genéricas de tooling.
_OTEL_TOOL_NAME_KEYS = (
    "gen_ai.tool.name",
    "gen_ai.tool_call.name",
    "tool.name",
    "tool_call.name",
    "ai.toolCall.name",
    "function.name",
)
_OTEL_TOOL_ID_KEYS = ("gen_ai.tool.call.id", "tool.call.id", "tool_call.id", "ai.toolCall.id")
_OTEL_TOOL_ARGS_KEYS = (
    "gen_ai.tool.call.arguments",
    "gen_ai.tool_call.arguments",
    "tool.input",
    "tool_call.arguments",
    "ai.toolCall.args",
    "input",
)
_OTEL_TOOL_RESULT_KEYS = (
    "gen_ai.tool.call.result",
    "gen_ai.tool_call.result",
    "tool.output",
    "tool_call.result",
    "ai.toolCall.result",
    "output",
)
_OTEL_FINAL_KEYS = (
    "gen_ai.response.text",
    "gen_ai.response.output",
    "gen_ai.output",
    "output.value",
    "eval.final",
)
_OTEL_CASE_KEYS = ("eval.case_id", "case_id", "eval.case")
_OTEL_REFUSED_KEYS = ("eval.refused", "refused")
_OTEL_ERROR_STATUS = frozenset({"ERROR", "error"})


def canonical(value: object) -> str:
    """Serialización estable de argumentos/resultados.

    Se usa para comparar llamadas y detectar bucles: el mismo objeto dict
    debe producir siempre el mismo string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - casos exóticos
        return str(value)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "sí", "si"}


def _first(attrs: dict[str, str], keys: tuple[str, ...]) -> str:
    return next((attrs[key] for key in keys if attrs.get(key)), "")


# --------------------------------------------------------------------------- #
# Langfuse
# --------------------------------------------------------------------------- #
def _langfuse_traces(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "traces"):
            inner = payload.get(key)
            if isinstance(inner, list):
                payload = inner
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("export de Langfuse no reconocido (se esperaba un objeto o una lista)")
    return [item for item in payload if isinstance(item, dict)]


def _langfuse_case_id(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    for key in ("case_id", "caseId", "eval_case_id"):
        if metadata.get(key):
            return str(metadata[key])
    return ""


def _langfuse_is_tool(obs: dict[str, Any], tool_names: frozenset[str]) -> bool:
    if str(obs.get("type") or "").upper() in _LANGFUSE_TOOL_TYPES:
        return True
    if obs.get("toolName") or obs.get("toolCallId"):
        return True
    name = str(obs.get("name") or "")
    return bool(name) and name in tool_names


def from_langfuse(payload: object, *, tool_names: tuple[str, ...] = ()) -> list[AgentTrace]:
    """Normaliza un export de Langfuse a ``AgentTrace``."""
    names = frozenset(tool_names)
    traces: list[AgentTrace] = []
    for raw in _langfuse_traces(payload):
        observations = [o for o in (raw.get("observations") or []) if isinstance(o, dict)]
        observations = [o for o in observations if _langfuse_is_tool(o, names)]
        observations.sort(key=lambda o: str(o.get("startTime") or ""))

        steps: list[ToolCall] = []
        for obs in observations:
            tool = str(obs.get("toolName") or obs.get("name") or "")
            if not tool:
                continue
            error = str(obs.get("level") or "").upper() == "ERROR" or bool(obs.get("statusMessage"))
            steps.append(
                ToolCall(
                    tool=tool,
                    args=canonical(obs.get("input")),
                    result=canonical(obs.get("output")),
                    status=STATUS_ERROR if error else STATUS_OK,
                )
            )

        metadata = raw.get("metadata")
        traces.append(
            AgentTrace(
                steps=tuple(steps),
                final=canonical(raw.get("output")),
                refused=_truthy(metadata.get("refused")) if isinstance(metadata, dict) else False,
                case_id=_langfuse_case_id(raw),
            )
        )
    return traces


# --------------------------------------------------------------------------- #
# OpenTelemetry (OTLP/JSON)
# --------------------------------------------------------------------------- #
def _otel_scalar(value: object) -> str:
    if not isinstance(value, dict):
        return canonical(value)
    for key in ("stringValue", "intValue", "boolValue", "doubleValue"):
        if key in value:
            return str(value[key])
    if "arrayValue" in value:
        items = (value.get("arrayValue") or {}).get("values") or []
        return canonical([_otel_scalar(item) for item in items])
    if "kvlistValue" in value:
        pairs = (value.get("kvlistValue") or {}).get("values") or []
        return canonical({str(p.get("key")): _otel_scalar(p.get("value")) for p in pairs})
    return canonical(value)


def _otel_attrs(span: dict[str, Any]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in span.get("attributes") or []:
        if isinstance(item, dict) and item.get("key"):
            attrs[str(item["key"])] = _otel_scalar(item.get("value"))
    return attrs


def _otel_spans(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "traceId" in payload or "spanId" in payload:
            return [payload]
        for key in ("resourceSpans", "resourceLogs"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError("export OTel no reconocido (falta resourceSpans/resourceLogs)")
    if not isinstance(payload, list):
        raise ValueError("export OTel no reconocido (se esperaba una lista de resources)")
    if payload and isinstance(payload[0], dict) and "spanId" in payload[0]:
        return [span for span in payload if isinstance(span, dict)]

    spans: list[dict[str, Any]] = []
    for resource in payload:
        if not isinstance(resource, dict):
            continue
        for scope_key in ("scopeSpans", "scopeLogs", "scopes"):
            for scope in resource.get(scope_key) or []:
                if isinstance(scope, dict):
                    spans.extend(s for s in (scope.get("spans") or []) if isinstance(s, dict))
    return spans


def from_otel(payload: object, *, tool_names: tuple[str, ...] = ()) -> list[AgentTrace]:
    """Normaliza un export OTLP/JSON a ``AgentTrace``.

    Los spans se agrupan por ``traceId`` y se ordenan por marca de tiempo.
    """
    names = frozenset(tool_names)
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for span in _otel_spans(payload):
        trace_id = str(span.get("traceId") or "0")
        if trace_id not in grouped:
            grouped[trace_id] = []
            order.append(trace_id)
        grouped[trace_id].append(span)

    traces: list[AgentTrace] = []
    for trace_id in order:
        spans = sorted(grouped[trace_id], key=lambda s: int(s.get("startTimeUnixNano") or 0))
        steps: list[ToolCall] = []
        final = ""
        refused = False
        case_id = ""
        for span in spans:
            attrs = _otel_attrs(span)
            name = _first(attrs, _OTEL_TOOL_NAME_KEYS) or (
                str(span.get("name") or "") if str(span.get("name") or "") in names else ""
            )
            if name:
                code = str((span.get("status") or {}).get("code") or "")
                steps.append(
                    ToolCall(
                        tool=name,
                        args=_first(attrs, _OTEL_TOOL_ARGS_KEYS),
                        result=_first(attrs, _OTEL_TOOL_RESULT_KEYS),
                        status=STATUS_ERROR if code.upper() in _OTEL_ERROR_STATUS else STATUS_OK,
                    )
                )
            final = final or _first(attrs, _OTEL_FINAL_KEYS)
            case_id = case_id or _first(attrs, _OTEL_CASE_KEYS)
            refused = refused or any(
                _truthy(attrs[key]) for key in _OTEL_REFUSED_KEYS if attrs.get(key)
            )

        traces.append(AgentTrace(steps=tuple(steps), final=final, refused=refused, case_id=case_id))
    return traces


# --------------------------------------------------------------------------- #
# Carga genérica
# --------------------------------------------------------------------------- #
def _unwrap(payload: object) -> object:
    if isinstance(payload, dict):
        for key in ("data", "traces"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return inner
    return payload


def detect_format(payload: object) -> str:
    """Adivina el formato del export: ``langfuse`` u ``otel``."""
    sample = _unwrap(payload)
    sample = sample[0] if isinstance(sample, list) and sample else sample
    if isinstance(sample, dict):
        if any(key in sample for key in ("resourceSpans", "resourceLogs", "scopeSpans")):
            return FORMAT_OTEL
        if "traceId" in sample or "spanId" in sample:
            return FORMAT_OTEL
        if "observations" in sample or "input" in sample:
            return FORMAT_LANGFUSE
    raise ValueError("formato de traza no reconocido (usa --format langfuse|otel)")


def load_traces(
    path: str | Path,
    fmt: str = FORMAT_AUTO,
    *,
    tool_names: tuple[str, ...] = (),
) -> list[AgentTrace]:
    """Lee un export JSON de trazas y lo normaliza a ``AgentTrace``."""
    payload = _unwrap(json.loads(Path(path).read_text(encoding="utf-8")))
    if fmt == FORMAT_AUTO:
        fmt = detect_format(payload)
    if fmt == FORMAT_LANGFUSE:
        return from_langfuse(payload, tool_names=tool_names)
    if fmt == FORMAT_OTEL:
        return from_otel(payload, tool_names=tool_names)
    raise ValueError(f"formato desconocido: {fmt}")
