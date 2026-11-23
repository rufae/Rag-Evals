"""ragevals para agentes: selección de herramientas, trayectoria y OWASP.

Medir si un agente elige bien las herramientas y sigue la trayectoria
correcta, con trazas de Langfuse/OpenTelemetry y pruebas de inyección
alineadas con el OWASP LLM Top 10.

Ejemplo mínimo::

    from ragevals.agent import AgentCase, evaluate_agent, agent_gates_passed
    from ragevals.agent.owasp import BUILTIN_ATTACKS

    cases = [AgentCase.from_dict({...}), *BUILTIN_ATTACKS]
    metrics = evaluate_agent(cases, runner=mi_agente, tool_catalog=TOOLS)
    assert agent_gates_passed(metrics)
"""

from ragevals.agent.metrics import (
    AGENT_DEFAULT_GATES,
    AgentMetrics,
    agent_gates_passed,
    has_loop,
    levenshtein,
    sequence_similarity,
)
from ragevals.agent.owasp import BUILTIN_ATTACKS, OWASP_LLM_TOP_10, classify_owasp, owasp_name
from ragevals.agent.report import (
    render_agent_markdown,
    write_agent_json,
    write_agent_markdown,
)
from ragevals.agent.runner import DEFAULT_MAX_STEPS, AgentRunner, evaluate_agent
from ragevals.agent.traces import detect_format, from_langfuse, from_otel, load_traces
from ragevals.agent.types import (
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_OK,
    AgentCase,
    AgentTrace,
    ToolCall,
    dump_agent_cases,
    load_agent_cases,
)

__all__ = [
    "AGENT_DEFAULT_GATES",
    "BUILTIN_ATTACKS",
    "DEFAULT_MAX_STEPS",
    "OWASP_LLM_TOP_10",
    "STATUS_BLOCKED",
    "STATUS_ERROR",
    "STATUS_OK",
    "AgentCase",
    "AgentMetrics",
    "AgentRunner",
    "AgentTrace",
    "ToolCall",
    "agent_gates_passed",
    "classify_owasp",
    "detect_format",
    "dump_agent_cases",
    "evaluate_agent",
    "from_langfuse",
    "from_otel",
    "has_loop",
    "levenshtein",
    "load_agent_cases",
    "load_traces",
    "owasp_name",
    "render_agent_markdown",
    "sequence_similarity",
    "write_agent_json",
    "write_agent_markdown",
]
