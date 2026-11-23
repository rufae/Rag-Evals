"""ragevals — evaluación de calidad para sistemas RAG, LLM y agentes.

Métricas reproducibles (recall@k, MRR, refusal, fugas), guardrails de salida
y gates de calidad para CI. Desacoplado de cualquier RAG concreto: tú aportas
la función de recuperación y `ragevals` la mide.

Para agentes (`ragevals.agent`): selección de herramientas, similitud de
trayectoria, detección de bucles y pruebas de inyección alineadas con el
OWASP LLM Top 10, sobre trazas de Langfuse/OpenTelemetry.

Basado en el framework de evaluación escrito para el proyecto CV-Web
(recuperación con umbrales calibrados, 49 casos dorados, recall@5=0,983).
"""

from ragevals.agent import (
    AGENT_DEFAULT_GATES,
    BUILTIN_ATTACKS,
    OWASP_LLM_TOP_10,
    AgentCase,
    AgentMetrics,
    AgentTrace,
    ToolCall,
    agent_gates_passed,
    classify_owasp,
    evaluate_agent,
    load_agent_cases,
    load_traces,
)
from ragevals.guards import GuardDecision, OutputGuard, is_explicit_extraction
from ragevals.metrics import DEFAULT_GATES, EvalMetrics, gates_passed
from ragevals.ranker import EmbeddingRanker
from ragevals.runner import RankedResult, evaluate_cases
from ragevals.types import EvalCase

__all__ = [
    "AGENT_DEFAULT_GATES",
    "BUILTIN_ATTACKS",
    "DEFAULT_GATES",
    "OWASP_LLM_TOP_10",
    "AgentCase",
    "AgentMetrics",
    "AgentTrace",
    "EmbeddingRanker",
    "EvalCase",
    "EvalMetrics",
    "GuardDecision",
    "OutputGuard",
    "RankedResult",
    "ToolCall",
    "agent_gates_passed",
    "classify_owasp",
    "evaluate_agent",
    "evaluate_cases",
    "gates_passed",
    "is_explicit_extraction",
    "load_agent_cases",
    "load_traces",
]

__version__ = "0.1.0"
