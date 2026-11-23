"""ragevals — evaluación de calidad para sistemas RAG y LLM.

Métricas reproducibles (recall@k, MRR, refusal, fugas), guardrails de salida
y gates de calidad para CI. Desacoplado de cualquier RAG concreto: tú aportas
la función de recuperación y `ragevals` la mide.

Basado en el framework de evaluación escrito para el proyecto CV-Web
(recuperación con umbrales calibrados, 49 casos dorados, recall@5=0,983).
"""

from ragevals.guards import GuardDecision, OutputGuard, is_explicit_extraction
from ragevals.metrics import DEFAULT_GATES, EvalMetrics, gates_passed
from ragevals.ranker import EmbeddingRanker
from ragevals.runner import RankedResult, evaluate_cases
from ragevals.types import EvalCase

__all__ = [
    "DEFAULT_GATES",
    "EmbeddingRanker",
    "EvalCase",
    "EvalMetrics",
    "GuardDecision",
    "OutputGuard",
    "RankedResult",
    "evaluate_cases",
    "gates_passed",
    "is_explicit_extraction",
]

__version__ = "0.1.0"
