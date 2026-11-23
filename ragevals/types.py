"""Tipos del dataset de evaluación (JSONL).

Cada caso: ``id``, ``category`` y ``question``. Las categorías soportadas:

- ``answerable``: tiene respuesta en la base → exige ``expected_sources``.
- ``out_of_domain`` (o la categoría de rechazo que configures): el sistema
  debe negarse/renunciar.
- ``injection``: intento de extracción (prompt, PII, jailbreak) → debe bloquearse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ANSWERABLE = "answerable"
INJECTION = "injection"
DEFAULT_REFUSAL_CATEGORIES = frozenset({"out_of_domain"})


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    expected_sources: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> EvalCase:
        expected = tuple(str(item) for item in (raw.get("expected_sources") or []))
        return cls(
            id=str(raw["id"]),
            category=str(raw["category"]),
            question=str(raw["question"]),
            expected_sources=expected,
        )


def load_cases(path: str | Path) -> list[EvalCase]:
    """Lee un dataset JSONL (una pregunta por línea)."""
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvalCase.from_dict(json.loads(line)))
    return cases
