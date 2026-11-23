"""Tipos del dataset de agentes y de las trazas de ejecución.

Un *caso de agente* describe una tarea y qué se espera de ella: qué
herramientas debe usar, cuáles está prohibido tocar y cuántos pasos como
máximo. Una *traza* es la secuencia real de llamadas a herramientas que hizo
el agente: la generas ejecutándolo tú con un runner, o la lees de un export
de Langfuse/OpenTelemetry (ver ``ragevals.agent.traces``).

Categorías (las mismas de ``ragevals.types``):

- ``answerable``: exige ``expected_tools`` (la trayectoria correcta).
- ``out_of_domain``: el agente debe renunciar sin tocar herramientas.
- ``injection``: ataque (ver ``ragevals.agent.owasp``) → no debe ejecutar nada.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ragevals.types import ANSWERABLE, INJECTION  # noqa: F401  (reexport útil)

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class ToolCall:
    """Una llamada a herramienta observada en la traza."""

    tool: str
    args: str = ""
    result: str = ""
    status: str = STATUS_OK

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> ToolCall:
        return cls(
            tool=str(raw["tool"]),
            args=str(raw.get("args") or ""),
            result=str(raw.get("result") or ""),
            status=str(raw.get("status") or STATUS_OK),
        )

    @property
    def signature(self) -> tuple[str, str]:
        """Par (herramienta, argumentos): la unidad de detección de bucles."""
        return (self.tool, self.args)


@dataclass(frozen=True)
class AgentTrace:
    """Trajectory observada: qué llamó el agente y qué respondió al final."""

    steps: tuple[ToolCall, ...] = field(default_factory=tuple)
    final: str = ""
    refused: bool = False
    case_id: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> AgentTrace:
        steps = tuple(ToolCall.from_dict(item) for item in (raw.get("steps") or []))
        return cls(
            steps=steps,
            final=str(raw.get("final") or ""),
            refused=bool(raw.get("refused") or False),
            case_id=str(raw.get("case_id") or ""),
        )

    @property
    def tools(self) -> tuple[str, ...]:
        """Secuencia de nombres de herramientas, en orden de ejecución."""
        return tuple(step.tool for step in self.steps)

    @property
    def texts(self) -> tuple[str, ...]:
        """Todos los textos observables: resultados + respuesta final."""
        return (*(step.result for step in self.steps), self.final)

    @property
    def refused_or_inert(self) -> bool:
        """¿Se negó explícitamente o no llegó a ejecutar nada?"""
        return self.refused or not self.steps


@dataclass(frozen=True)
class AgentCase:
    """Caso del dataset de evaluación de agentes."""

    id: str
    category: str
    question: str
    expected_tools: tuple[str, ...] = field(default_factory=tuple)
    forbidden_tools: tuple[str, ...] = field(default_factory=tuple)
    max_steps: int = 0  # 0 → se usa el ``max_steps`` global de la evaluación
    owasp: str = ""  # p. ej. "LLM01" (ver ragevals.agent.owasp)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> AgentCase:
        def _seq(key: str) -> tuple[str, ...]:
            return tuple(str(item) for item in (raw.get(key) or ()))

        return cls(
            id=str(raw["id"]),
            category=str(raw["category"]),
            question=str(raw["question"]),
            expected_tools=_seq("expected_tools"),
            forbidden_tools=_seq("forbidden_tools"),
            max_steps=int(raw.get("max_steps") or 0),
            owasp=str(raw.get("owasp") or ""),
        )

    @property
    def is_attack(self) -> bool:
        return self.category == INJECTION


def load_agent_cases(path: str | Path) -> list[AgentCase]:
    """Lee un dataset JSONL de agentes (un caso por línea)."""
    cases: list[AgentCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(AgentCase.from_dict(json.loads(line)))
    return cases


def dump_agent_cases(cases: list[AgentCase], path: str | Path) -> None:
    """Escribe un dataset JSONL de agentes (para exportar el catálogo propio)."""
    rows = [
        json.dumps(
            {
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "expected_tools": list(case.expected_tools),
                "forbidden_tools": list(case.forbidden_tools),
                **({"max_steps": case.max_steps} if case.max_steps else {}),
                **({"owasp": case.owasp} if case.owasp else {}),
            },
            ensure_ascii=False,
        )
        for case in cases
    ]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
