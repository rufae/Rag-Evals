"""Reportes de evaluación de agentes: JSON (máquinas) y Markdown (humanos/CI)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ragevals.agent.metrics import AGENT_DEFAULT_GATES, AgentMetrics, agent_gates_passed
from ragevals.agent.owasp import owasp_name


def write_agent_json(
    metrics: AgentMetrics, path: str | Path, *, gates: dict[str, float] | None = None
) -> None:
    payload = {
        "date": date.today().isoformat(),
        "metrics": metrics.to_dict(),
        "gates": {**AGENT_DEFAULT_GATES, **(gates or {})},
        "passed": agent_gates_passed(metrics, gates),
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _row(name: str, value: str, gate: str) -> tuple[str, str, str]:
    return (name, value, gate)


def render_agent_markdown(metrics: AgentMetrics, *, gates: dict[str, float] | None = None) -> str:
    active = {**AGENT_DEFAULT_GATES, **(gates or {})}
    ok = agent_gates_passed(metrics, gates)
    rows = [
        _row("tool recall", f"{metrics.tool_recall:.3f}", f"≥ {active['tool_recall']}"),
        _row("tool precision", f"{metrics.tool_precision:.3f}", "—"),
        _row("match exacto", f"{metrics.exact_tool_match:.3f}", "—"),
        _row("1ª herramienta", f"{metrics.first_tool_accuracy:.3f}", "—"),
        _row(
            "similitud de trayectoria",
            f"{metrics.trajectory_similarity:.3f}",
            f"≥ {active['trajectory_similarity']}",
        ),
        _row("eficiencia de pasos", f"{metrics.step_efficiency:.3f}", "—"),
        _row(
            "refusal recall",
            f"{metrics.refusal_recall:.3f} ({metrics.refusal_cases} casos)",
            f"≥ {active['refusal_recall']}",
        ),
        _row("refusal precision", f"{metrics.refusal_precision:.3f}", "—"),
        _row("bucles", str(metrics.loops), f"≤ {int(active['loops'])}"),
        _row("pasos excedidos", str(metrics.max_steps_exceeded), "—"),
        _row(
            "llamadas prohibidas",
            str(metrics.forbidden_tool_calls),
            f"≤ {int(active['forbidden_tool_calls'])}",
        ),
        _row("herramientas inventadas", str(metrics.unknown_tool_calls), "—"),
        _row("fugas (canario/guard)", str(metrics.leaks), f"≤ {int(active['leaks'])}"),
        _row(
            "ataques bloqueados",
            f"{metrics.attacks_blocked}/{metrics.attacks}",
            "—",
        ),
        _row(
            "ataques con éxito",
            str(metrics.attacks_succeeded),
            f"≤ {int(active['attacks_succeeded'])}",
        ),
        _row(
            "trazas ausentes",
            str(metrics.missing_traces),
            f"≤ {int(active['missing_traces'])}",
        ),
    ]
    lines = [
        f"# Evaluación de agente — {'✅ PASS' if ok else '❌ FAIL'}",
        "",
        "| Métrica | Valor | Gate |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {value} | {gate} |" for name, value, gate in rows]

    if metrics.by_owasp:
        lines += [
            "",
            "## OWASP LLM Top 10",
            "",
            "| ID | Categoría | Ataques | Bloqueados | Con éxito |",
            "|---|---|---|---|---|",
        ]
        for owasp_id in sorted(metrics.by_owasp):
            bucket = metrics.by_owasp[owasp_id]
            lines.append(
                f"| {owasp_id} | {owasp_name(owasp_id)} | {bucket['cases']} "
                f"| {bucket['blocked']} | {bucket['succeeded']} |"
            )

    lines.append("")
    return "\n".join(lines)


def write_agent_markdown(
    metrics: AgentMetrics, path: str | Path, *, gates: dict[str, float] | None = None
) -> None:
    Path(path).write_text(render_agent_markdown(metrics, gates=gates), encoding="utf-8")
