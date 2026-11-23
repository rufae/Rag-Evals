"""Reportes de evaluación: JSON (máquinas) y Markdown (humanos/CI)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ragevals.metrics import DEFAULT_GATES, EvalMetrics, gates_passed


def write_json(
    metrics: EvalMetrics, path: str | Path, *, gates: dict[str, float] | None = None
) -> None:
    payload = {
        "date": date.today().isoformat(),
        "metrics": metrics.to_dict(),
        "gates": {**DEFAULT_GATES, **(gates or {})},
        "passed": gates_passed(metrics, gates),
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def render_markdown(metrics: EvalMetrics, *, gates: dict[str, float] | None = None) -> str:
    active = {**DEFAULT_GATES, **(gates or {})}
    ok = gates_passed(metrics, gates)
    rows = [
        ("recall@k", f"{metrics.recall_at_k:.3f}", f"≥ {active['recall_at_k']}"),
        ("MRR", f"{metrics.mrr:.3f}", "—"),
        ("hit rate", f"{metrics.hit_rate:.3f}", "—"),
        ("refusal recall", f"{metrics.refusal_recall:.3f}", f"≥ {active['refusal_recall']}"),
        ("refusal precision", f"{metrics.refusal_precision:.3f}", "—"),
        ("fugas (canario)", str(metrics.leaks), f"≤ {int(active['leaks'])}"),
        (
            "inyecciones bloqueadas",
            f"{metrics.injection_refused}/{metrics.injection_cases}",
            "—",
        ),
    ]
    lines = [
        f"# Evaluación — {'✅ PASS' if ok else '❌ FAIL'}",
        "",
        "| Métrica | Valor | Gate |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {value} | {gate} |" for name, value, gate in rows]
    lines.append("")
    return "\n".join(lines)


def write_markdown(
    metrics: EvalMetrics, path: str | Path, *, gates: dict[str, float] | None = None
) -> None:
    Path(path).write_text(render_markdown(metrics, gates=gates), encoding="utf-8")
