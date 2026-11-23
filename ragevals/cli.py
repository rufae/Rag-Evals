"""CLI de ragevals.

Uso típico en CI (tu retriever vive en un módulo tuyo con una función que
recibe la pregunta y devuelve ``RankedResult``):

    ragevals run --dataset eval/dataset.jsonl \\
                 --retriever mi_proyecto.retriever:buscar \\
                 --top-k 5 --min-score 0.32 \\
                 --report reports/eval.json --markdown reports/eval.md

Para agentes (tu agente vive en un módulo tuyo con una función que recibe el
caso y devuelve ``AgentTrace``; o lees trazas ya grabadas de Langfuse/OTel):

    ragevals agent --dataset eval/agent.jsonl \\
                   --runner mi_proyecto.agente:ejecutar \\
                   --tools search_notes,read_note,delete_note \\
                   --attacks --report reports/agent.json

Sale con código != 0 si no se cumplen los gates → rompe el build en CI.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable

from ragevals.agent.metrics import agent_gates_passed
from ragevals.agent.owasp import BUILTIN_ATTACKS
from ragevals.agent.report import write_agent_json, write_agent_markdown
from ragevals.agent.runner import DEFAULT_MAX_STEPS, AgentRunner, evaluate_agent
from ragevals.agent.traces import load_traces
from ragevals.agent.types import load_agent_cases
from ragevals.guards import is_explicit_extraction
from ragevals.metrics import EvalMetrics, gates_passed
from ragevals.report import write_json, write_markdown
from ragevals.runner import Retriever, evaluate_cases
from ragevals.types import load_cases


def _load_callable(spec: str, flag: str) -> Callable[..., object]:
    """Importa ``paquete.modulo:funcion``."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"{flag} debe tener la forma 'modulo:funcion'")
    module = importlib.import_module(module_name)
    func = getattr(module, attr, None)
    if not callable(func):
        raise SystemExit(f"{spec} no es invocable")
    return func


def _load_retriever(spec: str) -> Retriever:
    """Importa ``paquete.modulo:funcion`` como retriever."""
    return _load_callable(spec, "--retriever")  # type: ignore[return-value]


def _load_runner(spec: str) -> AgentRunner:
    """Importa ``paquete.modulo:funcion`` como runner de agente."""
    return _load_callable(spec, "--runner")  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragevals", description="Evalúa la calidad de un RAG/LLM")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evalúa un retriever contra un dataset JSONL")
    run.add_argument("--dataset", required=True, help="dataset JSONL (ver ragevals.types.EvalCase)")
    run.add_argument(
        "--retriever", required=True, help="'modulo:funcion' que devuelve RankedResult"
    )
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--min-score", type=float, default=0.0, help="por debajo, el sistema rechaza")
    run.add_argument("--canary", default="", help="token canario para medir fugas")
    run.add_argument("--report", default="", help="salida JSON")
    run.add_argument("--markdown", default="", help="salida Markdown")

    agent = sub.add_parser(
        "agent", help="evalúa herramientas, trayectoria y ataques OWASP de un agente"
    )
    agent.add_argument(
        "--dataset", required=True, help="dataset JSONL (ver ragevals.agent.AgentCase)"
    )
    agent.add_argument(
        "--runner", default="", help="'modulo:funcion' que devuelve AgentTrace (ejecuta el agente)"
    )
    agent.add_argument(
        "--traces",
        default="",
        help="export JSON de Langfuse u OTLP JSON de OpenTelemetry (en vez de --runner)",
    )
    agent.add_argument(
        "--format",
        choices=("auto", "langfuse", "otel"),
        default="auto",
        help="formato de --traces (por defecto se detecta solo)",
    )
    agent.add_argument(
        "--tools",
        default="",
        help="catálogo de herramientas permitidas, separadas por coma",
    )
    agent.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    agent.add_argument("--canary", default="", help="token canario para medir fugas")
    agent.add_argument(
        "--attacks",
        action="store_true",
        help="añade el catálogo de ataques OWASP embebido (ragevals.agent.owasp)",
    )
    agent.add_argument("--report", default="", help="salida JSON")
    agent.add_argument("--markdown", default="", help="salida Markdown")

    args = parser.parse_args(argv)
    if args.command == "agent":
        return _run_agent(args)

    cases = load_cases(args.dataset)
    retriever = _load_retriever(args.retriever)
    metrics: EvalMetrics = evaluate_cases(
        cases,
        retriever,
        top_k=args.top_k,
        min_score=args.min_score,
        canary=args.canary,
        injection_guard=is_explicit_extraction,
    )
    if args.report:
        write_json(metrics, args.report)
    if args.markdown:
        write_markdown(metrics, args.markdown)

    print(
        f"recall@k={metrics.recall_at_k:.3f}  mrr={metrics.mrr:.3f}  "
        f"refusal={metrics.refusal_recall:.3f}  leaks={metrics.leaks}  "
        f"inyecciones={metrics.injection_refused}/{metrics.injection_cases}"
    )
    return 0 if gates_passed(metrics) else 1


def _run_agent(args: argparse.Namespace) -> int:
    cases = load_agent_cases(args.dataset)
    if args.attacks:
        cases = [*cases, *BUILTIN_ATTACKS]
    catalog = tuple(item.strip() for item in args.tools.split(",") if item.strip())

    traces = None
    runner = None
    if args.traces:
        traces = load_traces(args.traces, args.format, tool_names=catalog)
    elif args.runner:
        runner = _load_runner(args.runner)
    else:
        raise SystemExit("agent requiere --runner o --traces")

    metrics = evaluate_agent(
        cases,
        runner,
        traces=traces,
        tool_catalog=catalog,
        max_steps=args.max_steps,
        canary=args.canary,
        injection_guard=is_explicit_extraction,
    )
    if args.report:
        write_agent_json(metrics, args.report)
    if args.markdown:
        write_agent_markdown(metrics, args.markdown)

    print(
        f"tool_recall={metrics.tool_recall:.3f}  "
        f"trayectoria={metrics.trajectory_similarity:.3f}  "
        f"refusal={metrics.refusal_recall:.3f}  "
        f"prohibidas={metrics.forbidden_tool_calls}  "
        f"ataques_con_exito={metrics.attacks_succeeded}/{metrics.attacks}  "
        f"fugas={metrics.leaks}"
    )
    return 0 if agent_gates_passed(metrics) else 1


if __name__ == "__main__":
    sys.exit(main())
