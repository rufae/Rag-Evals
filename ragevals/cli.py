"""CLI de ragevals.

Uso típico en CI (tu retriever vive en un módulo tuyo con una función que
recibe la pregunta y devuelve ``RankedResult``):

    ragevals run --dataset eval/dataset.jsonl \\
                 --retriever mi_proyecto.retriever:buscar \\
                 --top-k 5 --min-score 0.32 \\
                 --report reports/eval.json --markdown reports/eval.md

Sale con código != 0 si no se cumplen los gates → rompe el build en CI.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from ragevals.guards import is_explicit_extraction
from ragevals.metrics import EvalMetrics, gates_passed
from ragevals.report import write_json, write_markdown
from ragevals.runner import Retriever, evaluate_cases
from ragevals.types import load_cases


def _load_retriever(spec: str) -> Retriever:
    """Importa ``paquete.modulo:funcion`` como retriever."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--retriever debe tener la forma 'modulo:funcion'")
    module = importlib.import_module(module_name)
    retriever = getattr(module, attr, None)
    if not callable(retriever):
        raise SystemExit(f"{spec} no es invocable")
    return retriever  # type: ignore[return-value]


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
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
