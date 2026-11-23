# ragevals — evaluación de calidad para RAG y LLM

[![CI](https://github.com/rufae/rag-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/rufae/rag-evals/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-39-brightgreen.svg)](tests/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Librería para **medir si tu RAG/LLM funciona de verdad**: métricas de
recuperación, rechazo y fugas reproducibles, guardrails de entrada/salida y
*gates* que rompen el build en CI cuando la calidad cae. Desacoplada de
cualquier stack: tú aportas la función de recuperación, `ragevals` la mide.

> Extraída y generalizada del framework de evaluación que escribí para un
> asistente RAG en producción (dataset dorado de 49 casos, `recall@5=0,983`,
> 0 fugas de PII/prompt).

## Qué mide

| Métrica | Qué responde |
|---|---|
| `recall@k` | ¿las fuentes correctas aparecen en el top-k? |
| `MRR` / `hit rate` | ¿qué tan arriba aparece la primera correcta? |
| `refusal recall/precision` | ¿se rechaza lo que debe rechazarse sin rechazar de más? |
| `leaks` (canario) | ¿se filtra contenido privado en los textos recuperados? |
| `injection blocked` | ¿se bloquean los intentos de jailbreak/extracción? |

## Uso

```bash
pip install ragevals
```

```python
from ragevals import EmbeddingRanker, evaluate_cases, gates_passed, is_explicit_extraction
from ragevals.types import load_cases

cases = load_cases("examples/dataset.jsonl")
ranker = EmbeddingRanker(chunk_matrix, source_ids, chunk_texts)

metrics = evaluate_cases(
    cases,
    ranker.retriever_for(query_vectors),
    top_k=5,
    min_score=0.32,  # calibrado: por debajo, el sistema rechaza
    canary="CANARIO_PRIVADO",
    injection_guard=is_explicit_extraction,
)
assert gates_passed(metrics)  # recall ≥ 0.9 · refusal ≥ 0.95 · 0 fugas
```

O desde la línea de comandos (ideal en CI):

```bash
ragevals run --dataset examples/dataset.jsonl \
             --retriever mi_proyecto.retriever:buscar \
             --min-score 0.32 --canary CANARIO_PRIVADO \
             --report reports/eval.json --markdown reports/eval.md
```

El comando sale con **código 1 si no se cumplen los gates** → tu pipeline se
rompe antes de que llegue a producción.

## Guardrails incluidos

- `is_explicit_extraction(text)`: detecta jailbreak/extracción de prompt/PII
  en la **entrada** (ES/EN).
- `OutputGuard`: última barrera de **salida** (streaming con búfer de
  retención): bloquea fugas del token canario o del prompt de sistema,
  emails/teléfonos/DNI/IBAN fuera de allowlist, URLs fuera de dominios
  permitidos y Markdown peligroso.

## Arquitectura

```
ragevals/
├── types.py     # dataset JSONL: answerable · out_of_domain · injection
├── runner.py    # evaluate_cases(cases, retriever) → EvalMetrics
├── metrics.py   # recall@k, MRR, refusal, leaks + gates_passed()
├── ranker.py    # EmbeddingRanker (coseno sobre vectores precalculados)
├── guards.py    # is_explicit_extraction + OutputGuard
├── report.py    # reportes JSON y Markdown
└── cli.py       # ragevals run (exit 1 si no hay gates)
```

## Desarrollo

```bash
pip install -e ".[dev]"
pytest -q          # 39 tests
ruff check .
```

## Licencia

[MIT](LICENSE)
