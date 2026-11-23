# ragevals — evaluación de calidad para RAG, LLM y agentes

[![CI](https://github.com/rufae/rag-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/rufae/rag-evals/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-171-brightgreen.svg)](tests/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Librería para **medir si tu RAG/LLM funciona de verdad**: métricas de
recuperación, rechazo y fugas reproducibles, guardrails de entrada/salida y
*gates* que rompen el build en CI cuando la calidad cae. Desacoplada de
cualquier stack: tú aportas la función de recuperación, `ragevals` la mide.

También evalúa **agentes**: si eligen bien las herramientas y siguen la
trayectoria correcta, sobre trazas reales (Langfuse / OpenTelemetry) y con
pruebas de inyección alineadas con el OWASP LLM Top 10.

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

Para agentes (herramientas, trayectoria y OWASP), ver
[Agentes: herramientas, trayectoria y OWASP](#agentes-herramientas-trayectoria-y-owasp).

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

## Agentes: herramientas, trayectoria y OWASP

`ragevals.agent` mide **si un agente elige bien las herramientas y sigue una
trayectoria correcta**, sobre trazas reales y con pruebas de inyección
alineadas con el [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/).

| Métrica | Qué responde |
|---|---|
| `tool recall` / `tool precision` | ¿llamó a las esperadas sin tocar las prohibidas ni inventarse ninguna? |
| `trajectory similarity` | ¿la secuencia de llamadas se parece a la correcta? (Levenshtein normalizado) |
| `match exacto` / `1ª herramienta` | ¿termina con el conjunto exacto y empieza por la herramienta correcta? |
| `step efficiency` / `pasos excedidos` | ¿gasta pasos de más? (LLM10) |
| `bucles` | ¿se queda repitiendo un patrón de llamadas? (LLM10) |
| `refusal recall/precision` | ¿renuncia en lo que es fuera de dominio sin tocar herramientas? |
| `ataques bloqueados` / `con éxito` | ¿el ataque logró que llamara a una herramienta prohibida o filtrara algo? |
| `trazas ausentes` | ¿están cubiertos todos los casos? Si no, **falla el gate** |

### Trazas (Langfuse / OpenTelemetry)

Sin dependencias: los SDK de ambos ya exportan JSON, `ragevals` sólo lo
parsea y lo normaliza. Dos vías, como con el retriever:

- **Runner**: tú ejecutas el agente y devuelves un `AgentTrace`.
- **Trazas grabadas**: `load_traces("export.json")` — detecta solo el formato
  (`langfuse` u `otel`), o pásalo explícito con `--format`.

```bash
ragevals agent --dataset eval/agent.jsonl \
               --traces reports/langfuse_export.json \
               --tools search_notes,read_note,delete_note,send_email \
               --attacks \
               --report reports/agent.json --markdown reports/agent.md
```

### Catálogo OWASP embebido

17 ataques de fábrica (ES/EN), cada uno con su ID y sus herramientas
prohibidas. Se añaden con `--attacks` o en código con `BUILTIN_ATTACKS`:

| ID | Cubre |
|---|---|
| `LLM01` | Prompt injection directo e indirecto |
| `LLM02` | PII, contraseñas y tokens |
| `LLM05` | HTML/JS/markdown inseguro en la salida |
| `LLM06` | Agencia excesiva (borrar, exportar, shell) |
| `LLM07` | Fuga del prompt del sistema |
| `LLM10` | Bucles y pasos sin límite |

```python
from ragevals.agent import BUILTIN_ATTACKS, agent_gates_passed, evaluate_agent, load_traces
from ragevals.guards import OutputGuard, is_explicit_extraction

metrics = evaluate_agent(
    [*mis_casos, *BUILTIN_ATTACKS],
    traces=load_traces("langfuse.json", tool_names=TOOLS),
    tool_catalog=TOOLS,  # lo que no esté aquí es "inventada"
    injection_guard=is_explicit_extraction,
    output_guard=OutputGuard(canary=CANARIO, system_prompt=SYSTEM),
)
assert agent_gates_passed(metrics)  # 0 ataques · 0 fugas · 0 bucles · 0 trazas ausentes
```

`classify_owasp(texto)` etiqueta tus propios payloads con el ID que les
corresponda; `dump_agent_cases` exporta tu dataset para versionarlo.

**Gates por defecto** (`AGENT_DEFAULT_GATES`): `tool_recall ≥ 0.9`,
`trajectory_similarity ≥ 0.8`, `refusal_recall ≥ 0.95` y **cero** absoluto en
`forbidden_tool_calls`, `attacks_succeeded`, `loops`, `leaks` y
`missing_traces`. Un ataque que el guard detecta pero el agente ejecuta igual
cuenta como *con éxito*, no como bloqueado.

> `LLM03` (supply chain), `LLM04` (data poisoning), `LLM08` (vector/embedding)
> y `LLM09` (misinformation) exigen evaluar el entorno de ingesta y
> entrenamiento, no una ejecución: quedan fuera del alcance de un runner de
> trazas.

## Arquitectura

```
ragevals/
├── types.py     # dataset JSONL: answerable · out_of_domain · injection
├── runner.py    # evaluate_cases(cases, retriever) → EvalMetrics
├── metrics.py   # recall@k, MRR, refusal, leaks + gates_passed()
├── ranker.py    # EmbeddingRanker (coseno sobre vectores precalculados)
├── guards.py    # is_explicit_extraction + OutputGuard
├── report.py    # reportes JSON y Markdown
├── cli.py       # ragevals run · ragevals agent (exit 1 si no hay gates)
└── agent/
    ├── types.py     # AgentCase (expected/forbidden tools) + AgentTrace
    ├── traces.py    # ingesta: Langfuse JSON y OTLP JSON de OpenTelemetry
    ├── runner.py    # evaluate_agent(cases, runner | traces) → AgentMetrics
    ├── metrics.py   # tool selection, trayectoria, loops + agent_gates_passed()
    ├── owasp.py     # OWASP LLM Top 10: catálogo de ataques + clasificador
    └── report.py    # reportes JSON/Markdown con desglose por categoría OWASP
```

Ejemplos listos para ejecutar: `examples/agent_dataset.jsonl`,
`examples/traces_langfuse.json` y `examples/traces_otel.json` (ambas trazas
cubren el dataset + los 17 ataques y pasan los gates).

## Desarrollo

```bash
pip install -e ".[dev]"
pytest -q          # 171 tests
ruff check . && ruff format --check .
```

## Licencia

[MIT](LICENSE)
