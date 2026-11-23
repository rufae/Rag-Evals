"""Tests del dataset (JSONL), los reportes y la CLI."""

import json
from pathlib import Path

import pytest

from ragevals.cli import main
from ragevals.metrics import EvalMetrics
from ragevals.report import render_markdown, write_json
from ragevals.types import EvalCase, load_cases


class TestDataset:
    def test_load_cases_jsonl(self, tmp_path: Path):
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            json.dumps(
                {"id": "a", "category": "answerable", "question": "q", "expected_sources": ["s1"]}
            )
            + "\n"
            + json.dumps({"id": "b", "category": "out_of_domain", "question": "q2"})
            + "\n",
            encoding="utf-8",
        )
        cases = load_cases(path)
        assert len(cases) == 2
        assert cases[0].expected_sources == ("s1",)
        assert cases[1].expected_sources == ()

    def test_case_from_dict_defaults(self):
        case = EvalCase.from_dict({"id": "x", "category": "answerable", "question": "q"})
        assert case.expected_sources == ()


class TestReport:
    METRICS = EvalMetrics(
        recall_at_k=0.95,
        mrr=0.8,
        hit_rate=0.9,
        refusal_recall=1.0,
        refusal_precision=1.0,
        leaks=0,
        answerable=10,
        refusal_cases=3,
        injection_cases=2,
        injection_refused=2,
    )

    def test_markdown_mentions_pass(self):
        assert "✅ PASS" in render_markdown(self.METRICS)
        assert "recall@k" in render_markdown(self.METRICS)

    def test_json_report(self, tmp_path: Path):
        out = tmp_path / "report.json"
        write_json(self.METRICS, out)
        payload = json.loads(out.read_text())
        assert payload["passed"] is True
        assert payload["metrics"]["recall_at_k"] == 0.95


class TestCli:
    def test_run_exit_zero_when_gates_pass(self, tmp_path: Path, monkeypatch, capsys):
        dataset = tmp_path / "d.jsonl"
        dataset.write_text(
            json.dumps(
                {"id": "a", "category": "answerable", "question": "q", "expected_sources": ["s"]}
            )
            + "\n",
            encoding="utf-8",
        )

        def fake_retriever(question):
            from ragevals.runner import RankedResult

            return RankedResult(sources=["s"], best_score=0.9, chunk_texts=[])

        import ragevals.cli as cli

        monkeypatch.setattr(cli, "_load_retriever", lambda spec: fake_retriever)
        code = main(["run", "--dataset", str(dataset), "--retriever", "x:y"])
        assert code == 0
        assert "recall@k" in capsys.readouterr().out

    def test_run_exit_one_when_gates_fail(self, tmp_path: Path, monkeypatch):
        dataset = tmp_path / "d.jsonl"
        dataset.write_text(
            json.dumps(
                {"id": "a", "category": "answerable", "question": "q", "expected_sources": ["s"]}
            )
            + "\n",
            encoding="utf-8",
        )

        def bad_retriever(question):
            from ragevals.runner import RankedResult

            return RankedResult(sources=[], best_score=0.1, chunk_texts=[])

        import ragevals.cli as cli

        monkeypatch.setattr(cli, "_load_retriever", lambda spec: bad_retriever)
        assert main(["run", "--dataset", str(dataset), "--retriever", "x:y"]) == 1

    def test_bad_retriever_spec(self):
        with pytest.raises(SystemExit):
            import ragevals.cli as cli

            cli._load_retriever("sin_dos_puntos")
