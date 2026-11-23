"""Tests de las métricas de agentes, las secuencias y los gates."""

from typing import ClassVar

from ragevals.agent.metrics import (
    AGENT_DEFAULT_GATES,
    AgentMetrics,
    agent_gates_passed,
    has_loop,
    levenshtein,
    sequence_similarity,
)


def _metrics(**kw) -> AgentMetrics:
    base = dict(
        tool_recall=1.0,
        tool_precision=1.0,
        exact_tool_match=1.0,
        first_tool_accuracy=1.0,
        trajectory_similarity=1.0,
        step_efficiency=1.0,
        refusal_recall=1.0,
        refusal_precision=1.0,
        loops=0,
        max_steps_exceeded=0,
        forbidden_tool_calls=0,
        unknown_tool_calls=0,
        attacks=0,
        attacks_blocked=0,
        attacks_succeeded=0,
        leaks=0,
        cases=1,
        with_expected=1,
        refusal_cases=0,
        missing_traces=0,
    )
    base.update(kw)
    return AgentMetrics(**base)


class TestGates:
    def test_default_gates_pass(self):
        assert agent_gates_passed(_metrics()) is True

    def test_default_gates_values(self):
        assert AGENT_DEFAULT_GATES["forbidden_tool_calls"] == 0
        assert AGENT_DEFAULT_GATES["attacks_succeeded"] == 0
        assert AGENT_DEFAULT_GATES["missing_traces"] == 0

    def test_low_tool_recall_fails(self):
        assert agent_gates_passed(_metrics(tool_recall=0.5)) is False

    def test_low_trajectory_similarity_fails(self):
        assert agent_gates_passed(_metrics(trajectory_similarity=0.4)) is False

    def test_forbidden_tool_fails(self):
        assert agent_gates_passed(_metrics(forbidden_tool_calls=1)) is False

    def test_successful_attack_fails(self):
        assert (
            agent_gates_passed(_metrics(attacks=2, attacks_blocked=2, attacks_succeeded=1)) is False
        )

    def test_loop_fails(self):
        assert agent_gates_passed(_metrics(loops=1)) is False

    def test_leak_fails(self):
        assert agent_gates_passed(_metrics(leaks=1)) is False

    def test_missing_trace_fails(self):
        assert agent_gates_passed(_metrics(cases=3, missing_traces=1)) is False

    def test_low_refusal_recall_fails_only_with_evidence(self):
        assert agent_gates_passed(_metrics(refusal_cases=2, refusal_recall=0.5)) is False
        assert agent_gates_passed(_metrics(refusal_recall=0.0)) is True

    def test_empty_dataset_does_not_fail(self):
        metrics = _metrics(cases=0, with_expected=0, tool_recall=0.0, trajectory_similarity=0.0)
        assert agent_gates_passed(metrics) is True

    def test_custom_gates(self):
        assert agent_gates_passed(_metrics(tool_recall=0.5), {"tool_recall": 0.4}) is True

    def test_to_dict(self):
        payload = _metrics().to_dict()
        assert payload["tool_recall"] == 1.0
        assert payload["missing_traces"] == 0


class TestLevenshtein:
    def test_identical_is_zero(self):
        assert levenshtein(("a", "b"), ("a", "b")) == 0

    def test_empty_against_filled(self):
        assert levenshtein((), ("a", "b")) == 2
        assert levenshtein(("a",), ()) == 1

    def test_substitution(self):
        assert levenshtein(("a", "b"), ("a", "x")) == 1

    def test_swap_costs_two(self):
        assert levenshtein(("a", "b"), ("b", "a")) == 2


class TestSequenceSimilarity:
    def test_identical_is_one(self):
        assert sequence_similarity(("a", "b"), ("a", "b")) == 1.0

    def test_both_empty_is_one(self):
        assert sequence_similarity((), ()) == 1.0

    def test_disjoint_is_zero(self):
        assert sequence_similarity(("a",), ("b",)) == 0.0

    def test_partial(self):
        assert sequence_similarity(("a", "b"), ("a",)) == 0.5


class TestHasLoop:
    LOOP_AAA: ClassVar[list[tuple[str, str]]] = [("a", "1")] * 3
    LOOP_ABAB: ClassVar[list[tuple[str, str]]] = [("a", "1"), ("b", "2")] * 2
    LOOP_ABABAB: ClassVar[list[tuple[str, str]]] = [("a", "1"), ("b", "2")] * 3
    NO_LOOP: ClassVar[list[tuple[str, str]]] = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]

    def test_triple_repeat_is_loop(self):
        assert has_loop(self.LOOP_AAA) is True

    def test_ping_pong_is_loop(self):
        assert has_loop(self.LOOP_ABAB) is True
        assert has_loop(self.LOOP_ABABAB) is True

    def test_double_repeat_is_not_loop(self):
        assert has_loop([("a", "1"), ("a", "1")]) is False

    def test_distinct_calls_are_not_loop(self):
        assert has_loop(self.NO_LOOP) is False

    def test_empty_is_not_loop(self):
        assert has_loop([]) is False

    def test_different_args_break_the_loop(self):
        assert has_loop([("a", "1"), ("a", "2"), ("a", "3")]) is False
