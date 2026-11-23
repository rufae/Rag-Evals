"""Guardrails de entrada y salida para pipelines LLM.

- ``is_explicit_extraction``: detecta intentos de extracción explícita
  (prompt del sistema, PII, jailbreak) en la entrada.
- ``OutputGuard``: última barrera de salida. Inspecciona el stream con un
  búfer de retención antes de emitir y bloquea fugas del token canario o del
  prompt de sistema, patrones sensibles (email, teléfono, DNI/NIE, IBAN),
  URLs fuera de la allowlist y respuestas demasiado largas.

Extraído del guard del proyecto CV-Web y generalizado (sin dependencias).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_RETENTION = 64
DEFAULT_MAX_LENGTH = 4000
DEFAULT_NGRAM_SIZE = 8

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(
    r"\b(?:\+\d{1,3}[\s.-]?)?(?:6\d{2}|7\d{2}|9\d{2})[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}\b"
)
_DNI = re.compile(r"\b[XYZ]?\d{7,8}[A-Z]\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[ ]?(?:\d{4}[ ]?){5}\b")
_URL = re.compile(r"https?://([^\s/)\]]+)", re.IGNORECASE)
_REMOTE_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(?:https?:)?//", re.IGNORECASE)
_JS_SCHEME = re.compile(r"javascript:\s*", re.IGNORECASE)
_NORMALIZE = re.compile(r"[^\wáéíóúüñ]+", re.IGNORECASE)

# Patrones de extracción explícita en la ENTRADA (jailbreak/PII/prompt).
_BLOCK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignora\s+(?:las\s+)?instrucciones",
        r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
        r"dime\s+el\s+(?:prompt|sistema)",
        r"(?:reveal|show|print)\s+(?:your|the)\s+(?:prompt|system)",
        r"confidencial|secreto",
        r"(?:password|contrase[nñ]a)\s*(?:=|:)",
        r"jailbreak",
        r"dan\s+mode",
    )
]


def is_explicit_extraction(text: str) -> bool:
    """¿El mensaje es un intento explícito de extracción/jailbreak?"""
    return any(pattern.search(text) for pattern in _BLOCK_PATTERNS)


@dataclass(frozen=True)
class GuardDecision:
    text: str
    reason: str | None = None

    @property
    def blocked(self) -> bool:
        return self.reason is not None


class OutputGuard:
    """Filtro de salida en streaming con búfer de retención."""

    def __init__(
        self,
        *,
        canary: str,
        system_prompt: str,
        allowlist: tuple[str, ...] = (),
        allowed_domains: tuple[str, ...] = (),
        retention: int = DEFAULT_RETENTION,
        max_length: int = DEFAULT_MAX_LENGTH,
        ngram_size: int = DEFAULT_NGRAM_SIZE,
    ) -> None:
        self._canary = canary
        self._retention = max(0, retention)
        self._max_length = max_length
        self._allowlist = tuple(item.strip().lower() for item in allowlist if item.strip())
        self._allowed_domains = tuple(d.strip().lower() for d in allowed_domains if d.strip())
        self._prompt_ngrams = self._make_ngrams(system_prompt, ngram_size)

    @staticmethod
    def _make_ngrams(text: str, size: int) -> set[str]:
        normalized = _NORMALIZE.sub(" ", text.lower()).split()
        return {" ".join(normalized[i : i + size]) for i in range(len(normalized) - size + 1)}

    def _violation(self, text: str) -> str | None:
        if self._canary and self._canary in text:
            return "canary"
        ngrams = self._make_ngrams(text, DEFAULT_NGRAM_SIZE)
        if self._prompt_ngrams & ngrams:
            return "system_prompt"
        lowered = text.lower()
        for pattern, name in ((_EMAIL, "email"), (_PHONE, "phone"), (_DNI, "dni"), (_IBAN, "iban")):
            if pattern.search(text) and not any(allowed in lowered for allowed in self._allowlist):
                return name
        for match in _URL.finditer(text):
            domain = match.group(1).lower()
            if self._allowed_domains and not any(domain.endswith(d) for d in self._allowed_domains):
                return "url"
        if _REMOTE_IMAGE.search(text) or _JS_SCHEME.search(text):
            return "unsafe_markdown"
        return None

    def check(self, text: str) -> GuardDecision:
        """Revisión de un fragmento completo (sin streaming)."""
        if len(text) > self._max_length:
            return GuardDecision(text="", reason="max_length")
        reason = self._violation(text)
        return GuardDecision(text="" if reason else text, reason=reason)
