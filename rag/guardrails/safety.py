from __future__ import annotations

import re

from pydantic import BaseModel


class SafetyDecision(BaseModel):
    allowed: bool
    reason: str | None = None


class SafetyGuard:
    """Conservative request-level safety and prompt-injection screening."""

    _unsafe_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bhow\s+to\s+(?:build|make)\s+(?:a\s+)?(?:bomb|explosive)\b",
            r"\b(?:steal|harvest)\s+(?:passwords?|credit\s+cards?)\b",
            r"\b(?:kill|hurt)\s+(?:myself|yourself|someone)\b",
            r"\bmalware\s+(?:payload|ransomware)\b",
        )
    ]
    _injection_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"ignore\s+(?:all\s+)?previous\s+instructions",
            r"reveal\s+(?:the\s+)?system\s+prompt",
            r"print\s+(?:all\s+)?(?:api\s+)?keys",
        )
    ]

    def check(self, query: str) -> SafetyDecision:
        if any(pattern.search(query) for pattern in self._unsafe_patterns):
            return SafetyDecision(allowed=False, reason="unsafe_request")
        if any(pattern.search(query) for pattern in self._injection_patterns):
            return SafetyDecision(allowed=False, reason="prompt_injection")
        return SafetyDecision(allowed=True)
