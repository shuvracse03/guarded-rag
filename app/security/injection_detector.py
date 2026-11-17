"""Heuristic prompt-injection detection for untrusted text.

This is a *signal*, not the security boundary. The boundary is application code: tenant/ACL
filtering before retrieval reaches the model, trust labelling, and post-generation citation checks.
"""
import re
from dataclasses import dataclass

REDACTION = "[redacted: instruction-like content]"

PATTERNS: dict[str, re.Pattern] = {
    "ignore_previous": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|system)\b.{0,30}"
        r"\b(instructions?|rules?|prompts?|policy|policies|guidelines?)\b",
        re.I,
    ),
    "fake_role_header": re.compile(r"^\s*[#>*\-\[\(]*\s*(system|assistant|developer|admin)\s*(prompt|message)?\s*[:\]\)>]", re.I | re.M),
    "chat_template_token": re.compile(r"</?\|?(im_start|im_end|system|assistant|untrusted_source|source)\b[^>\n]*>|\[/?INST\]|<<SYS>>", re.I),
    "privilege_escalation": re.compile(
        r"\b(grant|give|make|mark|treat|set)\b.{0,30}\b(admin|administrator|root|superuser|elevated|full access)\b", re.I
    ),
    "permission_override": re.compile(r"\b(bypass|disable|override|skip|waive)\b.{0,30}\b(permissions?|access control|acl|tenant|policy|filter|checks?)\b", re.I),
    "exfiltration": re.compile(
        r"\b(reveal|show|print|leak|disclose|output|dump|send|list)\b.{0,50}\b(other tenants?|another tenant|all tenants?|system prompt|api keys?|secrets?|credentials?|confidential|tickets? (of|from|for) (the )?\w+)\b",
        re.I,
    ),
    "role_hijack": re.compile(r"\b(you are now|from now on you|act as|pretend to be|new instructions?)\b", re.I),
    "concealment": re.compile(r"\b(do not|don't|never)\b.{0,20}\b(cite|mention|tell|reveal|disclose)\b", re.I),
    "hidden_chars": re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]"),
}


@dataclass(frozen=True)
class Detection:
    flags: list[str]
    flagged_lines: list[str]

    @property
    def flagged(self) -> bool:
        return bool(self.flags)


def detect(text: str) -> Detection:
    flags: list[str] = []
    lines: list[str] = []
    for name, pat in PATTERNS.items():
        if pat.search(text):
            flags.append(name)
    if flags:
        for line in text.splitlines():
            if any(p.search(line) for p in PATTERNS.values()):
                lines.append(line.strip())
    return Detection(flags=flags, flagged_lines=lines)


def redact(text: str) -> str:
    """Replace every instruction-like line with a placeholder (used for the model-facing copy)."""
    out = []
    for line in text.splitlines():
        out.append(REDACTION if any(p.search(line) for p in PATTERNS.values()) else line)
    return "\n".join(out)
