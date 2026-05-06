from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
import uuid
from pathlib import Path
from typing import Mapping, Protocol, Sequence


REVERSIBLE_SCHEMA_VERSION = "opf.reversible.v1"


class _DetectedSpanLike(Protocol):
    label: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class TokenizedSpan:
    """One source span after reversible token assignment."""

    label: str
    start: int
    end: int
    text: str
    token: str
    token_start: int
    token_end: int


@dataclass(frozen=True)
class VaultEntry:
    """One token-to-original-value mapping entry."""

    token: str
    label: str
    text: str
    canonical_text: str
    index: int

    def to_dict(self, *, include_values: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "label": self.label,
            "canonical_text": self.canonical_text if include_values else None,
            "index": self.index,
        }
        if include_values:
            payload["text"] = self.text
        return payload


def _placeholder_base_for_label(label: str) -> str:
    """Convert a span label into a stable placeholder base."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    return normalized or "REDACTED"


def _canonicalize_text(label: str, text: str) -> str:
    """Canonicalize values for stable mapping."""
    if label == "secret":
        return text
    return re.sub(r"\s+", " ", text.strip())


class ReversibleVault:
    """In-memory reversible token vault.

    Production callers should persist this through encrypted storage instead of
    writing the serialized payload as plaintext.
    """

    def __init__(
        self,
        *,
        vault_id: str | None = None,
        created_at_unix: float | None = None,
    ) -> None:
        self.vault_id = vault_id or uuid.uuid4().hex
        self.created_at_unix = (
            float(created_at_unix) if created_at_unix is not None else time.time()
        )
        self._entries_by_token: dict[str, VaultEntry] = {}
        self._token_by_key: dict[str, str] = {}
        self._counters_by_base: dict[str, int] = {}

    @staticmethod
    def key_for(label: str, text: str) -> str:
        canonical = _canonicalize_text(label, text)
        return f"{label}\0{canonical}"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ReversibleVault:
        schema_version = payload.get("schema_version")
        if schema_version != REVERSIBLE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported reversible vault schema_version: "
                f"{schema_version!r}"
            )

        vault_id = payload.get("vault_id")
        if not isinstance(vault_id, str) or not vault_id:
            raise ValueError("Vault payload must contain a non-empty vault_id")

        created_at_unix = payload.get("created_at_unix")
        vault = cls(
            vault_id=vault_id,
            created_at_unix=(
                float(created_at_unix)
                if isinstance(created_at_unix, (int, float))
                else None
            ),
        )

        entries = payload.get("entries")
        if not isinstance(entries, Mapping):
            raise ValueError("Vault payload must contain an entries object")

        for token, raw_entry in entries.items():
            if not isinstance(token, str):
                raise ValueError("Vault entry token must be a string")
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Vault entry for {token!r} must be an object")

            label = raw_entry.get("label")
            text = raw_entry.get("text")
            canonical_text = raw_entry.get("canonical_text")
            index = raw_entry.get("index")
            if not isinstance(label, str) or not label:
                raise ValueError(f"Vault entry {token!r} missing label")
            if not isinstance(text, str):
                raise ValueError(f"Vault entry {token!r} missing text")
            if not isinstance(canonical_text, str):
                raise ValueError(f"Vault entry {token!r} missing canonical_text")
            if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
                raise ValueError(f"Vault entry {token!r} has invalid index")

            entry = VaultEntry(
                token=token,
                label=label,
                text=text,
                canonical_text=canonical_text,
                index=index,
            )
            vault._entries_by_token[token] = entry
            vault._token_by_key[f"{label}\0{canonical_text}"] = token
            base = _placeholder_base_for_label(label)
            vault._counters_by_base[base] = max(
                vault._counters_by_base.get(base, 0),
                index,
            )

        return vault

    @classmethod
    def load(cls, path: str | Path) -> ReversibleVault:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Vault file must contain a JSON object")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(include_values=True), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    def to_dict(self, *, include_values: bool = True) -> dict[str, object]:
        return {
            "schema_version": REVERSIBLE_SCHEMA_VERSION,
            "vault_id": self.vault_id,
            "created_at_unix": self.created_at_unix,
            "entries": {
                token: entry.to_dict(include_values=include_values)
                for token, entry in sorted(self._entries_by_token.items())
            },
        }

    def issue_token(
        self,
        *,
        label: str,
        text: str,
        source_text: str,
    ) -> str:
        """Return an existing or newly allocated token for label+text."""
        canonical = _canonicalize_text(label, text)
        key = f"{label}\0{canonical}"
        existing = self._token_by_key.get(key)
        if existing is not None:
            return existing

        base = _placeholder_base_for_label(label)
        next_index = self._counters_by_base.get(base, 0) + 1
        while True:
            candidate = f"<{base}_{next_index}>"
            if candidate not in source_text and candidate not in self._entries_by_token:
                break
            next_index += 1

        self._counters_by_base[base] = next_index
        entry = VaultEntry(
            token=candidate,
            label=label,
            text=text,
            canonical_text=canonical,
            index=next_index,
        )
        self._entries_by_token[candidate] = entry
        self._token_by_key[key] = candidate
        return candidate

    def original_for_token(self, token: str) -> str | None:
        entry = self._entries_by_token.get(token)
        return None if entry is None else entry.text


def _validate_non_overlapping_spans(
    spans: Sequence[_DetectedSpanLike],
) -> list[_DetectedSpanLike]:
    ordered = sorted(spans, key=lambda span: (span.start, span.end, span.label))
    cursor = 0
    valid: list[_DetectedSpanLike] = []
    for span in ordered:
        if span.end <= span.start:
            continue
        if span.start < cursor:
            raise ValueError(
                "Reversible tokenization requires non-overlapping spans. "
                f"Overlapping span: {span!r}"
            )
        valid.append(span)
        cursor = span.end
    return valid


def apply_reversible_tokenization(
    *,
    text: str,
    spans: Sequence[_DetectedSpanLike],
    vault: ReversibleVault | None = None,
) -> tuple[str, tuple[TokenizedSpan, ...], ReversibleVault]:
    """Replace detected spans with stable per-value tokens."""
    resolved_vault = vault or ReversibleVault()
    ordered = _validate_non_overlapping_spans(spans)

    pieces: list[str] = []
    tokenized_spans: list[TokenizedSpan] = []
    source_cursor = 0
    output_cursor = 0

    for span in ordered:
        prefix = text[source_cursor : span.start]
        pieces.append(prefix)
        output_cursor += len(prefix)

        token = resolved_vault.issue_token(
            label=span.label,
            text=span.text,
            source_text=text,
        )
        token_start = output_cursor
        token_end = token_start + len(token)
        pieces.append(token)
        output_cursor = token_end

        tokenized_spans.append(
            TokenizedSpan(
                label=span.label,
                start=span.start,
                end=span.end,
                text=span.text,
                token=token,
                token_start=token_start,
                token_end=token_end,
            )
        )
        source_cursor = span.end

    suffix = text[source_cursor:]
    pieces.append(suffix)
    tokenized_text = "".join(pieces)
    return tokenized_text, tuple(tokenized_spans), resolved_vault


def restore_text(
    *,
    tokenized_text: str,
    vault: ReversibleVault,
) -> str:
    """Restore tokenized text by replacing vault tokens with original values."""
    restored = tokenized_text
    for token in sorted(vault._entries_by_token, key=len, reverse=True):
        original = vault.original_for_token(token)
        if original is None:
            continue
        restored = restored.replace(token, original)
    return restored
