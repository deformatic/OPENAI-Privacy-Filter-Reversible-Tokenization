from __future__ import annotations

from dataclasses import dataclass

import pytest

from opf._core.reversible import (
    ReversibleVault,
    apply_reversible_tokenization,
    restore_text,
)


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int
    text: str
    placeholder: str = "<PRIVATE>"


def test_reversible_tokenization_reuses_same_value_token():
    text = "Alice emailed Bob. Alice called Bob."
    spans = (
        Span("private_person", 0, 5, "Alice"),
        Span("private_person", 14, 17, "Bob"),
        Span("private_person", 19, 24, "Alice"),
        Span("private_person", 32, 35, "Bob"),
    )

    vault = ReversibleVault()
    tokenized_text, tokenized_spans, resolved_vault = apply_reversible_tokenization(
        text=text,
        spans=spans,
        vault=vault,
    )

    assert tokenized_text == (
        "<PRIVATE_PERSON_1> emailed <PRIVATE_PERSON_2>. "
        "<PRIVATE_PERSON_1> called <PRIVATE_PERSON_2>."
    )
    assert resolved_vault is vault
    assert tokenized_spans[0].token == "<PRIVATE_PERSON_1>"
    assert tokenized_spans[1].token == "<PRIVATE_PERSON_2>"
    assert tokenized_spans[2].token == "<PRIVATE_PERSON_1>"
    assert tokenized_spans[3].token == "<PRIVATE_PERSON_2>"


def test_different_values_under_same_label_increment_indexes():
    text = "Alice met Bob and Carol."
    spans = (
        Span("private_person", 0, 5, "Alice"),
        Span("private_person", 10, 13, "Bob"),
        Span("private_person", 18, 23, "Carol"),
    )

    tokenized_text, tokenized_spans, _ = apply_reversible_tokenization(
        text=text,
        spans=spans,
    )

    assert tokenized_text == (
        "<PRIVATE_PERSON_1> met <PRIVATE_PERSON_2> and <PRIVATE_PERSON_3>."
    )
    assert [span.token for span in tokenized_spans] == [
        "<PRIVATE_PERSON_1>",
        "<PRIVATE_PERSON_2>",
        "<PRIVATE_PERSON_3>",
    ]


def test_different_labels_do_not_share_tokens():
    text = "Value abc is email abc."
    spans = (
        Span("secret", 6, 9, "abc"),
        Span("private_email", 19, 22, "abc"),
    )

    tokenized_text, tokenized_spans, _ = apply_reversible_tokenization(
        text=text,
        spans=spans,
    )

    assert tokenized_spans[0].token == "<SECRET_1>"
    assert tokenized_spans[1].token == "<PRIVATE_EMAIL_1>"
    assert tokenized_text == "Value <SECRET_1> is email <PRIVATE_EMAIL_1>."


def test_restore_text_from_vault():
    text = "Alice emailed Bob."
    spans = (
        Span("private_person", 0, 5, "Alice"),
        Span("private_person", 14, 17, "Bob"),
    )

    vault = ReversibleVault()
    tokenized_text, _, _ = apply_reversible_tokenization(
        text=text,
        spans=spans,
        vault=vault,
    )

    assert restore_text(tokenized_text=tokenized_text, vault=vault) == text


def test_vault_save_load_preserves_mappings_and_counters(tmp_path):
    text = "Alice emailed Bob."
    spans = (
        Span("private_person", 0, 5, "Alice"),
        Span("private_person", 14, 17, "Bob"),
    )
    vault_path = tmp_path / "vault.json"

    vault = ReversibleVault(vault_id="test-vault", created_at_unix=1.0)
    apply_reversible_tokenization(text=text, spans=spans, vault=vault)
    vault.save(vault_path)

    loaded = ReversibleVault.load(vault_path)
    tokenized_text, tokenized_spans, _ = apply_reversible_tokenization(
        text="Carol called Alice.",
        spans=(
            Span("private_person", 0, 5, "Carol"),
            Span("private_person", 13, 18, "Alice"),
        ),
        vault=loaded,
    )

    assert loaded.vault_id == "test-vault"
    assert tokenized_text == "<PRIVATE_PERSON_3> called <PRIVATE_PERSON_1>."
    assert tokenized_spans[0].token == "<PRIVATE_PERSON_3>"
    assert tokenized_spans[1].token == "<PRIVATE_PERSON_1>"


def test_source_text_token_collision_skips_to_next_index():
    text = "Literal <PRIVATE_PERSON_1> mentions Alice."
    spans = (Span("private_person", 36, 41, "Alice"),)

    tokenized_text, tokenized_spans, _ = apply_reversible_tokenization(
        text=text,
        spans=spans,
    )

    assert tokenized_spans[0].token == "<PRIVATE_PERSON_2>"
    assert tokenized_text == "Literal <PRIVATE_PERSON_1> mentions <PRIVATE_PERSON_2>."


def test_overlapping_spans_raise_value_error():
    text = "Alice Bob"
    spans = (
        Span("private_person", 0, 5, "Alice"),
        Span("private_person", 3, 9, "ce Bob"),
    )

    with pytest.raises(ValueError, match="non-overlapping"):
        apply_reversible_tokenization(text=text, spans=spans)


def test_vault_metadata_output_excludes_raw_values():
    vault = ReversibleVault(vault_id="test-vault", created_at_unix=1.0)
    vault.issue_token(label="private_person", text="Alice", source_text="Alice")

    payload = vault.to_dict(include_values=False)
    entry = payload["entries"]["<PRIVATE_PERSON_1>"]

    assert "text" not in entry
    assert entry["canonical_text"] is None
