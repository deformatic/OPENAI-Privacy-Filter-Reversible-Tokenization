# OPENAI Privacy Filter: Reversible Tokenization Layer

This repository extends OpenAI Privacy Filter (OPF) with an opt-in
reversible tokenization layer.

OPF already detects privacy spans and replaces them with typed placeholders
such as `<PRIVATE_PERSON>` and `<PRIVATE_EMAIL>`. That is the right default
for irreversible redaction, but some production workflows need a stronger
middle ground: send privacy-preserving text to downstream systems, keep stable
entity references across a document or batch, and restore the original values
only inside an authorized boundary.

This project adds that middle layer.

```text
Alice emailed Bob. Alice's phone is 555-1111.
```

becomes:

```text
<PRIVATE_PERSON_1> emailed <PRIVATE_PERSON_2>. <PRIVATE_PERSON_1>'s phone is <PRIVATE_PHONE_1>.
```

with a separate vault:

```json
{
  "schema_version": "opf.reversible.v1",
  "vault_id": "7c1d...",
  "entries": {
    "<PRIVATE_PERSON_1>": {
      "label": "private_person",
      "text": "Alice",
      "canonical_text": "Alice",
      "index": 1
    }
  }
}
```

The important distinction: this is not anonymization. It is recoverable
pseudonymization. The tokenized text is useful only if the vault is protected
like source PII.

## Why This Exists

Plain redaction removes sensitive values, but it also destroys relationships
that many workflows still need:

- A reviewer may need to see that the same person appears multiple times.
- A downstream LLM task may need consistent placeholders for names, emails,
  phones, account numbers, or secrets.
- A data pipeline may need to restore original values after enrichment,
  approval, or internal processing.
- A service boundary may allow tokenized text to leave a secure enclave while
  requiring the token vault to stay inside it.

OPF already provides the hard part: local span detection with labels, offsets,
and original text. This repository keeps that model path intact and adds a
small reversible layer above the API.

## Design Goals

- Backward compatible: existing `redact()` behavior is unchanged.
- Explicit opt-in: reversible behavior is only used through `OPF.tokenize()`
  or `opf --recoverable`.
- Model agnostic: no checkpoint, decoder, Viterbi, training, or evaluation
  path is modified.
- Stable per value: the same `label + canonical_text` maps to the same token
  within a vault.
- Batch friendly: one vault can be reused across multiple inputs.
- Auditable: token mappings are serialized in a clear schema for development
  and testing.
- Security aware: the README and output schema call out that plaintext vaults
  are development-grade only.

## Architecture

The original OPF path remains:

```text
input text
  -> OPF model
  -> detected_spans
  -> typed placeholder replacement
  -> redacted_text
```

The reversible path adds one layer after detection:

```text
input text
  -> OPF model
  -> detected_spans
  -> Reversible Tokenization Layer
       - canonicalize label + value
       - reuse an existing token when possible
       - allocate the next token index for new values
       - write token -> original value into a vault
  -> tokenized_text
```

Restore is intentionally separate:

```text
tokenized_text + authorized vault access
  -> restore
  -> original-like text
```

At a service boundary this usually becomes:

```text
[Client]
   |
   v
[PII Tokenization API]
   |
   +-- OPF Detector
   |
   +-- Token Resolver
   |     - key: label + canonical_text
   |     - same value -> same token
   |     - new value -> next index
   |
   +-- Vault Writer
   |     - token -> original value
   |     - encrypted at rest in production
   |
   v
[tokenized_text returned downstream]

[Authorized restore request]
   |
   v
[Restore API]
   |
   +-- Vault Reader
   +-- token replacement
   |
   v
[restored_text]
```

## What Changed

The reversible implementation is deliberately small:

- `opf/_core/reversible.py`
  - `ReversibleVault`
  - `VaultEntry`
  - `TokenizedSpan`
  - `apply_reversible_tokenization()`
  - `restore_text()`

- `opf/_api.py`
  - `OPF.tokenize()`
  - `ReversibleRedactionResult`
  - module-level `restore()`

- `opf/__main__.py`
  - `--recoverable`
  - `--vault-in`
  - `--vault-out`
  - `--include-vault-values`

- `tests/test_reversible.py`
  - stable token reuse
  - cross-label isolation
  - vault save/load
  - restore
  - token collision avoidance
  - overlapping span rejection

## Token Assignment Rules

Tokens are generated from the detected span label:

```text
private_person -> <PRIVATE_PERSON_1>
private_email  -> <PRIVATE_EMAIL_1>
secret         -> <SECRET_1>
```

Within one vault:

- same label + same canonical text -> same token
- same label + different canonical text -> next index
- different label + same text -> different token family
- source text collision -> skip to the next available index
- overlapping spans -> raise `ValueError`

Examples:

```text
Alice emailed Bob. Alice called Bob.
```

```text
<PRIVATE_PERSON_1> emailed <PRIVATE_PERSON_2>. <PRIVATE_PERSON_1> called <PRIVATE_PERSON_2>.
```

```text
Value abc is email abc.
```

```text
Value <SECRET_1> is email <PRIVATE_EMAIL_1>.
```

Whitespace is normalized for most labels when building the stable key. The
`secret` label preserves exact text because small formatting differences may be
semantically meaningful for credentials, keys, and tokens.

## Security Model

Treat the vault as sensitive data.

Tokenized text without the vault should not be enough to recover original
values. Tokenized text with the vault is effectively the source PII.

The included JSON vault is intentionally simple so developers can inspect and
test the behavior locally. A production deployment should not store vaults as
plaintext JSON. At minimum, use:

- envelope encryption with KMS or an equivalent key-management boundary
- strict access controls separate from downstream tokenized-text consumers
- audit logs for every read, write, restore, and export
- TTL and deletion policies
- tenant or workspace scoping
- monitoring for vault exfiltration and unusual restore volume

For highly sensitive categories such as `secret`, decide whether recovery is
actually required. This implementation stores all detected spans by default
because it is a minimal reversible layer, but a production service may want a
recoverable-label allowlist or denylist.

## Quick Start

Install locally:

```bash
pip install -e .
```

By default, `opf` looks for a model at `OPF_CHECKPOINT` or
`~/.opf/privacy_filter`. If a model is missing from the default location, OPF
will download it.

Run ordinary redaction:

```bash
opf --device cpu "Alice was born on 1990-01-02."
```

Run reversible tokenization and write a vault:

```bash
opf --device cpu \
  --recoverable \
  --vault-out vault.json \
  --format json \
  "Alice emailed Bob. Alice's phone is 555-1111."
```

Print only tokenized text:

```bash
opf --device cpu \
  --recoverable \
  --format text \
  "Alice emailed Bob."
```

Reuse a vault across inputs:

```bash
opf --device cpu \
  --recoverable \
  --vault-in vault.json \
  --vault-out vault.json \
  --format json \
  "Bob called Alice."
```

Include raw vault values in JSON stdout only when you explicitly need it:

```bash
opf --device cpu \
  --recoverable \
  --include-vault-values \
  --format json \
  "Alice emailed Bob."
```

Prefer `--vault-out` over printing vault values to stdout, because stdout often
ends up in logs.

## Python API

```python
from opf import OPF, ReversibleVault, restore

redactor = OPF(device="cpu")
vault = ReversibleVault()

result = redactor.tokenize(
    "Alice emailed Bob. Alice's phone is 555-1111.",
    vault=vault,
)

print(result.tokenized_text)
# <PRIVATE_PERSON_1> emailed <PRIVATE_PERSON_2>. <PRIVATE_PERSON_1>'s phone is <PRIVATE_PHONE_1>.

restored = restore(result.tokenized_text, vault)
assert restored == "Alice emailed Bob. Alice's phone is 555-1111."
```

Use the same vault for stable cross-document tokens:

```python
vault = ReversibleVault()

r1 = redactor.tokenize("Alice emailed Bob.", vault=vault)
r2 = redactor.tokenize("Bob called Alice.", vault=vault)

print(r1.tokenized_text)
# <PRIVATE_PERSON_1> emailed <PRIVATE_PERSON_2>.

print(r2.tokenized_text)
# <PRIVATE_PERSON_2> called <PRIVATE_PERSON_1>.
```

Serialize and reload the vault:

```python
vault.save("vault.json")
vault = ReversibleVault.load("vault.json")
```

By default, result serialization does not include raw vault values:

```python
safe_payload = result.to_dict()
```

To include vault values intentionally:

```python
unsafe_payload = result.to_dict(include_vault_values=True)
```

## JSON Output Shape

Recoverable CLI/API JSON includes both detected spans and tokenized spans:

```json
{
  "schema_version": 1,
  "summary": {
    "output_mode": "typed",
    "span_count": 2,
    "by_label": {
      "private_person": 2
    },
    "decoded_mismatch": false,
    "reversible": true,
    "vault_id": "7c1d..."
  },
  "text": "Alice emailed Bob.",
  "detected_spans": [
    {
      "label": "private_person",
      "start": 0,
      "end": 5,
      "text": "Alice",
      "placeholder": "<PRIVATE_PERSON>"
    }
  ],
  "tokenized_spans": [
    {
      "label": "private_person",
      "start": 0,
      "end": 5,
      "text": "Alice",
      "token": "<PRIVATE_PERSON_1>",
      "token_start": 0,
      "token_end": 18
    }
  ],
  "tokenized_text": "<PRIVATE_PERSON_1> emailed <PRIVATE_PERSON_2>.",
  "vault_id": "7c1d..."
}
```

See `OUTPUT_SCHEMAS.md` for the full schema notes.

## Testing

Compile the package:

```bash
python3 -m compileall -q opf tests
```

Run reversible tokenization tests:

```bash
python3 -m pytest tests/test_reversible.py
```

If your system Python does not include project dependencies, install the package
and test dependencies in a virtual environment first.

## Repository Layout

- `opf/__main__.py`: unified CLI entrypoint for redact, eval, and train modes
- `opf/_api.py`: Python-facing API over runtime, redaction, and tokenization
- `opf/_core/reversible.py`: reversible tokenization and vault implementation
- `opf/_core/`: runtime loading, span conversion, and shared decoding logic
- `opf/_cli/`: command-line argument parsing and terminal rendering helpers
- `opf/_eval/`: dataset loading, preprocessing, metrics, and evaluation runners
- `opf/_train/`: local finetuning argument parsing and training runners
- `opf/_model/`: transformer implementation, checkpoint config, and weight loading
- `tests/test_reversible.py`: reversible tokenization unit tests
- `OUTPUT_SCHEMAS.md`: JSON response and export payload formats
- `FINETUNING.md`: focused finetuning workflow and demo-script guide
- `EVAL_AND_OUTPUT_MODES.md`: output mode documentation

## Model Details

Privacy Filter is a bidirectional token classification model with span
decoding. It is trained in phases, beginning with autoregressive pretraining.
The pretrained language model is then modified and post-trained as a
bidirectional banded-attention token classifier. Instead of generating text
token-by-token, it labels an input sequence in one forward pass and decodes
coherent spans with constrained Viterbi decoding.

Architecturally, the model implementation in this repository is a pre-norm
transformer encoder-style stack with:

- token embeddings
- 8 repeated transformer blocks
- grouped-query attention with rotary positional embeddings
- sparse mixture-of-experts feed-forward blocks
- a final token-classification head over privacy labels

Privacy Filter can detect these privacy span categories:

1. `account_number`
2. `private_address`
3. `private_email`
4. `private_person`
5. `private_phone`
6. `private_url`
7. `private_date`
8. `secret`

Each non-background span category is expanded into BIOES token classes:
`B-<label>`, `I-<label>`, `E-<label>`, `S-<label>`, plus the background class
`O`. The output head emits logits for these token-level classes, and the
decoder converts token labels into character spans.

## Sequence Decoding

After the token classifier produces per-token logits, OPF decodes labels with a
constrained Viterbi decoder instead of taking an independent argmax for each
token. The decoder enforces allowed BIOES boundary transitions and scores
complete label paths with start, transition, and end terms.

This improves span coherence and boundary stability, especially in noisy or
mixed-format text where local token decisions alone can produce fragmented or
inconsistent spans.

## Upstream Resources

- License: `Apache-2.0`
- Model weights: https://huggingface.co/openai/privacy-filter
- Demo: https://huggingface.co/spaces/openai/privacy-filter
- Model card: https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf

## Risks and Limitations

Privacy Filter is a redaction and data-minimization aid, not a complete
anonymization, compliance, or safety guarantee.

Known limitations include:

- missed spans for uncommon names, regional naming patterns, domain-specific
  identifiers, or novel secret formats
- over-redaction of public entities, organizations, locations, common nouns,
  placeholders, hashes, or synthetic examples
- fragmented or shifted boundaries in long, noisy, heavily punctuated, or
  layout-heavy text
- policy mismatch when an organization needs different privacy boundaries than
  the base label taxonomy provides
- possible performance drops on non-English text, non-Latin scripts, and
  domains far from the training distribution

Reversible tokenization adds a separate risk: vault exposure. A leaked vault can
restore original values from tokenized text. Production systems should isolate
vault access from downstream text processing and monitor restore operations.

## Recommended Production Pattern

Use this repository as a local implementation reference or prototype. For a
production service, split responsibilities:

- Detection service: runs OPF locally and emits spans.
- Token resolver: owns token allocation and stable value lookup.
- Vault service: encrypts and stores token-to-original mappings.
- Restore service: performs authorized reconstruction only.
- Downstream processors: receive tokenized text but never receive vault access.

That separation keeps the model path simple and makes the vault the explicit
security boundary.
