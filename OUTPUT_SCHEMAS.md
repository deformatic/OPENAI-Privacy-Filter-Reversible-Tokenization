# OPF Output Schemas

## 1. `opf` / `opf redact` JSON Output

Printed to stdout once per input example.

```json
{
  "schema_version": 1,
  "summary": {
    "output_mode": "typed",
    "span_count": 3,
    "by_label": {
      "private_person": 1,
      "private_date": 2
    },
    "decoded_mismatch": false
  },
  "text": "Alice was born on 1990-01-02.",
  "detected_spans": [
    {
      "label": "private_person",
      "start": 0,
      "end": 5,
      "text": "Alice",
      "placeholder": "<PRIVATE_PERSON>"
    }
  ],
  "redacted_text": "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."
}
```

Notes:

- In `--output-mode redacted`, every `detected_spans[*].label` becomes `redacted`.
- `warning` is present only when tokenizer decode does not exactly round-trip the input text.

## 2. `opf` / `opf redact --recoverable` JSON Output

Printed to stdout once per input example when `--recoverable --format json` is
enabled. Raw vault values are not included unless `--include-vault-values` is
explicitly provided.

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

When `--include-vault-values` is provided, the payload also includes a `vault`
object:

```json
{
  "schema_version": "opf.reversible.v1",
  "vault_id": "7c1d...",
  "created_at_unix": 1760000000.0,
  "entries": {
    "<PRIVATE_PERSON_1>": {
      "label": "private_person",
      "canonical_text": "Alice",
      "index": 1,
      "text": "Alice"
    }
  }
}
```

Security note: a reversible vault is a plaintext source of original PII when
stored as JSON. Production deployments should store vault payloads only behind
appropriate encryption, access controls, audit logging, and retention limits.

## 3. `opf eval` Predictions Output (`--predictions-out`)

Written as JSONL when requested.

```json
{
  "example_id": "stable-id",
  "text": "Alice was born on 1990-01-02.",
  "predicted_spans": {
    "private_person: Alice": [[0, 5]]
  }
}
```

Optional field:

- `token_logprobs_topk`: included only when `--predictions-token-logprobs-topk > 0`

Notes:

- This file is literal JSONL: one compact JSON object per line.

## Stability Notes

- `typed` and `untyped` are the evaluation terms.
- `typed` and `redacted` are the prediction-output terms.
- Additive fields may appear over time, but existing keys should remain stable unless `schema_version` changes for API/CLI JSON payloads.
