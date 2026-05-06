from __future__ import annotations

from dataclasses import dataclass
import functools
import json
import os
from pathlib import Path
from typing import Literal, TypeVar

from ._common.checkpoint_download import ensure_default_checkpoint
from ._core.decoding import ViterbiCRFDecoder, build_sequence_decoder
from ._common.constants import DEFAULT_MODEL_ENV_VAR, OUTPUT_MODES, SCHEMA_VERSION
from ._core.runtime import (
    DetectedSpan,
    PredictionResult,
    build_detection_summary,
    load_inference_runtime,
    predict_text,
)
from ._core.reversible import (
    ReversibleVault,
    TokenizedSpan,
    apply_reversible_tokenization,
    restore_text as restore_text_from_vault,
)


class _InheritType:
    """Sentinel type used to mark inherited per-call decode options."""

    def __repr__(self) -> str:
        return "INHERIT"


INHERIT = _InheritType()
T = TypeVar("T")


@dataclass(frozen=True)
class RedactionResult:
    """Structured result returned by the public OPF redaction API."""

    schema_version: int
    summary: dict[str, object]
    text: str
    detected_spans: tuple[DetectedSpan, ...]
    redacted_text: str
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the redaction result into a JSON-serializable dictionary.

        Returns:
            A dictionary containing the schema version, summary, original text,
            detected spans, redacted text, and optional warning.
        """
        payload: dict[str, object] = {
            "schema_version": int(self.schema_version),
            "summary": dict(self.summary),
            "text": self.text,
            "detected_spans": [
                {
                    "label": span.label,
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "placeholder": span.placeholder,
                }
                for span in self.detected_spans
            ],
            "redacted_text": self.redacted_text,
        }
        if self.warning is not None:
            payload["warning"] = self.warning
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the redaction result as JSON.

        Args:
            indent: Indentation level passed to ``json.dumps``. Use ``None`` for
                compact single-line JSON.

        Returns:
            A JSON string representation of the result.

        Raises:
            TypeError: If the payload cannot be serialized as JSON.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass(frozen=True)
class ReversibleRedactionResult:
    """Structured result returned by the reversible tokenization API."""

    schema_version: int
    summary: dict[str, object]
    text: str
    detected_spans: tuple[DetectedSpan, ...]
    tokenized_spans: tuple[TokenizedSpan, ...]
    tokenized_text: str
    vault: ReversibleVault
    warning: str | None = None

    def to_dict(
        self,
        *,
        include_vault: bool = False,
        include_vault_values: bool = False,
    ) -> dict[str, object]:
        """Convert the tokenization result into a JSON-serializable dictionary.

        Raw vault values are excluded unless ``include_vault_values`` is
        explicitly enabled.
        """
        payload: dict[str, object] = {
            "schema_version": int(self.schema_version),
            "summary": dict(self.summary),
            "text": self.text,
            "detected_spans": [
                {
                    "label": span.label,
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "placeholder": span.placeholder,
                }
                for span in self.detected_spans
            ],
            "tokenized_spans": [
                {
                    "label": span.label,
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "token": span.token,
                    "token_start": span.token_start,
                    "token_end": span.token_end,
                }
                for span in self.tokenized_spans
            ],
            "tokenized_text": self.tokenized_text,
            "vault_id": self.vault.vault_id,
        }
        if self.warning is not None:
            payload["warning"] = self.warning
        if include_vault or include_vault_values:
            payload["vault"] = self.vault.to_dict(
                include_values=include_vault_values,
            )
        return payload

    def to_json(
        self,
        *,
        indent: int | None = 2,
        include_vault: bool = False,
        include_vault_values: bool = False,
    ) -> str:
        """Serialize the tokenization result as JSON."""
        return json.dumps(
            self.to_dict(
                include_vault=include_vault,
                include_vault_values=include_vault_values,
            ),
            indent=indent,
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class DecodeOptions:
    """Per-call decode overrides for :meth:`OPF.redact`.

    Any field left as ``INHERIT`` uses the corresponding decoder setting from
    the ``OPF`` instance.
    """

    decode_mode: Literal["viterbi", "argmax"] | _InheritType = INHERIT
    viterbi_calibration_path: str | os.PathLike[str] | None | _InheritType = INHERIT


@dataclass(frozen=True)
class _DecoderConfig:
    """Fully resolved internal decoder configuration."""

    decode_mode: Literal["viterbi", "argmax"] = "viterbi"
    viterbi_calibration_path: str | None = None


def resolve_checkpoint_path(model: str | os.PathLike[str] | None) -> str:
    """Resolve a checkpoint path from an explicit value, env var, or default download."""
    if model is not None:
        return str(Path(model).expanduser())
    env_value = os.environ.get(DEFAULT_MODEL_ENV_VAR)
    if env_value:
        return str(Path(env_value).expanduser())
    return ensure_default_checkpoint()


def _redact_text(text: str, spans: tuple[DetectedSpan, ...]) -> str:
    """Apply placeholder substitutions for detected spans."""
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        pieces.append(span.placeholder)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _warning_for_prediction(result: PredictionResult) -> str | None:
    """Build the optional warning message for a prediction result."""
    if not result.decoded_mismatch:
        return None
    return (
        "Input text did not exactly match tokenizer round-trip decode; spans are based on "
        "decoded token text."
    )


def _merge_decoder_config(
    base: _DecoderConfig,
    *,
    decode: DecodeOptions | None = None,
) -> _DecoderConfig:
    """Merge per-call decode overrides into the base decoder configuration."""
    decode = DecodeOptions() if decode is None else decode

    def resolve(value: T | _InheritType, default: T) -> T:
        return default if value is INHERIT else value

    resolved = _DecoderConfig(
        decode_mode=str(resolve(decode.decode_mode, base.decode_mode)),
        viterbi_calibration_path=(
            base.viterbi_calibration_path
            if decode.viterbi_calibration_path is INHERIT
            else (
                str(Path(decode.viterbi_calibration_path).expanduser())
                if decode.viterbi_calibration_path is not None
                else None
            )
        ),
    )
    if resolved.decode_mode == "argmax":
        return _DecoderConfig(decode_mode="argmax")
    return resolved


def _canonicalize_decoder_config(config: _DecoderConfig) -> _DecoderConfig:
    """Normalize decoder settings for cache-key reuse."""
    if config.decode_mode == "argmax":
        return _DecoderConfig(decode_mode="argmax")
    return config


def _decode_options_from_config(config: _DecoderConfig) -> DecodeOptions:
    """Convert an internal decoder config back into public decode options."""
    return DecodeOptions(
        decode_mode=config.decode_mode,
        viterbi_calibration_path=config.viterbi_calibration_path,
    )


class OPF:
    """Public Python API for local OPF inference and redaction."""

    def __init__(
        self,
        *,
        model: str | os.PathLike[str] | None = None,
        context_window_length: int | None = None,
        trim_whitespace: bool = True,
        device: Literal["cpu", "cuda"] = "cuda",
        output_mode: Literal["typed", "redacted"] = "typed",
        decode_mode: Literal["viterbi", "argmax"] = "viterbi",
        discard_overlapping_predicted_spans: bool = False,
        output_text_only: bool = False,
    ) -> None:
        """Create a reusable local OPF redactor.

        Args:
            model: Checkpoint directory. If omitted, resolves from
                ``OPF_CHECKPOINT`` or uses ``~/.opf/privacy_filter``.
                If the model is missing in ``~/.opf/privacy_filter``, it will
                be downloaded.
            context_window_length: Optional override for the runtime context
                window length.
            trim_whitespace: Whether to trim whitespace from detected spans.
            device: Inference device name.
            output_mode: ``"typed"`` to preserve model labels or
                ``"redacted"`` to collapse them.
            decode_mode: ``"viterbi"`` or ``"argmax"``.
            discard_overlapping_predicted_spans: Whether to discard overlapping
                predicted spans per label.
            output_text_only: Whether :meth:`redact` should return only the
                redacted text instead of a structured result.

        Raises:
            ValueError: If ``output_mode`` is unsupported.
        """
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {output_mode!r}")
        self._checkpoint = resolve_checkpoint_path(model)
        self._context_window_length = context_window_length
        self._trim_whitespace = bool(trim_whitespace)
        self._device = str(device)
        self._output_mode = str(output_mode)
        self._discard_overlapping_predicted_spans = bool(
            discard_overlapping_predicted_spans
        )
        self._output_text_only = bool(output_text_only)
        self._decoder_config = _DecoderConfig(decode_mode=str(decode_mode))
        self._runtime = None
        self._decoders: dict[_DecoderConfig, ViterbiCRFDecoder | None] = {}

    def redact(
        self,
        text: str,
        *,
        decode: DecodeOptions | None = None,
    ) -> str | RedactionResult:
        """Run redaction on one input string.

        Args:
            text: Input text to redact.
            decode: Optional per-call decode overrides.

        Returns:
            The redacted text if ``output_text_only`` is enabled, otherwise a
            :class:`RedactionResult`.

        Raises:
            ValueError: If runtime or decode configuration is invalid.
            RuntimeError: If the checkpoint cannot be loaded or requires
                unsupported interactive confirmation.
        """
        runtime, decoder = self.get_prediction_components(decode=decode)
        prediction = predict_text(runtime, text, decoder=decoder)
        redacted_text = _redact_text(prediction.text, prediction.spans)
        if self._output_text_only:
            return redacted_text
        summary = build_detection_summary(
            output_mode=runtime.output_mode,
            labels=[span.label for span in prediction.spans],
            decoded_mismatch=prediction.decoded_mismatch,
        )
        return RedactionResult(
            schema_version=SCHEMA_VERSION,
            summary=summary,
            text=prediction.text,
            detected_spans=tuple(prediction.spans),
            redacted_text=redacted_text,
            warning=_warning_for_prediction(prediction),
        )

    def tokenize(
        self,
        text: str,
        *,
        vault: ReversibleVault | None = None,
        decode: DecodeOptions | None = None,
    ) -> ReversibleRedactionResult:
        """Run reversible tokenization on one input string.

        The returned vault contains token-to-original mappings and must be
        protected as sensitive data.
        """
        runtime, decoder = self.get_prediction_components(decode=decode)
        prediction = predict_text(runtime, text, decoder=decoder)
        tokenized_text, tokenized_spans, resolved_vault = (
            apply_reversible_tokenization(
                text=prediction.text,
                spans=prediction.spans,
                vault=vault,
            )
        )
        summary = build_detection_summary(
            output_mode=runtime.output_mode,
            labels=[span.label for span in prediction.spans],
            decoded_mismatch=prediction.decoded_mismatch,
        )
        summary["reversible"] = True
        summary["vault_id"] = resolved_vault.vault_id
        return ReversibleRedactionResult(
            schema_version=SCHEMA_VERSION,
            summary=summary,
            text=prediction.text,
            detected_spans=tuple(prediction.spans),
            tokenized_spans=tuple(tokenized_spans),
            tokenized_text=tokenized_text,
            vault=resolved_vault,
            warning=_warning_for_prediction(prediction),
        )

    def set_model_path(self, model_path: str | os.PathLike[str]) -> OPF:
        """Update the checkpoint directory used by this redactor.

        Args:
            model_path: New checkpoint directory path.

        Returns:
            ``self`` for fluent chaining.
        """
        self._checkpoint = resolve_checkpoint_path(model_path)
        self._invalidate_runtime()
        return self

    def set_device(self, *, device: Literal["cpu", "cuda"]) -> OPF:
        """Update the inference device.

        Args:
            device: New device name.

        Returns:
            ``self`` for fluent chaining.
        """
        self._device = str(device)
        self._invalidate_runtime()
        return self

    def set_output_mode(self, output_mode: Literal["typed", "redacted"]) -> OPF:
        """Update the output projection mode.

        Args:
            output_mode: ``"typed"`` or ``"redacted"``.

        Returns:
            ``self`` for fluent chaining.

        Raises:
            ValueError: If ``output_mode`` is unsupported.
        """
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {output_mode!r}")
        self._output_mode = str(output_mode)
        self._invalidate_runtime()
        return self

    def set_decode_mode(self, decode_mode: Literal["viterbi", "argmax"]) -> OPF:
        """Update the default decode mode for this redactor.

        Args:
            decode_mode: ``"viterbi"`` or ``"argmax"``.

        Returns:
            ``self`` for fluent chaining.
        """
        self._decoder_config = _DecoderConfig(
            decode_mode=str(decode_mode),
            viterbi_calibration_path=self._decoder_config.viterbi_calibration_path,
        )
        self._invalidate_decoder()
        return self

    def set_viterbi_decoder(
        self,
        *,
        calibration_path: str | os.PathLike[str] | None = None,
    ) -> OPF:
        """Configure the default Viterbi decoder settings for this redactor.

        Args:
            calibration_path: Optional path to a Viterbi calibration artifact.

        Returns:
            ``self`` for fluent chaining.
        """
        self._decoder_config = _DecoderConfig(
            decode_mode="viterbi",
            viterbi_calibration_path=(
                str(Path(calibration_path).expanduser())
                if calibration_path is not None
                else None
            ),
        )
        self._invalidate_decoder()
        return self

    def trim_whitespace(self, trim: bool = True) -> OPF:
        """Control whitespace trimming on detected character spans.

        Args:
            trim: Whether to trim leading and trailing span whitespace.

        Returns:
            ``self`` for fluent chaining.
        """
        self._trim_whitespace = bool(trim)
        self._invalidate_runtime()
        return self

    def output_text_only(self, text_only: bool = True) -> OPF:
        """Control whether :meth:`redact` returns only text or a full result.

        Args:
            text_only: Whether to return only the redacted text.

        Returns:
            ``self`` for fluent chaining.
        """
        self._output_text_only = bool(text_only)
        return self

    def _invalidate_runtime(self) -> None:
        """Drop the cached runtime and any cached decoders."""
        self._runtime = None
        self._decoders = {}

    def _invalidate_decoder(self) -> None:
        """Drop cached decoders while keeping the runtime."""
        self._decoders = {}

    def get_runtime(self):
        """Build or reuse the internal inference runtime for this redactor.

        Returns:
            The cached internal runtime object used for prediction.

        Raises:
            ValueError: If runtime configuration is invalid.
            RuntimeError: If the checkpoint cannot be loaded.
        """
        if self._runtime is None:
            self._runtime = load_inference_runtime(
                checkpoint=self._checkpoint,
                device_name=self._device,
                n_ctx_override=self._context_window_length,
                trim_span_whitespace=self._trim_whitespace,
                discard_overlapping_predicted_spans=self._discard_overlapping_predicted_spans,
                output_mode=self._output_mode,
            )
        return self._runtime

    def _resolve_effective_decoder_config(
        self, *, decode: DecodeOptions | None = None
    ) -> _DecoderConfig:
        """Resolve the effective decoder configuration for one call."""
        return _canonicalize_decoder_config(
            _merge_decoder_config(self._decoder_config, decode=decode)
        )

    def resolve_decode_options(
        self, *, decode: DecodeOptions | None = None
    ) -> DecodeOptions:
        """Resolve effective decode settings for one prediction call.

        Args:
            decode: Optional per-call overrides.

        Returns:
            A fully resolved :class:`DecodeOptions` object with inherited values
            replaced by the instance defaults.
        """
        merged = _merge_decoder_config(self._decoder_config, decode=decode)
        return _decode_options_from_config(merged)

    def get_prediction_components(self, *, decode: DecodeOptions | None = None):
        """Build or reuse the prediction runtime and decoder pair.

        Args:
            decode: Optional per-call decode overrides.

        Returns:
            A ``(runtime, decoder)`` pair used by the internal prediction path.

        Raises:
            ValueError: If runtime or decode configuration is invalid.
            RuntimeError: If the checkpoint cannot be loaded.
        """
        runtime = self.get_runtime()
        decoder_config = self._resolve_effective_decoder_config(decode=decode)
        decoder = self._get_decoder(runtime, decoder_config)
        return runtime, decoder

    def _get_decoder(self, runtime, decoder_config: _DecoderConfig):
        """Build or reuse the cached decoder for one effective configuration."""
        if decoder_config not in self._decoders:
            decoder, _ = build_sequence_decoder(
                decode_mode=decoder_config.decode_mode,
                label_info=runtime.label_info,
                viterbi_calibration_path=decoder_config.viterbi_calibration_path,
                checkpoint_dir=runtime.checkpoint,
            )
            self._decoders[decoder_config] = decoder
        return self._decoders[decoder_config]


@functools.lru_cache(maxsize=1)
def _default_redactor() -> OPF:
    """Return the cached default text-only redactor used by module-level redact."""
    return OPF(output_text_only=True)


def redact(text: str) -> str:
    """Redact one text string with the cached default OPF redactor.

    Args:
        text: Input text to redact.

    Returns:
        The redacted text only.

    Raises:
        ValueError: If model configuration or decode settings are invalid.
        RuntimeError: If the local checkpoint cannot be loaded.
    """
    return str(_default_redactor().redact(text))


def restore(tokenized_text: str, vault: ReversibleVault) -> str:
    """Restore text from a reversible vault."""
    return restore_text_from_vault(tokenized_text=tokenized_text, vault=vault)
