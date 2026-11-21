"""Selective high-accuracy Groq ASR pass for segments flagged by proofreading."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

from .audio_chunking import _decode_mono_16k, _write_flac
from .groq_transcription import _load_client, _request_transcription, _value
from .models import format_timestamp, timestamp_url


class ASRReviewError(RuntimeError):
    """Raised when selective ASR review cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class ASRReviewArtifact:
    input_path: Path
    json_path: Path
    markdown_path: Path
    audit_path: Path
    model: str
    candidate_count: int
    replacement_count: int
    audio_seconds: float
    billed_audio_seconds_estimate: float

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in ("input_path", "json_path", "markdown_path", "audit_path"):
            value[key] = str(value[key])
        return value


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ASRReviewError(f"cannot read proofreading JSON: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
        raise ASRReviewError("invalid proofreading JSON")
    return value


def _safe_candidate(candidate: str, original: str) -> bool:
    candidate = candidate.strip()
    if not candidate:
        return False
    original_length = max(1, len(original.strip()))
    ratio = len(candidate) / original_length
    if ratio < 0.45 or ratio > 2.2:
        return False
    words = candidate.split()
    if len(words) >= 8 and len(set(words)) <= max(2, len(words) // 4):
        return False
    return True


def _verified_term_hits(text: str, terms: list[str]) -> int:
    folded = text.casefold()
    return sum(1 for term in terms if term and term.casefold() in folded)


def _canonicalize_candidate_terms(text: str, terms: list[str]) -> str:
    result = text
    for term in sorted(terms, key=len, reverse=True):
        parts = [re.escape(part) for part in re.split(r"[\s_-]+", term) if part]
        if not parts:
            continue
        pattern = r"(?<![A-Za-z0-9])" + r"[\s_-]*".join(parts) + r"(?![A-Za-z0-9])"
        result = re.sub(pattern, term, result, flags=re.IGNORECASE)
    return result


def _patch_verified_term_differences(
    prior: str, candidate: str, terms: list[str]
) -> tuple[str, list[str]]:
    """Apply only candidate diff operations that overlap a new verified term."""
    if re.search(r"(?:https?://|www\.|\b[\w-]+\.[a-z]{2,}(?:/|\b))", prior, re.IGNORECASE):
        return prior, []
    new_terms = [term for term in terms if term in candidate and term not in prior]
    if not new_terms:
        return prior, []
    term_spans: list[tuple[int, int, str]] = []
    candidate_folded = candidate.casefold()
    for term in sorted(new_terms, key=len, reverse=True):
        start = candidate_folded.find(term.casefold())
        while start >= 0:
            term_spans.append((start, start + len(term), term))
            start = candidate_folded.find(term.casefold(), start + 1)
    opcodes = SequenceMatcher(None, prior, candidate, autojunk=False).get_opcodes()
    replacements: list[tuple[int, int, str]] = []
    applied: set[str] = set()
    used_candidate_spans: list[tuple[int, int]] = []
    for start, end, term in sorted(term_spans, key=lambda value: len(value[2]), reverse=True):
        if any(start < used_end and end > used_start for used_start, used_end in used_candidate_spans):
            continue
        mapped_ranges: list[tuple[int, int, int]] = []
        for index, (tag, i1, i2, j1, j2) in enumerate(opcodes):
            if tag == "equal" and j1 < end and j2 > start:
                overlap_start = max(start, j1)
                overlap_end = min(end, j2)
                mapped_ranges.append(
                    (
                        index,
                        i1 + overlap_start - j1,
                        i1 + overlap_end - j1,
                    )
                )
            elif tag != "equal" and (
                (j1 < end and j2 > start)
                or (j1 == j2 and start <= j1 <= end)
            ):
                mapped_ranges.append((index, i1, i2))
        if not mapped_ranges:
            continue
        source_start = min(i1 for _, i1, _ in mapped_ranges)
        source_end = max(i2 for _, _, i2 in mapped_ranges)
        replacement = candidate[start:end]

        # A one-character equal island can split one phonetic correction:
        # ``3の瀬`` -> ``このSan Jose``. Include its preceding edit so the
        # stale fragment does not survive as ``3のSan Jose``.
        first_index = min(index for index, _, _ in mapped_ranges)
        if first_index >= 2:
            previous_equal = opcodes[first_index - 1]
            previous_edit = opcodes[first_index - 2]
            if (
                previous_equal[0] == "equal"
                and previous_equal[2] - previous_equal[1] <= 1
                and previous_edit[0] != "equal"
            ):
                source_start = previous_edit[1]
                replacement = candidate[previous_edit[3]:end]

        # Replace the complete source token/phrase that aligns to the verified
        # term, but keep surrounding punctuation from the conservative text.
        while source_start < source_end and prior[source_start] in " \t\r\n、，,.":
            source_start += 1
        while source_end > source_start and prior[source_end - 1] in " \t\r\n、，,.":
            source_end -= 1
        if source_start >= source_end:
            continue
        replacements.append((source_start, source_end, replacement))
        used_candidate_spans.append((start, end))
        applied.add(term)

    # A shorter verified term can map into the same source phrase as a longer
    # one. Keep the longest patch and discard overlapping alternatives.
    replacements.sort(key=lambda value: (value[0], -(value[1] - value[0])))
    non_overlapping: list[tuple[int, int, str]] = []
    for replacement in replacements:
        if non_overlapping and replacement[0] < non_overlapping[-1][1]:
            continue
        non_overlapping.append(replacement)
    selected = prior
    for i1, i2, replacement in reversed(non_overlapping):
        selected = selected[:i1] + replacement + selected[i2:]
    return selected, sorted(applied)


def review_flagged_segments_groq(
    audio_file: str | Path,
    proofread_file: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "whisper-large-v3",
    language: str = "ja",
    terms: Iterable[str] = (),
    api_key: str | None = None,
    client: Any | None = None,
    request_interval_seconds: float = 3.1,
    sleep_func: Callable[[float], None] = time.sleep,
) -> ASRReviewArtifact:
    """Re-transcribe only uncertain timestamp ranges with full Whisper Large V3.

    The full-model text replaces a flagged segment only after deterministic
    length/repetition checks. The Turbo, proofread, and full-V3 versions are all
    retained in the audit file so no evidence is lost.
    """
    audio_path = Path(audio_file).resolve()
    input_path = Path(proofread_file).resolve()
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")
    if request_interval_seconds < 0:
        raise ValueError("request_interval_seconds must not be negative")
    payload = _load_payload(input_path)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reviewed = [dict(item) for item in payload["segments"]]
    flagged = [
        item
        for item in reviewed
        if bool(item.get("needs_review")) or str(item.get("confidence")) == "low"
    ]
    samples = None
    sample_rate = 16_000
    groq_client = None
    if flagged:
        samples, sample_rate = _decode_mono_16k(audio_path)
        groq_client = client or _load_client(api_key)
    prompt_terms = [str(term).strip() for term in terms if str(term).strip()]
    prompt = (", ".join(prompt_terms)[:800] or None)
    audit_items: list[dict[str, Any]] = []
    replacements = 0
    audio_seconds = 0.0
    by_id = {int(item["id"]): item for item in reviewed}

    for index, item in enumerate(flagged):
        if samples is None or groq_client is None:
            raise ASRReviewError("selective ASR review runtime was not initialized")
        start = max(0.0, float(item["start_seconds"]))
        end = min(samples.shape[1] / sample_rate, float(item["end_seconds"]))
        if end <= start:
            continue
        clip_path = destination / "clips" / f"segment-{int(item['id']):04d}.flac"
        _write_flac(
            clip_path,
            samples[:, round(start * sample_rate) : round(end * sample_rate)],
            sample_rate,
        )
        try:
            response = _request_transcription(
                groq_client,
                clip_path,
                model_name=model_name,
                language=language,
                prompt=prompt,
            )
        except Exception as exc:
            raise ASRReviewError(
                f"Groq selective review failed for segment {item['id']}: {exc}"
            ) from exc
        candidate_raw = str(_value(response, "text", "")).strip()
        candidate = _canonicalize_candidate_terms(candidate_raw, prompt_terms)
        prior = str(item.get("corrected_text") or item.get("asr_text") or "").strip()
        candidate_hits = _verified_term_hits(candidate, prompt_terms)
        prior_hits = _verified_term_hits(prior, prompt_terms)
        candidate_is_safe = _safe_candidate(candidate, prior)
        # Full V3 is additional evidence, not ground truth. Only promote it when
        # it adds an exact spelling from the verified glossary without failing
        # the information-loss guards. Otherwise retain the conservative
        # DeepSeek proofreading result and keep V3 only in the audit trail.
        selected, applied_terms = (
            _patch_verified_term_differences(prior, candidate, prompt_terms)
            if candidate_is_safe
            else (prior, [])
        )
        selected_reason = (
            "patched_verified_term_spellings"
            if applied_terms
            else "kept_proofread_no_verified_improvement"
            if candidate_is_safe
            else "kept_proofread_candidate_failed_safety_check"
        )
        replaced = selected != prior
        if replaced:
            replacements += 1
            target = by_id[int(item["id"])]
            target["corrected_text"] = selected
            target["changed"] = selected != str(target.get("asr_text") or "")
        audit_items.append(
            {
                "segment_id": int(item["id"]),
                "start_seconds": start,
                "end_seconds": end,
                "turbo_text": str(item.get("asr_text") or ""),
                "proofread_text": prior,
                "large_v3_text": candidate,
                "large_v3_raw_text": candidate_raw,
                "selected_text": selected,
                "replaced": replaced,
                "prior_verified_term_hits": prior_hits,
                "candidate_verified_term_hits": candidate_hits,
                "selected_reason": selected_reason,
                "applied_verified_terms": applied_terms,
                "original_note": str(item.get("note") or ""),
            }
        )
        audio_seconds += end - start
        if request_interval_seconds and index + 1 < len(flagged):
            sleep_func(request_interval_seconds)

    enhanced = dict(payload)
    enhanced["input_proofread"] = str(input_path)
    enhanced["selective_asr_review"] = {
        "provider": "groq",
        "model": model_name,
        "language": language,
        "candidate_count": len(audit_items),
        "replacement_count": replacements,
        "audio_seconds": audio_seconds,
        "billed_audio_seconds_estimate": sum(
            max(10.0, item["end_seconds"] - item["start_seconds"])
            for item in audit_items
        ),
        "preferred_terms": prompt_terms,
    }
    enhanced["segments"] = reviewed
    json_path = destination / "proofread_reviewed.json"
    markdown_path = destination / "proofread_reviewed.md"
    audit_path = destination / "asr_review.json"
    json_path.write_text(json.dumps(enhanced, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {"source_url": payload.get("source_url", ""), "model": model_name, "segments": audit_items},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    source_url = str(payload.get("source_url") or "")
    lines = [f"# {payload.get('title') or input_path.stem} — Selective ASR Review", ""]
    lines.append(f"- Review model: `{model_name}`")
    lines.append(f"- Reviewed segments: `{len(audit_items)}`")
    lines.append(f"- Replacements: `{replacements}`")
    lines.append("")
    for item in reviewed:
        label = format_timestamp(float(item["start_seconds"]))
        if source_url:
            label = f"[{label}]({timestamp_url(source_url, float(item['start_seconds']))})"
        review = " ⚠️ needs review" if item.get("needs_review") else ""
        lines.append(f"- {label} {item.get('corrected_text', '')}{review}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    billed = sum(max(10.0, item["end_seconds"] - item["start_seconds"]) for item in audit_items)
    return ASRReviewArtifact(
        input_path=input_path,
        json_path=json_path,
        markdown_path=markdown_path,
        audit_path=audit_path,
        model=model_name,
        candidate_count=len(audit_items),
        replacement_count=replacements,
        audio_seconds=audio_seconds,
        billed_audio_seconds_estimate=billed,
    )


def reselect_review_from_audit(
    proofread_file: str | Path,
    audit_file: str | Path,
    output_dir: str | Path,
    *,
    terms: Iterable[str] = (),
) -> ASRReviewArtifact:
    """Reapply deterministic selection to an existing Groq review audit.

    This intentionally performs no network calls. It is useful when selection
    policy or a verified glossary changes after expensive full-V3 candidates
    have already been collected.
    """
    input_path = Path(proofread_file).resolve()
    audit_path = Path(audit_file).resolve()
    payload = _load_payload(input_path)
    try:
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ASRReviewError(f"cannot read ASR review audit: {exc}") from exc
    audit_items = audit_payload.get("segments")
    if not isinstance(audit_items, list):
        raise ASRReviewError("invalid ASR review audit")

    prompt_terms = [str(term).strip() for term in terms if str(term).strip()]
    reviewed = [dict(item) for item in payload["segments"]]
    by_id = {int(item["id"]): item for item in reviewed}
    replacements = 0
    audio_seconds = 0.0
    refreshed_audit: list[dict[str, Any]] = []
    for old in audit_items:
        segment_id = int(old["segment_id"])
        if segment_id not in by_id:
            raise ASRReviewError(f"audit references unknown segment: {segment_id}")
        target = by_id[segment_id]
        prior = str(target.get("corrected_text") or target.get("asr_text") or "").strip()
        candidate_raw = str(old.get("large_v3_raw_text") or old.get("large_v3_text") or "").strip()
        candidate = _canonicalize_candidate_terms(candidate_raw, prompt_terms)
        candidate_is_safe = _safe_candidate(candidate, prior)
        selected, applied_terms = (
            _patch_verified_term_differences(prior, candidate, prompt_terms)
            if candidate_is_safe
            else (prior, [])
        )
        replaced = selected != prior
        if replaced:
            replacements += 1
            target["corrected_text"] = selected
            target["changed"] = selected != str(target.get("asr_text") or "")
        start = float(old["start_seconds"])
        end = float(old["end_seconds"])
        audio_seconds += max(0.0, end - start)
        refreshed = dict(old)
        refreshed.update(
            {
                "proofread_text": prior,
                "large_v3_text": candidate,
                "selected_text": selected,
                "replaced": replaced,
                "prior_verified_term_hits": _verified_term_hits(prior, prompt_terms),
                "candidate_verified_term_hits": _verified_term_hits(candidate, prompt_terms),
                "selected_reason": (
                    "patched_verified_term_spellings"
                    if applied_terms
                    else "kept_proofread_no_verified_improvement"
                    if candidate_is_safe
                    else "kept_proofread_candidate_failed_safety_check"
                ),
                "applied_verified_terms": applied_terms,
            }
        )
        refreshed_audit.append(refreshed)

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model_name = str(audit_payload.get("model") or "whisper-large-v3")
    billed = sum(
        max(10.0, float(item["end_seconds"]) - float(item["start_seconds"]))
        for item in refreshed_audit
    )
    enhanced = dict(payload)
    enhanced["input_proofread"] = str(input_path)
    enhanced["selective_asr_review"] = {
        "provider": "groq",
        "model": model_name,
        "language": "ja",
        "candidate_count": len(refreshed_audit),
        "replacement_count": replacements,
        "audio_seconds": audio_seconds,
        "billed_audio_seconds_estimate": billed,
        "preferred_terms": prompt_terms,
        "reselected_from_existing_audit": True,
    }
    enhanced["segments"] = reviewed
    json_path = destination / "proofread_reviewed.json"
    markdown_path = destination / "proofread_reviewed.md"
    destination_audit_path = destination / "asr_review.json"
    json_path.write_text(json.dumps(enhanced, ensure_ascii=False, indent=2), encoding="utf-8")
    destination_audit_path.write_text(
        json.dumps(
            {
                "source_url": payload.get("source_url", ""),
                "model": model_name,
                "segments": refreshed_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    source_url = str(payload.get("source_url") or "")
    lines = [f"# {payload.get('title') or input_path.stem} — Selective ASR Review", ""]
    lines.extend(
        [
            f"- Review model: `{model_name}`",
            f"- Reviewed segments: `{len(refreshed_audit)}`",
            f"- Replacements: `{replacements}`",
            "",
        ]
    )
    for item in reviewed:
        label = format_timestamp(float(item["start_seconds"]))
        if source_url:
            label = f"[{label}]({timestamp_url(source_url, float(item['start_seconds']))})"
        review = " ⚠️ needs review" if item.get("needs_review") else ""
        lines.append(f"- {label} {item.get('corrected_text', '')}{review}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ASRReviewArtifact(
        input_path=input_path,
        json_path=json_path,
        markdown_path=markdown_path,
        audit_path=destination_audit_path,
        model=model_name,
        candidate_count=len(refreshed_audit),
        replacement_count=replacements,
        audio_seconds=audio_seconds,
        billed_audio_seconds_estimate=billed,
    )
