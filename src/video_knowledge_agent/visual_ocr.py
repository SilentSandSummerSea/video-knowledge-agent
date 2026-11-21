"""Extract distinct frames from one cached clip and align local OCR to speech."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .models import timestamp_url


class VisualOcrError(RuntimeError):
    """A cached clip cannot be safely decoded or processed."""


@dataclass(frozen=True, slots=True)
class VisualOcrArtifact:
    json_path: Path
    frame_count: int
    text_line_count: int
    reused: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["json_path"] = str(self.json_path)
        return result


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VisualOcrError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VisualOcrError(f"expected JSON object in {path}")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transcript_segments(path_value: object) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.is_file():
        return []
    payload = _read_object(path)
    raw = payload.get("segments")
    if not isinstance(raw, list):
        return []
    segments = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            segment_id = int(item["id"])
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
        except (KeyError, ValueError, TypeError):
            continue
        text = str(item.get("corrected_text") or item.get("text") or item.get("asr_text") or "").strip()
        if 0 <= start < end and text:
            segments.append({"id": segment_id, "start_seconds": start, "end_seconds": end, "text": text})
    return segments


def _sample_frames(clip: Path, sample_every_seconds: float, max_frames: int,
                   change_threshold: float) -> list[tuple[float, Any]]:
    try:
        import av
        from PIL import ImageChops, ImageStat
    except ImportError as exc:
        raise VisualOcrError("frame extraction requires PyAV and Pillow; install .[ocr]") from exc
    try:
        with av.open(str(clip)) as container:
            if not any(stream.type == "video" for stream in container.streams):
                raise VisualOcrError("clip has no video stream")
            duration = float(container.duration / av.time_base) if container.duration else 0.0
            interval = max(sample_every_seconds, duration / (max_frames * 2))
            next_time = 0.0
            previous_thumb = None
            candidates: list[tuple[float, Any]] = []
            for frame in container.decode(video=0):
                at = frame.time
                if at is None or not math.isfinite(at) or at + 0.04 < next_time:
                    continue
                next_time = at + interval
                image = frame.to_image().convert("RGB")
                thumb = image.convert("L").resize((64, 36))
                difference = (
                    ImageStat.Stat(ImageChops.difference(thumb, previous_thumb)).mean[0]
                    if previous_thumb is not None else 255.0
                )
                if difference >= change_threshold:
                    candidates.append((max(0.0, float(at)), image))
                    previous_thumb = thumb
            if not candidates:
                raise VisualOcrError("clip contains no decodable frames")
    except VisualOcrError:
        raise
    except Exception as exc:
        raise VisualOcrError(f"cannot decode clip frames: {exc}") from exc
    if len(candidates) <= max_frames:
        return candidates
    if max_frames == 1:
        return [candidates[len(candidates) // 2]]
    indices = [round(index * (len(candidates) - 1) / (max_frames - 1)) for index in range(max_frames)]
    return [candidates[index] for index in indices]


def _rapidocr_engine(language: str) -> Callable[[Any], Any]:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise VisualOcrError("local OCR requires rapidocr and onnxruntime; install .[ocr]") from exc
    try:
        if language in {"ja", "jpn", "japanese", "japan"}:
            from rapidocr import ModelType, OCRVersion
            return RapidOCR(params={
                "Rec.lang_type": "japan",
                "Rec.model_type": ModelType.SMALL,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
            })
        return RapidOCR()
    except Exception as exc:
        raise VisualOcrError(f"cannot initialize local OCR model: {exc}") from exc


def _ocr_lines(result: Any, min_confidence: float) -> list[dict[str, Any]]:
    raw_texts = getattr(result, "txts", None)
    raw_scores = getattr(result, "scores", None)
    texts = list(raw_texts) if raw_texts is not None else []
    scores = list(raw_scores) if raw_scores is not None else []
    boxes = getattr(result, "boxes", None)
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    boxes = list(boxes) if boxes is not None else []
    lines = []
    for index, value in enumerate(texts):
        text = str(value).strip()
        confidence = float(scores[index]) if index < len(scores) else 0.0
        if not text or not math.isfinite(confidence) or confidence < min_confidence:
            continue
        lines.append({
            "text": text,
            "confidence": round(confidence, 4),
            "box": boxes[index] if index < len(boxes) else None,
        })
    return lines


def extract_moment_ocr(
    clip_file: str | Path,
    *,
    knowledge_root: str | Path = "data/outputs/knowledge",
    sample_every_seconds: float = 3.0,
    max_frames: int = 12,
    change_threshold: float = 8.0,
    min_confidence: float = 0.35,
    ocr_engine: Callable[[Any], Any] | None = None,
) -> VisualOcrArtifact:
    """Cache frame images and OCR, with absolute video times and nearby speech IDs."""
    if sample_every_seconds <= 0 or max_frames < 1 or max_frames > 30:
        raise ValueError("sampling interval must be positive and max_frames must be 1–30")
    if not 0 <= change_threshold <= 255 or not 0 <= min_confidence <= 1:
        raise ValueError("invalid frame-change or confidence threshold")
    clip = Path(clip_file).resolve()
    if not clip.is_file() or clip.stat().st_size == 0:
        raise VisualOcrError("clip is missing or empty")
    manifest = _read_object(clip.parent / "clip.json")
    video_id = str(manifest.get("video_id") or "")
    moment_id = str(manifest.get("moment_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id) or manifest.get("file") != clip.name:
        raise VisualOcrError("clip manifest does not match the video file")
    root = Path(knowledge_root).resolve()
    moments_payload = _read_object(root / video_id / "moments.json")
    metadata = _read_object(root / video_id / "metadata.json")
    if (moments_payload.get("video_id") != video_id
            or moments_payload.get("source_hash") != manifest.get("source_hash")):
        raise VisualOcrError("clip references stale or mismatched Moment data")
    matches = [item for item in moments_payload.get("moments", [])
               if isinstance(item, dict) and item.get("id") == moment_id]
    if len(matches) != 1:
        raise VisualOcrError("clip references an unknown Moment")
    moment = matches[0]
    source_url = str(metadata.get("source_url") or moments_payload.get("source_url") or "")
    clip_start = float(manifest["start_seconds"])
    clip_end = float(manifest["end_seconds"])
    if not source_url or clip_start < 0 or clip_end <= clip_start:
        raise VisualOcrError("clip has invalid time or source metadata")
    language = str(moments_payload.get("source_language") or "unknown").lower()
    fingerprint = hashlib.sha256(json.dumps({
        "clip_hash": _file_hash(clip),
        "source_hash": manifest["source_hash"],
        "sampling": sample_every_seconds,
        "max_frames": max_frames,
        "change_threshold": change_threshold,
        "min_confidence": min_confidence,
        "language": language,
        "schema": 1,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    destination = clip.parent / f"ocr-{fingerprint[:12]}"
    output_path = destination / "ocr.json"
    if output_path.is_file():
        try:
            cached = _read_object(output_path)
            frames = cached["frames"]
            if (cached.get("fingerprint") == fingerprint and isinstance(frames, list)
                    and all((destination / item["image_file"]).is_file() for item in frames)):
                return VisualOcrArtifact(output_path, len(frames),
                                         sum(len(item["lines"]) for item in frames), True)
        except (VisualOcrError, KeyError, TypeError, ValueError):
            pass
    engine = ocr_engine or _rapidocr_engine(language)
    sampled = _sample_frames(clip, sample_every_seconds, max_frames, change_threshold)
    speech = _transcript_segments(moments_payload.get("input_transcript"))
    allowed_ids = set(moment.get("segment_ids") or [])
    destination.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, (relative_time, image) in enumerate(sampled, start=1):
        absolute_time = min(clip_end, clip_start + relative_time)
        filename = f"frame_{index:03d}.jpg"
        image_path = destination / filename
        image.save(image_path, "JPEG", quality=90)
        try:
            result = engine(image)
            lines = _ocr_lines(result, min_confidence)
        except Exception as exc:
            raise VisualOcrError(f"OCR failed at {absolute_time:.2f}s: {exc}") from exc
        aligned = [item for item in speech if item["id"] in allowed_ids
                   and item["start_seconds"] <= absolute_time + 2
                   and item["end_seconds"] >= absolute_time - 2]
        frames.append({
            "relative_seconds": round(relative_time, 3),
            "video_seconds": round(absolute_time, 3),
            "source_url": timestamp_url(source_url, absolute_time),
            "image_file": filename,
            "lines": lines,
            "aligned_segment_ids": [item["id"] for item in aligned],
            "aligned_speech": [item["text"] for item in aligned],
        })
    payload = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "video_id": video_id,
        "moment_id": moment_id,
        "chapter_id": moment.get("chapter_id"),
        "source_language": language,
        "clip_file": str(clip),
        "clip_start_seconds": clip_start,
        "clip_end_seconds": clip_end,
        "alignment_tolerance_seconds": 2,
        "frames": frames,
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return VisualOcrArtifact(output_path, len(frames), sum(len(item["lines"]) for item in frames), False)
