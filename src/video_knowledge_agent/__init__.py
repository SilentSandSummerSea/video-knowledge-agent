"""Video Knowledge Agent core package."""

from .audio import LocalMediaArtifact, LocalMediaError, prepare_local_audio
from .groq_transcription import transcribe_audio_groq, transcribe_audio_groq_chunked
from .models import KnowledgeChunk, VideoSource, format_timestamp, timestamp_url
from .transcription import (
    TranscriptArtifact,
    TranscriptSegment,
    TranscriptionError,
    transcribe_audio,
)

__all__ = [
    "LocalMediaArtifact",
    "LocalMediaError",
    "KnowledgeChunk",
    "TranscriptArtifact",
    "TranscriptSegment",
    "TranscriptionError",
    "VideoSource",
    "prepare_local_audio",
    "format_timestamp",
    "timestamp_url",
    "transcribe_audio",
    "transcribe_audio_groq",
    "transcribe_audio_groq_chunked",
]
__version__ = "0.1.0"
