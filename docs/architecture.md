# Architecture

The public package is a **local-media-first** pipeline. Its input boundary is an existing video or audio file path. Optional website discovery/acquisition is a separate adapter and is not imported by the package. The project-root `.env` selects Groq or faster-whisper for ASR and DeepSeek or a loopback Ollama server for text processing; these choices are independent.

```text
external acquisition (optional/private) -> local audio path
  -> FFmpeg audio extraction -> Groq 120s/5s-overlap ASR or local faster-whisper
  -> DeepSeek or local Ollama proofreading -> chapters -> structured knowledge
  -> time-bounded Moments -> Markdown/JSON publication -> SQLite FTS5
  -> on-demand local clip extraction -> keyframes/OCR (local video only)
```

The core's `media_loader` callable is the extension point for externally obtained media. Its default implementation, `prepare_local_audio`, accepts only local files and emits normalized audio plus metadata. The pipeline manifest stores the source path and whether it has a video stream in ignored `data/work/`. Audio-only input cannot support clip extraction or OCR; when a local video is present, visual extraction remains available. Published citations use an internal `local-video://` or `local-audio://` reference unless a neighboring provenance sidecar supplies an original webpage URL. The core never fetches that URL.

This separation makes the processing code publishable without shipping automated download logic. It does **not** assert that any particular online video may be downloaded, processed, or redistributed. Groq and DeepSeek remain network services; users must account for privacy and their terms. Media, transcripts, keys, and generated knowledge are ignored by Git by default.
