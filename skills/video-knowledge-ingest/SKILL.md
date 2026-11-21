---
name: video-knowledge-ingest
description: Process a local video file through the video knowledge pipeline and add it to the searchable knowledge base. Use for /ana_video followed by a local file path, or when asked to ingest an already available local video. This skill does not download or resolve website URLs.
---

# Video Knowledge Ingest

Accept one existing local video file path. For `/ana_video`, strip the prefix and use the remaining path. Parse optional `--language`, `--output-language`, and preferred terms. If given only a website URL, explain that the public core needs a local file and ask the user to supply it or use an external, separately authorized acquisition adapter. Do not download media from this skill.

Run `scripts/ingest_video.py <local-video-file>` with the current Python environment. The script locates this repository relative to itself and follows the project-root `.env` mode settings: `ASR_PROVIDER=groq|local` and `TEXT_PROVIDER=deepseek|local`. API mode sends audio chunks to Groq and text to DeepSeek; local mode uses faster-whisper and a loopback Ollama server. The default output language is the source language. Never print API keys. The script then builds local Moments and an FTS5 index.

Completed stages are reusable. If the user requests a specific rerun, pass `--force-stage <stage>`; otherwise do not repeat paid stages. Report the video ID, executed/reused stages, published knowledge location, and estimated API cost. The resulting `local-video://` references are internal identifiers rather than public web links.
