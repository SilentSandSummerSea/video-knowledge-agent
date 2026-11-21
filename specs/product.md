# Product Specification

## Objective

Turn authorized, locally available video or audio into a searchable knowledge collection with timestamped evidence. Keep acquisition outside the public core.

## Required workflow

1. Accept one local video or audio path and validate it exists.
2. Extract and validate audio locally; generate a stable file fingerprint.
3. Upload overlapping audio chunks to Groq ASR, preserving source language and timestamps.
4. Proofread with DeepSeek, split chapters, extract evidence-grounded knowledge, and create Moments.
5. Publish per-chapter Markdown/JSON and build a local SQLite FTS5 index.
6. When visual evidence is needed and a local video exists, extract only the relevant short clip and run OCR on bounded keyframes. Audio-only sources skip this step explicitly.
7. Rerun safely: reuse completed stages and invalidate results when the input file changes.
8. Answer questions from indexed speech and optional OCR/frame evidence. Use the configured text provider, validate model-selected evidence IDs, and construct citations locally. Do not claim visual verification when no source video is available.

## Boundary

No public-core command accepts a website URL as a media source. A separate adapter may lawfully obtain or otherwise receive audio and pass its completed local path to `process`. API calls for ASR/text analysis remain allowed. Local-source citations are internal references until a local player resolver is attached; a local metadata sidecar may provide an original webpage citation without making the core download it.

## Evaluation

Test local input, cache reuse, changed-file invalidation, missing-source handling, chapter/Moment continuity, visual time alignment, and absence of download dependencies from the public package. Verify media, keys, and private acquisition tooling are excluded from Git.
