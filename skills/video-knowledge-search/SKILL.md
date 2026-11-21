---
name: video-knowledge-search
description: Search the user's local indexed video knowledge and answer with chapter, evidence, and timestamp links. Use automatically when the user asks what a processed video said, asks about facts, figures, people, events, conclusions, or procedures contained in their video library, or wants to locate the source moment. Do not use for general web questions or for processing a new video.
---

# Video Knowledge Search

When asked about a processed video, retrieve the matching knowledge items and their nearby source transcript, then answer the user's question. The helper reads local files only and does not generate an answer on its own.

## Query

Run the bundled script with the user's natural-language question:

```powershell
python .\skills\video-knowledge-search\scripts\search_knowledge.py --query '<question>' --limit 8
```

The script automatically rebuilds the local SQLite index when published knowledge is newer than the database. It is entirely local and must not invoke Groq, DeepSeek, OpenAI, or another external service.

Each hit includes `video_title`, `content`, and `context`: the evidence time range, a direct `evidence_url`, and nearby corrected transcript segments (normally 15 seconds before and after). `kind=moment` is a short chapter-bounded span of the original transcript. `context.precision` is `knowledge_evidence` for a structured knowledge item, `moment` for a short source span, `segment` when a matching transcript segment can be located inside a chapter, `chapter_only` when no precise sentence can be located, or `index_only` when the local transcript is missing. `needs_review` flags uncertain ASR text.

If the corpus may use a different language from the question, derive compact equivalents of the important nouns and technical terms in the likely source languages and run at most three focused queries total. Merge duplicate results by `id`. Do not invoke an external translation API or run repeated broad searches.

## Answer from evidence

- Read the relevant `context.segments` before answering, especially when the indexed item is long or awkwardly worded. Base factual claims on the indexed evidence and nearby source text; do not fill gaps from general knowledge.
- Prefer a specific structured knowledge item over a whole-chapter match.
- Use a matching `moment` when the question seeks the precise spoken passage or a structured item does not capture it. Its source URL starts at the moment's first transcript segment.
- Produce a direct answer and a short synthesis, rather than forwarding the raw search JSON. Attribute distinct claims to their matching video and time range.
- Answer in the user's interaction language (normally Chinese for this user), translating or summarizing only in the response. Never rewrite stored transcripts or knowledge merely to match the interaction language.
- Cite the time range and local chapter file. `local-video://` is an internal reference, not a browser-playable link; do not present it as a clickable public URL. When a local player integration is available, resolve the video ID and timestamp there.
- Treat `needs_review` as a reason to qualify uncertain names, numbers, or quotations. Avoid presenting a generated knowledge statement as a verbatim quote unless the transcript confirms it.
- If several videos conflict, identify each video rather than merging their claims.
- If the index lacks sufficient evidence, say so clearly and offer to search with a revised term or process the missing video.

The index database and published corpus belong to this repository's ignored `data/` directory. Do not modify transcript, chapter, or structured-knowledge source files while answering a query.
