# Knowledge Extraction Prompt Contract

Extract only claims supported by the supplied transcript or visual evidence.

For each item return:

- `topic`
- `knowledge_type`: fact, procedure, recommendation, warning, definition, or opinion
- `statement`
- `entities`
- `start_seconds`
- `end_seconds`
- `evidence_text`
- `confidence`

Do not merge conflicting claims. Do not infer missing settings, versions, numbers, or names. If evidence is insufficient, emit no item and explain the gap in diagnostics.
