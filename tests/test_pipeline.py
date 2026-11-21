import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from video_knowledge_agent.pipeline import PipelineError, process_video


class PipelineTests(unittest.TestCase):
    def test_local_mode_routes_both_stages_and_switching_back_reruns_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"video fixture")
            audio = root / "raw" / "source.flac"
            audio.parent.mkdir()
            audio.write_bytes(b"audio fixture")
            calls = []
            local_client = object()

            def media_loader(path, output_dir):
                return SimpleNamespace(video_id="MODETEST", source_url="local-video://MODETEST",
                                       title="Mode test", file_path=audio,
                                       source_path=source, video_path=source)

            def write_transcript(label, output_dir, **kwargs):
                calls.append((label, kwargs))
                path = Path(output_dir) / "transcript.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"language": "en"}), encoding="utf-8")
                return SimpleNamespace(json_path=path)

            def proofreader(path, output_dir, **kwargs):
                calls.append(("proofread", kwargs))
                raise RuntimeError("stop-before-text-call")

            options = dict(raw_root=root / "raw", work_root=root / "work",
                           media_loader=media_loader, proofreader=proofreader,
                           transcriber=lambda *a, **kw: write_transcript("groq", a[1], **kw),
                           local_transcriber=lambda *a, **kw: write_transcript("local", a[1], **kw),
                           local_text_client=local_client, show_progress=False)
            with patch.dict(os.environ, {"LOCAL_TEXT_MODEL": "fixture-model"}):
                with self.assertRaisesRegex(PipelineError, "stop-before-text-call"):
                    process_video(source, asr_provider="local", text_provider="local", **options)
            self.assertEqual(calls[0][0], "local")
            self.assertEqual(calls[0][1]["model_name"], "small")
            self.assertEqual(calls[1][1]["provider"], "local")
            self.assertIs(calls[1][1]["client"], local_client)
            calls.clear()
            with self.assertRaisesRegex(PipelineError, "stop-before-text-call"):
                process_video(source, asr_provider="groq", text_provider="deepseek", **options)
            self.assertEqual(calls[0][0], "groq")
            self.assertEqual(calls[1][1]["provider"], "deepseek")

    def test_audio_only_input_reaches_transcription_without_video_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.webm"
            source.write_bytes(b"audio-only fixture")
            audio = root / "raw" / "source.flac"
            audio.parent.mkdir()
            audio.write_bytes(b"normalized fixture")

            def media_loader(path, output_dir):
                return SimpleNamespace(
                    video_id="AUDIOONLY", source_url="local-audio://AUDIOONLY",
                    title="Audio only", file_path=audio, source_path=source,
                    video_path=None,
                )

            def stop_transcription(*args, **kwargs):
                raise RuntimeError("stop-before-paid-stage")

            with self.assertRaisesRegex(PipelineError, "stop-before-paid-stage"):
                process_video(
                    source, raw_root=root / "raw", work_root=root / "work",
                    media_loader=media_loader, transcriber=stop_transcription,
                    show_progress=False,
                )
            state = json.loads((root / "work" / "AUDIOONLY" / "pipeline.json").read_text(encoding="utf-8"))
            self.assertEqual(state["requested_file"], str(source.resolve()))
            self.assertIs(state["source_has_video"], False)
            self.assertIsNone(state["stages"]["extract_audio"]["video_path"])

    def test_runs_all_stages_then_resumes_without_repeating_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            progress_events = []
            text_stage_kwargs = {}

            video_file = root / "input.mp4"
            video_file.write_bytes(b"local fixture")

            def media_loader(path, output_dir):
                calls.append("extract_audio")
                audio = root / "raw" / "source.m4a"
                audio.parent.mkdir(parents=True, exist_ok=True)
                audio.write_bytes(b"audio")
                return SimpleNamespace(
                    video_id="BVTEST",
                    source_url="local-video://BVTEST",
                    title="测试视频",
                    file_path=audio,
                    video_path=video_file,
                )

            def transcriber(audio_path, output_dir, **kwargs):
                calls.append("transcribe")
                text_stage_kwargs["transcribe"] = kwargs
                path = Path(output_dir) / "transcript.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                detected = "ja" if kwargs["language"] == "auto" else kwargs["language"]
                path.write_text(json.dumps({"language": detected}), encoding="utf-8")
                return SimpleNamespace(json_path=path)

            def proofreader(transcript_path, output_dir, **kwargs):
                calls.append("proofread")
                text_stage_kwargs["proofread"] = kwargs
                path = Path(output_dir) / "proofread.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "source_url": "local-video://BVTEST",
                            "title": "测试视频",
                            "reasoning_effort": kwargs["reasoning_effort"],
                            "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5, "reasoning_tokens": 0},
                            "cost_estimate": {"estimated_cost_cny": 0.1, "pricing_period": "off_peak"},
                            "segments": [
                                {
                                    "id": 1,
                                    "start_seconds": 0,
                                    "end_seconds": 5,
                                    "corrected_text": "测试正文。",
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(json_path=path)

            def chapterer(proofread_path, output_dir, **kwargs):
                calls.append("chapters")
                text_stage_kwargs["chapters"] = kwargs
                path = Path(output_dir) / "chapters.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "source_url": "local-video://BVTEST",
                            "title": "测试视频",
                            "reasoning_effort": kwargs["reasoning_effort"],
                            "usage": {"input_tokens": 20, "cached_input_tokens": 0, "output_tokens": 10, "reasoning_tokens": 0},
                            "cost_estimate": {"estimated_cost_cny": 0.2, "pricing_period": "off_peak"},
                            "input_transcript": str(Path(proofread_path).resolve()),
                            "chapters": [
                                {
                                    "id": 1,
                                    "title": "测试章节",
                                    "start_segment_id": 1,
                                    "end_segment_id": 1,
                                    "start_seconds": 0,
                                    "end_seconds": 5,
                                    "summary": "测试摘要。",
                                    "topics": ["测试"],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(json_path=path)

            def knowledge_extractor(chapters_path, output_dir, **kwargs):
                calls.append("knowledge")
                text_stage_kwargs["knowledge"] = kwargs
                path = Path(output_dir) / "knowledge.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "reasoning_effort": kwargs["reasoning_effort"],
                            "usage": {"input_tokens": 30, "cached_input_tokens": 0, "output_tokens": 15, "reasoning_tokens": 0},
                            "cost_estimate": {"estimated_cost_cny": 0.3, "pricing_period": "off_peak"},
                            "chapters": [],
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(json_path=path)

            kwargs = {
                "raw_root": root / "raw",
                "work_root": root / "work",
                "publish_root": root / "published",
                "index_database": root / "index" / "knowledge.db",
                "media_loader": media_loader,
                "transcriber": transcriber,
                "proofreader": proofreader,
                "chapterer": chapterer,
                "knowledge_extractor": knowledge_extractor,
                "terms": ("ゲームデザイン",),
                "progress_callback": lambda stage, status, percent, detail: progress_events.append(
                    (stage, status, percent, detail)
                ),
            }
            first = process_video(video_file, **kwargs)
            second = process_video(video_file, **kwargs)

            self.assertEqual(calls, ["extract_audio", "transcribe", "proofread", "chapters", "knowledge", "extract_audio"])
            self.assertEqual(first.executed_stages, ("extract_audio", "transcribe", "proofread", "asr_review", "chapters", "moments", "knowledge"))
            self.assertEqual(second.executed_stages, ())
            self.assertEqual(second.reused_stages, ("extract_audio", "transcribe", "proofread", "asr_review", "chapters", "moments", "knowledge"))
            self.assertTrue(first.cost_estimate_complete)
            self.assertEqual(first.estimated_deepseek_cost_cny, 0.6)
            self.assertEqual(first.estimated_run_cost_cny, 0.6)
            self.assertEqual(second.estimated_run_cost_cny, 0.0)
            self.assertEqual(first.deepseek_usage["output_tokens"], 30)
            self.assertEqual(first.source_language, "ja")
            self.assertEqual(first.output_language, "ja")
            self.assertIn(("pipeline", "starting", 0, ""), progress_events)
            self.assertIn(("transcribe", "completed", 35, ""), progress_events)
            self.assertTrue(
                any(
                    stage == "transcribe" and status == "reused" and percent == 35
                    for stage, status, percent, _ in progress_events
                )
            )
            self.assertTrue(
                any(
                    stage == "pipeline" and status == "completed" and percent == 100
                    for stage, status, percent, _ in progress_events
                )
            )
            self.assertEqual(text_stage_kwargs["transcribe"]["language"], "auto")
            self.assertEqual(text_stage_kwargs["transcribe"]["prompt"], "ゲームデザイン")
            self.assertEqual(text_stage_kwargs["proofread"]["reasoning_effort"], "none")
            self.assertEqual(text_stage_kwargs["chapters"]["reasoning_effort"], "none")
            self.assertEqual(text_stage_kwargs["chapters"]["output_language"], "ja")
            self.assertEqual(text_stage_kwargs["knowledge"]["reasoning_effort"], "none")
            self.assertEqual(text_stage_kwargs["knowledge"]["output_language"], "ja")
            state = json.loads(second.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(state["stages"]["chapters"]["status"], "completed")
            self.assertEqual(state["stages"]["chapter_export"]["file_count"], 1)
            self.assertEqual(state["stages"]["moments"]["moment_count"], 1)
            self.assertEqual(state["stages"]["publish"]["chapter_count"], 1)
            metadata = json.loads(second.knowledge_metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["video_id"], "BVTEST")
            self.assertEqual(metadata["chapters"][0]["file"], "chapters/01-测试章节.md")
            self.assertTrue((second.knowledge_dir / metadata["chapters"][0]["file"]).is_file())
            self.assertEqual(metadata["structured_knowledge_file"], "structured_knowledge.json")
            self.assertEqual(metadata["moments_file"], "moments.json")
            self.assertEqual(metadata["moment_count"], 1)
            self.assertTrue(second.index_database_path.is_file())
            self.assertEqual(state["stages"]["index"]["knowledge_count"], 0)

            third = process_video(
                video_file, output_language="en", **kwargs
            )
            self.assertEqual(third.executed_stages, ("chapters", "knowledge"))
            self.assertEqual(
                third.reused_stages, ("extract_audio", "transcribe", "proofread", "asr_review", "moments")
            )
            self.assertEqual(third.output_language, "en")

            fourth = process_video(
                video_file,
                language="en",
                output_language="en",
                **kwargs,
            )
            self.assertEqual(
                fourth.executed_stages,
                ("transcribe", "proofread", "chapters", "moments", "knowledge"),
            )
            self.assertEqual(fourth.reused_stages, ("extract_audio",))
            self.assertEqual(fourth.source_language, "en")
            self.assertTrue(
                any(
                    stage == "asr_review" and status == "skipped" and percent == 62
                    for stage, status, percent, _ in progress_events
                )
            )


if __name__ == "__main__":
    unittest.main()
