import json
import unittest
from io import StringIO
from unittest import mock

from custom_asmr_srt_stack.transcription import (
    ModelEndpoint,
    TransformersWorkerClient,
    adapter_max_chunk_ms,
    build_gemini_url,
    parse_model_segments,
    parse_worker_response,
    transcribe_audio,
)


class TranscriptionAdapterTests(unittest.TestCase):
    def test_openai_compatible_adapter_posts_audio_and_parses_segments(self):
        calls = []

        def fake_post(url, headers, body):
            calls.append((url, headers, json.loads(body.decode("utf-8"))))
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "segments": [
                                        {
                                            "start_ms": 0,
                                            "end_ms": 1200,
                                            "channel": "MIX",
                                            "kind": "speech",
                                            "text": "ねえ",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        endpoint = ModelEndpoint(
            adapter="openai-compatible",
            endpoint_url="http://localhost:8000/v1",
            model_id="gemma-4-e4b",
            api_key="secret",
        )
        segments = transcribe_audio(endpoint, b"audio", mime_type="audio/wav", http_post=fake_post)

        self.assertEqual(calls[0][0], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(calls[0][1]["Authorization"], "Bearer secret")
        self.assertEqual(calls[0][2]["model"], "gemma-4-e4b")
        self.assertEqual(calls[0][2]["messages"][1]["content"][1]["input_audio"]["format"], "wav")
        self.assertEqual(segments[0].text, "ねえ")
        self.assertEqual(segments[0].id, "seg_000001")

    def test_gemini_adapter_posts_inline_audio_and_parses_segments(self):
        calls = []

        def fake_post(url, headers, body):
            calls.append((url, headers, json.loads(body.decode("utf-8"))))
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "segments": [
                                                {
                                                    "start_ms": 100,
                                                    "end_ms": 900,
                                                    "channel": "L",
                                                    "kind": "speech",
                                                    "text": "聞こえてる？",
                                                }
                                            ]
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            ]
                        }
                    }
                ]
            }

        endpoint = ModelEndpoint(
            adapter="gemini",
            endpoint_url="https://generativelanguage.googleapis.com",
            model_id="gemini-2.5-pro",
            api_key="secret",
        )
        segments = transcribe_audio(endpoint, b"audio", mime_type="audio/wav", channel="L", http_post=fake_post)

        self.assertEqual(
            calls[0][0],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
        )
        self.assertEqual(calls[0][1]["x-goog-api-key"], "secret")
        self.assertEqual(calls[0][2]["contents"][0]["parts"][1]["inline_data"]["mime_type"], "audio/wav")
        self.assertEqual(segments[0].channel, "L")

    def test_gemini_url_accepts_full_generate_content_url(self):
        endpoint = ModelEndpoint(
            adapter="gemini",
            endpoint_url="https://example.test/v1beta/models/gemini:generateContent",
            model_id="ignored",
        )

        self.assertEqual(build_gemini_url(endpoint), "https://example.test/v1beta/models/gemini:generateContent")

    def test_local_transformers_adapter_does_not_require_endpoint_url(self):
        endpoint = ModelEndpoint.from_json(
            {
                "adapter": "local-transformers",
                "model_id": "google/gemma-4-E4B-it",
            }
        )

        self.assertIsNone(endpoint.endpoint_url)
        self.assertEqual(adapter_max_chunk_ms(endpoint), 30_000)

    def test_local_qwen_asr_adapter_does_not_require_endpoint_url(self):
        endpoint = ModelEndpoint.from_json(
            {
                "adapter": "local-qwen-asr",
                "model_id": "Qwen/Qwen3-ASR-1.7B",
            }
        )

        self.assertIsNone(endpoint.endpoint_url)
        self.assertIsNone(adapter_max_chunk_ms(endpoint))

    def test_endpoint_adapter_requires_endpoint_url(self):
        with self.assertRaisesRegex(ValueError, "endpoint_url must not be empty"):
            ModelEndpoint.from_json(
                {
                    "adapter": "openai-compatible",
                    "model_id": "gemma-4-e4b",
                }
            )

    def test_local_transformers_adapter_uses_local_transcriber_boundary(self):
        calls = []

        def fake_local(endpoint, audio_bytes, mime_type, channel, source_language):
            calls.append((endpoint.model_id, audio_bytes, mime_type, channel, source_language))
            return (
                parse_model_segments(
                    json.dumps(
                        {
                            "segments": [
                                {
                                    "start_ms": 0,
                                    "end_ms": 10,
                                    "channel": channel,
                                    "kind": "speech",
                                    "text": "ねえ",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )
            )

        endpoint = ModelEndpoint(
            adapter="local-transformers",
            endpoint_url=None,
            model_id="google/gemma-4-E4B-it",
        )
        segments = transcribe_audio(
            endpoint,
            b"audio",
            mime_type="audio/wav",
            channel="L",
            local_transcribe_func=fake_local,
        )

        self.assertEqual(calls, [("google/gemma-4-E4B-it", b"audio", "audio/wav", "L", "ja")])
        self.assertEqual(segments[0].text, "ねえ")

    def test_local_qwen_asr_adapter_uses_local_transcriber_boundary(self):
        calls = []

        def fake_local(endpoint, audio_bytes, mime_type, channel, source_language):
            calls.append((endpoint.model_id, audio_bytes, mime_type, channel, source_language))
            return parse_model_segments(
                json.dumps(
                    {
                        "segments": [
                            {
                                "start_ms": 0,
                                "end_ms": 10,
                                "channel": channel,
                                "kind": "speech",
                                "text": "ねえ",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

        endpoint = ModelEndpoint(
            adapter="local-qwen-asr",
            endpoint_url=None,
            model_id="Qwen/Qwen3-ASR-1.7B",
        )
        segments = transcribe_audio(
            endpoint,
            b"audio",
            mime_type="audio/wav",
            channel="MIX",
            local_transcribe_func=fake_local,
        )

        self.assertEqual(calls, [("Qwen/Qwen3-ASR-1.7B", b"audio", "audio/wav", "MIX", "ja")])
        self.assertEqual(segments[0].text, "ねえ")

    def test_worker_response_is_parsed_as_segments(self):
        segments = parse_worker_response(
            json.dumps(
                {
                    "ok": True,
                    "segments": [
                        {
                            "start_ms": 0,
                            "end_ms": 100,
                            "channel": "MIX",
                            "kind": "speech",
                            "text": "はい",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(segments[0].id, "seg_000001")
        self.assertEqual(segments[0].text, "はい")

    def test_worker_response_failure_is_visible(self):
        with self.assertRaisesRegex(ValueError, "local transformers worker failed: load failed"):
            parse_worker_response(json.dumps({"ok": False, "error": "load failed"}))

    def test_worker_response_failure_uses_worker_name(self):
        with self.assertRaisesRegex(ValueError, "local Qwen ASR worker failed: load failed"):
            parse_worker_response(json.dumps({"ok": False, "error": "load failed"}), worker_name="local Qwen ASR worker")

    def test_transformers_worker_client_uses_json_lines_protocol(self):
        response = json.dumps(
            {
                "ok": True,
                "segments": [
                    {
                        "start_ms": 0,
                        "end_ms": 10,
                        "channel": "L",
                        "kind": "speech",
                        "text": "ねえ",
                    }
                ],
            },
            ensure_ascii=False,
        )

        class FakeProcess:
            def __init__(self):
                self.stdin = StringIO()
                self.stdout = StringIO(response + "\n")

            def poll(self):
                return None

        fake_process = FakeProcess()

        with mock.patch("custom_asmr_srt_stack.transcription.subprocess.Popen", return_value=fake_process) as popen:
            client = TransformersWorkerClient(("worker", "--stdio"))
            segments = client.transcribe(
                ModelEndpoint(
                    adapter="local-transformers",
                    endpoint_url=None,
                    model_id="google/gemma-4-E4B-it",
                ),
                b"audio",
                "audio/wav",
                "L",
                "ja",
            )

        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], ("worker", "--stdio"))
        request = json.loads(fake_process.stdin.getvalue())
        self.assertEqual(request["type"], "transcribe")
        self.assertEqual(request["model_id"], "google/gemma-4-E4B-it")
        self.assertEqual(request["channel"], "L")
        self.assertEqual(segments[0].text, "ねえ")

    def test_model_output_must_have_valid_timing(self):
        with self.assertRaisesRegex(ValueError, "end_ms must be greater"):
            parse_model_segments(
                json.dumps(
                    {
                        "segments": [
                            {
                                "start_ms": 1000,
                                "end_ms": 1000,
                                "channel": "MIX",
                                "kind": "speech",
                                "text": "だめ",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    def test_model_output_accepts_json_code_fence(self):
        segments = parse_model_segments(
            '```json\n{"segments":[{"start_ms":0,"end_ms":1,"channel":"R","kind":"speech","text":"はい"}]}\n```'
        )

        self.assertEqual(segments[0].channel, "R")
        self.assertEqual(segments[0].text, "はい")


if __name__ == "__main__":
    unittest.main()
