import json
import unittest

from custom_asmr_srt_stack.transcription import (
    ModelEndpoint,
    build_gemini_url,
    parse_model_segments,
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
