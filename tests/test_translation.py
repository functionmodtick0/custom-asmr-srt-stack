import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.cli import main
from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.srt import format_srt
from custom_asmr_srt_stack.translation import export_translation_json, parse_translated_texts


def sample_master() -> MasterDocument:
    return MasterDocument(
        source_language="ja",
        source_file="voice.wav",
        duration_ms=5000,
        segments=(
            Segment(
                id="seg_000001",
                start_ms=1000,
                end_ms=2000,
                channel="L",
                kind="speech",
                text="ねえ",
            ),
            Segment(
                id="seg_000002",
                start_ms=2100,
                end_ms=2600,
                channel="R",
                kind="breath",
                text="すう",
            ),
            Segment(
                id="seg_000003",
                start_ms=3000,
                end_ms=4000,
                channel="MIX",
                kind="speech",
                text="聞こえてる？",
            ),
        ),
    )


class TranslationJsonTests(unittest.TestCase):
    def test_translation_export_contains_only_ids_and_speech_text(self):
        exported = export_translation_json(sample_master())

        self.assertEqual(
            exported,
            {
                "format": "custom-asmr-translation-v1",
                "source_language": "ja",
                "items": [
                    {"id": "seg_000001", "text": "ねえ"},
                    {"id": "seg_000003", "text": "聞こえてる？"},
                ],
            },
        )

    def test_translated_texts_fail_on_missing_ids(self):
        translated = {
            "format": "custom-asmr-translated-v1",
            "source_language": "ja",
            "target_language": "ko",
            "items": [{"id": "seg_000001", "text": "저기"}],
        }

        with self.assertRaisesRegex(ValueError, "missing ids: seg_000003"):
            parse_translated_texts(sample_master(), translated)

    def test_translated_texts_fail_on_duplicate_ids(self):
        translated = {
            "format": "custom-asmr-translated-v1",
            "source_language": "ja",
            "target_language": "ko",
            "items": [
                {"id": "seg_000001", "text": "저기"},
                {"id": "seg_000001", "text": "저기"},
                {"id": "seg_000003", "text": "들려?"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "duplicate translated id"):
            parse_translated_texts(sample_master(), translated)

    def test_translated_srt_uses_original_timing_and_translated_text(self):
        translated = {
            "format": "custom-asmr-translated-v1",
            "source_language": "ja",
            "target_language": "ko",
            "items": [
                {"id": "seg_000001", "text": "저기"},
                {"id": "seg_000003", "text": "들려?"},
            ],
        }
        text_by_id = parse_translated_texts(sample_master(), translated)

        self.assertEqual(
            format_srt(sample_master(), text_by_id=text_by_id),
            "1\n00:00:01,000 --> 00:00:02,000\n저기\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n들려?\n",
        )

    def test_cli_exports_translation_json_and_translated_srt(self):
        master = sample_master()
        translated = {
            "format": "custom-asmr-translated-v1",
            "source_language": "ja",
            "target_language": "ko",
            "items": [
                {"id": "seg_000001", "text": "저기"},
                {"id": "seg_000003", "text": "들려?"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            master_path = root / "master.json"
            translation_path = root / "translation.json"
            translated_path = root / "translated.json"
            srt_path = root / "translated.srt"
            master_path.write_text(json.dumps(master.to_json(), ensure_ascii=False), encoding="utf-8")
            translated_path.write_text(json.dumps(translated, ensure_ascii=False), encoding="utf-8")

            self.assertEqual(main(["export-translation-json", str(master_path), "-o", str(translation_path)]), 0)
            self.assertEqual(
                json.loads(translation_path.read_text(encoding="utf-8"))["items"],
                [
                    {"id": "seg_000001", "text": "ねえ"},
                    {"id": "seg_000003", "text": "聞こえてる？"},
                ],
            )

            self.assertEqual(
                main(["json-to-srt", str(master_path), "-o", str(srt_path), "--translated", str(translated_path)]),
                0,
            )
            self.assertIn("저기", srt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
