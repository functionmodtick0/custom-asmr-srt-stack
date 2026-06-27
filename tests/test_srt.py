import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.cli import main
from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.srt import format_srt, parse_srt


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:02,500
ねえ
聞こえてる？

2
00:00:03.000 --> 00:00:04.000
うん
"""


class SrtConversionTests(unittest.TestCase):
    def test_parse_srt_creates_master_segments(self):
        master = parse_srt(SAMPLE_SRT, source_file="voice.wav")

        self.assertEqual(master.source_language, "ja")
        self.assertEqual(master.source_file, "voice.wav")
        self.assertEqual(master.duration_ms, 4000)
        self.assertEqual([segment.id for segment in master.segments], ["seg_000001", "seg_000002"])
        self.assertEqual(master.segments[0].channel, "MIX")
        self.assertEqual(master.segments[0].kind, "speech")
        self.assertEqual(master.segments[0].text, "ねえ\n聞こえてる？")

    def test_parse_srt_extracts_channel_label_from_text(self):
        master = parse_srt(
            "1\n00:00:01,000 --> 00:00:02,500\n[L] ねえ\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n[LR] うん\n"
        )

        self.assertEqual(master.segments[0].channel, "L")
        self.assertEqual(master.segments[0].text, "ねえ")
        self.assertEqual(master.segments[1].channel, "MIX")
        self.assertEqual(master.segments[1].text, "うん")

    def test_format_srt_exports_text_without_channel_labels(self):
        master = MasterDocument(
            source_language="ja",
            source_file="voice.wav",
            duration_ms=3000,
            segments=(
                Segment(
                    id="seg_000001",
                    start_ms=1000,
                    end_ms=2500,
                    channel="L",
                    kind="speech",
                    text="ねえ",
                ),
            ),
        )

        self.assertEqual(
            format_srt(master),
            "1\n00:00:01,000 --> 00:00:02,500\nねえ\n",
        )

    def test_parse_srt_rejects_invalid_timing(self):
        with self.assertRaisesRegex(ValueError, "end_ms must be greater"):
            parse_srt("1\n00:00:02,000 --> 00:00:01,000\n逆\n")

    def test_cli_converts_srt_to_json_and_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            srt_path = root / "input.srt"
            json_path = root / "master.json"
            out_srt_path = root / "out.srt"
            srt_path.write_text(SAMPLE_SRT, encoding="utf-8")

            self.assertEqual(
                main(["srt-to-json", str(srt_path), "-o", str(json_path), "--source-file", "voice.wav"]),
                0,
            )
            parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed_json["audio"]["source_file"], "voice.wav")

            self.assertEqual(main(["json-to-srt", str(json_path), "-o", str(out_srt_path)]), 0)
            self.assertEqual(out_srt_path.read_text(encoding="utf-8"), SAMPLE_SRT.replace(".", ","))


if __name__ == "__main__":
    unittest.main()
