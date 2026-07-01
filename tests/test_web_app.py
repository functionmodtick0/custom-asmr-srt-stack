import subprocess
import textwrap
import unittest
from html.parser import HTMLParser


class AdapterSelectParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_adapter_select = False
        self.options = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "select" and attrs_dict.get("id") == "adapterInput":
            self.in_adapter_select = True
        if self.in_adapter_select and tag == "option":
            self.options.append(attrs_dict.get("value"))

    def handle_endtag(self, tag):
        if tag == "select" and self.in_adapter_select:
            self.in_adapter_select = False


class WebAppBehaviorTests(unittest.TestCase):
    def run_app_assertions(self, assertions: str) -> subprocess.CompletedProcess[str]:
        script = r"""
            const assert = require("assert");
            const fs = require("fs");
            const vm = require("vm");

            const elements = new Map();
            function element(id) {
              if (!elements.has(id)) {
                elements.set(id, {
                  id,
                  value: "",
                  disabled: false,
                  hidden: false,
                  textContent: "",
                  style: {},
                  dataset: {},
                  children: [],
                  classList: { add() {}, remove() {}, toggle() {} },
                  addEventListener() {},
                  append(...children) { this.children.push(...children); },
                  replaceChildren(...children) { this.children = children; },
                  querySelectorAll() { return []; },
                  removeAttribute() {},
                  load() {},
                  close() {},
                  showModal() {},
                  click() {},
                  getBoundingClientRect() { return { width: 640, height: 120 }; },
                  getContext() {
                    return {
                      scale() {},
                      clearRect() {},
                      fillRect() {},
                      beginPath() {},
                      moveTo() {},
                      lineTo() {},
                      stroke() {},
                    };
                  },
                });
              }
              return elements.get(id);
            }

            const context = {
              console,
              require,
              Blob: class Blob {},
              Element: class Element {},
              URL: { createObjectURL() { return "blob:test"; }, revokeObjectURL() {} },
              localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
              window: {
                devicePixelRatio: 1,
                addEventListener() {},
                clearTimeout() {},
                setTimeout() { return 1; },
                URL: { createObjectURL() { return "blob:test"; }, revokeObjectURL() {} },
              },
              document: {
                getElementById: element,
                createElement(tag) {
                  const node = element(`created-${tag}-${Math.random()}`);
                  node.tagName = tag.toUpperCase();
                  return node;
                },
              },
              fetch() {
                throw new Error("fetch should not be called in this test");
              },
            };
            context.window.window = context.window;
            context.window.document = context.document;
            vm.createContext(context);
            vm.runInContext(fs.readFileSync("web/app.js", "utf8"), context);
        """
        result = subprocess.run(
            ["node", "-e", textwrap.dedent(script + "\n" + assertions)],
            check=False,
            text=True,
            capture_output=True,
        )
        return result

    def test_local_granite_adapter_does_not_require_endpoint_settings(self):
        result = self.run_app_assertions(
            r"""
            assert.strictEqual(context.isLocalAdapter("local-granite-asr"), true);
            const adapter = elements.get("adapterInput");
            const endpoint = elements.get("endpointInput");
            const apiKey = elements.get("apiKeyInput");
            const model = elements.get("modelInput");
            adapter.value = "local-granite-asr";
            endpoint.value = "http://127.0.0.1:8000/v1";
            apiKey.value = "secret";
            context.syncModelFormForAdapter();

            assert.strictEqual(endpoint.disabled, true);
            assert.strictEqual(apiKey.disabled, true);
            assert.strictEqual(endpoint.value, "");
            assert.strictEqual(apiKey.value, "");
            assert.match(model.placeholder, /granite-speech-4\.1-2b/);
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_case_preview_uses_first_remaining_review_segment(self):
        result = self.run_app_assertions(
            r"""
            const item = {
              reference_master: {
                segments: [
                  { id: "seg_000001", start_ms: 0, end_ms: 500, channel: "MIX", text: "済み", needs_review: false },
                  { id: "seg_000002", start_ms: 1234, end_ms: 3456, channel: "L", text: "確認する", needs_review: true },
                  { id: "seg_000003", start_ms: 5000, end_ms: 6000, channel: "R", text: "後", needs_review: true },
                ],
              },
            };

            const first = context.firstReviewSegment(item);
            assert.strictEqual(first.id, "seg_000002");
            assert.strictEqual(context.reviewSegmentPreview(first), "0:01.234 - 0:03.456 · 確認する");
            assert.strictEqual(context.reviewSegmentPreview(null), "검수 flag 없음");
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_case_list_shows_remaining_review_duration(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              const pathInput = elements.get("reviewPackPathInput");
              pathInput.value = "/cases";
              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review/load");
                assert.deepStrictEqual(JSON.parse(options.body), { path: "/cases" });
                return {
                  ok: true,
                  async json() {
                    return {
                      kind: "review-case-set",
                      items: [
                        {
                          id: "front-a",
                          reference_type: "pseudo-gold",
                          duration_ms: 120000,
                          segments: 3,
                          review_count: 2,
                          review_duration_ms: 3222,
                          audio: "audio/front-a.wav",
                          reference: "references/front-a.master.json",
                          reference_master: {
                            segments: [
                              { id: "seg_000001", start_ms: 0, end_ms: 1000, channel: "MIX", text: "済み", needs_review: false },
                              { id: "seg_000002", start_ms: 1234, end_ms: 3456, channel: "L", text: "確認する", needs_review: true },
                            ],
                          },
                        },
                        {
                          id: "front-b",
                          reference_type: "pseudo-gold",
                          duration_ms: 120000,
                          segments: 1,
                          review_count: 0,
                          review_duration_ms: 0,
                          audio: "audio/front-b.wav",
                          reference: "references/front-b.master.json",
                          reference_master: { segments: [] },
                        },
                      ],
                    };
                  },
                };
              };

              await context.loadReviewPath();

              assert.strictEqual(elements.get("segmentCount").textContent, "2 review cases · 2 flags · 0:03.222");
              const firstRow = elements.get("segmentList").children[0];
              const counts = firstRow.children[1];
              assert.strictEqual(counts.children[0].textContent, "3 segments");
              assert.strictEqual(counts.children[1].textContent, "2 review flags");
              assert.strictEqual(counts.children[2].textContent, "0:03.222");
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_pack_item_hides_candidate_row_when_candidate_is_absent(self):
        result = self.run_app_assertions(
            r"""
            const referenceOnly = {
              priority_rank: 1,
              case_id: "front-a",
              reference_id: "seg_000002",
              start_ms: 1000,
              end_ms: 2000,
              reasons: ["reference-needs-review"],
              reference_channel: "L",
              reference_text: "確認",
              candidate_id: null,
              candidate_channel: null,
              candidate_text: "",
            };
            const candidateItem = {
              ...referenceOnly,
              candidate_id: "seg_000002",
              candidate_channel: "R",
              candidate_text: "候補",
              reasons: ["text"],
            };
            const auditOverlapItem = {
              ...referenceOnly,
              reference_id: "seg_000001",
              candidate_id: "seg_000002",
              candidate_channel: "L",
              candidate_text: "",
              reasons: ["reference-same-channel-overlap"],
            };
            const channelAuditItem = {
              ...referenceOnly,
              candidate_id: null,
              candidate_channel: "R",
              candidate_text: "",
              reasons: ["reference-channel-energy-mismatch"],
              left_dbfs: -37.536,
              right_dbfs: -32.968,
              delta_db: -4.568,
            };

            const referenceRow = context.renderReviewPackItem(referenceOnly, 0);
            const candidateRow = context.renderReviewPackItem(candidateItem, 1);
            const auditOverlapRow = context.renderReviewPackItem(auditOverlapItem, 2);
            const channelAuditRow = context.renderReviewPackItem(channelAuditItem, 3);
            const referenceMeta = referenceRow.children[0];
            const referenceTexts = referenceRow.children[2];
            const candidateTexts = candidateRow.children[2];
            const auditOverlapTexts = auditOverlapRow.children[2];
            const channelAuditTexts = channelAuditRow.children[2];

            assert.strictEqual(referenceMeta.children[3].textContent, "seg_000002");
            assert.strictEqual(context.reviewPackHasCandidate(referenceOnly), false);
            assert.strictEqual(context.reviewPackHasCandidate(candidateItem), true);
            assert.strictEqual(context.reviewPackHasCandidate(auditOverlapItem), true);
            assert.strictEqual(context.reviewPackHasCandidate(channelAuditItem), true);
            assert.strictEqual(referenceTexts.children.length, 1);
            assert.strictEqual(referenceTexts.children[0].children[0].textContent, "REF L");
            assert.strictEqual(candidateTexts.children.length, 2);
            assert.strictEqual(candidateTexts.children[1].children[0].textContent, "CAND R");
            assert.strictEqual(auditOverlapTexts.children.length, 2);
            assert.strictEqual(auditOverlapTexts.children[1].children[0].textContent, "REF2 L");
            assert.strictEqual(auditOverlapTexts.children[1].children[1].textContent, "seg_000002");
            assert.strictEqual(channelAuditTexts.children.length, 2);
            assert.strictEqual(channelAuditTexts.children[1].children[0].textContent, "ENERGY R");
            assert.strictEqual(
              channelAuditTexts.children[1].children[1].textContent,
              "L -37.5 dBFS · R -33.0 dBFS · delta -4.6 dB",
            );
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_pack_source_case_button_opens_referenced_case_segment(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              const sourceButton = elements.get("sourceCaseButton");
              const pathInput = elements.get("reviewPackPathInput");
              pathInput.value = "/packs/review-case-pack";
              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review/load");
                assert.deepStrictEqual(JSON.parse(options.body), { path: "/packs/review-case-pack" });
                return {
                  ok: true,
                  async json() {
                    return {
                      kind: "review-pack",
                      source_case_index: "/cases/case-index.json",
                      items: [
                        {
                          priority_rank: 1,
                          case_id: "front-a",
                          reference_id: "seg_000002",
                          start_ms: 1000,
                          end_ms: 2000,
                          reasons: ["reference-needs-review"],
                          reference_channel: "L",
                          reference_text: "確認",
                          clip_url: "/api/review-pack/clip?x=1",
                        },
                      ],
                    };
                  },
                };
              };

              await context.loadReviewPath();
              assert.strictEqual(sourceButton.hidden, false);
              assert.strictEqual(sourceButton.disabled, true);

              context.selectReviewPackItem(0, false);
              assert.strictEqual(sourceButton.disabled, false);

              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review-case/load");
                assert.deepStrictEqual(JSON.parse(options.body), { path: "/cases/case-index.json" });
                return {
                  ok: true,
                  async json() {
                    return {
                      kind: "review-case-set",
                      case_index_path: "/cases/case-index.json",
                      items: [
                        {
                          id: "front-a",
                          audio_url: "/api/review-case/audio?x=1",
                          reference_master: {
                            format: "custom-asmr-master-v1",
                            source_language: "ja",
                            audio: { source_file: "front-a.wav", duration_ms: 3000 },
                            segments: [
                              { id: "seg_000001", start_ms: 0, end_ms: 1000, channel: "MIX", kind: "speech", text: "前", needs_review: false },
                              { id: "seg_000002", start_ms: 1000, end_ms: 2000, channel: "L", kind: "speech", text: "確認", needs_review: true },
                            ],
                          },
                        },
                      ],
                    };
                  },
                };
              };

              await context.openSelectedReviewPackSourceCase();
              assert.strictEqual(elements.get("audioPlayer").src, "/api/review-case/audio?x=1");
              assert.strictEqual(elements.get("segmentCount").textContent, "front-a · 2 segments");
              assert.strictEqual(elements.get("selectedLabel").textContent, "seg_000002");
              assert.strictEqual(elements.get("reviewDoneButton").hidden, false);
              assert.strictEqual(elements.get("sourceCaseButton").hidden, true);
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_pack_source_case_keeps_channel_audit_status_hint(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              const pathInput = elements.get("reviewPackPathInput");
              pathInput.value = "/packs/channel-audit-pack";
              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review/load");
                assert.deepStrictEqual(JSON.parse(options.body), { path: "/packs/channel-audit-pack" });
                return {
                  ok: true,
                  async json() {
                    return {
                      kind: "review-pack",
                      source_case_index: "/cases/case-index.json",
                      items: [
                        {
                          priority_rank: 1,
                          case_id: "front-a",
                          reference_id: "seg_000002",
                          start_ms: 1000,
                          end_ms: 2000,
                          reasons: ["reference-channel-energy-mismatch"],
                          reference_channel: "L",
                          reference_text: "",
                          candidate_channel: "R",
                          left_dbfs: -37.536,
                          right_dbfs: -32.968,
                          delta_db: -4.568,
                          clip_url: "/api/review-pack/clip?x=1",
                        },
                      ],
                    };
                  },
                };
              };

              await context.loadReviewPath();
              context.selectReviewPackItem(0, false);

              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review-case/load");
                assert.deepStrictEqual(JSON.parse(options.body), { path: "/cases/case-index.json" });
                return {
                  ok: true,
                  async json() {
                    return {
                      kind: "review-case-set",
                      case_index_path: "/cases/case-index.json",
                      items: [
                        {
                          id: "front-a",
                          audio_url: "/api/review-case/audio?x=1",
                          reference_master: {
                            format: "custom-asmr-master-v1",
                            source_language: "ja",
                            audio: { source_file: "front-a.wav", duration_ms: 3000 },
                            segments: [
                              { id: "seg_000002", start_ms: 1000, end_ms: 2000, channel: "L", kind: "speech", text: "確認", needs_review: true },
                            ],
                          },
                        },
                      ],
                    };
                  },
                };
              };

              await context.openSelectedReviewPackSourceCase();
              assert.strictEqual(elements.get("selectedLabel").textContent, "seg_000002");
              assert.strictEqual(
                elements.get("statusText").textContent,
                "front-a/seg_000002 · ENERGY R · L -37.5 dBFS · R -33.0 dBFS · delta -4.6 dB",
              );
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_adapter_select_exposes_all_local_adapters(self):
        parser = AdapterSelectParser()
        with open("web/index.html", encoding="utf-8") as html_file:
            parser.feed(html_file.read())

        self.assertIn("local-transformers", parser.options)
        self.assertIn("local-qwen-asr", parser.options)
        self.assertIn("local-qwen-hf-asr", parser.options)
        self.assertIn("local-cohere-asr", parser.options)
        self.assertIn("local-granite-asr", parser.options)


if __name__ == "__main__":
    unittest.main()
