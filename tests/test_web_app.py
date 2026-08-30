import json
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


class ElementIdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        del tag
        element_id = dict(attrs).get("id")
        if element_id:
            self.ids.add(element_id)


class WebAppBehaviorTests(unittest.TestCase):
    def run_app_assertions(
        self,
        assertions: str,
        *,
        missing_element_ids: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        script = r"""
            const assert = require("assert");
            const fs = require("fs");
            const vm = require("vm");

            const elements = new Map();
            const missingElementIds = new Set(MISSING_ELEMENT_IDS);
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
                  play() {},
                  pause() {},
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
                getElementById(id) {
                  return missingElementIds.has(id) ? null : element(id);
                },
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
        """.replace("MISSING_ELEMENT_IDS", json.dumps(missing_element_ids))
        result = subprocess.run(
            ["node", "-e", textwrap.dedent(script + "\n" + assertions)],
            check=False,
            text=True,
            capture_output=True,
        )
        return result

    def test_primary_app_starts_without_diagnostics_controls(self):
        diagnostics_ids = (
            "reviewPackPathInput",
            "loadReviewPackButton",
            "sourceCaseButton",
            "applyEnergyChannelButton",
            "reviewDoneButton",
            "caseListButton",
            "nextCaseButton",
        )
        result = self.run_app_assertions(
            r"""
            assert.strictEqual(elements.get("segmentCount").textContent, "0 segments");
            for (const id of missingElementIds) {
              assert.strictEqual(elements.has(id), false);
            }
        """,
            missing_element_ids=diagnostics_ids,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_controls_are_isolated_to_diagnostics_page(self):
        primary = ElementIdParser()
        diagnostics = ElementIdParser()
        with open("web/index.html", encoding="utf-8") as html_file:
            primary.feed(html_file.read())
        with open("web/diagnostics.html", encoding="utf-8") as html_file:
            diagnostics.feed(html_file.read())

        review_ids = {
            "reviewPackPathInput",
            "loadReviewPackButton",
            "sourceCaseButton",
            "applyEnergyChannelButton",
            "reviewDoneButton",
            "caseListButton",
            "nextCaseButton",
        }
        self.assertTrue(review_ids.isdisjoint(primary.ids))
        self.assertTrue(review_ids.issubset(diagnostics.ids))

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

    def test_local_qwen_hf_adapter_points_to_the_selected_product_model(self):
        result = self.run_app_assertions(
            r"""
            assert.strictEqual(
              context.modelPlaceholderForAdapter("local-qwen-hf-asr"),
              ".casrt/models/qwen3-asr-1.7b-ja-anime-galgame-hf-5a6a789ceb2f22d2b8606743b13a8159af218362",
            );
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_case_preview_uses_first_remaining_review_segment(self):
        result = self.run_app_assertions(
            r"""
            const item = {
              reference_master: {
                segments: [
                  { id: "seg_000001", start_ms: 0, end_ms: 500, channel: "MIX", text: "済み", needs_review: false, content_reviewed: true },
                  { id: "seg_000002", start_ms: 1234, end_ms: 3456, channel: "L", text: "確認する", needs_review: false, content_reviewed: false },
                  { id: "seg_000003", start_ms: 5000, end_ms: 6000, channel: "R", text: "後", needs_review: true },
                ],
              },
            };

            const first = context.firstReviewSegment(item);
            assert.strictEqual(first.id, "seg_000002");
            assert.strictEqual(context.reviewSegmentPreview(first), "0:01.234 - 0:03.456 · 確認する");
            assert.strictEqual(context.reviewSegmentPreview(null), "미검수 없음");
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_content_review_completion_and_edits_update_review_evidence(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              const segment = {
                id: "seg_000001",
                start_ms: 0,
                end_ms: 1000,
                channel: "L",
                kind: "speech",
                text: "確認",
                needs_review: false,
                content_reviewed: false,
                channel_reviewed: true,
              };
              const master = {
                format: "custom-asmr-master-v1",
                source_language: "ja",
                audio: { source_file: "front.wav", duration_ms: 2000 },
                segments: [segment],
              };
              elements.get("reviewPackPathInput").value = "/cases";
              context.fetch = async () => ({
                ok: true,
                async json() {
                  return {
                    kind: "review-case-set",
                    case_index_path: "/cases/case-index.json",
                    items: [
                      {
                        id: "front",
                        audio_url: "/api/review-case/audio?x=1",
                        reference_master: master,
                      },
                    ],
                  };
                },
              });
              await context.loadReviewPath();
              context.loadReviewCaseItem(0);
              assert.strictEqual(elements.get("reviewDoneButton").disabled, false);

              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review-case/save-reference");
                const payload = JSON.parse(options.body);
                assert.strictEqual(payload.master.segments[0].needs_review, false);
                assert.strictEqual(payload.master.segments[0].content_reviewed, true);
                assert.strictEqual(payload.master.segments[0].channel_reviewed, true);
                return {
                  ok: true,
                  async json() {
                    return {
                      segments: 1,
                      review_count: 0,
                      review_duration_ms: 0,
                      content_reviewed_count: 1,
                      content_unreviewed_count: 0,
                      content_unreviewed_duration_ms: 0,
                    };
                  },
                };
              };

              await context.markSelectedReviewDone();
              assert.strictEqual(segment.content_reviewed, true);
              assert.strictEqual(elements.get("reviewDoneButton").disabled, true);

              context.commitSegmentTime(segment, "start_ms", { value: "100" });
              assert.strictEqual(segment.content_reviewed, false);
              assert.strictEqual(segment.channel_reviewed, false);

              segment.content_reviewed = true;
              assert.strictEqual(context.commitSegmentText(segment, "修正"), true);
              assert.strictEqual(segment.text, "修正");
              assert.strictEqual(segment.content_reviewed, false);

              segment.content_reviewed = true;
              segment.channel_reviewed = true;
              assert.strictEqual(context.commitSegmentChannel(segment, "R"), true);
              assert.strictEqual(segment.channel, "R");
              assert.strictEqual(segment.content_reviewed, true);
              assert.strictEqual(segment.channel_reviewed, false);

              context.setSegmentReviewFlag(segment, true);
              assert.strictEqual(segment.needs_review, true);
              assert.strictEqual(segment.content_reviewed, false);
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_segment_playback_waits_for_audio_metadata(self):
        result = self.run_app_assertions(
            r"""
            const player = elements.get("audioPlayer");
            let loadedData = null;
            let playCount = 0;
            player.readyState = 0;
            player.currentTime = 0;
            player.addEventListener = (type, callback, options) => {
              assert.strictEqual(type, "loadeddata");
              assert.strictEqual(options.once, true);
              loadedData = callback;
            };
            player.play = () => { playCount += 1; };

            context.playSegment({ id: "seg_000001", start_ms: 98513, end_ms: 120000 });

            assert.strictEqual(player.currentTime, 0);
            assert.strictEqual(playCount, 0);
            assert.ok(loadedData);

            player.readyState = 2;
            loadedData();

            assert.strictEqual(player.currentTime, 98.513);
            assert.strictEqual(playCount, 1);
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_case_editor_shows_overlapping_candidate_context(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              elements.get("reviewPackPathInput").value = "/cases";
              context.fetch = async () => ({
                ok: true,
                async json() {
                  return {
                    kind: "review-case-set",
                    case_index_path: "/cases/case-index.json",
                    items: [
                      {
                        id: "front-a",
                        audio_url: "/api/review-case/audio?x=1",
                        candidate_id: "bro-stereo",
                        reference_master: {
                          format: "custom-asmr-master-v1",
                          source_language: "ja",
                          audio: { source_file: "front-a.wav", duration_ms: 3000 },
                          segments: [
                            { id: "ref_1", start_ms: 0, end_ms: 2000, channel: "L", kind: "speech", text: "基準", needs_review: true },
                          ],
                        },
                        candidate_master: {
                          format: "custom-asmr-master-v1",
                          source_language: "ja",
                          audio: { source_file: "front-a.wav", duration_ms: 3000 },
                          segments: [
                            { id: "cand_1", start_ms: 250, end_ms: 1250, channel: "L", kind: "speech", text: "候補一", needs_review: true },
                            { id: "cand_2", start_ms: 1500, end_ms: 2500, channel: "R", kind: "speech", text: "候補二", needs_review: true },
                            { id: "cand_3", start_ms: 1750, end_ms: 2250, channel: "MIX", kind: "speech", text: "混合候補", needs_review: true },
                            { id: "cand_4", start_ms: 2500, end_ms: 3000, channel: "L", kind: "speech", text: "範囲外", needs_review: true },
                          ],
                        },
                      },
                    ],
                  };
                },
              });

              await context.loadReviewPath();
              context.loadReviewCaseItem(0);

              assert.strictEqual(elements.get("segmentCount").textContent, "front-a · 1 ref · 4 cand");
              const row = elements.get("segmentList").children[0];
              const textStack = row.children[2];
              assert.strictEqual(textStack.className, "segment-text-stack");
              const candidateContext = textStack.children[1];
              assert.strictEqual(candidateContext.className, "candidate-context");
              assert.strictEqual(candidateContext.children.length, 2);
              assert.match(candidateContext.children[0].textContent, /CAND L/);
              assert.match(candidateContext.children[0].textContent, /候補一/);
              assert.match(candidateContext.children[1].textContent, /CAND MIX/);
              assert.match(candidateContext.children[1].textContent, /混合候補/);
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
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
                          content_unreviewed_count: 2,
                          content_unreviewed_duration_ms: 3222,
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
                          content_unreviewed_count: 0,
                          content_unreviewed_duration_ms: 0,
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

              assert.strictEqual(
                elements.get("segmentCount").textContent,
                "2 review cases · 2 content pending · 2 flags · 0:03.222",
              );
              const firstRow = elements.get("segmentList").children[0];
              const counts = firstRow.children[1];
              assert.strictEqual(counts.children[0].textContent, "3 segments");
              assert.strictEqual(counts.children[1].textContent, "2 content pending");
              assert.strictEqual(counts.children[2].textContent, "2 review flags");
              assert.strictEqual(counts.children[3].textContent, "0:03.222");
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
              review_clip_start_ms: 1200,
              review_clip_end_ms: 1700,
            };
            const mergedChannelAuditItem = {
              ...channelAuditItem,
              reasons: ["reference-needs-review", "reference-channel-energy-mismatch"],
            };

            const referenceRow = context.renderReviewPackItem(referenceOnly, 0);
            const candidateRow = context.renderReviewPackItem(candidateItem, 1);
            const auditOverlapRow = context.renderReviewPackItem(auditOverlapItem, 2);
            const channelAuditRow = context.renderReviewPackItem(channelAuditItem, 3);
            const mergedChannelAuditRow = context.renderReviewPackItem(mergedChannelAuditItem, 4);
            const referenceMeta = referenceRow.children[0];
            const channelAuditMeta = channelAuditRow.children[0];
            const referenceTexts = referenceRow.children[2];
            const candidateTexts = candidateRow.children[2];
            const auditOverlapTexts = auditOverlapRow.children[2];
            const channelAuditTexts = channelAuditRow.children[2];
            const mergedChannelAuditTexts = mergedChannelAuditRow.children[2];

            assert.strictEqual(referenceMeta.children[3].textContent, "seg_000002");
            assert.strictEqual(channelAuditMeta.children[4].textContent, "focus 0:01.200 - 0:01.700");
            assert.strictEqual(context.reviewPackHasCandidate(referenceOnly), false);
            assert.strictEqual(context.reviewPackHasCandidate(candidateItem), true);
            assert.strictEqual(context.reviewPackHasCandidate(auditOverlapItem), true);
            assert.strictEqual(context.reviewPackHasCandidate(channelAuditItem), true);
            assert.strictEqual(context.reviewPackHasCandidate(mergedChannelAuditItem), true);
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
            assert.strictEqual(mergedChannelAuditTexts.children.length, 2);
            assert.strictEqual(mergedChannelAuditTexts.children[1].children[0].textContent, "ENERGY R");
            assert.strictEqual(
              mergedChannelAuditTexts.children[1].children[1].textContent,
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
                          candidate_id: "seg_000001",
                          start_ms: 1000,
                          end_ms: 2000,
                          reasons: ["reference-same-channel-overlap"],
                          reference_channel: "L",
                          reference_text: "確認",
                          candidate_channel: "L",
                          overlap_ms: 20,
                          clip_url: "/api/review-pack/clip?x=1",
                        },
                      ],
                    };
                  },
                };
              };

              await context.loadReviewPath();
              assert.strictEqual(sourceButton.hidden, false);
              assert.strictEqual(sourceButton.disabled, false);

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
              assert.ok(elements.get("segmentList").children[0].className.includes("is-secondary-reference"));
              assert.ok(!elements.get("segmentList").children[1].className.includes("is-secondary-reference"));
              assert.strictEqual(elements.get("reviewDoneButton").hidden, false);
              assert.strictEqual(elements.get("sourceCaseButton").hidden, true);
              assert.strictEqual(
                elements.get("statusText").textContent,
                "front-a/seg_000002 · reference-same-channel-overlap · REF L · REF2 L seg_000001 · overlap 0:00.020",
              );
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_pack_hides_resolved_items_and_skips_their_priority_slots(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              elements.get("reviewPackPathInput").value = "/packs/resumable";
              context.fetch = async () => ({
                ok: true,
                async json() {
                  return {
                    kind: "review-pack",
                    source_case_index: "/cases/case-index.json",
                    total_item_count: 3,
                    pending_item_count: 2,
                    resolved_item_count: 1,
                    duration_summary: {
                      source_item_duration_ms_sum: 3000,
                      effective_item_duration_ms_sum: 3000,
                      clip_duration_ms_sum: 3600,
                      clip_duration_ms_max: 1200,
                      focus_item_count: 0,
                    },
                    items: [
                      {
                        priority_rank: 1,
                        case_id: "front-a",
                        reference_id: "seg_done",
                        start_ms: 0,
                        end_ms: 1000,
                        clip_start_ms: 0,
                        clip_end_ms: 1200,
                        reasons: ["reference-content-unreviewed"],
                        source_review_resolved: true,
                        clip_url: "/api/review-pack/clip?x=1",
                      },
                      {
                        priority_rank: 2,
                        case_id: "front-a",
                        reference_id: "seg_pending_1",
                        start_ms: 1000,
                        end_ms: 2000,
                        clip_start_ms: 900,
                        clip_end_ms: 2100,
                        reasons: ["reference-content-unreviewed"],
                        source_review_resolved: false,
                        clip_url: "/api/review-pack/clip?x=2",
                      },
                      {
                        priority_rank: 3,
                        case_id: "front-b",
                        reference_id: "seg_pending_2",
                        start_ms: 2000,
                        end_ms: 3000,
                        clip_start_ms: 1900,
                        clip_end_ms: 3100,
                        reasons: ["reference-content-unreviewed"],
                        source_review_resolved: false,
                        clip_url: "/api/review-pack/clip?x=3",
                      },
                    ],
                  };
                },
              });

              await context.loadReviewPath();

              assert.strictEqual(elements.get("segmentCount").textContent, "2 pending · 3 total · listen 0:02.400");
              assert.strictEqual(elements.get("segmentList").children.length, 2);
              assert.strictEqual(elements.get("segmentList").children[0].dataset.index, "1");
              assert.strictEqual(elements.get("segmentList").children[1].dataset.index, "2");
              assert.strictEqual(context.nextReviewPackIndex(), 1);
              assert.strictEqual(context.reviewPackSelectedOrDefaultSourceItem().reference_id, "seg_pending_1");

              context.selectReviewPackItem(1, false);
              assert.strictEqual(context.nextReviewPackIndex(), 2);
              context.selectReviewPackItem(2, false);
              assert.strictEqual(context.nextReviewPackIndex(), null);
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_pack_source_case_button_uses_next_case_without_selected_clip(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              const sourceButton = elements.get("sourceCaseButton");
              const pathInput = elements.get("reviewPackPathInput");
              pathInput.value = "/packs/combined-reference-pack";
              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review/load");
                assert.deepStrictEqual(JSON.parse(options.body), { path: "/packs/combined-reference-pack" });
                return {
                  ok: true,
                  async json() {
                    return {
                      kind: "review-pack",
                      source_case_index: "/cases/case-index.json",
                      case_count: 2,
                      next_case_id: "front-b",
                      duration_summary: {
                        source_item_duration_ms_sum: 10000,
                        effective_item_duration_ms_sum: 5000,
                        clip_duration_ms_sum: 6000,
                        clip_duration_ms_max: 6000,
                        focus_item_count: 2,
                      },
                      items: [
                        {
                          priority_rank: 1,
                          case_id: "front-a",
                          reference_id: "seg_000001",
                          start_ms: 0,
                          end_ms: 1000,
                          reasons: ["reference-channel-energy-mismatch"],
                          clip_url: "/api/review-pack/clip?x=1",
                        },
                        {
                          priority_rank: 2,
                          case_id: "front-b",
                          reference_id: "seg_000003",
                          start_ms: 2000,
                          end_ms: 3000,
                          reasons: ["reference-needs-review"],
                          clip_url: "/api/review-pack/clip?x=2",
                        },
                      ],
                    };
                  },
                };
              };

              await context.loadReviewPath();
              assert.strictEqual(
                elements.get("segmentCount").textContent,
                "2 review clips · 2 cases · listen 0:06.000 · focus 0:05.000/0:10.000 · next front-b",
              );
              assert.strictEqual(sourceButton.hidden, false);
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
                          audio_url: "/api/review-case/audio?a=1",
                          reference_master: {
                            format: "custom-asmr-master-v1",
                            source_language: "ja",
                            audio: { source_file: "front-a.wav", duration_ms: 3000 },
                            segments: [
                              { id: "seg_000001", start_ms: 0, end_ms: 1000, channel: "L", kind: "speech", text: "前", needs_review: true },
                            ],
                          },
                        },
                        {
                          id: "front-b",
                          audio_url: "/api/review-case/audio?b=1",
                          reference_master: {
                            format: "custom-asmr-master-v1",
                            source_language: "ja",
                            audio: { source_file: "front-b.wav", duration_ms: 3000 },
                            segments: [
                              { id: "seg_000003", start_ms: 2000, end_ms: 3000, channel: "R", kind: "speech", text: "次", needs_review: true },
                            ],
                          },
                        },
                      ],
                    };
                  },
                };
              };

              await context.openSelectedReviewPackSourceCase();
              assert.strictEqual(elements.get("audioPlayer").src, "/api/review-case/audio?b=1");
              assert.strictEqual(elements.get("segmentCount").textContent, "front-b · 1 segments");
              assert.strictEqual(elements.get("selectedLabel").textContent, "seg_000003");
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_pack_next_clip_button_advances_priority_items(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              const pathInput = elements.get("reviewPackPathInput");
              pathInput.value = "/packs/review-pack";
              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review/load");
                assert.deepStrictEqual(JSON.parse(options.body), { path: "/packs/review-pack" });
                return {
                  ok: true,
                  async json() {
                    return {
                      kind: "review-pack",
                      items: [
                        {
                          priority_rank: 1,
                          start_ms: 0,
                          end_ms: 1000,
                          reasons: ["reference-channel-energy-mismatch"],
                          clip_url: "/api/review-pack/clip?x=1",
                        },
                        {
                          priority_rank: 2,
                          start_ms: 2000,
                          end_ms: 3000,
                          reasons: ["reference-needs-review"],
                          clip_url: "/api/review-pack/clip?x=2",
                        },
                      ],
                    };
                  },
                };
              };

              await context.loadReviewPath();
              assert.strictEqual(elements.get("segmentCount").textContent, "2 review clips");
              assert.strictEqual(elements.get("nextCaseButton").hidden, false);
              assert.strictEqual(elements.get("nextCaseButton").textContent, "다음 clip");
              assert.strictEqual(elements.get("nextCaseButton").disabled, false);

              await context.openNextAction();
              assert.strictEqual(elements.get("selectedLabel").textContent, "#1 0:00.000 - 0:01.000");
              assert.strictEqual(elements.get("audioPlayer").src, "/api/review-pack/clip?x=1");
              assert.strictEqual(elements.get("nextCaseButton").disabled, false);

              await context.openNextAction();
              assert.strictEqual(elements.get("selectedLabel").textContent, "#2 0:02.000 - 0:03.000");
              assert.strictEqual(elements.get("audioPlayer").src, "/api/review-pack/clip?x=2");
              assert.strictEqual(elements.get("nextCaseButton").disabled, true);
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_pack_source_editor_advances_directly_to_next_issue(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              const pathInput = elements.get("reviewPackPathInput");
              pathInput.value = "/packs/channel-audit-pack";
              context.fetch = async (path) => {
                assert.strictEqual(path, "/api/review/load");
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
                          reference_id: "seg_000001",
                          start_ms: 0,
                          end_ms: 1000,
                          reasons: ["reference-channel-energy-mismatch"],
                          reference_channel: "L",
                          candidate_channel: "R",
                          clip_url: "/api/review-pack/clip?x=1",
                        },
                        {
                          priority_rank: 2,
                          case_id: "front-b",
                          reference_id: "seg_000002",
                          start_ms: 1000,
                          end_ms: 2000,
                          reasons: ["reference-channel-energy-uncertain"],
                          reference_channel: "R",
                          candidate_channel: "MIX",
                          clip_url: "/api/review-pack/clip?x=2",
                        },
                      ],
                    };
                  },
                };
              };
              await context.loadReviewPath();
              context.selectReviewPackItem(0, false);

              context.fetch = async (path) => {
                assert.strictEqual(path, "/api/review-case/load");
                return {
                  ok: true,
                  async json() {
                    return {
                      kind: "review-case-set",
                      case_index_path: "/cases/case-index.json",
                      items: [
                        {
                          id: "front-a",
                          audio_url: "/api/review-case/audio?a=1",
                          reference_master: {
                            segments: [
                              { id: "seg_000001", start_ms: 0, end_ms: 1000, channel: "L", kind: "speech", text: "前", needs_review: false },
                            ],
                          },
                        },
                        {
                          id: "front-b",
                          audio_url: "/api/review-case/audio?b=1",
                          reference_master: {
                            segments: [
                              { id: "seg_000002", start_ms: 1000, end_ms: 2000, channel: "R", kind: "speech", text: "次", needs_review: false },
                            ],
                          },
                        },
                      ],
                    };
                  },
                };
              };
              await context.openSelectedReviewPackSourceCase();
              assert.strictEqual(elements.get("nextCaseButton").textContent, "다음 issue");
              assert.strictEqual(elements.get("nextCaseButton").disabled, false);

              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review-case/save-reference");
                const payload = JSON.parse(options.body);
                assert.ok(["front-a", "front-b"].includes(payload.case_id));
                return {
                  ok: true,
                  async json() {
                    return { segments: 1, review_count: 0, review_duration_ms: 0 };
                  },
                };
              };
              await context.openNextAction();

              assert.strictEqual(elements.get("audioPlayer").src, "/api/review-case/audio?b=1");
              assert.strictEqual(elements.get("segmentCount").textContent, "front-b · 1 segments");
              assert.strictEqual(elements.get("selectedLabel").textContent, "seg_000002");
              assert.strictEqual(elements.get("nextCaseButton").textContent, "다음 issue");
              assert.strictEqual(elements.get("nextCaseButton").disabled, true);

              await context.returnToReviewCases();
              assert.strictEqual(elements.get("selectedLabel").textContent, "#2 0:01.000 - 0:02.000");
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
                          reasons: ["reference-needs-review", "reference-channel-energy-mismatch"],
                          reference_channel: "L",
                          reference_text: "",
                          candidate_channel: "R",
                          left_dbfs: -37.536,
                          right_dbfs: -32.968,
                          delta_db: -4.568,
                          review_clip_start_ms: 1200,
                          review_clip_end_ms: 1700,
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
              assert.strictEqual(elements.get("applyEnergyChannelButton").hidden, false);
              assert.strictEqual(elements.get("applyEnergyChannelButton").disabled, false);
              assert.strictEqual(elements.get("applyEnergyChannelButton").textContent, "ENERGY R 적용");
              assert.strictEqual(elements.get("caseListButton").textContent, "pack 목록");
              assert.strictEqual(
                elements.get("statusText").textContent,
                "front-a/seg_000002 · ENERGY R · L -37.5 dBFS · R -33.0 dBFS · delta -4.6 dB · focus 0:01.200 - 0:01.700",
              );
              context.playSegment({ id: "seg_000002", start_ms: 1000, end_ms: 2000 });
              assert.strictEqual(elements.get("audioPlayer").currentTime, 1.2);

              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review-case/save-reference");
                const payload = JSON.parse(options.body);
                assert.strictEqual(payload.case_index_path, "/cases/case-index.json");
                assert.strictEqual(payload.case_id, "front-a");
                assert.strictEqual(payload.master.segments[0].channel, "R");
                assert.strictEqual(payload.master.segments[0].needs_review, true);
                assert.strictEqual(payload.master.segments[0].channel_reviewed, false);
                return {
                  ok: true,
                  async json() {
                    return {
                      segments: 1,
                      review_count: 1,
                      review_duration_ms: 1000,
                    };
                  },
                };
              };

              await context.applyEnergyChannelToSelectedSegment();
              assert.strictEqual(elements.get("applyEnergyChannelButton").disabled, true);
              assert.strictEqual(
                elements.get("statusText").textContent,
                "seg_000002 channel을 ENERGY R로 저장했습니다.",
              );

              await context.returnToReviewCases();
              assert.strictEqual(elements.get("segmentCount").textContent, "1 review clips");
              assert.strictEqual(elements.get("selectedLabel").textContent, "#1 0:01.000 - 0:02.000");
              assert.strictEqual(elements.get("caseListButton").hidden, true);
              assert.strictEqual(
                elements.get("statusText").textContent,
                "원래 review pack 목록으로 돌아왔습니다.",
              );
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
        """,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_channel_audit_review_can_accept_existing_reference_channel(self):
        result = self.run_app_assertions(
            r"""
            (async () => {
              const pathInput = elements.get("reviewPackPathInput");
              pathInput.value = "/packs/channel-audit-pack";
              const reviewPackItem = {
                priority_rank: 1,
                case_id: "front-a",
                reference_id: "seg_000001",
                start_ms: 0,
                end_ms: 1000,
                reasons: ["reference-channel-energy-mismatch"],
                reference_channel: "L",
                candidate_channel: "R",
                left_dbfs: -32,
                right_dbfs: -28,
                delta_db: -4,
                source_review_requirements: ["channel"],
                source_review_resolved: false,
                clip_url: "/api/review-pack/clip?x=1",
              };
              context.fetch = async () => ({
                ok: true,
                async json() {
                  return {
                    kind: "review-pack",
                    source_case_index: "/cases/case-index.json",
                    items: [reviewPackItem],
                  };
                },
              });
              await context.loadReviewPath();
              context.selectReviewPackItem(0, false);

              context.fetch = async () => ({
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
                          audio: { source_file: "front-a.wav", duration_ms: 1000 },
                          segments: [
                            {
                              id: "seg_000001",
                              start_ms: 0,
                              end_ms: 1000,
                              channel: "L",
                              kind: "speech",
                              text: "確認",
                              needs_review: false,
                              content_reviewed: false,
                              channel_reviewed: false,
                            },
                          ],
                        },
                      },
                    ],
                  };
                },
              });
              await context.openSelectedReviewPackSourceCase();

              const doneButton = elements.get("reviewDoneButton");
              assert.strictEqual(doneButton.hidden, false);
              assert.strictEqual(doneButton.disabled, false);
              assert.strictEqual(doneButton.textContent, "Channel 검수 완료");

              let saveCount = 0;
              context.fetch = async (path, options) => {
                assert.strictEqual(path, "/api/review-case/save-reference");
                const payload = JSON.parse(options.body);
                const segment = payload.master.segments[0];
                saveCount += 1;
                assert.strictEqual(segment.channel, "L");
                assert.strictEqual(segment.needs_review, false);
                assert.strictEqual(segment.content_reviewed, false);
                assert.strictEqual(segment.channel_reviewed, saveCount === 1);
                return {
                  ok: true,
                  async json() {
                    return { segments: 1, review_count: 0, review_duration_ms: 0 };
                  },
                };
              };

              await context.markSelectedReviewDone();
              assert.strictEqual(doneButton.disabled, true);
              assert.strictEqual(reviewPackItem.source_review_resolved, true);
              assert.strictEqual(context.nextReviewPackSourceIndex(), null);
              assert.strictEqual(elements.get("statusLabel").textContent, "Channel 검수 저장됨");
              assert.strictEqual(
                elements.get("statusText").textContent,
                "seg_000001 channel 판정을 저장했습니다.",
              );

              const segment = context.selectedSegment();
              context.commitSegmentTime(segment, "start_ms", { value: "100" });
              assert.strictEqual(segment.start_ms, 100);
              assert.strictEqual(segment.channel_reviewed, false);
              await context.saveCurrentMasterNow();
              assert.strictEqual(reviewPackItem.source_review_resolved, false);
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
