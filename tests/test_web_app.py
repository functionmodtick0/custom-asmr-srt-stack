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
