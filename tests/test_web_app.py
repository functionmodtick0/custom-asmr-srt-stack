import subprocess
import textwrap
import unittest


class WebAppBehaviorTests(unittest.TestCase):
    def test_local_granite_adapter_does_not_require_endpoint_settings(self):
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
                  classList: { add() {}, remove() {}, toggle() {} },
                  addEventListener() {},
                  append() {},
                  replaceChildren() {},
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
        """

        result = subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
