/**
 * "Suggest" button + dropdown for the Snart Staged Prompt Builder.
 *
 * Clicking Suggest asks the backend (/snart/staged/suggest) for the next step's
 * options and populates the "choice" dropdown. Picking from the dropdown appends
 * the choice to the node's "choices" widget. Repeat per step, then run the node
 * to write the final prompt.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXT = "Snart";
const ROUTE = "/snart/staged/suggest";
const NODE_TYPE = "SnartStagedPrompter";

function getWidget(node, name) {
  return node?.widgets?.find((w) => w.name === name);
}

function widgetValue(node, name) {
  const w = getWidget(node, name);
  return w ? String(w.value ?? "").trim() : "";
}

function notify(message, severity = "info") {
  console.log(`[${EXT}] ${severity}: ${message}`);
  try {
    app.extensionManager?.toast?.add?.({ severity, summary: EXT, detail: message, life: 6000 });
  } catch (error) {
    /* toast unavailable - console log above is enough */
  }
}

app.registerExtension({
  name: "snart.staged-suggest",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated?.apply(this, args);

      // dropdown the Suggest button populates
      const choice = this.addWidget(
        "combo",
        "choice",
        "",
        (value) => {
          if (!value) return;
          const choices = getWidget(this, "choices");
          if (choices) {
            const lines = String(choices.value ?? "")
              .split("\n")
              .map((l) => l.trim())
              .filter(Boolean);
            lines.push(String(value));
            choices.value = lines.join("\n");
            this.graph?.setDirtyCanvas(true, true);
          }
        },
        { serialize: false }
      );
      choice.options = choice.options || {};
      choice.options.values = [""];

      // Suggest button
      this.addWidget(
        "button",
        "🎲 Suggest",
        null,
        async () => {
          const prompt = widgetValue(this, "prompt");
          if (!prompt) {
            notify("Enter a prompt first.", "warn");
            return;
          }
          const numSteps = Number(widgetValue(this, "num_steps") || 0);
          const chosen = String(widgetValue(this, "choices"))
            .split("\n")
            .map((l) => l.trim())
            .filter(Boolean);
          const step = chosen.length + 1;
          if (numSteps > 0 && step > numSteps) {
            notify(`All ${numSteps} steps chosen - run the node for the final prompt.`, "warn");
            return;
          }

          const payload = {
            prompt,
            chosen,
            step,
            num_suggestions: Number(widgetValue(this, "num_suggestions") || 2),
            broadness: widgetValue(this, "broadness") || "simple",
            llm_backend: widgetValue(this, "llm_backend"),
            api_endpoint: widgetValue(this, "api_endpoint"),
            model_name: widgetValue(this, "model_name"),
            temperature: Number(widgetValue(this, "temperature") || 0.7),
          };

          let data;
          try {
            const resp = await api.fetchApi(ROUTE, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = await resp.json();
          } catch (error) {
            notify(`Suggest failed: ${error.message}`, "error");
            return;
          }

          if (!data?.success || !data.options?.length) {
            notify(data?.error || "No options returned.", "warn");
            return;
          }

          choice.options.values = ["", ...data.options];
          choice.value = "";
          this.graph?.setDirtyCanvas(true, true);
        },
        { serialize: false }
      );

      return result;
    };
  },
});
