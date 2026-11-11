/**
 * "Suggest" button + dropdown for the Snart Staged Prompt Builder.
 *
 * Clicking Suggest asks the backend (/snart/staged/suggest) for the next step's
 * options and populates the "choice" dropdown. Picking from the dropdown appends
 * the choice to the node's "choices" widget. Repeat per step, then run the node
 * to write the final prompt.
 *
 * All UI additions are wrapped in try/catch so a frontend hiccup can never stop
 * the node from being placed in a workflow.
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
      try {
        addSuggestUI(this);
      } catch (error) {
        console.error(`[${EXT}] Could not add Suggest UI:`, error);
      }
      return result;
    };
  },
});

function addSuggestUI(node) {
  // dropdown the Suggest button populates
  const choice = node.addWidget(
    "combo",
    "choice",
    "",
    (value) => {
      if (!value) return;
      const choices = getWidget(node, "choices");
      if (choices) {
        const lines = String(choices.value ?? "")
          .split("\n")
          .map((l) => l.trim())
          .filter(Boolean);
        lines.push(String(value));
        choices.value = lines.join("\n");
        node.graph?.setDirtyCanvas(true, true);
      }
    },
    { serialize: false, values: [""] }
  );
  choice.options = choice.options || {};
  choice.options.values = choice.options.values || [""];

  // Suggest button
  node.addWidget(
    "button",
    "🎲 Suggest",
    null,
    async () => {
      const prompt = widgetValue(node, "prompt");
      if (!prompt) {
        notify("Enter a prompt first.", "warn");
        return;
      }
      const numSteps = Number(widgetValue(node, "num_steps") || 0);
      const chosen = String(widgetValue(node, "choices"))
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
        num_suggestions: Number(widgetValue(node, "num_suggestions") || 2),
        broadness: widgetValue(node, "broadness") || "simple",
        llm_backend: widgetValue(node, "llm_backend"),
        api_endpoint: widgetValue(node, "api_endpoint"),
        model_name: widgetValue(node, "model_name"),
        temperature: Number(widgetValue(node, "temperature") || 0.7),
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

      choice.options = choice.options || {};
      choice.options.values = ["", ...data.options];
      choice.value = "";
      node.graph?.setDirtyCanvas(true, true);
    },
    { serialize: false }
  );
}
