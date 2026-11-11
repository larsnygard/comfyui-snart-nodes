/**
 * "Suggest" button for the Snart Staged Prompt Builder.
 *
 * The "choice" dropdown is a normal node input (declared in INPUT_TYPES), so the
 * frontend's standard combo mechanism drives it. Clicking Suggest:
 *   1. appends the current "choice" pick to "choices",
 *   2. asks the backend (/snart/staged/suggest) for the next step's options,
 *   3. repopulates the "choice" dropdown with them.
 * Repeat per step, then run the node to write the final prompt.
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
        addSuggestButton(this);
      } catch (error) {
        console.error(`[${EXT}] Could not add Suggest button:`, error);
      }
      return result;
    };
  },
});

function addSuggestButton(node) {
  node.addWidget(
    "button",
    "🎲 Suggest",
    null,
    async () => {
      const choice = getWidget(node, "choice");
      const choices = getWidget(node, "choices");

      // 1. append the current pick (if any) to choices
      const curChoice = choice ? String(choice.value ?? "").trim() : "";
      if (curChoice) {
        const lines = String(choices?.value ?? "")
          .split("\n")
          .map((l) => l.trim())
          .filter(Boolean);
        if (!lines.includes(curChoice)) lines.push(curChoice);
        if (choices) choices.value = lines.join("\n");
      }

      const chosen = String(choices?.value ?? "")
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean);
      const numSteps = Number(widgetValue(node, "num_steps") || 0);
      const step = chosen.length + 1;

      if (numSteps > 0 && step > numSteps) {
        notify(`All ${numSteps} steps chosen - run the node for the final prompt.`, "warn");
        return;
      }

      const prompt = widgetValue(node, "prompt");
      if (!prompt) {
        notify("Enter a prompt first.", "warn");
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

      // 2. repopulate the choice dropdown
      if (choice) {
        choice.options = choice.options || {};
        choice.options.values = ["", ...data.options];
        choice.value = "";
      }
      node.graph?.setDirtyCanvas(true, true);
    },
    { serialize: false }
  );
}
