/**
 * Expanding dropdowns for the Snart Prompt Builder.
 *
 * Clicking 🎲 Start fetches step 1's options and adds a "Step 1" dropdown.
 * Picking from it appends the choice to "choices" and adds a "Step 2" dropdown,
 * and so on until num_steps is reached. The node grows one dropdown per step.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXT = "Snart";
const ROUTE = "/snart/staged/suggest";
const NODE_TYPE = "SnartPromptBuilder";

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
    /* toast unavailable */
  }
}

async function fetchOptions(node, step, chosen) {
  const payload = {
    prompt: widgetValue(node, "prompt"),
    chosen,
    step,
    num_suggestions: Number(widgetValue(node, "num_suggestions") || 2),
    broadness: widgetValue(node, "broadness") || "simple",
    llm_backend: widgetValue(node, "llm_backend"),
    api_endpoint: widgetValue(node, "api_endpoint"),
    model_name: widgetValue(node, "model_name"),
    temperature: Number(widgetValue(node, "temperature") || 0.7),
  };
  const resp = await api.fetchApi(ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  if (!data?.success) throw new Error(data?.error || "no options returned");
  return data.options || [];
}

function appendChoice(node, value) {
  const choices = getWidget(node, "choices");
  if (!choices) return;
  const lines = String(choices.value ?? "")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (!lines.includes(value)) lines.push(value);
  choices.value = lines.join("\n");
}

function currentChoices(node) {
  return String(getWidget(node, "choices")?.value ?? "")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

async function addStep(node, step) {
  const chosen = currentChoices(node);
  let options;
  try {
    options = await fetchOptions(node, step, chosen);
  } catch (error) {
    notify(`Step ${step} failed: ${error.message}`, "error");
    return;
  }
  if (!options.length) {
    notify("No options returned.", "warn");
    return;
  }

  node.addWidget(
    "combo",
    `Step ${step}`,
    "",
    async (value) => {
      if (!value) return;
      appendChoice(node, value);
      node.graph?.setDirtyCanvas(true, true);
      const numSteps = Number(widgetValue(node, "num_steps") || 0);
      if (step < numSteps) {
        await addStep(node, step + 1);
      } else {
        notify("All steps done - connect the text output to an enhancer.", "success");
      }
    },
    { serialize: false, values: ["", ...options] }
  );
  node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "snart.prompt-builder",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated?.apply(this, args);

      try {
        this.addWidget(
          "button",
          "🎲 Start",
          null,
          async () => {
            const prompt = widgetValue(this, "prompt");
            if (!prompt) {
              notify("Enter a prompt first.", "warn");
              return;
            }
            // clear previous run
            const choices = getWidget(this, "choices");
            if (choices) choices.value = "";
            // drop any old step dropdowns
            const stepNames = this.widgets
              .filter((w) => /^Step \d+$/.test(w.name))
              .map((w) => w.name);
            for (const name of stepNames) {
              const idx = this.widgets.findIndex((w) => w.name === name);
              if (idx >= 0) this.widgets.splice(idx, 1);
            }
            this.setSize?.(this.computeSize?.());
            await addStep(this, 1);
          },
          { serialize: false }
        );
      } catch (error) {
        console.error(`[${EXT}] Could not add Start button:`, error);
      }

      return result;
    };
  },
});
