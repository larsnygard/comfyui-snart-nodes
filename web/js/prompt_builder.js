/**
 * Prompt Builder: all step dropdowns + random checkboxes are shown up front
 * (one per num_steps), populated in sequence when you click 🎲 Start. The
 * choices box sits below the step dropdowns. Changing num_steps rebuilds them.
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

function updateProgress(node) {
  const prog = getWidget(node, "📊 Progress");
  if (!prog) return;
  const numSteps = Number(widgetValue(node, "num_steps") || 0);
  const chosen = currentChoices(node).length;
  prog.value = `${chosen} / ${numSteps} steps chosen`;
  node.graph?.setDirtyCanvas(true, true);
}

function removeStepWidgets(node) {
  const names = node.widgets
    .map((w) => w.name)
    .filter((n) => /^Step \d+$/.test(n) || /^🎲 Step \d+$/.test(n));
  for (const name of names) {
    const idx = node.widgets.findIndex((w) => w.name === name);
    if (idx >= 0) node.widgets.splice(idx, 1);
  }
}

function moveChoicesToBottom(node) {
  const idx = node.widgets.findIndex((w) => w.name === "choices");
  if (idx >= 0) {
    const [choices] = node.widgets.splice(idx, 1);
    node.widgets.push(choices);
  }
}

async function onStepPicked(node, step, value) {
  if (!value) return;
  // guard against double-trigger (programmatic value set + manual callback)
  if (currentChoices(node).length >= step) return;
  appendChoice(node, value);
  updateProgress(node);
  const numSteps = Number(widgetValue(node, "num_steps") || 0);
  if (step < numSteps) {
    await populateStep(node, step + 1);
  } else {
    notify("All steps done - connect the text output to an enhancer.", "success");
  }
}

async function populateStep(node, step) {
  const combo = getWidget(node, `Step ${step}`);
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
  if (combo) {
    combo.options = combo.options || {};
    combo.options.values = ["", ...options];
    combo.value = "";
  }
  node.graph?.setDirtyCanvas(true, true);

  const rand = getWidget(node, `🎲 Step ${step}`);
  if (rand && rand.value) {
    const pick = options[Math.floor(Math.random() * options.length)];
    if (combo) combo.value = pick;
    await onStepPicked(node, step, pick);
  }
}

function rebuildSteps(node) {
  removeStepWidgets(node);
  const numSteps = Number(widgetValue(node, "num_steps") || 0);
  for (let i = 1; i <= numSteps; i++) {
    node.addWidget("toggle", `🎲 Step ${i}`, false, null, { serialize: false });
    node.addWidget(
      "combo",
      `Step ${i}`,
      "",
      async (value) => {
        if (value) await onStepPicked(node, i, value);
      },
      { serialize: false, values: [""] }
    );
  }
  moveChoicesToBottom(node);
  node.graph?.setDirtyCanvas(true, true);
}

function wireNumSteps(node) {
  const w = getWidget(node, "num_steps");
  if (!w || w._snartWired) return;
  w._snartWired = true;
  const orig = w.callback;
  w.callback = function (value) {
    if (orig) orig.apply(this, arguments);
    setTimeout(() => {
      try {
        rebuildSteps(node);
      } catch (error) {
        console.error(`[${EXT}] rebuild failed:`, error);
      }
    }, 0);
  };
}

app.registerExtension({
  name: "snart.prompt-builder",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated?.apply(this, args);

      try {
        this.addWidget("string", "📊 Progress", "", null, { serialize: false });
        updateProgress(this);

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
            const choices = getWidget(this, "choices");
            if (choices) choices.value = "";
            updateProgress(this);
            await populateStep(this, 1);
          },
          { serialize: false }
        );

        rebuildSteps(this);
        wireNumSteps(this);
      } catch (error) {
        console.error(`[${EXT}] Could not init builder UI:`, error);
      }

      return result;
    };
  },
});
