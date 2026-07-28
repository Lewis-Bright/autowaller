const MODULE_ID = "foundry-autowaller";
const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 15 * 60 * 1000;
const MINIMUM_CONFIDENCE = 0.65;

Hooks.once("init", () => {
  game.settings.register(MODULE_ID, "apiUrl", {
    name: "Service URL",
    hint: "The URL of your deployed Autowaller API.",
    scope: "world",
    config: true,
    type: String,
    default: "",
    restricted: true
  });

  game.settings.register(MODULE_ID, "apiKey", {
    name: "API key",
    hint: "Private key used to access your Autowaller service.",
    scope: "world",
    config: true,
    type: String,
    default: "",
    restricted: true
  });
});

Hooks.on("getSceneControlButtons", (controls) => {
  const wallsControl = Array.isArray(controls)
    ? controls.find((control) => control.name === "walls")
    : controls?.walls;
  if (!wallsControl) return;

  const tool = {
    name: "autowaller",
    title: "Auto Wall",
    icon: "fas fa-wand-magic-sparkles",
    button: true,
    visible: game.user.isGM,
    onClick: confirmAndRun
  };

  if (Array.isArray(wallsControl.tools)) wallsControl.tools.push(tool);
  else wallsControl.tools.autowaller = tool;
});

async function confirmAndRun() {
  if (!game.user.isGM) {
    ui.notifications.warn("Only a GM can use Auto Wall.");
    return;
  }
  if (!canvas?.scene) {
    ui.notifications.warn("Open a scene first.");
    return;
  }
  if (!apiKey()) {
    ui.notifications.error(
      "Enter the Autowaller API key in Configure Settings → Module Settings first."
    );
    return;
  }
  if (!apiUrl()) {
    ui.notifications.error(
      "Enter the Autowaller service URL in Configure Settings → Module Settings first."
    );
    return;
  }

  const confirmed = window.confirm(
    `Automatically detect and apply walls to “${canvas.scene.name}”?`
  );
  if (!confirmed) return;

  await runAutowaller();
}

async function runAutowaller() {
  const scene = canvas.scene;
  const sceneId = scene.id;
  let jobId = null;
  ui.notifications.info("Auto Wall is analysing this scene…");

  try {
    const blob = await getBackgroundBlob(scene);
    const dimensions = sceneAnalysisDimensions(scene);
    const contentType = supportedContentType(blob.type);
    const job = await api("/jobs", {
      method: "POST",
      body: JSON.stringify({
        width: dimensions.width,
        height: dimensions.height,
        contentType
      })
    });
    jobId = job.jobId;
    console.info(`${MODULE_ID} | created job ${jobId}`, {
      sceneId,
      sceneName: scene.name,
      dimensions
    });

    const upload = await fetch(job.uploadUrl, {
      method: "PUT",
      headers: { "content-type": contentType },
      body: blob
    });
    if (!upload.ok) throw new Error(`Map upload failed (${upload.status}).`);

    await api(`/jobs/${job.jobId}/start`, {
      method: "POST",
      body: "{}"
    });
    const resultUrl = await waitForResult(job.jobId);
    const resultResponse = await fetch(resultUrl);
    if (!resultResponse.ok) {
      throw new Error("Could not download the completed wall plan.");
    }

    const plan = await resultResponse.json();
    validatePlan(plan, dimensions);
    if (canvas.scene?.id !== sceneId) {
      throw new Error("The active scene changed while Auto Wall was running.");
    }

    const walls = plan.walls
      .filter((wall) => wall.confidence >= MINIMUM_CONFIDENCE)
      .map((wall) => wallData(wall, job.jobId, dimensions));
    if (!walls.length) throw new Error("No confident wall segments were detected.");

    console.info(`${MODULE_ID} | job ${jobId} applying ${walls.length} walls`, {
      diagnostics: plan.diagnostics
    });
    await scene.createEmbeddedDocuments("Wall", walls);
    console.info(`${MODULE_ID} | job ${jobId} complete`);
    ui.notifications.info(`Auto Wall applied ${walls.length} wall segments.`);
  } catch (error) {
    console.error(`${MODULE_ID} | job ${jobId || "not-created"} failed`, error);
    const trace = jobId ? ` Job: ${jobId}` : "";
    ui.notifications.error(`${error.message || "Auto Wall failed."}${trace}`);
  }
}

async function api(path, options = {}) {
  const response = await fetch(`${apiUrl()}${path}`, {
    ...options,
    headers: {
      authorization: `Bearer ${apiKey()}`,
      "content-type": "application/json",
      ...(options.headers || {})
    }
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Autowaller request failed (${response.status}).`);
  }
  return body;
}

function apiKey() {
  return String(game.settings.get(MODULE_ID, "apiKey") || "").trim();
}

function apiUrl() {
  return String(game.settings.get(MODULE_ID, "apiUrl") || "")
    .trim()
    .replace(/\/+$/, "");
}

async function getBackgroundBlob(scene) {
  const source = scene.background?.src || scene.img;
  if (!source) throw new Error("The current scene has no background image.");

  const response = await fetch(new URL(source, window.location.href));
  if (!response.ok) {
    throw new Error(`Could not load the scene image (${response.status}).`);
  }
  return response.blob();
}

function supportedContentType(contentType) {
  if (["image/png", "image/jpeg", "image/webp"].includes(contentType)) {
    return contentType;
  }
  throw new Error("The scene background must be a PNG, JPEG, or WebP image.");
}

function sceneAnalysisDimensions(scene) {
  const dimensions = canvas.dimensions;
  return {
    width: Math.round(dimensions.sceneWidth || scene.width),
    height: Math.round(dimensions.sceneHeight || scene.height),
    x: Number(dimensions.sceneX || 0),
    y: Number(dimensions.sceneY || 0)
  };
}

async function waitForResult(jobId) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const job = await api(`/jobs/${jobId}`);
    if (job.status === "complete") return job.resultUrl;
    if (job.status === "failed") {
      throw new Error(job.error || "Wall detection failed.");
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  throw new Error("Wall detection did not finish within 15 minutes.");
}

function validatePlan(plan, expected) {
  if (plan?.schemaVersion !== 1 || !Array.isArray(plan.walls)) {
    throw new Error("The service returned an unsupported wall-plan format.");
  }
  if (
    plan.scene?.width !== expected.width ||
    plan.scene?.height !== expected.height
  ) {
    throw new Error("The returned wall plan does not match this scene.");
  }
  if (plan.walls.length > 2000) {
    throw new Error("The returned wall plan contains too many segments.");
  }
  for (const wall of plan.walls) {
    if (
      !Array.isArray(wall.c) ||
      wall.c.length !== 4 ||
      !wall.c.every(Number.isFinite)
    ) {
      throw new Error("The returned wall plan contains invalid coordinates.");
    }
  }
}

function wallData(wall, runId, offset) {
  const movement = CONST.WALL_MOVEMENT_TYPES?.NORMAL ?? 20;
  const sense = CONST.WALL_SENSE_TYPES?.NORMAL ?? 1;
  const doorNone = CONST.WALL_DOOR_TYPES?.NONE ?? 0;
  const [x1, y1, x2, y2] = wall.c;
  return {
    c: [x1 + offset.x, y1 + offset.y, x2 + offset.x, y2 + offset.y],
    move: movement,
    sight: sense,
    light: sense,
    sound: sense,
    door: doorNone,
    flags: {
      [MODULE_ID]: {
        runId,
        confidence: wall.confidence
      }
    }
  };
}
