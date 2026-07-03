const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function formData(form) {
  const data = new FormData(form);
  const out = {};
  for (const [key, value] of data.entries()) {
    out[key] = value;
  }
  for (const input of form.querySelectorAll('input[type="checkbox"]')) {
    out[input.name] = input.checked;
  }
  return out;
}

function optionalPath(value) {
  return value && value.trim() ? value.trim() : null;
}

function fileUrl(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function activate(view) {
  $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $$(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
}

async function refreshHealth() {
  try {
    const health = await api("/health");
    $("#health").textContent = `${health.status} | ${health.role}`;
  } catch (error) {
    $("#health").textContent = error.message;
  }
}

async function loadStyles() {
  const styles = await api("/api/styles");
  $("#styles-list").innerHTML = styles
    .map((style) => `<div class="style-pill"><strong>${style.id}</strong><br>${style.name}</div>`)
    .join("");
}

async function runInspect() {
  const data = formData($("#inspect-form"));
  const result = await api("/api/inspect", {
    method: "POST",
    body: JSON.stringify({
      request: {
        input_path: data.input_path,
        debug_dir: optionalPath(data.debug_dir),
      },
    }),
  });
  $("#inspect-output").textContent = pretty({
    classification: result.page_classification,
    size: result.size,
    preprocess: result.preprocess,
    panels: result.panels.length,
    tiles: result.tiles.length,
    debug_files: result.debug_files,
  });
}

async function queueColorize() {
  const data = formData($("#colorize-form"));
  const request = {
    input_path: data.input_path,
    output_path: data.output_path,
    debug_dir: optionalPath(data.debug_dir),
    style_path: optionalPath(data.style_path),
    user_hints_path: optionalPath(data.user_hints_path),
    use_mock_backend: data.use_mock_backend,
  };
  const job = await api("/api/colorize", {
    method: "POST",
    body: JSON.stringify({ request, ads_token: null, premium_account_id: null }),
  });
  $("#colorize-output").textContent = pretty(job);
  await pollJob(job.id, data.output_path);
}

async function pollJob(id, outputPath) {
  for (let i = 0; i < 80; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const job = await api(`/api/jobs/${id}`);
    $("#colorize-output").textContent = pretty(job);
    if (job.status === "complete") {
      const image = $("#preview-image");
      image.src = `${fileUrl(outputPath)}&t=${Date.now()}`;
      image.style.display = "block";
      await refreshJobs();
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "job failed");
    }
  }
  throw new Error("job timed out");
}

async function queueLinkColorize() {
  const data = formData($("#link-form"));
  const request = {
    input_path: "",
    output_path: "",
    style_path: optionalPath(data.style_path),
    use_mock_backend: data.use_mock_backend,
  };
  const job = await api("/api/link/colorize", {
    method: "POST",
    body: JSON.stringify({
      source_url: data.source_url,
      output_dir: data.output_dir,
      downloads_dir: optionalPath(data.downloads_dir),
      max_pages: Number(data.max_pages || 8),
      continue_on_error: true,
      template: request,
      user_agent: null,
    }),
  });
  $("#link-output").textContent = pretty(job);
  await pollLinkJob(job.id);
}

async function pollLinkJob(id) {
  for (let i = 0; i < 120; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const job = await api(`/api/jobs/${id}`);
    $("#link-output").textContent = pretty(job);
    if (job.status === "complete") {
      const result = await api(`/api/jobs/${id}/result`);
      $("#link-output").textContent = pretty(result);
      const firstPage = result.result?.pages?.find((page) => page.status === "complete");
      if (firstPage?.output_path) {
        const image = $("#link-preview-image");
        image.src = `${fileUrl(firstPage.output_path)}&t=${Date.now()}`;
        image.style.display = "block";
      }
      await refreshJobs();
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "link job failed");
    }
  }
  throw new Error("link job timed out");
}

async function importChapter() {
  const data = formData($("#chapter-form"));
  const result = await api("/api/chapter/import", {
    method: "POST",
    body: JSON.stringify({
      input_path: data.input_path,
      output_dir: data.output_dir,
      manifest_path: optionalPath(data.manifest_path),
      max_pages: Number(data.max_pages || 500),
      overwrite: true,
    }),
  });
  $("#chapter-output").textContent = pretty(result);
}

async function exportChapter() {
  const data = formData($("#chapter-form"));
  const result = await api("/api/chapter/export", {
    method: "POST",
    body: JSON.stringify({
      input_dir: data.export_input_dir,
      output_file: data.export_file,
      manifest_path: optionalPath(data.manifest_path),
    }),
  });
  $("#chapter-output").textContent = pretty(result);
}

async function refreshJobs() {
  const jobs = await api("/api/jobs");
  $("#jobs-table").innerHTML = jobs
    .map(
      (job) => `<tr>
        <td>${job.id}</td>
        <td>${job.status}</td>
        <td>${job.output_path || ""}</td>
        <td><button data-job="${job.id}" class="result-btn">Open</button></td>
      </tr>`,
    )
    .join("");
  $$(".result-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const result = await api(`/api/jobs/${button.dataset.job}/result`);
      $("#inspect-output").textContent = pretty(result);
      activate("inspect");
    });
  });
}

async function saveCorrection() {
  const data = formData($("#correction-form"));
  const payload = {
    project: data.project,
    page: data.page,
    x: Number(data.x),
    y: Number(data.y),
    width: Number(data.width),
    height: Number(data.height),
    label: optionalPath(data.label),
    old: data.old,
    new: data.new,
    character: optionalPath(data.character),
  };
  const result = await api("/api/corrections", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  $("#memory-output").textContent = pretty(result);
}

async function exportCorrectionHints() {
  const data = formData($("#correction-form"));
  const result = await api("/api/projects/export-hints", {
    method: "POST",
    body: JSON.stringify({
      project: data.project,
      page: data.page,
      output: data.hints_output,
    }),
  });
  $("#memory-output").textContent = pretty(result);
}

async function runDoctor() {
  const data = formData($("#doctor-form"));
  const result = await api("/api/doctor", {
    method: "POST",
    body: JSON.stringify({
      config_path: data.config_path,
      styles_dir: data.styles_dir,
      metadata_path: data.metadata_path,
      jobs_dir: data.jobs_dir,
      studio_dir: data.studio_dir,
      sample_input: data.sample_input,
      python: data.python,
    }),
  });
  $("#doctor-output").textContent = pretty(result);
}

function wire() {
  $$(".tab").forEach((button) => button.addEventListener("click", () => activate(button.dataset.view)));
  $("#inspect-run").addEventListener("click", () => runInspect().catch((error) => $("#inspect-output").textContent = error.message));
  $("#colorize-run").addEventListener("click", () => queueColorize().catch((error) => $("#colorize-output").textContent = error.message));
  $("#link-run").addEventListener("click", () => queueLinkColorize().catch((error) => $("#link-output").textContent = error.message));
  $("#chapter-import").addEventListener("click", () => importChapter().catch((error) => $("#chapter-output").textContent = error.message));
  $("#chapter-export").addEventListener("click", () => exportChapter().catch((error) => $("#chapter-output").textContent = error.message));
  $("#jobs-refresh").addEventListener("click", () => refreshJobs());
  $("#correction-save").addEventListener("click", () => saveCorrection().catch((error) => $("#memory-output").textContent = error.message));
  $("#correction-export-hints").addEventListener("click", () => exportCorrectionHints().catch((error) => $("#memory-output").textContent = error.message));
  $("#doctor-run").addEventListener("click", () => runDoctor().catch((error) => $("#doctor-output").textContent = error.message));
}

wire();
refreshHealth();
loadStyles().catch((error) => $("#styles-list").textContent = error.message);
refreshJobs().catch(() => {});
