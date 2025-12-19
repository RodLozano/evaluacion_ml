const API_BASE = "/api";

/**
 * Helpers
 */
function $(id) {
  return document.getElementById(id);
}

function formatDate(s) {
  if (!s) return "";
  // Si viene ISO, lo dejamos tal cual. Si quieres, lo formateamos.
  return s.replace("T", " ").replace("Z", "");
}

async function safeJson(res) {
  const txt = await res.text();
  try { return JSON.parse(txt); } catch { return txt; }
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

const PipelineDashboard = {
  async init() {
    await this.loadDatasets();
    await this.loadRuns();

    $("create-run-form").addEventListener("submit", this.createRun.bind(this));
  },

  async loadDatasets() {
    const sel = $("run-dataset");
    sel.innerHTML = `<option value="">Cargando datasets...</option>`;

    const res = await fetch(`${API_BASE}/datasets/`);
    if (!res.ok) {
      sel.innerHTML = `<option value="">(Error cargando datasets)</option>`;
      return;
    }

    const datasets = await res.json();
    sel.innerHTML = "";

    if (!datasets.length) {
      sel.innerHTML = `<option value="">(No hay CSVs en data/raw)</option>`;
      return;
    }

    sel.innerHTML = `<option value="">Selecciona dataset (obligatorio)</option>`;
    datasets.forEach((ds) => {
      const opt = document.createElement("option");
      opt.value = ds;
      opt.textContent = ds;
      sel.appendChild(opt);
    });
  },

  async loadRuns() {
    const res = await fetch(`${API_BASE}/runs/`);
    const runs = await res.json();

    const tbody = document.querySelector("#runs-table tbody");
    tbody.innerHTML = "";

    runs.forEach((run) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${run.id}</td>
        <td>${escapeHtml(run.name || "")}</td>
        <td>${escapeHtml(run.dataset || "")}</td>
        <td>${escapeHtml(run.status || "")}</td>
        <td>${formatDate(run.created_at)}</td>
        <td><a href="/pipeline/${run.id}/">Ver</a></td>
      `;
      tbody.appendChild(tr);
    });
  },

  async createRun(e) {
    e.preventDefault();

    const msg = $("create-run-msg");
    msg.textContent = "";

    const name = $("run-name").value?.trim() || "";
    const dataset = $("run-dataset").value?.trim() || "";
    const configText = $("run-config").value?.trim() || "";

    if (!dataset) {
      msg.textContent = "Selecciona un dataset antes de crear la ejecución.";
      return;
    }

    let config = {};
    if (configText) {
      try {
        config = JSON.parse(configText);
      } catch {
        msg.textContent = "Config JSON inválido.";
        return;
      }
    }

    const payload = { name, dataset, config };

    const res = await fetch(`${API_BASE}/runs/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await safeJson(res);
      msg.textContent = `Error creando run: ${typeof err === "string" ? err : JSON.stringify(err)}`;
      return;
    }

    $("run-name").value = "";
    // No borramos dataset para facilitar crear varias runs
    // $("run-dataset").value = dataset;
    // config opcional: lo dejamos
    msg.textContent = "Run creada correctamente.";

    await this.loadRuns();
  },
};


const PipelineRunDetail = {
  runId: null,
  lastLogTime: null, // ISO datetime; backend filtra created_at__gt
  polling: null,
  running: false,

  async init(runId) {
    this.runId = runId;

    $("start-btn").addEventListener("click", () => this.start());
    $("refresh-btn").addEventListener("click", () => this.refreshAll());

    await this.refreshAll();
    this.startPolling();
  },

  async refreshAll() {
    await this.refreshStatus();
    await this.loadArtifacts(); // aunque no haya terminado, lista lo que exista
  },

  async start() {
    const btn = $("start-btn");
    btn.disabled = true;

    const res = await fetch(`${API_BASE}/runs/${this.runId}/start/`, { method: "POST" });
    if (!res.ok) {
      const err = await safeJson(res);
      alert(`No se pudo iniciar: ${typeof err === "string" ? err : JSON.stringify(err)}`);
    }

    // Reiniciamos consola/offset para esta ejecución
    // $("log-console").textContent = "";
    // this.lastLogTime = null;

    await this.refreshStatus();
  },

  async refreshStatus() {
    const res = await fetch(`${API_BASE}/runs/${this.runId}/`);
    const run = await res.json();

    $("status").textContent = run.status || "";
    $("dataset").textContent = run.dataset ? `Dataset: ${run.dataset}` : "";

    // UX básica: botón start se deshabilita si está running o ya terminó
    const finished = run.status === "SELECTED" || run.status === "FAILED";
    const running = run.status === "RUNNING" || run.status === "DATA" || run.status === "PREPROCESS" ||
                    run.status === "TRAINING" || run.status === "EVALUATED" || run.status === "SELECTED";

    this.running = running && !finished;

    const btn = $("start-btn");
    if (finished) {
      btn.disabled = true;
      btn.textContent = run.status === "FAILED" ? "✖ Pipeline fallida" : "✔ Pipeline finalizada";
    } else {
      btn.disabled = this.running; // si ya está en marcha, no reiniciar
      btn.textContent = this.running ? "⏳ Ejecutando..." : "▶ Ejecutar pipeline";
    }

    // Si falla, mostramos error en consola (si existe)
    if (run.status === "FAILED" && run.error_message) {
      const c = $("log-console");
      if (!c.textContent.includes("FAILED:")) {
        c.textContent += `\n[system] FAILED: ${run.error_message}\n`;
      }
    }
  },

  async pollLogs() {
    let url = `${API_BASE}/runs/${this.runId}/logs/`;
    if (this.lastLogTime) {
      // backend espera ISO datetime
      url += `?since=${encodeURIComponent(this.lastLogTime)}`;
    }

    const res = await fetch(url);
    if (!res.ok) return;

    const logs = await res.json();
    const consoleEl = $("log-console");

    // Si llegan logs, los pintamos
    logs.forEach((log) => {
      const step = log.step || "console";
      const msg = log.message || "";
      consoleEl.textContent += `[${step}] ${msg}\n`;
      this.lastLogTime = log.created_at; // ISO
    });

    if (logs.length) {
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }

    await this.refreshStatus();

    const statusText = $("status").textContent;
    if (statusText === "SELECTED" || statusText === "FAILED") {
      await this.loadArtifacts();
      clearInterval(this.polling);
    }
  },

  startPolling() {
    if (this.polling) clearInterval(this.polling);
    this.polling = setInterval(() => this.pollLogs(), 1500);
  },

  async loadArtifacts() {
    const res = await fetch(`${API_BASE}/runs/${this.runId}/artifacts/`);
    if (!res.ok) return;

    const artifacts = await res.json();
    const ul = $("artifacts-list");
    ul.innerHTML = "";

    if (!artifacts.length) {
      ul.innerHTML = `<li style="opacity:0.7;">(Sin artifacts todavía)</li>`;
      return;
    }

    artifacts.forEach((a) => {
      // Con los cambios de backend: a.url y a.download_url existen
      const viewUrl = a.url || "";
      const dlUrl = a.download_url || "";

      const li = document.createElement("li");

      const name = escapeHtml(a.name || a.path || "artifact");
      const kind = escapeHtml(a.kind || "");

      // "Ver" abre el recurso (imagen/csv), "Descargar" fuerza endpoint de descarga
      li.innerHTML = `
        <span style="opacity:0.8;">[${kind}]</span>
        <a href="${viewUrl}" target="_blank" rel="noreferrer">${name}</a>
        ${dlUrl ? ` - <a href="${dlUrl}" target="_blank" rel="noreferrer">Descargar</a>` : ""}
      `;

      ul.appendChild(li);
    });
  },
};
