/* Dashboard behaviour.
 *
 * Two request shapes only. Read-only views GET a JSON endpoint and render it.
 * Actions POST to /api/actions/<name>, get a job id back, and then poll
 * /api/jobs/<id> until the job leaves the "running" state, streaming the captured
 * log lines into the drawer. Nothing is faked client-side: every number on screen
 * came from a file on disk or from the API answering a request.
 */

(function () {
  "use strict";

  var API_URL = document.body.dataset.apiUrl || "";
  var REFRESH_MS = (parseInt(document.body.dataset.refresh, 10) || 5) * 1000;
  var pollTimer = null;
  var statusTimer = null;

  // ---------- small helpers ----------

  function $(selector) { return document.querySelector(selector); }
  function $$(selector) { return Array.prototype.slice.call(document.querySelectorAll(selector)); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function clear(node) { while (node && node.firstChild) { node.removeChild(node.firstChild); } }

  function fmt(value, digits) {
    if (value === null || value === undefined || value === "") { return "—"; }
    if (typeof value === "number") { return value.toFixed(digits === undefined ? 3 : digits); }
    return String(value);
  }

  function bytes(value) {
    if (!value) { return "0 B"; }
    var units = ["B", "KB", "MB", "GB"];
    var index = 0;
    var size = Number(value);
    while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
    return size.toFixed(index === 0 ? 0 : 1) + " " + units[index];
  }

  function toast(message, kind) {
    var node = $("#toast");
    node.textContent = message;
    node.className = "toast" + (kind ? " " + kind : "");
    node.hidden = false;
    window.clearTimeout(node._timer);
    node._timer = window.setTimeout(function () { node.hidden = true; }, 4200);
  }

  function getJSON(url) {
    return fetch(url, { headers: { Accept: "application/json" } }).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) { throw new Error(body.error || ("HTTP " + response.status)); }
        return body;
      });
    });
  }

  function postJSON(url, payload) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {})
    }).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) { throw new Error(body.error || ("HTTP " + response.status)); }
        return body;
      });
    });
  }

  function setBusy(isBusy) {
    $$("button[data-action-button]").forEach(function (button) { button.disabled = isBusy; });
    $("#hdr-job").textContent = isBusy ? "running" : "idle";
  }

  // ---------- jobs ----------

  function startAction(name, url, payload, onDone) {
    setBusy(true);
    openDrawer(name, "running");
    postJSON(url, payload)
      .then(function (job) { pollJob(job.job_id, name, onDone); })
      .catch(function (error) {
        setBusy(false);
        setDrawerStatus("failed");
        appendDrawer(String(error.message || error));
        toast(String(error.message || error), "fail");
      });
  }

  function pollJob(jobId, name, onDone) {
    window.clearTimeout(pollTimer);
    getJSON("/api/jobs/" + jobId)
      .then(function (job) {
        renderDrawerLogs(job.logs || []);
        setDrawerStatus(job.status);
        if (job.status === "running") {
          pollTimer = window.setTimeout(function () { pollJob(jobId, name, onDone); }, 1200);
          return;
        }
        setBusy(false);
        if (job.status === "succeeded") {
          toast(name + " finished in " + fmt(job.duration_seconds, 1) + "s", "ok");
        } else {
          toast(name + " failed: " + job.error, "fail");
        }
        refreshStatus();
        refreshJobs();
        if (onDone) { onDone(job); }
      })
      .catch(function (error) {
        setBusy(false);
        setDrawerStatus("failed");
        toast(String(error.message || error), "fail");
      });
  }

  function openDrawer(title, status) {
    $("#drawer").hidden = false;
    $("#drawer-title").textContent = title;
    setDrawerStatus(status);
    $("#drawer-log").textContent = "";
  }

  function setDrawerStatus(status) {
    var node = $("#drawer-status");
    node.textContent = status;
    node.className = "tag" + (status === "succeeded" ? " ok" : status === "failed" ? " fail" : "");
  }

  function appendDrawer(line) {
    var node = $("#drawer-log");
    node.textContent += (node.textContent ? "\n" : "") + line;
    node.scrollTop = node.scrollHeight;
  }

  function renderDrawerLogs(lines) {
    var node = $("#drawer-log");
    node.textContent = lines.map(function (line) {
      try {
        var parsed = JSON.parse(line);
        var extras = Object.keys(parsed)
          .filter(function (key) {
            return ["ts", "level", "logger", "message", "request_id"].indexOf(key) === -1;
          })
          .map(function (key) { return key + "=" + JSON.stringify(parsed[key]); })
          .join(" ");
        return (parsed.level || "INFO").padEnd(7) + " " + (parsed.message || "") + (extras ? "  " + extras : "");
      } catch (error) {
        return line;
      }
    }).join("\n");
    node.scrollTop = node.scrollHeight;
  }

  function refreshJobs() {
    getJSON("/api/jobs").then(function (data) {
      var body = $("#jobs-table").querySelector("tbody");
      clear(body);
      (data.jobs || []).forEach(function (job) {
        var row = el("tr");
        row.appendChild(el("td", null, job.name));
        var status = el("td");
        status.appendChild(el("span", "tag " + (job.status === "succeeded" ? "ok" : job.status === "failed" ? "fail" : ""), job.status));
        row.appendChild(status);
        row.appendChild(el("td", "num", fmt(job.duration_seconds, 1) + "s"));
        row.appendChild(el("td", null, job.created_at));
        body.appendChild(row);
      });
      if (!(data.jobs || []).length) {
        var empty = el("tr");
        var cell = el("td", null, "No actions have been run yet. Start with “Run the whole pipeline”.");
        cell.colSpan = 4;
        empty.appendChild(cell);
        body.appendChild(empty);
      }
      setBusy(Boolean(data.busy));
    }).catch(function () { /* the jobs table is not worth a toast */ });
  }

  // ---------- overview ----------

  function renderSpine(stages) {
    var list = $("#spine-nodes");
    clear(list);
    stages.forEach(function (stage) {
      var item = el("li", "spine-node" + (stage.done ? " done" : ""));
      var button = el("button");
      button.type = "button";
      button.appendChild(el("span", "spine-marker"));
      button.appendChild(el("span", "spine-label", stage.label));
      button.appendChild(el("span", "spine-detail", stage.detail));
      button.addEventListener("click", function () {
        var map = {
          data: "#panel-data", version: "#panel-data", train: "#panel-training",
          evaluate: "#panel-training", serve: "#panel-api", monitor: "#panel-monitoring",
          deploy: "#panel-deployment"
        };
        var target = document.querySelector(map[stage.key] || "#panel-overview");
        if (target) { target.scrollIntoView({ behavior: "smooth", block: "start" }); }
      });
      item.appendChild(button);
      list.appendChild(item);
    });
  }

  function stat(label, value, sub, kind) {
    var node = el("div", "stat" + (kind ? " " + kind : ""));
    node.appendChild(el("div", "stat-label", label));
    node.appendChild(el("div", "stat-value", value));
    if (sub) { node.appendChild(el("div", "stat-sub", sub)); }
    return node;
  }

  function renderOverview(status) {
    var grid = $("#stat-grid");
    clear(grid);

    var dataset = status.dataset || {};
    var processed = dataset.processed || {};
    var model = status.model || {};
    var baseline = model.baseline || {};
    var metrics = baseline.metrics || {};
    var api = status.api || {};
    var perf = status.perf_report;

    grid.appendChild(stat("Images", processed.total_images || dataset.raw_total || 0,
      (dataset.raw_total || 0) + " raw", processed.total_images ? "ok" : "warn"));
    grid.appendChild(stat("Test accuracy", metrics.accuracy !== undefined ? fmt(metrics.accuracy, 4) : "—",
      metrics.f1 !== undefined ? "F1 " + fmt(metrics.f1, 3) : "not evaluated",
      metrics.accuracy !== undefined ? "ok" : "warn"));
    grid.appendChild(stat("Checkpoint", model.exists ? bytes(model.bytes) : "none",
      model.exists ? "artifacts/model.pkl" : "run training", model.exists ? "ok" : "warn"));
    grid.appendChild(stat("API", api.reachable ? "up" : "down",
      api.base_url || "", api.reachable ? "ok" : "fail"));
    grid.appendChild(stat("Tracked runs", (status.runs || []).length,
      status.mlflow_available ? "mlflow + local store" : "local store", "ok"));
    grid.appendChild(stat("Perf check", perf ? (perf.passed ? "PASS" : "FAIL") : "—",
      perf ? "accuracy " + fmt((perf.metrics || {}).accuracy, 3) : "not run",
      perf ? (perf.passed ? "ok" : "fail") : "warn"));

    var list = $("#stage-list");
    clear(list);
    (status.stages || []).forEach(function (stage) {
      var item = el("li", stage.done ? "done" : "");
      item.appendChild(el("span", "stage-dot"));
      item.appendChild(el("span", "stage-name", stage.label));
      item.appendChild(el("span", "stage-detail", stage.detail));
      list.appendChild(item);
    });

    var done = (status.stages || []).filter(function (s) { return s.done; }).length;
    $("#overview-note").textContent = done + " of " + (status.stages || []).length + " stages complete";
    $("#hdr-model").textContent = model.exists
      ? ((baseline.model || {}).model_type || "model") + " · " + bytes(model.bytes)
      : "none";
  }

  function renderData(status) {
    var dataset = status.dataset || {};
    var processed = dataset.processed || {};
    var counts = processed.counts || {};
    var body = $("#split-table").querySelector("tbody");
    clear(body);
    ["train", "val", "test"].forEach(function (split) {
      var bucket = counts[split] || {};
      var row = el("tr");
      row.appendChild(el("td", null, split));
      row.appendChild(el("td", "num", bucket.cat || 0));
      row.appendChild(el("td", "num", bucket.dog || 0));
      row.appendChild(el("td", "num", bucket.total || 0));
      body.appendChild(row);
    });

    var kv = $("#data-kv");
    clear(kv);
    var ratios = processed.split_ratios || {};
    [
      ["Geometry", processed.image_size ? processed.image_size + "×" + processed.image_size + " RGB" : "—"],
      ["Split", ratios.train ? [ratios.train, ratios.val, ratios.test].map(function (r) { return Math.round(r * 100) + "%"; }).join(" / ") : "—"],
      ["Split method", "sha256(" + (processed.hash_salt || "salt") + " + path)"],
      ["Dataset digest", (processed.dataset_digest || "—").slice(0, 24)],
      ["Manifest", dataset.manifest_exists ? "data/processed/manifest.csv" : "not written"]
    ].forEach(function (pair) {
      kv.appendChild(el("dt", null, pair[0]));
      kv.appendChild(el("dd", null, pair[1]));
    });

    var versioning = status.versioning || {};
    var lock = versioning.lock || {};
    var git = versioning.git || {};
    var vkv = $("#version-kv");
    clear(vkv);
    [
      ["DVC installed", versioning.dvc_installed ? "yes" : "no — pip install dvc"],
      ["DVC repo", versioning.dvc_initialised ? "initialised" : "not initialised"],
      ["dvc.yaml", versioning.dvc_yaml ? "present" : "missing"],
      ["Lock digest", (lock.combined_digest || "—").slice(0, 24)],
      ["Raw tracked", lock.raw ? lock.raw.files + " files · " + bytes(lock.raw.bytes) : "—"],
      ["Processed tracked", lock.processed ? lock.processed.files + " files · " + bytes(lock.processed.bytes) : "—"],
      ["Git branch", git.branch || "—"],
      ["Git commit", (git.commit || "—") + (git.dirty ? " (dirty)" : "")]
    ].forEach(function (pair) {
      vkv.appendChild(el("dt", null, pair[0]));
      vkv.appendChild(el("dd", null, pair[1]));
    });
  }

  function renderTraining(status) {
    var model = status.model || {};
    var baseline = model.baseline || {};
    var metrics = baseline.metrics || {};
    var history = (model.history || {}).history || {};
    var meta = baseline.model || {};

    var kv = $("#training-kv");
    clear(kv);
    var lastVal = (history.val_accuracy || []).slice(-1)[0];
    [
      ["Model type", meta.model_type || "—"],
      ["Epochs run", (model.history || {}).epochs_run || "—"],
      ["Best epoch", (model.history || {}).best_epoch || "—"],
      ["Final val accuracy", lastVal !== undefined ? fmt(lastVal, 4) : "—"],
      ["Test accuracy", fmt(metrics.accuracy, 4)],
      ["Precision / recall", fmt(metrics.precision, 3) + " / " + fmt(metrics.recall, 3)],
      ["F1", fmt(metrics.f1, 4)],
      ["ROC AUC", fmt(metrics.roc_auc, 4)],
      ["Log loss", fmt(metrics.log_loss, 4)],
      ["Trained at", meta.trained_at || "—"],
      ["Checkpoint", model.exists ? model.checkpoint + " (" + bytes(model.bytes) + ")" : "none"]
    ].forEach(function (pair) {
      kv.appendChild(el("dt", null, pair[0]));
      kv.appendChild(el("dd", null, pair[1]));
    });

    var promotion = model.promotion || {};
    var pkv = $("#promotion-kv");
    clear(pkv);
    [
      ["Decision", promotion.promoted === undefined ? "not run" : (promotion.promoted ? "PROMOTED" : "REJECTED")],
      ["Reason", promotion.reason || "—"],
      ["Threshold", promotion.threshold !== undefined ? fmt(promotion.threshold, 3) : "—"],
      ["Registered name", promotion.registered_model_name || "—"],
      ["Checkpoint match", promotion.checkpoint_matches_run === undefined ? "—"
        : (promotion.checkpoint_matches_run ? "the scored run's weights are on disk" : "MISMATCH — the file on disk was not the model that was scored")],
      ["MLflow registry", promotion.mlflow_registered ? "registered" : "local decision only"],
      ["Decided at", promotion.decided_at || "—"]
    ].forEach(function (pair) {
      pkv.appendChild(el("dt", null, pair[0]));
      pkv.appendChild(el("dd", null, pair[1]));
    });

    var grid = $("#plot-grid");
    clear(grid);
    (status.plots || []).forEach(function (plot) {
      var figure = el("figure");
      var image = el("img");
      image.src = plot.url + "?t=" + Date.now();
      image.alt = plot.name;
      image.loading = "lazy";
      figure.appendChild(image);
      figure.appendChild(el("figcaption", null, plot.name));
      grid.appendChild(figure);
    });
    if (!(status.plots || []).length) {
      grid.appendChild(el("p", "hint", "Train and evaluate a model to generate the curves."));
    }
  }

  function renderRuns(status) {
    var body = $("#runs-table").querySelector("tbody");
    clear(body);
    (status.runs || []).forEach(function (run) {
      var row = el("tr", "clickable");
      var metrics = run.metrics || {};
      var key = metrics.test_accuracy !== undefined ? "test_accuracy " + fmt(metrics.test_accuracy, 4)
        : metrics.val_accuracy !== undefined ? "val_accuracy " + fmt(metrics.val_accuracy, 4)
        : metrics.live_accuracy !== undefined ? "live_accuracy " + fmt(metrics.live_accuracy, 4)
        : "—";
      row.appendChild(el("td", null, run.started_at));
      row.appendChild(el("td", null, run.run_name));
      row.appendChild(el("td", null, (run.tags || {}).stage || "—"));
      var status_cell = el("td");
      status_cell.appendChild(el("span", "tag " + (run.status === "FINISHED" ? "ok" : run.status === "FAILED" ? "fail" : ""), run.status));
      row.appendChild(status_cell);
      row.appendChild(el("td", null, key));
      row.appendChild(el("td", null, run.backend));
      row.addEventListener("click", function () { showRun(run.run_id); });
      body.appendChild(row);
    });
    if (!(status.runs || []).length) {
      var empty = el("tr");
      var cell = el("td", null, "No runs recorded yet.");
      cell.colSpan = 6;
      empty.appendChild(cell);
      body.appendChild(empty);
    }
    $("#mlflow-note").textContent = status.mlflow_available
      ? "M1.3 — logging to MLflow and to the local run store"
      : "M1.3 — MLflow not installed here; logging to the local run store";
  }

  function showRun(runId) {
    getJSON("/api/runs/" + runId).then(function (run) {
      $("#run-detail-card").hidden = false;
      $("#run-detail-title").textContent = run.run_name + " · " + run.run_id;
      $("#run-params").textContent = JSON.stringify(run.params || {}, null, 2);
      $("#run-metrics").textContent = JSON.stringify(run.metrics || {}, null, 2);
      var list = $("#run-artifacts");
      clear(list);
      (run.artifacts || []).forEach(function (name) {
        var item = el("li");
        item.appendChild(el("span", "chip", name));
        list.appendChild(item);
      });
      $("#run-detail-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }).catch(function (error) { toast(String(error.message || error), "fail"); });
  }

  function renderPerf(report) {
    var verdict = $("#perf-verdict");
    if (!report) {
      verdict.textContent = "not run";
      verdict.className = "tag";
      return;
    }
    verdict.textContent = report.passed ? "PASS" : "FAIL";
    verdict.className = "tag " + (report.passed ? "ok" : "fail");

    var gates = $("#perf-gates");
    clear(gates);
    (report.gates || []).forEach(function (gate) {
      var item = el("li");
      item.appendChild(el("span", "gate-name", (gate.passed ? "✓ " : "✗ ") + gate.name));
      item.appendChild(el("span", "gate-detail", gate.detail));
      gates.appendChild(item);
    });

    var body = $("#perf-table").querySelector("tbody");
    clear(body);
    var live = report.metrics || {};
    var base = report.baseline_metrics || {};
    ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"].forEach(function (key) {
      if (live[key] === undefined) { return; }
      var delta = base[key] !== undefined ? live[key] - base[key] : null;
      var row = el("tr");
      row.appendChild(el("td", null, key));
      row.appendChild(el("td", "num", fmt(base[key], 4)));
      row.appendChild(el("td", "num", fmt(live[key], 4)));
      row.appendChild(el("td", "num", delta === null ? "—" : (delta >= 0 ? "+" : "") + delta.toFixed(4)));
      body.appendChild(row);
    });

    var table = $("#perf-confusion");
    clear(table);
    var head = el("tr");
    ["", "pred cat", "pred dog"].forEach(function (label) { head.appendChild(el("th", null, label)); });
    table.appendChild(head);
    (report.confusion_matrix || []).forEach(function (row, index) {
      var tr = el("tr");
      tr.appendChild(el("th", null, index === 0 ? "true cat" : "true dog"));
      row.forEach(function (value) { tr.appendChild(el("td", "num", value)); });
      table.appendChild(tr);
    });

    var kv = $("#perf-latency");
    clear(kv);
    var latency = report.latency_ms || {};
    [
      ["Endpoint", report.endpoint],
      ["Sample", report.sample_size + " of " + report.requested + " requested"],
      ["Model version", report.model_version || "—"],
      ["Mean latency", fmt(latency.mean, 1) + " ms"],
      ["p95 latency", fmt(latency.p95, 1) + " ms"],
      ["Checked at", report.checked_at]
    ].forEach(function (pair) {
      kv.appendChild(el("dt", null, pair[0]));
      kv.appendChild(el("dd", null, pair[1]));
    });
  }

  function refreshStatus() {
    return getJSON("/api/status").then(function (status) {
      renderSpine(status.stages || []);
      renderOverview(status);
      renderData(status);
      renderTraining(status);
      renderRuns(status);
      renderPerf(status.perf_report);
      $("#curl-block").textContent =
        "curl -s " + API_URL + "/health\n\n" +
        "curl -s -X POST " + API_URL + "/predict \\\n" +
        "  -F \"file=@data/processed/test/dog/dog_00000.jpg;type=image/jpeg\"";
      return status;
    }).catch(function (error) {
      $("#overview-note").textContent = "status unavailable: " + (error.message || error);
    });
  }

  // ---------- metrics, logs, pods ----------

  function refreshMetrics() {
    return getJSON("/api/metrics").then(function (data) {
      var grid = $("#metric-grid");
      clear(grid);
      if (!data.reachable) {
        grid.appendChild(stat("API", "unreachable", data.error || data.base_url, "fail"));
        return;
      }
      var summary = data.summary || {};
      var latency = summary.latency_ms || {};
      grid.appendChild(stat("Predictions", summary.predictions_total || 0,
        Object.keys(summary.predictions_by_label || {}).map(function (k) {
          return k + " " + summary.predictions_by_label[k];
        }).join(" · ") || "none yet", "ok"));
      grid.appendChild(stat("HTTP requests", summary.http_requests_total || 0,
        Object.keys(summary.http_by_status || {}).map(function (k) {
          return k + ": " + summary.http_by_status[k];
        }).join(" · "), "ok"));
      grid.appendChild(stat("Errors", summary.errors_total || 0,
        Object.keys(summary.errors_by_code || {}).join(", ") || "none",
        (summary.errors_total || 0) > 0 ? "warn" : "ok"));
      grid.appendChild(stat("Mean latency", fmt(latency.mean, 1) + " ms",
        "p50 " + fmt(latency.p50, 1) + " · p95 " + fmt(latency.p90, 1) + " · p99 " + fmt(latency.p99, 1), "ok"));
      grid.appendChild(stat("Mean confidence", fmt(summary.confidence_mean, 3),
        "across served predictions", "ok"));
      grid.appendChild(stat("Uptime", fmt(summary.uptime_seconds, 0) + " s",
        summary.model_ready ? "model loaded" : "no model", summary.model_ready ? "ok" : "warn"));
      $("#metrics-raw").textContent = data.raw || "";
    }).catch(function (error) { toast(String(error.message || error), "fail"); });
  }

  function renderLogs(bundle) {
    $("#log-source-tag").textContent = bundle.source || "—";
    $("#log-message").textContent = bundle.message || "";
    var view = $("#log-view");
    clear(view);
    var records = bundle.records || [];
    if (!records.length) {
      view.appendChild(el("div", "log-line", "No log records for this source yet."));
      return;
    }
    records.slice(-400).forEach(function (record) {
      var line = el("div", "log-line");
      line.appendChild(el("span", "log-ts", (record.ts || "").replace("T", " ").replace("Z", "")));
      line.appendChild(el("span", "log-level " + (record.level || "INFO"), record.level || "INFO"));
      line.appendChild(el("span", "log-logger", record.pod ? record.pod : (record.logger || "-")));
      var extras = Object.keys(record).filter(function (key) {
        return ["ts", "level", "logger", "message", "pod", "source", "request_id"].indexOf(key) === -1;
      }).map(function (key) { return key + "=" + JSON.stringify(record[key]); }).join(" ");
      line.appendChild(el("span", "log-msg", (record.message || "") + (extras ? "   " + extras : "")));
      view.appendChild(line);
    });
    view.scrollTop = view.scrollHeight;
  }

  function refreshLogs() {
    var source = $("#log-source").value;
    return getJSON("/api/logs?source=" + encodeURIComponent(source)).then(renderLogs)
      .catch(function (error) { toast(String(error.message || error), "fail"); });
  }
  function refreshContainers() {
    return getJSON("/api/containers").then(function (data) {
      $("#containers-source").textContent = data.source || "—";
      var body = $("#containers-table").querySelector("tbody");
      clear(body);
      var containers = data.containers || data.pods || [];
      if (!containers.length) {
        var empty = el("tr");
        var cell = el("td", null, data.error
          ? data.error
          : "No containers visible. Start the stack with `docker compose -f docker/docker-compose.yml up`.");
        cell.colSpan = 3;
        empty.appendChild(cell);
        body.appendChild(empty);
        return;
      }
      containers.forEach(function (c) {
        var row = el("tr");
        row.appendChild(el("td", null, c.name));
        row.appendChild(el("td", null, c.status || "—"));
        row.appendChild(el("td", null, c.image || "—"));
        body.appendChild(row);
      });
    }).catch(function () { /* containers are optional */ });
  }

  function refreshDeployment() {
    return getJSON("/api/deployment").then(function (data) {
      [["#workflow-list", data.workflows], ["#docker-list", data.docker]]
        .forEach(function (pair) {
          var list = $(pair[0]);
          clear(list);
          (pair[1] || []).forEach(function (file) {
            var item = el("li");
            var button = el("button", null, file.name);
            button.type = "button";
            button.addEventListener("click", function () {
              getJSON("/api/file?path=" + encodeURIComponent(file.path)).then(function (body) {
                $("#file-preview").textContent = body.text;
              }).catch(function (error) {
                $("#file-preview").textContent = String(error.message || error);
              });
            });
            item.appendChild(button);
            list.appendChild(item);
          });
        });
    }).catch(function () { /* deployment listing is optional */ });
  }

  // ---------- wiring ----------

  function bindAction(selector, name, url, payloadFn) {
    var button = $(selector);
    if (!button) { return; }
    button.setAttribute("data-action-button", "1");
    button.addEventListener("click", function () {
      startAction(name, url, payloadFn ? payloadFn() : {});
    });
  }

  function renderPrediction(target, data) {
    clear(target);
    var body = data.body || {};
    if (data.status_code !== 200) {
      target.appendChild(el("div", "verdict-line fail", "HTTP " + data.status_code));
      target.appendChild(el("pre", "console", JSON.stringify(body, null, 2)));
      return;
    }
    if (data.preview) {
      var image = el("img");
      image.src = data.preview;
      image.alt = "submitted image";
      target.appendChild(image);
    }
    var verdictClass = data.correct === false ? "verdict-line fail" : "verdict-line ok";
    var headline = body.label + " · " + (body.confidence * 100).toFixed(1) + "% confidence";
    if (data.true_label) {
      headline += "  (true: " + data.true_label + (data.correct ? " ✓" : " ✗") + ")";
    }
    target.appendChild(el("div", verdictClass, headline));
    var probabilities = body.probabilities || {};
    target.appendChild(el("div", "stat-sub", Object.keys(probabilities).map(function (key) {
      return key + " " + fmt(probabilities[key], 4);
    }).join("  ·  ") + "   |   inference " + fmt(body.latency_ms, 1) + " ms, round trip " + fmt(data.round_trip_ms, 1) + " ms"));
    target.appendChild(el("pre", "console", JSON.stringify(body, null, 2)));
  }

  function init() {
    // Data panel
    bindAction("#btn-generate", "generate-data", "/api/actions/generate-data", function () {
      return { per_class: parseInt($("#per-class").value, 10) || 300 };
    });
    bindAction("#btn-preprocess", "preprocess", "/api/actions/preprocess");
    bindAction("#btn-lock", "dataset-lock", "/api/actions/dataset-lock");

    $("#btn-samples").addEventListener("click", function () {
      getJSON("/api/sample-images?seed=" + Math.floor(Math.random() * 1000)).then(function (data) {
        var card = $("#samples-card");
        card.hidden = false;
        var grid = $("#samples-grid");
        clear(grid);
        (data.samples || []).forEach(function (sample) {
          var figure = el("figure");
          var image = el("img");
          image.src = sample.data_url;
          image.alt = sample.class_name;
          figure.appendChild(image);
          figure.appendChild(el("figcaption", null, sample.class_name + " · " + sample.split + " · " + sample.sha256_12));
          grid.appendChild(figure);
        });
        if (!(data.samples || []).length) {
          grid.appendChild(el("p", "hint", data.message || "No processed images yet."));
        }
      }).catch(function (error) { toast(String(error.message || error), "fail"); });
    });

    $$("[data-dvc]").forEach(function (button) {
      button.setAttribute("data-action-button", "1");
      button.addEventListener("click", function () {
        var command = button.getAttribute("data-dvc");
        startAction("dvc " + command, "/api/actions/dvc", { command: command }, function (job) {
          var result = job.result || {};
          $("#dvc-console").textContent =
            "$ " + (result.command || ("dvc " + command)) + "\n" +
            "exit " + result.returncode + "\n\n" +
            (result.stdout || "") + (result.stderr ? "\n" + result.stderr : "");
        });
      });
    });

    // Training panel
    bindAction("#btn-train", "train", "/api/actions/train", function () {
      return {
        model_type: $("#model-type").value,
        epochs: parseInt($("#epochs").value, 10),
        learning_rate: parseFloat($("#learning-rate").value)
      };
    });
    bindAction("#btn-evaluate", "evaluate", "/api/actions/evaluate");
    bindAction("#btn-promote", "promote", "/api/actions/promote");
    bindAction("#btn-run-all", "full-pipeline", "/api/actions/pipeline", function () {
      return { include_data: true, per_class: parseInt($("#per-class").value, 10) || 300 };
    });

    // API panel
    $$("[data-probe]").forEach(function (button) {
      button.addEventListener("click", function () {
        var path = button.getAttribute("data-probe");
        postJSON("/api/probe", { path: path }).then(function (data) {
          $("#probe-console").textContent =
            "GET " + API_URL + data.endpoint + "\n" +
            (data.reachable ? "HTTP " + data.status_code + "  ·  " + data.latency_ms + " ms\n\n" : "unreachable\n\n") +
            JSON.stringify(data.body || { error: data.error }, null, 2);
        }).catch(function (error) { toast(String(error.message || error), "fail"); });
      });
    });

    $("#btn-reload-model").addEventListener("click", function () {
      postJSON("/api/reload-model", {}).then(function (data) {
        var body = data.body || {};
        $("#probe-console").textContent =
          "POST " + API_URL + "/reload\nHTTP " + data.status_code + "\n\n" +
          JSON.stringify(body, null, 2);
        if (data.status_code === 200) {
          toast("API reloaded the checkpoint trained at " + (((body.model || {}).training || {}).trained_at || "?"), "ok");
        } else {
          toast("reload failed", "fail");
        }
        refreshStatus();
      }).catch(function (error) { toast(String(error.message || error), "fail"); });
    });

    $("#btn-predict").addEventListener("click", function () {
      var input = $("#predict-file");
      if (!input.files || !input.files.length) {
        toast("Choose an image first", "fail");
        return;
      }
      var form = new FormData();
      form.append("file", input.files[0]);
      fetch("/api/predict", { method: "POST", body: form })
        .then(function (response) { return response.json(); })
        .then(function (data) { renderPrediction($("#predict-result"), data); })
        .catch(function (error) { toast(String(error.message || error), "fail"); });
    });

    $("#btn-predict-sample").addEventListener("click", function () {
      postJSON("/api/predict-sample", {}).then(function (data) {
        renderPrediction($("#predict-result"), data);
        refreshMetrics();
      }).catch(function (error) { toast(String(error.message || error), "fail"); });
    });

    // Monitoring panel
    $("#btn-load-test").addEventListener("click", function () {
      toast("sending 20 predictions…");
      postJSON("/api/load-test", { count: 20 }).then(function (data) {
        toast("burst done · p95 " + fmt(data.latency_ms.p95, 1) + " ms · accuracy " + fmt(data.accuracy_on_burst, 3), "ok");
        refreshMetrics();
        refreshLogs();
      }).catch(function (error) { toast(String(error.message || error), "fail"); });
    });

    $("#btn-metrics-raw").addEventListener("click", function () {
      var node = $("#metrics-raw");
      node.hidden = !node.hidden;
    });

    $("#btn-logs").addEventListener("click", refreshLogs);
    $("#log-source").addEventListener("change", refreshLogs);

    // The dashboard supports Docker and local file tails as log sources.

    // Performance panel
    bindAction("#btn-perf", "perf-check", "/api/actions/perf-check", function () {
      return { sample_size: parseInt($("#perf-sample").value, 10) || 40 };
    });

    // Header + drawer
    $("#btn-refresh").addEventListener("click", function () {
      refreshStatus();
      refreshMetrics();
      refreshLogs();
      refreshContainers();
      refreshJobs();
      toast("refreshed");
    });
    $("#drawer-close").addEventListener("click", function () { $("#drawer").hidden = true; });

    getJSON("/api/model-card").then(function (data) {
      if (data.exists) { $("#model-card").textContent = data.markdown; }
    }).catch(function () { /* the card is optional */ });

    refreshStatus();
    refreshMetrics();
    refreshLogs();
    refreshContainers();
    refreshJobs();
    refreshDeployment();

    statusTimer = window.setInterval(function () {
      if (document.hidden) { return; }
      refreshStatus();
      refreshMetrics();
      refreshJobs();
    }, REFRESH_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
