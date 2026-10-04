const element = (id) => document.getElementById(id);
const isNested = (type) => type.endsWith("[]") || type.startsWith("MAP");
const isNumeric = (type) => ["DOUBLE", "FLOAT", "INTEGER", "BIGINT", "float", "integer"].includes(type);

function options(select, entries, selected = []) {
  select.replaceChildren(...entries.map(([value, label, note]) => {
    const option = new Option(label, value, false, selected.includes(value));
    option.title = note ?? label;
    return option;
  }));
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function response(url, body) {
  const result = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!result.ok) {
    const error = await result.json();
    throw new Error(typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail));
  }
  return result;
}

export class Analysis {
  constructor(showError) {
    this.showError = showError;
    this.worker = null;
    this.viewer = null;
    this.data = null;
    this.table = null;
    this.state = null;
    this.request = null;
    this.schema = {};
    this.busy = false;
    this.setBusy(true);
  }

  initialize(project, tables, restoreQuery) {
    this.tables = tables;
    this.restoreQuery = restoreQuery;
    this.storageKey = `traj-analysis:${project.root}`;
    this.updateSavedViews();
    element("analysis-source").onchange = async () => {
      this.configureColumns();
      await this.load(this.state, false);
    };
    element("load-columns").onclick = async () => {
      element("load-columns").closest("details").open = false;
      await this.load(this.state, false);
    };
    element("distribution").onclick = () => this.preset("distribution");
    element("compare").onclick = () => this.preset("compare");
    element("relationship").onclick = () => this.preset("relationship");
    element("add-panel").onclick = async () => {
      await this.viewer.addPanel({ ...await this.viewer.save(), title: "Analysis" });
    };
    element("reset-view").onclick = () => this.preset("distribution");
    element("export-data").onclick = () => this.viewer.download();
    element("save-view").onclick = async () => {
      const name = element("view-name").value.trim();
      if (!name) throw new Error("Enter a view name before saving");
      const views = this.savedViews();
      const previous = views.findIndex((view) => view.name === name);
      const view = { name, ...await this.snapshot() };
      if (previous < 0) views.push(view);
      else views[previous] = view;
      localStorage.setItem(this.storageKey, JSON.stringify(views));
      this.updateSavedViews(name);
    };
    element("saved-views").onchange = async () => {
      const view = this.savedViews().find((view) => view.name === element("saved-views").value);
      if (view) await this.restore(view);
    };
    element("delete-view").onclick = () => {
      const name = element("saved-views").value;
      if (!name) throw new Error("Choose the saved view to delete");
      localStorage.setItem(this.storageKey, JSON.stringify(this.savedViews().filter((view) => view.name !== name)));
      this.updateSavedViews();
    };
    element("export-view").onclick = async () => {
      download("analysis-view.json", JSON.stringify(await this.snapshot(), null, 2), "application/json");
    };
    element("import-view").onclick = () => element("view-file").click();
    element("view-file").onchange = async (event) => {
      const file = event.target.files[0];
      event.target.value = "";
      if (file) await this.restore(JSON.parse(await file.text()));
    };
  }

  savedViews() {
    const views = JSON.parse(localStorage.getItem(this.storageKey) ?? "[]");
    if (!Array.isArray(views)) throw new Error("Saved analysis views must be a JSON array");
    return views;
  }

  updateSavedViews(name = "") {
    options(element("saved-views"), [["", "Choose a view"], ...this.savedViews().map((view) => [view.name, view.name])], [name]);
  }

  configureColumns() {
    const feature = element("analysis-source").value;
    const selected = this.table.columns.filter((column) => feature
      ? column.type === "VARCHAR" && !column.name.endsWith("__status") && !column.name.endsWith("__detail")
        && !column.name.endsWith("__evidence")
      : !isNested(column.type)).map((column) => column.name);
    options(element("analysis-columns"), this.table.columns.map((column) =>
      [column.name, column.name, column.note]), selected);
  }

  async show(table, state, load = true) {
    if (!this.table) this.setBusy(false);
    const changed = this.table?.name !== table.name;
    this.table = table;
    this.state = state;
    if (changed) {
      options(element("analysis-source"), [["", "Trajectory features"], ...table.columns.filter((column) =>
        isNested(column.type)).map((column) => [column.name, `Labels: ${column.name}`, column.note])]);
      this.configureColumns();
    }
    if (!load) return;
    if (changed || !this.viewer || this.request.search !== state.search || this.request.where !== state.where) {
      await this.load(state, !changed && !!this.viewer);
    } else {
      await this.viewer.resize();
    }
  }

  setBusy(value) {
    this.busy = value;
    for (const node of document.querySelectorAll(".analysis-controls button, .analysis-controls select, #tables button, .modes button, #apply, #reset")) {
      node.disabled = value;
    }
    for (const id of ["distribution", "compare", "relationship", "add-panel", "export-data", "export-view", "save-view", "reset-view"]) {
      element(id).disabled = value || !this.viewer;
    }
  }

  async dispose() {
    if (this.viewer) {
      const previous = this.viewer;
      this.viewer = null;
      previous.remove();
      await previous.delete();
    }
    if (this.data) {
      await this.data.delete();
      this.data = null;
    }
  }

  async load(state, preserve = true, restoredWorkspace = null) {
    if (this.busy) throw new Error("Analysis is loading");
    this.setBusy(true);
    this.showError("");
    try {
      const workspace = restoredWorkspace ?? (preserve && this.viewer ? await this.viewer.saveWorkspace() : null);
      await this.dispose();
      this.request = {
        table: this.table.name, search: state.search, where: state.where,
        feature: element("analysis-source").value || null,
        columns: Array.from(element("analysis-columns").selectedOptions, (option) => option.value),
      };
      element("analysis-info").textContent = "Loading all matching rows...";
      if (!this.worker) {
        const { createWorker } = await import("./vendor/perspective.js");
        this.worker = await createWorker();
      }
      const result = await response("api/analysis", this.request);
      this.data = await this.worker.table(await result.arrayBuffer(), { name: "analysis" });
      this.schema = await this.data.schema();
      this.viewer = document.createElement("perspective-viewer");
      this.viewer.setAttribute("theme", window.matchMedia("(prefers-color-scheme: dark)").matches ? "Pro Dark" : "Pro Light");
      element("analysis").replaceChildren(this.viewer);
      await this.viewer.load(this.worker);
      const entries = Object.entries(this.schema);
      options(element("analysis-measure"), entries.filter(([name]) => name !== "key").map(([name]) => [name, name]),
        [this.request.feature ? "__value" : entries.find(([name, type]) => name !== "__count" && isNumeric(type))?.[0] ?? "__count"]);
      options(element("analysis-group"), entries.filter(([name]) => name !== "__count").map(([name]) => [name, name]),
        [this.request.feature ? "__label" : this.schema.dataset ? "dataset"
          : entries.find(([name, type]) => name !== "key" && type === "string")?.[0] ?? entries[0][0]]);
      this.populationInfo = `${Number(result.headers.get("X-Analysis-Rows")).toLocaleString()} input rows; all matching rows loaded.`;
      if (workspace) await this.viewer.restoreWorkspace(workspace);
      else await this.applyPreset(this.request.feature ? "labels" : "distribution");
      element("analysis-info").textContent = this.populationInfo;
    } catch (error) {
      element("analysis-info").textContent = "Analysis could not be loaded.";
      await this.dispose();
      throw error;
    } finally {
      this.setBusy(false);
    }
  }

  async preset(kind) {
    if (this.busy) throw new Error("Analysis is loading");
    this.setBusy(true);
    this.showError("");
    try {
      await this.applyPreset(kind);
    } finally {
      this.setBusy(false);
    }
  }

  async applyPreset(kind) {
    const column = element("analysis-measure").value;
    const numeric = isNumeric(this.schema[column]);
    let config = { table: "analysis", plugin: "Y Bar", group_by: [], split_by: [], filter: [], sort: [],
      expressions: {}, columns: [column], aggregates: { [column]: column === "__count" ? "sum" : numeric ? "avg" : "count" }, settings: true };
    if (kind === "distribution") {
      const stats = await (await response("api/profile", { ...this.request, column })).json();
      if (numeric && stats.valid) {
        const width = stats.bin_width;
        const start = stats.bin_start;
        config = { ...config, columns: ["__count"], group_by: ["__distribution_bin"], aggregates: { __count: "sum" },
          filter: [[column, "is not null"]], sort: [["__distribution_bin", "asc"]],
          expressions: { __distribution_bin: `min(floor(("${column}" - ${start}) / ${width}), ${stats.bins - 1}) * ${width} + ${start}` } };
      } else if (!numeric) {
        config = { ...config, columns: ["__count"], group_by: [column], aggregates: { __count: "sum" },
          sort: [["__count", "desc"]] };
      } else {
        config.plugin = "Datagrid";
      }
      this.populationInfo = `${stats.total.toLocaleString()} input rows; ${stats.nulls.toLocaleString()} null values in ${column}.`;
    } else if (kind === "compare") {
      config.group_by = [element("analysis-group").value];
    } else if (kind === "labels") {
      config.group_by = ["__label"];
      config.aggregates = { __value: "sum" };
      config.sort = [["__value", "desc"]];
    } else if (kind === "relationship") {
      const other = Object.entries(this.schema).find(([name, type]) => name !== column && name !== "__count" && isNumeric(type));
      if (!numeric || !other) throw new Error("Relationship requires at least two loaded numeric columns");
      config.plugin = "X/Y Scatter";
      config.columns = [column, other[0]];
      config.aggregates = {};
    }
    await this.viewer.restore(config);
    element("analysis-info").textContent = this.populationInfo;
  }

  async snapshot() {
    if (!this.viewer || this.busy) throw new Error("Load analysis before saving a view");
    return { version: 1, ...this.request, workspace: await this.viewer.saveWorkspace() };
  }

  async restore(snapshot) {
    if (snapshot.version !== 1 || !snapshot.workspace || !this.tables.some((table) => table.name === snapshot.table)) {
      throw new Error("The analysis view has an unsupported version or an unknown source table");
    }
    await this.restoreQuery({ ...snapshot, search: snapshot.search ?? "", where: snapshot.where ?? "" });
    element("analysis-source").value = snapshot.feature ?? "";
    if (element("analysis-source").value !== (snapshot.feature ?? "")) throw new Error("The label feature no longer exists");
    this.configureColumns();
    const names = new Set(snapshot.columns);
    for (const option of element("analysis-columns").options) option.selected = names.has(option.value);
    if (Array.from(element("analysis-columns").selectedOptions).length !== names.size) throw new Error("Saved columns no longer exist");
    await this.load(this.state, false, snapshot.workspace);
    element("view-name").value = snapshot.name ?? "";
  }
}
