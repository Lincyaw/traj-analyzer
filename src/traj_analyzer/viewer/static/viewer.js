import { Analysis } from "./analysis.js";

const state = { table: null, search: "", where: "", mode: "analysis" };
const errorBox = document.getElementById("error");
let grid = null;
let currentTable = null;
const analysis = new Analysis(showError);

window.addEventListener("unhandledrejection", (event) => showError(event.reason.message ?? String(event.reason)));

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

function showError(text) {
  errorBox.textContent = text;
  errorBox.hidden = !text;
}

function formatNumber(value) {
  return value.toLocaleString("en-US", { maximumFractionDigits: 3 });
}

function nullCell() {
  return element("span", "null", "null");
}

function chips(items) {
  const box = element("div", "chips");
  for (const [label, value] of items) {
    const chip = element("span", "chip", label);
    if (value !== undefined) {
      chip.append(element("b", "", formatNumber(value)));
    }
    box.append(chip);
  }
  return box;
}

function formatter(column) {
  if (column.name === "key") {
    return (cell) => element("span", "key", cell.getValue());
  }
  if (column.name.endsWith("__status")) {
    return (cell) => {
      const value = cell.getValue();
      return value === null ? nullCell() : element("span", value === "ok" ? "status-ok" : "status-bad", value);
    };
  }
  if (column.type === "DOUBLE") {
    return (cell) => {
      const value = cell.getValue();
      return value === null ? nullCell() : element("span", "num", formatNumber(value));
    };
  }
  if (column.type === "BOOLEAN") {
    return (cell) => {
      const value = cell.getValue();
      return value === null ? nullCell() : element("span", `pill ${value}`, String(value));
    };
  }
  if (column.type.endsWith("[]")) {
    return (cell) => {
      const value = cell.getValue();
      if (value === null) {
        return nullCell();
      }
      if (value.length === 0) {
        return element("span", "null", "empty");
      }
      return chips(value.map((item) => [typeof item === "number" ? formatNumber(item) : item]));
    };
  }
  if (column.type.startsWith("MAP")) {
    return (cell) => {
      const value = cell.getValue();
      if (value === null) {
        return nullCell();
      }
      const entries = Object.entries(value).sort((a, b) => b[1] - a[1]);
      return entries.length === 0 ? element("span", "null", "empty") : chips(entries);
    };
  }
  return (cell) => {
    const value = cell.getValue();
    return value === null ? nullCell() : element("span", "", value);
  };
}

function cellTooltip(event, cell) {
  const value = cell.getValue();
  if (value === null) {
    return false;
  }
  return typeof value === "object" ? JSON.stringify(value, null, 1) : String(value);
}

function headerNote(column) {
  return () => {
    const note = element("div", "note");
    note.append(element("div", "note-name", column.name));
    note.append(element("div", "note-body", column.note));
    note.append(element("div", "note-type", `Column type: ${column.type}`));
    return note;
  };
}

async function show(table, loadAnalysis = true) {
  state.table = table.name;
  currentTable = table;
  for (const tab of document.querySelectorAll("#tables button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.table === table.name));
  }
  showError("");
  if (state.mode === "analysis") {
    if (grid) {
      grid.destroy();
      grid = null;
    }
    await analysis.show(table, state, loadAnalysis);
  } else {
    showRows(table);
  }
}

function showRows(table) {
  if (grid) {
    grid.destroy();
  }
  grid = new Tabulator("#grid", {
    height: "100%",
    layout: "fitData",
    placeholder: "No matching rows",
    pagination: true,
    paginationMode: "remote",
    paginationSize: 50,
    paginationSizeSelector: [25, 50, 100, 200, 500],
    paginationCounter: "rows",
    paginationButtonCount: 7,
    sortMode: "remote",
    filterMode: "remote",
    ajaxURL: "api/rows",
    ajaxConfig: "POST",
    ajaxContentType: "json",
    ajaxParams: () => ({ table: state.table, search: state.search, where: state.where }),
    dataSendParams: { page: "page", size: "size", sort: "sorters", filter: "filters" },
    columns: table.columns.map((column) => ({
      title: column.name,
      field: column.name,
      headerTooltip: headerNote(column),
      headerFilter: "input",
      headerFilterPlaceholder: "filter",
      formatter: formatter(column),
      tooltip: cellTooltip,
      hozAlign: column.type === "DOUBLE" ? "right" : "left",
      minWidth: 90,
      maxWidth: 340,
      frozen: column.name === "key",
    })),
  });
  grid.on("dataLoaded", () => showError(""));
  grid.on("dataLoadError", async (response) => {
    const body = await response.json();
    grid.clearData();
    showError(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail, null, 1));
  });
}

async function apply() {
  state.search = document.getElementById("search").value;
  state.where = document.getElementById("where").value;
  if (state.mode === "analysis") {
    await analysis.load(state);
  } else {
    await grid.setPage(1);
  }
}

async function reset() {
  document.getElementById("search").value = "";
  document.getElementById("where").value = "";
  state.search = "";
  state.where = "";
  if (state.mode === "analysis") {
    await analysis.load(state);
  } else {
    showRows(currentTable);
  }
}

async function mode(value) {
  state.mode = value;
  for (const id of ["analysis", "analysis-controls", "view-controls", "analysis-info"]) {
    document.getElementById(id).hidden = value !== "analysis";
  }
  document.getElementById("grid").hidden = value !== "rows";
  document.getElementById("analysis-mode").setAttribute("aria-pressed", String(value === "analysis"));
  document.getElementById("table-mode").setAttribute("aria-pressed", String(value === "rows"));
  showError("");
  if (value === "analysis") {
    if (grid) {
      grid.destroy();
      grid = null;
    }
    await analysis.show(currentTable, state);
  } else {
    showRows(currentTable);
  }
}

async function start() {
  const [project, tables] = await Promise.all([
    fetch("api/project").then((response) => response.json()),
    fetch("api/tables").then((response) => response.json()),
  ]);
  document.getElementById("project").textContent = project.name;
  document.getElementById("project").title = project.root;
  document.title = `${project.name} · traj viewer`;
  analysis.initialize(project, tables, async (snapshot) => {
    document.getElementById("search").value = snapshot.search;
    document.getElementById("where").value = snapshot.where;
    state.search = snapshot.search;
    state.where = snapshot.where;
    await show(tables.find((table) => table.name === snapshot.table), false);
  });
  const nav = document.getElementById("tables");
  for (const table of tables) {
    const tab = element("button", "", table.name);
    tab.dataset.table = table.name;
    tab.title = `${table.rows} rows, ${table.columns.length} columns`;
    tab.append(element("span", "count", String(table.columns.length)));
    tab.addEventListener("click", () => show(table));
    nav.append(tab);
  }
  document.getElementById("apply").addEventListener("click", apply);
  document.getElementById("reset").addEventListener("click", reset);
  document.getElementById("analysis-mode").addEventListener("click", () => mode("analysis"));
  document.getElementById("table-mode").addEventListener("click", () => mode("rows"));
  for (const id of ["search", "where"]) {
    document.getElementById(id).addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        apply();
      }
    });
  }
  await show(tables[0]);
}

start();
