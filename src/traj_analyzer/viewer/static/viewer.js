const state = { table: null, search: "", where: "" };
const errorBox = document.getElementById("error");
let grid = null;

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

function show(table) {
  state.table = table.name;
  for (const tab of document.querySelectorAll("#tables button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.table === table.name));
  }
  showError("");
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

function apply() {
  state.search = document.getElementById("search").value;
  state.where = document.getElementById("where").value;
  grid.setPage(1);
}

function reset() {
  document.getElementById("search").value = "";
  document.getElementById("where").value = "";
  state.search = "";
  state.where = "";
  grid.clearHeaderFilter();
  grid.clearSort();
  grid.setPage(1);
}

async function start() {
  const [project, tables] = await Promise.all([
    fetch("api/project").then((response) => response.json()),
    fetch("api/tables").then((response) => response.json()),
  ]);
  document.getElementById("project").textContent = project.name;
  document.getElementById("project").title = project.root;
  document.title = `${project.name} · traj viewer`;
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
  for (const id of ["search", "where"]) {
    document.getElementById(id).addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        apply();
      }
    });
  }
  show(tables[0]);
}

start();
