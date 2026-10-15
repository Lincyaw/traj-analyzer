// The review pages: the batches, the trajectories worth a look, one trajectory's course, and the features.
// The address holds the page and its parameters, so any view can be shared as a link.

import { get } from "./api.js";
import { clear, el } from "./dom.js";
import * as features from "./views/features.js";
import * as overview from "./views/overview.js";
import * as trajectories from "./views/trajectories.js";
import * as trajectory from "./views/trajectory.js";

const PAGES = [
  { id: "overview", title: "Overview", view: overview },
  { id: "trajectories", title: "Trajectories", view: trajectories },
  { id: "features", title: "Features", view: features },
];
const DETAIL = { id: "trajectory", view: trajectory };

const main = document.getElementById("main");
const errorBox = document.getElementById("error");
let rendering = 0;

function showError(text) {
  errorBox.textContent = text;
  errorBox.hidden = !text;
}

window.addEventListener("unhandledrejection", (event) => showError(event.reason?.message ?? String(event.reason)));

export function address(page, params = {}, path = []) {
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== "");
  const query = new URLSearchParams(entries).toString();
  const parts = [page, ...path.map(encodeURIComponent)].join("/");
  return `#/${parts}${query ? `?${query}` : ""}`;
}

export function navigate(page, params = {}, path = []) {
  location.hash = address(page, params, path);
}

function parse() {
  const [pathPart, queryPart] = location.hash.replace(/^#\/?/, "").split("?");
  const [page, ...rest] = pathPart.split("/");
  return { page: page || PAGES[0].id, path: rest.map(decodeURIComponent), params: Object.fromEntries(new URLSearchParams(queryPart ?? "")) };
}

async function render() {
  const { page, path, params } = parse();
  const found = PAGES.find((candidate) => candidate.id === page) ?? (page === DETAIL.id ? DETAIL : null);
  for (const button of document.querySelectorAll("#pages button")) {
    button.setAttribute("aria-current", String(button.dataset.page === page));
  }
  showError("");
  if (!found) {
    navigate(PAGES[0].id);
    return;
  }
  const token = ++rendering;
  const container = el("section", { class: `page page-${found.id}` });
  clear(main).append(container);
  try {
    await found.view.render(container, { params, path, navigate, address });
  } catch (error) {
    if (token === rendering) {
      showError(error.message ?? String(error));
    }
  }
}

async function start() {
  const project = await get("project");
  document.getElementById("project").textContent = project.name;
  document.getElementById("project").title = project.root;
  document.title = `${project.name} · traj review`;
  const nav = document.getElementById("pages");
  for (const page of PAGES) {
    nav.append(el("button", { dataset: { page: page.id }, text: page.title, on: { click: () => navigate(page.id) } }));
  }
  window.addEventListener("hashchange", render);
  await render();
}

start();
