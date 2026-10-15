// How every feature holds up: whether it tells trajectories apart, how often it explains an anomaly, what follows
// its events, and what the people who followed it found. Each count opens the trajectories behind it.

import { get } from "../api.js";
import { append, chip, el, fmt } from "../dom.js";

const COLUMNS = [
  { id: "name", title: "Feature", text: (f) => fmt.words(f.name), sort: (f) => f.name,
    hint: "The feature; hover for its definition. Code features run on every trajectory, judgments are model calls." },
  { id: "role", title: "Role", text: (f) => f.role ?? "–", sort: (f) => f.role ?? "",
    hint: "From traj evaluate: distinguishing features vary between trajectories beyond their noise; invariant ones hardly vary; dependent ones follow another feature; context features copy metadata." },
  { id: "noise", title: "Noise", text: (f) => fmt.number(f.noise, 2), sort: (f) => f.noise ?? -1, numeric: true,
    hint: "How much a second measurement of a judgment differs from the first; 0 for code." },
  { id: "explains", title: "Explains", text: (f) => fmt.number(f.explains), sort: (f) => f.explains, numeric: true, link: (f) => ({ feature: f.name }),
    hint: "Trajectories whose anomaly reasons name the feature: it puts them far from their neighbours." },
  { id: "events", title: "Events", text: (f) => (f.events ? `${fmt.number(f.events.unrecovered)} / ${fmt.number(f.events.total)}` : "–"), sort: (f) => f.events?.unrecovered ?? -1, numeric: true, link: (f) => (f.events ? { rule: f.name } : null),
    hint: "Unrecovered events / all events the rule found." },
  { id: "consequence", title: "Consequence lift", text: (f) => (f.consequence_lift === null ? "–" : `${fmt.number(f.consequence_lift, 2)}×`), sort: (f) => f.consequence_lift ?? -1, numeric: true,
    hint: "How much more often the user pushes back or interrupts after a turn with the event than after any turn; from traj anomalies." },
  { id: "tail", title: "Tail precision", text: (f) => fmt.percent(f.tail_precision), sort: (f) => f.tail_precision ?? -1, numeric: true,
    hint: "Share of the trajectories at the feature's tail that the mining tail check found worth a look." },
  { id: "human", title: "People found", text: (f) => (f.human_verdicts ? `${fmt.percent(f.human_precision)} of ${f.human_verdicts}` : "–"), sort: (f) => (f.human_verdicts ? f.human_precision : -1), numeric: true, link: (f) => (f.human_verdicts ? { feature: f.name, reviewed: "true" } : null),
    hint: "Share of the verdicts people recorded on this feature that found the trajectory worth their look." },
  { id: "alert", title: "Alert", text: (f) => (f.alert ? "yes" : "–"), sort: (f) => (f.alert ? 1 : 0),
    hint: "Whether the feature raises alerts: its tail precision and lift passed the thresholds of traj.yaml." },
  { id: "shift", title: "Largest shift", text: (f) => (f.shift ? `${f.shift.direction === "up" ? "▲" : "▼"} ${fmt.number(Math.abs(f.shift.effect), 1)} in ${f.shift.batch}` : "–"), sort: (f) => Math.abs(f.shift?.effect ?? 0), numeric: true, link: (f) => (f.shift ? { batch: f.shift.batch, feature: f.name } : null),
    hint: "The batch where the feature lies furthest from the baseline, with the standardized effect." },
];

export async function render(container, { navigate }) {
  const features = await get("review/features");
  const state = { sort: "explains", descending: true, search: "", signal: true };
  container.append(el("h1", { text: "Features" }));
  container.append(el("p", { class: "lead", text: "Each row is a feature with how it holds up. A feature is effective when it explains anomalies or finds events that people and consequences confirm; hover a column title for its meaning, click a count to open the trajectories behind it." }));
  const search = el("input", { type: "search", placeholder: "Filter by name", "aria-label": "Filter features by name" });
  const signal = el("input", { type: "checkbox", checked: true });
  const controls = el("div", { class: "filters" },
    el("label", { class: "filter" }, el("span", { class: "filter-name", text: "Name" }), search),
    el("label", { class: "filter check" }, signal, " only features with a signal: they explain an anomaly, found an event, or were reviewed"),
  );
  const table = el("table", { class: "grid features" });
  container.append(controls, table);
  const draw = () => {
    const shown = features
      .filter((f) => !state.search || f.name.includes(state.search))
      .filter((f) => !state.signal || f.explains > 0 || f.events || f.human_verdicts || f.alert)
      .sort((a, b) => compare(COLUMNS.find((c) => c.id === state.sort), a, b, state.descending));
    table.replaceChildren(head(state, (id) => {
      state.descending = state.sort === id ? !state.descending : true;
      state.sort = id;
      draw();
    }), body(shown, navigate));
  };
  search.addEventListener("input", () => { state.search = search.value.trim(); draw(); });
  signal.addEventListener("change", () => { state.signal = signal.checked; draw(); });
  draw();
}

function compare(column, a, b, descending) {
  const [x, y] = [column.sort(a), column.sort(b)];
  const order = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
  return (descending ? -order : order) || a.name.localeCompare(b.name);
}

function head(state, onsort) {
  return el("thead", {}, el("tr", {}, COLUMNS.map((column) => el("th", {
    class: `${column.numeric ? "num" : ""} sortable ${state.sort === column.id ? (state.descending ? "desc" : "asc") : ""}`.trim(),
    title: column.hint,
    on: { click: () => onsort(column.id) },
  }, column.title))));
}

function body(features, navigate) {
  return el("tbody", {}, features.map((feature) => el("tr", {}, COLUMNS.map((column) => {
    const cell = el("td", { class: column.numeric ? "num" : "" });
    const target = column.link?.(feature);
    const text = column.text(feature);
    if (column.id === "name") {
      cell.title = `${feature.description}\n\n${feature.kind} feature of ${feature.operator}${feature.unit ? `, per ${feature.unit}` : ""}`;
      append(cell, [el("b", { text }), " ", chip(feature.kind, { kind: "muted" }), feature.unit ? chip(`per ${feature.unit}`, { kind: "muted" }) : null, feature.event ? chip(`event on ${feature.event}`, { kind: "event" }) : null]);
    } else if (target && text !== "–") {
      cell.append(el("button", { class: "link", text, on: { click: () => navigate("trajectories", target) } }));
    } else {
      cell.textContent = text;
    }
    return cell;
  }))));
}
