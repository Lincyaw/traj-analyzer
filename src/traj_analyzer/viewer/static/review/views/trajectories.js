// The trajectories worth a look, most anomalous first, each with one line saying why; narrowed by batch, event,
// judged label, how turns ended, feature and review state.

import { filters as loadFilters, post } from "../api.js";
import { chip, el, fmt, notice, select } from "../dom.js";

const SORTS = [
  ["anomaly", "anomaly rank"],
  ["score", "anomaly score"],
  ["duration", "longest"],
  ["turns", "most turns"],
  ["steps", "most steps"],
];
const REVIEWED = [["", "any"], ["true", "reviewed"], ["false", "not reviewed"]];

export async function render(container, { params, navigate, address }) {
  const filters = await loadFilters();
  const request = toRequest(params);
  const open = (changes) => navigate("trajectories", { ...params, page: undefined, ...changes });
  container.append(el("h1", { text: "Trajectories" }), filterBar(filters, params, open));
  const data = await post("review/trajectories", request);
  if (!data.rows.length) {
    container.append(notice("No trajectory matches these filters."));
    return;
  }
  container.append(el("p", { class: "lead", text: `${fmt.number(data.total)} trajectories, page ${data.page} of ${data.pages}. Open one with a click or Enter; move with the arrow keys.` }));
  const dominant = filters.ended_by[0]?.value;
  const list = el("div", { class: "rows", tabindex: 0, role: "list" });
  const rows = data.rows.map((row) => {
    const link = address("trajectory", { step: firstStep(row) }, [row.key]);
    const node = rowNode(row, dominant, link, filters);
    list.append(node);
    return node;
  });
  container.append(list, pager(data, (page) => navigate("trajectories", { ...params, page })));
  keyboard(list, rows);
}

function toRequest(params) {
  const request = { page: Number(params.page ?? 1), size: 50, sort: params.sort || "anomaly", search: params.search ?? "" };
  for (const name of ["batch", "stratum", "rule", "ended_by", "feature"]) {
    if (params[name]) {
      request[name] = params[name];
    }
  }
  if (params.group_by && params.group !== undefined) {
    request.group = { by: params.group_by.split(","), value: params.group };
  }
  if (params.judged) {
    const [feature, ...label] = params.judged.split(":");
    request.judged = { feature, label: label.join(":") };
  }
  if (params.reviewed) {
    request.reviewed = params.reviewed === "true";
  }
  return request;
}

function filterBar(filters, params, open) {
  const any = ["", "any", ""];
  const judged = filters.judged.map((item) => [`${item.feature}:${item.label}`, `${fmt.words(item.feature)}: ${item.label} (${fmt.number(item.count)})${item.event ? " ⚑" : ""}`, item.description]);
  const bar = el("div", { class: "filters" },
    select("Batch", [any, ...filters.batches.map((b) => [b, b, ""])], params.batch, (v) => open({ batch: v })),
    select("Stratum", [any, ...filters.strata.map((s) => [s, s, ""])], params.stratum, (v) => open({ stratum: v })),
    select("Unrecovered event", [any, ...filters.rules.map((r) => [r.rule, `${fmt.words(r.rule)} (${fmt.number(r.unrecovered)})`, r.description])], params.rule, (v) => open({ rule: v }),
      { title: "An expectation broken somewhere in the trajectory that no later action made up for." }),
    select("Judged", [any, ...judged], params.judged, (v) => open({ judged: v }),
      { title: "A label a judgment gave to some message or window of the trajectory; ⚑ marks the labels that count as events." }),
    select("A turn ended by", [any, ...filters.ended_by.map((e) => [e.value, `${e.value} (${fmt.number(e.count)})`, ""])], params.ended_by, (v) => open({ ended_by: v }),
      { title: filters.ended_by_description }),
    select("Feature", [any, ...filters.features.map((f) => [f.name, fmt.words(f.name), f.description])], params.feature, (v) => open({ feature: v }),
      { title: "A feature among the trajectory's anomaly reasons, or the rule of one of its unrecovered events." }),
    select("Reviewed", REVIEWED, params.reviewed, (v) => open({ reviewed: v })),
    select("Order", SORTS.map(([v, t]) => [v, t, ""]), params.sort || "anomaly", (v) => open({ sort: v })),
  );
  const search = el("input", { type: "search", placeholder: "Search the key", value: params.search ?? "", "aria-label": "Search the key" });
  search.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      open({ search: search.value });
    }
  });
  bar.append(el("label", { class: "filter" }, el("span", { class: "filter-name", text: "Key" }), search));
  if (params.group_by && params.group !== undefined) {
    bar.append(el("label", { class: "filter" }, el("span", { class: "filter-name", text: `Group by ${params.group_by.split(",").join(" and ")}` }),
      chip(params.group, { kind: "muted", title: "Remove this group", onclick: () => open({ group_by: undefined, group: undefined }) })));
  }
  if (Object.keys(params).some((name) => name !== "page" && name !== "sort" && params[name])) {
    bar.append(el("button", { class: "ghost", text: "Clear filters", on: { click: () => open(Object.fromEntries(Object.keys(params).map((name) => [name, undefined]))) } }));
  }
  return bar;
}

function firstStep(row) {
  if (row.events.shown.length) {
    return row.events.shown[0].step;
  }
  return undefined;
}

function rowNode(row, dominant, link, filters) {
  const node = el("a", { class: "row", href: link, role: "listitem" });
  node.append(el("div", { class: "row-head" },
    row.anomaly_rank === null ? null : el("span", { class: "rank", text: `#${row.anomaly_rank}`, title: `Anomaly rank ${row.anomaly_rank}, score ${fmt.number(row.anomaly_score)}: outlier ${fmt.number(row.outlier)} plus events ${fmt.number(row.events_score)}.` }),
    el("span", { class: "key", text: fmt.id(row.key), title: row.key }),
    row.batch ? chip(row.batch, { kind: "muted" }) : null,
    row.stratum ? chip(row.stratum, { kind: "muted" }) : null,
    verdictMarks(row.verdicts),
  ));
  node.append(el("div", { class: "row-why" }, why(row)));
  node.append(el("div", { class: "row-tags" },
    row.judged.map((item) => chip(`${fmt.words(item.feature)}: ${item.label}`, { kind: item.event ? "alert" : "", title: `${item.description}\nThe last judged unit, from step #${item.step}.` })),
    Object.entries(row.ended_by).filter(([value]) => value !== dominant).map(([value, count]) => chip(`${count} turn${count === 1 ? "" : "s"} ended by ${value}`, { kind: "warn", title: filters.ended_by_description })),
    Object.entries(row.events.counts).map(([rule, counts]) => chip(fmt.events(rule, counts), { kind: counts.unrecovered ? "event" : "muted" })),
  ));
  node.append(el("div", { class: "row-stats" },
    stat("turns", row.turns), stat("steps", row.steps), stat("duration", fmt.duration(row.duration_s)),
    stat("failed calls", row.failed_calls), stat("asks", row.asks),
  ));
  return node;
}

function why(row) {
  const parts = [];
  const event = row.events.shown[0];
  if (event) {
    parts.push(el("span", { class: "why-event" }, el("b", { text: fmt.words(event.rule) }), event.note ? `: ${event.note}` : "", el("span", { class: "step-ref", text: ` #${event.step}` })));
  }
  const reason = row.reasons[0];
  if (reason) {
    const direction = reason.deviation > 0 ? "above" : "below";
    parts.push(el("span", { class: "why-reason", title: reason.description }, el("b", { text: fmt.words(reason.column) }), ` ${fmt.value(reason.value)}, far ${direction} its neighbours`));
  }
  if (!parts.length) {
    parts.push(el("span", { class: "muted", text: "Nothing sets this trajectory apart." }));
  }
  return parts.flatMap((part, index) => (index ? [el("span", { class: "sep", text: " · " }), part] : [part]));
}

function verdictMarks(verdicts) {
  if (!verdicts.length) {
    return null;
  }
  const worth = verdicts.filter((v) => v.worth).length;
  const text = `${worth ? `✓ ${worth}` : ""}${worth && verdicts.length > worth ? " " : ""}${verdicts.length > worth ? `✗ ${verdicts.length - worth}` : ""}`;
  return chip(text, { kind: "reviewed", title: verdicts.map((v) => `${v.feature}${v.step === null ? "" : ` #${v.step}`}: ${v.worth ? "worth a look" : "not worth it"}${v.note ? ` — ${v.note}` : ""}`).join("\n") });
}

function stat(name, value) {
  return el("span", { class: "stat" }, el("b", { text: typeof value === "number" ? fmt.number(value) : (value ?? "–") }), ` ${name}`);
}

function pager(data, go) {
  return el("div", { class: "pager" },
    el("button", { class: "ghost", text: "← Previous", disabled: data.page <= 1, on: { click: () => go(data.page - 1) } }),
    el("span", { class: "muted", text: `Page ${data.page} of ${data.pages}` }),
    el("button", { class: "ghost", text: "Next →", disabled: data.page >= data.pages, on: { click: () => go(data.page + 1) } }),
  );
}

function keyboard(list, rows) {
  let selected = -1;
  const mark = (index) => {
    rows.forEach((row, i) => row.classList.toggle("selected", i === index));
    selected = index;
    rows[index]?.scrollIntoView({ block: "nearest" });
  };
  list.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      mark(Math.min(rows.length - 1, selected + 1));
    } else if (event.key === "ArrowUp") {
      mark(Math.max(0, selected - 1));
    } else if (event.key === "Enter" && selected >= 0) {
      rows[selected].click();
    } else {
      return;
    }
    event.preventDefault();
  });
}
