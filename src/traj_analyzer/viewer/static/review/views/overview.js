// The trajectories in groups of the person's choosing: how much the users had to step in and how long the turns
// took in each group, how many of its trajectories stand out, and the features where it differs from the others.

import { get } from "../api.js";
import { bar, el, fmt, notice, option } from "../dom.js";

export async function render(container, { params, navigate }) {
  const groupings = await get("review/groupings");
  const by = params.by ? params.by.split(",") : null;
  const data = await get("review/overview", { by: by ?? undefined });
  container.append(el("h1", { text: "How each group is doing" }));
  container.append(groupPicker(groupings, data, (names) => navigate("overview", { by: names.join(",") })));
  const chosen = data.by.map((name) => groupings.find((g) => g.name === name)?.label ?? name).join(" and ");
  const strata = data.strata.length ? ` among trajectories alike in ${data.strata.join(" and ")}` : "";
  container.append(el("p", { class: "lead" },
    `One row per value of ${chosen}. The columns say how much the users had to step in and how long the turns took; hover a title for the definition. `,
    `Under a group with at least ${data.min_batch} trajectories, the features where it differs most from the other groups${strata}, `,
    "each as the group's average against the others' average, ordered by the size of the difference in standard deviations. ",
    "Click a group to see its trajectories, most unusual first; click a feature to see the trajectories of the group where that feature stands out.",
  ));
  if (!data.rows.length) {
    container.append(notice("No trajectory has a value for this grouping."));
    return;
  }
  const largest = Math.max(1, ...data.rows.flatMap((row) => row.differences.map((d) => Math.abs(d.effect))));
  const table = el("table", { class: "grid batches" });
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: chosen }),
    el("th", { class: "num", text: "Trajectories" }),
    data.metrics.map((metric) => el("th", { class: "num", text: fmt.title(metric.name), title: metric.description })),
    el("th", { class: "num", text: "Unusual", title: "Trajectories of the group among the tenth of the project with the highest anomaly scores: far from their neighbours, or with events no later action made up for. Empty before traj anomalies has run." }),
  )));
  const body = el("tbody");
  for (const row of data.rows) {
    const open = (extra = {}) => navigate("trajectories", { group_by: data.by.join(","), group: row.group, ...extra });
    body.append(el("tr", { class: "clickable", title: `Open the ${fmt.number(row.trajectories)} trajectories of ${row.group}`, on: { click: () => open() } },
      el("td", {}, el("b", { text: row.group })),
      el("td", { class: "num", text: fmt.number(row.trajectories) }),
      data.metrics.map((metric) => el("td", { class: "num", text: fmt.metric(metric.name, row.metrics[metric.name]) })),
      el("td", { class: "num" }, row.unusual === null ? "–" : [fmt.number(row.unusual), el("span", { class: "muted", text: row.trajectories ? ` (${fmt.percent(row.unusual / row.trajectories)})` : "" })]),
    ));
    if (row.differences.length) {
      body.append(el("tr", { class: "shifts" }, el("td", { colspan: data.metrics.length + 3 },
        el("div", { class: "shift-list" },
          el("span", { class: "muted", text: `${row.group} against the others:` }),
          row.differences.map((difference) => differenceItem(difference, largest, () => open({ feature: difference.feature }))),
        ),
      )));
    }
  }
  table.append(body);
  container.append(table);
}

function groupPicker(groupings, data, onchange) {
  const current = data.by.join(",");
  const configured = data.configured.join(",");
  const select = el("select", { "aria-label": "Group by", on: { change: () => onchange(select.value.split(",")) } });
  if (configured && !groupings.some((g) => g.name === configured)) {
    select.append(option(configured, `${data.configured.join(" and ")} (batches of traj.yaml)`, { selected: current === configured }));
  }
  for (const kind of ["metadata", "feature"]) {
    const group = el("optgroup", { label: kind === "metadata" ? "Metadata" : "Features" });
    for (const grouping of groupings.filter((g) => g.kind === kind)) {
      group.append(option(grouping.name, grouping.label, { title: grouping.description, selected: grouping.name === current }));
    }
    select.append(group);
  }
  return el("div", { class: "filters" }, el("label", { class: "filter" }, el("span", { class: "filter-name", text: "Group by" }), select));
}

function differenceItem(difference, largest, open) {
  const value = (number) => (difference.unit === "share" ? fmt.percent(number) : fmt.number(number, 2));
  const title = `${difference.description}\n\nAverage ${value(difference.own)} over ${fmt.number(difference.n)} trajectories of the group, `
    + `${value(difference.others)} over ${fmt.number(difference.n_others)} of the others; ${fmt.signed(difference.effect)} standard deviations.`;
  const kind = difference.effect > 0 ? "up" : "down";
  return el("button", { class: "shift", title, on: { click: open } },
    el("span", { class: "shift-name", text: fmt.column(difference.column, difference.feature) }),
    el("span", { class: `versus ${kind}` }, el("b", { text: value(difference.own) }), el("span", { class: "muted", text: " vs " }), value(difference.others)),
    bar(Math.abs(difference.effect) / largest, kind),
  );
}
