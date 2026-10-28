// The hunts: what the last finished run of each found, and the trajectories it read or its marks point to, each
// with its verdict and the steps worth a look.

import { get } from "../api.js";
import { chip, el, fmt, notice, select } from "../dom.js";

const STATUS = [
  ["", "any", ""],
  ["found", "confirmed", "A reader found the trajectory shows the need."],
  ["unread", "marked, unread", "The kept probes mark it and no reader read it."],
  ["rejected", "rejected", "A reader found it does not show the need."],
];
const KINDS = { found: "alert", unread: "event", rejected: "muted" };

export async function render(container, { params, navigate, address }) {
  if (!params.name) {
    await renderList(container, navigate);
    return;
  }
  const data = await get("review/hunt", { name: params.name });
  container.append(el("h1", { text: `Hunt ${data.hunt}` }), el("p", { class: "lead", text: data.need }));
  if (data.run === null) {
    container.append(notice(`No run of this hunt has finished; run \`traj hunt run ${data.hunt}\`.`));
    return;
  }
  container.append(summaryLine(data), facetTable(data), batchTable(data), probeTable(data), checks(data, address));
  container.append(el("h2", { text: "Trajectories" }), trajectoryList(data, params, navigate, address));
}

async function renderList(container, navigate) {
  const hunts = await get("review/hunts");
  container.append(el("h1", { text: "Hunts" }));
  container.append(el("p", { class: "lead", text: "Each hunt searches every trajectory for what its need describes: probes find positions mechanically, readers confirm a sample, and the probes it keeps mark every trajectory. Open a hunt for its findings and the steps worth a look." }));
  if (!hunts.length) {
    container.append(notice("The project has no hunt; a hunt is a file hunts/<name>.yaml."));
    return;
  }
  const table = el("table", { class: "grid" });
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "Hunt" }), el("th", { text: "Need" }), el("th", { class: "num", text: "Run" }),
    el("th", { class: "num", text: "Rounds" }), el("th", { class: "num", text: "Confirmed" }),
    el("th", { class: "num", text: "Read", title: "Trajectories read of all the hunt covers." }),
    el("th", { class: "num", text: "Facets" }), el("th", { class: "num", text: "Probes kept" }),
  )));
  table.append(el("tbody", {}, hunts.map((hunt) => el("tr", { class: "clickable", on: { click: () => navigate("hunts", { name: hunt.hunt }) } },
    el("td", {}, el("b", { text: hunt.hunt })),
    el("td", { class: "muted", text: preview(hunt.need, 140), title: hunt.need }),
    el("td", { class: "num", text: hunt.run === null ? "none finished" : fmt.number(hunt.run) }),
    el("td", { class: "num", text: hunt.run === null ? "–" : `${hunt.rounds}${hunt.converged ? "" : " (stopped)"}` }),
    el("td", { class: "num", text: hunt.run === null ? "–" : fmt.number(hunt.found) }),
    el("td", { class: "num", text: hunt.run === null ? "–" : `${fmt.number(hunt.coverage.read)} of ${fmt.number(hunt.coverage.trajectories)}` }),
    el("td", { class: "num", text: hunt.run === null ? "–" : fmt.number(hunt.facets) }),
    el("td", { class: "num", text: hunt.run === null ? "–" : fmt.number(hunt.probes) }),
  ))));
  container.append(table);
}

function summaryLine(data) {
  const coverage = data.coverage;
  const bound = coverage.miss_share_upper === null ? "no random reads of what the probes miss" :
    `at most ${fmt.percent(coverage.miss_share_upper)} of the ${fmt.number(coverage.unread_unhit)} unread trajectories no probe hits show the need, about ${fmt.number(coverage.expected_missed, 0)}`;
  return el("div", { class: "filters" },
    el("span", { class: "stat" }, el("b", { text: `run ${data.run}` })),
    el("span", { class: "stat" }, el("b", { text: fmt.number(data.rounds) }), ` rounds, ${data.converged ? "ended with no new finding" : "stopped at the most rounds"}`),
    el("span", { class: "stat" }, el("b", { text: fmt.number(data.found) }), " confirmed"),
    el("span", { class: "stat" }, el("b", { text: fmt.number(coverage.read) }), ` of ${fmt.number(coverage.trajectories)} read`),
    el("span", { class: "muted", text: `With 95% confidence, ${bound}.` }),
  );
}

function facetTable(data) {
  const marked = {};
  for (const [feature, facet] of Object.entries(data.marks.features)) {
    marked[facet] = { feature, count: data.marks.marked[feature] };
  }
  const table = el("table", { class: "grid" });
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "Facet" }), el("th", { text: "Edge" }), el("th", { class: "num", text: "Confirmed" }),
    el("th", { class: "num", text: "Marked", title: "Trajectories the kept probes of the facet mark, read or not." }),
    el("th", { text: "Definition" }),
  )));
  table.append(el("tbody", {}, data.facets.map((facet) => el("tr", {},
    el("td", {}, el("b", { text: fmt.words(facet.name) })),
    el("td", { text: facet.edge }),
    el("td", { class: "num", text: fmt.number(facet.found) }),
    el("td", { class: "num", text: marked[facet.name] ? fmt.number(marked[facet.name].count) : "–" }),
    el("td", { class: "muted", text: facet.definition }),
  ))));
  return section("Facets", "The kinds of behaviour the need covers. Each is a feature of the marks, named like the facet, so the trajectory list and the batches can be narrowed by it.", table);
}

function batchTable(data) {
  const table = el("table", { class: "grid" });
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "Batch" }), el("th", { text: "Stratum" }), el("th", { class: "num", text: "Trajectories" }),
    el("th", { class: "num", text: "Read" }), el("th", { class: "num", text: "Confirmed" }),
    el("th", { class: "num", text: "Marked, unread", title: "Trajectories the active probes hit that no reader read." }),
    el("th", { text: "Confirmed by facet" }),
  )));
  table.append(el("tbody", {}, data.batches.map((row) => el("tr", {},
    el("td", {}, el("b", { text: row.batch })), el("td", { text: row.stratum }),
    el("td", { class: "num", text: fmt.number(row.trajectories) }), el("td", { class: "num", text: fmt.number(row.read) }),
    el("td", { class: "num", text: `${fmt.number(row.found)} (${fmt.percent(row.trajectories ? row.found / row.trajectories : null)})` }),
    el("td", { class: "num", text: fmt.number(row.unread_hits) }),
    el("td", {}, Object.entries(row.facets).map(([facet, count]) => chip(`${fmt.words(facet)} ${count}`, { kind: "muted" }))),
  ))));
  return section("Batches", "Per batch and stratum of traj.yaml: how many trajectories were read and confirmed.", table);
}

function probeTable(data) {
  const table = el("table", { class: "grid" });
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "Probe" }), el("th", { text: "Facet" }), el("th", { class: "num", text: "Hits" }),
    el("th", { class: "num", text: "Read" }), el("th", { class: "num", text: "Confirmed" }),
    el("th", { class: "num", text: "Precision" }), el("th", { class: "num", text: "Unread hits" }),
  )));
  table.append(el("tbody", {}, data.probes.map((probe) => el("tr", {},
    el("td", {}, el("b", { text: probe.name })), el("td", { text: fmt.words(probe.facet) }),
    el("td", { class: "num", text: fmt.number(probe.hits) }), el("td", { class: "num", text: fmt.number(probe.read) }),
    el("td", { class: "num", text: fmt.number(probe.found) }), el("td", { class: "num", text: fmt.percent(probe.precision) }),
    el("td", { class: "num", text: fmt.number(data.unverified[probe.name] ?? 0) }),
  ))));
  return section("Probes kept", `The active probes of the last round, which mark every trajectory; ${data.inactive_probes} other probes were retired or failed.`, table);
}

function checks(data, address) {
  const calibration = data.calibration;
  const reviews = data.reviews;
  const link = (key) => el("a", { class: "link", href: address("trajectory", {}, [key]), text: fmt.id(key), title: key });
  return el("div", { class: "filters" },
    el("div", { class: "panel" },
      el("h2", { text: "Blind reads" }),
      calibration.pairs ? el("p", {},
        `${fmt.number(calibration.pairs)} trajectories a probe hit were read again without the hits: `,
        el("b", { text: `${fmt.number(calibration.found_shown)} confirmed with the hits, ${fmt.number(calibration.found_blind)} without` }),
        `; ${fmt.percent(calibration.agreement)} of the pairs agree.`) : el("p", { class: "muted", text: "No trajectory was read blind." }),
      calibration.differ.length ? el("p", {}, "Read differently: ", calibration.differ.map((key) => [link(key), " "])) : null,
    ),
    el("div", { class: "panel" },
      el("h2", { text: "Your verdicts" }),
      reviews.verdicts ? el("p", {}, `${fmt.number(reviews.verdicts)} verdicts on the marks of this hunt; `,
        el("b", { text: `${fmt.number(reviews.agree)} agree with the hunt` }),
        reviews.unread ? `, ${fmt.number(reviews.unread)} on trajectories the hunt did not read` : "", ".") :
        el("p", { class: "muted", text: "No verdict yet: open a trajectory, pick the feature of a facet, and record whether it shows it." }),
      reviews.differ.length ? el("p", {}, "Disagree: ", reviews.differ.map((item) => [link(item.key), " "])) : null,
    ),
  );
}

function trajectoryList(data, params, navigate, address) {
  const open = (changes) => navigate("hunts", { ...params, ...changes });
  const facets = [["", "any", ""], ...data.facets.map((facet) => [facet.name, fmt.words(facet.name), facet.definition])];
  const shown = data.trajectories
    .filter((row) => !params.status || row.status === params.status)
    .filter((row) => !params.facet || row.facet === params.facet || row.marks.some((mark) => mark.facet === params.facet));
  const wrapper = el("div", {},
    el("div", { class: "filters" },
      select("Verdict", STATUS, params.status, (value) => open({ status: value })),
      select("Facet", facets, params.facet, (value) => open({ facet: value })),
      el("span", { class: "muted", text: `${fmt.number(shown.length)} of ${fmt.number(data.trajectories.length)} trajectories` }),
    ),
  );
  if (!shown.length) {
    wrapper.append(notice("No trajectory matches these filters."));
    return wrapper;
  }
  wrapper.append(el("div", { class: "rows" }, shown.map((row) => trajectoryRow(row, address))));
  return wrapper;
}

function trajectoryRow(row, address) {
  const first = row.steps?.length ? Math.min(...row.steps) : row.marks[0]?.step;
  const node = el("a", { class: "row", href: address("trajectory", { step: first }, [row.key]) });
  node.append(el("div", { class: "row-head" },
    el("span", { class: "key", text: fmt.id(row.key), title: row.key }),
    chip(STATUS.find(([value]) => value === row.status)[1], { kind: KINDS[row.status] }),
    row.facet ? chip(fmt.words(row.facet), { kind: "muted" }) : null,
    row.batch ? chip(row.batch, { kind: "muted" }) : null,
    row.stratum ? chip(row.stratum, { kind: "muted" }) : null,
    row.steps?.length ? el("span", { class: "muted" }, " at ", row.steps.map((step) => el("span", { class: "step-ref", text: `#${step} ` }))) : null,
  ));
  if (row.why) {
    node.append(el("div", { class: "row-why", text: row.why }));
  }
  if (row.marks.length) {
    node.append(el("div", { class: "row-tags" }, row.marks.slice(0, 8).map((mark) =>
      chip(`#${mark.step} ${fmt.words(mark.facet)}`, { kind: "event", title: mark.note })),
    row.marks.length > 8 ? el("span", { class: "muted", text: ` +${row.marks.length - 8} marks` }) : null));
  }
  return node;
}

function section(title, lead, body) {
  return el("section", { class: "hunt-section" }, el("h2", { text: title }), el("p", { class: "muted", text: lead }), body);
}

function preview(text, limit) {
  const flat = String(text ?? "").replace(/\s+/g, " ").trim();
  return flat.length > limit ? `${flat.slice(0, limit)}…` : flat;
}
