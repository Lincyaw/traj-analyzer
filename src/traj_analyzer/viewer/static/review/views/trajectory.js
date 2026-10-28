// One trajectory: its course turn by turn on the left, the steps on the right, why it stands out, and the verdict a
// person leaves about what a feature pointed them to.

import { filters as loadFilters, get, post } from "../api.js";
import { append, bar, chip, clear, el, fmt } from "../dom.js";

const PREVIEW = 160;

export async function render(container, { params, path, navigate, address }) {
  const key = path[0];
  if (!key) {
    navigate("trajectories");
    return;
  }
  const [data, filters] = await Promise.all([get("review/trajectory", { key }), loadFilters()]);
  const view = new TrajectoryView(container, key, data, filters, address);
  view.render(params.step === undefined ? null : Number(params.step));
}

class TrajectoryView {
  constructor(container, key, data, filters, address) {
    this.container = container;
    this.key = key;
    this.data = data;
    this.filters = filters;
    this.address = address;
    this.turns = data.turns;
    this.events = data.turns.flatMap((turn) => turn.events).filter((event) => !event.recovered).sort((a, b) => a.step - b.step);
    this.selected = null;
    this.loaded = null;
    this.verdicts = data.verdicts;
  }

  render(step) {
    const { trajectory, anomaly } = this.data;
    this.container.append(this.header(trajectory, anomaly));
    this.course = el("div", { class: "course" }, this.turns.map((turn) => this.turnNode(turn)));
    this.inspector = el("div", { class: "inspector" });
    this.aside = el("aside", { class: "aside" }, this.whyPanel(), this.huntPanel(), this.verdictPanel());
    this.container.append(el("div", { class: "detail" }, this.course, this.inspector, this.aside));
    const first = step ?? this.events[0]?.step ?? this.turns[0]?.start ?? 0;
    this.select(first);
  }

  header(trajectory, anomaly) {
    const metadata = Object.entries(trajectory.metadata).filter(([, value]) => value !== null && typeof value !== "object");
    return el("header", { class: "detail-head" },
      el("div", { class: "detail-title" },
        el("button", { class: "ghost", text: "← Back", on: { click: () => history.back() } }),
        el("h1", { text: fmt.id(this.key), title: this.key }),
        anomaly ? chip(anomaly.batch, { kind: "muted" }) : null,
        anomaly ? chip(anomaly.stratum, { kind: "muted" }) : null,
        anomaly ? chip(`anomaly #${anomaly.anomaly_rank}`, { kind: "rank", title: `Score ${fmt.number(anomaly.anomaly_score)}: outlier ${fmt.number(anomaly.outlier)} plus events ${fmt.number(anomaly.events_score)}.` }) : null,
        el("span", { class: "stat" }, el("b", { text: fmt.number(trajectory.steps) }), " steps"),
        el("span", { class: "stat" }, el("b", { text: fmt.number(trajectory.turns) }), " turns"),
        this.events.length ? el("span", { class: "event-nav" },
          el("button", { class: "ghost", text: "◂ event", title: "The previous unrecovered event", on: { click: () => this.jump(-1) } }),
          el("span", { class: "muted", text: `${this.events.length} unrecovered events` }),
          el("button", { class: "ghost", text: "event ▸", title: "The next unrecovered event", on: { click: () => this.jump(1) } }),
        ) : el("span", { class: "muted", text: "No unrecovered event" }),
      ),
      metadata.length ? el("details", { class: "metadata" }, el("summary", { text: "Metadata" }),
        el("dl", {}, metadata.map(([name, value]) => [el("dt", { text: name }), el("dd", { text: fmt.value(value) })]))) : null,
    );
  }

  // The course

  turnNode(turn) {
    const node = el("article", { class: "turn", dataset: { start: turn.start, end: turn.end } });
    const title = turn.index === null ? "Before the first message" : `Turn ${turn.index}`;
    const dominant = this.filters.ended_by[0]?.value;
    node.append(el("header", { class: "turn-head", on: { click: () => this.select(turn.start) } },
      el("span", { class: "turn-title", text: title }),
      el("span", { class: "step-ref", text: `#${turn.start}–#${turn.end}` }),
      turn.ended_by && turn.ended_by !== dominant ? chip(`ended by ${turn.ended_by}`, { kind: "warn", title: this.filters.ended_by_description }) : null,
      turn.duration_s ? el("span", { class: "muted", text: fmt.duration(turn.duration_s) }) : null,
      turn.judgments.map((item) => chip(`${fmt.words(item.feature)}: ${fmt.value(item.value)}`, {
        kind: item.event ? "alert" : "", title: `${item.description}${item.evidence ? `\n\n${item.evidence}` : ""}`,
      })),
    ));
    if (turn.prompt !== null) {
      node.append(el("p", { class: "prompt", text: preview(turn.prompt), title: "The message the user typed", on: { click: () => this.select(turn.start) } }));
    }
    const summary = actionSummary(turn.actions);
    if (summary.length) {
      const details = el("details", { class: "actions" }, el("summary", {}, summary.map((text) => el("span", { class: "act", text }))));
      details.append(el("ul", {}, turn.actions.map((action) => el("li", { class: `action ${action.outcome}`, on: { click: () => this.select(action.call) } },
        el("span", { class: "step-ref", text: `#${action.call}` }),
        el("b", { text: action.act }),
        el("span", { class: "muted", text: actionText(action) }),
        action.outcome !== "ok" ? chip(action.error ? `${action.outcome}: ${fmt.words(action.error)}` : action.outcome, { kind: action.outcome === "error" ? "bad" : "muted" }) : null,
      ))));
      node.append(details);
    }
    for (const notice of turn.notices) {
      node.append(el("p", { class: "mark notice", on: { click: () => this.select(notice.step) } }, el("span", { class: "step-ref", text: `#${notice.step}` }), ` ${fmt.words(notice.name)}: `, el("span", { class: "muted", text: preview(notice.content, 80) })));
    }
    for (const event of turn.events) {
      node.append(el("p", { class: `mark event ${event.recovered ? "recovered" : ""}`, title: event.description, on: { click: () => this.select(event.step) } },
        el("span", { class: "step-ref", text: `#${event.step}` }), " ⚑ ", el("b", { text: fmt.words(event.rule) }),
        event.note ? `: ${event.note}` : "",
        event.recovered ? el("span", { class: "muted", text: ` · recovered at #${event.recovered_at}` }) : el("span", { class: "unrecovered", text: " · unrecovered" }),
      ));
    }
    return node;
  }

  // The steps

  async select(step) {
    this.selected = step;
    const turn = this.turns.find((candidate) => candidate.start <= step && step <= candidate.end) ?? this.turns.at(-1);
    for (const node of this.course.querySelectorAll(".turn")) {
      const current = Number(node.dataset.start) === turn.start;
      node.classList.toggle("current", current);
      if (current) {
        node.scrollIntoView({ block: "nearest" });
      }
    }
    for (const node of this.course.querySelectorAll(".mark, .action")) {
      node.classList.toggle("selected", node.textContent.startsWith(`#${step}`) || node.querySelector(".step-ref")?.textContent === `#${step}`);
    }
    if (this.loaded?.start !== turn.start) {
      const steps = await get("review/steps", { key: this.key, start: turn.start, end: turn.end });
      this.loaded = { start: turn.start, steps };
      append(clear(this.inspector), [
        el("div", { class: "inspector-head", text: turn.index === null ? "Steps before the first message" : `Steps of turn ${turn.index}` }),
        steps.map((item) => this.stepNode(item)),
      ]);
    }
    for (const node of this.inspector.querySelectorAll(".step")) {
      const current = Number(node.dataset.step) === step;
      node.classList.toggle("selected", current);
      if (current) {
        node.scrollIntoView({ block: "start" });
      }
    }
    if (this.stepField) {
      this.stepField.value = String(step);
    }
  }

  stepNode(item) {
    return el("article", { class: `step ${item.role}`, dataset: { step: item.step } },
      el("header", {},
        el("span", { class: "step-ref", text: `#${item.step}` }),
        el("b", { text: item.role }), ` · ${item.kind}`, item.name ? ` · ${item.name}` : "",
        item.is_error ? chip("error", { kind: "bad" }) : null,
        item.timestamp ? el("span", { class: "muted", text: ` ${item.timestamp}` }) : null,
      ),
      el("pre", { text: item.content }),
    );
  }

  jump(direction) {
    if (!this.events.length) {
      return;
    }
    const index = this.events.findIndex((event) => event.step === this.selected);
    const next = index < 0 ? (direction > 0 ? 0 : this.events.length - 1) : (index + direction + this.events.length) % this.events.length;
    this.select(this.events[next].step);
  }

  // Why it stands out

  whyPanel() {
    const panel = el("section", { class: "panel" }, el("h2", { text: "Why it stands out" }));
    const features = this.data.features;
    const reasons = features.filter((feature) => feature.reason !== null);
    const largest = Math.max(1, ...reasons.map((feature) => Math.abs(feature.reason.deviation)));
    if (!reasons.length) {
      panel.append(el("p", { class: "muted", text: "No feature puts this trajectory far from its neighbours." }));
    }
    for (const feature of reasons) {
      panel.append(this.featureLine(feature, largest));
    }
    const counts = {};
    for (const event of this.turns.flatMap((turn) => turn.events)) {
      counts[event.rule] ??= { total: 0, unrecovered: 0 };
      counts[event.rule].total += 1;
      counts[event.rule].unrecovered += event.recovered ? 0 : 1;
    }
    if (Object.keys(counts).length) {
      panel.append(el("h3", { text: "Events" }), el("div", { class: "tags" }, Object.entries(counts).map(([rule, count]) =>
        chip(fmt.events(rule, count), {
          kind: count.unrecovered ? "event" : "muted",
          onclick: () => {
            const first = this.events.find((event) => event.rule === rule);
            if (first) {
              this.select(first.step);
            }
          },
        }))));
    }
    const others = features.filter((feature) => feature.reason === null);
    if (others.length) {
      panel.append(el("details", { class: "all-features" }, el("summary", { text: `All ${features.length} features` }),
        others.map((feature) => this.featureLine(feature, null))));
    }
    return panel;
  }

  featureLine(feature, largest) {
    const typical = feature.typical === null || feature.typical === undefined ? "" : ` · typical ${fmt.value(feature.typical)}`;
    const line = el("div", { class: "feature-line", title: `${feature.description}\n\n${feature.kind} feature of ${feature.operator}` },
      el("span", { class: "feature-name", text: fmt.words(feature.name) }),
      el("span", { class: "feature-value" }, el("b", { text: featureValue(feature) }), typical),
    );
    if (feature.reason && largest) {
      line.append(bar(feature.reason.deviation / largest, feature.reason.deviation > 0 ? "up" : "down"));
    }
    if (feature.evidence.length) {
      line.append(el("span", { class: "evidence" }, feature.evidence.slice(0, 8).map((item) =>
        el("button", { class: "step-ref link", text: `#${item.step}`, title: item.note || "A step the value rests on", on: { click: () => this.select(item.step) } }))));
    }
    line.addEventListener("click", () => { this.featureField.value = feature.name; });
    return line;
  }

  // What the hunts read in it

  huntPanel() {
    if (!this.data.hunts.length) {
      return null;
    }
    return el("section", { class: "panel" }, el("h2", { text: "Hunts" }), this.data.hunts.map((verdict) => el("div", { class: "feature-line" },
      el("span", { class: "feature-name" }, el("a", { class: "link", href: this.address("hunts", { name: verdict.hunt }), text: verdict.hunt })),
      chip(verdict.found ? `confirmed: ${fmt.words(verdict.facet)}` : "rejected", { kind: verdict.found ? "alert" : "muted", title: `Read in round ${verdict.round}, drawn as ${verdict.source}.` }),
      el("span", { class: "evidence" }, (verdict.steps ?? []).map((step) =>
        el("button", { class: "step-ref link", text: `#${step}`, title: "A step the verdict cites", on: { click: () => this.select(step) } }))),
      el("p", { class: "muted", text: verdict.why }),
    )));
  }

  // Verdicts

  verdictPanel() {
    const names = [...new Set([
      ...this.data.features.filter((f) => f.reason !== null).map((f) => f.name),
      ...this.events.map((event) => event.rule),
      ...this.data.features.map((f) => f.name),
    ])];
    this.featureField = el("select", { "aria-label": "Feature" }, names.map((name) => el("option", { value: name, text: fmt.words(name) })));
    this.stepField = el("input", { type: "number", min: 0, placeholder: "step", "aria-label": "Step" });
    const note = el("input", { type: "text", placeholder: "What you saw", "aria-label": "Note" });
    const result = el("p", { class: "muted" });
    const submit = async (worth) => {
      const body = { key: this.key, feature: this.featureField.value, worth, note: note.value, step: this.stepField.value === "" ? null : Number(this.stepField.value) };
      const answer = await post("review/verdict", body);
      this.verdicts.push(answer.verdict);
      result.textContent = `Recorded. ${fmt.words(answer.feature ?? body.feature)}: ${fmt.percent(answer.precision)} of ${answer.verdicts} verdicts found the trajectory worth a look.`;
      note.value = "";
      this.renderVerdicts(list);
    };
    const list = el("ul", { class: "verdicts" });
    this.renderVerdicts(list);
    return el("section", { class: "panel verdict" },
      el("h2", { text: "Your verdict" }),
      el("p", { class: "muted", text: "Was this trajectory worth your look because of what the feature found at the step? Verdicts go to reviews/verdicts.jsonl and measure the feature's precision." }),
      el("div", { class: "verdict-form" }, this.featureField, this.stepField, note),
      el("div", { class: "verdict-buttons" },
        el("button", { class: "primary", text: "Worth a look", on: { click: () => submit(true) } }),
        el("button", { class: "ghost", text: "Not worth it", on: { click: () => submit(false) } }),
      ),
      result, list,
    );
  }

  renderVerdicts(list) {
    append(clear(list), this.verdicts.map((verdict) => el("li", { class: verdict.worth ? "worth" : "not" },
      `${verdict.worth ? "✓" : "✗"} ${fmt.words(verdict.feature)}`,
      verdict.step === null ? "" : el("button", { class: "step-ref link", text: ` #${verdict.step}`, on: { click: () => this.select(verdict.step) } }),
      verdict.note ? el("span", { class: "muted", text: ` — ${verdict.note}` }) : null,
    )));
  }
}

// An event feature holds its events; its value reads as their count.
function featureValue(feature) {
  if (feature.type === "event" && Array.isArray(feature.value)) {
    const unrecovered = feature.value.filter((event) => !event.recovered).length;
    return `${feature.value.length} event${feature.value.length === 1 ? "" : "s"}, ${unrecovered} unrecovered`;
  }
  return fmt.value(feature.value);
}

function preview(text, limit = PREVIEW) {
  const flat = String(text ?? "").replace(/\s+/g, " ").trim();
  return flat.length > limit ? `${flat.slice(0, limit)}…` : flat;
}

function actionSummary(actions) {
  const counts = {};
  for (const action of actions) {
    counts[action.act] ??= { total: 0, failed: 0 };
    counts[action.act].total += 1;
    counts[action.act].failed += action.outcome === "error" ? 1 : 0;
  }
  return Object.entries(counts).map(([act, count]) => `${act} ×${count.total}${count.failed ? ` (${count.failed} failed)` : ""}`);
}

function actionText(action) {
  if (action.text) {
    return action.text;
  }
  const objects = (action.objects ?? []).map((object) => object.name).filter(Boolean);
  return objects.length ? objects.join(", ") : action.tool;
}
