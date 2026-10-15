// Small helpers to build the pages: elements, formatting and the words shown for the names of the tables.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) {
      continue;
    }
    if (name === "class") {
      node.className = value;
    } else if (name === "text") {
      node.textContent = value;
    } else if (name === "on") {
      for (const [event, handler] of Object.entries(value)) {
        node.addEventListener(event, handler);
      }
    } else if (name === "dataset") {
      Object.assign(node.dataset, value);
    } else if (value === true) {
      node.setAttribute(name, "");
    } else {
      node.setAttribute(name, value);
    }
  }
  append(node, children);
  return node;
}

export function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) {
      continue;
    }
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  node.replaceChildren();
  return node;
}

export const fmt = {
  // "ask_share" reads "ask share": the names of the tables, as words.
  words(name) {
    return String(name).replace(/_/g, " ");
  },
  // "ask_share" reads "Ask share": a column title.
  title(name) {
    const words = fmt.words(name);
    return words.charAt(0).toUpperCase() + words.slice(1);
  },
  // A column of the wide table reads as its feature and, after a colon, the label or part it holds.
  column(column, feature) {
    if (feature && column !== feature && column.startsWith(`${feature}__`)) {
      return `${fmt.words(feature)}: ${fmt.words(column.slice(feature.length + 2))}`;
    }
    return fmt.words(column);
  },
  number(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "–";
    }
    if (Number.isInteger(value)) {
      return value.toLocaleString("en-US");
    }
    return value.toLocaleString("en-US", { maximumFractionDigits: digits });
  },
  percent(value) {
    return value === null || value === undefined ? "–" : `${(value * 100).toFixed(1)}%`;
  },
  duration(seconds) {
    if (seconds === null || seconds === undefined) {
      return "–";
    }
    if (seconds < 60) {
      return `${Math.round(seconds)} s`;
    }
    if (seconds < 3600) {
      return `${(seconds / 60).toFixed(1)} min`;
    }
    return `${(seconds / 3600).toFixed(1)} h`;
  },
  // A metric of the oversight table, formatted by what its name says it holds.
  metric(name, value) {
    if (/_share$/.test(name)) {
      return fmt.percent(value);
    }
    if (/duration|_s$/.test(name)) {
      return fmt.duration(value);
    }
    return fmt.number(value);
  },
  // A feature value of any type, as a short text.
  value(value) {
    if (value === null || value === undefined) {
      return "–";
    }
    if (typeof value === "boolean") {
      return value ? "yes" : "no";
    }
    if (typeof value === "number") {
      return fmt.number(value, 3);
    }
    if (Array.isArray(value)) {
      return value.length ? value.map((item) => fmt.value(item)).join(", ") : "none";
    }
    if (typeof value === "object") {
      const entries = Object.entries(value);
      return entries.length ? entries.map(([k, v]) => `${k} ${fmt.value(v)}`).join(", ") : "none";
    }
    return String(value);
  },
  id(key) {
    return String(key).split("/").pop();
  },
  // "blind retry ×3 (+2 recovered)", or "blind retry: 2 recovered" when none is left unrecovered.
  events(rule, counts) {
    const recovered = counts.total - counts.unrecovered;
    if (!counts.unrecovered) {
      return `${fmt.words(rule)}: ${recovered} recovered`;
    }
    return `${fmt.words(rule)} ×${counts.unrecovered}${recovered ? ` (+${recovered} recovered)` : ""}`;
  },
  signed(value, digits = 2) {
    return value === null || value === undefined ? "–" : `${value > 0 ? "+" : ""}${fmt.number(value, digits)}`;
  },
};

export function chip(text, { kind = "", title = "", onclick = null } = {}) {
  const node = el("span", { class: `chip ${kind}`.trim(), text, title: title || undefined });
  if (onclick) {
    node.classList.add("link");
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      onclick(event);
    });
  }
  return node;
}

export function bar(ratio, kind = "") {
  const width = `${Math.max(0, Math.min(1, Math.abs(ratio))) * 100}%`;
  return el("span", { class: `meter ${kind}`.trim() }, el("span", { class: "fill", style: `width: ${width}` }));
}

export function notice(text) {
  return el("div", { class: "notice", text });
}

export function option(value, text, { title = "", selected = false } = {}) {
  return el("option", { value, text, title: title || undefined, selected });
}

// A select whose change navigates: the options are given as [value, text, title].
export function select(label, options, current, onchange, { title = "" } = {}) {
  const node = el("select", { "aria-label": label, title: title || undefined, on: { change: () => onchange(node.value) } });
  for (const [value, text, hint] of options) {
    node.append(option(value, text, { title: hint, selected: value === (current ?? "") }));
  }
  return el("label", { class: "filter" }, el("span", { class: "filter-name", text: label }), node);
}
