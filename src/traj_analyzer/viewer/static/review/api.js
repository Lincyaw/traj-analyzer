// Calls to the review endpoints; every failure becomes an Error carrying the server's detail.

async function request(path, init) {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // The body was not JSON; the status line is the detail.
    }
    throw new Error(detail);
  }
  return response.json();
}

function query(params) {
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== "");
  const text = new URLSearchParams(entries).toString();
  return text ? `?${text}` : "";
}

export function get(path, params = {}) {
  return request(`api/${path}${query(params)}`);
}

export function post(path, body) {
  return request(`api/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

let filtersPromise = null;

// The values the trajectory list can be narrowed by, fetched once per page load.
export function filters() {
  filtersPromise ??= get("review/filters");
  return filtersPromise;
}
