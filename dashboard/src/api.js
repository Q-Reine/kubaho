export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function request(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  households: () => request("/households"),
  household: (id) => request(`/households/${id}`),
  alerts: (id) => request(`/households/${id}/alerts?limit=30`),
  restlessness: (id) => request(`/households/${id}/restlessness?nights=14`),
  bathroom: (id) => request(`/households/${id}/bathroom?days=7`),
  sms: (id) => request(`/households/${id}/sms?limit=20`),
  ack: (alertId, outcome) =>
    request(`/alerts/${alertId}/ack`, { method: "POST", body: JSON.stringify({ outcome, by: "dashboard" }) }),
};

// The API stores naive UTC timestamps; show them in Rwanda time (UTC+2).
export function parseUtc(s) {
  if (!s) return null;
  return new Date(/[zZ]|[+-]\d\d:\d\d$/.test(s) ? s : `${s}Z`);
}

const fmt = (opts) => new Intl.DateTimeFormat("en-GB", { timeZone: "Africa/Kigali", ...opts });
const dateTimeFmt = fmt({ day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });
const timeFmt = fmt({ hour: "2-digit", minute: "2-digit", hour12: false });

export const formatDateTime = (s) => (s ? dateTimeFmt.format(parseUtc(s)) : "—");
export const formatTime = (s) => (s ? timeFmt.format(parseUtc(s)) : "—");

export function formatShortDate(isoDate) {
  // "2026-11-09" (a calendar date, no time zone) -> "9 Nov"
  const [y, m, d] = isoDate.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
}
