import { useCallback, useEffect, useState } from "react";
import { api, formatTime } from "./api.js";
import { STRINGS } from "./i18n.js";
import { Alerts } from "./components/Alerts.jsx";
import { Restlessness } from "./components/Restlessness.jsx";
import { Bathroom } from "./components/Bathroom.jsx";
import { SmsLog } from "./components/SmsLog.jsx";
import { StatusChip } from "./components/ui.jsx";

const REFRESH_MS = 10_000;

function load(key, fallback) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}
function save(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: keep in memory only */
  }
}

export default function App() {
  const [lang, setLang] = useState(() => load("kubaho.lang", "rw"));
  const t = STRINGS[lang];
  const [households, setHouseholds] = useState([]);
  const [selected, setSelected] = useState(() => load("kubaho.household", null));
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [updatedAt, setUpdatedAt] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const list = await api.households();
      setHouseholds(list);
      const id = selected && list.some((h) => h.id === selected) ? selected : list[0]?.id;
      if (id && id !== selected) setSelected(id);
      if (id) {
        const [household, alerts, restlessness, bathroom, sms] = await Promise.all([
          api.household(id),
          api.alerts(id),
          api.restlessness(id),
          api.bathroom(id),
          api.sms(id),
        ]);
        setData({ id, household, alerts, restlessness, bathroom, sms });
      }
      setError(false);
      setUpdatedAt(new Date().toISOString());
    } catch {
      setError(true);
    }
  }, [selected]);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, REFRESH_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const choose = (id) => {
    setSelected(id);
    save("kubaho.household", id);
    setData((d) => (d && d.id === id ? d : null));
  };
  const toggleLang = (l) => {
    setLang(l);
    save("kubaho.lang", l);
  };

  const onAck = async (alertId, outcome) => {
    await api.ack(alertId, outcome);
    refresh();
  };

  const hh = data?.household;
  const elder = hh?.people.find((p) => p.role === "elder");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/icon.svg" alt="" width="32" height="32" />
          <div>
            <h1>Kubaho</h1>
            <p className="muted small">{t.tagline}</p>
          </div>
        </div>
        <div className="lang" role="group" aria-label="Language / Ururimi">
          {["rw", "en"].map((l) => (
            <button key={l} className={lang === l ? "active" : ""} onClick={() => toggleLang(l)} aria-pressed={lang === l}>
              {l === "rw" ? "Kinyarwanda" : "English"}
            </button>
          ))}
        </div>
      </header>

      {error && (
        <div className="banner" role="alert">
          <span aria-hidden="true">!</span> {t.offlineError}
        </div>
      )}

      <nav className="households" aria-label={t.households}>
        {households.map((h) => (
          <button
            key={h.id}
            className={`hh ${h.id === data?.id || h.id === selected ? "selected" : ""}`}
            onClick={() => choose(h.id)}
            aria-current={h.id === selected}
          >
            <span className="hh-name">{h.elder ?? h.name}</span>
            <span className="muted small">
              {h.id} · {h.village}
            </span>
            <span className="hh-meta small">
              {h.open_alerts > 0 ? (
                <span className="badge critical">
                  <span aria-hidden="true">!</span> {h.open_alerts} {t.openAlerts}
                </span>
              ) : (
                <span className="badge good">
                  <span aria-hidden="true">✓</span> 0 {t.openAlerts}
                </span>
              )}
              <span className="muted">
                {h.nodes_online}/{h.nodes_total} {t.sensorsOnline}
              </span>
            </span>
          </button>
        ))}
      </nav>

      {!data && <p className="muted pad">{households.length ? t.loading : error ? "" : t.loading}</p>}

      {data && (
        <main>
          <section className="card household-head">
            <div>
              <h2>{elder?.name ?? hh.name}</h2>
              <p className="muted">
                {hh.id} · {t.village}: {hh.village} · {hh.sector}
              </p>
            </div>
            <div className="chips">
              {hh.nodes.map((n) => (
                <StatusChip
                  key={n.id}
                  tone={n.status === "online" ? "good" : n.status === "offline" ? "critical" : "neutral"}
                  label={`${t.zone[n.zone] ?? n.zone}: ${t.node[n.status] ?? n.status}`}
                  title={`${n.id} · ${n.adapter_version}`}
                />
              ))}
            </div>
          </section>

          {/* Two columns on desktop; on phones the columns dissolve and CSS order puts SMS last. */}
          <div className="grid">
            <div className="col">
              <Alerts alerts={data.alerts} t={t} onAck={onAck} />
              <SmsLog messages={data.sms} t={t} />
            </div>
            <div className="col">
              <Restlessness nights={data.restlessness} t={t} />
              <Bathroom summary={data.bathroom} t={t} />
            </div>
          </div>
        </main>
      )}

      <footer className="muted small pad">
        {updatedAt && `${t.updated} ${formatTime(updatedAt)} · `}Kubaho research prototype — not a certified medical device.
      </footer>
    </div>
  );
}
