import { useState } from "react";
import { formatDateTime, formatTime } from "../api.js";
import { Card, StatusChip } from "./ui.jsx";

function tone(alert) {
  if (alert.status === "acknowledged") return alert.ack_outcome === "confirmed" ? "serious" : "good";
  if (alert.severity === "urgent") return "critical";
  if (alert.severity === "non_urgent") return "warning";
  return "neutral";
}

export function Alerts({ alerts, t, onAck }) {
  const [busy, setBusy] = useState(null);

  const ack = async (id, outcome) => {
    setBusy(id);
    try {
      await onAck(id, outcome);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Card title={t.alerts} className="alerts">
      {alerts.length === 0 && <p className="muted">{t.noAlerts}</p>}
      <ul className="alert-list">
        {alerts.map((a) => {
          const pending = a.severity === "urgent" && (a.status === "open" || a.status === "escalated");
          const statusLabel =
            a.status === "acknowledged" && a.ack_outcome
              ? `${t.status.acknowledged} · ${t.outcome[a.ack_outcome]}`
              : t.status[a.status] ?? a.status;
          return (
            <li key={a.id} className={`alert ${pending ? "pending" : ""}`}>
              <div className="alert-top">
                <strong>{t.type[a.type] ?? a.type}</strong>
                <time className="muted small" dateTime={a.created_at}>
                  {formatDateTime(a.created_at)}
                </time>
              </div>
              <div className="alert-meta">
                <StatusChip tone={tone(a)} label={statusLabel} />
                <span className="muted small">{t.severity[a.severity] ?? a.severity}</span>
                {a.confidence != null && (
                  <span className="muted small">
                    {t.confidence}: {Math.round(a.confidence * 100)}%
                  </span>
                )}
              </div>
              {a.escalated_at && (
                <p className="muted small">
                  {t.escalatedAt} {formatTime(a.escalated_at)}
                </p>
              )}
              {a.ack_by && (
                <p className="muted small">
                  {t.acknowledgedBy} {a.ack_by} · {formatTime(a.ack_at)}
                </p>
              )}
              {pending && (
                <div className="ack-buttons">
                  <button className="btn primary" disabled={busy === a.id} onClick={() => ack(a.id, "confirmed")}>
                    {t.realFall}
                  </button>
                  <button className="btn" disabled={busy === a.id} onClick={() => ack(a.id, "false_alarm")}>
                    {t.falseAlarm}
                  </button>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </Card>
  );
}
