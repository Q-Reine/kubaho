import { formatDateTime } from "../api.js";
import { Card } from "./ui.jsx";

export function SmsLog({ messages, t }) {
  return (
    <Card title={t.sms} className="sms">
      {messages.length === 0 && <p className="muted">{t.noSms}</p>}
      <ul className="sms-list">
        {messages.map((m) => (
          <li key={m.id} className={m.direction === "in" ? "in" : "out"}>
            <div className="sms-meta muted small">
              <span>
                {m.direction === "in" ? t.from : t.to} {m.phone}
                {m.recipient_role && ` · ${t.role[m.recipient_role] ?? m.recipient_role}`}
              </span>
              <span>
                {formatDateTime(m.created_at)} · {m.status}
              </span>
            </div>
            <p>{m.message}</p>
          </li>
        ))}
      </ul>
    </Card>
  );
}
