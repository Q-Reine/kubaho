const ICONS = { good: "✓", warning: "!", serious: "!", critical: "!", neutral: "•" };

// Status is always icon + text; colour only reinforces it.
export function StatusChip({ tone = "neutral", label, title }) {
  return (
    <span className={`chip ${tone}`} title={title}>
      <span className="chip-icon" aria-hidden="true">
        {ICONS[tone]}
      </span>
      {label}
    </span>
  );
}

export function Card({ title, help, children, actions, className = "" }) {
  return (
    <section className={`card ${className}`}>
      <div className="card-head">
        <h3>{title}</h3>
        {actions}
      </div>
      {help && <p className="muted small help">{help}</p>}
      {children}
    </section>
  );
}

export function Legend({ items }) {
  return (
    <ul className="legend">
      {items.map((it) => (
        <li key={it.label}>
          <span
            className={`swatch ${it.kind ?? "box"}`}
            style={it.kind === "line" ? { borderColor: it.color } : { background: it.color }}
            aria-hidden="true"
          />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

export function ChartTooltip({ title, rows }) {
  return (
    <div className="tooltip">
      <div className="tooltip-title">{title}</div>
      {rows.map((r) => (
        <div key={r.label} className="tooltip-row">
          {r.color && <span className="swatch box" style={{ background: r.color }} aria-hidden="true" />}
          <span className="tooltip-label">{r.label}</span>
          <span className="tooltip-value">{r.value}</span>
        </div>
      ))}
    </div>
  );
}
