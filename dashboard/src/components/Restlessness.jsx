import { useState } from "react";
import { Bar, CartesianGrid, Cell, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatShortDate } from "../api.js";
import { usePalette } from "../theme.js";
import { Card, ChartTooltip, Legend } from "./ui.jsx";

export function Restlessness({ nights, t }) {
  const c = usePalette();
  const [table, setTable] = useState(false);
  const rows = nights.map((n) => ({ ...n, label: formatShortDate(n.night) }));

  const tooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const n = payload[0].payload;
    return (
      <ChartTooltip
        title={`${t.night} ${n.label}`}
        rows={[
          { label: t.restlessMinutes, value: n.restless_minutes, color: n.flagged ? c.series2 : c.series1 },
          { label: t.baseline, value: n.baseline_mean ?? "—" },
          { label: t.zScore, value: n.z_score ?? "—" },
        ]}
      />
    );
  };

  return (
    <Card
      title={t.restlessness}
      className="restlessness"
      help={t.restlessnessHelp}
      actions={
        <button className="link" onClick={() => setTable((v) => !v)}>
          {table ? t.hideTable : t.showTable}
        </button>
      }
    >
      <Legend
        items={[
          { label: t.restlessMinutes, color: c.series1 },
          { label: t.flaggedNight, color: c.series2 },
          { label: t.baseline, color: c.baseline, kind: "line" },
        ]}
      />
      <div className="chart">
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -16 }} barCategoryGap="30%">
            <CartesianGrid vertical={false} stroke={c.grid} />
            <XAxis
              dataKey="label"
              tick={{ fill: c.muted, fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: c.axis }}
              interval="preserveStartEnd"
              minTickGap={8}
            />
            <YAxis tick={{ fill: c.muted, fontSize: 11 }} tickLine={false} axisLine={false} allowDecimals={false} />
            <Tooltip content={tooltip} cursor={{ fill: c.grid, opacity: 0.5 }} />
            <Bar dataKey="restless_minutes" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false}>
              {rows.map((r) => (
                <Cell key={r.night} fill={r.flagged ? c.series2 : c.series1} />
              ))}
            </Bar>
            <Line
              dataKey="baseline_mean"
              stroke={c.baseline}
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {table && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t.night}</th>
                <th className="num">{t.restlessMinutes}</th>
                <th className="num">{t.baseline}</th>
                <th className="num">{t.zScore}</th>
              </tr>
            </thead>
            <tbody>
              {[...rows].reverse().map((r) => (
                <tr key={r.night}>
                  <td>
                    {r.label} {r.flagged && <span aria-label={t.flaggedNight}>!</span>}
                  </td>
                  <td className="num">{r.restless_minutes}</td>
                  <td className="num">{r.baseline_mean ?? "—"}</td>
                  <td className="num">{r.z_score ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
