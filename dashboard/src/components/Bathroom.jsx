import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatDateTime, formatShortDate } from "../api.js";
import { usePalette } from "../theme.js";
import { Card, ChartTooltip, Legend, StatusChip } from "./ui.jsx";

export function Bathroom({ summary, t }) {
  const c = usePalette();
  const rows = summary.days.map((d) => ({ ...d, label: formatShortDate(d.date) }));

  const tooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <ChartTooltip
        title={d.label}
        rows={[
          { label: t.visits, value: d.visits, color: c.series1 },
          { label: t.nightVisits, value: d.night_visits, color: c.series2 },
          { label: t.duration, value: `${Math.round(d.total_minutes)} ${t.min}` },
        ]}
      />
    );
  };

  return (
    <Card title={t.bathroom} className="bathroom" help={t.bathroomHelp.replace("{min}", summary.dwell_threshold_minutes)}>
      <Legend
        items={[
          { label: t.visits, color: c.series1 },
          { label: t.nightVisits, color: c.series2 },
        ]}
      />
      <div className="chart">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -24 }} barGap={2} barCategoryGap="28%">
            <CartesianGrid vertical={false} stroke={c.grid} />
            <XAxis dataKey="label" tick={{ fill: c.muted, fontSize: 11 }} tickLine={false} axisLine={{ stroke: c.axis }} />
            <YAxis tick={{ fill: c.muted, fontSize: 11 }} tickLine={false} axisLine={false} allowDecimals={false} />
            <Tooltip content={tooltip} cursor={{ fill: c.grid, opacity: 0.5 }} />
            <Bar dataKey="visits" fill={c.series1} radius={[4, 4, 0, 0]} maxBarSize={16} isAnimationActive={false} />
            <Bar dataKey="night_visits" fill={c.series2} radius={[4, 4, 0, 0]} maxBarSize={16} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h4>{t.recentVisits}</h4>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>{t.entered}</th>
              <th className="num">{t.duration}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {summary.visits.slice(0, 8).map((v) => (
              <tr key={v.id}>
                <td>{formatDateTime(v.enter)}</td>
                <td className="num">
                  {Math.round(v.minutes)} {t.min}
                  {v.in_progress && ` (${t.inProgress})`}
                </td>
                <td>{v.over_threshold && <StatusChip tone="critical" label={t.tooLong} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
