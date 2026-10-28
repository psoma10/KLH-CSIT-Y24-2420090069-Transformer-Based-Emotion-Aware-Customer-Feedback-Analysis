import { memo } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from "recharts";
import { getBucketColor, labelToBucket } from "../lib/emotions.js";

/**
 * Horizontal bar chart, sorted descending, colored by business bucket.
 * `scores` maps label -> numeric value; pass `domain: [0, 1]` for probability
 * scores (Analyze page) or omit it to let Recharts auto-scale raw counts
 * (Dashboard distribution chart).
 *
 * `labelInterval` controls Y-axis tick density. Recharts' default ("preserveEnd")
 * silently drops every other category label when the chart is short relative to
 * the number of bars - pass 0 to force all 28 labels to render.
 */
const AXIS = { fontSize: 12, fill: "#64748b" };

function ChartTooltip({ active, payload, format }) {
  if (!active || !payload?.length) return null;
  const { label, value } = payload[0].payload;
  const bucket = labelToBucket(label);
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg">
      <div className="flex items-center gap-2 font-semibold capitalize text-slate-900">
        <span
          className="h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: getBucketColor(label) }}
        />
        {label}
      </div>
      <div className="mt-1 tabular-nums text-slate-600">{format(value)}</div>
      {bucket && <div className="mt-0.5 capitalize text-slate-400">{bucket}</div>}
    </div>
  );
}

// Memoized: re-laying out 28 recharts bars on every keystroke is the
// single most expensive render on the Analyze page.
function EmotionBarChart({
  scores,
  height = 560,
  domain,
  valueFormatter,
  labelInterval,
  yAxisWidth = 100,
}) {
  const data = Object.entries(scores)
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);
  const format = valueFormatter ?? ((v) => v.toFixed(4));
  const isPercent = domain && domain[1] === 1;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          domain={domain ?? [0, "auto"]}
          tick={AXIS}
          tickLine={false}
          axisLine={{ stroke: "#cbd5e1" }}
          tickFormatter={isPercent ? (v) => `${Math.round(v * 100)}%` : undefined}
        />
        <YAxis
          type="category"
          dataKey="label"
          width={yAxisWidth}
          tick={AXIS}
          tickLine={false}
          axisLine={false}
          interval={labelInterval}
          tickFormatter={(l) => l.charAt(0).toUpperCase() + l.slice(1)}
        />
        <Tooltip cursor={{ fill: "#f1f5f9" }} content={<ChartTooltip format={format} />} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18}>
          {data.map((entry) => (
            <Cell key={entry.label} fill={getBucketColor(entry.label)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default memo(EmotionBarChart);
