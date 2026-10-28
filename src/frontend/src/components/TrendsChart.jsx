import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { BUCKET_NAMES, getBucketColor } from "../lib/emotions.js";

/** Pivots the flat {date, bucket, count} rows from /api/analytics/trends into one object per date. */
function pivotByDate(rows) {
  const byDate = new Map();
  for (const { date, bucket, count } of rows) {
    if (!byDate.has(date)) byDate.set(date, { date });
    byDate.get(date)[bucket] = count;
  }
  return Array.from(byDate.values()).sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
}

const TICK = { fontSize: 12, fill: "#64748b" };
const capitalize = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);

export default function TrendsChart({ trends, height = 320 }) {
  const data = pivotByDate(trends);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
        <XAxis dataKey="date" tick={TICK} tickLine={false} axisLine={{ stroke: "#cbd5e1" }} minTickGap={24} />
        <YAxis allowDecimals={false} tick={TICK} tickLine={false} axisLine={false} width={44} />
        <Tooltip
          contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12 }}
          formatter={(value, name) => [value, capitalize(name)]}
        />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, paddingTop: 8 }} formatter={capitalize} />
        {BUCKET_NAMES.map((bucket) => (
          <Area
            key={bucket}
            type="monotone"
            dataKey={bucket}
            stackId="1"
            stroke={getBucketColor(bucket)}
            fill={getBucketColor(bucket)}
            fillOpacity={0.55}
            strokeWidth={1.5}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
