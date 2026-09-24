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

export default function TrendsChart({ trends, height = 320 }) {
  const data = pivotByDate(trends);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
        <Tooltip />
        <Legend />
        {BUCKET_NAMES.map((bucket) => (
          <Area
            key={bucket}
            type="monotone"
            dataKey={bucket}
            stackId="1"
            stroke={getBucketColor(bucket)}
            fill={getBucketColor(bucket)}
            fillOpacity={0.6}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
