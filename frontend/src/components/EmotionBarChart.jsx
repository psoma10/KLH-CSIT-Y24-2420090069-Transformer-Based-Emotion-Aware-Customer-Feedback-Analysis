import { memo } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from "recharts";
import { getBucketColor } from "../lib/emotions.js";

/**
 * Horizontal bar chart, sorted descending, colored by business bucket.
 * `scores` maps label -> numeric value; pass `domain: [0, 1]` for probability
 * scores (Analyze page) or omit it to let Recharts auto-scale raw counts
 * (Dashboard distribution chart).
 *
 * `labelInterval` controls Y-axis tick density. Recharts' default ("preserveEnd")
 * silently drops every other category label when the chart is short relative to
 * the number of bars — pass 0 to force all 28 labels to render.
 */
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

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 16 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" domain={domain ?? [0, "auto"]} />
        <YAxis
          type="category"
          dataKey="label"
          width={yAxisWidth}
          tick={{ fontSize: 12 }}
          interval={labelInterval}
        />
        <Tooltip formatter={(value) => format(value)} labelFormatter={(label) => label} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((entry) => (
            <Cell key={entry.label} fill={getBucketColor(entry.label)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default memo(EmotionBarChart);
