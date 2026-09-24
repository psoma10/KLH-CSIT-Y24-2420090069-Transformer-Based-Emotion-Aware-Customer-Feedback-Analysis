import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { getBucketColor } from "../lib/emotions.js";

/**
 * A ring with nothing in the middle reads as decoration rather than data — the
 * center label is what turns it into a number the viewer actually takes away.
 */
export default function BucketDonut({ buckets, height = 280 }) {
  const total = buckets.reduce((sum, b) => sum + b.count, 0);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={buckets}
          dataKey="count"
          nameKey="bucket"
          innerRadius="55%"
          outerRadius="80%"
          paddingAngle={2}
        >
          {buckets.map((entry) => (
            <Cell key={entry.bucket} fill={getBucketColor(entry.bucket)} />
          ))}
        </Pie>
        {total > 0 && (
          <>
            <text
              x="50%"
              y="48%"
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-slate-900 text-2xl font-bold"
            >
              {total.toLocaleString()}
            </text>
            <text
              x="50%"
              y="60%"
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-slate-400 text-xs font-medium uppercase tracking-wide"
            >
              reviews
            </text>
          </>
        )}
        <Tooltip formatter={(value) => [`${value} (${((value / total) * 100).toFixed(0)}%)`, ""]} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
