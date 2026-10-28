import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { getBucketColor } from "../lib/emotions.js";

/**
 * A ring with nothing in the middle reads as decoration rather than data - the
 * center label is what turns it into a number the viewer actually takes away.
 */
function DonutTooltip({ active, payload, total }) {
  if (!active || !payload?.length) return null;
  const { name, value } = payload[0];
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg">
      <div className="flex items-center gap-2 font-semibold capitalize text-slate-900">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: getBucketColor(name) }} />
        {name}
      </div>
      <div className="mt-1 tabular-nums text-slate-600">
        {value.toLocaleString()} ({total ? ((value / total) * 100).toFixed(0) : 0}%)
      </div>
    </div>
  );
}

export default function BucketDonut({ buckets, height = 280 }) {
  const total = buckets.reduce((sum, b) => sum + b.count, 0);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
        <Pie
          data={buckets}
          dataKey="count"
          nameKey="bucket"
          innerRadius="58%"
          outerRadius="82%"
          paddingAngle={2}
          cy="46%"
          stroke="#fff"
          strokeWidth={2}
        >
          {buckets.map((entry) => (
            <Cell key={entry.bucket} fill={getBucketColor(entry.bucket)} />
          ))}
        </Pie>
        {total > 0 && (
          <>
            <text
              x="50%"
              y="43%"
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-slate-900 text-2xl font-bold"
            >
              {total.toLocaleString()}
            </text>
            <text
              x="50%"
              y="53%"
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-slate-400 text-xs font-medium uppercase tracking-wide"
            >
              reviews
            </text>
          </>
        )}
        <Tooltip content={<DonutTooltip total={total} />} />
        <Legend
          verticalAlign="bottom"
          iconType="circle"
          iconSize={8}
          formatter={(v) => <span className="text-xs capitalize text-slate-600">{v}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
