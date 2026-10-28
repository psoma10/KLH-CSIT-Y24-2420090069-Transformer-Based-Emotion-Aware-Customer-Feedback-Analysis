import { useMemo, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList } from "recharts";
import Card, { CardTitle } from "../components/ui/Card.jsx";
import PageHeader from "../components/ui/PageHeader.jsx";
import { cn } from "../lib/utils.js";
import data from "../data/model_comparison.json";

// lowerBetter metrics: best = smallest value.
const METRICS = [
  { key: "f1_macro", label: "Macro F1" },
  { key: "sensitivity_macro", label: "Sensitivity" },
  { key: "specificity_macro", label: "Specificity" },
  { key: "mcc_macro", label: "MCC" },
  { key: "kappa_macro", label: "Cohen's κ" },
  { key: "auc_roc_macro", label: "AUC-ROC" },
  { key: "pr_auc_macro", label: "PR-AUC" },
  { key: "log_loss", label: "Log loss ↓", lowerBetter: true },
  { key: "hamming_loss", label: "Hamming loss ↓", lowerBetter: true },
];

const GROUPS = [
  { id: "baseline", label: "Baseline", plural: "Baselines", color: "#94a3b8" },
  { id: "hybrid", label: "Hybrid", plural: "Hybrids", color: "#818cf8" },
  { id: "proposed", label: "Proposed", plural: "Proposed hybrid", color: "#4338ca" },
];
const GROUP_COLOR = Object.fromEntries(GROUPS.map((g) => [g.id, g.color]));

const fmt = (m) => (m?.mean == null ? "—" : m.mean.toFixed(4));
const shortName = (n) => n.replace("PROPOSED: ", "");

function bestValues(models) {
  const best = {};
  for (const { key, lowerBetter } of METRICS) {
    const vals = models.map((m) => m.metrics[key]?.mean).filter((v) => v != null);
    best[key] = lowerBetter ? Math.min(...vals) : Math.max(...vals);
  }
  return best;
}

function GroupChip({ group, children }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-semibold text-slate-700">
      <span className="h-2 w-2 rounded-full" style={{ background: group.color }} />
      {children ?? group.plural}
    </span>
  );
}

function Kpi({ metric, model, value }) {
  const group = GROUPS.find((g) => g.id === model.group);
  return (
    <div className="min-w-0 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
        Best {metric.label.replace(" ↓", "")}
        {metric.lowerBetter && <span className="normal-case"> (lowest)</span>}
      </div>
      <div className="mt-1 text-2xl font-bold tabular-nums text-slate-900">{value.toFixed(4)}</div>
      <div className="mt-2 flex min-w-0 items-center gap-2">
        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: group.color }} />
        <span className="truncate text-sm text-slate-600" title={model.name}>
          {shortName(model.name)}
        </span>
      </div>
    </div>
  );
}

export default function Compare() {
  const [metricKey, setMetricKey] = useState("f1_macro");
  const { models, perLabel } = data;
  const best = useMemo(() => bestValues(models), [models]);
  const metric = METRICS.find((m) => m.key === metricKey);

  const kpis = ["f1_macro", "auc_roc_macro", "mcc_macro", "log_loss"].map((key) => {
    const m = METRICS.find((x) => x.key === key);
    const model = models.find((x) => x.metrics[key]?.mean === best[key]);
    return { metric: m, model, value: best[key] };
  });

  const chartData = models
    .map((m) => ({ name: shortName(m.name), value: m.metrics[metricKey]?.mean ?? 0, group: m.group }))
    .sort((a, b) => (metric.lowerBetter ? a.value - b.value : b.value - a.value));
  const maxVal = Math.max(...chartData.map((d) => d.value));

  // Per-label F1: best model per row (ties are rare; first wins) and per-row range for shading.
  const labelInfo = perLabel.map((row) => {
    const entries = Object.entries(row.f1).filter(([, v]) => v != null);
    const bestId = entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    const vals = entries.map(([, v]) => v);
    return { bestId, min: Math.min(...vals), max: Math.max(...vals) };
  });

  const hasSeeds = models.some((m) => m.metrics.f1_macro?.std != null);

  return (
    <div className="min-w-0">
      <PageHeader
        title="Model comparison"
        description={`${data.dataset}. Macro-averaged over labels, thresholds tuned on the validation split. Values are exactly as produced by results/comparison/metrics_summary.csv${
          hasSeeds ? "; ± is the standard deviation over seeds for models trained with more than one seed" : ""
        }.`}
      />

      <div className="space-y-6">
        {/* KPI tiles */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {kpis.map((k) => (
            <Kpi key={k.metric.key} {...k} />
          ))}
        </div>

        {/* Metrics table */}
        <Card className="min-w-0">
          <CardTitle>Metrics</CardTitle>
          <div className="max-h-[560px] overflow-auto rounded-lg border border-slate-200">
            <table className="w-full min-w-[900px] border-separate border-spacing-0 text-sm">
              <thead>
                <tr>
                  <th className="sticky left-0 top-0 z-30 border-b border-slate-200 bg-slate-50 px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Model
                  </th>
                  {METRICS.map((m) => (
                    <th
                      key={m.key}
                      className="sticky top-0 z-20 whitespace-nowrap border-b border-slate-200 bg-slate-50 px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wide text-slate-500"
                    >
                      {m.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {GROUPS.map((g) => {
                  const rows = models.filter((m) => m.group === g.id);
                  return [
                    <tr key={`h-${g.id}`}>
                      <td className="sticky left-0 z-10 border-b border-slate-200 bg-slate-100 px-4 py-2">
                        <GroupChip group={g} />
                      </td>
                      <td colSpan={METRICS.length} className="border-b border-slate-200 bg-slate-100" />
                    </tr>,
                    ...rows.map((m) => {
                      const prop = m.group === "proposed";
                      return (
                        <tr key={m.id} className="group">
                          <td
                            className={cn(
                              "sticky left-0 z-10 whitespace-nowrap border-b border-slate-100 px-4 py-2.5 font-medium text-slate-900 group-hover:bg-slate-50",
                              prop ? "bg-indigo-50" : "bg-white"
                            )}
                          >
                            {m.name}
                          </td>
                          {METRICS.map(({ key }) => {
                            const v = m.metrics[key];
                            const isBest = v?.mean != null && v.mean === best[key];
                            return (
                              <td
                                key={key}
                                className={cn(
                                  "whitespace-nowrap border-b border-slate-100 px-4 py-2.5 text-right tabular-nums",
                                  isBest
                                    ? "bg-emerald-50 font-bold text-emerald-800"
                                    : cn("text-slate-700 group-hover:bg-slate-50", prop && "bg-indigo-50/50")
                                )}
                              >
                                {fmt(v)}
                                {v?.std != null && (
                                  <span className="ml-1 text-xs font-normal text-slate-400">±{v.std.toFixed(4)}</span>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    }),
                  ];
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-emerald-200 align-middle" />
            Best per column. ↓ lower is better. Models without ± were trained with a single seed, so their variance is
            unknown; small differences between models (especially among the top hybrids) are within seed noise.
          </p>
        </Card>

        {/* Bar chart */}
        <Card className="min-w-0">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <CardTitle className="mb-0">
              {metric.label.replace(" ↓", "")} by model
              <span className="ml-2 font-normal text-slate-500">
                {metric.lowerBetter ? "(lower is better)" : "(higher is better)"}
              </span>
            </CardTitle>
            <label className="flex items-center gap-2 text-sm text-slate-600">
              Metric
              <select
                value={metricKey}
                onChange={(e) => setMetricKey(e.target.value)}
                className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
              >
                {METRICS.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="mb-3 flex flex-wrap gap-4 text-xs text-slate-600">
            {GROUPS.map((g) => (
              <span key={g.id} className="flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-sm" style={{ background: g.color }} />
                {g.label}
              </span>
            ))}
          </div>
          <div className="w-full min-w-0">
            <ResponsiveContainer width="100%" height={chartData.length * 34 + 40}>
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 0 }} barCategoryGap={6}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, Math.ceil(maxVal * 20) / 20]}
                  tickFormatter={(v) => v.toFixed(2)}
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={190}
                  tick={{ fontSize: 11, fill: "#334155" }}
                  axisLine={false}
                  tickLine={false}
                  interval={0}
                />
                <Tooltip
                  cursor={{ fill: "#f1f5f9" }}
                  formatter={(value) => [value.toFixed(4), metric.label.replace(" ↓", "")]}
                  contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12 }}
                />
                <Bar dataKey="value" name={metric.label} radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {chartData.map((d) => (
                    <Cell key={d.name} fill={GROUP_COLOR[d.group]} />
                  ))}
                  <LabelList
                    dataKey="value"
                    position="right"
                    formatter={(v) => v.toFixed(3)}
                    style={{ fontSize: 11, fill: "#334155" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-xs text-slate-500">Sorted by value. Axis starts at 0 so bar lengths are not exaggerated.</p>
        </Card>

        {/* Per-label F1 heatmap */}
        <Card className="min-w-0">
          <CardTitle>Per-label F1</CardTitle>
          <div className="max-h-[640px] overflow-auto rounded-lg border border-slate-200">
            <table className="w-full min-w-[1100px] border-separate border-spacing-0 text-sm">
              <thead>
                <tr>
                  <th className="sticky left-0 top-0 z-30 border-b border-slate-200 bg-slate-50 px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Label
                  </th>
                  <th className="sticky top-0 z-20 border-b border-slate-200 bg-slate-50 px-3 py-2.5 text-right text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Support
                  </th>
                  {models.map((m) => (
                    <th
                      key={m.id}
                      className="sticky top-0 z-20 min-w-[84px] border-b border-slate-200 bg-slate-50 px-2 py-2 text-center align-bottom text-[11px] font-semibold leading-tight text-slate-600"
                      style={{ boxShadow: `inset 0 -3px 0 ${GROUP_COLOR[m.group]}` }}
                      title={m.name}
                    >
                      {m.group === "proposed" ? "★ " : ""}
                      {shortName(m.name)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {perLabel.map((row, i) => {
                  const info = labelInfo[i];
                  const span = info.max - info.min || 1;
                  return (
                    <tr key={row.label}>
                      <td className="sticky left-0 z-10 whitespace-nowrap border-b border-slate-100 bg-white px-3 py-1.5 font-medium capitalize text-slate-900">
                        {row.label}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-1.5 text-right tabular-nums text-slate-500">
                        {row.support}
                      </td>
                      {models.map((m) => {
                        const v = row.f1[m.id];
                        const isBest = info.bestId === m.id;
                        const alpha = v == null ? 0 : 0.06 + 0.5 * ((v - info.min) / span);
                        return (
                          <td
                            key={m.id}
                            className={cn(
                              "border-b border-slate-100 px-2 py-1.5 text-center tabular-nums",
                              isBest ? "font-bold text-slate-900" : "text-slate-700"
                            )}
                            style={{
                              backgroundColor: v == null ? undefined : `rgba(79, 70, 229, ${alpha.toFixed(3)})`,
                              boxShadow: isBest ? "inset 0 0 0 2px rgba(67, 56, 202, 0.85)" : undefined,
                            }}
                          >
                            {v == null ? "—" : v.toFixed(3)}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Cell shading is scaled within each label row (darker = higher F1 for that label); the best model per label
            is bold and outlined. Column header underline colour shows the model group.
          </p>
        </Card>
      </div>
    </div>
  );
}
