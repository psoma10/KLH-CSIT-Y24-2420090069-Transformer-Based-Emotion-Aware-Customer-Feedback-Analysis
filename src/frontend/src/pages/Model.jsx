import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getModelInfo } from "../lib/api.js";
import Card, { CardTitle } from "../components/ui/Card.jsx";
import PageHeader from "../components/ui/PageHeader.jsx";
import { Table, Thead, Tbody, Th, Td } from "../components/ui/Table.jsx";

// sklearn's classification_report(output_dict=True) mixes per-label rows in
// with aggregate rows (accuracy/macro avg/weighted avg/samples avg) — only
// the per-label rows belong in the F1-by-label table below.
const AGGREGATE_KEYS = new Set(["accuracy", "macro avg", "weighted avg", "samples avg", "micro avg"]);

function extractPerLabelF1(perLabel) {
  if (!perLabel) return [];
  return Object.entries(perLabel)
    .filter(([key, value]) => !AGGREGATE_KEYS.has(key) && typeof value === "object" && "f1-score" in value)
    .map(([label, metrics]) => ({ label, f1: metrics["f1-score"], support: metrics.support }))
    .sort((a, b) => b.f1 - a.f1);
}

const HEADER = <PageHeader title="Model" description="Held-out test performance of the deployed classifier." />;

function Stat({ label, value, small }) {
  return (
    <Card className="!p-4 sm:!p-5">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-1 truncate font-bold tabular-nums text-slate-900 ${small ? "text-xl leading-9" : "text-3xl"}`}>
        {value}
      </p>
    </Card>
  );
}

export default function Model() {
  const query = useQuery({ queryKey: ["model-info"], queryFn: getModelInfo, retry: false });

  if (query.isLoading) {
    return (
      <div className="mx-auto max-w-5xl">
        {HEADER}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-24 animate-pulse rounded-xl bg-slate-100" />
          ))}
        </div>
      </div>
    );
  }

  if (query.error?.status === 404) {
    return (
      <div className="mx-auto max-w-2xl">
        {HEADER}
        <Card className="border-amber-300 bg-amber-50">
          <CardTitle className="text-amber-800">Model not yet trained</CardTitle>
          <p className="text-sm text-amber-700">
            No training/evaluation metrics were found. This is expected on a fresh checkout — run the training
            pipeline first:
          </p>
          <pre className="mt-3 overflow-x-auto rounded-md bg-amber-100 p-3 text-xs text-amber-900">
            python ml/train_roberta.py --config ml/configs/train_config.yaml{"\n"}
            python ml/evaluate.py --config ml/configs/train_config.yaml
          </pre>
          <p className="mt-2 text-sm text-amber-700">
            Once <code className="rounded bg-amber-100 px-1">test_metrics.json</code> exists in the model artifact
            directory, refresh this page.
          </p>
        </Card>
      </div>
    );
  }

  if (query.isError) {
    return (
      <div className="mx-auto max-w-2xl">
        {HEADER}
        <Card className="border-red-300 bg-red-50">
          <p className="text-sm text-red-700">{query.error.message}</p>
        </Card>
      </div>
    );
  }

  const { model_version, test_metrics } = query.data;
  const perLabelRows = extractPerLabelF1(test_metrics?.per_label);
  const fmt = (v) => (v != null ? v.toFixed(4) : "—");

  return (
    <div className="mx-auto max-w-5xl">
      {HEADER}
      <div className="space-y-6">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Stat label="Model" value={model_version} small />
          <Stat label="Macro F1" value={fmt(test_metrics?.macro_f1)} />
          <Stat label="Micro F1" value={fmt(test_metrics?.micro_f1)} />
        </div>

        {perLabelRows.length === 0 && (
          <Card>
            <p className="text-sm text-slate-500">No per-label metrics available for this model.</p>
          </Card>
        )}

        {perLabelRows.length > 0 && (
          <div className="grid gap-6 lg:grid-cols-5">
            <Card className="min-w-0 lg:col-span-3">
              <CardTitle>F1 by label</CardTitle>
              <ResponsiveContainer width="100%" height={Math.max(360, perLabelRows.length * 22 + 40)}>
                <BarChart data={perLabelRows} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                  <XAxis
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 12, fill: "#64748b" }}
                    tickLine={false}
                    axisLine={{ stroke: "#cbd5e1" }}
                  />
                  <YAxis
                    type="category"
                    dataKey="label"
                    width={104}
                    interval={0}
                    tick={{ fontSize: 12, fill: "#475569" }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    cursor={{ fill: "#f1f5f9" }}
                    contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12 }}
                    formatter={(value) => [value.toFixed(4), "F1"]}
                  />
                  <Bar dataKey="f1" radius={[0, 4, 4, 0]} fill="#4f46e5" barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card className="min-w-0 lg:col-span-2">
              <CardTitle>Per-label detail</CardTitle>
              <div className="max-h-[640px] overflow-y-auto">
                <Table>
                  <Thead>
                    <tr>
                      <Th>Label</Th>
                      <Th className="text-right">F1</Th>
                      <Th className="text-right">Support</Th>
                    </tr>
                  </Thead>
                  <Tbody>
                    {perLabelRows.map((row) => (
                      <tr key={row.label}>
                        <Td className="capitalize">{row.label}</Td>
                        <Td className="text-right tabular-nums">{row.f1.toFixed(4)}</Td>
                        <Td className="text-right tabular-nums">{row.support}</Td>
                      </tr>
                    ))}
                  </Tbody>
                </Table>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
