import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getModelInfo } from "../lib/api.js";
import Card, { CardTitle } from "../components/ui/Card.jsx";
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

export default function Model() {
  const query = useQuery({ queryKey: ["model-info"], queryFn: getModelInfo, retry: false });

  if (query.isLoading) {
    return <p className="text-sm text-slate-500">Loading model info…</p>;
  }

  if (query.error?.status === 404) {
    return (
      <div className="mx-auto max-w-2xl">
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
      <Card className="mx-auto max-w-2xl border-red-300 bg-red-50">
        <p className="text-sm text-red-700">{query.error.message}</p>
      </Card>
    );
  }

  const { model_version, test_metrics } = query.data;
  const perLabelRows = extractPerLabelF1(test_metrics?.per_label);

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <Card>
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Model version</p>
            <p className="text-xl font-bold text-slate-900">{model_version}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Macro F1</p>
            <p className="text-3xl font-bold text-slate-900">
              {test_metrics?.macro_f1 != null ? test_metrics.macro_f1.toFixed(4) : "—"}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Micro F1</p>
            <p className="text-3xl font-bold text-slate-900">
              {test_metrics?.micro_f1 != null ? test_metrics.micro_f1.toFixed(4) : "—"}
            </p>
          </div>
        </div>
      </Card>

      {perLabelRows.length > 0 && (
        <>
          <Card>
            <CardTitle>F1 by label</CardTitle>
            <ResponsiveContainer width="100%" height={560}>
              <BarChart data={perLabelRows} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 16 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" domain={[0, 1]} />
                <YAxis type="category" dataKey="label" width={100} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value) => value.toFixed(4)} />
                <Bar dataKey="f1" radius={[0, 4, 4, 0]} fill="#0f172a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <CardTitle>Per-label detail</CardTitle>
            <Table>
              <Thead>
                <tr>
                  <Th>Label</Th>
                  <Th>F1</Th>
                  <Th>Support</Th>
                </tr>
              </Thead>
              <Tbody>
                {perLabelRows.map((row) => (
                  <tr key={row.label}>
                    <Td className="capitalize">{row.label}</Td>
                    <Td>{row.f1.toFixed(4)}</Td>
                    <Td>{row.support}</Td>
                  </tr>
                ))}
              </Tbody>
            </Table>
          </Card>
        </>
      )}
    </div>
  );
}
