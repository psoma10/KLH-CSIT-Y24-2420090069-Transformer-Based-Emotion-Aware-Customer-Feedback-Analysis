// Diverging red/blue scale: positive attribution (pushes the predicted
// emotion up) renders warm, negative renders cool, near-zero stays neutral.
// Opacity is driven by magnitude relative to the largest |attribution| in
// this explanation, so the strongest tokens in any given review always read
// as fully saturated regardless of the model's raw SHAP value range.
function tokenStyle(value, maxAbs) {
  if (maxAbs === 0) return { backgroundColor: "transparent" };
  const intensity = Math.min(1, Math.abs(value) / maxAbs);
  const alpha = 0.12 + intensity * 0.68;
  const color = value >= 0 ? `rgba(220, 38, 38, ${alpha})` : `rgba(37, 99, 235, ${alpha})`;
  return { backgroundColor: color };
}

export default function TokenHeatmap({ tokens, attributions }) {
  const maxAbs = attributions.reduce((max, v) => Math.max(max, Math.abs(v)), 0);

  return (
    <div className="flex flex-wrap gap-1 leading-loose">
      {tokens.map((token, i) => (
        <span
          key={`${token}-${i}`}
          title={`${token}: ${attributions[i].toFixed(4)}`}
          className="rounded px-1.5 py-0.5 text-sm text-slate-900"
          style={tokenStyle(attributions[i], maxAbs)}
        >
          {token}
        </span>
      ))}
    </div>
  );
}
