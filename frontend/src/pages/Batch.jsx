import { useCallback, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { uploadBatch, getJob, getReviews } from "../lib/api.js";
import Card, { CardTitle } from "../components/ui/Card.jsx";
import Button from "../components/ui/Button.jsx";
import ProgressBar from "../components/ui/ProgressBar.jsx";
import BucketDonut from "../components/BucketDonut.jsx";
import { BUCKET_NAMES } from "../lib/emotions.js";
import { cn } from "../lib/utils.js";

const TERMINAL_STATUSES = ["done", "failed"];

const SAMPLE_CSV = `review_id,product_id,review_text,star_rating,review_date
1,b09-anc-headphones,"Battery life is incredible and the noise cancellation is on another level.",5,2026-01-14
2,b07-espresso,"Steam wand stopped working after two weeks. Very disappointed.",1,2026-01-15
3,b0c-standing-desk,"Assembly was confusing but the desk itself feels sturdy.",3,2026-01-16
`;

function downloadSampleCsv() {
  const blob = new Blob([SAMPLE_CSV], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sample_reviews.csv";
  a.click();
  URL.revokeObjectURL(url);
}

/** Client-side bucket tally for the reviews this batch just wrote. There is no
 * job-scoped analytics endpoint, so this fetches the most recent rows (the
 * job's own row count, newest first) right after completion — an honest
 * approximation for a demo, not a guarantee against concurrent uploads. */
function useBatchResults(job) {
  const enabled = job?.status === "done" && job.total_rows > 0;
  const query = useQuery({
    queryKey: ["batch-results", job?.id],
    queryFn: () => getReviews({ limit: job.total_rows, offset: 0 }),
    enabled,
  });

  const buckets = useMemo(() => {
    if (!query.data) return [];
    const counts = Object.fromEntries(BUCKET_NAMES.map((b) => [b, 0]));
    for (const review of query.data) {
      if (review.business_bucket && counts[review.business_bucket] != null) {
        counts[review.business_bucket] += 1;
      }
    }
    return BUCKET_NAMES.map((bucket) => ({ bucket, count: counts[bucket] })).filter(
      (b) => b.count > 0
    );
  }, [query.data]);

  return { ...query, buckets };
}

function DropZone({ file, onFile, disabled }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const dropped = e.dataTransfer.files?.[0];
      if (dropped) onFile(dropped);
    },
    [onFile]
  );

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={disabled ? undefined : handleDrop}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors",
        disabled && "cursor-not-allowed opacity-60",
        isDragging ? "border-slate-500 bg-slate-50" : "border-slate-300 hover:border-slate-400"
      )}
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 24 24"
        className="h-8 w-8 text-slate-400"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 8.25L12 3.75m0 0L7.5 8.25M12 3.75v12"
        />
      </svg>
      {file ? (
        <p className="text-sm font-medium text-slate-900">{file.name}</p>
      ) : (
        <>
          <p className="text-sm font-medium text-slate-700">
            Drag a CSV here, or click to choose a file
          </p>
          <p className="text-xs text-slate-400">.csv up to the configured upload limit</p>
        </>
      )}
      <input
        type="file"
        accept=".csv"
        disabled={disabled}
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        className="sr-only"
      />
    </label>
  );
}

export default function Batch() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);

  const uploadMutation = useMutation({
    mutationFn: uploadBatch,
    onSuccess: (job) => setJobId(job.id),
  });

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    enabled: Boolean(jobId),
    // Stop polling once the job reaches a terminal state — refetchInterval
    // is re-evaluated against the latest query data on every tick.
    refetchInterval: (query) => (TERMINAL_STATUSES.includes(query.state.data?.status) ? false : 2000),
  });

  const job = jobQuery.data;
  const results = useBatchResults(job);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    setJobId(null);
    uploadMutation.mutate(file);
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <Card>
        <CardTitle>Upload a review CSV</CardTitle>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm text-slate-500">
            Columns required:{" "}
            <code className="rounded bg-slate-100 px-1 py-0.5 text-xs">
              review_id, product_id, review_text, star_rating, review_date
            </code>
          </p>
          <button
            type="button"
            onClick={downloadSampleCsv}
            className="whitespace-nowrap text-xs font-medium text-slate-600 underline decoration-slate-300 underline-offset-2 hover:text-slate-900"
          >
            Download sample CSV
          </button>
        </div>
        <form onSubmit={handleSubmit} className="space-y-3">
          <DropZone file={file} onFile={setFile} disabled={uploadMutation.isPending} />
          <Button type="submit" disabled={!file || uploadMutation.isPending}>
            {uploadMutation.isPending ? "Uploading…" : "Upload & process"}
          </Button>
        </form>
        {uploadMutation.isError && <p className="mt-2 text-sm text-red-700">{uploadMutation.error.message}</p>}
      </Card>

      {jobId && (
        <Card>
          <CardTitle>Job {jobId}</CardTitle>
          {jobQuery.isLoading && <p className="text-sm text-slate-500">Fetching job status…</p>}
          {jobQuery.isError && <p className="text-sm text-red-700">{jobQuery.error.message}</p>}
          {job && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <span className="font-medium capitalize">{job.status}</span>
                <span className="text-slate-500">
                  ({job.processed_rows} / {job.total_rows} rows)
                </span>
              </div>
              <ProgressBar
                value={job.processed_rows}
                max={job.total_rows}
                barClassName={job.status === "failed" ? "bg-red-600" : "bg-slate-900"}
              />
              {job.status === "done" && (
                <p className="text-sm text-green-700">
                  Completed at {job.completed_at ? new Date(job.completed_at).toLocaleString() : "—"}.
                </p>
              )}
              {job.status === "failed" && <p className="text-sm text-red-700">{job.error ?? "Job failed."}</p>}
            </div>
          )}
        </Card>
      )}

      {job?.status === "done" && (
        <Card>
          <CardTitle>Results</CardTitle>
          {results.isLoading && <p className="text-sm text-slate-500">Loading results…</p>}
          {results.isError && (
            <p className="text-sm text-amber-700">
              Rows processed, but the results summary couldn't load: {results.error.message}
            </p>
          )}
          {!results.isLoading && !results.isError && results.buckets.length === 0 && (
            <p className="text-sm text-slate-500">
              No confident predictions in this batch — every row landed under its threshold.
            </p>
          )}
          {results.buckets.length > 0 && <BucketDonut buckets={results.buckets} height={220} />}
        </Card>
      )}
    </div>
  );
}
