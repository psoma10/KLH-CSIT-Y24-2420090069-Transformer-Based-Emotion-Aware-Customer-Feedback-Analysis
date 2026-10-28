import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDistribution, getBuckets, getTrends, getReviews } from "../lib/api.js";
import { BUCKET_NAMES, getBucketColor } from "../lib/emotions.js";
import Card, { CardTitle } from "../components/ui/Card.jsx";
import PageHeader from "../components/ui/PageHeader.jsx";
import EmotionBarChart from "../components/EmotionBarChart.jsx";
import BucketDonut from "../components/BucketDonut.jsx";
import TrendsChart from "../components/TrendsChart.jsx";
import ReviewsTable from "../components/ReviewsTable.jsx";

const PAGE_SIZE = 10;
const SELECT_CLASS =
  "rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm capitalize text-slate-700 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200";

function BucketSelect({ value, onChange, label }) {
  return (
    <select aria-label={label} className={SELECT_CLASS} value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">All buckets</option>
      {BUCKET_NAMES.map((bucket) => (
        <option key={bucket} value={bucket}>
          {bucket}
        </option>
      ))}
    </select>
  );
}

function CardHeader({ title, children }) {
  return (
    <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
      <CardTitle className="mb-0">{title}</CardTitle>
      {children}
    </div>
  );
}

function Skeleton({ height = 240 }) {
  return <div className="animate-pulse rounded-lg bg-slate-100" style={{ height }} aria-busy="true" />;
}

function Empty({ children }) {
  return (
    <div className="flex min-h-[160px] items-center justify-center rounded-lg border border-dashed border-slate-200 bg-slate-50 px-4 text-center text-sm text-slate-500">
      {children}
    </div>
  );
}

function ErrorMsg({ error }) {
  return <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{error.message}</p>;
}

function StatTile({ label, value, color, loading }) {
  return (
    <Card className="!p-4 sm:!p-5">
      <div className="flex items-center gap-2">
        {color && <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: color }} />}
        <p className="truncate text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      </div>
      {loading ? (
        <div className="mt-2 h-8 w-16 animate-pulse rounded bg-slate-100" />
      ) : (
        <p className="mt-1 text-2xl font-bold tabular-nums text-slate-900">{value}</p>
      )}
    </Card>
  );
}

export default function Dashboard() {
  const [distributionBucket, setDistributionBucket] = useState("");
  const [reviewsBucket, setReviewsBucket] = useState("");
  const [offset, setOffset] = useState(0);

  const distributionQuery = useQuery({
    queryKey: ["distribution", distributionBucket],
    queryFn: () => getDistribution(distributionBucket || undefined),
  });
  const bucketsQuery = useQuery({ queryKey: ["buckets"], queryFn: getBuckets });
  const trendsQuery = useQuery({ queryKey: ["trends"], queryFn: getTrends });
  const reviewsQuery = useQuery({
    queryKey: ["reviews", reviewsBucket, offset],
    queryFn: () => getReviews({ limit: PAGE_SIZE, offset, businessBucket: reviewsBucket || undefined }),
  });

  const distributionScores = Object.fromEntries(
    (distributionQuery.data ?? []).map((item) => [item.label, item.count])
  );

  const bucketCounts = Object.fromEntries((bucketsQuery.data ?? []).map((b) => [b.bucket, b.count]));
  const total = (bucketsQuery.data ?? []).reduce((sum, b) => sum + b.count, 0);

  return (
    <div className="mx-auto max-w-6xl">
      <PageHeader
        title="Dashboard"
        description="Emotion and business-bucket analytics across every review logged so far."
      />

      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <StatTile label="Total reviews" value={total.toLocaleString()} loading={bucketsQuery.isLoading} />
          {BUCKET_NAMES.map((bucket) => (
            <StatTile
              key={bucket}
              label={bucket}
              value={(bucketCounts[bucket] ?? 0).toLocaleString()}
              color={getBucketColor(bucket)}
              loading={bucketsQuery.isLoading}
            />
          ))}
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card className="min-w-0">
            <CardHeader title="Emotion distribution">
              <BucketSelect
                label="Filter distribution by bucket"
                value={distributionBucket}
                onChange={setDistributionBucket}
              />
            </CardHeader>
            {distributionQuery.isLoading && <Skeleton height={480} />}
            {distributionQuery.isError && <ErrorMsg error={distributionQuery.error} />}
            {distributionQuery.isSuccess && distributionQuery.data.length === 0 && (
              <Empty>No predictions logged yet.</Empty>
            )}
            {distributionQuery.isSuccess && distributionQuery.data.length > 0 && (
              <EmotionBarChart scores={distributionScores} height={480} valueFormatter={(v) => `${v} reviews`} />
            )}
          </Card>

          <Card className="min-w-0">
            <CardHeader title="Business bucket distribution" />
            {bucketsQuery.isLoading && <Skeleton height={320} />}
            {bucketsQuery.isError && <ErrorMsg error={bucketsQuery.error} />}
            {bucketsQuery.isSuccess && bucketsQuery.data.length === 0 && <Empty>No predictions logged yet.</Empty>}
            {bucketsQuery.isSuccess && bucketsQuery.data.length > 0 && (
              <BucketDonut buckets={bucketsQuery.data} height={360} />
            )}
          </Card>
        </div>

        <Card className="min-w-0">
          <CardHeader title="Trends over time" />
          {trendsQuery.isLoading && <Skeleton height={320} />}
          {trendsQuery.isError && <ErrorMsg error={trendsQuery.error} />}
          {trendsQuery.isSuccess && trendsQuery.data.length === 0 && <Empty>No dated reviews logged yet.</Empty>}
          {trendsQuery.isSuccess && trendsQuery.data.length > 0 && <TrendsChart trends={trendsQuery.data} />}
        </Card>

        <Card className="min-w-0">
          <CardHeader title="Recent reviews">
            <BucketSelect
              label="Filter reviews by bucket"
              value={reviewsBucket}
              onChange={(v) => {
                setReviewsBucket(v);
                setOffset(0);
              }}
            />
          </CardHeader>
          {reviewsQuery.isError && <ErrorMsg error={reviewsQuery.error} />}
          <ReviewsTable
            reviews={reviewsQuery.data ?? []}
            isLoading={reviewsQuery.isLoading}
            limit={PAGE_SIZE}
            offset={offset}
            onPageChange={setOffset}
          />
        </Card>
      </div>
    </div>
  );
}
