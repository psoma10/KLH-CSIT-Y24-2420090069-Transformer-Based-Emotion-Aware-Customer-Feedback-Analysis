import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDistribution, getBuckets, getTrends, getReviews } from "../lib/api.js";
import { BUCKET_NAMES } from "../lib/emotions.js";
import Card, { CardTitle } from "../components/ui/Card.jsx";
import EmotionBarChart from "../components/EmotionBarChart.jsx";
import BucketDonut from "../components/BucketDonut.jsx";
import TrendsChart from "../components/TrendsChart.jsx";
import ReviewsTable from "../components/ReviewsTable.jsx";

const PAGE_SIZE = 10;

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

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <CardTitle className="mb-0">Emotion distribution</CardTitle>
            <select
              className="rounded-md border border-slate-300 px-2 py-1 text-sm"
              value={distributionBucket}
              onChange={(e) => setDistributionBucket(e.target.value)}
            >
              <option value="">All buckets</option>
              {BUCKET_NAMES.map((bucket) => (
                <option key={bucket} value={bucket}>
                  {bucket}
                </option>
              ))}
            </select>
          </div>
          {distributionQuery.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
          {distributionQuery.isError && <p className="text-sm text-red-700">{distributionQuery.error.message}</p>}
          {distributionQuery.isSuccess && distributionQuery.data.length === 0 && (
            <p className="text-sm text-slate-400">No predictions logged yet.</p>
          )}
          {distributionQuery.isSuccess && distributionQuery.data.length > 0 && (
            <EmotionBarChart scores={distributionScores} height={480} valueFormatter={(v) => `${v} reviews`} />
          )}
        </Card>

        <Card>
          <CardTitle>Business bucket distribution</CardTitle>
          {bucketsQuery.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
          {bucketsQuery.isError && <p className="text-sm text-red-700">{bucketsQuery.error.message}</p>}
          {bucketsQuery.isSuccess && bucketsQuery.data.length === 0 && (
            <p className="text-sm text-slate-400">No predictions logged yet.</p>
          )}
          {bucketsQuery.isSuccess && bucketsQuery.data.length > 0 && <BucketDonut buckets={bucketsQuery.data} />}
        </Card>
      </div>

      <Card>
        <CardTitle>Trends over time</CardTitle>
        {trendsQuery.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
        {trendsQuery.isError && <p className="text-sm text-red-700">{trendsQuery.error.message}</p>}
        {trendsQuery.isSuccess && trendsQuery.data.length === 0 && (
          <p className="text-sm text-slate-400">No dated reviews logged yet.</p>
        )}
        {trendsQuery.isSuccess && trendsQuery.data.length > 0 && <TrendsChart trends={trendsQuery.data} />}
      </Card>

      <Card>
        <div className="mb-3 flex items-center justify-between">
          <CardTitle className="mb-0">Recent reviews</CardTitle>
          <select
            className="rounded-md border border-slate-300 px-2 py-1 text-sm"
            value={reviewsBucket}
            onChange={(e) => {
              setReviewsBucket(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">All buckets</option>
            {BUCKET_NAMES.map((bucket) => (
              <option key={bucket} value={bucket}>
                {bucket}
              </option>
            ))}
          </select>
        </div>
        {reviewsQuery.isError && <p className="text-sm text-red-700">{reviewsQuery.error.message}</p>}
        <ReviewsTable
          reviews={reviewsQuery.data ?? []}
          isLoading={reviewsQuery.isLoading}
          limit={PAGE_SIZE}
          offset={offset}
          onPageChange={setOffset}
        />
      </Card>
    </div>
  );
}
