import { Table, Thead, Tbody, Th, Td } from "./ui/Table.jsx";
import Badge from "./ui/Badge.jsx";
import Button from "./ui/Button.jsx";
import { getBucketColor } from "../lib/emotions.js";

export default function ReviewsTable({ reviews, isLoading, limit, offset, onPageChange }) {
  return (
    <div className="space-y-3">
      <Table>
        <Thead>
          <tr>
            <Th>Review</Th>
            <Th>Product</Th>
            <Th>Rating</Th>
            <Th>Top emotion</Th>
            <Th>Bucket</Th>
            <Th>Date</Th>
          </tr>
        </Thead>
        <Tbody>
          {reviews.map((review) => (
            <tr key={review.id}>
              <Td className="max-w-[16rem] truncate sm:max-w-md" title={review.review_text}>{review.review_text}</Td>
              <Td className="whitespace-nowrap">{review.product_id ?? "—"}</Td>
              <Td className="whitespace-nowrap tabular-nums">{review.star_rating ?? "—"}</Td>
              <Td className="capitalize">{review.top_label ?? "—"}</Td>
              <Td>
                {review.business_bucket ? (
                  <Badge color={getBucketColor(review.business_bucket)}>{review.business_bucket}</Badge>
                ) : (
                  "—"
                )}
              </Td>
              <Td className="whitespace-nowrap">{review.review_date ?? "—"}</Td>
            </tr>
          ))}
          {!isLoading && reviews.length === 0 && (
            <tr>
              <Td colSpan={6} className="py-10 text-center text-sm text-slate-400">
                No reviews found.
              </Td>
            </tr>
          )}
        </Tbody>
      </Table>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Button variant="secondary" disabled={offset === 0} onClick={() => onPageChange(Math.max(0, offset - limit))}>
          Previous
        </Button>
        <span className="text-sm text-slate-500">{reviews.length === 0 ? "No results" : <>Showing {offset + 1}–{offset + reviews.length}</>}</span>
        <Button
          variant="secondary"
          disabled={reviews.length < limit}
          onClick={() => onPageChange(offset + limit)}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
