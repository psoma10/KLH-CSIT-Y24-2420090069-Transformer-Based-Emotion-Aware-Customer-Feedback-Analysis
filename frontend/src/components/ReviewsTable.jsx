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
              <Td className="max-w-md truncate">{review.review_text}</Td>
              <Td>{review.product_id ?? "—"}</Td>
              <Td>{review.star_rating ?? "—"}</Td>
              <Td className="capitalize">{review.top_label ?? "—"}</Td>
              <Td>
                {review.business_bucket ? (
                  <Badge color={getBucketColor(review.business_bucket)}>{review.business_bucket}</Badge>
                ) : (
                  "—"
                )}
              </Td>
              <Td>{review.review_date ?? "—"}</Td>
            </tr>
          ))}
          {!isLoading && reviews.length === 0 && (
            <tr>
              <Td colSpan={6} className="py-6 text-center text-slate-400">
                No reviews found.
              </Td>
            </tr>
          )}
        </Tbody>
      </Table>
      <div className="flex items-center justify-between">
        <Button variant="secondary" disabled={offset === 0} onClick={() => onPageChange(Math.max(0, offset - limit))}>
          Previous
        </Button>
        <span className="text-sm text-slate-500">Showing {offset + 1}–{offset + reviews.length}</span>
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
