// Single source of truth for backend calls. Page components must import
// from here rather than calling fetch() directly, so the base URL, error
// handling, and request shapes stay consistent across the app.

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const API_PREFIX = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // response had no JSON body — fall back to statusText
    }
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = res.status;
    error.detail = detail;
    throw error;
  }

  if (res.status === 204) return null;
  return res.json();
}

/**
 * Segment-aware analysis: scores each part of a review separately so praise in
 * one sentence cannot cancel out a complaint in the next.
 *
 * `persist: false` (the default) is a preview — nothing is written server-side.
 * The live-typing panel relies on this: it calls analyzeReview on every
 * keystroke, and a persisting call there would fill the Dashboard with one row
 * per debounce tick. Pass `persist: true` only from the actual Submit action.
 */
export function analyzeReview(text, { persist = false, starRating, productId } = {}) {
  return request(`${API_PREFIX}/analyze`, {
    method: "POST",
    body: JSON.stringify({
      text,
      persist,
      ...(starRating ? { star_rating: starRating } : {}),
      ...(productId ? { product_id: productId } : {}),
    }),
  });
}

export function explainReview(text, label) {
  return request(`${API_PREFIX}/explain`, {
    method: "POST",
    body: JSON.stringify({ text, ...(label ? { label } : {}) }),
  });
}

export function getReviews({ limit = 50, offset = 0, businessBucket, productId } = {}) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (businessBucket) params.set("business_bucket", businessBucket);
  if (productId) params.set("product_id", productId);
  return request(`${API_PREFIX}/reviews?${params.toString()}`);
}

export function uploadBatch(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request(`${API_PREFIX}/batch`, {
    method: "POST",
    body: formData,
  });
}

export function getJob(jobId) {
  return request(`${API_PREFIX}/jobs/${jobId}`);
}

export function getDistribution(businessBucket) {
  const params = new URLSearchParams();
  if (businessBucket) params.set("business_bucket", businessBucket);
  const qs = params.toString();
  return request(`${API_PREFIX}/analytics/distribution${qs ? `?${qs}` : ""}`);
}

export function getBuckets() {
  return request(`${API_PREFIX}/analytics/buckets`);
}

export function getTrends() {
  return request(`${API_PREFIX}/analytics/trends`);
}

export function getModelInfo() {
  return request(`${API_PREFIX}/model/info`);
}

