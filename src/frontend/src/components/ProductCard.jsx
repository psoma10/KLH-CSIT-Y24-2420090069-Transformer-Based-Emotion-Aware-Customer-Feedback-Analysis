import { memo } from "react";
import { StarDisplay } from "./StarRating.jsx";
import RatingHistogram from "./RatingHistogram.jsx";
import { formatPrice, formatCount } from "../lib/demoProducts.js";
import { cn } from "../lib/utils.js";

/** Gradient tile standing in for a product photo (no image assets in repo). */
function ProductImage({ product }) {
  return (
    <div
      className="relative flex aspect-square w-full items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-white shadow-inner"
      aria-hidden="true"
    >
      <div
        className="absolute inset-0 opacity-[0.09] transition-opacity duration-500"
        style={{
          background: `radial-gradient(circle at 30% 25%, ${product.accent} 0%, transparent 62%)`,
        }}
      />
      <div
        className="absolute inset-x-0 bottom-0 h-1/3 opacity-[0.06]"
        style={{ background: `linear-gradient(to top, ${product.accent}, transparent)` }}
      />
      <span className="relative select-none text-7xl drop-shadow-sm transition-transform duration-500 ease-out group-hover:scale-105 motion-reduce:transition-none sm:text-8xl">
        {product.emoji}
      </span>
    </div>
  );
}

function ProductSwitcher({ products, selectedId, onSelect }) {
  return (
    <div
      role="tablist"
      aria-label="Choose a demo product"
      className="flex gap-2"
    >
      {products.map((p) => {
        const active = p.id === selectedId;
        return (
          <button
            key={p.id}
            type="button"
            role="tab"
            aria-selected={active}
            aria-label={`View ${p.brand} ${p.title}`}
            onClick={() => onSelect(p.id)}
            className={cn(
              "flex h-14 w-14 items-center justify-center rounded-lg border-2 text-2xl transition-all duration-200 ease-out",
              "hover:-translate-y-0.5 hover:shadow-md motion-reduce:transition-none motion-reduce:hover:translate-y-0",
              "focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 focus-visible:ring-offset-2",
              active
                ? "border-orange-400 bg-orange-50 shadow-sm ring-1 ring-orange-200"
                : "border-slate-200 bg-white opacity-70 hover:opacity-100"
            )}
          >
            <span aria-hidden="true">{p.emoji}</span>
          </button>
        );
      })}
    </div>
  );
}

// Memoized: Analyze re-renders on every keystroke, but this card's props
// only change when the visitor picks a different product.
function ProductCard({ products, product, onSelect }) {
  const discount = Math.round(
    ((product.listPrice - product.price) / product.listPrice) * 100
  );

  return (
    <section
      aria-label="Product details"
      className="group grid gap-8 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]"
    >
      <div className="space-y-4">
        <ProductImage product={product} />
        <ProductSwitcher products={products} selectedId={product.id} onSelect={onSelect} />
      </div>

      <div className="min-w-0 space-y-4">
        <div className="space-y-1.5">
          <a
            href="#!"
            onClick={(e) => e.preventDefault()}
            className="text-sm font-medium text-sky-700 transition-colors hover:text-orange-600 hover:underline"
          >
            Visit the {product.brand} Store
          </a>
          <h1 className="text-xl font-semibold leading-snug text-slate-900 sm:text-2xl">
            {product.title}
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="text-sm font-semibold text-slate-900">
            {product.rating.toFixed(1)}
          </span>
          <StarDisplay
            value={product.rating}
            size="md"
            label={`${product.rating} out of 5 stars, ${formatCount(product.ratingCount)} ratings`}
          />
          <span className="text-sm text-sky-700 transition-colors hover:text-orange-600 hover:underline">
            {formatCount(product.ratingCount)} ratings
          </span>
        </div>

        <hr className="border-slate-200" />

        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="rounded bg-red-600 px-2 py-0.5 text-sm font-semibold text-white">
            -{discount}%
          </span>
          <span className="text-3xl font-medium tracking-tight text-slate-900">
            {formatPrice(product.price)}
          </span>
          <span className="text-sm text-slate-500">
            List: <s>{formatPrice(product.listPrice)}</s>
          </span>
        </div>

        <p
          className={cn(
            "text-lg font-medium",
            product.inStock ? "text-green-700" : "text-red-700"
          )}
        >
          {product.inStock ? "In Stock" : "Temporarily out of stock"}
        </p>

        <ul className="space-y-1.5 text-sm leading-relaxed text-slate-700">
          {product.bullets.map((b) => (
            <li key={b} className="flex items-start gap-2.5">
              <span
                aria-hidden="true"
                className="mt-[0.5em] h-1.5 w-1.5 shrink-0 rounded-full bg-slate-400"
              />
              <span>{b}</span>
            </li>
          ))}
        </ul>

        <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4">
          <h2 className="mb-3 text-sm font-semibold text-slate-900">Customer reviews</h2>
          <RatingHistogram histogram={product.histogram} />
        </div>
      </div>
    </section>
  );
}

export default memo(ProductCard);
