import clsx from "clsx";

/** Merge Tailwind class lists conditionally (no tailwind-merge — clsx is enough here). */
export function cn(...inputs) {
  return clsx(...inputs);
}
