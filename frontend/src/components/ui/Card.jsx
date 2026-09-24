import { cn } from "../../lib/utils.js";

export default function Card({ className, children, ...props }) {
  return (
    <div
      className={cn("rounded-lg border border-slate-200 bg-white p-5 shadow-sm", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardTitle({ className, children, ...props }) {
  return (
    <h3 className={cn("mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500", className)} {...props}>
      {children}
    </h3>
  );
}
