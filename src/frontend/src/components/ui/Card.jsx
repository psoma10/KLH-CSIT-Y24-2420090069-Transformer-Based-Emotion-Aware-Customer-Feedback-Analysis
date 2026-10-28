import { cn } from "../../lib/utils.js";

export default function Card({ className, children, ...props }) {
  return (
    <div
      className={cn("rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardTitle({ className, children, ...props }) {
  return (
    <h3 className={cn("mb-4 text-sm font-semibold text-slate-900", className)} {...props}>
      {children}
    </h3>
  );
}
