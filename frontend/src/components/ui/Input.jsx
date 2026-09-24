import { cn } from "../../lib/utils.js";

export default function Input({ className, ...props }) {
  return (
    <input
      className={cn(
        "rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-slate-500 focus:outline-none focus:ring-1 focus:ring-slate-500",
        className
      )}
      {...props}
    />
  );
}
