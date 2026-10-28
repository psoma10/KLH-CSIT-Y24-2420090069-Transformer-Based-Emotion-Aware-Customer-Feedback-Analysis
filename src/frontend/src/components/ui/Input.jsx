import { cn } from "../../lib/utils.js";

export default function Input({ className, ...props }) {
  return (
    <input
      className={cn(
        "rounded-lg border border-slate-300 bg-white px-3 py-2 shadow-sm text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20",
        className
      )}
      {...props}
    />
  );
}
