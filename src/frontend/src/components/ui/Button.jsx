import { cn } from "../../lib/utils.js";

const VARIANTS = {
  primary: "bg-indigo-600 text-white shadow-sm hover:bg-indigo-700 disabled:bg-slate-300 disabled:shadow-none",
  secondary: "bg-white text-slate-700 border border-slate-300 shadow-sm hover:bg-slate-50 disabled:text-slate-400",
  ghost: "text-slate-600 hover:bg-slate-100 disabled:text-slate-400",
};
const SIZES = { sm: "px-3 py-1.5 text-xs", md: "px-4 py-2 text-sm" };

export default function Button({ variant = "primary", size = "md", className, children, ...props }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed",
        SIZES[size],
        VARIANTS[variant],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
