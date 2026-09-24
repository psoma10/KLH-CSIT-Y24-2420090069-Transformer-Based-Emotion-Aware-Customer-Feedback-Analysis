import { cn } from "../../lib/utils.js";

export default function Badge({ color, className, children }) {
  return (
    <span
      className={cn("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium text-white", className)}
      style={color ? { backgroundColor: color } : undefined}
    >
      {children}
    </span>
  );
}
