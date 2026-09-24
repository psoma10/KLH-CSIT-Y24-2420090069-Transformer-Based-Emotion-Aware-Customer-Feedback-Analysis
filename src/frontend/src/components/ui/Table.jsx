import { cn } from "../../lib/utils.js";

export function Table({ className, children }) {
  return (
    <div className="overflow-x-auto">
      <table className={cn("w-full text-left text-sm", className)}>{children}</table>
    </div>
  );
}

export function Thead({ children }) {
  return <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">{children}</thead>;
}

export function Th({ className, children, ...props }) {
  return (
    <th className={cn("px-3 py-2 font-semibold", className)} {...props}>
      {children}
    </th>
  );
}

export function Tbody({ children }) {
  return <tbody className="divide-y divide-slate-100">{children}</tbody>;
}

export function Td({ className, children, ...props }) {
  return (
    <td className={cn("px-3 py-2 align-top", className)} {...props}>
      {children}
    </td>
  );
}
