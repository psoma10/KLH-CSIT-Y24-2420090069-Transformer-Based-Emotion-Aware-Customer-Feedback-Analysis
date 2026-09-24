import { Routes, Route, NavLink } from "react-router-dom";
import Analyze from "./pages/Analyze.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Batch from "./pages/Batch.jsx";
import Model from "./pages/Model.jsx";
import { cn } from "./lib/utils.js";

const NAV_LINKS = [
  { to: "/", label: "Analyze", end: true },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/batch", label: "Batch" },
  { to: "/model", label: "Model" },
];

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-4">
          <span className="text-sm font-semibold text-slate-900">Emotion-Aware Feedback Analysis</span>
          <nav className="flex gap-4">
            {NAV_LINKS.map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    "text-sm font-medium",
                    isActive ? "text-slate-900" : "text-slate-500 hover:text-slate-700"
                  )
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8">
        <Routes>
          <Route path="/" element={<Analyze />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/model" element={<Model />} />
        </Routes>
      </main>
    </div>
  );
}
