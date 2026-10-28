import { useState } from "react";
import { Routes, Route, NavLink, useLocation } from "react-router-dom";
import Analyze from "./pages/Analyze.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Batch from "./pages/Batch.jsx";
import Model from "./pages/Model.jsx";
import Compare from "./pages/Compare.jsx";
import { cn } from "./lib/utils.js";

const NAV_LINKS = [
  { to: "/", label: "Analyze", end: true },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/batch", label: "Batch" },
  { to: "/model", label: "Model" },
  { to: "/compare", label: "Comparison" },
];

const linkClass = ({ isActive }) =>
  cn(
    "rounded-lg px-3 py-2 text-sm font-medium transition-colors",
    isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
  );

function Logo() {
  return (
    <NavLink to="/" className="flex items-center gap-2.5">
      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-sm">
        <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="9" />
          <path d="M8 14s1.5 2 4 2 4-2 4-2" />
          <path d="M9 9h.01M15 9h.01" />
        </svg>
      </span>
      <span className="leading-tight">
        <span className="block text-sm font-bold text-slate-900">EmotionLens</span>
        <span className="hidden text-[11px] text-slate-500 sm:block">Emotion-aware feedback analysis</span>
      </span>
    </NavLink>
  );
}

export default function App() {
  const [open, setOpen] = useState(false);
  useLocation(); // re-render on route change so the mobile menu closes via onClick below

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <Logo />
          <nav className="hidden items-center gap-1 md:flex" aria-label="Main">
            {NAV_LINKS.map(({ to, label, end }) => (
              <NavLink key={to} to={to} end={end} className={linkClass}>
                {label}
              </NavLink>
            ))}
          </nav>
          <button
            type="button"
            className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 md:hidden"
            aria-label="Toggle menu"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
            </svg>
          </button>
        </div>
        {open && (
          <nav className="flex flex-col gap-1 border-t border-slate-200 bg-white px-4 py-3 md:hidden" aria-label="Mobile">
            {NAV_LINKS.map(({ to, label, end }) => (
              <NavLink key={to} to={to} end={end} className={linkClass} onClick={() => setOpen(false)}>
                {label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<Analyze />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/model" element={<Model />} />
          <Route path="/compare" element={<Compare />} />
        </Routes>
      </main>
    </div>
  );
}
