import type { ReactNode } from "react";

export function Panel({
  title,
  hint,
  children,
  className = "",
}: {
  title: string;
  hint?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-lg border border-slate-800 bg-slate-900/40 ${className}`}
      aria-label={title}
    >
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-slate-800 px-4 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-300">
          {title}
        </h2>
        {hint ? <p className="text-xs text-slate-500">{hint}</p> : null}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  unit,
  note,
  tone = "default",
}: {
  label: string;
  value: string | number;
  unit?: string;
  note?: string;
  tone?: "default" | "good" | "bad";
}) {
  const toneClass =
    tone === "good" ? "text-emerald-400" : tone === "bad" ? "text-rose-400" : "text-slate-100";
  return (
    <div className="min-w-0">
      <dt className="truncate text-xs uppercase tracking-wider text-slate-500">{label}</dt>
      <dd className={`mt-1 font-mono text-2xl tabular-nums ${toneClass}`}>
        {value}
        {unit ? <span className="ml-1 text-sm text-slate-500">{unit}</span> : null}
      </dd>
      {note ? <p className="mt-1 text-xs leading-snug text-slate-500">{note}</p> : null}
    </div>
  );
}

/** Horizontal bar. `max` keeps bars comparable across rows. */
export function Bar({
  value,
  max,
  className = "bg-sky-500",
}: {
  value: number;
  max: number;
  className?: string;
}) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
      <div className={`h-full rounded-full ${className}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Pill({ children, tone = "slate" }: { children: ReactNode; tone?: string }) {
  return (
    <span
      className={`inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[11px] text-${tone}-300 ring-1 ring-inset ring-${tone}-500/30`}
    >
      {children}
    </span>
  );
}
