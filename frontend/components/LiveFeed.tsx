"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CLASS_COLOR, CLASS_DOT, type FeedEvent } from "@/lib/types";

const SPEEDS = [
  { label: "1x", ms: 260 },
  { label: "4x", ms: 65 },
  { label: "20x", ms: 14 },
];

/**
 * Replays a recorded run. The events, the predictions and the log lines are all real output
 * from the Python pipeline; only the pacing is synthetic, so the page needs no backend.
 */
export function LiveFeed({ events }: { events: FeedEvent[] }) {
  const [cursor, setCursor] = useState(24);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState<FeedEvent | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setCursor((c) => (c >= events.length ? 24 : c + 1));
    }, SPEEDS[speed].ms);
    return () => clearInterval(id);
  }, [running, speed, events.length]);

  const visible = useMemo(
    () => events.slice(Math.max(0, cursor - 26), cursor).reverse(),
    [events, cursor],
  );

  const seen = events.slice(0, cursor);
  const escalated = seen.filter((e) => e.action === "escalate" || e.resolved).length;
  const corrected = seen.filter((e) => e.resolved && e.resolved !== e.predicted).length;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div>
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <button
            onClick={() => setRunning((r) => !r)}
            className="rounded border border-slate-700 px-3 py-1 text-xs font-medium text-slate-200 transition hover:border-slate-500 hover:bg-slate-800"
          >
            {running ? "Pause" : "Play"}
          </button>
          <div className="flex gap-1" role="group" aria-label="Replay speed">
            {SPEEDS.map((s, i) => (
              <button
                key={s.label}
                onClick={() => setSpeed(i)}
                aria-pressed={speed === i}
                className={`rounded px-2 py-1 font-mono text-xs transition ${
                  speed === i
                    ? "bg-sky-500/15 text-sky-300 ring-1 ring-inset ring-sky-500/40"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
          <p className="ml-auto font-mono text-xs tabular-nums text-slate-500">
            {cursor.toLocaleString()} / {events.length.toLocaleString()} events ·{" "}
            {escalated.toLocaleString()} escalated · {corrected.toLocaleString()} corrected
          </p>
        </div>

        <div
          ref={listRef}
          className="h-[26rem] overflow-hidden rounded-lg border border-slate-800 bg-slate-950/60 font-mono text-xs"
        >
          {visible.map((e, i) => {
            const shown = e.resolved ?? e.predicted;
            const wrong = shown !== e.truth;
            return (
              <button
                key={`${e.ts}-${e.service}-${i}`}
                onClick={() => setSelected(e)}
                className={`flex w-full items-center gap-3 border-b border-slate-900 px-3 py-1.5 text-left transition hover:bg-slate-900 ${
                  i === 0 ? "animate-[fadeIn_240ms_ease-out]" : ""
                } ${selected?.ts === e.ts && selected?.service === e.service ? "bg-slate-900" : ""}`}
              >
                <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${CLASS_DOT[shown]}`} />
                <span className="w-12 shrink-0 tabular-nums text-slate-600">
                  {e.ts.toFixed(0)}
                </span>
                <span className="w-32 shrink-0 truncate text-slate-400">{e.service}</span>
                <span className="w-16 shrink-0 tabular-nums text-slate-300">
                  {e.latency_ms.toFixed(0)}ms
                </span>
                <span className="w-14 shrink-0 tabular-nums text-slate-500">
                  {(e.error_rate * 100).toFixed(1)}%
                </span>
                <span className={`w-40 shrink-0 truncate ${CLASS_COLOR[shown]}`}>{shown}</span>
                {e.resolved && e.resolved !== e.predicted ? (
                  <span className="shrink-0 rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-300 ring-1 ring-inset ring-amber-500/30">
                    log corrected
                  </span>
                ) : null}
                {wrong ? (
                  <span className="shrink-0 text-[10px] text-rose-400/70">
                    truth {e.truth}
                  </span>
                ) : null}
                <span className="ml-auto shrink-0 truncate text-slate-600">
                  {e.breach ? "SLO breach" : ""}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <aside className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
        {selected ? (
          <div className="space-y-3 text-sm">
            <div>
              <p className="text-xs uppercase tracking-wider text-slate-500">Event</p>
              <p className="font-mono text-slate-200">
                {selected.service} <span className="text-slate-600">@{selected.ts.toFixed(0)}s</span>
              </p>
              <p className="font-mono text-xs text-slate-500">{selected.version}</p>
            </div>

            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs">
              {[
                ["latency", `${selected.latency_ms.toFixed(0)}ms`],
                ["errors", `${(selected.error_rate * 100).toFixed(2)}%`],
                ["cpu", `${selected.cpu_pct.toFixed(0)}%`],
                ["mem", `${selected.mem_pct.toFixed(0)}%`],
                ["rps", selected.rps.toFixed(0)],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <dt className="text-slate-600">{k}</dt>
                  <dd className="tabular-nums text-slate-300">{v}</dd>
                </div>
              ))}
            </dl>

            <div>
              <p className="text-xs uppercase tracking-wider text-slate-500">Log line</p>
              <p className="mt-1 break-words rounded bg-slate-900 p-2 font-mono text-[11px] leading-relaxed text-slate-400">
                {selected.log}
              </p>
            </div>

            <div className="space-y-1.5 border-t border-slate-800 pt-3 text-xs">
              <Row label="metrics say" value={selected.predicted} cls />
              <Row label="conformal set" value={selected.set.join(", ")} />
              {selected.resolved ? (
                <Row label="log says" value={selected.resolved} cls />
              ) : null}
              {selected.remediation ? (
                <Row label="remediation" value={selected.remediation} />
              ) : null}
              <Row label="truth" value={selected.truth} cls />
            </div>

            {selected.evidence ? (
              <div>
                <p className="text-xs uppercase tracking-wider text-slate-500">Evidence</p>
                <p className="mt-1 font-mono text-[11px] text-amber-300/90">
                  &ldquo;{selected.evidence}&rdquo;
                </p>
              </div>
            ) : null}
          </div>
        ) : (
          <p className="text-sm leading-relaxed text-slate-500">
            Pick an event to see its metrics, its log line, and how the metrics and the log
            disagreed. Rows tagged{" "}
            <span className="rounded bg-amber-500/10 px-1 text-[10px] text-amber-300">
              log corrected
            </span>{" "}
            are ones the classifier got wrong and reading the log fixed.
          </p>
        )}
      </aside>
    </div>
  );
}

function Row({ label, value, cls = false }: { label: string; value: string; cls?: boolean }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-slate-600">{label}</span>
      <span className={`font-mono ${cls ? CLASS_COLOR[value] ?? "text-slate-300" : "text-slate-300"}`}>
        {value}
      </span>
    </div>
  );
}
