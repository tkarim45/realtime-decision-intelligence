import fs from "node:fs/promises";
import path from "node:path";
import { LiveFeed } from "@/components/LiveFeed";
import { Bar, Panel, Stat } from "@/components/primitives";
import { CLASS_COLOR, CLASS_DOT, type Snapshot } from "@/lib/types";

async function load(): Promise<Snapshot> {
  const raw = await fs.readFile(path.join(process.cwd(), "public/snapshot.json"), "utf8");
  return JSON.parse(raw) as Snapshot;
}

export default async function Page() {
  const s = await load();
  const bestBaseline = Math.max(
    ...s.policies.filter((p) => !p.name.startsWith("causal")).map((p) => p.value),
  );
  const causal = s.policies.find((p) => p.name.startsWith("causal, T"));
  const maxAbs = Math.max(...s.policies.map((p) => Math.abs(p.value)));
  const tight = s.policies.find((p) => p.name.includes("10%"));
  const loose = s.policies.find((p) => p.name.includes("25%"));

  return (
    <main className="mx-auto max-w-[1400px] px-5 py-8 lg:px-8">
      <header className="mb-8">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">
            realtime-decision-intelligence
          </h1>
          <span className="rounded bg-emerald-500/10 px-2 py-0.5 font-mono text-[11px] text-emerald-400 ring-1 ring-inset ring-emerald-500/30">
            {s.hotPath.p99.toFixed(3)}ms p99
          </span>
          <a
            href="https://github.com/tkarim45/realtime-decision-intelligence"
            className="ml-auto text-xs text-slate-500 underline-offset-4 hover:text-slate-300 hover:underline"
          >
            source
          </a>
        </div>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
          Detects incidents in service telemetry, works out which ones are worth acting on, and
          picks the remediation. Everything below is a recorded run of the Python pipeline over{" "}
          {s.generatedFrom.events.toLocaleString()} synthetic events across{" "}
          {s.generatedFrom.services} services. The replay is paced for reading; the numbers are
          not simulated.
        </p>
      </header>

      <dl className="mb-6 grid grid-cols-2 gap-4 rounded-lg border border-slate-800 bg-slate-900/40 p-5 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="hot path p50" value={s.hotPath.p50.toFixed(3)} unit="ms" />
        <Stat label="hot path p99" value={s.hotPath.p99.toFixed(3)} unit="ms" tone="good" />
        <Stat label="throughput" value={s.hotPath.throughput.toLocaleString()} unit="ev/s" />
        <Stat
          label="macro-F1"
          value={s.reasoning.macroF1After.toFixed(3)}
          tone="good"
          note={`${s.reasoning.macroF1Before.toFixed(3)} before log reading`}
        />
        <Stat
          label="policy value"
          value={causal ? causal.value.toFixed(1) : "n/a"}
          tone="good"
          note={`vs ${bestBaseline.toFixed(1)} best baseline`}
        />
        <Stat
          label="events acted on"
          value={`${((s.pipeline.acted / s.pipeline.processed) * 100).toFixed(1)}%`}
          note={`${s.pipeline.acted.toLocaleString()} of ${s.pipeline.processed.toLocaleString()}`}
        />
      </dl>

      <Panel
        title="Live feed"
        hint="each row is one event, scored and routed. Click to inspect."
        className="mb-6"
      >
        <LiveFeed events={s.feed} />
      </Panel>

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <Panel title="Reading the log line" hint="for the events the metrics cannot separate">
          <p className="mb-4 text-sm leading-relaxed text-slate-400">
            A retry storm and a bad release move CPU, latency and errors together, so they
            overlap on every metric the scorer sees. The classifier is confidently wrong rather
            than uncertain, which is why widening the conformal set does not help. The log line
            names the failing upstream.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wider text-slate-500">
                  <th className="pb-2 font-medium">stage</th>
                  <th className="pb-2 text-right font-medium">macro-F1</th>
                  <th className="pb-2 text-right font-medium">dep_failure recall</th>
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums">
                <tr className="border-b border-slate-900">
                  <td className="py-2 font-sans text-slate-400">classifier alone</td>
                  <td className="py-2 text-right text-slate-300">
                    {s.reasoning.macroF1Before.toFixed(3)}
                  </td>
                  <td className="py-2 text-right text-rose-400">
                    {s.reasoning.depRecallBefore.toFixed(2)}
                  </td>
                </tr>
                <tr>
                  <td className="py-2 font-sans text-slate-400">plus log reading</td>
                  <td className="py-2 text-right text-emerald-400">
                    {s.reasoning.macroF1After.toFixed(3)}
                  </td>
                  <td className="py-2 text-right text-emerald-400">
                    {s.reasoning.depRecallAfter.toFixed(2)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Only {s.reasoning.escalatedPct}% of events are escalated. Reading every line would
            cost more than it returns.
          </p>
        </Panel>

        <Panel title="Choosing the action" hint="breaches avoided, net of what the actions cost">
          <p className="mb-4 text-sm leading-relaxed text-slate-400">
            Every remediation costs something and the wrong one has negative effect, so the
            question is incremental rather than &ldquo;how alarming is this&rdquo;. The
            risk-threshold baselines are strong: they apply the correct runbook action for
            whatever class was predicted.
          </p>
          <ul className="space-y-2.5">
            {s.policies.map((p) => {
              const isCausal = p.name.startsWith("causal, T");
              const negative = p.value < 0;
              return (
                <li key={p.name}>
                  <div className="flex items-baseline justify-between gap-3 text-sm">
                    <span className={isCausal ? "text-slate-100" : "text-slate-400"}>
                      {p.name}
                    </span>
                    <span
                      className={`shrink-0 font-mono tabular-nums ${
                        negative
                          ? "text-rose-400"
                          : isCausal
                            ? "text-emerald-400"
                            : "text-slate-300"
                      }`}
                    >
                      {p.value.toFixed(1)}
                      <span className="ml-2 text-xs text-slate-600">
                        {p.treated.toLocaleString()} treated
                      </span>
                    </span>
                  </div>
                  <div className="mt-1">
                    <Bar
                      value={Math.abs(p.value)}
                      max={maxAbs}
                      className={
                        negative ? "bg-rose-500" : isCausal ? "bg-emerald-500" : "bg-slate-600"
                      }
                    />
                  </div>
                </li>
              );
            })}
          </ul>
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            The same risk rule scores {tight ? tight.value.toFixed(1) : "n/a"} at the top 10% and{" "}
            {loose ? loose.value.toFixed(1) : "n/a"} at the top 25%. Different cutoff, opposite
            sign. The causal policy picks its own operating point and treats fewer events than
            the threshold it beats.
          </p>
        </Panel>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Panel title="Per class" hint="test split" className="lg:col-span-2">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wider text-slate-500">
                  <th className="pb-2 font-medium">class</th>
                  <th className="pb-2 text-right font-medium">precision</th>
                  <th className="pb-2 text-right font-medium">recall</th>
                  <th className="pb-2 text-right font-medium">F1</th>
                  <th className="pb-2 text-right font-medium">support</th>
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums">
                {Object.entries(s.classifier.perClass).map(([cls, m]) => (
                  <tr key={cls} className="border-b border-slate-900 last:border-0">
                    <td className="py-2">
                      <span className="flex items-center gap-2 font-sans">
                        <span className={`h-1.5 w-1.5 rounded-full ${CLASS_DOT[cls]}`} />
                        <span className={CLASS_COLOR[cls]}>{cls}</span>
                      </span>
                    </td>
                    <td className="py-2 text-right text-slate-300">{m.precision.toFixed(2)}</td>
                    <td
                      className={`py-2 text-right ${m.recall < 0.7 ? "text-rose-400" : "text-slate-300"}`}
                    >
                      {m.recall.toFixed(2)}
                    </td>
                    <td className="py-2 text-right text-slate-300">{m.f1.toFixed(2)}</td>
                    <td className="py-2 text-right text-slate-600">
                      {m.support.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            dependency_failure is the weak class on metrics alone, and it is mistaken for
            bad_deploy. That is the confusion the log-reading step exists to resolve.
          </p>
        </Panel>

        <Panel title="Drift" hint="windows of 600 events">
          <p className="mb-4 text-sm leading-relaxed text-slate-400">
            An outage wants remediation, drift wants retraining. A monitor that confuses them
            retrains on the outage.
          </p>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">clean, has incidents</span>
                <span className="font-mono tabular-nums text-emerald-400">
                  {s.drift.cleanRate}%
                </span>
              </div>
              <div className="mt-1">
                <Bar value={s.drift.cleanRate} max={100} className="bg-emerald-500" />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">shifted baseline</span>
                <span className="font-mono tabular-nums text-sky-400">{s.drift.driftedRate}%</span>
              </div>
              <div className="mt-1">
                <Bar value={s.drift.driftedRate} max={100} className="bg-sky-500" />
              </div>
            </div>
          </div>
          <p className="mt-4 border-t border-slate-800 pt-3 text-xs leading-relaxed text-slate-500">
            Naive PSI against a reference that includes incidents alerts on {s.drift.naiveClean}{" "}
            clean windows against {s.drift.naiveDrifted} drifted ones, which is no discrimination
            at all. Building the reference from healthy ticks only and requiring the median to
            move as well is what separates them.
          </p>
        </Panel>
      </div>

      <footer className="mt-10 border-t border-slate-800 pt-5 text-xs leading-relaxed text-slate-500">
        <p className="max-w-3xl">
          <strong className="text-slate-400">Honest caveats.</strong> The telemetry is synthetic,
          generated with labelled incidents so every claim is measurable against ground truth.
          Latency figures come from one laptop and vary with load. The log-reading numbers come
          from a deterministic offline reader that matches the phrases the generator writes, so
          treat them as proof the wiring works rather than as a result about any language model.
        </p>
      </footer>
    </main>
  );
}
