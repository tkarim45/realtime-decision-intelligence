export type IncidentClass =
  | "normal"
  | "memory_leak"
  | "dependency_failure"
  | "traffic_spike"
  | "bad_deploy";

export interface FeedEvent {
  ts: number;
  service: string;
  version: string;
  latency_ms: number;
  error_rate: number;
  cpu_pct: number;
  mem_pct: number;
  rps: number;
  log: string;
  truth: IncidentClass;
  breach: boolean;
  predicted: IncidentClass;
  set: string[];
  action: "act" | "escalate";
  resolved?: IncidentClass;
  evidence?: string;
  explanation?: string;
  remediation?: string;
}

export interface Snapshot {
  generatedFrom: { ticks: number; seed: number; events: number; services: number };
  hotPath: { p50: number; p95: number; p99: number; throughput: number };
  classifier: {
    macroF1: number;
    perClass: Record<string, { precision: number; recall: number; f1: number; support: number }>;
    confusion: number[][];
    labels: string[];
  };
  reasoning: {
    macroF1Before: number;
    macroF1After: number;
    depRecallBefore: number;
    depRecallAfter: number;
    escalatedPct: number;
  };
  policies: { name: string; value: number; treated: number }[];
  actionCost: Record<string, number>;
  drift: { cleanRate: number; driftedRate: number; naiveClean: number; naiveDrifted: number };
  pipeline: {
    processed: number;
    escalated: number;
    reclassified: number;
    anomalies: number;
    acted: number;
    policyValue: number;
    actions: Record<string, number>;
    incidents: Record<string, number>;
  };
  feed: FeedEvent[];
}

/** One colour per incident class, used everywhere so a class always reads the same. */
export const CLASS_COLOR: Record<string, string> = {
  normal: "text-slate-400",
  memory_leak: "text-amber-400",
  dependency_failure: "text-rose-400",
  traffic_spike: "text-sky-400",
  bad_deploy: "text-violet-400",
};

export const CLASS_DOT: Record<string, string> = {
  normal: "bg-slate-500",
  memory_leak: "bg-amber-400",
  dependency_failure: "bg-rose-400",
  traffic_spike: "bg-sky-400",
  bad_deploy: "bg-violet-400",
};

export const REMEDIATION: Record<string, string> = {
  memory_leak: "restart",
  dependency_failure: "failover",
  traffic_spike: "scale_out",
  bad_deploy: "rollback",
};
