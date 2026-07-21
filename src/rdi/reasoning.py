"""Log reading for the events the metrics can't classify.

The hot path gets `dependency_failure` right about half the time and mistakes the rest for
`bad_deploy`. That isn't a tuning problem. A retry storm and a bad deploy push CPU, latency and
error rate up together, so the two classes overlap on every metric the scorer sees, and the
model is confidently wrong rather than uncertain (it assigns ~0.97 to the wrong class, which is
why conformal sets don't hedge it either).

What separates them is the log line. `upstream session-store 503; retrying (7/9 attempts)`
names a failing dependency. `NullPointerException in RecommendHandler.validate (v1.5.3)` names
a bad release. Same metrics, different remediation: failover versus rollback, and picking wrong
costs an outage either way.

So this layer reads text for the events worth spending time on, and it runs off the hot path.
A model call takes hundreds of milliseconds against a sub-millisecond budget, so it only ever
sees events the fast path already flagged.

Two readers ship. `MockReader` matches on the phrases the generator emits, needs no network,
and is what the tests and any CI run use. It is a fixture, not a result: it scores near-perfect
by construction because it was written against the same vocabulary the generator writes, so
treat its accuracy as proof the wiring works and nothing more. `ClaudeReader` calls a real
model and is the number worth quoting.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from rdi.events import REMEDIATIONS

BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

CLASSES = ["memory_leak", "dependency_failure", "traffic_spike", "bad_deploy", "normal"]


@dataclass
class Verdict:
    """What the reader concluded, plus the text it leaned on."""
    incident: str
    remediation: str
    explanation: str
    evidence: str = ""
    grounded: bool = False
    source: str = "mock"


@dataclass
class ReasonerStats:
    calls: int = 0
    failures: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latencies_ms: list[float] = field(default_factory=list)


PROMPT = """You are triaging one alert from a service reliability system.

Metrics for {service} at t={ts:.0f}s:
  latency_ms {latency_ms:.0f} (baseline multiple {lat_ratio:.1f}x)
  error_rate {error_rate:.3f}
  cpu_pct {cpu_pct:.0f} (baseline multiple {cpu_ratio:.1f}x)
  mem_pct {mem_pct:.0f}
  rps {rps:.0f}
  version {version}

Log line:
  {log}

The metrics alone cannot separate a failing upstream dependency from a bad release: both raise
latency, errors and CPU together. The log line is the deciding evidence.

Classify as exactly one of: memory_leak, dependency_failure, traffic_spike, bad_deploy, normal.

Reply with JSON only:
{{"incident": "<class>", "evidence": "<the exact phrase from the log that decided it>",
  "explanation": "<one sentence, cite a real metric value>"}}"""


def _context(event: dict, features: dict | None = None) -> dict:
    f = features or {}
    return {
        "service": event["service"], "ts": event["ts"], "log": event["log"],
        "latency_ms": event["latency_ms"], "error_rate": event["error_rate"],
        "cpu_pct": event["cpu_pct"], "mem_pct": event["mem_pct"], "rps": event["rps"],
        "version": event["version"],
        "lat_ratio": f.get("latency_over_baseline", 1.0),
        "cpu_ratio": f.get("cpu_over_baseline", 1.0),
    }


def is_grounded(explanation: str, event: dict) -> bool:
    """Does the explanation quote a number that actually appears in the event?

    A fluent explanation that invents its numbers is worse than no explanation, because it
    reads as evidence. This only checks that some real value shows up, which is a floor rather
    than a guarantee.
    """
    nums = {
        f"{event['latency_ms']:.0f}", f"{event['cpu_pct']:.0f}", f"{event['mem_pct']:.0f}",
        f"{event['rps']:.0f}", f"{event['error_rate']:.3f}", f"{event['error_rate']:.2f}",
        str(int(event["latency_ms"])), event["version"],
    }
    return any(n and n in explanation for n in nums)


class MockReader:
    """Deterministic phrase matching. No network, no key, no cost.

    Written against the vocabulary in `events._log_line`, so it is close to perfect on this
    generator by construction. That makes it a wiring fixture, not evidence about real models.
    """

    source = "mock"

    PATTERNS = [
        ("dependency_failure", re.compile(
            r"upstream \S+ (?:timeout|503)|connection pool exhausted|retrying \(")),
        ("bad_deploy", re.compile(r"Exception in \w+\.\w+|returning 500")),
        ("memory_leak", re.compile(r"GC pause|heap at \d+%|old-gen")),
        ("traffic_spike", re.compile(r"queue depth|worker pool saturated|requests shed")),
    ]

    def __init__(self) -> None:
        self.stats = ReasonerStats()

    def read(self, event: dict, features: dict | None = None) -> Verdict:
        self.stats.calls += 1
        log = event["log"]
        for cls, pat in self.PATTERNS:
            m = pat.search(log)
            if m:
                expl = (f"{event['service']} shows {log.split(';')[0].strip()} "
                        f"at {event['latency_ms']:.0f}ms latency.")
                return Verdict(cls, REMEDIATIONS[cls], expl, m.group(0),
                               is_grounded(expl, event), self.source)
        expl = f"{event['service']} logged a routine request at {event['latency_ms']:.0f}ms."
        return Verdict("normal", "none", expl, "", is_grounded(expl, event), self.source)


class ClaudeReader:
    """Real model. Bedrock when AWS creds are present, otherwise the Anthropic API."""

    def __init__(self, model: str | None = None, region: str | None = None,
                 max_tokens: int = 300) -> None:
        self.stats = ReasonerStats()
        self.max_tokens = max_tokens
        if os.getenv("AWS_ACCESS_KEY_ID"):
            from anthropic import AnthropicBedrock
            self._client = AnthropicBedrock(
                aws_region=region or os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            self.model = model or BEDROCK_MODEL
            self.source = "bedrock"
        elif os.getenv("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            self._client = Anthropic()
            self.model = model or ANTHROPIC_MODEL
            self.source = "anthropic"
        else:
            # Failing here beats constructing a client that raises on the first call, halfway
            # through an eval run.
            raise RuntimeError(
                "no credentials: set AWS_ACCESS_KEY_ID (Bedrock) or ANTHROPIC_API_KEY. "
                "Use MockReader for offline runs."
            )

    def read(self, event: dict, features: dict | None = None) -> Verdict:
        import time
        self.stats.calls += 1
        t0 = time.perf_counter()
        try:
            r = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(**_context(event, features))}],
            )
            self.stats.latencies_ms.append((time.perf_counter() - t0) * 1000)
            self.stats.tokens_in += r.usage.input_tokens
            self.stats.tokens_out += r.usage.output_tokens
            text = r.content[0].text.strip()
        except Exception as exc:
            self.stats.failures += 1
            return Verdict("normal", "none", f"reader failed: {type(exc).__name__}", "",
                           False, self.source)

        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0) if m else text)
        except Exception:
            self.stats.failures += 1
            return Verdict("normal", "none", f"unparseable reply: {text[:80]}", "",
                           False, self.source)

        cls = data.get("incident", "normal")
        if cls not in CLASSES:
            cls = "normal"
        expl = str(data.get("explanation", ""))
        return Verdict(cls, REMEDIATIONS.get(cls, "none"), expl,
                       str(data.get("evidence", "")), is_grounded(expl, event), self.source)


def resolve(events: list[dict], predictions: list[str], reader, only: set[str] | None = None,
            features: list[dict] | None = None) -> list[str]:
    """Send the events worth a second look to the reader, keep the rest as scored.

    `only` names the hot-path predictions that are worth escalating. Everything else keeps its
    fast-path label, which is the point: reading every event would cost more than it returns.
    """
    only = only or {"bad_deploy", "dependency_failure"}
    out = []
    for i, (e, pred) in enumerate(zip(events, predictions, strict=True)):
        if pred in only:
            out.append(reader.read(e, (features or [None] * len(events))[i]).incident)
        else:
            out.append(pred)
    return out
