"""The stream consumer — claim a batch, compute features, ack.

Ack-AFTER-processing is the entire at-least-once guarantee. If the consumer dies mid-batch,
the unacked events are still in the broker's pending set and get redelivered to whoever picks
up next, so nothing is lost. Ack-before-processing would be faster and would silently drop
every event in flight at the moment of the crash — which is the bug this ordering exists to
not have. The trade is duplicates on redelivery, which are countable and deduped by offset.

There is no model here yet: M1 proves the stream is durable and the features are train=serve.
Scoring arrives in M2 and slots in where `_process` computes features.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rdi.broker import Broker
from rdi.features import OnlineFeatures


@dataclass
class Results:
    processed: int = 0
    duplicates: int = 0
    features: list = field(default_factory=list)   # (offset, feature vector)
    latencies_ms: list = field(default_factory=list)


class Consumer:
    def __init__(self, name: str, broker: Broker, group: str,
                 die_after: int | None = None) -> None:
        self.name = name
        self.broker = broker
        self.group = group
        self.die_after = die_after  # crash simulation: claim, then return without acking
        self.features = OnlineFeatures()
        self.results = Results()
        self._seen_offsets: set[int] = set()

    def _process(self, event: dict) -> None:
        feats = self.features.update_and_extract(event)
        self.results.features.append((event["offset"], feats))
        self.results.processed += 1

    def run(self, batch: int = 32, idle_rounds: int = 3, redeliver_after_s: float = 0.2):
        idle = 0
        while idle < idle_rounds:
            events = self.broker.claim(self.group, self.name, batch)
            if not events:
                events = self.broker.redeliver(self.group, self.name,
                                               older_than_s=redeliver_after_s, n=batch)
            if not events:
                idle += 1
                time.sleep(0.05)
                continue
            idle = 0
            for e in events:
                if self.die_after is not None and self.results.processed >= self.die_after:
                    return self.results  # simulated crash: claimed but never acked
                t0 = time.perf_counter()
                if e["offset"] in self._seen_offsets:
                    # At-least-once means duplicates are expected, not exceptional. Dedup by
                    # offset so redelivery can't double-count or corrupt the feature windows.
                    self.results.duplicates += 1
                else:
                    self._seen_offsets.add(e["offset"])
                    self._process(e)
                self.broker.ack(self.group, e["offset"])
                self.results.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return self.results
