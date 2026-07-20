"""Durable event log with consumer groups. Redis-Streams semantics, file-backed.

Same five-method contract as Redis Streams
(the exact shape of Redis XADD/XREADGROUP/XACK/XAUTOCLAIM), so swapping in real Redis is one
adapter, but implemented on a local append-only log, which is what makes at-least-once
delivery *provable* on a laptop instead of merely claimed.

  append(event), durable write (JSONL), monotonic offsets
  claim(group, consumer, n), deliver up to n unacked events; they enter the group's
                                   PENDING set (the in-flight ledger)
  ack(group, offset), remove from PENDING; the event is done
  redeliver(group, older_than_s), re-queue pending entries whose consumer went quiet
                                   (crashed before ack) → AT-LEAST-ONCE
  lag(group), backlog + in-flight depth

Two changes from that version, both because throughput is measured against an SLO here and
the durability claim has to survive a real kill:

  * The file handle stays open. Reopening per append made the write path an
    open/write/close syscall trio per event, which would put a floor under throughput
    that has nothing to do with the model.
  * `fsync=True` is a real option. The original said "fsync-able" but never
    called it, so a power-cut would have lost the page cache. Off by default (it costs ~an
    order of magnitude); on for the durability test, which is the one place the claim is
    actually being made.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass


@dataclass
class Pending:
    offset: int
    consumer: str
    delivered_at: float
    deliveries: int = 1


class Broker:
    def __init__(self, path: str, fsync: bool = False) -> None:
        self.path = path
        self.fsync = fsync
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self._groups: dict[str, dict] = {}  # group -> {"next": int, "pending": {offset: Pending}}
        if os.path.exists(path):
            with open(path) as f:
                self._events = [json.loads(line) for line in f if line.strip()]
        # Deliberately long-lived, released by close()/__exit__, a context manager here would
        # mean reopening per append, which is the syscall cost this class exists to avoid.
        self._fh = open(path, "a")  # noqa: SIM115

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Broker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- producer ----
    def append(self, event: dict) -> int:
        with self._lock:
            offset = len(self._events)
            rec = {"offset": offset, **event}
            self._events.append(rec)
            self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())  # survives power loss, not just process death
            return offset

    # ---- consumer group ----
    def _group(self, name: str) -> dict:
        return self._groups.setdefault(name, {"next": 0, "pending": {}})

    def claim(self, group: str, consumer: str, n: int = 10) -> list[dict]:
        with self._lock:
            g = self._group(group)
            out = []
            while len(out) < n and g["next"] < len(self._events):
                off = g["next"]
                g["next"] += 1
                g["pending"][off] = Pending(off, consumer, time.monotonic())
                out.append(self._events[off])
            return out

    def ack(self, group: str, offset: int) -> None:
        with self._lock:
            self._group(group)["pending"].pop(offset, None)

    def redeliver(self, group: str, consumer: str, older_than_s: float = 1.0,
                  n: int = 10) -> list[dict]:
        """Claim pending entries stuck longer than older_than_s, their consumer died."""
        now = time.monotonic()
        with self._lock:
            g = self._group(group)
            stale = [p for p in g["pending"].values()
                     if now - p.delivered_at >= older_than_s][:n]
            out = []
            for p in stale:
                p.consumer = consumer
                p.delivered_at = now
                p.deliveries += 1
                out.append(self._events[p.offset] | {"_redelivery": p.deliveries})
            return out

    def lag(self, group: str) -> dict:
        with self._lock:
            g = self._group(group)
            return {"appended": len(self._events), "consumed_next": g["next"],
                    "backlog": len(self._events) - g["next"], "pending": len(g["pending"])}
