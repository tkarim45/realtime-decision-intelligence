"""Tests for the durable log and at-least-once delivery.

The headline is test_crash_mid_stream_loses_zero_events. The rest exist so that when it
passes, it passes for the documented reason rather than by accident.
"""
from __future__ import annotations

import json
import time

import pytest

from rdi.broker import Broker
from rdi.consumer import Consumer
from rdi.events import generate


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "stream.jsonl")


def _filled(path: str, n_ticks: int = 30, **kw) -> Broker:
    b = Broker(path, **kw)
    for e in generate(n_ticks=n_ticks, seed=13):
        b.append(e)
    return b


# ---- log durability ----

def test_append_returns_monotonic_offsets(log_path):
    with Broker(log_path) as b:
        offsets = [b.append(e) for e in generate(n_ticks=5)]
    assert offsets == list(range(len(offsets)))


def test_events_survive_a_fresh_process(log_path):
    """Durability means a *new* Broker over the same file sees everything."""
    with _filled(log_path, n_ticks=10) as b:
        n = b.lag("g")["appended"]
    with Broker(log_path) as reopened:
        assert reopened.lag("g")["appended"] == n
        assert len(reopened.claim("g", "c", 10_000)) == n


def test_log_is_valid_jsonl_with_offsets(log_path):
    with _filled(log_path, n_ticks=5):
        pass
    with open(log_path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    assert [r["offset"] for r in recs] == list(range(len(recs)))
    assert all("service" in r for r in recs)


def test_fsync_mode_still_round_trips(log_path):
    """fsync is the one place the durability claim is actually made, so it has to work."""
    with _filled(log_path, n_ticks=5, fsync=True) as b:
        n = b.lag("g")["appended"]
    with Broker(log_path) as reopened:
        assert reopened.lag("g")["appended"] == n


# ---- consumer groups ----

def test_claim_moves_events_into_pending(log_path):
    with _filled(log_path, n_ticks=10) as b:
        total = b.lag("g")["appended"]
        got = b.claim("g", "c1", 5)
        assert len(got) == 5
        assert b.lag("g")["pending"] == 5
        assert b.lag("g")["backlog"] == total - 5


def test_ack_clears_pending(log_path):
    with _filled(log_path, n_ticks=10) as b:
        for e in b.claim("g", "c1", 5):
            b.ack("g", e["offset"])
        assert b.lag("g")["pending"] == 0


def test_groups_are_independent(log_path):
    """Two groups each see the whole stream, that's what makes fan-out possible."""
    with _filled(log_path, n_ticks=10) as b:
        total = b.lag("g")["appended"]
        assert len(b.claim("scoring", "c", 10_000)) == total
        assert len(b.claim("drift", "c", 10_000)) == total


def test_claim_never_delivers_the_same_offset_twice_to_a_group(log_path):
    with _filled(log_path, n_ticks=20) as b:
        seen = [e["offset"] for _ in range(5) for e in b.claim("g", "c", 25)]
    assert len(seen) == len(set(seen))


# ---- at-least-once ----

def test_pending_entries_are_redelivered_when_a_consumer_goes_quiet(log_path):
    with _filled(log_path, n_ticks=10) as b:
        b.claim("g", "dead", 4)  # claimed, never acked, the consumer died here
        time.sleep(0.15)
        rescued = b.redeliver("g", "rescue", older_than_s=0.1, n=10)
        assert len(rescued) == 4
        assert all(e["_redelivery"] == 2 for e in rescued)


def test_redeliver_leaves_live_consumers_alone(log_path):
    """A consumer that is merely slow must not have its work stolen."""
    with _filled(log_path, n_ticks=10) as b:
        b.claim("g", "busy", 4)
        assert b.redeliver("g", "thief", older_than_s=10.0) == []


def test_acked_events_are_never_redelivered(log_path):
    with _filled(log_path, n_ticks=10) as b:
        for e in b.claim("g", "c", 4):
            b.ack("g", e["offset"])
        time.sleep(0.15)
        assert b.redeliver("g", "rescue", older_than_s=0.1) == []


# ---- the S1 claim ----

def test_crash_mid_stream_loses_zero_events(log_path):
    """S1: kill a consumer mid-stream, restart, assert 0 events lost.

    The crash consumer claims a batch and dies partway through without acking. Its in-flight
    events sit in PENDING; the rescue consumer redelivers and finishes them. Processed counts
    across both must sum to the full stream.
    """
    with _filled(log_path, n_ticks=50) as b:
        total = b.lag("g")["appended"]
        crash = Consumer("crash", b, "g", die_after=80)
        crash.run()
        assert crash.results.processed == 80, "crash consumer did not die where expected"
        assert b.lag("g")["pending"] > 0, "nothing was in flight, the crash proved nothing"

        time.sleep(0.25)
        rescue = Consumer("rescue", b, "g")
        r2 = rescue.run()

        assert crash.results.processed + r2.processed == total  # zero loss
        assert b.lag("g")["pending"] == 0                       # nothing left in flight


def test_every_offset_is_processed_exactly_once_across_a_crash(log_path):
    """Stronger than the count: counts can tie while the *identities* are wrong."""
    with _filled(log_path, n_ticks=50) as b:
        total = b.lag("g")["appended"]
        crash = Consumer("crash", b, "g", die_after=80)
        crash.run()
        time.sleep(0.25)
        rescue = Consumer("rescue", b, "g")
        rescue.run()

    done = [o for c in (crash, rescue) for o, _ in c.results.features]
    assert sorted(done) == list(range(total))


def test_duplicates_are_deduped_not_double_counted(log_path):
    """The at-least-once trade: redelivery brings duplicates, and they must be absorbed."""
    with _filled(log_path, n_ticks=30) as b:
        total = b.lag("g")["appended"]
        b.claim("g", "ghost", 20)  # in flight, never acked
        time.sleep(0.25)
        c = Consumer("worker", b, "g")
        c.run()
        assert c.results.processed == total
        offsets = [o for o, _ in c.results.features]
        assert len(offsets) == len(set(offsets))
