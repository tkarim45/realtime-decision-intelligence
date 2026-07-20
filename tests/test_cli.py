"""Tests for the `rdi-stream` CLI.

The stream is meant to be piped (`rdi-stream | jq`, `| head`), so the pipe contract is part
of the artifact, not a detail.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from rdi.cli import main


def _run(args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        f"{sys.executable} -m rdi.cli {args}",
        shell=True, capture_output=True, text=True,
    )


def test_emits_valid_jsonl(capsys):
    assert main(["--n", "5"]) == 0
    lines = capsys.readouterr().out.strip().split("\n")
    assert len(lines) == 5 * 6
    for line in lines:
        assert json.loads(line)["service"]


def test_summary_prints_ground_truth_breakdown(capsys):
    assert main(["--n", "50", "--summary"]) == 0
    out = capsys.readouterr().out
    assert "breaching (label=1)" in out
    assert "incident_type" in out


def test_summary_emits_no_jsonl(capsys):
    main(["--n", "20", "--summary"])
    assert "{" not in capsys.readouterr().out


def test_head_does_not_raise_broken_pipe():
    """`rdi-stream | head` is documented usage; a stack trace there is a broken artifact."""
    proc = subprocess.run(
        f"{sys.executable} -m rdi.cli --n 200 | head -3",
        shell=True, capture_output=True, text=True,
    )
    assert "BrokenPipeError" not in proc.stderr
    assert proc.stderr.strip() == ""
    assert len(proc.stdout.strip().split("\n")) == 3


def test_drifted_flag_yields_no_incidents(capsys):
    main(["--n", "20", "--drifted"])
    events = [json.loads(line) for line in capsys.readouterr().out.strip().split("\n")]
    assert all(e["incident_type"] is None for e in events)


@pytest.mark.parametrize("flag", ["--n", "--seed", "--incidents-per-service"])
def test_flags_accept_values(flag, capsys):
    assert main([flag, "2", "--n", "5", "--summary"]) == 0
