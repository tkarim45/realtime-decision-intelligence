.PHONY: install test lint stream summary drift recover parity score uncertainty loadtest pipeline decide

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

# Labelled synthetic telemetry as JSONL on stdout.
stream:
	rdi-stream --n 600

summary:
	rdi-stream --n 600 --summary

# Distribution shift with zero incidents, for testing the drift monitor.
drift:
	rdi-stream --n 600 --drifted --summary

# Kill a consumer mid-stream and lose nothing; offline features match online.
recover:
	rdi-demo recover

parity:
	rdi-demo parity

# Incident classifier, temporal vs shuffled split.
score:
	rdi-demo score

# Unsupervised detector and conformal prediction sets.
uncertainty:
	rdi-demo uncertainty

# Hot-path latency, and why the detector is not on it.
loadtest:
	rdi-demo loadtest

# Every layer end to end on one stream.
pipeline:
	rdi-demo pipeline

# The intervention policy against the baselines it has to beat.
decide:
	rdi-demo decide
