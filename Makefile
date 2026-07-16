.PHONY: install test lint stream summary drift recover parity score

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

# M0 artifact: labeled synthetic AIOps stream as JSONL on stdout.
stream:
	rdi-stream --n 600

summary:
	rdi-stream --n 600 --summary

# The M5 specificity fixture: distribution shift, zero incidents injected.
drift:
	rdi-stream --n 600 --drifted --summary

# M1 artifacts: S1 (kill a consumer, lose nothing) and train=serve parity.
recover:
	rdi-demo recover

parity:
	rdi-demo parity

# M2: hot-path incident classifier, temporal vs random split.
score:
	rdi-demo score
