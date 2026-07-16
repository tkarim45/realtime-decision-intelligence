.PHONY: install test lint stream summary drift

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
