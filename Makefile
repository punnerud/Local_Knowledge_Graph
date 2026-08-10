PY := .venv/bin/python

.PHONY: all venv lint test test-ollama measure sweep bench loops eval check-numbers run clean

# One gate. Lint, tests and the documented numbers pass together or the build is
# not green -- keeping the numbers in a separate optional target is how a README
# drifts away from what the code actually does.
all: lint test check-numbers

# pyproject.toml is the single source of truth for dependencies; there is no
# requirements.txt to drift out of sync with it.
venv:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"
	$(PY) -m playwright install chromium --only-shell

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest tests/ -q -m "not ollama"

# Requires a running Ollama with the models named in the README.
test-ollama:
	$(PY) -m pytest tests/ -q -m ollama

# Regenerates docs/claims/*.json from a live Ollama.
measure:
	$(PY) scripts/measure.py

# How well does each layer of a local model separate topics? Needs torch.
sweep:
	$(PY) scripts/layer_sweep.py

# Exact scan versus an approximate index, at several store sizes.
bench:
	$(PY) scripts/bench_search.py

check-numbers:
	$(PY) scripts/check_numbers.py

run:
	$(PY) app.py

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ tests/__pycache__ embeddings.db embeddings.ann

# Is repetition actually the problem, or does the model drift instead?
loops:
	$(PY) scripts/measure_loops.py

# Is the thinking improving the answer? Needs a live Ollama.
eval:
	$(PY) scripts/eval.py --repeats 4 --label validated
