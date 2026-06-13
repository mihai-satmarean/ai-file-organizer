.PHONY: install install-all test lint run clean

# Mirror what CI does: core + LLM SDKs + test tooling (skips the heavy torch extra).
install:
	python -m pip install --upgrade pip
	pip install -e ".[llm,dev]"

# Everything, including the offline local backend (large).
install-all:
	pip install -e ".[all]"

test:
	pytest -q

# Same command CI runs, end to end, from a clean checkout.
ci: install test

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -not -path './venv/*' -exec rm -rf {} +
