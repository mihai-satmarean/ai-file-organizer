"""Shared pytest setup: put the project root on sys.path and provide a fake provider."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeProvider:
    """Stand-in for an LLM provider — returns canned JSON, records no real usage."""

    label = "Fake"

    def __init__(self, payload):
        self._payload = payload
        self.calls = []
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

    def status_line(self):
        return "Fake provider"

    def usage_summary(self):
        return "📊 LLM usage: fake"

    def complete_json(self, system, user, *, schema=None, max_tokens=4096):
        self.calls.append({"system": system, "user": user, "schema": schema})
        payload = self._payload(user) if callable(self._payload) else self._payload
        return payload


@pytest.fixture
def make_fake_provider():
    return FakeProvider
