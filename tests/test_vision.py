"""Vision: image loading, neutral->provider rendering, classifier payload + fallback."""

from pathlib import Path

import pytest

from categories import create_waterloo_example_categories as create_default_categories
from extractors.base import ExtractedContent
from llm.classifier import LLMClassifier
from llm.images import is_image_path, load_image_part
from llm.providers import _anthropic_content, _openai_content

from conftest import FakeProvider


def _png(path: Path):
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (80, 60), (123, 50, 200)).save(path)
    return path


def _content(path):
    p = Path(path)
    return ExtractedContent(p, p.name, p.suffix, "", {}, True)


# ---- image loading -------------------------------------------------------

def test_is_image_path():
    assert is_image_path("a.PNG") and is_image_path("b.jpg")
    assert not is_image_path("c.txt") and not is_image_path("d.pdf")


def test_load_image_part(tmp_path):
    part = load_image_part(_png(tmp_path / "x.png"), max_dim=32)
    assert part is not None
    assert part["type"] == "image"
    assert part["media_type"] == "image/jpeg"
    assert isinstance(part["data"], str) and len(part["data"]) > 0


def test_load_image_part_missing_returns_none(tmp_path):
    assert load_image_part(tmp_path / "nope.png") is None


# ---- neutral -> provider rendering --------------------------------------

def test_render_passthrough_string():
    assert _anthropic_content("hello") == "hello"
    assert _openai_content("hello") == "hello"


def test_anthropic_render_parts():
    parts = [
        {"type": "text", "text": "label"},
        {"type": "image", "media_type": "image/jpeg", "data": "QUJD"},
    ]
    out = _anthropic_content(parts)
    assert out[0] == {"type": "text", "text": "label"}
    assert out[1]["type"] == "image"
    assert out[1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}


def test_openai_render_parts():
    parts = [{"type": "image", "media_type": "image/jpeg", "data": "QUJD"}]
    out = _openai_content(parts)
    assert out[0]["type"] == "image_url"
    assert out[0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


# ---- classifier payload --------------------------------------------------

def _classifier(payload, **kw):
    return LLMClassifier(create_default_categories(), FakeProvider(payload), **kw)


def test_vision_sends_image_part(tmp_path):
    img = _png(tmp_path / "vacation.png")
    payload = {"results": [{"index": 0, "path": "Personal/Photos/Camera", "confidence": 0.9}]}
    clf = _classifier(payload, vision=True)
    clf.classify_batch([_content(img)], show_progress=False)
    sent = clf.provider.calls[0]["user"]
    assert isinstance(sent, list)
    assert any(p["type"] == "image" for p in sent)


def test_vision_disabled_sends_string(tmp_path):
    img = _png(tmp_path / "vacation.png")
    clf = _classifier({"results": []}, vision=False)
    clf.classify_batch([_content(img)], show_progress=False)
    assert isinstance(clf.provider.calls[0]["user"], str)


def test_non_image_uses_string_even_with_vision(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    clf = _classifier({"results": []}, vision=True)
    clf.classify_batch([_content(txt)], show_progress=False)
    # No images in the batch => cheaper string form.
    assert isinstance(clf.provider.calls[0]["user"], str)


def test_vision_unsupported_retries_text_only(tmp_path):
    img = _png(tmp_path / "p.png")

    class FlakyVisionProvider:
        def __init__(self):
            self.calls = []

        def complete_json(self, system, user, *, schema=None, max_tokens=4096):
            self.calls.append(user)
            if isinstance(user, list):
                raise RuntimeError("this model has no vision")
            return {"results": [{"index": 0, "path": "Personal/Photos/Screenshots", "confidence": 0.7}]}

    clf = LLMClassifier(create_default_categories(), FlakyVisionProvider(), vision=True)
    [res] = clf.classify_batch([_content(img)], show_progress=False)
    # Retried with the string payload and succeeded — not lost to 'Other'.
    assert res.category_path == "Personal/Photos/Screenshots"
    assert len(clf.provider.calls) == 2
    assert isinstance(clf.provider.calls[0], list)
    assert isinstance(clf.provider.calls[1], str)
