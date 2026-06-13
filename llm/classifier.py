"""
LLM-based hierarchical classifier.

Instead of embedding every file and every category and comparing vectors, this
asks an LLM to place each file directly into the category tree. It is more
accurate on messy/ambiguous files and works with any provider.

Token frugality (the explicit goal):
  * Batching — the category tree is sent ONCE per batch and many files are
    classified in a single request, amortizing the (large, fixed) tree prompt.
  * Caching — on Anthropic the tree lives in a cached system prompt, so every
    batch after the first pays ~0.1x for that prefix.
  * Truncation — each file contributes only a short snippet (filename + a
    capped slice of content), not its whole text.
  * No thinking / temperature 0 / tight max_tokens — classification is simple
    and the output is small structured JSON.

Returned results reuse `classifier.hierarchical.ClassificationResult` so the
LLM backend is a drop-in replacement for the local embedding backend.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from categories.tree import CategoryNode, CategoryTree
from classifier.hierarchical import ClassificationResult
from extractors.base import ExtractedContent
from llm.providers import BaseProvider
from llm.images import is_image_path, load_image_part

log = logging.getLogger("file_organizer.classifier")


# JSON schema for structured output (used directly by Anthropic; described in
# the prompt for other providers). No numeric constraints — unsupported by
# structured-output validators.
_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "path": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["index", "path", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def _outline(tree: CategoryTree) -> str:
    """Render the category tree as a compact indented outline with descriptions."""
    lines: list[str] = []

    def walk(node: CategoryNode, depth: int):
        indent = "  " * depth
        desc = node.description.strip()
        if len(desc) > 160:
            desc = desc[:160].rstrip() + "…"
        lines.append(f"{indent}- {node.name}: {desc}")
        for child in node.children:
            walk(child, depth + 1)

    for root in tree.root_categories:
        walk(root, 0)
    lines.append(f"- {tree.other_category.name}: {tree.other_category.description}")
    return "\n".join(lines)


class LLMClassifier:
    """Classifies files into a CategoryTree using an LLM provider."""

    def __init__(
        self,
        category_tree: CategoryTree,
        provider: BaseProvider,
        *,
        batch_size: int = 20,
        max_chars_per_file: int = 1200,
        max_output_tokens: int = 4096,
        min_confidence: float = 0.0,
        review_folder: str = "Review",
        vision: bool = True,
        vision_max_dim: int = 1024,
        concurrency: int = 4,
    ):
        self.category_tree = category_tree
        self.provider = provider
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)
        self.max_chars_per_file = max_chars_per_file
        self.max_output_tokens = max_output_tokens
        # Files the model places with confidence below this go to `review_folder`
        # instead of being filed (set 0.0 to disable).
        self.min_confidence = min_confidence
        self.review_folder = review_folder
        # When enabled, image files are sent to the model as actual (downscaled)
        # images so they're classified by content, not just filename.
        self.vision = vision
        self.vision_max_dim = vision_max_dim

        # Precompute once: the outline (system prompt) and the valid-path index.
        self._outline_text = _outline(category_tree)
        self._valid_leaf_paths = {p for p, _ in category_tree.get_all_leaf_paths()}
        self._node_paths = self._build_node_index(category_tree)

    @staticmethod
    def _build_node_index(tree: CategoryTree) -> dict[str, CategoryNode]:
        index: dict[str, CategoryNode] = {}

        def walk(node: CategoryNode, prefix: str):
            path = f"{prefix}/{node.name}" if prefix else node.name
            index[path] = node
            for child in node.children:
                walk(child, path)

        for root in tree.root_categories:
            walk(root, "")
        index[tree.other_category.name] = tree.other_category
        return index

    # ---- prompt construction -------------------------------------------------

    def _system_prompt(self) -> str:
        return (
            "You are a file organizer. Classify each file into exactly one folder "
            "from the category tree below.\n\n"
            "Category tree (indentation shows nesting):\n"
            f"{self._outline_text}\n\n"
            "Some files are images, included inline after their label — classify "
            "them by what they actually show.\n\n"
            "Rules:\n"
            "1. Choose the DEEPEST folder that clearly fits the file.\n"
            "2. If a file fits a parent category but no specific child, use that "
            "parent's 'Other' subfolder (e.g. 'School/University/Other').\n"
            "3. If nothing fits at the top level, use 'Other'.\n"
            "4. Use the full path with '/' separators, exactly matching names in the tree.\n"
            "5. confidence is your certainty from 0.0 to 1.0.\n\n"
            "Respond with ONLY a JSON object of the form:\n"
            '{"results": [{"index": <int>, "path": "<folder/path>", "confidence": <0..1>}]}\n'
            "Include one entry for every file index given, and nothing else."
        )

    def _user_prompt(self, contents: list[ExtractedContent]) -> str:
        lines = ["Classify these files:\n"]
        for i, content in enumerate(contents):
            snippet = content.combined_content[: self.max_chars_per_file]
            snippet = " ".join(snippet.split())  # collapse whitespace to save tokens
            lines.append(f"[{i}] name={content.file_path.name!r}\n    content: {snippet}")
        return "\n".join(lines)

    def _user_payload(self, contents: list[ExtractedContent]):
        """
        Build the user message. Returns a plain string when there are no images
        (cheaper, same as before); otherwise a list of neutral content parts with
        the actual image bytes interleaved after each image file's label.
        """
        if not self.vision:
            return self._user_prompt(contents)

        parts: list[dict] = [{"type": "text", "text": "Classify these files:\n"}]
        any_image = False
        for i, content in enumerate(contents):
            label = f"[{i}] name={content.file_path.name!r}"
            image_part = None
            if is_image_path(content.file_path):
                image_part = load_image_part(content.file_path, max_dim=self.vision_max_dim)
            if image_part is not None:
                any_image = True
                parts.append({"type": "text", "text": f"{label}\n    image:"})
                parts.append(image_part)
            else:
                snippet = " ".join(content.combined_content[: self.max_chars_per_file].split())
                parts.append({"type": "text", "text": f"{label}\n    content: {snippet}"})

        # No images actually loaded — use the cheaper string form.
        return parts if any_image else self._user_prompt(contents)

    # ---- path normalization --------------------------------------------------

    def _normalize_path(self, raw: str) -> tuple[str, bool]:
        """
        Map a model-returned path onto a real folder in the tree.

        Returns (path, is_other). Guarantees the path corresponds to a real,
        creatable destination so the output folder structure stays consistent.
        """
        if not raw:
            return "Other", True
        path = raw.strip().strip("/")

        # Exact leaf match — best case.
        if path in self._valid_leaf_paths:
            return path, path.split("/")[-1].lower() == "other"

        # Resolve to the deepest existing node along the path, then leaf-ify.
        parts = path.split("/")
        resolved: list[str] = []
        node: Optional[CategoryNode] = None
        for part in parts:
            candidate = "/".join(resolved + [part])
            match = self._node_paths.get(candidate) or self._case_insensitive_node(candidate)
            if match is None:
                break
            resolved.append(match.name)
            node = match

        if node is None:
            return "Other", True

        resolved_path = "/".join(resolved)
        # If we landed on a non-leaf, route into its 'Other' child.
        if node.children:
            other = node.get_child_by_name("Other")
            if other is not None:
                return f"{resolved_path}/Other", True
        return resolved_path, resolved_path.split("/")[-1].lower() == "other"

    def _case_insensitive_node(self, candidate: str) -> Optional[CategoryNode]:
        lowered = candidate.lower()
        for path, node in self._node_paths.items():
            if path.lower() == lowered:
                return node
        return None

    # ---- classification ------------------------------------------------------

    def _call_model(self, contents: list[ExtractedContent]) -> Optional[dict]:
        """One model call. Retries vision payloads as text-only. Returns None on failure."""
        payload = self._user_payload(contents)
        try:
            return self.provider.complete_json(
                self._system_prompt(), payload,
                schema=_RESULT_SCHEMA, max_tokens=self.max_output_tokens,
            )
        except Exception as e:
            # If we sent images, the model may not support vision — retry text-only.
            if isinstance(payload, list):
                try:
                    return self.provider.complete_json(
                        self._system_prompt(), self._user_prompt(contents),
                        schema=_RESULT_SCHEMA, max_tokens=self.max_output_tokens,
                    )
                except Exception as e2:
                    log.warning("Batch of %d failed even without images: %s", len(contents), e2)
                    return None
            log.warning("Batch of %d failed: %s", len(contents), e)
            return None

    def _classify_one_batch(self, contents: list[ExtractedContent]) -> list[ClassificationResult]:
        data = self._call_model(contents)

        # On failure (API error, or JSON truncated because the batch was too big),
        # split and retry — only genuinely-failing single files fall back to Other.
        if data is None:
            if len(contents) > 1:
                mid = len(contents) // 2
                log.debug("Splitting failed batch of %d into %d + %d",
                          len(contents), mid, len(contents) - mid)
                return (self._classify_one_batch(contents[:mid])
                        + self._classify_one_batch(contents[mid:]))
            log.warning("Routing %s to 'Other' after repeated failure.",
                        contents[0].file_path.name)
            return [self._fallback(contents[0])]

        by_index: dict[int, dict] = {}
        for entry in data.get("results", []):
            try:
                by_index[int(entry["index"])] = entry
            except (KeyError, ValueError, TypeError):
                continue

        results: list[ClassificationResult] = []
        for i, content in enumerate(contents):
            entry = by_index.get(i)
            if entry is None:
                results.append(self._fallback(content))
                continue
            path, is_other = self._normalize_path(str(entry.get("path", "")))
            try:
                confidence = float(entry.get("confidence", 0.0))
            except (ValueError, TypeError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            # Route uncertain placements to a Review folder instead of filing them.
            if self.min_confidence > 0.0 and confidence < self.min_confidence:
                path, is_other = self.review_folder, False
            results.append(ClassificationResult(
                file_path=str(content.file_path),
                category_path=path,
                confidence_scores={"llm": confidence},
                final_confidence=confidence,
                is_other=is_other,
                classification_chain=path.split("/"),
            ))
        return results

    @staticmethod
    def _fallback(content: ExtractedContent) -> ClassificationResult:
        return ClassificationResult(
            file_path=str(content.file_path),
            category_path="Other",
            confidence_scores={"llm": 0.0},
            final_confidence=0.0,
            is_other=True,
            classification_chain=["Other"],
        )

    def estimate_usage(self, contents: list[ExtractedContent]) -> tuple[int, int]:
        """
        Rough pre-run estimate: (approx_input_tokens, num_requests). Deliberately
        crude (~4 chars/token); used only to warn the user before spending.
        """
        num_requests = max(1, math.ceil(len(contents) / self.batch_size))
        system_tokens = len(self._system_prompt()) // 4
        total = num_requests * system_tokens
        for c in contents:
            snippet = c.combined_content[: self.max_chars_per_file]
            total += max(8, len(snippet) // 4)
            if self.vision and is_image_path(c.file_path):
                total += 1300  # rough cost of one downscaled image
        return total, num_requests

    def diagnose(self, content: ExtractedContent, top_k: int = 3) -> list[dict]:
        """
        Explain a single file: return the top candidate folders the model would
        consider, best-first, each with a confidence and a one-line reason. Useful
        for understanding (and sanity-checking) the model's self-reported confidence.
        """
        system = (
            "You are a file organizer. Given the category tree and ONE file "
            "(name, content, and possibly an image), return the top candidate "
            "folders, best first, each with a confidence (0.0-1.0) and a one-line "
            "reason.\n\n"
            f"Category tree:\n{self._outline_text}\n\n"
            'Respond with ONLY JSON: {"candidates": [{"path": "...", '
            '"confidence": <0..1>, "reason": "..."}]}'
        )
        schema = {
            "type": "object",
            "properties": {"candidates": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["path", "confidence", "reason"],
                "additionalProperties": False,
            }}},
            "required": ["candidates"],
            "additionalProperties": False,
        }
        data = self.provider.complete_json(
            system, self._user_payload([content]), schema=schema, max_tokens=1024
        )

        out: list[dict] = []
        for cand in data.get("candidates", [])[:top_k]:
            path, _ = self._normalize_path(str(cand.get("path", "")))
            try:
                conf = max(0.0, min(1.0, float(cand.get("confidence", 0.0))))
            except (ValueError, TypeError):
                conf = 0.0
            out.append({"path": path, "confidence": conf,
                        "reason": str(cand.get("reason", ""))[:200]})
        return out

    def classify(self, content: ExtractedContent) -> ClassificationResult:
        """Classify a single file (convenience; batch is far more token-efficient)."""
        return self._classify_one_batch([content])[0]

    def classify_batch(
        self,
        contents: list[ExtractedContent],
        show_progress: bool = True,
    ) -> list[ClassificationResult]:
        """
        Classify many files, one API request per `batch_size` files.

        Batches run concurrently (up to `concurrency`). The first batch runs
        alone to warm the provider's prompt cache (concurrent requests with an
        identical cached prefix would each pay full price), then the rest fan out.
        """
        batches = [
            contents[i:i + self.batch_size]
            for i in range(0, len(contents), self.batch_size)
        ]
        if not batches:
            return []

        progress = None
        if show_progress:
            try:
                from tqdm import tqdm
                progress = tqdm(total=len(batches), desc="   Classifying (LLM)", unit="batch")
            except ImportError:
                progress = None

        def run(batch):
            out = self._classify_one_batch(batch)
            if progress is not None:
                progress.update(1)
            return out

        per_batch: list[list[ClassificationResult]] = [None] * len(batches)
        try:
            # Warm the cache with the first batch before fanning out.
            per_batch[0] = run(batches[0])

            if self.concurrency > 1 and len(batches) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
                    futures = {ex.submit(run, b): i for i, b in enumerate(batches[1:], start=1)}
                    for fut in as_completed(futures):
                        per_batch[futures[fut]] = fut.result()
            else:
                for i in range(1, len(batches)):
                    per_batch[i] = run(batches[i])
        finally:
            if progress is not None:
                progress.close()

        results: list[ClassificationResult] = []
        for batch_results in per_batch:
            results.extend(batch_results)
        return results
