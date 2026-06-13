# AI File Organizer

A command-line tool that automatically sorts messy folders into a clean, hierarchical category tree using either a free **offline semantic-embedding** model or a **bring-your-own-key Large Language Model** — Claude, GPT, Gemini, or any OpenAI-compatible provider. It combines **multi-provider LLM classification** (with **vision**-based image understanding, **prompt caching**, **structured JSON outputs**, **concurrent** requests, and exponential-backoff retries) with **crash-safe file operations** (append-only journaling and one-click undo), secret-safe **API key management**, and content extraction from PDF, Word, PowerPoint, Excel, and images via OCR.

## Features

- **Two interchangeable backends** — a free, fully offline embedding classifier (sentence-transformers), or a bring-your-own-key LLM for higher accuracy.
- **Every major provider through one key** — Anthropic Claude, OpenAI, Google Gemini, Groq, Mistral, DeepSeek, Together, OpenRouter, a local Ollama server, or any custom OpenAI-compatible endpoint.
- **Vision-based image sorting** — image files are sent to multimodal models and classified by what they actually show, not just their filename.
- **Token-frugal LLM mode** — request batching, Anthropic prompt caching, content truncation, and schema-constrained outputs keep cost low; each run prints a token/cache-hit summary.
- **Concurrent classification** — batches run in parallel with a cache-warming first request, plus automatic retry/backoff on rate limits and server errors.
- **LLM category discovery** — let the model propose a category tree from your own files, or supply a custom tree as JSON/YAML.
- **Crash-safe organization** — every move is journaled to disk as it happens and an undo script is generated even if the run is interrupted; dry-run and copy modes included.
- **Secret-safe keys** — API keys come from the environment, a gitignored `.env`, or a hidden prompt; they are masked in output and never written to config, logs, or undo scripts.
- **Privacy controls** — exclude sensitive files via `.organizerignore` or `--exclude` globs, a notice before any data leaves your machine, and a fully offline mode.
- **Spend guard** — a pre-run token estimate and a hard file-count cap prevent surprise bills.
- **Low-confidence Review folder** — uncertain placements are quarantined instead of mis-filed.
- **Interactive setup wizard** plus content extraction for PDF, DOCX, PPTX, XLSX, images (OCR), and text/code files.

## How It Works

Files flow through a four-stage pipeline: **scan → extract → classify → move.**

1. **Scan** discovers files recursively, skipping hidden/system files and anything matched by a `.organizerignore` file or `--exclude` glob.
2. **Extract** pulls text from each file using a format-specific extractor (PDF, Word, PowerPoint, Excel, image OCR, or plain text), with character-encoding detection.
3. **Classify** assigns each file a path in a hierarchical category tree using the selected backend.
4. **Move** relocates (or copies) files into the destination tree, recording every operation to a crash-safe journal and emitting an undo script.

**Classification backends.** The *local* backend embeds filenames and content with sentence-transformers and matches them to category descriptions by cosine similarity, walking the tree level by level and falling back to an "Other" folder when confidence drops below a per-node threshold — entirely offline, no key, no network. The *LLM* backend instead asks a language model to place each file directly into the tree, which handles messy or ambiguous files far better.

**Token frugality.** The LLM backend classifies many files per request so the (large, fixed) category tree is sent once per batch rather than once per file. On Anthropic that tree lives in a cached system prompt (`cache_control: ephemeral`), so every batch after the first is dramatically cheaper. Responses are constrained with JSON-schema structured outputs (with defensive parsing as a fallback for providers that only support JSON-object mode), and per-file content is truncated to a short snippet.

**Vision.** Image files are downscaled and re-encoded with Pillow, then sent to the model as actual images so a screenshot of a receipt lands in Finance, not Photos. If the chosen model has no vision support, that batch is automatically retried text-only — nothing breaks.

**Concurrency & resilience.** Batches are classified in parallel via a thread pool; the first batch runs alone to warm the prompt cache before the rest fan out. Rate-limit (429) and server (5xx) errors are retried with exponential backoff, and a batch that errors or returns truncated JSON is recursively split and retried so only genuinely failing single files fall back to "Other."

**Safety.** The tool refuses to write its output inside the folder being organized, confirms before a real move, and writes each move to an append-only journal (flushed with `fsync`) plus a generated undo script — so even an interrupted run is fully recoverable. Low-confidence LLM placements can be routed to a `Review/` folder, and a spend guard estimates token usage and caps the number of files before any request is made.

## Architecture

The codebase is organized around small, swappable components:

- **Provider layer** — a `BaseProvider` interface with two implementations: an Anthropic provider (official `anthropic` SDK, prompt caching + structured outputs) and a single OpenAI-compatible provider (official `openai` SDK) that drives OpenAI, Gemini, Groq, Mistral, DeepSeek, Together, OpenRouter, Ollama, and custom endpoints. A neutral content format is rendered to each SDK's multimodal message shape, and token usage is accumulated behind a lock for thread safety.
- **Extractor registry** — pluggable extractors selected by file extension/MIME type, each producing a uniform `ExtractedContent`.
- **Category tree** — a recursive `CategoryTree` of nodes with descriptions and thresholds, automatic "Other" fallback folders at every level, and JSON/YAML (de)serialization; the LLM classifier normalizes model-returned paths back onto real tree nodes.
- **File mover** — duplicate handling (rename/skip/overwrite), dry-run simulation, incremental journaling, and undo-script generation.
- **Secrets module** — layered key resolution (environment → `.env` → hidden `getpass` prompt) with masking and `0600` file permissions, never persisting a key to config.
- **Config dataclass** — typed settings loaded from YAML/JSON that strips any secret-looking fields, with CLI flags overriding on top.

## Skills Demonstrated

- LLM application development — multi-provider file classification via the Anthropic and OpenAI Python SDKs
- Provider abstraction (strategy pattern) — one interface over Claude, OpenAI, Gemini, Groq, Mistral, DeepSeek, Together, OpenRouter, and Ollama
- Prompt caching — Anthropic ephemeral `cache_control` on a shared system prompt to slash repeat-batch cost
- Structured outputs — JSON-schema-constrained model responses with a defensive JSON-extraction fallback
- Computer vision / multimodal AI — downscaled images (Pillow) sent to vision models, with text-only fallback for non-vision models
- Semantic embeddings — offline cosine-similarity classification with sentence-transformers
- Hierarchical classification — recursive category tree with "Other" fallbacks and model-path normalization
- Concurrency & multithreading — `ThreadPoolExecutor` batch parallelism with a cache-warming first request
- Thread-safe state — mutex-guarded token-usage accounting across parallel requests
- Rate-limit handling — exponential-backoff retries on HTTP 429 / 5xx
- Fault tolerance — recursive batch split-and-retry on API errors or truncated JSON
- Crash-safe file operations — append-only journal flushed with `fsync` and interrupt-safe undo-script generation
- Secrets management — environment / `.env` / hidden-prompt key resolution, masked display, `0600` permissions, never logged
- Content extraction — PDF (PyPDF2), Word (python-docx), PowerPoint (python-pptx), Excel (openpyxl), image OCR (pytesseract), encoding detection (chardet)
- CLI design — an argparse command-line interface plus an interactive setup wizard
- Cost engineering — pre-run token estimation and a hard file-count spend guard
- Privacy engineering — `.organizerignore` / glob excludes and a data-egress notice
- Automatic dependency management — on-demand provider-SDK installation
- Packaging — `pyproject.toml` with optional-dependency extras and a console entry point
- Automated testing — 66 pytest tests including an end-to-end organize→undo flow and concurrency-ordering checks
- CI/CD — GitHub Actions matrix across Python 3.9, 3.11, and 3.12
- Design patterns — registry (extractors), strategy (providers), and graceful degradation throughout

## Tech Stack

- **Language:** Python 3.9+
- **LLM APIs / SDKs:** Anthropic API (`anthropic`), OpenAI API (`openai`) — the latter also drives Google Gemini, Groq, Mistral, DeepSeek, Together AI, OpenRouter, and local Ollama via OpenAI-compatible endpoints
- **Local ML:** sentence-transformers, PyTorch, scikit-learn, NumPy
- **Content extraction:** Pillow, pytesseract (Tesseract OCR), PyPDF2, python-docx, python-pptx, openpyxl, chardet
- **Utilities:** tqdm, PyYAML
- **Tooling:** pytest, setuptools (`pyproject.toml`), GitHub Actions

## Getting Started

### Install

Install as a package (gives you a `file-organizer` command):

```bash
pip install -e ".[llm]"      # core extraction + Anthropic/OpenAI SDKs
pip install -e ".[local]"    # add the free offline backend (large; pulls in torch)
pip install -e ".[all]"      # everything, including dev/test tools
```

(The LLM SDKs also auto-install on first use, based on the provider you pick.)

For OCR on images, install Tesseract: `brew install tesseract` (macOS) or `sudo apt-get install tesseract-ocr` (Debian/Ubuntu).

### Choose a backend

```bash
# Free, offline (default) — local embedding model
file-organizer ./downloads ./organized

# Bring-your-own-key LLM (more accurate)
file-organizer --backend llm --provider anthropic ./downloads ./organized

# List every supported provider and the env var it reads
file-organizer --list-providers
```

| Provider key        | Service               | Key env var                         |
|---------------------|-----------------------|-------------------------------------|
| `anthropic`         | Claude                | `ANTHROPIC_API_KEY`                 |
| `openai`            | OpenAI                | `OPENAI_API_KEY`                    |
| `gemini`            | Google Gemini         | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `groq`              | Groq                  | `GROQ_API_KEY`                      |
| `mistral`           | Mistral               | `MISTRAL_API_KEY`                   |
| `deepseek`          | DeepSeek              | `DEEPSEEK_API_KEY`                  |
| `together`          | Together AI           | `TOGETHER_API_KEY`                  |
| `openrouter`        | OpenRouter            | `OPENROUTER_API_KEY`                |
| `ollama`            | Local Ollama server   | _(none required)_                   |
| `openai-compatible` | Any custom endpoint   | `LLM_API_KEY` (+ `--base-url`)      |

Pick the model with `--model` (e.g. `--model claude-haiku-4-5` for the cheapest Claude, or `--model gpt-4o-mini`).

### API keys & security

You supply your own key, and it is treated as a secret:

- Keys are read from, in order: the **environment**, a gitignored **`.env`** file, then a **hidden prompt**.
- A key is **never** written to config, logs, or undo scripts, and is shown only as a masked fingerprint like `sk-…a1b2`.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # option A: environment
cp .env.example .env                     # option B: edit .env (gitignored)
# option C: just run it and enter the key at the hidden prompt
```

### Interactive mode

Run with no arguments (or `-i`) for a guided wizard that walks through backend, provider, key, model, folders, and action:

```bash
file-organizer
```

### Categories

```bash
# Let the LLM propose a category tree from your files, then save it
file-organizer --provider anthropic --discover-categories ./downloads

# Organize using a custom category tree (JSON or YAML)
file-organizer --backend llm --categories-file categories.json ./downloads ./organized
```

A category file looks like:

```json
{
  "categories": [
    {"name": "Finance", "description": "bank statements, taxes, invoices, receipts",
     "children": [{"name": "Taxes", "description": "tax returns, T4, W2, CRA, IRS"}]}
  ]
}
```

### Safety, privacy & cost

- **Review folder:** `--min-confidence 0.6` routes uncertain LLM placements to `Review/`.
- **Spend guard:** runs print a token estimate and refuse more than `--max-files` files (default 2000).
- **Exclude files:** add globs to `.organizerignore`, or pass `--exclude '*.key' --exclude 'secret*'`.
- **Confirmations & undo:** real moves prompt first (skip with `--yes`); an undo script is written to `organized/_logs/`.
- **Speed:** batches classify in parallel (`--concurrency`, default 4) with automatic retry/backoff.

### Diagnose a file

```bash
file-organizer ... --preview ./downloads        # preview classifications, no changes
python train.py diagnose --backend llm --provider anthropic <file>   # top candidate folders + reasons
```

### Configuration

Settings can also live in a YAML/JSON config (`--config config.yaml`). It supports backend choice, provider/model, batch size, concurrency, confidence threshold, vision, and more. **API keys are never read from or written to config files** — any `api_key` field is dropped with a warning.

### Testing

```bash
make ci        # install ".[llm,dev]" and run the suite
pytest -q      # or just run the tests
```

### License

MIT — see [LICENSE](LICENSE).
