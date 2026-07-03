#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
AI-Powered File Organizer

Main entry point for organizing files using AI-based classification.

Usage:
    python main.py [input_directory] [output_directory]
    python main.py --help
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import sys

# Add the project directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import OrganizerConfig
from categories import create_default_categories, CategoryTree
from extractors import ExtractorRegistry
from classifier import HierarchicalClassifier
from organizer import DirectoryScanner, FileMover


class FileOrganizer:
    """
    Main application class that orchestrates the file organization process.
    """
    
    def __init__(self, config: Optional[OrganizerConfig] = None):
        """
        Initialize the file organizer.
        
        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or OrganizerConfig()
        
        # Initialize components
        self.scanner = DirectoryScanner(
            skip_hidden=self.config.skip_hidden_files,
            max_file_size_mb=self.config.max_file_size_mb,
            ignore_patterns=self.config.exclude_patterns,
        )
        
        self.extractor = ExtractorRegistry(
            enable_ocr=self.config.enable_ocr
        )
        
        self.category_tree = self._load_categories()
        self.classifier: Optional[HierarchicalClassifier] = None
        
        self.mover = FileMover(
            output_directory=Path(self.config.output_directory),
            dry_run=self.config.dry_run,
            copy_instead_of_move=self.config.copy_instead_of_move,
            handle_duplicates=self.config.handle_duplicates,
            create_log=self.config.create_logs,
            create_undo=self.config.create_undo_script,
        )
    
    def _load_categories(self) -> CategoryTree:
        """Load category tree: a custom file if given, otherwise the defaults."""
        if self.config.custom_categories_file:
            path = Path(self.config.custom_categories_file).expanduser()
            if not path.is_file():
                raise ValueError(f"Category file not found: {path}")
            print(f"   Using categories from {path}")
            return CategoryTree.from_file(path)
        return create_default_categories()
    
    def _init_classifier(self) -> None:
        """Initialize the classifier (lazy loading). Picks backend from config."""
        if self.classifier is not None:
            return

        if self.config.backend == "llm":
            from llm import build_provider, LLMClassifier
            provider = build_provider(
                self.config.llm_provider,
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                auto_install=self.config.llm_auto_install_sdk,
                max_retries=self.config.llm_max_retries,
                project_dir=Path(__file__).parent,
            )
            print(f"   Using {provider.status_line()}")
            self.classifier = LLMClassifier(
                category_tree=self.category_tree,
                provider=provider,
                batch_size=self.config.llm_batch_size,
                max_chars_per_file=self.config.llm_max_chars_per_file,
                min_confidence=self.config.llm_min_confidence,
                vision=self.config.llm_vision,
                vision_max_dim=self.config.llm_vision_max_dim,
                concurrency=self.config.llm_concurrency,
            )
        else:
            from classifier import Embedder
            embedder = Embedder(model_name=self.config.embedding_model)
            self.classifier = HierarchicalClassifier(
                category_tree=self.category_tree,
                embedder=embedder
            )
    
    def organize(
        self,
        input_dirs: Optional[list[str]] = None,
        output_dir: Optional[str] = None,
        show_progress: bool = True,
        assume_yes: bool = False,
    ) -> dict:
        """
        Main method to organize files.

        Args:
            input_dirs: Input directories to scan. Uses config if not provided.
            output_dir: Output directory. Uses config if not provided.
            show_progress: Whether to show progress bars.
            assume_yes: Skip the confirmation prompt before moving files.

        Returns:
            Dictionary with organization results.
        """
        # Use provided paths or config defaults
        input_directories = [Path(d) for d in (input_dirs or self.config.input_directories)]
        output_directory = Path(output_dir or self.config.output_directory)

        # Safety: refuse to place the output inside (or equal to) an input dir —
        # that would re-scan organized files and move our own logs on a re-run.
        _guard_output_not_in_input(input_directories, output_directory)

        # Update mover output directory
        self.mover.output_directory = output_directory

        print("=" * 60)
        print("AI-Powered File Organizer")
        print("=" * 60)
        
        # Step 1: Scan for files
        print(f"\n📁 Scanning directories...")
        scan_result = self.scanner.scan_multiple(input_directories, self.config.recursive_scan)
        
        print(f"   Found {scan_result.supported_files} files to process")
        print(f"   Skipped {scan_result.skipped_files} files")
        
        if scan_result.errors:
            print(f"   ⚠️  Errors: {len(scan_result.errors)}")
            for error in scan_result.errors[:5]:
                print(f"      - {error}")
        
        if not scan_result.file_paths:
            print("\n❌ No files found to organize!")
            return {"status": "no_files", "scan_result": scan_result}
        
        # Step 2: Extract content
        print(f"\n📄 Extracting content from {len(scan_result.file_paths)} files...")
        
        extracted_contents = []
        extraction_errors = []
        
        files_to_process = scan_result.file_paths
        if show_progress:
            try:
                from tqdm import tqdm
                files_to_process = tqdm(files_to_process, desc="   Extracting")
            except ImportError:
                pass
        
        for file_path in files_to_process:
            content = self.extractor.extract(file_path)
            if content.extraction_success:
                extracted_contents.append(content)
            else:
                extraction_errors.append((file_path, content.error_message))
        
        print(f"   Successfully extracted: {len(extracted_contents)}")
        if extraction_errors:
            print(f"   Extraction errors: {len(extraction_errors)}")
        
        # Spend guard: refuse oversized LLM runs before building the provider.
        if self.config.backend == "llm" and self.config.llm_max_files:
            n = len(extracted_contents)
            if n > self.config.llm_max_files:
                raise ValueError(
                    f"{n} files exceeds the --max-files limit of {self.config.llm_max_files}. "
                    f"Re-run with --max-files {n} (or higher) to allow it."
                )

        # Step 3: Classify files
        print(f"\n🤖 Classifying files...")
        self._init_classifier()

        # Privacy notice + rough cost estimate before spending anything.
        if self.config.backend == "llm":
            if self.config.show_data_notice:
                print(_data_notice(self.config))
            est_tokens, est_requests = self.classifier.estimate_usage(extracted_contents)
            print(f"   ≈{est_tokens:,} input tokens across {est_requests} request(s) (rough estimate)")

        results = self.classifier.classify_batch(extracted_contents, show_progress=show_progress)
        classifications = [
            (content.file_path, result.category_path, result.final_confidence)
            for content, result in zip(extracted_contents, results)
        ]
        
        # Show classification summary
        category_counts = {}
        for _, cat_path, _ in classifications:
            root = cat_path.split('/')[0]
            category_counts[root] = category_counts.get(root, 0) + 1
        
        print(f"   Classification summary:")
        for cat, count in sorted(category_counts.items()):
            print(f"      {cat}: {count} files")

        # Show what the LLM run cost, in tokens (proves caching is working).
        if self.config.backend == "llm" and self.config.show_token_usage:
            print(f"   {self.classifier.provider.usage_summary()}")

        # Step 4: Organize files
        action = "Simulating" if self.config.dry_run else "Moving" if not self.config.copy_instead_of_move else "Copying"

        # Confirm before a real MOVE (irreversible-ish) when run interactively.
        if (not self.config.dry_run and not self.config.copy_instead_of_move
                and not assume_yes and sys.stdin is not None and sys.stdin.isatty()):
            answer = input(
                f"\n⚠️  About to MOVE {len(classifications)} files into "
                f"{output_directory}. Proceed? (y/N): "
            ).strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted — nothing moved.")
                return {"status": "aborted", "scan_result": scan_result,
                        "classifications": classifications}

        print(f"\n📦 {action} files to {output_directory}...")

        org_result = self.mover.organize_files(classifications, show_progress=show_progress)

        # Show final summary
        print("\n" + "=" * 60)
        if org_result.interrupted:
            print("⚠️  Organization Interrupted — partial run recorded")
        else:
            print("✅ Organization Complete!")
        print("=" * 60)
        print(org_result.get_summary())

        if org_result.undo_path:
            print(f"\n🔄 Undo script: {org_result.undo_path}")
            if org_result.interrupted:
                print("   (run it to put the already-moved files back)")

        return {
            "status": "success",
            "scan_result": scan_result,
            "classifications": classifications,
            "organization_result": org_result
        }
    
    def preview(
        self,
        input_dirs: Optional[list[str]] = None,
        limit: int = 10
    ) -> None:
        """
        Preview classification for a few files without organizing.
        
        Args:
            input_dirs: Input directories to scan.
            limit: Maximum number of files to preview.
        """
        input_directories = [Path(d) for d in (input_dirs or self.config.input_directories)]
        
        print("Preview Mode - No files will be moved")
        print("=" * 60)
        
        # Scan
        scan_result = self.scanner.scan_multiple(input_directories, self.config.recursive_scan)
        files = scan_result.file_paths[:limit]
        
        if not files:
            print("No files found!")
            return
        
        print(f"\nPreviewing {len(files)} files:\n")

        self._init_classifier()

        if self.config.backend == "llm" and self.config.show_data_notice:
            print(_data_notice(self.config))

        contents = [self.extractor.extract(file_path) for file_path in files]
        results = self.classifier.classify_batch(contents, show_progress=False)

        for content, result in zip(contents, results):
            print(f"📄 {content.file_path.name}")
            print(f"   → {result.category_path} (confidence: {result.final_confidence:.1%})")
            print()
    
    def show_categories(self) -> None:
        """Display the category tree."""
        print("Category Structure:")
        print("=" * 60)
        self.category_tree.print_tree()


def _setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure diagnostic logging. UX output stays on print(); this is for warnings/debug."""
    level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _read_ignore_patterns(input_dirs: list, extra: list) -> list:
    """Collect exclude globs from --exclude plus a .organizerignore in each input dir."""
    patterns = list(extra or [])
    for d in input_dirs:
        ignore_file = Path(d).expanduser() / ".organizerignore"
        if ignore_file.is_file():
            for line in ignore_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


def _data_notice(config) -> str:
    kinds = "filenames and text snippets"
    if config.llm_vision:
        kinds += " (and images)"
    return (f"   ℹ️  {kinds} will be sent to the '{config.llm_provider}' API "
            "for classification. Use --no-vision to withhold images; --backend local to stay offline.")


def _guard_output_not_in_input(input_dirs: list, output_dir) -> None:
    """Refuse to write the output inside (or equal to) any input directory."""
    out = Path(output_dir).expanduser().resolve()
    for d in input_dirs:
        ind = Path(d).expanduser().resolve()
        if out == ind or out.is_relative_to(ind):
            raise ValueError(
                f"Output '{out}' is inside input '{ind}'. Pick an output folder "
                "outside the folder you're organizing, or organized files will be "
                "re-scanned (and logs moved) on the next run."
            )


def _run_discovery(config: OrganizerConfig, out_path: Path) -> None:
    """Learn a category tree from a sample of the input files and save it."""
    import json
    from llm import build_provider, discover_categories
    from categories import CategoryTree

    scanner = DirectoryScanner(
        skip_hidden=config.skip_hidden_files,
        max_file_size_mb=config.max_file_size_mb,
    )
    extractor = ExtractorRegistry(enable_ocr=config.enable_ocr)
    input_dirs = [Path(d) for d in config.input_directories]

    print("📁 Scanning for a sample of files to learn from...")
    scan = scanner.scan_multiple(input_dirs, config.recursive_scan)
    if not scan.file_paths:
        raise ValueError("No files found to learn categories from.")

    # Sample at the path level so we don't extract the whole folder.
    paths = scan.file_paths
    cap = 60
    if len(paths) > cap:
        step = len(paths) / cap
        paths = [paths[int(i * step)] for i in range(cap)]
    contents = [extractor.extract(p) for p in paths]

    provider = build_provider(
        config.llm_provider,
        model=config.llm_model,
        base_url=config.llm_base_url,
        auto_install=config.llm_auto_install_sdk,
        project_dir=Path(__file__).parent,
    )
    print(f"   Using {provider.status_line()}")
    print(f"🤖 Proposing a category tree from {len(contents)} sampled files...")
    tree_dict = discover_categories(provider, contents)

    out_path = out_path.expanduser()
    out_path.write_text(json.dumps(tree_dict, indent=2), encoding="utf-8")

    print(f"\n✅ Saved proposed categories to {out_path}")
    print(f"   {provider.usage_summary()}")
    print("\nProposed structure:")
    CategoryTree.from_file(out_path).print_tree()
    print(
        f"\nEdit {out_path} to taste, then organize with it:\n"
        f"  python main.py --backend llm --categories-file {out_path} "
        f"{config.input_directories[0]} {config.output_directory}"
    )


def _print_providers() -> None:
    """List supported LLM providers and the env vars they read (no keys shown)."""
    from llm import list_presets

    print("Supported LLM providers (use with --backend llm --provider <name>):")
    print("=" * 70)
    for preset in list_presets():
        env = " / ".join(preset.env_names)
        default_model = preset.default_model or "(set with --model)"
        print(f"\n  {preset.key}  —  {preset.label}")
        print(f"      key env var : {env}" + ("  (optional)" if preset.keyless else ""))
        print(f"      default model: {default_model}")
        if preset.base_url:
            print(f"      endpoint     : {preset.base_url}")
    print(
        "\nKeys are read from the environment, a .env file, or a hidden prompt — "
        "never stored in config or logs.\n"
        "Example:  python main.py --backend llm --provider openai ~/Downloads ~/Organized"
    )


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="AI-Powered File Organizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local (free, offline) backend
  python main.py ./messy_folder ./organized
  python main.py --dry-run ./downloads
  python main.py --preview ./documents
  python main.py --show-categories

  # LLM backend (bring your own API key — any provider)
  python main.py --list-providers
  python main.py --backend llm --provider anthropic ./downloads ./organized
  python main.py --provider openai --model gpt-4o-mini --preview ./downloads
  python main.py --provider gemini ./downloads ./organized

API keys are read from environment variables or a .env file (never stored).
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        default="./input",
        help="Input directory to organize (default: ./input)"
    )
    
    parser.add_argument(
        "output",
        nargs="?",
        default="./organized",
        help="Output directory for organized files (default: ./organized)"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Simulate organization without moving files"
    )
    
    parser.add_argument(
        "--copy", "-c",
        action="store_true",
        help="Copy files instead of moving them"
    )
    
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview classification for a few files"
    )
    
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=10,
        help="Number of files to preview (default: 10)"
    )
    
    parser.add_argument(
        "--show-categories",
        action="store_true",
        help="Display the category structure"
    )
    
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable OCR for images (slower)"
    )
    
    parser.add_argument(
        "--backend",
        choices=["local", "llm"],
        help="Classification backend: 'local' (free, offline embeddings) or "
             "'llm' (bring-your-own-key, any provider). Default: local."
    )

    parser.add_argument(
        "--provider",
        help="LLM provider when --backend llm (e.g. anthropic, openai, gemini, "
             "groq, mistral, deepseek, together, openrouter, ollama, "
             "openai-compatible). See --list-providers."
    )

    parser.add_argument(
        "--model",
        help="Model to use. For --backend llm this is the LLM model id; for "
             "--backend local it is the embedding model."
    )

    parser.add_argument(
        "--embedding-model",
        help="Embedding model for the local backend (alias for --model)."
    )

    parser.add_argument(
        "--base-url",
        help="Custom endpoint URL for OpenAI-compatible providers."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Files per LLM request (higher = fewer requests / cheaper). Default: 20."
    )

    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List supported LLM providers and the env vars they read, then exit."
    )

    parser.add_argument(
        "--no-auto-install",
        action="store_true",
        help="Don't auto-install a provider's SDK; require it to be installed already."
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch the guided interactive setup (backend, provider, key, model, run)."
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt before moving files."
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        help="LLM backend: route files below this confidence (0-1) to a Review/ folder."
    )

    parser.add_argument(
        "--categories-file",
        help="Use a custom category tree (JSON/YAML) instead of the built-in defaults."
    )

    parser.add_argument(
        "--discover-categories",
        action="store_true",
        help="Use the LLM to propose a category tree from your files, save it, and exit. "
             "Saves to --categories-file if given, else ./categories.json."
    )

    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="LLM backend: don't send image files to the model as images (filename/OCR only)."
    )

    parser.add_argument(
        "--max-files",
        type=int,
        help="Refuse LLM runs with more than this many files (spend guard; 0 = no cap)."
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of parallel LLM requests (default 4)."
    )

    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob pattern of files to skip (repeatable). Also reads .organizerignore."
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose diagnostic logging."
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only log errors."
    )

    parser.add_argument(
        "--no-notice",
        action="store_true",
        help="Suppress the notice that file data is sent to the LLM provider."
    )

    parser.add_argument(
        "--config", "-f",
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    args = parser.parse_args()
    _setup_logging(args.verbose, args.quiet)

    # Informational mode that needs no config/key.
    if args.list_providers:
        _print_providers()
        return

    # Launch the interactive wizard when asked, or when run bare at a terminal.
    launch_wizard = args.interactive or (len(sys.argv) == 1 and sys.stdin.isatty())

    # Load or create config
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in ('.yml', '.yaml'):
            config = OrganizerConfig.from_yaml(config_path)
        else:
            config = OrganizerConfig.from_json(config_path)
    else:
        config = OrganizerConfig()

    # Apply CLI overrides
    config.input_directories = [args.input]
    config.output_directory = args.output
    config.dry_run = args.dry_run
    config.copy_instead_of_move = args.copy
    config.enable_ocr = args.enable_ocr

    if args.backend:
        config.backend = args.backend
    if args.provider:
        config.llm_provider = args.provider
        # Choosing a provider implies the LLM backend unless told otherwise.
        if not args.backend:
            config.backend = "llm"
    if args.base_url:
        config.llm_base_url = args.base_url
    if args.batch_size:
        config.llm_batch_size = args.batch_size
    if args.no_auto_install:
        config.llm_auto_install_sdk = False
    if args.min_confidence is not None:
        config.llm_min_confidence = args.min_confidence
    if args.categories_file:
        config.custom_categories_file = args.categories_file
    if args.no_vision:
        config.llm_vision = False
    if args.max_files is not None:
        config.llm_max_files = args.max_files
    if args.concurrency is not None:
        config.llm_concurrency = args.concurrency
    if args.no_notice:
        config.show_data_notice = False
    # Exclude globs come from --exclude and any .organizerignore in the inputs.
    config.exclude_patterns = _read_ignore_patterns(config.input_directories, args.exclude)

    # --model targets whichever backend is active; --embedding-model is explicit.
    if args.embedding_model:
        config.embedding_model = args.embedding_model
    if args.model:
        if config.backend == "llm":
            config.llm_model = args.model
        else:
            config.embedding_model = args.model

    # Category discovery is its own terminal action (needs an LLM provider).
    if args.discover_categories:
        out_path = Path(args.categories_file) if args.categories_file else Path("categories.json")
        try:
            _run_discovery(config, out_path)
        except (ImportError, RuntimeError, ValueError) as e:
            print(f"\n❌ {e}")
            sys.exit(1)
        return

    # Interactive wizard handles its own flow (config, key, model, run).
    if launch_wizard:
        from cli import run_interactive
        try:
            run_interactive(config)
        except (ImportError, RuntimeError, ValueError) as e:
            print(f"\n❌ {e}")
            sys.exit(1)
        return

    # Create organizer
    organizer = FileOrganizer(config)

    # Handle different modes. Setup problems (missing SDK, missing key, bad
    # provider) are expected user errors — show a clean message, not a traceback.
    try:
        if args.show_categories:
            organizer.show_categories()
            return

        if args.preview:
            organizer.preview(limit=args.preview_limit)
            return

        # Run organization
        organizer.organize(show_progress=not args.no_progress, assume_yes=args.yes)
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"\n❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
