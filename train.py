#!/usr/bin/env python3
"""
Training script for the file organizer.

Usage:
    python train.py diagnose <file>          # See why a file is classified a certain way
    python train.py add-example <file> <category>  # Add a training example
    python train.py add-folder <folder> <category> # Add all files in folder as examples
    python train.py review <folder>          # Interactive review session
    python train.py suggest                  # Get threshold suggestions
    python train.py apply                    # Apply suggested thresholds
"""

import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from categories import create_default_categories
from trainer import Trainer, interactive_training_session


TRAINING_DATA_FILE = Path(__file__).parent / "training_data.json"


def get_trainer(model: str = "all-MiniLM-L6-v2") -> Trainer:
    """Get or create a trainer instance."""
    from classifier.embedder import Embedder
    
    tree = create_default_categories()
    embedder = Embedder(model_name=model)
    trainer = Trainer(category_tree=tree, embedder=embedder)
    
    # Load existing training data if available
    if TRAINING_DATA_FILE.exists():
        trainer.load(TRAINING_DATA_FILE)
    
    return trainer


def cmd_diagnose(args):
    """Diagnose how a file would be classified."""
    file_path = Path(args.file).expanduser()
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    backend = getattr(args, "backend", "local")
    if backend == "llm":
        _diagnose_llm(args, file_path)
    else:
        trainer = get_trainer(args.model)
        trainer.print_diagnosis(file_path)


def _diagnose_llm(args, file_path: Path) -> None:
    """LLM-backed diagnosis: show the top candidate folders with reasons."""
    from llm import build_provider, LLMClassifier
    from extractors import ExtractorRegistry

    tree = create_default_categories()
    provider = build_provider(
        getattr(args, "provider", None) or "anthropic",
        model=getattr(args, "llm_model", None),
        project_dir=Path(__file__).parent,
    )
    classifier = LLMClassifier(category_tree=tree, provider=provider, vision=True)
    content = ExtractorRegistry(enable_ocr=False).extract(file_path)

    print(f"\n{'='*60}\nDiagnosis for: {file_path.name}\n{'='*60}")
    print(f"Provider: {provider.status_line()}\n")
    candidates = classifier.diagnose(content)
    if not candidates:
        print("No candidates returned.")
        return
    print("Top candidate folders (best first):")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c['path']}  ({c['confidence']:.0%})")
        print(f"       {c['reason']}")
    print(f"\n   {provider.usage_summary()}")


def cmd_add_example(args):
    """Add a single file as a training example."""
    trainer = get_trainer(args.model)
    file_path = Path(args.file).expanduser()
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    trainer.add_example(file_path, args.category)
    trainer.save(TRAINING_DATA_FILE)


def cmd_add_folder(args):
    """Add all files in a folder as training examples."""
    trainer = get_trainer(args.model)
    folder_path = Path(args.folder).expanduser()
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    count = trainer.add_examples_from_folder(folder_path, args.category)
    print(f"\nAdded {count} examples for {args.category}")
    trainer.save(TRAINING_DATA_FILE)


def cmd_review(args):
    """Interactive review session."""
    trainer = get_trainer(args.model)
    folder_path = Path(args.folder).expanduser()
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    # Get files recursively
    files = []
    for f in folder_path.rglob("*"):
        if f.is_file() and not f.name.startswith('.'):
            # Skip common non-document files
            if f.suffix.lower() in {'.ds_store', '.localized', ''}:
                continue
            files.append(f)
    
    if not files:
        print(f"No files found in {folder_path}")
        return
    
    print(f"Found {len(files)} files")
    
    if args.limit:
        files = files[:args.limit]
    
    interactive_training_session(trainer, files)
    trainer.save(TRAINING_DATA_FILE)


def cmd_suggest(args):
    """Show threshold suggestions based on training data."""
    trainer = get_trainer(args.model)
    
    if not trainer.training_data:
        print("No training data available. Add some examples first:")
        print("  python train.py add-example <file> <category>")
        print("  python train.py add-folder <folder> <category>")
        return
    
    suggestions = trainer.suggest_threshold_adjustments()
    
    if not suggestions:
        print("Not enough training data to make suggestions.")
        return
    
    print("\nThreshold Suggestions")
    print("=" * 60)
    
    for category, data in suggestions.items():
        print(f"\n{category}")
        print(f"  Current threshold: {data['current_threshold']}")
        print(f"  Suggested:         {data['suggested_threshold']}")
        print(f"  Based on {data['num_examples']} examples")
        print(f"  Example similarities: min={data['example_similarities']['min']}, "
              f"max={data['example_similarities']['max']}, "
              f"mean={data['example_similarities']['mean']}")


def cmd_apply(args):
    """Apply threshold suggestions."""
    trainer = get_trainer(args.model)
    
    if not trainer.training_data:
        print("No training data available.")
        return
    
    suggestions = trainer.suggest_threshold_adjustments()
    
    if not suggestions:
        print("No suggestions available.")
        return
    
    print("\nApplying threshold adjustments:")
    trainer.apply_threshold_suggestions(suggestions)
    
    print("\nNote: To make these permanent, update the thresholds in")
    print("categories/default_categories.py")


def cmd_show_categories(args):
    """Show the category tree."""
    tree = create_default_categories()
    tree.print_tree()


def cmd_stats(args):
    """Show training statistics."""
    trainer = get_trainer(args.model)
    
    print("\nTraining Statistics")
    print("=" * 60)
    print(f"Corrections recorded: {len(trainer.corrections)}")
    print(f"Categories with examples: {len(trainer.training_data)}")
    
    total_examples = 0
    for cat_path, data in sorted(trainer.training_data.items()):
        count = len(data.examples)
        total_examples += count
        print(f"  {cat_path}: {count} examples")
    
    print(f"\nTotal examples: {total_examples}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and calibrate the file organizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See why a file is classified a certain way (local embeddings)
  python train.py diagnose ~/Downloads/assignment3.pdf

  # Or ask an LLM for the top candidate folders + reasons
  python train.py diagnose --backend llm --provider anthropic ~/Downloads/assignment3.pdf

  # Note: add-example / add-folder / suggest / apply / stats tune the LOCAL
  # embedding backend's thresholds — they don't apply to the LLM backend.

  # Add a training example
  python train.py add-example ~/Documents/cs246_notes.pdf "School/University/CS 246"

  # Add all files in a folder as examples for a category  
  python train.py add-folder ~/Documents/CS246 "School/University/CS 246"

  # Interactive review of files in a folder
  python train.py review ~/Downloads --limit 20

  # See suggested threshold adjustments
  python train.py suggest

  # Show current training stats
  python train.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Global model argument
    model_help = "Embedding model: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (accurate)"
    
    # diagnose (works with both backends)
    p_diagnose = subparsers.add_parser("diagnose", help="Diagnose how a file would be classified")
    p_diagnose.add_argument("file", help="File to diagnose")
    p_diagnose.add_argument("--backend", choices=["local", "llm"], default="local",
                            help="Use the local embedding backend or an LLM provider.")
    p_diagnose.add_argument("--provider", help="LLM provider when --backend llm (default: anthropic).")
    p_diagnose.add_argument("--llm-model", help="LLM model id when --backend llm.")
    p_diagnose.add_argument("--model", default="all-mpnet-base-v2", help=model_help)
    p_diagnose.set_defaults(func=cmd_diagnose)
    
    # add-example
    p_add = subparsers.add_parser("add-example", help="Add a training example")
    p_add.add_argument("file", help="File to add")
    p_add.add_argument("category", help="Correct category path (e.g., 'School/University/CS 246')")
    p_add.add_argument("--model", default="all-mpnet-base-v2", help=model_help)
    p_add.set_defaults(func=cmd_add_example)
    
    # add-folder
    p_folder = subparsers.add_parser("add-folder", help="Add folder contents as examples")
    p_folder.add_argument("folder", help="Folder containing examples")
    p_folder.add_argument("category", help="Category for all files in folder")
    p_folder.add_argument("--model", default="all-mpnet-base-v2", help=model_help)
    p_folder.set_defaults(func=cmd_add_folder)
    
    # review
    p_review = subparsers.add_parser("review", help="Interactive review session")
    p_review.add_argument("folder", help="Folder to review")
    p_review.add_argument("--limit", type=int, help="Max files to review")
    p_review.add_argument("--model", default="all-mpnet-base-v2", help=model_help)
    p_review.set_defaults(func=cmd_review)
    
    # suggest
    p_suggest = subparsers.add_parser("suggest", help="Show threshold suggestions")
    p_suggest.add_argument("--model", default="all-mpnet-base-v2", help=model_help)
    p_suggest.set_defaults(func=cmd_suggest)
    
    # apply
    p_apply = subparsers.add_parser("apply", help="Apply threshold suggestions")
    p_apply.add_argument("--model", default="all-mpnet-base-v2", help=model_help)
    p_apply.set_defaults(func=cmd_apply)
    
    # categories
    p_cats = subparsers.add_parser("categories", help="Show category tree")
    p_cats.set_defaults(func=cmd_show_categories)
    
    # stats
    p_stats = subparsers.add_parser("stats", help="Show training statistics")
    p_stats.add_argument("--model", default="all-mpnet-base-v2", help=model_help)
    p_stats.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
