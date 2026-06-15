import argparse
from pathlib import Path

from lexetta_lcp.CompLexPerAnnotator.model import load_trained, push_to_hub


def main():
    parser = argparse.ArgumentParser(description="Push a trained per-annotator model to the Hugging Face Hub")
    parser.add_argument("path", help="Run directory of the trained per-annotator model")
    parser.add_argument("repo_id", help="Target Hub repo id, e.g. my-org/complex-per-annotator")
    parser.add_argument("--public", action="store_true", help="Create the repo as public (default: private)")
    parser.add_argument("--commit-message", default=None, help="Commit message for the upload")
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (falls back to the cached `huggingface-cli login` if omitted)",
    )
    args = parser.parse_args()

    model_dir = Path(args.path) / "model"

    print(f"Loading model from {model_dir}...")
    model, tokenizer = load_trained(str(model_dir))

    print(f"Pushing to {args.repo_id} ({'public' if args.public else 'private'})...")
    push_to_hub(
        model,
        tokenizer,
        repo_id=args.repo_id,
        private=not args.public,
        commit_message=args.commit_message,
        token=args.token,
    )
    print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
