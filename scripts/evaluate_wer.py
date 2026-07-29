from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import mean

from jiwer import wer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from asr.transcriber import Transcriber  # noqa: E402


WORD_PATTERN = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def normalize_for_wer(text: str) -> str:
    """Normalize reference and hypothesis text before WER calculation."""
    normalized = text.lower().replace("ё", "е")
    normalized = WORD_PATTERN.sub(" ", normalized)
    return " ".join(normalized.split())


def read_reference(reference_path: Path) -> str:
    """Read and validate a reference transcript."""
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference transcript not found: {reference_path}"
        )

    text = reference_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(
            f"Reference transcript is empty: {reference_path}"
        )

    return text


def transcribe_audio(
    transcriber: Transcriber,
    audio_path: Path,
) -> str:
    """Run ASR and join all non-empty segment texts."""
    segments = transcriber.transcribe(audio_path)

    if not segments:
        raise RuntimeError(
            f"ASR returned no segments for: {audio_path}"
        )

    transcript = " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    )

    if not transcript:
        raise RuntimeError(
            f"ASR returned an empty transcript for: {audio_path}"
        )

    return transcript


def evaluate_dataset(
    calls_dir: Path,
    references_dir: Path,
) -> list[tuple[str, float]]:
    """Calculate WER for every WAV file in the dataset."""
    if not calls_dir.is_dir():
        raise FileNotFoundError(
            f"Calls directory not found: {calls_dir}"
        )

    if not references_dir.is_dir():
        raise FileNotFoundError(
            f"References directory not found: {references_dir}"
        )

    audio_files = sorted(calls_dir.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found in: {calls_dir}"
        )

    transcriber = Transcriber()
    results: list[tuple[str, float]] = []

    for audio_path in audio_files:
        reference_path = references_dir / f"{audio_path.stem}.txt"

        reference_text = read_reference(reference_path)
        hypothesis_text = transcribe_audio(
            transcriber=transcriber,
            audio_path=audio_path,
        )

        normalized_reference = normalize_for_wer(reference_text)
        normalized_hypothesis = normalize_for_wer(hypothesis_text)

        if not normalized_reference:
            raise ValueError(
                f"Reference became empty after normalization: "
                f"{reference_path}"
            )

        if not normalized_hypothesis:
            raise RuntimeError(
                f"ASR hypothesis became empty after normalization: "
                f"{audio_path}"
            )

        score = float(
            wer(
                normalized_reference,
                normalized_hypothesis,
            )
        )

        results.append((audio_path.name, score))

    return results


def print_markdown_table(
    results: list[tuple[str, float]],
) -> None:
    """Print results in README-compatible Markdown."""
    print("| Файл | WER |")
    print("|---|---:|")

    for filename, score in results:
        print(f"| `{filename}` | `{score:.4f}` |")

    average_score = mean(
        score for _, score in results
    )

    print()
    print(
        f"Средний WER по набору: `{average_score:.4f}`"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate WER for test audio files using "
            "reference transcripts."
        )
    )

    parser.add_argument(
        "--calls-dir",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "calls",
        help=(
            "Directory containing WAV files. "
            "Default: test_data/calls"
        ),
    )

    parser.add_argument(
        "--references-dir",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "references",
        help=(
            "Directory containing reference TXT files. "
            "Default: test_data/references"
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run WER evaluation."""
    args = parse_args()

    try:
        results = evaluate_dataset(
            calls_dir=args.calls_dir,
            references_dir=args.references_dir,
        )
    except Exception as exc:
        print(
            f"WER evaluation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print_markdown_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())