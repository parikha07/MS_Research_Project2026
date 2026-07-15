#!/usr/bin/env python3
"""Image-only, zero-shot emotion recognition on EmoSet with the Groq API.

The image is the only sample-specific information sent to Grok. Ground-truth
labels are read locally and are used only after inference for evaluation.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageOps
from tqdm import tqdm


EMOTION_MAP = {
    "a": "awe",
    "b": "contentment",
    "c": "excitement",
    "d": "anger",
    "e": "sadness",
    "f": "amusement",
    "g": "fear",
    "h": "disgust",
}
EMOTIONS = tuple(EMOTION_MAP.values())
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUTPUT_COLUMNS = [
    "mode", "task", "sample_id", "image_filename", "predicted_label",
    "prediction_letter", "ground_truth", "correct", "reasoning",
    "visual_evidence", "raw_llm_output", "image_path", "model", "status",
    "api_status", "error", "error_message", "latency_seconds", "retry_count",
    "prompt_tokens", "completion_tokens", "total_tokens", "raw_response",
]

LABEL_ALIASES = {
    "amused": "amusement", "amusing": "amusement",
    "angry": "anger",
    "awed": "awe",
    "content": "contentment", "contented": "contentment",
    "disgusted": "disgust",
    "excited": "excitement",
    "fearful": "fear", "scared": "fear",
    "sad": "sadness",
}


def canonical_label(value: object) -> Optional[str]:
    if value is None:
        return None
    label = re.sub(r"[^a-z]+", " ", str(value).strip().lower()).strip()
    label = LABEL_ALIASES.get(label, label)
    return label if label in EMOTIONS else None


def resolve_column(fieldnames: Iterable[str], requested: Optional[str], candidates: List[str]) -> Optional[str]:
    names = list(fieldnames)
    lookup = {name.lower(): name for name in names}
    if requested:
        if requested in names:
            return requested
        return lookup.get(requested.lower())
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def find_image(root: Path, value: str, csv_dir: Path) -> Optional[Path]:
    candidate = Path(value)
    attempts = [candidate] if candidate.is_absolute() else [csv_dir / candidate, root / candidate]
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()

    basename = candidate.name
    matches = list(root.rglob(basename))
    return matches[0].resolve() if len(matches) == 1 else None


def load_from_csv(
    root: Path,
    csv_path: Path,
    image_column: Optional[str],
    label_column: Optional[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {csv_path}")
        image_col = resolve_column(
            reader.fieldnames, image_column,
            ["image_path", "image", "path", "filepath", "file_name", "filename", "name"],
        )
        label_col = resolve_column(
            reader.fieldnames, label_column,
            ["emotion", "label", "category", "ground_truth", "class"],
        )
        if image_col is None:
            raise ValueError(f"Could not identify an image column. Columns: {reader.fieldnames}")

        for index, row in enumerate(reader, start=1):
            raw_path = (row.get(image_col) or "").strip()
            image_path = find_image(root, raw_path, csv_path.parent) if raw_path else None
            if image_path is None:
                print(f"Warning: skipping row {index}; image not found: {raw_path}")
                continue
            ground_truth = canonical_label(row.get(label_col)) if label_col else None
            if label_col and ground_truth is None:
                print(f"Warning: row {index} has an unrecognized label: {row.get(label_col)!r}")
            rows.append({
                "sample_id": str(row.get("id") or row.get("image_id") or index),
                "image_path": image_path,
                "ground_truth": ground_truth,
            })
    return rows


def load_from_folders(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for image_path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        relative_parents = image_path.relative_to(root).parents
        ground_truth = next(
            (label for parent in relative_parents
             for label in [canonical_label(parent.name)] if label),
            None,
        )
        rows.append({
            "sample_id": image_path.stem,
            "image_path": image_path.resolve(),
            "ground_truth": ground_truth,
        })
    return rows


def load_from_official_split(root: Path, split: str) -> List[Dict[str, object]]:
    """Load the official EmoSet train.json, val.json, or test.json format.

    EmoSet distributions may place the image and annotation paths in different
    positions. Detect the image path by its extension rather than assuming a
    fixed index. Only the label, ID, and image path are needed here.
    """
    split_path = root / f"{split}.json"
    with split_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {split_path}")

    rows: List[Dict[str, object]] = []
    skipped = 0

    # Some EmoSet-118K distributions store only annotation paths in the split
    # files. Build one image index so those entries can be matched reliably
    # without performing an expensive recursive search for every sample.
    image_root = root / "image"
    image_index: Dict[str, Path] = {}
    if image_root.is_dir():
        for path in image_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_index[path.stem.lower()] = path.resolve()

    for index, item in enumerate(data, start=1):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            skipped += 1
            if skipped <= 10:
                print(f"Warning: skipping malformed {split} entry {index}: {item!r}")
            continue
        ground_truth = canonical_label(item[0])
        image_candidates = [
            (root / str(value)).resolve()
            for value in item[2:]
            if Path(str(value)).suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_path = next((path for path in image_candidates if path.is_file()), None)

        # Local EmoSet variant: item contains an annotation path but no image
        # path. Example annotation/awe/awe_06628.json -> image/awe/awe_06628.jpg.
        if image_path is None:
            annotation_values = [
                Path(str(value)) for value in item[2:]
                if Path(str(value)).suffix.lower() == ".json"
            ]
            for annotation_path in annotation_values:
                stem = annotation_path.stem.lower()
                indexed_path = image_index.get(stem)
                if indexed_path is not None:
                    image_path = indexed_path
                    break

        if ground_truth is None:
            skipped += 1
            if skipped <= 10:
                print(f"Warning: skipping entry {index}; invalid emotion label: {item[0]!r}")
            continue
        if image_path is None:
            skipped += 1
            if skipped <= 10:
                print(
                    f"Warning: skipping entry {index}; no matching image found "
                    f"for: {item[2:]!r}"
                )
            continue
        rows.append({
            "sample_id": str(item[1]),
            "image_path": image_path,
            "ground_truth": ground_truth,
        })
    if skipped:
        print(f"Skipped {skipped} invalid or unmatched {split} entries in total.")
    return rows


def load_emoset(
    dataset_path: str,
    metadata_csv: Optional[str],
    image_column: Optional[str],
    label_column: Optional[str],
    split: str,
    limit: Optional[int],
) -> List[Dict[str, object]]:
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    if metadata_csv:
        csv_path = Path(metadata_csv).expanduser().resolve()
        samples = load_from_csv(root, csv_path, image_column, label_column)
    elif (root / f"{split}.json").is_file():
        print(f"Using official EmoSet split: {root / f'{split}.json'}")
        samples = load_from_official_split(root, split)
    else:
        samples = load_from_folders(root)

    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No images were found. Check the dataset path and metadata columns.")
    return samples


def encode_image(image_path: Path, max_side: int = 2048) -> str:
    """Normalize an image to RGB JPEG and return a base64 data URL."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_label_from_text(text: str) -> str:
    """Use the same label-extraction strategies as the reference baseline."""
    if not text:
        return "x"
    text = text.strip().lower()
    if len(text) == 1 and text in "abcdefgh":
        return text

    valid_letters = [char for char in reversed(text) if char in "abcdefgh"]
    if valid_letters:
        return valid_letters[0]

    response_pattern = r"(?:response|answer|result)?\s*:?\s*([abcdefgh])"
    matches = re.findall(response_pattern, text)
    if matches:
        return matches[-1]

    common_phrases = [
        r"the answer is\s*([abcdefgh])",
        r"i choose\s*([abcdefgh])",
        r"my response is\s*([abcdefgh])",
        r"therefore\s*([abcdefgh])",
        r"so\s*([abcdefgh])",
        r"thus\s*([abcdefgh])",
        r"final answer\s*:?\s*([abcdefgh])",
        r"conclusion\s*:?\s*([abcdefgh])",
    ]
    for pattern in common_phrases:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]

    matches = re.findall(r"\b([abcdefgh])\b", text)
    if matches:
        return matches[-1]

    parts = re.split(r"[.,!?;:\n\t\s]+", text)
    for part in reversed(parts):
        if len(part) == 1 and part in "abcdefgh":
            return part

    for char in text:
        if char in "abcdefgh":
            return char
    return "x"


def parse_baseline_response(text: str) -> tuple[str, str, str]:
    """Parse Reasoning/Final Answer exactly as in the reference baseline."""
    generated_text = text.strip()
    if "Final Answer:" in generated_text:
        parts = generated_text.split("Final Answer:")
        reasoning = parts[0].strip()
        if reasoning.startswith("Reasoning:"):
            reasoning = reasoning[10:].strip()
        prediction_letter = extract_label_from_text(parts[1].strip())
    else:
        reasoning = generated_text
        if reasoning.startswith("Reasoning:"):
            reasoning = reasoning[10:].strip()
        prediction_letter = extract_label_from_text(generated_text)
    predicted_label = EMOTION_MAP.get(prediction_letter, "unknown")
    return prediction_letter, predicted_label, reasoning


class GroqEmoSetAnalyzer:
    def __init__(self, api_key: str, model: str, timeout: float, max_retries: int) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install openai pillow") from exc
        # Groq exposes an OpenAI-compatible Responses API with vision support.
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout,
        )
        self.model = model
        self.max_retries = max_retries

    def classify(self, image_path: Path) -> Dict[str, object]:
        prompt = """Analyze the primary emotion expressed in this image using the dataset's ground-truth emotion schema.

**Classification Options:**
a = awe
b = contentment
c = excitement
d = anger
e = sadness
f = amusement
g = fear
h = disgust

**Instructions:**
First, explain your reasoning step-by-step. Then provide your final answer as a single letter.

**Format your response exactly as:**
Reasoning: [your analysis]
Final Answer: [letter]"""
        try:
            image_url = encode_image(image_path)
        except Exception as exc:
            return {
                "prediction_letter": "x", "predicted_emotion": "unknown", "reasoning": "", "status": "failed",
                "error": f"Image loading failed: {exc}", "latency_seconds": 0.0,
                "retry_count": 0,
                "prompt_tokens": "", "completion_tokens": "", "total_tokens": "",
                "raw_response": "",
            }

        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": image_url, "detail": "high"},
                            {"type": "input_text", "text": prompt},
                        ],
                    }],
                    store=False,
                    temperature=0,
                    max_output_tokens=512,
                )
                latency = time.perf_counter() - started
                raw = response.output_text.strip()
                prediction_letter, prediction, reasoning = parse_baseline_response(raw)
                usage = getattr(response, "usage", None)
                return {
                    "predicted_emotion": prediction,
                    "prediction_letter": prediction_letter,
                    "reasoning": reasoning,
                    "status": "success",
                    "error": "",
                    "retry_count": attempt,
                    "latency_seconds": round(latency, 4),
                    "prompt_tokens": getattr(usage, "input_tokens", "") if usage else "",
                    "completion_tokens": getattr(usage, "output_tokens", "") if usage else "",
                    "total_tokens": getattr(usage, "total_tokens", "") if usage else "",
                    "raw_response": raw,
                }
            except Exception as exc:
                if attempt >= self.max_retries:
                    return {
                        "prediction_letter": "x", "predicted_emotion": "unknown", "reasoning": "", "status": "failed",
                        "error": str(exc), "latency_seconds": round(time.perf_counter() - started, 4),
                        "retry_count": attempt,
                        "prompt_tokens": "", "completion_tokens": "", "total_tokens": "",
                        "raw_response": "",
                    }
                delay = min(60.0, (2 ** attempt) + random.random())
                print(f"Request failed ({exc}); retrying in {delay:.1f}s")
                time.sleep(delay)
        raise AssertionError("unreachable")


def completed_keys(output_path: Path, model: str, retry_failed: bool) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        keys = set()
        for row in csv.DictReader(handle):
            done = row.get("status") == "success" or not retry_failed
            if done and row.get("model") == model:
                keys.add(str(Path(row["image_path"]).resolve()))
        return keys


def append_result(output_path: Path, row: Dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def save_json_results(output_path: Path, model: str) -> Path:
    """Create a detailed JSON companion file from the checkpointed CSV."""
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    model_rows = [row for row in csv_rows if row.get("model") == model]
    result_rows = []
    for row in model_rows:
        ground_truth = canonical_label(row.get("ground_truth"))
        prediction = canonical_label(row.get("predicted_emotion"))
        result_rows.append({
            "sample_id": row.get("sample_id"),
            "image_path": row.get("image_path"),
            "prediction_letter": row.get("prediction_letter") or None,
            "predicted_label": prediction,
            "ground_truth": ground_truth,
            "correct": prediction == ground_truth if prediction and ground_truth else None,
            "reasoning": row.get("reasoning") or None,
            "visual_evidence": row.get("visual_evidence") or row.get("reasoning") or None,
            "success": row.get("status") == "success",
            "api_status": row.get("status"),
            "error_message": row.get("error") or None,
            "retry_count": int(row["retry_count"]) if row.get("retry_count") else 0,
            "latency_seconds": float(row["latency_seconds"]) if row.get("latency_seconds") else None,
            "prompt_tokens": int(row["prompt_tokens"]) if row.get("prompt_tokens") else None,
            "completion_tokens": int(row["completion_tokens"]) if row.get("completion_tokens") else None,
            "total_tokens": int(row["total_tokens"]) if row.get("total_tokens") else None,
            "raw_llm_output": row.get("raw_llm_output") or row.get("raw_response") or None,
        })

    # Keep only operational run counts here. Aggregate evaluation metrics are
    # intentionally deferred to the separate end-of-experiment reporting step.
    summary = {
        "total_stored_rows": len(result_rows),
        "successful_requests": sum(r["success"] for r in result_rows),
        "failed_requests": sum(not r["success"] for r in result_rows),
    }

    json_output = {
        "results": {"image_only": {"emotion": result_rows}},
        "summary": {"image_only": {"emotion": summary}},
        "config": {
            "dataset": "EmoSet",
            "experiment": "image-only zero-shot emotion classification",
            "model": model,
            "emotion_labels": list(EMOTIONS),
            "caption_used": False,
            "ground_truth_sent_to_model": False,
            "resume_enabled": True,
        },
    }

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_output, handle, indent=2, ensure_ascii=False)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Groq image-only zero-shot classification on EmoSet")
    parser.add_argument("--dataset-path", required=True, help="EmoSet root or image directory")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="test",
        help="Official EmoSet JSON split to run (default: test)",
    )
    parser.add_argument("--metadata-csv", help="Optional CSV containing image paths and hidden ground truth")
    parser.add_argument("--image-column", help="CSV image-path column; auto-detected if omitted")
    parser.add_argument("--label-column", help="CSV emotion-label column; auto-detected if omitted")
    parser.add_argument(
        "--output-path", default="results/grok_emoset_image_results.csv",
        help="CSV output path or basename; the matching JSON is created automatically",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        help="Groq vision model",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed rows when resuming")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. In PowerShell use: "
            "$env:GROQ_API_KEY='your_groq_key'"
        )

    samples = load_emoset(
        args.dataset_path, args.metadata_csv, args.image_column, args.label_column,
        args.split, args.limit
    )
    output_path = Path(args.output_path).expanduser()
    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_suffix(".csv")
    output_path = output_path.resolve()
    done = completed_keys(output_path, args.model, args.retry_failed)
    pending = [s for s in samples if str(Path(s["image_path"]).resolve()) not in done]
    labels_found = sum(s["ground_truth"] is not None for s in samples)
    completed_count = len(samples) - len(pending)
    print("\n" + "=" * 72)
    print("EMOSET IMAGE-ONLY EXPERIMENT STATUS")
    print("=" * 72)
    print(f"Split: {args.split}")
    print(f"Selected dataset size: {len(samples):,} images")
    print(f"Images with local ground truth: {labels_found:,}")
    print(f"Already completed (skipped): {completed_count:,}")
    print(f"Remaining to process: {len(pending):,}")
    print(f"Model: {args.model}")
    print(f"Output CSV: {output_path}")
    print("=" * 72 + "\n")

    analyzer = GroqEmoSetAnalyzer(api_key, args.model, args.timeout, args.max_retries)
    with tqdm(
        total=len(samples),
        initial=completed_count,
        desc="Image-Only Emotion Analysis",
        unit="image",
        dynamic_ncols=True,
    ) as progress:
        for sample in pending:
            image_path = Path(sample["image_path"])
            result = analyzer.classify(image_path)
            prediction = result.get("predicted_emotion")
            ground_truth = sample.get("ground_truth")
            row = {
                **sample,
                **result,
                "mode": "image_only",
                "task": "emotion",
                "image_filename": image_path.name,
                "predicted_label": prediction,
                "image_path": str(image_path.resolve()),
                "ground_truth": ground_truth or "",
                "correct": (prediction == ground_truth) if prediction and ground_truth else "",
                "visual_evidence": result.get("reasoning", ""),
                "raw_llm_output": result.get("raw_response", ""),
                "api_status": result.get("status", ""),
                "error_message": result.get("error", ""),
                "model": args.model,
            }
            append_result(output_path, row)
            progress.set_postfix(
                file=image_path.name,
                result=prediction or "unknown",
                status=result.get("status", "unknown"),
                refresh=False,
            )
            progress.update(1)

    json_path = save_json_results(output_path, args.model)
    print(f"CSV results saved to: {output_path}")
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
