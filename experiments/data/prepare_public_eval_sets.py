import argparse
import json
from pathlib import Path


def export_wikitext2(output_path: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    kept = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip()
            if not text:
                continue
            if text.startswith("="):
                continue
            if len(text.split()) < 5:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def export_chinese_webtext(output_path: Path, min_quality: float) -> int:
    from datasets import load_dataset

    ds = load_dataset("CASIA-LM/ChineseWebText2.0", split="test")
    kept = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in ds:
            text = str(item["text"]).strip()
            if not text:
                continue
            if len(text) < 20:
                continue
            quality = float(item.get("quality_score", 0.0))
            if quality < min_quality:
                continue
            record = {
                "text": text,
                "quality_score": quality,
                "domain": item.get("domain"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Export public evaluation corpora to local jsonl files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/data",
        help="Directory where exported jsonl files will be written.",
    )
    parser.add_argument(
        "--cn-min-quality",
        type=float,
        default=0.5,
        help="Minimum quality_score kept for ChineseWebText2.0 test rows.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wikitext_out = output_dir / "wikitext2_test_public_eval.jsonl"
    chinese_out = output_dir / "chinesewebtext2_test_public_eval.jsonl"

    wikitext_count = export_wikitext2(wikitext_out)
    chinese_count = export_chinese_webtext(chinese_out, min_quality=args.cn_min_quality)

    print(f"[OK] exported {wikitext_count} rows -> {wikitext_out}")
    print(f"[OK] exported {chinese_count} rows -> {chinese_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
