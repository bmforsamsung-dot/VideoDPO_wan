#!/usr/bin/env python3
"""Extract video_name entries whose captions describe gravity-related falling events."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _iter_examples(payload):
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        for key in ("data", "train", "examples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return
        # Fallback: treat dict values as row-like objects
        for value in payload.values():
            if isinstance(value, dict) and ("caption" in value or "captions" in value):
                yield value


def _caption_text(example: dict) -> str:
    captions = example.get("captions")
    if isinstance(captions, list):
        return " ".join(str(x) for x in captions)
    if isinstance(captions, str):
        return captions
    for key in ("caption", "text", "description"):
        v = example.get(key)
        if isinstance(v, str):
            return v
    return ""


def _is_gravity_fall_caption(text: str, pattern: re.Pattern[str]) -> bool:
    if not text:
        return False
    return bool(pattern.search(text.lower()))


def extract_video_names(input_json: Path, output_txt: Path) -> int:
    with input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # "falling" 맥락(추락/넘어짐/낙하) 키워드를 느슨하게 매칭.
    pattern = re.compile(
        r"\b(fall(?:s|ing|en)?|fell|drop(?:s|ped|ping)?|plunge(?:s|d|ing)?|"
        r"tumble(?:s|d|ing)?|slip(?:s|ped|ping)?|trip(?:s|ped|ping)?|"
        r"topple(?:s|d|ing)?|collapse(?:s|d|ing)?)\b|"
        r"(추락|낙하|떨어지|미끄러|넘어지|전도)"
    )

    seen = set()
    matched: list[str] = []
    for example in _iter_examples(payload):
        if not isinstance(example, dict):
            continue
        caption_text = _caption_text(example)
        if not _is_gravity_fall_caption(caption_text, pattern):
            continue
        video_name = example.get("video_name") or example.get("video") or example.get("id")
        if not isinstance(video_name, str):
            continue
        if video_name in seen:
            continue
        seen.add(video_name)
        matched.append(video_name)

    matched.sort()
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(matched) + ("\n" if matched else ""), encoding="utf-8")
    return len(matched)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("data/wisa-80k.json"),
        help="Path to wisa-80k.json",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("data/wisa-80k_gravity_fall_video_names.txt"),
        help="Output txt path containing only video_name values",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = extract_video_names(args.input_json, args.output_txt)
    print(f"Done. matched_video_names={count} output={args.output_txt}")


if __name__ == "__main__":
    main()
