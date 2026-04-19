from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"

def _normalize_gemini_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return "models/gemini-2.5-flash"
    return m if m.startswith("models/") else f"models/{m}"

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of a single top-level JSON object from model text.
    Handles occasional code fences and trailing commas.
    """
    if text is None:
        return ""
    s = str(text).strip()
    s = _FENCE_RE.sub("", s).strip()

    # If model returns extra text, take the first {...} block.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    # Remove common trailing commas: {"a":1,}
    for _ in range(5):
        new_s = _TRAILING_COMMA_RE.sub(r"\1", s)
        if new_s == s:
            break
        s = new_s
    return s


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _weighted_sample(
    rng: np.random.Generator, items: np.ndarray, weights: np.ndarray, k: int
) -> list[str]:
    weights = weights.astype("float64")
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        idx = rng.choice(len(items), size=k, replace=False)
        return [str(items[i]) for i in idx]
    p = weights / weights.sum()
    idx = rng.choice(len(items), size=k, replace=False if k <= len(items) else True, p=p)
    return [str(items[i]) for i in np.atleast_1d(idx)]


@dataclass(frozen=True)
class NameCandidates:
    first_female: list[str]
    first_male: list[str]
    last: list[str]


def load_dvv_name_candidates(
    etunimet_xlsx: str,
    sukunimet_xlsx: str,
    seed: int,
    k_first: int = 200,
    k_last: int = 200,
) -> NameCandidates:
    rng = np.random.default_rng(seed)

    fn = pd.read_excel(etunimet_xlsx)
    fn = fn.rename(columns={fn.columns[0]: "name", fn.columns[1]: "count"})
    fn["name"] = fn["name"].astype(str).str.strip()
    fn["count"] = pd.to_numeric(fn["count"], errors="coerce").fillna(0)
    fn = fn[fn["name"].ne("") & fn["count"].gt(0)]

    # DVV file here is not gender-separated; we provide same pool for both to keep it grounded.
    first_pool = fn["name"].to_numpy()
    first_w = fn["count"].to_numpy(dtype="float64")

    ln = pd.read_excel(sukunimet_xlsx)
    ln = ln.rename(columns={ln.columns[0]: "name", ln.columns[1]: "count"})
    ln["name"] = ln["name"].astype(str).str.strip()
    ln["count"] = pd.to_numeric(ln["count"], errors="coerce").fillna(0)
    ln = ln[ln["name"].ne("") & ln["count"].gt(0)]
    last_pool = ln["name"].to_numpy()
    last_w = ln["count"].to_numpy(dtype="float64")

    first_candidates = _weighted_sample(rng, first_pool, first_w, k_first)
    last_candidates = _weighted_sample(rng, last_pool, last_w, k_last)

    return NameCandidates(
        first_female=first_candidates,
        first_male=first_candidates,
        last=last_candidates,
    )


def gemini_generate_json(
    api_key: str,
    model: str,
    system_prompt: str,
    user_obj: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    model_name = _normalize_gemini_model(model)
    url = f"{GEMINI_API}/{model_name}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(user_obj, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }

    backoff_s = 1.0
    last_err: Exception | None = None
    for attempt in range(1, 6):
        try:
            r = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"retryable status {r.status_code}", response=r)
            r.raise_for_status()
            data = r.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                raise RuntimeError(f"Unexpected Gemini response shape: {data}")

            cleaned = _extract_json_object(text)
            return json.loads(cleaned)
        except Exception as e:
            last_err = e
            if attempt >= 5:
                break
            time.sleep(backoff_s)
            backoff_s = min(20.0, backoff_s * 2)

    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: Enrich skeleton CSV rows into narrative JSON via an LLM.")
    ap.add_argument("--in-csv", default="personas_10000.csv", help="Input skeleton CSV")
    ap.add_argument("--out-jsonl", default="personas_enriched.jsonl", help="Output JSONL path")
    ap.add_argument("--system-prompt", default="phase3_prompt_system.txt", help="System prompt file")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name (e.g. gemini-2.5-flash)")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Env var holding Gemini API key")
    ap.add_argument("--limit", type=int, default=0, help="Limit rows (0=all)")
    ap.add_argument("--resume", action="store_true", help="Resume by skipping already-written rows")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between calls seconds")
    ap.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    ap.add_argument("--seed", type=int, default=42, help="Seed for name shortlist sampling")
    ap.add_argument("--dvv-first", default="data/dvv_names/etunimet.xlsx", help="DVV first names XLSX")
    ap.add_argument("--dvv-last", default="data/dvv_names/sukunimet.xlsx", help="DVV surnames XLSX")
    ap.add_argument("--name-k-first", type=int, default=200, help="How many DVV first names to offer")
    ap.add_argument("--name-k-last", type=int, default=200, help="How many DVV surnames to offer")
    args = ap.parse_args()

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set env var {args.api_key_env}=... or change --api-key-env.")

    system_prompt = read_text(args.system_prompt)

    # Pre-sample a DVV-grounded shortlist; we pass it to every row to reduce name hallucination.
    candidates = load_dvv_name_candidates(
        args.dvv_first,
        args.dvv_last,
        seed=args.seed,
        k_first=args.name_k_first,
        k_last=args.name_k_last,
    )
    name_candidates_obj = {
        "first_female": candidates.first_female,
        "first_male": candidates.first_male,
        "last": candidates.last,
    }

    df = pd.read_csv(args.in_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already = 0
    if args.resume and out_path.exists():
        already = sum(1 for _ in out_path.open("r", encoding="utf-8"))

    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(df.itertuples(index=False), start=0):
            if args.resume and i < already:
                continue

            skeleton = row._asdict()
            payload = {"skeleton": skeleton, "name_candidates": name_candidates_obj}

            try:
                enriched = gemini_generate_json(
                    api_key=api_key,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_obj=payload,
                    timeout_s=max(5.0, args.timeout),
                )
            except Exception as e:
                # minimal retry with lower temperature by asking again
                time.sleep(1.0)
                enriched = gemini_generate_json(
                    api_key=api_key,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_obj=payload,
                    timeout_s=max(5.0, args.timeout),
                )

            # Basic validation: ensure required keys exist
            if not isinstance(enriched, dict) or "skeleton" not in enriched or "name" not in enriched:
                raise RuntimeError(f"Invalid enriched object at row {i}: {enriched}")

            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

