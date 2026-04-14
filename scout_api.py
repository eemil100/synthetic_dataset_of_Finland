from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests


DEFAULT_TOPIC_QUERIES = [
    "Population structure",
    "Age distribution",
    "Household income",
    # Additional grounding dimensions for synthetic personas
    "Education",
    "Educational level",
    "Occupation",
    "Employment",
    "Language",
    "Mother tongue",
]

GEO_QUERIES = [
    "municipality",
    "postal code",
    "postcode",
    "zip code",
    "postal area",
    "kunta",
    "postinumero",
]


DEFAULT_BASES = [
    # StatFin (English)
    "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin",
    # Paavo (postal code area open data) – commonly hosted as its own PXWeb database
    "https://pxdata.stat.fi/PXWeb/api/v1/en/Postinumeroalueittainen_avoin_tieto",
    # Alternate hostname sometimes used for PXWeb (kept as fallback)
    "https://pxnet2.stat.fi/PXWeb/api/v1/en/Postinumeroalueittainen_avoin_tieto",
]


_WORD_RE = re.compile(r"[A-Za-zÅÄÖåäö]+(?:'[A-Za-z]+)?|\d+")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(s or "")}


def _score(text: str, queries: Iterable[str]) -> int:
    text_tokens = _tokens(text)
    score = 0
    for q in queries:
        q_tokens = _tokens(q)
        if not q_tokens:
            continue
        overlap = len(text_tokens.intersection(q_tokens))
        if overlap == len(q_tokens):
            score += 3
        elif overlap > 0:
            score += 1
    return score



def _norm_base(url: str) -> str:
    return url.rstrip("/")


def _join(base: str, path: str) -> str:
    base = _norm_base(base)
    path = path.strip("/")
    if not path:
        return base + "/"
    return f"{base}/{path}/"


def _get_json(session: requests.Session, url: str, timeout_s: float) -> Any:
    backoff_s = 1.0
    for attempt in range(1, 6):
        resp = session.get(url, timeout=timeout_s)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()

        retry_after = resp.headers.get("retry-after")
        if retry_after:
            try:
                wait_s = float(retry_after)
            except ValueError:
                wait_s = backoff_s
        else:
            wait_s = backoff_s

        time.sleep(max(0.0, wait_s))
        backoff_s = min(30.0, backoff_s * 2)

    resp.raise_for_status()
    return resp.json()


@dataclass(frozen=True)
class TableHit:
    api_base: str
    table_id: str
    path: str
    title: str
    score: int
    geo_score: int


@dataclass(frozen=True)
class CrawlResult:
    api_base: str
    pages_crawled: int
    hits: list[TableHit]
    error: str | None


def _is_leaf_table(node: Any) -> bool:
    # PXWeb directory listing typically returns a list of objects; table items often have:
    # { "id": "...", "text": "...", "type": "t" } or { ..., "type": "table" }
    if not isinstance(node, dict):
        return False
    t = (node.get("type") or "").lower()
    return t in {"t", "table"}


def _node_id_text(node: dict[str, Any]) -> tuple[str, str]:
    return str(node.get("id") or ""), str(node.get("text") or "")


def discover_tables(
    api_base: str,
    topic_queries: list[str],
    max_pages: int,
    sleep_s: float,
    timeout_s: float,
) -> CrawlResult:
    """
    BFS crawl of a PXWeb API tree, starting at api_base.
    For each directory path P, we call GET {api_base}/{P}/ which returns children.
    """
    api_base = _norm_base(api_base)
    session = requests.Session()
    session.headers.update({"User-Agent": "statfin-scout/1.0 (+python requests)"})

    hits: list[TableHit] = []
    visited: set[str] = set()

    q: deque[str] = deque([""])  # '' means root
    pages = 0
    last_error: str | None = None

    while q and pages < max_pages:
        rel = q.popleft()
        if rel in visited:
            continue
        visited.add(rel)

        url = _join(api_base, rel)
        try:
            data = _get_json(session, url, timeout_s=timeout_s)
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            continue

        pages += 1

        if isinstance(data, list):
            for child in data:
                if not isinstance(child, dict):
                    continue
                cid, ctext = _node_id_text(child)
                if not cid:
                    continue

                if _is_leaf_table(child):
                    # Table endpoint is {api_base}/{rel}/{cid}
                    table_path = "/".join([p for p in [rel.strip("/"), cid] if p])
                    title = ctext
                    hay = f"{title} {table_path}"
                    score = _score(hay, topic_queries)
                    geo_score = _score(hay, GEO_QUERIES)
                    # Keep tables that have any topical match; you can widen later by setting --min-score 0
                    if score > 0:
                        hits.append(
                            TableHit(
                                api_base=api_base,
                                table_id=cid,
                                path=table_path,
                                title=title,
                                score=score,
                                geo_score=geo_score,
                            )
                        )
                else:
                    # Directory node: enqueue {rel}/{cid}
                    next_rel = "/".join([p for p in [rel.strip("/"), cid] if p])
                    q.append(next_rel)

        if sleep_s > 0:
            time.sleep(sleep_s)

    return CrawlResult(api_base=api_base, pages_crawled=pages, hits=hits, error=last_error)


def write_summary(
    out_path: str,
    topic_queries: list[str],
    crawl_results: list[CrawlResult],
    max_pages: int,
    geo_only: bool,
) -> None:
    all_hits: list[TableHit] = []
    for cr in crawl_results:
        all_hits.extend(cr.hits)

    if geo_only:
        all_hits = [h for h in all_hits if h.geo_score > 0]

    results_sorted = sorted(
        all_hits,
        key=lambda h: (-(h.score), -(h.geo_score), h.api_base, h.table_id, h.path),
    )

    lines: list[str] = []
    lines.append("API table scout summary")
    lines.append("")
    lines.append(f"Topics: {', '.join(topic_queries)}")
    lines.append(f"Max pages per API base: {max_pages}")
    lines.append(f"API bases tried: {len(crawl_results)}")
    lines.append(f"Geo-only filter: {geo_only}")
    lines.append("")
    lines.append("Crawl status:")
    for cr in crawl_results:
        if cr.pages_crawled > 0 and cr.error is None:
            status = "ok"
        elif cr.pages_crawled > 0 and cr.error is not None:
            status = "partial"
        else:
            status = "failed"
        lines.append(
            f"  - {cr.api_base} | {status} | pages={cr.pages_crawled} | hits={len(cr.hits)}"
            + (f" | last_error={cr.error}" if cr.error else "")
        )
    lines.append("")
    lines.append(f"Matches found: {len(results_sorted)}")
    lines.append("")
    lines.append("Format: [topic_score, geo_score] table_id | path | title | api_base")
    lines.append("")

    for h in results_sorted:
        lines.append(
            f"[{h.score},{h.geo_score}] {h.table_id} | {h.path} | {h.title} | {h.api_base}"
        )

    lines.append("")
    lines.append("Note:")
    lines.append(
        "  - 'table_id' is the leaf ID; the full PXWeb path is in 'path'."
    )
    lines.append(
        "  - If you want broader results, rerun with --min-score 0 to include everything crawled."
    )
    lines.append(
        "  - If you want only municipality/postal-code flavored tables, rerun with --geo-only."
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Scout StatFin + Paavo PXWeb APIs for relevant tables."
    )
    p.add_argument(
        "--out",
        default="api_tables.txt",
        help="Output file for table summary (default: api_tables.txt)",
    )
    p.add_argument(
        "--base",
        action="append",
        default=[],
        help="PXWeb API base URL to crawl. Can be provided multiple times.",
    )
    p.add_argument(
        "--topic",
        action="append",
        default=[],
        help="Topic query term to match (can be provided multiple times). "
        "If omitted, uses a default list (population/age/income/education/occupation/language).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Max directory pages to crawl per base (default: 500)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Sleep between requests in seconds (default: 0.05)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds (default: 20)",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=1,
        help="Minimum topic match score to include (default: 1)",
    )
    p.add_argument(
        "--geo-only",
        action="store_true",
        help="Only include tables that also look municipality/postal-code related",
    )
    args = p.parse_args(argv)

    bases = [_norm_base(b) for b in (args.base or [])] or DEFAULT_BASES
    topic_queries = args.topic or DEFAULT_TOPIC_QUERIES
    crawl_results: list[CrawlResult] = []

    for base in bases:
        cr = discover_tables(
            api_base=base,
            topic_queries=topic_queries,
            max_pages=args.max_pages,
            sleep_s=max(0.0, args.sleep),
            timeout_s=max(1.0, args.timeout),
        )
        if args.min_score > 0:
            filtered = [h for h in cr.hits if h.score >= args.min_score]
            cr = CrawlResult(
                api_base=cr.api_base,
                pages_crawled=cr.pages_crawled,
                hits=filtered,
                error=cr.error,
            )
        crawl_results.append(cr)

    write_summary(
        out_path=args.out,
        topic_queries=topic_queries,
        crawl_results=crawl_results,
        max_pages=args.max_pages,
        geo_only=bool(args.geo_only),
    )

    total = sum(len(cr.hits) for cr in crawl_results)
    print(f"Wrote {args.out} with {total} matching tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

