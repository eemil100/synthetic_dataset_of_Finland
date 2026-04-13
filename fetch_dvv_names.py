from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


# DVV publishes the name-frequency spreadsheets via Avoindata (CKAN).
# Dataset (package) id/name (as of 2026): "vaestotietojarjestelman-suomalaisten-nimiaineistot"
DEFAULT_CKAN_BASE = "https://avoindata.suomi.fi/data"
# Note: the public dataset page currently uses the slug "none" in its URL
# (https://avoindata.suomi.fi/data/en_GB/dataset/none), and CKAN package_show works with id=none.
DEFAULT_PACKAGE_ID = "none"


@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    format: str
    created: str | None
    last_modified: str | None


def _get_json(url: str, timeout_s: float) -> Any:
    r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "dvv-names-fetch/1.0"})
    r.raise_for_status()
    return r.json()


def _package_show(ckan_base: str, package_id: str, timeout_s: float) -> dict[str, Any]:
    api = ckan_base.rstrip("/") + "/api/3/action/package_show"
    data = _get_json(f"{api}?id={package_id}", timeout_s=timeout_s)
    if not isinstance(data, dict) or not data.get("success"):
        raise RuntimeError(f"CKAN package_show failed for id={package_id!r}")
    result = data.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("CKAN package_show returned unexpected payload")
    return result


def _extract_resources(pkg: dict[str, Any]) -> list[Resource]:
    out: list[Resource] = []
    for r in pkg.get("resources", []) or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or r.get("description") or "resource")
        url = str(r.get("url") or "")
        fmt = str(r.get("format") or "")
        if not url:
            continue
        out.append(
            Resource(
                name=name,
                url=url,
                format=fmt,
                created=r.get("created"),
                last_modified=r.get("last_modified"),
            )
        )
    return out


def _pick_best(resources: list[Resource], kind: str) -> Resource | None:
    """
    kind: 'first' or 'last'
    """
    # DVV resources are typically named "Etunimitilasto ..." and "Sukunimitilasto ..."
    kind_terms = (
        ["etunimi", "etunimitilasto", "forename", "first"]
        if kind == "first"
        else ["sukunimi", "sukunimitilasto", "surname", "last"]
    )

    def score(res: Resource) -> tuple[int, int, int]:
        name_l = (res.name or "").lower()
        url_l = (res.url or "").lower()
        fmt_l = (res.format or "").lower()
        s = 0
        if "xlsx" in fmt_l or url_l.endswith(".xlsx"):
            s += 3
        if any(t in name_l or t in url_l for t in kind_terms):
            s += 3
        # Penalize selecting the opposite kind if both exist
        if kind == "first" and ("sukunimi" in name_l or "sukunimi" in url_l):
            s -= 5
        if kind == "last" and ("etunimi" in name_l or "etunimi" in url_l):
            s -= 5
        # Prefer newer resources if CKAN timestamps exist (string compare works for ISO timestamps)
        ts = res.last_modified or res.created or ""
        # Secondary tie-breaker: prefer more specific titles
        specific = 1 if ("tilasto" in name_l) else 0
        return (s, 1 if ts else 0, specific)

    ranked = sorted(resources, key=score, reverse=True)
    return ranked[0] if ranked else None


def _download(url: str, dest: Path, timeout_s: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s, headers={"User-Agent": "dvv-names-fetch/1.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download DVV name frequency lists (first + surname).")
    ap.add_argument("--ckan-base", default=DEFAULT_CKAN_BASE, help="CKAN base URL (default: avoindata.suomi.fi)")
    ap.add_argument("--package-id", default=DEFAULT_PACKAGE_ID, help="CKAN dataset/package id/name")
    ap.add_argument("--out-dir", default="data/dvv_names", help="Output directory (default: data/dvv_names)")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds (default: 30)")
    args = ap.parse_args()

    pkg = _package_show(args.ckan_base, args.package_id, timeout_s=max(1.0, args.timeout))
    resources = _extract_resources(pkg)
    if not resources:
        raise SystemExit("No resources found in the DVV CKAN package.")

    first = _pick_best(resources, "first")
    last = _pick_best(resources, "last")

    out_dir = Path(args.out_dir)
    downloaded = []

    if first:
        dest = out_dir / "etunimet.xlsx"
        _download(first.url, dest, timeout_s=max(1.0, args.timeout))
        downloaded.append(str(dest))

    if last:
        dest = out_dir / "sukunimet.xlsx"
        _download(last.url, dest, timeout_s=max(1.0, args.timeout))
        downloaded.append(str(dest))

    if not downloaded:
        raise SystemExit("Could not identify first/surname resources to download.")

    print("Downloaded:")
    for p in downloaded:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

