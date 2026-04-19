from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


STATFIN_BASE = "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin"

# Core tables (municipality-level where available)
TABLE_AGE_BY_AREA = f"{STATFIN_BASE}/vaerak/statfin_vaerak_pxt_11re.px"
TABLE_EDU_BY_MUNI = f"{STATFIN_BASE}/vkour/statfin_vkour_pxt_12bq.px"
TABLE_ACTIVITY_BY_AREA = f"{STATFIN_BASE}/tyokay/statfin_tyokay_pxt_115b.px"
TABLE_INCOME_DECILE_BY_AREA = f"{STATFIN_BASE}/tjt/statfin_tjt_pxt_12hh.px"
TABLE_FIN_SWE_BY_AREA = f"{STATFIN_BASE}/vaerak/statfin_vaerak_pxt_11s2.px"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ordered_category_codes(dim: dict[str, Any]) -> list[str]:
    cat = (dim.get("category") or {})
    idx = cat.get("index")
    labels = cat.get("label") or {}
    if isinstance(idx, dict):
        # code -> position
        return [c for c, _ in sorted(idx.items(), key=lambda kv: kv[1])]
    if isinstance(idx, list):
        return [str(x) for x in idx]
    # Fallback: use label keys as-is
    return [str(k) for k in labels.keys()]


def jsonstat_to_frame(js: dict[str, Any]) -> pd.DataFrame:
    ids: list[str] = list(js["id"])
    sizes: list[int] = list(js["size"])
    dims: dict[str, Any] = js["dimension"]
    values = js.get("value", [])

    levels: list[list[str]] = []
    for dim_id in ids:
        dim = dims[dim_id]
        codes = _ordered_category_codes(dim)
        labels = (dim.get("category") or {}).get("label") or {}
        levels.append([str(labels.get(c, c)) for c in codes])

    mi = pd.MultiIndex.from_product(levels, names=ids)
    s = pd.Series(values, index=mi, name="value", dtype="float64")
    df = s.reset_index()

    # Normalize possible nulls (PXWeb may return None for missing)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df


def pxweb_get_meta(url: str, timeout_s: float) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "statfin-phase2/1.0"})
    r.raise_for_status()
    return r.json()


def pxweb_post(url: str, query: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    r = requests.post(
        url,
        json=query,
        timeout=timeout_s,
        headers={"User-Agent": "statfin-phase2/1.0"},
    )
    r.raise_for_status()
    return r.json()


def _pick_latest_year(meta: dict[str, Any]) -> str:
    for v in meta.get("variables", []):
        if v.get("code") == "Vuosi":
            values = v.get("values") or []
            if not values:
                raise ValueError("No Year values found")
            # Values are typically ascending strings
            return str(values[-1])
    raise ValueError("No Year variable (Vuosi) found")


def _pick_values_by_label(
    meta: dict[str, Any],
    code: str,
    want_any_of: Iterable[str],
    fallback: str | None = None,
) -> list[str]:
    want = {w.strip().lower() for w in want_any_of}
    for v in meta.get("variables", []):
        if v.get("code") != code:
            continue
        values = list(v.get("values") or [])
        texts = list(v.get("valueTexts") or [])
        for val, txt in zip(values, texts):
            if str(txt).strip().lower() in want:
                return [str(val)]
        if fallback is not None and fallback in values:
            return [str(fallback)]
        return [str(values[0])] if values else []
    return []


def _all_area_values(meta: dict[str, Any]) -> list[str]:
    for v in meta.get("variables", []):
        if v.get("code") == "Alue":
            vals = [str(x) for x in (v.get("values") or [])]
            # Drop whole country if present
            return [x for x in vals if x not in {"SSS", "SSS "} and not x.upper().startswith("SSS")]
    raise ValueError("No Area variable (Alue) found")


@dataclass(frozen=True)
class DimensionDist:
    # municipality label -> (categories, probs)
    by_area: dict[str, tuple[np.ndarray, np.ndarray]]


def _make_dist(df: pd.DataFrame, area_col: str, cat_col: str) -> DimensionDist:
    by_area: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for area, g in df.groupby(area_col, sort=False):
        cats = g[cat_col].astype(str).to_numpy()
        w = g["value"].to_numpy(dtype="float64")
        w = np.clip(w, 0, None)
        total = float(w.sum())
        if total <= 0:
            continue
        by_area[str(area)] = (cats, (w / total))
    return DimensionDist(by_area=by_area)


def _drop_totals(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].astype(str)
    mask = ~s.str.fullmatch(r"(?i)total|all|both sexes|sexes total|age total|industries total", na=False)
    mask &= ~s.str.contains(r"(?i)total", na=False)
    return df.loc[mask].copy()


def _sample_conditional(
    rng: np.random.Generator,
    areas: np.ndarray,
    dist: DimensionDist,
    default_choice: str,
) -> np.ndarray:
    out = np.empty(len(areas), dtype=object)
    for area in np.unique(areas):
        idx = np.where(areas == area)[0]
        key = str(area)
        if key not in dist.by_area:
            out[idx] = default_choice
            continue
        cats, probs = dist.by_area[key]
        out[idx] = rng.choice(cats, size=len(idx), p=probs)
    return out.astype(str)


def _income_bracket(decile_label: str) -> str:
    # decile label is usually "1st decile", "2nd decile", ... or similar
    s = str(decile_label)
    num = None
    for token in s.split():
        token = token.strip().rstrip(".,")
        if token.isdigit():
            num = int(token)
            break
    if num is None:
        # Try leading digit like "1st decile"
        if s and s[0].isdigit():
            num = int(s[0])
    if num is None:
        roman = {
            "I": 1,
            "II": 2,
            "III": 3,
            "IV": 4,
            "V": 5,
            "VI": 6,
            "VII": 7,
            "VIII": 8,
            "IX": 9,
            "X": 10,
        }
        num = roman.get(s.strip().upper())
    if num is None:
        return "Unknown"
    if 1 <= num <= 3:
        return "Low"
    if 4 <= num <= 7:
        return "Medium"
    if 8 <= num <= 10:
        return "High"
    return "Unknown"


def fetch_or_load_csv(cache_path: Path, fetch_fn) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    df = fetch_fn()
    _ensure_dir(cache_path.parent)
    df.to_csv(cache_path, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2: Generate statistically grounded persona skeletons (10k by default)."
    )
    ap.add_argument("--n", type=int, default=10_000, help="Number of personas to generate (default: 10000)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument("--out-csv", default="personas_10000.csv", help="Output CSV path")
    ap.add_argument("--out-schema", default="persona_schema.json", help="Output schema JSON path")
    ap.add_argument("--cache-dir", default="data/cache_phase2", help="Cache directory for pulled tables")
    ap.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds (default: 60)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between API requests seconds (default: 0.2)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cache_dir = Path(args.cache_dir)

    def sleep():
        if args.sleep > 0:
            import time

            time.sleep(args.sleep)

    # 1) Age distribution by municipality (Age x Area), latest year, both sexes total
    def fetch_age():
        meta = pxweb_get_meta(TABLE_AGE_BY_AREA, timeout_s=args.timeout)
        year = _pick_latest_year(meta)
        areas = _all_area_values(meta)
        sex_total = _pick_values_by_label(meta, "Sukupuoli", ["Total", "Both sexes"])
        info = _pick_values_by_label(meta, "Tiedot", ["Population"], fallback=None) or [meta["variables"][-1]["values"][0]]
        query = {
            "query": [
                {"code": "Alue", "selection": {"filter": "item", "values": areas}},
                {"code": "Ikä", "selection": {"filter": "all", "values": ["*"]}},
                {"code": "Sukupuoli", "selection": {"filter": "item", "values": sex_total}},
                {"code": "Vuosi", "selection": {"filter": "item", "values": [year]}},
                {"code": "Tiedot", "selection": {"filter": "item", "values": info}},
            ],
            "response": {"format": "JSON-stat2"},
        }
        js = pxweb_post(TABLE_AGE_BY_AREA, query, timeout_s=args.timeout)
        df = jsonstat_to_frame(js)
        # Rename dims to stable english names
        df = df.rename(columns={"Alue": "Area", "Ikä": "Age", "Sukupuoli": "Sex", "Vuosi": "Year", "Tiedot": "Information"})
        return df[["Area", "Age", "value"]]

    age_df = fetch_or_load_csv(cache_dir / "age_by_area.csv", fetch_age)
    sleep()

    # 2) Gender distribution by municipality (Sex x Area), latest year, age total
    def fetch_gender():
        meta = pxweb_get_meta(TABLE_AGE_BY_AREA, timeout_s=args.timeout)
        year = _pick_latest_year(meta)
        areas = _all_area_values(meta)
        # pick male/female if present
        male = _pick_values_by_label(meta, "Sukupuoli", ["Males", "Men", "Male"])
        female = _pick_values_by_label(meta, "Sukupuoli", ["Females", "Women", "Female"])
        if not male or not female:
            # fallback: first two non-total values
            for v in meta.get("variables", []):
                if v.get("code") == "Sukupuoli":
                    vals = list(v.get("values") or [])
                    male, female = [vals[0]], [vals[1]] if len(vals) > 1 else [vals[0]]
        age_total = _pick_values_by_label(meta, "Ikä", ["Total", "All ages", "Age total"])
        info = _pick_values_by_label(meta, "Tiedot", ["Population"], fallback=None) or [meta["variables"][-1]["values"][0]]
        query = {
            "query": [
                {"code": "Alue", "selection": {"filter": "item", "values": areas}},
                {"code": "Ikä", "selection": {"filter": "item", "values": age_total}},
                {"code": "Sukupuoli", "selection": {"filter": "item", "values": male + female}},
                {"code": "Vuosi", "selection": {"filter": "item", "values": [year]}},
                {"code": "Tiedot", "selection": {"filter": "item", "values": info}},
            ],
            "response": {"format": "JSON-stat2"},
        }
        js = pxweb_post(TABLE_AGE_BY_AREA, query, timeout_s=args.timeout)
        df = jsonstat_to_frame(js).rename(columns={"Alue": "Area", "Sukupuoli": "Sex", "Ikä": "Age", "Tiedot": "Information"})
        return df[["Area", "Sex", "value"]]

    gender_df = fetch_or_load_csv(cache_dir / "gender_by_area.csv", fetch_gender)
    sleep()

    # 3) Education level distribution by municipality (Education level x Area), latest year, age total, gender total
    def fetch_education():
        meta = pxweb_get_meta(TABLE_EDU_BY_MUNI, timeout_s=args.timeout)
        year = _pick_latest_year(meta)
        areas = _all_area_values(meta)
        age_total = _pick_values_by_label(meta, "Ikä", ["Total", "All ages", "Age total"])
        sex_total = _pick_values_by_label(meta, "Sukupuoli", ["Total", "Both sexes", "Sexes total"])
        info = _pick_values_by_label(meta, "Tiedot", ["Population"], fallback=None) or [meta["variables"][-1]["values"][0]]
        query = {
            "query": [
                {"code": "Vuosi", "selection": {"filter": "item", "values": [year]}},
                {"code": "Alue", "selection": {"filter": "item", "values": areas}},
                {"code": "Ikä", "selection": {"filter": "item", "values": age_total}},
                {"code": "Sukupuoli", "selection": {"filter": "item", "values": sex_total}},
                {"code": "Koulutusaste", "selection": {"filter": "all", "values": ["*"]}},
                {"code": "Tiedot", "selection": {"filter": "item", "values": info}},
            ],
            "response": {"format": "JSON-stat2"},
        }
        js = pxweb_post(TABLE_EDU_BY_MUNI, query, timeout_s=args.timeout)
        df = jsonstat_to_frame(js).rename(columns={"Alue": "Area", "Koulutusaste": "Education"})
        return df[["Area", "Education", "value"]]

    edu_df = fetch_or_load_csv(cache_dir / "education_by_area.csv", fetch_education)
    sleep()

    # 4) Main type of activity by municipality, latest year (proxy for occupation)
    def fetch_activity():
        meta = pxweb_get_meta(TABLE_ACTIVITY_BY_AREA, timeout_s=args.timeout)
        year = _pick_latest_year(meta)
        areas = _all_area_values(meta)
        sex_total = _pick_values_by_label(meta, "Sukupuoli", ["Total", "Both sexes", "Sexes total"])
        age_total = _pick_values_by_label(meta, "Ikä", ["Total", "All ages", "Age total"])
        info = _pick_values_by_label(meta, "Tiedot", ["Population"], fallback=None) or [meta["variables"][-1]["values"][0]]

        query = {
            "query": [
                {"code": "Alue", "selection": {"filter": "item", "values": areas}},
                {"code": "Pääasiallinen toiminta", "selection": {"filter": "all", "values": ["*"]}},
                {"code": "Ikä", "selection": {"filter": "item", "values": age_total}},
                {"code": "Sukupuoli", "selection": {"filter": "item", "values": sex_total}},
                {"code": "Vuosi", "selection": {"filter": "item", "values": [year]}},
                {"code": "Tiedot", "selection": {"filter": "item", "values": info}},
            ],
            "response": {"format": "JSON-stat2"},
        }
        js = pxweb_post(TABLE_ACTIVITY_BY_AREA, query, timeout_s=args.timeout)
        df = jsonstat_to_frame(js).rename(columns={"Alue": "Area", "Pääasiallinen toiminta": "Activity"})
        return df[["Area", "Activity", "value"]]

    act_df = fetch_or_load_csv(cache_dir / "activity_by_area.csv", fetch_activity)
    sleep()

    # 5) Income decile distribution by municipality (Decile x Area), latest year
    def fetch_income_decile():
        meta = pxweb_get_meta(TABLE_INCOME_DECILE_BY_AREA, timeout_s=args.timeout)
        year = _pick_latest_year(meta)
        areas = _all_area_values(meta)
        info = _pick_values_by_label(meta, "Tiedot", ["Persons", "Number of persons"], fallback=None) or [meta["variables"][-1]["values"][0]]
        query = {
            "query": [
                {"code": "Vuosi", "selection": {"filter": "item", "values": [year]}},
                {"code": "Tulokymmenys tai fraktiiliryhmä", "selection": {"filter": "all", "values": ["*"]}},
                {"code": "Alue", "selection": {"filter": "item", "values": areas}},
                {"code": "Tiedot", "selection": {"filter": "item", "values": info}},
            ],
            "response": {"format": "JSON-stat2"},
        }
        js = pxweb_post(TABLE_INCOME_DECILE_BY_AREA, query, timeout_s=args.timeout)
        df = jsonstat_to_frame(js).rename(
            columns={"Alue": "Area", "Tulokymmenys tai fraktiiliryhmä": "Income decile"}
        )
        return df[["Area", "Income decile", "value"]]

    inc_df = fetch_or_load_csv(cache_dir / "income_decile_by_area.csv", fetch_income_decile)
    sleep()

    # 6) Finnish/Swedish/Other language shares by municipality (computed from table)
    def fetch_language():
        meta = pxweb_get_meta(TABLE_FIN_SWE_BY_AREA, timeout_s=args.timeout)
        year = _pick_latest_year(meta)
        areas = _all_area_values(meta)
        sex_total = _pick_values_by_label(meta, "Sukupuoli", ["Total", "Both sexes", "Sexes total"])
        age_total = _pick_values_by_label(meta, "Ikä", ["Total", "All ages", "Age total"])
        query = {
            "query": [
                {"code": "Vuosi", "selection": {"filter": "item", "values": [year]}},
                {"code": "Alue", "selection": {"filter": "item", "values": areas}},
                {"code": "Sukupuoli", "selection": {"filter": "item", "values": sex_total}},
                {"code": "Ikä", "selection": {"filter": "item", "values": age_total}},
                {"code": "Tiedot", "selection": {"filter": "all", "values": ["*"]}},
            ],
            "response": {"format": "JSON-stat2"},
        }
        js = pxweb_post(TABLE_FIN_SWE_BY_AREA, query, timeout_s=args.timeout)
        df = jsonstat_to_frame(js).rename(columns={"Alue": "Area", "Tiedot": "Language group"})
        return df[["Area", "Language group", "value"]]

    lang_df = fetch_or_load_csv(cache_dir / "language_by_area.csv", fetch_language)

    # Build dists
    muni_pop = age_df.groupby("Area", as_index=True)["value"].sum()
    muni_labels = muni_pop.index.to_numpy(dtype=str)
    muni_probs = (muni_pop.to_numpy(dtype="float64") / float(muni_pop.sum())).clip(0, 1)

    age_dist = _make_dist(age_df, "Area", "Age")
    gender_dist = _make_dist(gender_df, "Area", "Sex")
    edu_df = _drop_totals(edu_df, "Education")
    inc_df = _drop_totals(inc_df, "Income decile")

    edu_dist = _make_dist(edu_df, "Area", "Education")
    act_df = _drop_totals(act_df, "Activity")
    act_dist = _make_dist(act_df, "Area", "Activity")
    inc_dist = _make_dist(inc_df, "Area", "Income decile")
    lang_dist = _make_dist(lang_df, "Area", "Language group")

    # Sample N municipalities (weighted by population)
    n = int(args.n)
    areas = rng.choice(muni_labels, size=n, p=muni_probs)

    # Conditional samples
    ages = _sample_conditional(rng, areas, age_dist, default_choice="0")
    sexes = _sample_conditional(rng, areas, gender_dist, default_choice="Total")
    edus = _sample_conditional(rng, areas, edu_dist, default_choice="Unknown")
    acts = _sample_conditional(rng, areas, act_dist, default_choice="Unknown")
    decs = _sample_conditional(rng, areas, inc_dist, default_choice="Unknown")
    langs = _sample_conditional(rng, areas, lang_dist, default_choice="Unknown")

    # Normalize fields
    def to_int_age(a: str) -> int:
        try:
            return int(float(a))
        except ValueError:
            # some age labels can be ranges; pick midpoint if possible
            s = a.replace("years", "").replace("year", "").strip()
            if "-" in s:
                lo, hi = s.split("-", 1)
                try:
                    return int((int(lo) + int(hi)) / 2)
                except Exception:
                    return 0
            return 0

    age_int = np.array([to_int_age(x) for x in ages], dtype=int)
    income_bracket = np.array([_income_bracket(x) for x in decs], dtype=str)

    df_out = pd.DataFrame(
        {
            "age": age_int,
            "gender": sexes,
            "municipality": areas,
            "education_level": edus,
            "main_activity": acts,
            "income_decile": decs,
            "income_bracket": income_bracket,
            "language_group": langs,
        }
    )

    # ---------------------------------------------------------
    # POST-SAMPLING CORRECTIONS (Fixing the Independent Sampling bug of {Age, Education, Activity, Income} dimensions)
    # ---------------------------------------------------------
    
    # 1. Babies and Toddlers (Age 0-6)
    mask_child = df_out["age"] < 7
    df_out.loc[mask_child, "main_activity"] = "Child (Outside labour force)"
    df_out.loc[mask_child, "education_level"] = "No degree (Early childhood)"
    df_out.loc[mask_child, "income_decile"] = "Unknown"
    df_out.loc[mask_child, "income_bracket"] = "Unknown"

    # 2. School-aged Children (Age 7-15)
    mask_pupil = (df_out["age"] >= 7) & (df_out["age"] <= 15)
    df_out.loc[mask_pupil, "main_activity"] = "Pupil/Student"
    df_out.loc[mask_pupil, "education_level"] = "Comprehensive school (ongoing)"
    df_out.loc[mask_pupil, "income_decile"] = "Unknown"
    df_out.loc[mask_pupil, "income_bracket"] = "Unknown"

    # 3. Young Adults / High Schoolers (Age 16-18)
    # Demote anyone with a Bachelor's/Master's who is under 19
    mask_teen = (df_out["age"] >= 16) & (df_out["age"] <= 18)
    mask_too_educated = mask_teen & df_out["education_level"].str.contains(r"Bachelor|Master|Doctor", case=False, na=False)
    df_out.loc[mask_too_educated, "education_level"] = "Upper secondary education"
    
    # 4. Retirees (Age 65+)
    # Statistically, the vast majority are pensioners.
    mask_retiree = df_out["age"] >= 65
    df_out.loc[mask_retiree, "main_activity"] = "Pensioners"
    
    # ---------------------------------------------------------

    # Save CSV
    df_out.to_csv(args.out_csv, index=False)

    # Save schema JSON (simple, human-readable)
    schema = {
        "type": "object",
        "description": "Statistically grounded persona skeleton (no free-text guessing).",
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "gender": {"type": "string"},
            "municipality": {"type": "string", "description": "StatFin area label (municipality)"},
            "education_level": {"type": "string"},
            "main_activity": {"type": "string", "description": "Main type of activity (proxy for occupation)"},
            "income_decile": {"type": "string"},
            "income_bracket": {"type": "string", "enum": ["Low", "Medium", "High", "Unknown"]},
            "language_group": {"type": "string", "description": "Finnish/Swedish speaking counts from StatFin table; remaining languages are not separately represented here."},
        },
        "example": {
            "age": 34,
            "gender": "Females",
            "municipality": "Tampere",
            "main_activity": "Employed",
            "income_bracket": "High",
            "language_group": "Finnish-speaking population",
            "education_level": "Upper secondary education",
        },
    }
    Path(args.out_schema).write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_csv} with {len(df_out)} rows.")
    print(f"Wrote {args.out_schema}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

