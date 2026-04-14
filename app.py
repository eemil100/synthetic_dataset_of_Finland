from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


DEFAULT_GEOJSON_PATH = "data/fin_municipalities.geojson"
DEFAULT_GEOJSON_URL = "https://raw.githubusercontent.com/varmais/maakunnat/master/kunnat.geojson"

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"

def _normalize_gemini_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return "models/gemini-2.5-flash"
    return m if m.startswith("models/") else f"models/{m}"


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _norm_name(s: str) -> str:
    s = str(s).strip().lower()
    # Fix common bilingual municipality name mismatches between different datasets.
    bilingual = {
        "ingå": "inkoo",
        "jakobstad": "pietarsaari",
        "kimitoön": "kemiönsaari",
        "korsholm": "mustasaari",
        "kristinestad": "kristiinankaupunki",
        "kronoby": "kruunupyy",
        "larsmo": "luoto",
        "malax": "maalahti",
        "mariehamn": "maarianhamina",
        "nykarleby": "uusikaarlepyy",
        "närpes": "närpiö",
        "pargas": "parainen",
        "pedersöre": "pedersören kunta",
        "raseborg": "raasepori",
        "vörå": "vöyri",
    }
    return bilingual.get(s, s)


@st.cache_data(show_spinner=False)
def load_skeletons(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "municipality" not in df.columns:
        raise ValueError("CSV must have a 'municipality' column")
    df["municipality_norm"] = df["municipality"].map(_norm_name)
    return df


@st.cache_data(show_spinner=False)
def load_enriched_jsonl(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sk = obj.get("skeleton") or {}
            if not isinstance(sk, dict):
                continue
            # flatten a few narrative fields for optional filtering
            sk["name_full"] = ((obj.get("name") or {}).get("full")) if isinstance(obj.get("name"), dict) else None
            sk["transport_primary"] = obj.get("transport_primary")
            sk["life_goal"] = obj.get("life_goal")
            rows.append(sk)
    df = pd.DataFrame(rows)
    df["municipality_norm"] = df["municipality"].map(_norm_name)
    return df


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_geojson(url: str, dest_path: str) -> None:
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def _extract_feature_muni_name(feat: dict[str, Any]) -> Optional[str]:
    props = feat.get("properties") or {}
    if not isinstance(props, dict):
        return None
    for key in ["Kunta", "name", "NAME", "municipality", "Municipality"]:
        if key in props and props[key]:
            return str(props[key])
    return None


def _geojson_muni_index(geo: dict[str, Any]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    feats = geo.get("features") or []
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        name = _extract_feature_muni_name(feat)
        if not name:
            continue
        idx[_norm_name(name)] = feat
    return idx


def gemini_simulate(
    api_key: str,
    model: str,
    system_prompt: str,
    question: str,
    persona: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    model_name = _normalize_gemini_model(model)
    url = f"{GEMINI_API}/{model_name}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": json.dumps({"question": question, "persona": persona}, ensure_ascii=False)}
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 256,
            "responseMimeType": "application/json",
        },
    }
    r = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def heuristic_sentiment(question: str, df_muni: pd.DataFrame, seed: int) -> Tuple[float, str]:
    """
    Offline baseline: lightweight heuristic using income_bracket + car/public transport hints in question.
    """
    q = question.lower()
    rng = np.random.default_rng(seed)
    base = 0.0
    if any(k in q for k in ["increase taxes", "raise taxes", "higher taxes", "tax increase"]):
        # higher income slightly more opposed on avg; lower income slightly more supportive
        br = df_muni.get("income_bracket")
        if br is not None:
            w = br.value_counts(normalize=True).to_dict()
            base += 0.25 * w.get("Low", 0) - 0.15 * w.get("High", 0)
    if any(k in q for k in ["train", "rail", "public transport", "bus", "tram"]):
        base += 0.10
    if any(k in q for k in ["road", "highway", "parking"]):
        base -= 0.05
    base += float(rng.normal(0, 0.05))
    base = float(np.clip(base, -1.0, 1.0))
    if base > 0.2:
        resp = "Generally supportive, but wants clear benefits and fair funding."
    elif base < -0.2:
        resp = "Generally opposed, preferring lower costs and cautious spending."
    else:
        resp = "Mixed or uncertain, weighing costs against practical benefits."
    return base, resp


def municipality_scores(
    df: pd.DataFrame,
    question: str,
    sample_per_muni: int,
    max_munis: int,
    use_llm: bool,
    gemini_key: str,
    gemini_model: str,
    system_prompt: str,
    sleep_s: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    munis = df["municipality"].dropna().astype(str).unique().tolist()
    munis = sorted(munis)
    if max_munis and len(munis) > max_munis:
        munis = munis[:max_munis]

    out_rows = []
    prog = st.progress(0.0)
    status = st.empty()

    for i, muni in enumerate(munis):
        g = df[df["municipality"] == muni]
        if g.empty:
            continue
        n = min(sample_per_muni, len(g))
        g_s = g.sample(n=n, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))

        sentiments = []
        responses = []
        if use_llm:
            for _, r in g_s.iterrows():
                persona = r.dropna().to_dict()
                try:
                    sim = gemini_simulate(
                        api_key=gemini_key,
                        model=gemini_model,
                        system_prompt=system_prompt,
                        question=question,
                        persona=persona,
                        timeout_s=60.0,
                    )
                    s = float(sim.get("sentiment", 0.0))
                    s = float(np.clip(s, -1.0, 1.0))
                    sentiments.append(s)
                    responses.append(str(sim.get("response", "")).strip())
                except Exception:
                    # fall back to heuristic for that persona
                    s, resp = heuristic_sentiment(question, g, seed + i)
                    sentiments.append(s)
                    responses.append(resp)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        else:
            s, resp = heuristic_sentiment(question, g, seed + i)
            sentiments.append(s)
            responses.append(resp)

        avg = float(np.mean(sentiments)) if sentiments else 0.0
        out_rows.append(
            {
                "municipality": muni,
                "municipality_norm": _norm_name(muni),
                "sentiment": avg,
                "sample_n": int(n),
                "response_hint": next((x for x in responses if x), ""),
            }
        )

        prog.progress((i + 1) / max(1, len(munis)))
        status.write(f"Scored {i+1}/{len(munis)} municipalities…")

    status.empty()
    prog.empty()
    return pd.DataFrame(out_rows)


def sentiment_color(v: float) -> str:
    # blue (negative) -> white (neutral) -> red (positive)
    v = float(np.clip(v, -1.0, 1.0))
    if v >= 0:
        r = int(255)
        g = int(255 * (1 - v))
        b = int(255 * (1 - v))
    else:
        r = int(255 * (1 + v))
        g = int(255 * (1 + v))
        b = int(255)
    return f"#{r:02x}{g:02x}{b:02x}"


def make_map(geo: dict[str, Any], scores: pd.DataFrame) -> folium.Map:
    idx = dict(zip(scores["municipality_norm"], scores["sentiment"]))
    resp_idx = dict(zip(scores["municipality_norm"], scores.get("response_hint", "")))
    n_idx = dict(zip(scores["municipality_norm"], scores.get("sample_n", 0)))

    m = folium.Map(location=[64.5, 26.0], zoom_start=5, tiles="cartodbpositron", control_scale=True)

    def style_fn(feat: dict[str, Any]) -> dict[str, Any]:
        name = _extract_feature_muni_name(feat) or ""
        key = _norm_name(name)
        v = idx.get(key, 0.0)
        return {
            "fillColor": sentiment_color(v),
            "color": "#333333",
            "weight": 0.6,
            "fillOpacity": 0.65,
        }

    def tooltip_fields(feat: dict[str, Any]) -> str:
        name = _extract_feature_muni_name(feat) or ""
        key = _norm_name(name)
        v = idx.get(key, 0.0)
        hint = resp_idx.get(key, "")
        n = n_idx.get(key, 0)
        return f"{name}<br>Sentiment: {v:+.2f} (n={n})<br>{hint}"

    gj = folium.GeoJson(
        geo,
        name="Municipalities",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=[], aliases=[], labels=False),
    )
    # Replace tooltip content via onEachFeature-like approach using Popup for simplicity
    for feat in geo.get("features", []):
        if not isinstance(feat, dict):
            continue
        folium.GeoJson(
            feat,
            style_function=style_fn,
            tooltip=folium.Tooltip(tooltip_fields(feat), sticky=True),
        ).add_to(m)

    return m


st.set_page_config(page_title="Synthetic Finland – Opinion Map", layout="wide")

st.title("Synthetic Finland – Interactive Opinion Map")
st.caption("Phase 4: ask a question, simulate persona responses, map sentiment by municipality.")

with st.sidebar:
    st.header("Data")
    geo_path = st.text_input("Municipality GeoJSON path", value=DEFAULT_GEOJSON_PATH)
    if not Path(geo_path).exists():
        st.warning("GeoJSON not found. You can download a free municipality GeoJSON.")
        if st.button("Download GeoJSON (free)"):
            download_geojson(DEFAULT_GEOJSON_URL, geo_path)
            st.success(f"Downloaded to {geo_path}")

    data_mode = st.radio("Persona data source", ["Skeleton CSV", "Enriched JSONL"], index=0)
    if data_mode == "Skeleton CSV":
        in_path = st.text_input("Input CSV", value="personas_10000.csv")
    else:
        in_path = st.text_input("Input JSONL", value="personas_enriched.jsonl")

    st.header("Simulation")
    question = st.text_area(
        "Question",
        value="Should we increase taxes to fund better train lines?",
        height=80,
    )
    sample_per_muni = st.slider("Personas sampled per municipality", 1, 30, 5)
    max_munis = st.slider("Max municipalities to score (cost control)", 10, 320, 120)

    st.header("LLM")
    use_llm = st.checkbox("Use Gemini (LLM) for simulation", value=False)
    gemini_model = st.text_input("Gemini model", value="gemini-2.5-flash")
    api_key_env = st.text_input("API key env var", value="GEMINI_API_KEY")
    sleep_s = st.slider("Sleep between LLM calls (sec)", 0.0, 2.0, 0.2, 0.1)
    seed = st.number_input("Seed", value=42, step=1)

    st.header("Run")
    run = st.button("Simulate and map")


if run:
    if not Path(geo_path).exists():
        st.error("Missing GeoJSON. Download it in the sidebar first.")
        st.stop()
    if not Path(in_path).exists():
        st.error(f"Missing input data file: {in_path}")
        st.stop()
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    with st.spinner("Loading data…"):
        geo = load_geojson(geo_path)
        if data_mode == "Skeleton CSV":
            df = load_skeletons(in_path)
        else:
            df = load_enriched_jsonl(in_path)

    system_prompt = _read_text("phase4_prompt_system.txt")
    api_key = os.getenv(api_key_env, "") if use_llm else ""
    if use_llm and not api_key:
        st.error(f"LLM enabled but missing env var {api_key_env}.")
        st.stop()

    st.info("Scoring municipalities (this can take a while with LLM enabled)…")
    scores = municipality_scores(
        df=df,
        question=question.strip(),
        sample_per_muni=int(sample_per_muni),
        max_munis=int(max_munis),
        use_llm=bool(use_llm),
        gemini_key=api_key,
        gemini_model=gemini_model.strip(),
        system_prompt=system_prompt,
        sleep_s=float(sleep_s),
        seed=int(seed),
    )

    st.subheader("Map")
    m = make_map(geo, scores)
    st_folium(m, width=1100, height=700)

    st.subheader("Scores (table)")
    st.dataframe(scores.sort_values("sentiment", ascending=False), use_container_width=True)

else:
    st.write("Configure data + question in the sidebar, then click **Simulate and map**.")

