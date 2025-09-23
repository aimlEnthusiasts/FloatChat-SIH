import os
import json
import re
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
from decouple import config
from argopy import DataFetcher, set_options
import google.generativeai as genai

import visualize

# -------------------- CONFIG --------------------

API_KEY = config("GEMINI_API_KEY", default="").strip()
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        MODEL = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.warning(f"Gemini model init failed: {e}")
        MODEL = None
else:
    st.info(
        "No Gemini API key found — explanation features will use a fallback parser."
    )
    MODEL = None

# Use 0-360 longitude convention to avoid dateline issues in argopy
set_options(api_timeout=300, longitude_convention="360")

# -------------------- CONSTANTS --------------------
OCEAN_BBOX = {
    "indian ocean": [20, 120, -40, 30],
    "atlantic ocean": [-80, 20, -40, 70],
    "pacific west": [120, 180, -50, 60],
    "pacific east": [-180, -70, -50, 60],
    "southern ocean": [-180, 180, -90, -50],
    "arctic ocean": [-180, 180, 50, 90],
}

VAR_MAP = {
    "temperature": ["temp", "temp_adjusted", "temp_qc"],
    "salinity": ["psal", "psal_adjusted", "psal_qc"],
    "pressure": ["pres", "pres_adjusted", "pres_qc"],
}


# -------------------- HELPERS --------------------
def parse_json_from_text(text: str):
    """Return first JSON object found in text, or None."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def normalize_longitude(lon):
    try:
        lon = float(lon)
    except Exception:
        return None
    if lon < 0:
        lon = lon + 360
    return lon % 360


def normalize_bbox(bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_min_n = normalize_longitude(lon_min)
    lon_max_n = normalize_longitude(lon_max)
    lat_min_n = max(-90, min(90, float(lat_min)))
    lat_max_n = max(-90, min(90, float(lat_max)))
    return [lon_min_n, lon_max_n, lat_min_n, lat_max_n]


def split_bbox_if_wrap(bbox_norm):
    lon_min, lon_max, lat_min, lat_max = bbox_norm
    # If normalization mapped a global band (e.g., -180..180) to equal bounds (e.g., 180..180),
    # treat it as the full wrap in [0, 360) to satisfy argopy's strict lon_min < lon_max.
    if lon_min == lon_max:
        return [[0.0, 359.9999, lat_min, lat_max]]
    if lon_min < lon_max:
        return [[lon_min, lon_max, lat_min, lat_max]]
    # Dateline-crossing case
    return [[lon_min, 360.0, lat_min, lat_max], [0.0, lon_max, lat_min, lat_max]]


def expand_variables(natural_vars):
    if not natural_vars:
        return []
    expanded = []
    for v in natural_vars:
        k = v.lower()
        if k in VAR_MAP:
            expanded.extend(VAR_MAP[k])
        else:
            expanded.append(k)
    out = []
    seen = set()
    for x in expanded:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def detect_ocean(user_text: str):
    if not user_text:
        return None, None
    t = re.sub(r"[^\w\s]", " ", user_text.lower())
    for key in OCEAN_BBOX.keys():
        if key in t:
            if "pacific" in key and "west" not in key and "east" not in key:
                return "pacific", None
            return key, OCEAN_BBOX[key]
    if "pacific" in t:
        return "pacific", None
    if "atlantic" in t:
        return "atlantic ocean", OCEAN_BBOX["atlantic ocean"]
    if "indian" in t:
        return "indian ocean", OCEAN_BBOX["indian ocean"]
    if "southern" in t or "antarctic" in t:
        return "southern ocean", OCEAN_BBOX["southern ocean"]
    if "arctic" in t:
        return "arctic ocean", OCEAN_BBOX["arctic ocean"]
    return None, None


def infer_params_from_user_text(text: str):
    """Try to infer start_date, end_date, variables, region directly from user text."""
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if not dates:
        # also accept dd-mm-yyyy or dd/mm/yyyy? keep only ISO for reliability
        return None
    start = dates[0]
    end = dates[0] if len(dates) == 1 else dates[1]
    # find variable keywords
    vars_found = []
    for k in VAR_MAP.keys():
        if k in text.lower():
            vars_found.append(k)
    if not vars_found:
        # try plural forms or words like "temp", "sal"
        if re.search(r"\btemp\b|\btemperature\b|\bdeg c\b|\bdegc\b", text.lower()):
            vars_found = ["temperature"]
        elif re.search(r"\bsalinity\b|\bsal\b", text.lower()):
            vars_found = ["salinity"]
    # region inference
    region = ""
    for k in ["pacific", "atlantic", "indian", "southern", "arctic"]:
        if k in text.lower():
            region = k + (" ocean" if k != "pacific" else "")
            break
    params = {
        "start_date": start,
        "end_date": end,
        "variables": vars_found or ["temperature"],
        "region": region,
    }
    return params


def extract_parameters_from_text(text: str):
    """
    Use Gemini (if available) to return either:
    - JSON: {start_date, end_date, variables, region}
    - Plain text explanation (no JSON)
    If Gemini returns plain text, we attempt to infer params from user text automatically.
    Returns: (params_dict_or_None, raw_model_text, inferred_from_user_flag)
    """
    inferred = False
    if MODEL is None:
        params = infer_params_from_user_text(text)
        if params:
            return params, "Fallback parser used (no Gemini).", True
        return None, "Explanation requested (fallback, no Gemini):\n\n" + text, False

    system_prompt = (
        "You are a helper that extracts parameters for Argo ocean data queries.\n"
        "If the user requests data, output STRICT JSON only with keys: start_date, end_date, variables, region\n"
        'Example: {"start_date":"2023-01-01","end_date":"2023-12-31","variables":["temperature"],"region":"Indian Ocean"}\n'
        "If the user only wants an explanation (no data), return plain English (no JSON)."
    )
    try:
        resp = MODEL.generate_content(system_prompt + "\nUser: " + text)
        raw = resp.text
    except Exception as e:
        # Gemini call failed — try inference
        params = infer_params_from_user_text(text)
        if params:
            return (
                params,
                f"(Gemini call failed) {e}\nInferred params from user text.",
                True,
            )
        return None, f"(Gemini call failed) {e}\nUser: {text}", False

    params = parse_json_from_text(raw)
    if params:
        # make sure single-date is handled downstream
        return params, raw, False

    # model returned explanation (plain text). Try to infer parameters from user text now.
    params_inferred = infer_params_from_user_text(text)
    if params_inferred:
        return (
            params_inferred,
            raw
            + "\n\n(Parameters inferred from user input because model returned an explanation.)",
            True,
        )

    return None, raw, False


def fetch_region_chunk(bbox_segment, start_date, end_date):
    lon_min, lon_max, lat_min, lat_max = bbox_segment
    chunks = []
    delta = timedelta(days=5)
    current = start_date
    while current <= end_date:
        chunk_end = min(current + delta, end_date)
        start_iso = current.strftime("%Y-%m-%dT00:00:00Z")
        end_iso = chunk_end.strftime("%Y-%m-%dT23:59:59Z")
        try:
            df = (
                DataFetcher(
                    src="erddap",
                    dataset="phy",
                    erddap_server="https://www.ifremer.fr/erddap/",
                )
                .region(
                    [lon_min, lon_max, lat_min, lat_max, 0, 2000, start_iso, end_iso]
                )
                .to_dataframe()
                .reset_index()
            )
            df.columns = [c.lower() for c in df.columns]
            chunks.append(df)
        except Exception as e:
            st.error(f"Fetch error for {current.date()}→{chunk_end.date()}: {e}")
        current = chunk_end + timedelta(seconds=1)
    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True)


def fetch_argo_for_bbox(bbox, start_date, end_date):
    bbox_norm = normalize_bbox(bbox)
    segments = split_bbox_if_wrap(bbox_norm)
    dfs = []
    for seg in segments:
        df_seg = fetch_region_chunk(seg, start_date, end_date)
        if df_seg is not None and not df_seg.empty:
            dfs.append(df_seg)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def explain_with_gemini(df: pd.DataFrame, query: str) -> str:
    if MODEL is None:
        cols = list(df.columns[:6])
        nrows = min(5, len(df))
        return f"(No Gemini) columns: {cols}. Showing first {nrows} rows."
    df_sample = df.head(10).to_csv(index=False)
    prompt = (
        "You are an oceanography expert. Explain the following Argo dataset in simple terms.\n\n"
        f"User query: {query}\n\nSample data (first 10 rows):\n{df_sample}\n\n"
        "Keep the explanation short: mention which parameters are present and what they indicate."
    )
    try:
        resp = MODEL.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"(Gemini call failed) {e}"


# -------------------- STREAMLIT UI --------------------
st.set_page_config(
    page_title="Argo Chatbot (robust single-date + inference)", layout="wide"
)
st.title("🌊 Argo Chatbot — robust single-date handling & inference")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "datasets" not in st.session_state:
    st.session_state.datasets = []

col_left, col_right = st.columns([0.85, 0.15])
with col_right:
    if st.button("🧹 Clear Chat & Data"):
        st.session_state.messages = []
        st.session_state.datasets = []
        for k in list(st.session_state.keys()):
            if k.startswith("show_full_"):
                del st.session_state[k]
        # st.experimental_rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about oceans, Argo data, or marine science...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    params, raw_model_text, inferred_flag = extract_parameters_from_text(user_input)

    # If params is None -> explanation-only (raw_model_text contains that)
    if not params:
        assistant_text = raw_model_text or "Sorry — I could not produce an explanation."
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
    else:
        # post-process params: single-date => end_date = start_date; default variable; default region inference
        if "start_date" in params and (
            "end_date" not in params or not params.get("end_date")
        ):
            params["end_date"] = params["start_date"]
        if "variables" not in params or not params.get("variables"):
            params["variables"] = ["temperature"]
        if "region" not in params or not params.get("region"):
            inferred_ocean, _ = detect_ocean(user_input)
            if inferred_ocean:
                params["region"] = (
                    inferred_ocean if inferred_ocean != "pacific" else "Pacific Ocean"
                )
            else:
                params["region"] = ""

        required = ("start_date", "end_date", "variables", "region")
        missing = [k for k in required if k not in params]
        if missing:
            msg = f"Missing keys from parsed params: {missing}. Raw model output:\n\n{raw_model_text}"
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
        else:
            # parse dates
            try:
                start_date = datetime.strptime(params["start_date"], "%Y-%m-%d")
                end_date = datetime.strptime(params["end_date"], "%Y-%m-%d")
            except Exception:
                err = "Invalid date format. Use YYYY-MM-DD."
                st.session_state.messages.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"):
                    st.markdown(err)
                st.stop()

            end_date = min(end_date, datetime.utcnow())

            ocean_key, bbox = detect_ocean(params.get("region", "") or "")
            if not ocean_key:
                region_raw = params.get("region", "").strip().lower()
                if region_raw in OCEAN_BBOX:
                    ocean_key, bbox = region_raw, OCEAN_BBOX[region_raw]
                else:
                    err = "Please specify a valid ocean (e.g., 'Indian Ocean', 'Atlantic Ocean', 'Pacific Ocean')."
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(err)
                    st.stop()

            if ocean_key == "pacific":
                fetch_list = [
                    ("Pacific West", OCEAN_BBOX["pacific west"]),
                    ("Pacific East", OCEAN_BBOX["pacific east"]),
                ]
            elif ocean_key in ("pacific west", "pacific east"):
                fetch_list = [(ocean_key.title(), OCEAN_BBOX[ocean_key])]
            else:
                if bbox is None:
                    bbox = OCEAN_BBOX.get(ocean_key)
                    if bbox is None:
                        st.error("Could not resolve bounding box for the region.")
                        st.stop()
                fetch_list = [(ocean_key.title(), bbox)]

            all_dfs = []
            with st.spinner("Fetching Argo data..."):
                for label, bb in fetch_list:
                    df_region = fetch_argo_for_bbox(bb, start_date, end_date)
                    if df_region is not None and not df_region.empty:
                        all_dfs.append(df_region)

            if not all_dfs:
                msg = "No data retrieved for your query region/date."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
            else:
                df_full = pd.concat(all_dfs, ignore_index=True)
                df_full.columns = [c.lower() for c in df_full.columns]
                for c in ("latitude", "longitude", "time"):
                    if c not in df_full.columns:
                        df_full[c] = pd.NA

                requested_vars = params.get("variables") or []
                expanded = expand_variables(requested_vars)
                selected = [v for v in expanded if v in df_full.columns]
                if not selected:
                    guesses = []
                    for cand_list in VAR_MAP.values():
                        for cand in cand_list:
                            if cand in df_full.columns:
                                guesses.append(cand)
                    selected = list(dict.fromkeys(guesses))

                base_cols = [
                    c for c in ("latitude", "longitude", "time") if c in df_full.columns
                ]
                final_cols = base_cols + selected
                final_cols = [
                    c for i, c in enumerate(final_cols) if c not in final_cols[:i]
                ]
                df_selected = df_full[final_cols] if final_cols else df_full.copy()

                ds_id = len(st.session_state.datasets)
                st.session_state.datasets.append(
                    {
                        "id": ds_id,
                        "ocean": ocean_key.title(),
                        "start_date": start_date.date().isoformat(),
                        "end_date": end_date.date().isoformat(),
                        "variables": requested_vars,
                        "df": df_selected,
                    }
                )

                compute_avg = bool(
                    re.search(
                        r"\b(average|avg|mean|mean temperature|mean salinity)\b",
                        user_input.lower(),
                    )
                )
                stats_text = ""
                if compute_avg and selected:
                    stats_lines = []
                    for col in selected:
                        try:
                            vals = pd.to_numeric(df_selected[col], errors="coerce")
                            mean_val = vals.mean()
                            if pd.notna(mean_val):
                                stats_lines.append(f"{col}: mean={mean_val:.3f}")
                        except Exception:
                            continue
                    if stats_lines:
                        stats_text = "\n\n" + "\n".join([f"• {s}" for s in stats_lines])

                summary = f"✅ Fetched data for {ocean_key.title()} from {start_date.date()} to {end_date.date()}."
                if stats_text:
                    summary += "\n\nComputed statistics (requested):" + stats_text
                # if we inferred params from user text (because Gemini returned explanation),
                # show a one-line notice so user knows inference happened
                if inferred_flag:
                    summary += (
                        "\n\n(Note: parameters were inferred from your query text.)"
                    )

                st.session_state.messages.append(
                    {"role": "assistant", "content": summary}
                )
                with st.chat_message("assistant"):
                    st.markdown(summary)
                    explanation = (
                        explain_with_gemini(df_selected, user_input)
                        if MODEL
                        else "(No Gemini) local summary used."
                    )
                    st.markdown(f"**Analysis:** {explanation}")

                # Render visualizations in a stable container below the chat bubble
                visualize.show_visualizations(
                    df_selected,
                    variables=selected,
                    dataset_id=ds_id
                )

# -------------------- HISTORY PANEL --------------------
if st.session_state.datasets:
    st.markdown("---")
    st.subheader("📚 Previous Queries")
    for ds in st.session_state.datasets:
        st.markdown(f"**Query #{ds['id']} — {ds['ocean']}**")
        st.markdown(
            f"{ds['start_date']} → {ds['end_date']} — variables: {', '.join(ds['variables']) if ds['variables'] else 'none'}"
        )
        try:
            st.dataframe(ds["df"].head(7))
        except Exception:
            st.write("(Preview unavailable)")
        show_key = f"show_full_{ds['id']}"
        checked = st.checkbox(
            "📌 Show full data",
            value=st.session_state.get(show_key, False),
            key=show_key,
        )
        if checked:
            with st.expander("📊 Full Data (expand/minimize)", expanded=True):
                st.dataframe(ds["df"])
        csv_bytes = ds["df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download this dataset as CSV",
            data=csv_bytes,
            file_name=f"argo_{ds['ocean']}_{ds['id']}.csv",
            mime="text/csv",
            key=f"download_{ds['id']}",
        )

