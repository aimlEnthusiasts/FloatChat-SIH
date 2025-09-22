import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from argopy import DataFetcher, set_options
import google.generativeai as genai
import json, re, os
from  dotenv  import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

# --- GEMINI CONFIG ---
# Replace with your Gemini API key (or set it as an env var)
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

OCEAN_BBOX = {
    "indian ocean": [20, 120, -40, 30],
    "atlantic ocean": [-80, 20, -40, 70],
    "pacific ocean": [120, -70, -50, 60],
    "southern ocean": [-180, 180, -90, -50],
    "arctic ocean": [-180, 180, 50, 90]
}

VAR_MAP = {
    "temperature": ["temp", "temp_adjusted", "temp_qc"],
    "salinity": ["psal", "psal_adjusted", "psal_qc"],
    "pressure": ["pres", "pres_adjusted", "pres_qc"]
}


def parse_json_from_text(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None


def expand_variables(natural_vars):
    expanded = []
    for var in natural_vars:
        var = var.lower()
        if var in VAR_MAP:
            expanded.extend(VAR_MAP[var])
        else:
            expanded.append(var.lower())
    return expanded


def detect_ocean(text):
    text = text.lower()
    for ocean, bbox in OCEAN_BBOX.items():
        if ocean in text:
            return ocean, bbox
    return None, None


def extract_parameters_from_text(text):
    system_prompt = """
    You are a helper that extracts parameters for Argo ocean data queries. 
    Output strictly as JSON with keys: start_date, end_date, variables, region
    Example: {"start_date":"2023-01-01","end_date":"2023-12-31","variables":["temperature","salinity"],"region":"Indian Ocean"}
    And if the data is not requested and just explaination is requested or something like that then do not return any data, just return the explaination.
    Example: Explain me the oceanography or explain a previous query.
    
    """
    context = system_prompt + "\nUser: " + text + "\nAssistant:"
    response = model.generate_content(context).text
    params = parse_json_from_text(response)
    return params

def explain_with_gemini(df: pd.DataFrame, query: str) -> str:
    """Ask Gemini to explain the fetched dataset in plain language."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # We pass both query and a sample of the dataframe
    df_sample = df.head(10).to_csv(index=False)  # only send preview to save tokens
    prompt = f"""
    You are an oceanography expert. 
    Explain the following Argo dataset in very simple terms.

    User query: {query}

    Sample data (first 10 rows):
    {df_sample}

    Keep the explanation short and clear:
    - What parameters are included (e.g., temperature, salinity, oxygen, etc.)
    - What these parameters indicate about the ocean
    - How this data answers the user’s query
    """


    response = model.generate_content(prompt)
    return response.text
# --- Streamlit UI ---
st.set_page_config(page_title="Argo Chatbot", layout="wide")

st.title("🌊 Oceanography & Argo Chatbot with Data Fetching")

# --- Persistent state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# store each successful query's result in a list so previews/history persist
if "datasets" not in st.session_state:
    st.session_state.datasets = []  # list of dicts: {id, ocean, start_date, end_date, variables, df}

# --- Clear history button ---
col_left, col_right = st.columns([0.85, 0.15])
with col_right:
    if st.button("🧹 Clear Chat History"):
        # clear messages and datasets + related show flags
        st.session_state.messages = []
        st.session_state.datasets = []
        # remove any show_full_x keys
        keys_to_remove = [k for k in list(st.session_state.keys()) if k.startswith("show_full_")]
        for k in keys_to_remove:
            del st.session_state[k]
        st.success("Chat history and data cleared.")

# Render previous chat messages (keeps history visible)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask about oceans, Argo data, or marine science...")

if user_input:
    # append user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    data_request_keywords = ["temperature", "salinity", "argo", "ocean", "pressure", "data", "profile", "download"]
    is_data_query = any(word in user_input.lower() for word in data_request_keywords)

    if is_data_query:
        params = extract_parameters_from_text(user_input)
        if not params:
            answer = "Sorry, I couldn't understand what data you want. Please specify region, dates, and variables."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            variables = expand_variables(params.get("variables", []))
            region_text = params.get("region", "")
            ocean_name, bbox = detect_ocean(region_text)
            if not bbox:
                answer = "Please specify a valid ocean region (e.g., Indian Ocean, Atlantic Ocean)."
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
            else:
                try:
                    start_date = datetime.strptime(params.get("start_date"), "%Y-%m-%d")
                    end_date = datetime.strptime(params.get("end_date"), "%Y-%m-%d")
                    today = datetime.utcnow()
                    if end_date > today:
                        end_date = today
                except Exception:
                    answer = "Invalid date format. Please provide start and end date as YYYY-MM-DD."
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                else:
                    lon_min, lon_max, lat_min, lat_max = bbox
                    set_options(api_timeout=300)
                    delta = timedelta(days=5)
                    all_datasets = []
                    current = start_date
                    total_days = (end_date - start_date).days
                    total_chunks = max(1, (total_days // delta.days) + 1)
                    progress = st.progress(0)
                    chunk_index = 0

                    with st.spinner("Fetching Argo data..."):
                        while current <= end_date:
                            chunk_end = min(current + delta, end_date)
                            start_iso = current.strftime("%Y-%m-%dT00:00:00Z")
                            end_iso = chunk_end.strftime("%Y-%m-%dT23:59:59Z")
                            try:
                                df_chunk = DataFetcher(
                                    src="erddap",
                                    dataset="phy",
                                    erddap_server="https://www.ifremer.fr/erddap/"
                                ).region(
                                    [lon_min, lon_max, lat_min, lat_max, 0, 2000, start_iso, end_iso]
                                ).to_dataframe().reset_index()

                                # normalize columns to lowercase
                                df_chunk.columns = [c.lower() for c in df_chunk.columns]

                                all_datasets.append(df_chunk)
                            except Exception as e:
                                st.error(f"Error fetching data chunk {current.date()} to {chunk_end.date()}: {e}")
                                # skip this chunk but continue
                            # move to next chunk (add 1 second to avoid infinite loop when chunk_end == current)
                            current = chunk_end + timedelta(seconds=1)
                            chunk_index += 1
                            progress.progress(min(1.0, chunk_index / total_chunks))

                    if all_datasets:
                        df_full = pd.concat(all_datasets, ignore_index=True)
                        df_full.columns = [c.lower() for c in df_full.columns]

                        # Ensure base columns exist; if missing, create them as empty
                        base_expected = ["latitude", "longitude", "time"]
                        for col in base_expected:
                            if col not in df_full.columns:
                                df_full[col] = pd.NA

                        selected_vars = [v for v in variables if v in df_full.columns]

                        # If nothing matched from requested variables, try to find likely columns
                        if not selected_vars and params.get("variables"):
                            guessed = []
                            for want in params.get("variables"):
                                mapped = VAR_MAP.get(want.lower(), [])
                                for m in mapped:
                                    if m in df_full.columns:
                                        guessed.append(m)
                            selected_vars = list(dict.fromkeys(guessed))

                        # Build final dataframe to present
                        base_cols = [c for c in base_expected if c in df_full.columns]
                        final_cols = base_cols + selected_vars
                        # avoid duplicate columns
                        final_cols = [c for i, c in enumerate(final_cols) if c not in final_cols[:i]]

                        if not selected_vars:
                            warning_msg = "Warning: None of the requested variables were found in the returned dataset. Showing available columns."
                            st.warning(warning_msg)

                        df_selected = df_full[final_cols] if final_cols else df_full.copy()

                        # Save dataset to session_state so history + toggles persist per query
                        ds_id = len(st.session_state.datasets)
                        st.session_state.datasets.append({
                            "id": ds_id,
                            "ocean": ocean_name.title(),
                            "start_date": start_date.date().isoformat(),
                            "end_date": end_date.date().isoformat(),
                            "variables": params.get("variables", []),
                            "df": df_selected
                        })

                        answer = (
                            f"✅ Fetched Argo data for {ocean_name.title()} from {start_date.date()} "
                            f"to {end_date.date()} with variables: {', '.join(params.get('variables',[]))}."
                        )
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            explanation = explain_with_gemini(df_selected, user_input)
                            st.markdown(f"**Gemini Analysis:** {explanation}")

                    else:
                        answer = "No data was retrieved for your query period and region."
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)

    else:
        # Non-data question -> pass full chat context to the model
        context = ""
        for msg in st.session_state.messages:
            context += f"\n{msg['role'].capitalize()}: {msg['content']}"
        with st.spinner("Generating answer..."):
            response_text = model.generate_content(context).text
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

# --- Render Data History (previews + persistent toggles) ---
if st.session_state.datasets:
    st.markdown("---")
    st.markdown("### 📚 Previous Data Queries")
    for ds in st.session_state.datasets:
        st.markdown(f"**Query #{ds['id']} — {ds['ocean']}**  ")
        st.markdown(f"{ds['start_date']} to {ds['end_date']}  — variables: {', '.join(ds['variables'])}")

        # Preview
        try:
            st.markdown("Preview (first 7 rows)")
            st.dataframe(ds['df'].head(7))
        except Exception:
            st.write("(Unable to preview this dataset)")

        # Per-query persistent toggle using a checkbox (reliable across reruns)
        show_key = f"show_full_{ds['id']}"
        checked = st.checkbox("📌 Show full data", value=st.session_state.get(show_key, False), key=show_key)

        if checked:
            with st.expander("📊 Full Data (expand/minimize)", expanded=True):
                # Use a container for optional styling
                st.dataframe(ds['df'])

        # Per-dataset download button
        csv_data = ds['df'].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download this dataset as CSV",
            data=csv_data,
            file_name=f"argo_data_query_{ds['id']}.csv",
            mime="text/csv",
            key=f"download_{ds['id']}"
        )

    st.markdown("---")

# End of file
