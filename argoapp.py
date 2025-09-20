import streamlit as st
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from argopy import DataFetcher, set_options
import google.generativeai as genai
import json, re

# --- GEMINI CONFIG ---
genai.configure(api_key="AIzaSyAZtfncaOI_QslbbCt5ZmSUwbTboZ5tGyI")
model = genai.GenerativeModel("gemini-1.5-flash")

OCEAN_BBOX = {
    "indian ocean": [20, 120, -40, 30],
    "atlantic ocean": [-80, 20, -40, 70],
    "pacific ocean": [120, -70, -50, 60],
    "southern ocean": [-180, 180, -90, -50],
    "arctic ocean": [-180, 180, 50, 90]
}

VAR_MAP = {
    "temperature": ["TEMP", "TEMP_ADJUSTED", "TEMP_QC"],
    "salinity": ["PSAL", "PSAL_ADJUSTED", "PSAL_QC"],
    "pressure": ["PRES", "PRES_ADJUSTED", "PRES_QC"]
}

def parse_json_from_text(text):
    try:
        return json.loads(text)
    except:
        # Try to extract JSON inside text
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
            expanded.append(var.upper())
    return expanded

def detect_ocean(text):
    text = text.lower()
    for ocean, bbox in OCEAN_BBOX.items():
        if ocean in text:
            return ocean, bbox
    return None, None

def extract_parameters_from_text(text):
    """
    Use LLM to convert natural language to structured query parameters
    If that fails, fallback to manual extraction (basic).
    """
    system_prompt = """
    You are a helper that extracts parameters for Argo ocean data queries.
    Output strictly as JSON with keys: start_date, end_date, variables, region
    Example: {"start_date":"2024-01-01","end_date":"2025-12-31","variables":["temperature","salinity"],"region":"Indian Ocean"}
    """
    context = system_prompt + "\nUser: " + text + "\nAssistant:"
    response = model.generate_content(context).text
    params = parse_json_from_text(response)
    return params

# Start Streamlit app
st.title("🌊 Oceanography & Argo Chatbot with Data Fetching")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about oceans, Argo data, or marine science...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Detect if query likely requests Argo data by keywords
    data_request_keywords = ["temperature", "salinity", "argo", "ocean", "pressure", "data", "profile", "download"]
    is_data_query = any(word in user_input.lower() for word in data_request_keywords)

    if is_data_query:
        # Try extract parameters for data fetch from user query
        params = extract_parameters_from_text(user_input)
        if not params:
            answer = "Sorry, I couldn't understand what data you want. Please specify region, dates, and variables."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            # Expand variables, detect region bbox
            variables = expand_variables(params.get("variables", []))
            region_text = params.get("region", "")
            ocean_name, bbox = detect_ocean(region_text)

            if not bbox:
                answer = "Please specify a valid ocean region (e.g., Indian Ocean, Atlantic Ocean)."
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
            else:
                # Fetch Argo data in chunks
                start_date = datetime.strptime(params.get("start_date"), "%Y-%m-%d")
                end_date = datetime.strptime(params.get("end_date"), "%Y-%m-%d")
                lon_min, lon_max, lat_min, lat_max = bbox

                set_options(api_timeout=300)
                delta = timedelta(days=15)
                all_datasets = []
                current = start_date

                with st.spinner("Fetching Argo data..."):
                    while current < end_date:
                        chunk_end = min(current + delta, end_date)
                        try:
                            ds_chunk = DataFetcher(src="erddap", dataset="phy").region(
                                [lon_min, lon_max, lat_min, lat_max, 0, 2000, current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")]
                            ).to_xarray()
                            all_datasets.append(ds_chunk)
                        except Exception as e:
                            st.error(f"Error fetching data chunk {current.date()} to {chunk_end.date()}: {e}")
                            break
                        current = chunk_end

                if all_datasets:
                    final_ds = xr.concat(all_datasets, dim="N_PROF")
                    selected_vars = [v for v in variables if v in final_ds.data_vars]
                    if not selected_vars:
                        answer = "No matching data variables found in Argo dataset for your query."
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                    else:
                        df_full = final_ds[selected_vars].to_dataframe().reset_index()

                        # Prepare response summary
                        answer = (
                            f"Fetched Argo data for {ocean_name.title()} from {params.get('start_date')} "
                            f"to {params.get('end_date')} with variables: {', '.join(params.get('variables',[]))}.\n\n"
                            "Here's a preview:"
                        )
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            st.dataframe(df_full.head(7))

                        csv_data = df_full.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Full Argo Data CSV",
                            data=csv_data,
                            file_name="argo_ocean_data.csv",
                            mime="text/csv"
                        )
                else:
                    answer = "No data was retrieved for your query period and region."
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
    else:
        # General oceanography/Argo chatbot reply
        context = ""
        for msg in st.session_state.messages:
            context += f"\n{msg['role'].capitalize()}: {msg['content']}"

        with st.spinner("Generating answer..."):
            response_text = model.generate_content(context).text
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
