# FloatChat-SIH — Argo Chatbot

This Streamlit app lets you chat about Argo float oceanographic data. It can parse your natural-language query to fetch Argo profiles from ERDDAP via `argopy`, then summarize or compute simple stats. If a Gemini API key is provided, it also generates short explanations.

## Features
- Natural language parameter extraction with Gemini or a local fallback
- Robust single-date handling (uses the same date for start/end)
- Ocean region detection with special handling for the Pacific (wraps 0/360)
- Incremental ERDDAP fetching by 5-day chunks for reliability
- Variable expansion (e.g., "temperature" -> `temp`, `temp_adjusted`, `temp_qc`)
- History panel with dataset previews and CSV download

## Requirements
- Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Optional: Gemini API Key
Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Without a key, the app still works and will use a local fallback for parameter extraction and explanations.

## Run the app

```bash
streamlit run argo.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage tips
- Dates must be in `YYYY-MM-DD` format. Single date is allowed.
- Regions: say "Indian Ocean", "Atlantic Ocean", "Pacific Ocean". The Pacific is split across the dateline and fetched in two parts automatically.
- Variables: ask for temperature, salinity, or pressure. The app will expand to relevant columns found in the dataset.

## Notes
- Data source: Ifremer ERDDAP Argo dataset (`phy`).
- Long-running queries are chunked by 5 days to be resilient to API timeouts.
