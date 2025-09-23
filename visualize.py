# visualize.py
import pandas as pd
import plotly.express as px
import streamlit as st

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None



def map_view(df: pd.DataFrame, color_var: str | None = None, title: str = "Float locations"):
    """Interactive map of float locations."""
    if {"latitude", "longitude"}.issubset(df.columns):
        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color=color_var if color_var and color_var in df.columns else None,
            hover_name="time" if "time" in df.columns else None,
            zoom=2,
            height=400,
            title=title,
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Latitude/Longitude columns not found for map view.")


# def profile_plot(df: pd.DataFrame, var: str, title: str | None = None):
#     """Depth (pressure) vs. variable profiles."""
#     if "pres" in df.columns and var in df.columns:
#         fig = px.scatter(
#             df,
#             x=var,
#             y="pres",
#             color="time" if "time" in df.columns else None,
#             labels={"pres": "Pressure (dbar)", var: var},
#             title=title or f"{var} profile(s)"
#         )
#         fig.update_yaxes(autorange="reversed")  # Depth increases downward
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.warning(f"Cannot plot profile — 'pres' or '{var}' column missing.")

def profile_plot(df: pd.DataFrame, var: str, title: str | None = None):
    """Depth (pressure) vs. variable profiles."""
    pres_col = pick_first_existing(df, ["pres", "pressure", "pres_adjusted"])
    var_col = pick_first_existing(df, [var, f"{var}_adjusted", f"{var}_qc"])

    if pres_col and var_col:
        fig = px.scatter(
            df,
            x=var_col,
            y=pres_col,
            color="time" if "time" in df.columns else None,
            labels={pres_col: "Pressure (dbar)", var_col: var_col},
            title=title or f"{var_col} profile(s)"
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Cannot plot profile — pressure or '{var}' column missing.")


# def time_series(df: pd.DataFrame, var: str, title: str | None = None):
#     """Time series of a variable."""
#     if "time" in df.columns and var in df.columns:
#         df = df.copy()
#         df["time"] = pd.to_datetime(df["time"], errors="coerce")
#         df = df.sort_values("time")
#         fig = px.line(
#             df,
#             x="time",
#             y=var,
#             title=title or f"{var} over time",
#             labels={"time": "Date/Time", var: var}
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.warning(f"Cannot plot time series — 'time' or '{var}' column missing.")

def time_series(df: pd.DataFrame, var: str, title: str | None = None):
    """Time series of a variable."""
    var_col = pick_first_existing(df, [var, f"{var}_adjusted", f"{var}_qc"])
    if "time" in df.columns and var_col:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")
        fig = px.line(
            df,
            x="time",
            y=var_col,
            title=title or f"{var_col} over time",
            labels={"time": "Date/Time", var_col: var_col}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Cannot plot time series — 'time' or '{var}' column missing.")


def show_visualizations(df: pd.DataFrame, variables: list[str], dataset_id: int):
    """Show all available visualizations (map, profile, time series) together."""
    st.markdown("### 🔎 Visualizations")

    if not variables:
        st.info("No variable selected for visualization.")
        return

    var = variables[0]  # just use the first variable for plotting

    # Map
    st.subheader("🌍 Float Locations")
    map_view(df, color_var=var)

    # Profile
    st.subheader("📉 Vertical Profile")
    profile_plot(df, var)

    # Time Series
    st.subheader("⏳ Time Series")
    time_series(df, var)
