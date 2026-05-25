import streamlit as st
import plotly.graph_objects as go
from src.load_data import load_train
from src.preprocess import add_rul

st.set_page_config(page_title="Predictive Maintenance Platform", layout="wide")


@st.cache_data
def get_data():
    df = load_train()
    return add_rul(df)


try:
    df = get_data()
except FileNotFoundError:
    st.error("Dataset not found. Place `train_FD001.txt` in the `data/` folder. See README for setup instructions.")
    st.stop()

st.title("Predictive Maintenance Platform")
st.caption("NASA C-MAPSS FD001 — Turbofan Engine Degradation Dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Total Engines", df["engine_id"].nunique())
col2.metric("Total Rows", f"{len(df):,}")
col3.metric("Average RUL", round(df["RUL"].mean(), 1))

st.divider()

engine_id = st.selectbox("Select Engine", sorted(df["engine_id"].unique()))

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
selected_sensors = st.multiselect(
    "Select Sensors",
    sensor_cols,
    default=["sensor_2", "sensor_3", "sensor_4"],
)

if selected_sensors:
    engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
    fig = go.Figure()
    for sensor in selected_sensors:
        fig.add_trace(go.Scatter(
            x=engine_df["cycle"],
            y=engine_df[sensor],
            mode="lines",
            name=sensor,
        ))
    fig.update_layout(
        title=f"Sensor Readings — Engine {engine_id}",
        xaxis_title="Cycle",
        yaxis_title="Sensor Value",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one sensor to display the chart.")

st.divider()

st.subheader("Dataset Preview")
st.caption("First 100 rows of training data with RUL")
st.dataframe(df.head(100), use_container_width=True)
