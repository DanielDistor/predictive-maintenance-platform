import streamlit as st
import plotly.graph_objects as go
from src.load_data import load_train
from src.preprocess import add_rul

st.set_page_config(page_title="Predictive Maintenance Platform", layout="wide")

RISK_COLORS = {"Critical": "#ef4444", "Warning": "#f59e0b", "Healthy": "#10b981"}

INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]


@st.cache_data
def get_data():
    df = load_train()
    return add_rul(df)


try:
    df = get_data()
except FileNotFoundError:
    st.error("Dataset not found. Place `train_FD001.txt` in the `data/` folder. See README.")
    st.stop()

fleet = df.groupby("engine_id")["cycle"].max().reset_index()
fleet.columns = ["engine_id", "max_cycle"]
fleet["risk"] = fleet["max_cycle"].apply(
    lambda x: "Critical" if x < 150 else ("Warning" if x < 250 else "Healthy")
)
fleet = fleet.sort_values("max_cycle").reset_index(drop=True)

engine_labels = {
    row["engine_id"]: f"Engine {row['engine_id']}  —  {row['max_cycle']} cycles  ({row['risk']})"
    for _, row in fleet.iterrows()
}

tab_about, tab_dashboard = st.tabs(["About the Dataset", "Dashboard"])

# ── Tab 1: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.title("NASA C-MAPSS Turbofan Engine Dataset")
    st.caption("Commercial Modular Aero-Propulsion System Simulation")

    st.markdown("""
    ### What is this dataset?

    The **C-MAPSS** dataset was created by NASA to simulate how turbofan aircraft engines
    degrade over time. Each engine starts in a healthy state and is run through repeated
    operational cycles until it fails. The dataset records 21 sensor readings per cycle,
    letting us study how the engine changes as it wears out.

    The goal is to predict the **Remaining Useful Life (RUL)** — how many more cycles an
    engine can run before it breaks down.

    ---
    ### Key Terms
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Engine ID**
        A unique number (1–100) identifying each individual turbofan engine.
        Each engine is independent — its own run-to-failure story.

        **Cycle**
        One operational cycle of the engine (think: one flight). Each engine
        starts at cycle 1 and runs until it fails.

        **Remaining Useful Life (RUL)**
        How many cycles an engine has left before failure.
        `RUL = total_cycles − current_cycle`
        At the last cycle, RUL = 0 (the engine just failed).

        **Operational Settings (op_1, op_2, op_3)**
        Environmental and throttle conditions during each cycle — things like
        altitude and flight regime.
        """)
    with col2:
        st.markdown("""
        **Sensors (sensor_1 to sensor_21)**
        Physical measurements taken from the engine every cycle:
        - Fan speed and pressure
        - Core temperature
        - Fuel flow rate
        - Exhaust gas temperature

        Not all 21 sensors are equally useful — some are nearly constant.
        The dashboard only shows the 13 sensors that actually change over time.

        **FD001 (this dataset)**
        One of four C-MAPSS subsets. Single operating condition, one failure
        mode — the simplest and best starting point.

        **Degradation**
        Gradual wear of engine components over cycles. Sensors that trend
        upward or downward over an engine's lifetime are capturing degradation.
        """)

    st.divider()
    st.markdown("### Dataset at a Glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training Engines", fleet["engine_id"].nunique())
    m2.metric("Total Rows", f"{len(df):,}")
    m3.metric("Sensors", 21)
    m4.metric("Avg Engine Life", f"{fleet['max_cycle'].mean():.0f} cycles")

    st.divider()
    st.markdown("""
    ### Why Does This Matter?

    Predicting equipment failure before it happens lets maintenance teams schedule
    repairs proactively instead of reacting to breakdowns. This is **predictive maintenance**.

    - Avoid unplanned downtime (a grounded aircraft costs ~$150k/hour)
    - Extend equipment life by servicing at exactly the right time
    - Reduce costs vs. fixed-schedule servicing

    C-MAPSS is a benchmark dataset used by researchers worldwide to develop and test
    predictive maintenance algorithms.

    ### What's Next (Phase 2)
    - **Feature engineering** — extract degradation trends from raw sensors
    - **RUL prediction model** — train ML to forecast remaining life
    - **Health score** — a single 0–100 condition indicator per engine
    - **Risk alerts** — flag engines approaching failure
    """)

# ── Tab 2: Dashboard ──────────────────────────────────────────────────────────
with tab_dashboard:
    st.title("Fleet Health Monitor")
    st.caption("Engine degradation analysis — NASA C-MAPSS FD001")

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_critical = (fleet["risk"] == "Critical").sum()
    n_warning  = (fleet["risk"] == "Warning").sum()
    n_healthy  = (fleet["risk"] == "Healthy").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Engines", len(fleet))
    c2.metric("Avg Engine Life", f"{fleet['max_cycle'].mean():.0f} cycles")
    c3.metric("Critical  (<150 cycles)", n_critical)
    c4.metric("Healthy  (>250 cycles)", n_healthy)

    # ── Fleet bar chart ───────────────────────────────────────────────────────
    st.subheader("Engine Lifetime Overview")
    st.caption("Each bar = one engine. Height = total cycles before failure. Sorted shortest → longest.")

    fig_fleet = go.Figure()
    for risk in ["Critical", "Warning", "Healthy"]:
        subset = fleet[fleet["risk"] == risk]
        fig_fleet.add_trace(go.Bar(
            x=list(range(len(subset))),
            y=subset["max_cycle"],
            name=risk,
            marker_color=RISK_COLORS[risk],
            customdata=subset["engine_id"],
            hovertemplate="Engine %{customdata}<br>Lasted <b>%{y} cycles</b><extra></extra>",
        ))
    fig_fleet.update_layout(
        barmode="stack",
        xaxis=dict(title="Engines (sorted by lifetime)", showticklabels=False),
        yaxis_title="Total Cycles Before Failure",
        legend=dict(orientation="h", y=1.1, x=0),
        height=280,
        margin=dict(t=10, b=30, l=40, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_fleet, width="stretch")

    st.divider()

    # ── Engine Inspector ──────────────────────────────────────────────────────
    st.subheader("Engine Inspector")

    engine_id = st.selectbox(
        "Select an engine",
        options=sorted(fleet["engine_id"].unique()),
        format_func=lambda x: engine_labels[x],
    )

    engine_df   = df[df["engine_id"] == engine_id].sort_values("cycle")
    engine_info = fleet[fleet["engine_id"] == engine_id].iloc[0]
    risk        = engine_info["risk"]
    max_cycle   = int(engine_info["max_cycle"])
    color       = RISK_COLORS[risk]
    risk_icon   = {"Critical": "🔴", "Warning": "🟡", "Healthy": "🟢"}[risk]

    col_card, col_rul, col_sensor = st.columns([1, 2, 2])

    # Health card
    with col_card:
        st.markdown(f"""
        <div style="background:{color}18;border-left:4px solid {color};
                    padding:20px;border-radius:10px;height:100%;">
            <div style="font-size:11px;color:{color};font-weight:700;
                        text-transform:uppercase;letter-spacing:1px;">
                {risk_icon} {risk}
            </div>
            <div style="font-size:32px;font-weight:800;margin:10px 0 4px;">
                Engine {engine_id}
            </div>
            <div style="font-size:13px;opacity:0.6;margin-bottom:4px;">
                Total life
            </div>
            <div style="font-size:22px;font-weight:700;color:{color};">
                {max_cycle} cycles
            </div>
            <div style="margin-top:12px;font-size:12px;opacity:0.5;">
                Started at RUL {max_cycle - 1}<br>
                Ended at RUL 0 (failure)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RUL decline chart
    with col_rul:
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(
            x=engine_df["cycle"],
            y=engine_df["RUL"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=color, width=2),
            fillcolor=f"{color}28",
            hovertemplate="Cycle %{x}<br><b>RUL: %{y}</b><extra></extra>",
        ))
        fig_rul.update_layout(
            title=dict(text="Remaining Useful Life", font=dict(size=13)),
            xaxis_title="Cycle",
            yaxis_title="Cycles remaining",
            height=280,
            margin=dict(t=40, b=40, l=50, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_rul, width="stretch")

    # Sensor degradation chart
    with col_sensor:
        selected_sensors = st.multiselect(
            "Sensors to compare (normalized 0–1)",
            INFORMATIVE_SENSORS,
            default=["sensor_2", "sensor_3", "sensor_4"],
        )
        if selected_sensors:
            fig_sensors = go.Figure()
            for sensor in selected_sensors:
                s = engine_df[sensor]
                rng = s.max() - s.min()
                s_norm = (s - s.min()) / (rng if rng > 0 else 1)
                fig_sensors.add_trace(go.Scatter(
                    x=engine_df["cycle"],
                    y=s_norm,
                    mode="lines",
                    name=sensor,
                    hovertemplate=f"<b>{sensor}</b><br>Cycle %{{x}}<br>Raw: %{{customdata:.2f}}<extra></extra>",
                    customdata=engine_df[sensor],
                ))
            fig_sensors.update_layout(
                title=dict(text="Sensor Degradation Trends", font=dict(size=13)),
                xaxis_title="Cycle",
                yaxis_title="Normalized value (0 = min, 1 = max)",
                height=280,
                margin=dict(t=40, b=40, l=50, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.35),
            )
            st.plotly_chart(fig_sensors, width="stretch")
        else:
            st.info("Select at least one sensor above.")
