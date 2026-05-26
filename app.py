import streamlit as st
import plotly.graph_objects as go
from src.load_data import load_train
from src.preprocess import add_rul

st.set_page_config(
    page_title="Predictive Maintenance Command Center",
    page_icon="✈️",
    layout="wide",
)

# ── Risk tier config ──────────────────────────────────────────────────────────
TIERS = [
    ("Critical",  "#ef4444", (0,   150)),
    ("High",      "#f97316", (150, 175)),
    ("Elevated",  "#eab308", (175, 210)),
    ("Moderate",  "#14b8a6", (210, 260)),
    ("Healthy",   "#22c55e", (260, 999)),
]
TIER_COLORS = {name: color for name, color, _ in TIERS}
INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]


def get_tier(max_cycle):
    for name, _, (lo, hi) in TIERS:
        if lo <= max_cycle < hi:
            return name
    return "Healthy"


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_train()
    return add_rul(df)


try:
    df = get_data()
except FileNotFoundError:
    st.error("Dataset not found. Place `train_FD001.txt` in `data/`. See README.")
    st.stop()

fleet = df.groupby("engine_id")["cycle"].max().reset_index()
fleet.columns = ["engine_id", "max_cycle"]
fleet["tier"] = fleet["max_cycle"].apply(get_tier)
fleet_sorted = fleet.sort_values("max_cycle").reset_index(drop=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Predictive Maintenance")
    st.caption("Command Center")
    st.divider()

    st.subheader("Navigation")
    page = st.radio(
        "",
        ["About the Dataset", "Fleet Overview", "Engine Deep Dive"],
        label_visibility="collapsed",
    )
    st.divider()

    st.subheader("Dataset Info")
    st.markdown(f"""
    This dashboard uses the **NASA C-MAPSS FD001** dataset, which contains simulated
    run-to-failure data for {fleet['engine_id'].nunique()} turbofan aircraft engines.
    Across all engines, there are **{len(df):,} recorded cycles**, with an average engine
    lifetime of **{fleet['max_cycle'].mean():.0f} cycles** before failure. Each cycle
    captures readings from 21 onboard sensors, of which 13 show meaningful degradation
    trends over time.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — About the Dataset
# ═════════════════════════════════════════════════════════════════════════════
if page == "About the Dataset":
    st.markdown("# 📖 About the Dataset")
    st.caption("NASA C-MAPSS — Commercial Modular Aero-Propulsion System Simulation")
    st.divider()

    st.markdown("""
    ### What is this dataset?

    The **C-MAPSS** dataset was created by NASA to simulate how turbofan aircraft engines
    degrade over time. Each engine starts healthy and runs through repeated cycles until it
    fails. 21 sensor readings are recorded every cycle, capturing how the engine changes
    as it wears out.

    The goal of this platform is to predict **Remaining Useful Life (RUL)** — how many cycles
    an engine has left before breakdown — so that maintenance can be scheduled at exactly the
    right time, avoiding both surprise failures and costly over-servicing.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Engine ID** is a unique number from 1 to 100 assigned to each turbofan engine in
        the dataset. Every engine is tracked independently from its first cycle until failure.

        **Cycle** represents one complete operational cycle, roughly equivalent to one flight.
        Each engine starts at cycle 1 and the data ends at the cycle where the engine failed.

        **RUL (Remaining Useful Life)** is the number of cycles left before an engine fails.
        It is calculated as the total lifetime minus the current cycle, so it counts down to
        zero at the moment of failure.

        **op_1, op_2, op_3** are operational settings such as altitude, throttle angle, and
        fan speed command. In FD001, these are nearly constant across all engines.
        """)
    with col2:
        st.markdown("""
        **Sensors 1 through 21** capture physical measurements inside the engine, including
        temperatures, pressures, fan and core speeds, and fuel flow rates. Only 13 of the 21
        sensors show meaningful change over an engine's lifetime. The other 8 are essentially
        flat and provide no useful signal for predicting failure.

        **FD001** is the simplest subset of the C-MAPSS collection. It uses a single operating
        condition and a single failure mode, making it the ideal starting point for building and
        validating predictive maintenance methods. The dataset includes 100 training engines,
        all run to failure.

        **Degradation** is the gradual wear captured as sensor readings drift upward or downward
        over an engine's lifetime. These trends are the signal the model learns from to estimate
        how much useful life remains.
        """)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training Engines", fleet["engine_id"].nunique())
    m2.metric("Total Rows", f"{len(df):,}")
    m3.metric("Informative Sensors", 13)
    m4.metric("Avg Engine Life", f"{fleet['max_cycle'].mean():.0f} cycles")

    st.divider()
    st.markdown("""
    ### Why Predictive Maintenance?

    A single unplanned engine shop visit costs roughly $500,000 in emergency labor, parts,
    and flight cancellations. A scheduled visit for the same work costs around $50,000.
    That is a ten-times difference — driven entirely by whether the failure was anticipated.
    Predictive maintenance uses sensor data to anticipate failures before they happen, so
    airlines can service engines at the right time rather than reacting after something breaks.

    ### What is Coming in Phase 2

    The next phase will add an LSTM neural network that predicts RUL from rolling sensor
    windows, confidence intervals to show prediction uncertainty, engineered health features,
    and a composite health score from 0 to 100 per engine.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Fleet Overview
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Fleet Overview":
    st.markdown("# ✈️ Fleet Health Overview")
    st.caption("Engine lifetime analysis — NASA C-MAPSS FD001")
    st.divider()

    # Risk tier counts
    tier_names = [t[0] for t in TIERS]
    counts = {name: (fleet["tier"] == name).sum() for name in tier_names}

    cols = st.columns(5)
    for col, (name, color, _) in zip(cols, TIERS):
        col.markdown(f"""
        <div style="background:{color}18;border:1px solid {color}44;
                    padding:16px 12px;border-radius:10px;text-align:center;">
            <div style="font-size:11px;color:{color};font-weight:700;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
                {name}
            </div>
            <div style="font-size:36px;font-weight:800;color:{color};">
                {counts[name]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Fleet Engine Lifetimes — Sorted Most Critical First")
    st.caption("Each bar = one engine. Height = total cycles before failure. Error bars show ±10% variability estimate.")

    fig = go.Figure()
    for name, color, _ in TIERS:
        subset = fleet_sorted[fleet_sorted["tier"] == name]
        fig.add_trace(go.Bar(
            x=subset.index,
            y=subset["max_cycle"],
            name=name,
            marker_color=color,
            error_y=dict(
                type="data",
                array=(subset["max_cycle"] * 0.10).tolist(),
                visible=True,
                color=color,
                thickness=1,
                width=2,
            ),
            customdata=subset["engine_id"],
            hovertemplate="<b>Engine %{customdata}</b><br>Lifetime: %{y} cycles<extra></extra>",
        ))

    fig.add_hline(y=175, line_dash="dot", line_color="#ef4444", line_width=1,
                  annotation_text="Critical threshold (175)", annotation_font_color="#ef4444",
                  annotation_position="top left")
    fig.add_hline(y=210, line_dash="dot", line_color="#eab308", line_width=1,
                  annotation_text="Elevated threshold (210)", annotation_font_color="#eab308",
                  annotation_position="top left")

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Engines (sorted by lifetime — most critical first)", showticklabels=False),
        yaxis_title="Total Cycles Before Failure",
        legend=dict(orientation="h", y=1.08, x=0),
        height=420,
        margin=dict(t=20, b=30, l=50, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Engine Deep Dive
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Engine Deep Dive":
    st.markdown("# 🔍 Engine Deep Dive")
    st.caption("Select an engine to inspect its full lifecycle and sensor degradation")
    st.divider()

    tier_icons = {"Critical": "🔴", "High": "🟠", "Elevated": "🟡",
                  "Moderate": "🟢", "Healthy": "✅"}

    engine_labels = {
        row["engine_id"]: f"Engine {row['engine_id']}  —  {tier_icons[row['tier']]} {row['tier'].upper()}"
        for _, row in fleet.iterrows()
    }

    engine_id = st.selectbox(
        "Select Engine to Analyze",
        options=sorted(fleet["engine_id"].unique()),
        format_func=lambda x: engine_labels[x],
    )

    engine_df   = df[df["engine_id"] == engine_id].sort_values("cycle")
    engine_info = fleet[fleet["engine_id"] == engine_id].iloc[0]
    tier        = engine_info["tier"]
    max_cycle   = int(engine_info["max_cycle"])
    color       = TIER_COLORS[tier]
    icon        = tier_icons[tier]
    avg_life    = fleet["max_cycle"].mean()

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Engine Life", f"{max_cycle} cycles")
    m2.metric("Max RUL at Start", f"{max_cycle - 1} cycles")
    m3.metric("vs Fleet Average", f"{max_cycle - avg_life:+.0f} cycles",
              delta_color="normal" if max_cycle >= avg_life else "inverse")
    m4.metric("Risk Tier", f"{icon} {tier}")

    # Alert banner for critical / high
    if tier == "Critical":
        st.error(f"🔴 **CRITICAL** — This engine had one of the shortest lifetimes in the fleet ({max_cycle} cycles). Engines in this range are highest priority for maintenance review.")
    elif tier == "High":
        st.warning(f"🟠 **HIGH RISK** — This engine's lifetime ({max_cycle} cycles) is below the fleet average of {avg_life:.0f} cycles.")

    st.divider()

    col_rul, col_sensor = st.columns(2)

    # RUL decline
    with col_rul:
        st.markdown("**Remaining Useful Life — Full Lifecycle**")
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(
            x=engine_df["cycle"],
            y=engine_df["RUL"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=color, width=2.5),
            fillcolor=f"{color}28",
            hovertemplate="Cycle %{x}<br><b>RUL: %{y} cycles</b><extra></extra>",
        ))
        fig_rul.add_hline(y=0, line_color="#555", line_width=1)
        fig_rul.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Remaining Useful Life (cycles)",
            height=320,
            margin=dict(t=10, b=40, l=55, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_rul, width="stretch")

    # Sensor degradation
    with col_sensor:
        st.markdown("**Sensor Degradation Trends**")
        selected = st.multiselect(
            "Sensors (normalized 0–1 per sensor so trends are comparable)",
            INFORMATIVE_SENSORS,
            default=["sensor_2", "sensor_3", "sensor_4"],
        )
        if selected:
            fig_s = go.Figure()
            for sensor in selected:
                s = engine_df[sensor]
                rng = s.max() - s.min()
                s_norm = (s - s.min()) / (rng if rng > 0 else 1)
                fig_s.add_trace(go.Scatter(
                    x=engine_df["cycle"],
                    y=s_norm,
                    mode="lines",
                    name=sensor,
                    hovertemplate=f"<b>{sensor}</b><br>Cycle %{{x}}<br>Raw value: %{{customdata:.3f}}<extra></extra>",
                    customdata=engine_df[sensor],
                ))
            fig_s.update_layout(
                xaxis_title="Cycle",
                yaxis_title="Normalized value (0 = min, 1 = max)",
                height=320,
                margin=dict(t=10, b=40, l=55, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig_s, width="stretch")
        else:
            st.info("Select at least one sensor.")

    st.divider()
    st.markdown("**How does this engine compare to the fleet?**")

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        x=["Shortest Engine", "This Engine", "Fleet Average", "Longest Engine"],
        y=[int(fleet["max_cycle"].min()), max_cycle,
           int(avg_life), int(fleet["max_cycle"].max())],
        marker_color=[TIER_COLORS["Critical"], color,
                      TIER_COLORS["Moderate"], TIER_COLORS["Healthy"]],
        text=[f"{int(fleet['max_cycle'].min())} cycles", f"{max_cycle} cycles",
              f"{int(avg_life)} cycles", f"{int(fleet['max_cycle'].max())} cycles"],
        textposition="outside",
        hovertemplate="%{x}: <b>%{y} cycles</b><extra></extra>",
    ))
    fig_compare.update_layout(
        yaxis_title="Total Cycles",
        height=280,
        margin=dict(t=20, b=20, l=50, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_compare, width="stretch")

