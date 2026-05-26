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
    st.markdown("# 📖 About This Project")
    st.caption("Predicting aircraft engine failure before it happens. NASA C-MAPSS FD001.")
    st.divider()

    st.markdown("""
    ### The Problem

    Imagine you run an airline with 100 engines flying every day. Right now, most airlines
    service engines on a fixed schedule (every 500 flights, say) whether the engine needs
    it or not. That is like changing your car's oil every three months even if you have only
    driven 200 miles.

    The problem is that some engines degrade faster than others. A fixed schedule means you
    are either servicing perfectly healthy engines too early, wasting around **$50,000 per
    unnecessary visit**, or missing a degrading engine that fails mid-air, which costs
    **$500,000 or more** and puts lives at risk. That is a ten-times difference driven
    entirely by whether the failure was anticipated.

    This platform is built to change that. Instead of guessing based on a calendar, it reads
    live sensor data from each engine and predicts exactly how many cycles of useful life that
    engine has left.
    """)

    st.divider()

    st.markdown("### The Dataset")
    st.markdown("""
    The data comes from NASA's **C-MAPSS** simulation (Commercial Modular Aero-Propulsion
    System Simulation), which models how real turbofan aircraft engines wear down over time.
    Each of the 100 engines in the dataset starts healthy and runs cycle by cycle (one cycle
    is roughly one flight) until it fails. Every cycle, 21 sensors record what is happening
    inside the engine: temperatures, pressures, fan speeds, fuel flow, and more.

    Not all 21 sensors are useful. Eight of them barely change across an engine's entire
    lifetime and carry no signal about degradation. This platform focuses on the 13 sensors
    that do drift meaningfully as wear accumulates, because those are the ones that actually
    tell you something is changing.
    """)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training Engines", fleet["engine_id"].nunique())
    m2.metric("Total Cycles Recorded", f"{len(df):,}")
    m3.metric("Informative Sensors", 13)
    m4.metric("Avg Engine Lifetime", f"{fleet['max_cycle'].mean():.0f} cycles")

    st.divider()

    st.markdown("### Key Terms")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Engine ID** is a unique identifier from 1 to 100 assigned to each turbofan engine.
        Every engine is tracked independently from its very first cycle until the cycle it fails.

        **Cycle** is one complete operational run, roughly equivalent to one flight. Each engine
        starts at cycle 1, and the recording ends at whatever cycle the engine finally breaks down.

        **RUL (Remaining Useful Life)** is how many cycles an engine has left before it fails.
        At the very last cycle before failure, RUL is zero. Earlier in the engine's life, RUL
        is high. Watching it count down is how you know when to act.
        """)
    with col2:
        st.markdown("""
        **Sensors** are the 21 physical measurements recorded each cycle, including exhaust
        temperature, high-pressure turbine efficiency, and bypass ratio. They are the raw signal
        the model reads to understand how degraded an engine is. Only 13 of these change in ways
        that actually correlate with wear.

        **Degradation** is the gradual drift in sensor readings as an engine wears out. Some
        sensors trend upward over time, others trend downward. The pattern of those trends,
        taken together, is what reveals that an engine is approaching failure.

        **FD001** is the specific subset of C-MAPSS used here. It is the simplest version: one
        operating condition, one failure mode, 100 engines all run to failure. It is the right
        place to start building and validating a predictive maintenance system.
        """)

    st.divider()
    st.markdown("""
    ### What is Coming in Phase 2

    Phase 1, what you are looking at now, is the foundation: clean data, correct RUL
    calculations, and a dashboard that makes the dataset legible. Phase 2 will train an LSTM
    neural network on the sensor sequences to actually predict RUL for engines it has never
    seen before. It will also add confidence intervals so the system can say not just "6 cycles
    left" but "between 1 and 10 cycles, with 90% confidence." In safety-critical
    systems, knowing how uncertain a prediction is matters just as much as the prediction itself.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Fleet Overview
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Fleet Overview":
    st.markdown("# ✈️ Fleet Health Overview")
    st.caption("Engine lifetime analysis across 100 turbofan engines. NASA C-MAPSS FD001.")
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

    st.markdown("### Fleet Engine Lifetimes, Sorted Most Critical First")
    st.caption("Each bar represents one engine. Bar height shows total cycles before failure. Error bars show a 10% variability estimate.")

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
        xaxis=dict(title="Engines sorted by lifetime, most critical first", showticklabels=False),
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
        row["engine_id"]: f"Engine {row['engine_id']}  {tier_icons[row['tier']]} {row['tier'].upper()}"
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
        st.error(f"🔴 **CRITICAL** This engine had one of the shortest lifetimes in the fleet at {max_cycle} cycles. Engines in this range are highest priority for maintenance review.")
    elif tier == "High":
        st.warning(f"🟠 **HIGH RISK** This engine's lifetime of {max_cycle} cycles is below the fleet average of {avg_life:.0f} cycles.")

    st.divider()

    col_rul, col_sensor = st.columns(2)

    # RUL decline
    with col_rul:
        st.markdown("**Remaining Useful Life over Full Lifecycle**")
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
            "Sensors (each normalized to a 0 to 1 scale so trends are comparable)",
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

