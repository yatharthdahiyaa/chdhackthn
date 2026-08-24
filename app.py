"""
app.py
------
Streamlit dashboard for the Unified Analytics Platform MVP.

Tabs:
  1. Overview  — ranked entity risk list + summary metrics
  2. Entity Profile — cross-source timeline + identity card
  3. Graph Explorer — interactive network graph (PyVis)
  4. Geo Map — tower/transaction geo overlay (Plotly)

Usage:
    streamlit run app.py
"""

import os
import json
import pickle
import math
import tempfile
from pathlib import Path
from datetime import timezone, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network

from config import (
    RISK_FILE, EVENTS_FILE, ENTITIES_FILE, GRAPH_FILE,
    GROUND_TRUTH_FILE, OUTPUT_DIR, ATM_LOCATIONS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FusionWatch — Unified Analytics Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background override */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    color: #ffffff !important;
}

/* Risk tier badges */
.badge-high {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-medium {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-low {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Cards */
.entity-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    transition: all 0.2s;
}
.entity-card:hover {
    border-color: #3b82f6;
    transform: translateY(-1px);
}

/* Header */
.platform-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    border: 1px solid #1d4ed8;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}

/* Data source pills */
.source-pill-cdr    { background: #1d4ed8; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
.source-pill-ipdr   { background: #7c3aed; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
.source-pill-tx     { background: #059669; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
.source-pill-social { background: #d97706; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }

/* Timeline entries */
.timeline-cdr    { border-left: 3px solid #3b82f6; }
.timeline-ipdr   { border-left: 3px solid #8b5cf6; }
.timeline-tx     { border-left: 3px solid #10b981; }
.timeline-social { border-left: 3px solid #f59e0b; }

/* Explanation box */
.explanation-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #475569;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #cbd5e1;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_risk_scores():
    if not os.path.exists(RISK_FILE):
        return pd.DataFrame()
    df = pd.read_parquet(RISK_FILE)
    return df


@st.cache_data(show_spinner=False)
def load_events():
    if not os.path.exists(EVENTS_FILE):
        return pd.DataFrame()
    df = pd.read_parquet(EVENTS_FILE)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_entities():
    if not os.path.exists(ENTITIES_FILE):
        return pd.DataFrame()
    return pd.read_parquet(ENTITIES_FILE)


@st.cache_resource(show_spinner=False)
def load_graph():
    if not os.path.exists(GRAPH_FILE):
        return None, {}, {}, []
    with open(GRAPH_FILE, "rb") as f:
        data = pickle.load(f)
    return (
        data.get("graph"),
        data.get("root_map", {}),
        data.get("validation", {}),
        data.get("rules", []),
    )


@st.cache_data(show_spinner=False)
def load_ground_truth():
    if not os.path.exists(GROUND_TRUTH_FILE):
        return {}
    with open(GROUND_TRUTH_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIER_COLORS = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
SOURCE_COLORS = {
    "CDR": "#3b82f6",
    "IPDR": "#8b5cf6",
    "TRANSACTIONS": "#10b981",
    "SOCIAL": "#f59e0b",
}

def tier_badge(tier: str) -> str:
    cls = f"badge-{tier.lower()}"
    return f'<span class="{cls}">{tier}</span>'


def source_pill(source: str) -> str:
    s = source.lower().replace("transactions", "tx")
    return f'<span class="source-pill-{s}">{source}</span>'


def score_bar(score: float, width_px: int = 200) -> str:
    pct = int(score * 100)
    color = "#ef4444" if score >= 0.7 else "#f59e0b" if score >= 0.4 else "#10b981"
    return (
        f'<div style="background:#1e293b;border-radius:6px;height:8px;width:{width_px}px;">'
        f'<div style="background:{color};border-radius:6px;height:8px;width:{int(pct/100*width_px)}px;"></div>'
        f'</div>'
        f'<span style="font-size:0.85rem;color:#94a3b8;margin-left:6px;">{pct}%</span>'
    )


def parse_attrs(attrs_str) -> dict:
    try:
        return json.loads(attrs_str) if isinstance(attrs_str, str) else {}
    except Exception:
        return {}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def safe_int(v, default: int = 0) -> int:
    try:
        if v is None or pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def run_full_pipeline():
    """Runs the full 4-stage pipeline."""
    import generate_synthetic_data
    import ingest
    import entity_resolution
    import analytics

    generate_synthetic_data.main()
    ingest.run()
    entity_resolution.run()
    analytics.run()
    st.cache_data.clear()
    st.cache_resource.clear()


def render_sidebar(risk_df: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0;">
            <div style="font-size:2rem;">🔍</div>
            <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;">FusionWatch</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">Unified Analytics Platform</div>
        </div>
        <hr style="border-color:#334155;margin:12px 0;">
        """, unsafe_allow_html=True)

        st.markdown("**Data Sources**")
        for src, color in SOURCE_COLORS.items():
            label = "Transactions" if src == "TRANSACTIONS" else src
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                f'<div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>'
                f'<span style="font-size:0.85rem;">{label}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:#334155;margin:12px 0;'>", unsafe_allow_html=True)

        if not risk_df.empty:
            st.markdown("**Risk Distribution**")
            tier_counts = risk_df["risk_tier"].value_counts()
            for tier in ["HIGH", "MEDIUM", "LOW"]:
                count = tier_counts.get(tier, 0)
                color = TIER_COLORS[tier]
                pct = int(count / len(risk_df) * 100) if len(risk_df) > 0 else 0
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;margin:6px 0;">'
                    f'<span style="color:{color};font-weight:600;">{tier}</span>'
                    f'<span style="color:#94a3b8;">{count} ({pct}%)</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='border-color:#334155;margin:12px 0;'>", unsafe_allow_html=True)
        if st.button("🔄 Re-run Pipeline", use_container_width=True):
            with st.spinner("Running full analytics pipeline..."):
                run_full_pipeline()
            st.rerun()

        st.markdown("<hr style='border-color:#334155;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.7rem;color:#475569;text-align:center;">'
            '⚠️ Synthetic data only<br>Demo / research use<br>'
            'No real persons or accounts</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

def render_overview(risk_df: pd.DataFrame, events_df: pd.DataFrame):
    # Header
    st.markdown("""
    <div class="platform-header">
        <div style="font-size:2.5rem;">🔍</div>
        <div>
            <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;">FusionWatch</div>
            <div style="font-size:0.9rem;color:#64748b;">
                Unified CDR · IPDR · Financial · Social/OSINT Analytics Platform
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if risk_df.empty:
        st.error("No risk scores found. Run the pipeline first: `python generate_synthetic_data.py && python ingest.py && python entity_resolution.py && python analytics.py`")
        return

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Entities", len(risk_df))
    with col2:
        high_count = (risk_df["risk_tier"] == "HIGH").sum()
        st.metric("🔴 High Risk", high_count)
    with col3:
        med_count = (risk_df["risk_tier"] == "MEDIUM").sum()
        st.metric("🟡 Medium Risk", med_count)
    with col4:
        total_events = len(events_df) if not events_df.empty else 0
        st.metric("Total Events", f"{total_events:,}")
    with col5:
        avg_risk = risk_df["risk_score"].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk score distribution
    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        st.markdown("#### Risk Score Distribution")
        fig_hist = px.histogram(
            risk_df, x="risk_score", nbins=20, color="risk_tier",
            color_discrete_map=TIER_COLORS,
            template="plotly_dark",
            labels={"risk_score": "Composite Risk Score", "count": "Entities"},
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            height=280,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart2:
        st.markdown("#### Events by Source")
        if not events_df.empty:
            src_counts = events_df["source"].value_counts().reset_index()
            src_counts.columns = ["source", "count"]
            fig_pie = px.pie(
                src_counts, names="source", values="count",
                color="source",
                color_discrete_map=SOURCE_COLORS,
                template="plotly_dark",
                hole=0.55,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                showlegend=True,
                legend=dict(font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Flagged entities table
    st.markdown("---")
    st.markdown("#### 🚨 Flagged & Anomalous Entities")

    filter_tier = st.multiselect(
        "Filter by risk tier",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM"],
        key="overview_tier_filter",
    )

    display_df = risk_df[risk_df["risk_tier"].isin(filter_tier)].copy()

    for _, row in display_df.head(30).iterrows():
        col_a, col_b, col_c = st.columns([3, 1, 4])
        with col_a:
            st.markdown(
                f'<div style="font-weight:600;color:#f1f5f9;">{row.get("name","") or row["entity_id"]}</div>'
                f'<div style="font-size:0.78rem;color:#64748b;">{row["entity_id"]} · '
                f'Community #{row.get("community_id","?")} · '
                f'PageRank {safe_float(row.get("pagerank",0)):.4f}</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(tier_badge(row["risk_tier"]), unsafe_allow_html=True)
            st.markdown(
                f'<div style="margin-top:4px;">{score_bar(safe_float(row.get("risk_score", 0)))}</div>',
                unsafe_allow_html=True,
            )
        with col_c:
            explanation = row.get("explanation", "")
            st.markdown(
                f'<div class="explanation-box">{explanation[:220]}{"…" if len(str(explanation)) > 220 else ""}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2 — Entity Profile
# ---------------------------------------------------------------------------

def render_entity_profile(risk_df: pd.DataFrame, events_df: pd.DataFrame):
    st.markdown("#### 🔎 Entity Profile")

    if risk_df.empty:
        st.warning("Pipeline output not found. Please run the pipeline first.")
        return

    # Search
    search = st.text_input("Search entity by name or ID", placeholder="e.g. E003 or Raj")
    if search:
        mask = (
            risk_df["entity_id"].str.contains(search, case=False, na=False) |
            risk_df["name"].str.contains(search, case=False, na=False)
        )
        matches = risk_df[mask]
    else:
        matches = risk_df

    if matches.empty:
        st.info("No matching entities found.")
        return

    selected_id = st.selectbox(
        "Select entity",
        matches["entity_id"].tolist(),
        format_func=lambda eid: f"{eid} — {risk_df[risk_df['entity_id']==eid]['name'].values[0] if not risk_df[risk_df['entity_id']==eid].empty else ''}",
    )

    row = risk_df[risk_df["entity_id"] == selected_id]
    if row.empty:
        st.warning("Entity not found.")
        return
    row = row.iloc[0]

    # Identity card
    col_id, col_scores = st.columns([2, 3])

    with col_id:
        tier_color = TIER_COLORS[row["risk_tier"]]
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1e293b,#0f172a);
                    border:1px solid {tier_color};
                    border-radius:16px;padding:20px;">
            <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;margin-bottom:4px;">
                {row.get('name','') or selected_id}
            </div>
            <div style="margin-bottom:16px;">{tier_badge(row['risk_tier'])}</div>
            <div style="font-size:0.82rem;color:#94a3b8;">
                <div style="margin:4px 0;"><b>Entity ID:</b> {selected_id}</div>
                <div style="margin:4px 0;"><b>Community:</b> #{row.get('community_id','?')}</div>
                <div style="margin:4px 0;"><b>PageRank:</b> {safe_float(row.get('pagerank',0)):.5f}</div>
                <div style="margin:4px 0;"><b>Degree (in/out):</b> {safe_int(row.get('degree_in',0))}/{safe_int(row.get('degree_out',0))}</div>
                <div style="margin:4px 0;"><b>Risk Score:</b> {safe_float(row.get('risk_score',0)):.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_scores:
        st.markdown("**Source-Level Scores**")
        score_data = {
            "CDR": safe_float(row.get("cdr_score", 0)),
            "IPDR": safe_float(row.get("ipdr_score", 0)),
            "Transactions": safe_float(row.get("tx_score", 0)),
            "Social": safe_float(row.get("social_score", 0)),
            "Graph": safe_float(row.get("graph_score", 0)),
        }
        colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#f43f5e"]
        fig_bar = go.Figure(go.Bar(
            x=list(score_data.values()),
            y=list(score_data.keys()),
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in score_data.values()],
            textposition="auto",
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            xaxis=dict(range=[0, 1], gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Explanation
    explanation = row.get("explanation", "")
    if explanation:
        st.markdown(f'<div class="explanation-box">💡 {explanation}</div>', unsafe_allow_html=True)

    # NER entities
    ner_raw = row.get("ner_entities", "[]")
    try:
        ner_ents = json.loads(ner_raw) if isinstance(ner_raw, str) else []
    except Exception:
        ner_ents = []
    if ner_ents:
        st.markdown("**Named Entities Mentioned in Posts (spaCy NER)**")
        ner_cols = st.columns(min(len(ner_ents), 5))
        for i, ent in enumerate(ner_ents[:10]):
            with ner_cols[i % 5]:
                label_colors = {"PERSON": "#3b82f6", "ORG": "#8b5cf6",
                                "GPE": "#10b981", "LOC": "#f59e0b"}
                color = label_colors.get(ent.get("label", ""), "#64748b")
                st.markdown(
                    f'<div style="background:{color}22;border:1px solid {color};'
                    f'border-radius:8px;padding:6px 10px;margin:4px 0;font-size:0.8rem;">'
                    f'<span style="color:{color};font-weight:600;">{ent["label"]}</span>'
                    f'<br><span style="color:#f1f5f9;">{ent["text"]}</span></div>',
                    unsafe_allow_html=True,
                )

    # Cross-source timeline
    st.markdown("---")
    st.markdown("**📅 Cross-Source Event Timeline**")

    entity_events = events_df[events_df["entity_id"] == selected_id].copy()
    if entity_events.empty:
        st.info("No events found for this entity.")
        return

    entity_events = entity_events.sort_values("timestamp_utc")

    # Timeline chart
    fig_tl = go.Figure()
    for src, color in SOURCE_COLORS.items():
        sub = entity_events[entity_events["source"] == src]
        if sub.empty:
            continue
        fig_tl.add_trace(go.Scatter(
            x=sub["timestamp_utc"],
            y=[src] * len(sub),
            mode="markers",
            name=src,
            marker=dict(color=color, size=8, opacity=0.8,
                        line=dict(color=color, width=1)),
            hovertemplate=(
                f"<b>{src}</b><br>"
                "%{x|%Y-%m-%d %H:%M}<br>"
                "Type: %{customdata}<extra></extra>"
            ),
            customdata=sub["event_type"].tolist(),
        ))

    fig_tl.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8")),
        xaxis=dict(gridcolor="#1e293b", title="Timeline"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # Event table
    st.markdown("**Recent Events**")
    display_cols = ["timestamp_utc", "source", "event_type"]
    tbl = entity_events[display_cols].tail(50).sort_values("timestamp_utc", ascending=False)
    tbl["timestamp_utc"] = tbl["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M UTC")

    styler = tbl.style
    if hasattr(styler, "map"):
        styler = styler.map(
            lambda v: f"color: {SOURCE_COLORS.get(v, '#94a3b8')}" if v in SOURCE_COLORS else "",
            subset=["source"],
        )
    elif hasattr(styler, "applymap"):
        styler = styler.applymap(
            lambda v: f"color: {SOURCE_COLORS.get(v, '#94a3b8')}" if v in SOURCE_COLORS else "",
            subset=["source"],
        )
    st.dataframe(
        styler,
        use_container_width=True,
        height=300,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Graph Explorer
# ---------------------------------------------------------------------------

def render_graph_explorer(G, risk_df: pd.DataFrame):
    st.markdown("#### Entity Graph Explorer")

    if G is None:
        st.warning("Entity graph not found. Run `python entity_resolution.py` first.")
        return

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        max_nodes = st.slider("Max nodes to display", 10, min(50, G.number_of_nodes()), 30)
    with col_ctrl2:
        color_by = st.selectbox("Color nodes by", ["Risk Tier", "Community", "Source Degree"])
    with col_ctrl3:
        max_edges_per_type = st.slider("Max edges per type", 20, 200, 50)

    # Build display subgraph: top-N nodes by degree
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
    sub_G = G.subgraph(top_nodes).copy()

    # CRITICAL: Aggregate MultiDiGraph to simple DiGraph with edge counts
    # This prevents 8,000+ edges from going to vis.js which would freeze the browser
    simple_G = nx.DiGraph()
    edge_weight_map: dict = {}  # (u, v, etype) -> count
    for u, v, data in sub_G.edges(data=True):
        if u == v:
            continue
        etype = data.get("edge_type", "RELATED")
        key = (u, v, etype)
        edge_weight_map[key] = edge_weight_map.get(key, 0) + 1

    # Sort by weight, cap per edge-type
    from collections import defaultdict
    per_type_counts: dict = defaultdict(int)
    MAX_PER_TYPE = max_edges_per_type
    aggregated_edges = []
    for (u, v, etype), cnt in sorted(edge_weight_map.items(), key=lambda x: -x[1]):
        if per_type_counts[etype] < MAX_PER_TYPE:
            aggregated_edges.append((u, v, etype, cnt))
            per_type_counts[etype] += 1
            simple_G.add_edge(u, v, edge_type=etype, weight=cnt)

    # Build PyVis network — physics DISABLED to prevent freeze
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0a0e1a",
        font_color="#e2e8f0",
        directed=True,
    )
    # Disable physics on load — user can re-enable with the UI button
    net.set_options("""{
        "physics": {
            "enabled": false
        },
        "layout": {
            "randomSeed": 42
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "navigationButtons": true
        },
        "edges": {
            "smooth": {"type": "curvedCW", "roundness": 0.1}
        }
    }""")

    # Risk score lookup
    risk_lookup = {}
    tier_lookup  = {}
    comm_lookup  = {}
    if not risk_df.empty:
        risk_lookup = dict(zip(risk_df["entity_id"], risk_df["risk_score"]))
        tier_lookup  = dict(zip(risk_df["entity_id"], risk_df["risk_tier"]))
        comm_lookup  = dict(zip(risk_df["entity_id"],
                                risk_df.get("community_id", pd.Series(dtype=int))))

    COMM_PALETTE = [
        "#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#f43f5e",
        "#06b6d4", "#84cc16", "#ec4899", "#6366f1", "#14b8a6",
    ]

    for node in simple_G.nodes:
        ndata = G.nodes.get(node, {})
        risk  = safe_float(risk_lookup.get(node, 0))
        tier  = tier_lookup.get(node, "LOW")
        comm  = safe_int(comm_lookup.get(node, 0))

        if color_by == "Risk Tier":
            color = TIER_COLORS.get(tier, "#64748b")
        elif color_by == "Community":
            color = COMM_PALETTE[comm % len(COMM_PALETTE)]
        else:
            deg = G.degree(node)
            intensity = min(int(deg / 20 * 255), 255)
            color = f"#{intensity:02x}6f{255-intensity:02x}"

        label = ndata.get("name", node)[:20] or node
        size  = max(12, min(40, 12 + int(risk * 28) + G.degree(node) // 5))
        title = (
            f"<b>{label}</b><br>"
            f"ID: {node}<br>"
            f"Risk: {risk:.2f} ({tier})<br>"
            f"Community: #{comm}<br>"
            f"Degree: {G.degree(node)}"
        )

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=3 if tier in ("HIGH", "MEDIUM") else 1,
            borderWidthSelected=5,
        )

    edge_type_colors = {
        "CALLED": "#3b82f6",
        "TRANSACTED": "#10b981",
        "MENTIONED": "#f59e0b",
    }
    for u, v, etype, cnt in aggregated_edges:
        ecolor = edge_type_colors.get(etype, "#475569")
        width  = min(1 + cnt // 10, 5)
        net.add_edge(
            u, v,
            color=ecolor,
            width=width,
            arrows="to",
            title=f"{etype} (x{cnt})",
        )

    # Render to temp HTML file
    tmp_html = os.path.join(tempfile.gettempdir(), "fw_graph.html")
    net.save_graph(tmp_html)
    with open(tmp_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Legend
    legend_html = """
    <div style="display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap;">
        <span style="color:#ef4444;font-size:0.8rem;">&#9679; HIGH risk</span>
        <span style="color:#f59e0b;font-size:0.8rem;">&#9679; MEDIUM risk</span>
        <span style="color:#10b981;font-size:0.8rem;">&#9679; LOW risk</span>
        <span style="font-size:0.8rem;color:#64748b;">|</span>
        <span style="color:#3b82f6;font-size:0.8rem;">-- CDR (Called)</span>
        <span style="color:#10b981;font-size:0.8rem;">-- TX (Transacted)</span>
        <span style="color:#f59e0b;font-size:0.8rem;">-- Social (Mentioned)</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    components.html(html_content, height=620, scrolling=False)

    st.caption(
        f"Showing {simple_G.number_of_nodes()} nodes | "
        f"{simple_G.number_of_edges()} aggregated edges "
        f"({sum(per_type_counts.values())} displayed, physics disabled for performance)"
    )


# ---------------------------------------------------------------------------
# Tab 4 — Geo Map
# ---------------------------------------------------------------------------

def render_geo_map(events_df: pd.DataFrame, risk_df: pd.DataFrame):
    st.markdown("#### 🗺️ Geospatial Overlay")
    st.caption("CDR tower locations, ATM sites, and geotagged social posts")

    if events_df.empty:
        st.warning("Events not loaded.")
        return

    geo_points = []

    # CDR tower locations
    cdr_evts = events_df[events_df["source"] == "CDR"].copy()
    for _, row in cdr_evts.iterrows():
        attrs = parse_attrs(row["attributes_json"])
        lat = attrs.get("tower_lat")
        lon = attrs.get("tower_lon")
        if lat and lon:
            try:
                geo_points.append({
                    "lat": float(lat), "lon": float(lon),
                    "type": "CDR Tower",
                    "entity_id": row["entity_id"],
                    "label": f"CDR: {attrs.get('call_type','')} @ {str(row['timestamp_utc'])[:16]}",
                    "color": SOURCE_COLORS["CDR"],
                    "size": 5,
                })
            except Exception:
                pass

    # Social geotagged posts
    soc_evts = events_df[events_df["source"] == "SOCIAL"].copy()
    for _, row in soc_evts.iterrows():
        attrs = parse_attrs(row["attributes_json"])
        geotag = attrs.get("geotag")
        if geotag and isinstance(geotag, dict):
            lat = geotag.get("lat")
            lon = geotag.get("lon")
            if lat and lon:
                geo_points.append({
                    "lat": float(lat), "lon": float(lon),
                    "type": "Social Post",
                    "entity_id": row["entity_id"],
                    "label": f"Post: {attrs.get('text','')[:50]}",
                    "color": SOURCE_COLORS["SOCIAL"],
                    "size": 8,
                })

    # ATM locations
    for i, (lat, lon) in enumerate(ATM_LOCATIONS):
        geo_points.append({
            "lat": lat, "lon": lon,
            "type": "ATM Site",
            "entity_id": "ATM",
            "label": f"ATM #{i+1} (Chandigarh region)",
            "color": "#f43f5e",
            "size": 15,
        })

    if not geo_points:
        st.info("No geo-located events found in the current dataset.")
        return

    geo_df = pd.DataFrame(geo_points)

    # Sample for performance
    max_pts = 2000
    if len(geo_df) > max_pts:
        geo_df = geo_df.sample(max_pts, random_state=42)

    # Merge risk tier for color enhancement
    if not risk_df.empty:
        geo_df = geo_df.merge(
            risk_df[["entity_id", "risk_tier"]].rename(columns={"risk_tier": "entity_risk"}),
            on="entity_id",
            how="left",
        )
        geo_df["entity_risk"] = geo_df["entity_risk"].fillna("LOW")

    map_kwargs = dict(
        lat="lat",
        lon="lon",
        color="type",
        size="size",
        hover_name="label",
        hover_data={"entity_id": True, "lat": False, "lon": False, "size": False},
        color_discrete_map={
            "CDR Tower":   SOURCE_COLORS["CDR"],
            "Social Post": SOURCE_COLORS["SOCIAL"],
            "ATM Site":    "#f43f5e",
        },
        zoom=7,
        center={"lat": 30.9, "lon": 76.9},
        template="plotly_dark",
    )
    if hasattr(px, "scatter_mapbox"):
        fig_map = px.scatter_mapbox(geo_df, mapbox_style="open-street-map", **map_kwargs)
    else:
        fig_map = px.scatter_map(geo_df, map_style="open-street-map", **map_kwargs)
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(
            bgcolor="rgba(15,23,42,0.9)",
            bordercolor="#334155",
            font=dict(color="#94a3b8"),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.caption(f"{len(geo_df):,} geo-points rendered (CDR towers, geotagged posts, ATM sites)")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # Load data
    with st.spinner("Loading pipeline outputs…"):
        risk_df   = load_risk_scores()
        events_df = load_events()
        entities_df = load_entities()
        G, root_map, validation, rules = load_graph()

    # Check if pipeline has been run
    data_missing = risk_df.empty or events_df.empty

    render_sidebar(risk_df)

    # Validation status banner
    if not data_missing and validation:
        all_pass = all(v.get("pass", False) for v in validation.values())
        banner_color = "#10b981" if all_pass else "#f59e0b"
        banner_icon  = "✅" if all_pass else "⚠️"
        banner_text  = (
            f"{banner_icon} Entity resolution ground-truth: "
            + " · ".join(f"{k}: {'PASS' if v.get('pass') else 'FAIL'}"
                         for k, v in validation.items())
        )
        st.markdown(
            f'<div style="background:{banner_color}22;border:1px solid {banner_color};'
            f'border-radius:8px;padding:8px 16px;margin-bottom:12px;'
            f'font-size:0.85rem;color:{banner_color};">{banner_text}</div>',
            unsafe_allow_html=True,
        )

    if data_missing:
        st.info("👋 Welcome to **FusionWatch**! Initializing pipeline outputs...")
        with st.spinner("🚀 Running pipeline (Generating synthetic data -> Ingestion -> Entity Resolution -> Analytics)..."):
            run_full_pipeline()
        st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔎 Entity Profile",
        "🕸️ Graph Explorer",
        "🗺️ Geo Map",
    ])

    with tab1:
        render_overview(risk_df, events_df)

    with tab2:
        render_entity_profile(risk_df, events_df)

    with tab3:
        render_graph_explorer(G, risk_df)

    with tab4:
        render_geo_map(events_df, risk_df)


if __name__ == "__main__":
    main()
