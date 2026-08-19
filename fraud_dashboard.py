from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fraud & Chargeback Dashboard", layout="wide", page_icon="🛡️")


@st.cache_data
def load_data(path: str = "transactional-sample.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["date"] = df["transaction_date"].dt.date
    df["hour"] = df["transaction_date"].dt.hour
    df["day_of_week"] = df["transaction_date"].dt.day_name()
    df["bin"] = df["card_number"].astype(str).str[:6]
    df["has_cbk"] = df["has_cbk"].astype(bool)
    return df


@st.cache_data
def build_entity_graph(df: pd.DataFrame, max_nodes: int = 200) -> nx.Graph:
    """User <-> Device <-> Card link graph, edges weighted by shared transactions."""
    G = nx.Graph()
    for row in df.itertuples(index=False):
        entities = []
        if pd.notna(row.user_id):
            entities.append((f"user_{row.user_id}", "user"))
        if pd.notna(row.device_id):
            entities.append((f"device_{row.device_id}", "device"))
        if pd.notna(row.card_number):
            entities.append((f"card_{row.card_number}", "card"))
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                u, ut = entities[i]
                v, vt = entities[j]
                G.add_node(u, type=ut)
                G.add_node(v, type=vt)
                if G.has_edge(u, v):
                    G[u][v]["count"] += 1
                    G[u][v]["cbk"] += int(row.has_cbk)
                else:
                    G.add_edge(u, v, count=1, cbk=int(row.has_cbk))
    if len(G.nodes) > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
    return G


def recommended_action(rate: float) -> str:
    if rate >= 0.5:
        return "🔴 Block / manual review"
    if rate >= 0.2:
        return "🟠 Step-up authentication"
    return "🟢 Monitor"


def risk_table(df: pd.DataFrame, entity_col: str, min_txn: int) -> pd.DataFrame:
    g = (
        df.groupby(entity_col)["has_cbk"]
        .agg(transactions="count", cbk_rate="mean")
        .reset_index()
    )
    g = g[g["transactions"] >= min_txn].sort_values("cbk_rate", ascending=False)
    g["recommended_action"] = g["cbk_rate"].apply(recommended_action)
    g["cbk_rate"] = (g["cbk_rate"] * 100).round(1).astype(str) + "%"
    return g.rename(columns={entity_col: entity_col.replace("_", " ").title()})


# ---------------------------------------------------------------------------
# Data + sidebar filters
# ---------------------------------------------------------------------------
df_raw = load_data()

st.sidebar.header("Filters")

min_date, max_date = df_raw["date"].min(), df_raw["date"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

merchant_options = ["All"] + sorted(df_raw["merchant_id"].unique().tolist())
merchant_sel = st.sidebar.selectbox("Merchant", merchant_options)

bin_options = ["All"] + sorted(df_raw["bin"].unique().tolist())
bin_sel = st.sidebar.selectbox("BIN", bin_options)

min_txn = st.sidebar.slider("Minimum transactions per entity (risk tables & charts)", 1, 50, 10)

df = df_raw.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    df = df[(df["date"] >= start) & (df["date"] <= end)]
if merchant_sel != "All":
    df = df[df["merchant_id"] == merchant_sel]
if bin_sel != "All":
    df = df[df["bin"] == bin_sel]

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("🛡️ Fraud & Chargeback Analysis Dashboard")
st.caption(
 "Same EDA, explorable live."
)

if df.empty:
    st.warning("No transactions match the current filters.")
    st.stop()

overall_rate = df["has_cbk"].mean()
total_txn = len(df)
amount_at_risk = df.loc[df["has_cbk"], "transaction_amount"].sum()
total_amount = df["transaction_amount"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions", f"{total_txn:,}")
k2.metric("Chargeback Rate", f"{overall_rate:.1%}")
k3.metric("$ at Risk (chargebacks)", f"${amount_at_risk:,.0f}")
k4.metric("Total Volume", f"${total_amount:,.0f}")

st.divider()

# ---------------------------------------------------------------------------
# Daily volume + chargeback trend
# ---------------------------------------------------------------------------
daily = (
    df.groupby("date")
    .agg(volume=("transaction_id", "count"), cbk_rate=("has_cbk", "mean"))
    .reset_index()
)
fig_trend = go.Figure()
fig_trend.add_bar(
    x=daily["date"], y=daily["volume"], name="Volume", marker_color="steelblue", opacity=0.6
)
fig_trend.add_trace(
    go.Scatter(
        x=daily["date"],
        y=daily["cbk_rate"] * 100,
        name="Chargeback rate %",
        mode="lines+markers",
        line=dict(color="crimson"),
        yaxis="y2",
    )
)
fig_trend.update_layout(
    title="Daily Transaction Volume & Chargeback Rate",
    yaxis=dict(title="Volume"),
    yaxis2=dict(title="Chargeback rate (%)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=60),
)
st.plotly_chart(fig_trend, width='stretch')

# ---------------------------------------------------------------------------
# Chargeback rate by merchant / BIN
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    merch = (
        df.groupby("merchant_id")["has_cbk"]
        .agg(transactions="count", cbk_rate="mean")
        .reset_index()
    )
    merch = merch[merch["transactions"] >= min_txn].sort_values("cbk_rate", ascending=False).head(20)
    if merch.empty:
        st.info(f"No merchants with ≥{min_txn} transactions in this filter.")
    else:
        merch["merchant_id"] = merch["merchant_id"].astype(str)
        fig_m = px.bar(
            merch,
            x="merchant_id",
            y="cbk_rate",
            hover_data=["transactions"],
            title=f"Chargeback Rate by Merchant (≥{min_txn} txns, top 20)",
            labels={"cbk_rate": "Chargeback rate", "merchant_id": "Merchant ID"},
            color="cbk_rate",
            color_continuous_scale="Reds",
        )
        fig_m.update_yaxes(tickformat=".0%")
        fig_m.update_xaxes(type="category")
        fig_m.add_hline(
            y=overall_rate, line_dash="dash", line_color="gray",
            annotation_text="Portfolio avg", annotation_position="top left",
        )
        fig_m.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_m, width='stretch')

with col2:
    bins_df = (
        df.groupby("bin")["has_cbk"].agg(transactions="count", cbk_rate="mean").reset_index()
    )
    bins_df = bins_df[bins_df["transactions"] >= min_txn].sort_values("cbk_rate", ascending=False).head(20)
    if bins_df.empty:
        st.info(f"No BINs with ≥{min_txn} transactions in this filter.")
    else:
        fig_b = px.bar(
            bins_df,
            x="bin",
            y="cbk_rate",
            hover_data=["transactions"],
            title=f"Chargeback Rate by BIN (≥{min_txn} txns, top 20)",
            labels={"cbk_rate": "Chargeback rate", "bin": "Card BIN"},
            color="cbk_rate",
            color_continuous_scale="Reds",
        )
        fig_b.update_yaxes(tickformat=".0%")
        fig_b.update_xaxes(type="category")
        fig_b.add_hline(
            y=overall_rate, line_dash="dash", line_color="gray",
            annotation_text="Portfolio avg", annotation_position="top left",
        )
        fig_b.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_b, width='stretch')

# ---------------------------------------------------------------------------
# Amount distribution by chargeback status
# ---------------------------------------------------------------------------

df_hist = df.copy()
df_hist["log_amount"] = np.log10(df_hist["transaction_amount"])

fig_hist = px.histogram(
    df_hist,
    x="log_amount",
    color="has_cbk",
    nbins=60,
    barmode="overlay",
    opacity=0.65,
    color_discrete_map={True: "crimson", False: "steelblue"},
    labels={"has_cbk": "Chargeback", "log_amount": "Amount ($, log scale)"},
    title="Transaction Amount Distribution by Chargeback Status",
)

tick_dollars = [1, 5, 10, 50, 100, 500, 1000, 5000]
min_amt, max_amt = df["transaction_amount"].min(), df["transaction_amount"].max()
tick_dollars = [v for v in tick_dollars if min_amt * 0.8 <= v <= max_amt * 1.2]
fig_hist.update_xaxes(
    tickvals=[np.log10(v) for v in tick_dollars],
    ticktext=[f"${v:,.0f}" for v in tick_dollars],
)
st.plotly_chart(fig_hist, width='stretch')

# ---------------------------------------------------------------------------
# Top risky entities — operational worklist
# ---------------------------------------------------------------------------
st.subheader("Top Risky Entities — Operational Worklist")
tab_users, tab_devices, tab_cards = st.tabs(["Users", "Devices", "Cards"])

with tab_users:
    st.dataframe(risk_table(df, "user_id", min_txn), width='stretch', hide_index=True)
with tab_devices:
    st.dataframe(risk_table(df, "device_id", min_txn), width='stretch', hide_index=True)
with tab_cards:
    st.dataframe(risk_table(df, "card_number", min_txn), width='stretch', hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Fraud ring network graph
# ---------------------------------------------------------------------------
st.subheader("Fraud Ring Network — Entity Link Graph")
st.caption(
    "User ↔ Device ↔ Card graph built from shared transactions. Clusters where transactions "
    "are concentrated and chargebacks run high are the clearest sign of an organized ring "
    "(see Step 1.6 of the report)."
)

gc1, gc2 = st.columns(2)
min_edge_txn = gc1.slider("Min transactions per cluster", 1, 10, 3)
min_cluster_rate = gc2.slider("Min chargeback rate per cluster", 0.0, 1.0, 0.3, step=0.05)

G = build_entity_graph(df)

clusters = []
for comp in nx.connected_components(G):
    sub = G.subgraph(comp)
    total = sum(d["count"] for _, _, d in sub.edges(data=True))
    cbk = sum(d["cbk"] for _, _, d in sub.edges(data=True))
    rate = cbk / total if total else 0
    if total >= min_edge_txn and rate >= min_cluster_rate:
        clusters.append({"sub": sub, "total": total, "cbk": cbk, "rate": rate, "nodes": len(sub.nodes)})

clusters.sort(key=lambda c: c["rate"], reverse=True)
st.write(
    f"**{len(clusters)} suspicious cluster(s) found** "
    f"(≥{min_edge_txn} txns, ≥{min_cluster_rate:.0%} chargeback rate) out of "
    f"{G.number_of_nodes()} entities analyzed (sampled to top 200 by connectivity)."
)

if clusters:
    cluster_idx = 0
    if len(clusters) > 1:
        cluster_idx = st.selectbox(
            "Cluster to inspect",
            options=list(range(len(clusters))),
            format_func=lambda i: f"#{i+1} — {clusters[i]['nodes']} nodes, "
            f"{clusters[i]['total']} txns, {clusters[i]['rate']:.0%} CBK rate",
        )
    top = clusters[cluster_idx]
    sub = top["sub"]
    pos = nx.spring_layout(sub, k=1.5, seed=42)
    color_map = {"user": "#6baed6", "device": "#74c476", "card": "#fb6a4a"}

    edge_x, edge_y, edge_hover = [], [], []
    for u, v, d in sub.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color="#999"), hoverinfo="none", mode="lines"
    )

    node_x, node_y, node_color, node_text = [], [], [], []
    for node in sub.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(color_map[sub.nodes[node]["type"]])
        node_text.append(node)
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(color=node_color, size=20, line=dict(width=1, color="black")),
        hoverinfo="text",
    )

    fig_net = go.Figure(data=[edge_trace, node_trace])
    fig_net.update_layout(
        title=f"Cluster #{cluster_idx+1} — {top['nodes']} nodes, {top['total']} txns, {top['rate']:.0%} chargeback rate",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=520,
        margin=dict(t=60),
    )
    st.plotly_chart(fig_net, width='stretch')
    st.caption("🔵 User · 🟢 Device · 🔴 Card")
else:
    st.info("No clusters match the current thresholds — try lowering the minimums above.")

st.divider()

# ---------------------------------------------------------------------------
# ML risk scoring (Step 4 / Layer 2 of the report)
# ---------------------------------------------------------------------------
st.subheader("🎯 ML Risk Scoring — Layer 2")
st.caption(
    "Gradient-boosted trees model trained on the engineered feature store "
    "(`fs_transactional_data.csv`), scoring each transaction with a continuous "
    "risk score and routing it to one of three tiers instead of a single hard "
    "cutoff. Train with `python train_risk_model.py`."
)

MODEL_PATH = Path("risk_model.joblib")
SCORES_PATH = Path("risk_scores_test.csv")

if not MODEL_PATH.exists() or not SCORES_PATH.exists():
    st.warning(
        "No trained model found yet. Run `python train_risk_model.py` in this folder "
        "to train the model and generate `risk_model.joblib` / `risk_scores_test.csv`, "
        "then reload this page."
    )
else:
    @st.cache_resource
    def load_model(path: Path):
        return joblib.load(path)

    @st.cache_data
    def load_scores(path: Path) -> pd.DataFrame:
        s = pd.read_csv(path)
        s["transaction_date"] = pd.to_datetime(s["transaction_date"])
        s["bin"] = s["card_number"].astype(str).str[:6]
        return s

    bundle = load_model(MODEL_PATH)
    scores_df = load_scores(SCORES_PATH)
    low_thr, high_thr = bundle["low_threshold"], bundle["high_threshold"]

    if merchant_sel != "All":
        scores_df = scores_df[scores_df["merchant_id"] == merchant_sel]
    if bin_sel != "All":
        scores_df = scores_df[scores_df["bin"] == bin_sel]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Held-out test transactions", f"{bundle['test_size']:,}")
    m1.caption("Most recent 20% of transactions by date — never seen during training")
    m2.metric("ROC-AUC", f"{bundle['test_roc_auc']:.3f}")
    m3.metric("PR-AUC", f"{bundle['test_pr_auc']:.3f}", help=f"Baseline (random) = {bundle['test_cbk_rate']:.3f}")
    m4.metric("Tier thresholds", f"{low_thr:.2f} / {high_thr:.2f}")

    if scores_df.empty:
        st.info("No held-out transactions match the current Merchant/BIN filter.")
    else:
        left, right = st.columns([3, 2])

        with left:
            fig_score = px.histogram(
                scores_df,
                x="risk_score",
                color="has_cbk",
                nbins=40,
                barmode="overlay",
                opacity=0.65,
                range_x=[0, 1],
                color_discrete_map={True: "crimson", False: "steelblue"},
                labels={"has_cbk": "Chargeback", "risk_score": "Model risk score"},
                title="Risk Score Distribution (held-out test set)",
            )
            fig_score.add_vline(x=low_thr, line_dash="dash", line_color="orange",
                                 annotation_text="auto-approve / step-up")
            fig_score.add_vline(x=high_thr, line_dash="dash", line_color="crimson",
                                 annotation_text="step-up / decline")
            st.plotly_chart(fig_score, width='stretch')

        with right:
            tier_order = ["Auto-approve", "Step-up authentication", "Manual review / decline"]
            tier_summary = (
                scores_df.groupby("tier")["has_cbk"]
                .agg(transactions="count", chargeback_rate="mean")
                .reindex(tier_order)
                .dropna(how="all")
                .reset_index()
            )
            fig_tier = px.bar(
                tier_summary,
                x="tier",
                y="chargeback_rate",
                text="transactions",
                color="tier",
                color_discrete_map={
                    "Auto-approve": "#2ca02c",
                    "Step-up authentication": "#ff9f1c",
                    "Manual review / decline": "#d62728",
                },
                title="Chargeback Rate by Decision Tier",
                labels={"chargeback_rate": "Chargeback rate", "tier": ""},
            )
            fig_tier.update_traces(texttemplate="%{text} txns", textposition="outside")
            fig_tier.update_yaxes(tickformat=".0%")
            fig_tier.update_layout(showlegend=False, margin=dict(t=60))
            st.plotly_chart(fig_tier, width='stretch')

        fig_imp = px.bar(
            bundle["feature_importance"].head(10).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Features (permutation importance, scored on PR-AUC)",
            labels={"importance": "Importance (PR-AUC drop)", "feature": ""},
        )
        fig_imp.update_layout(margin=dict(t=60))
        st.plotly_chart(fig_imp, width='stretch')

        st.markdown("**Scored transactions (held-out test set)**")
        tier_filter = st.multiselect(
            "Show tiers", options=tier_order, default=tier_order, key="ml_tier_filter"
        )
        display_cols = [
            "transaction_id", "transaction_date", "merchant_id", "user_id",
            "transaction_amount", "risk_score", "tier", "has_cbk",
        ]
        table = (
            scores_df[scores_df["tier"].isin(tier_filter)][display_cols]
            .sort_values("risk_score", ascending=False)
            .reset_index(drop=True)
        )
        table["risk_score"] = table["risk_score"].round(3)
        st.dataframe(table, width='stretch', hide_index=True)

st.divider()
st.caption("Data: transactional-sample.csv · Companion dashboard to the REPORT case-study deliverable.")
