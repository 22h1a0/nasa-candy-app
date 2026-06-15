"""
Nassau Candy Distributor
Factory Reallocation & Shipping Optimization Dashboard
"""

import os, sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_APP_DIR)
sys.path.insert(0, _APP_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_prep import (
    load_and_prepare, FACTORIES, PRODUCT_FACTORY, REGION_COORDS, haversine_km
)
from models_train import (
    train_and_evaluate, load_model,
    predict_lead_time, simulate_factory_reassignment, generate_top_recommendations
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy | Factory Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#0f1117; }
  [data-testid="stSidebar"]          { background:#1a1d27; border-right:1px solid #2d3148; }
  .block-container                   { padding-top:1.5rem; }

  .kpi-card {
    background: linear-gradient(135deg,#1e2235,#252942);
    border:1px solid #3a3f5c; border-radius:12px;
    padding:18px 22px; text-align:center;
  }
  .kpi-val   { font-size:2rem; font-weight:700; color:#7c83f5; }
  .kpi-label { font-size:.78rem; color:#9095b0; margin-top:2px; letter-spacing:.05em; text-transform:uppercase; }
  .kpi-delta { font-size:.82rem; margin-top:4px; }

  .section-title {
    font-size:1.1rem; font-weight:600; color:#c8caf0;
    border-left:3px solid #7c83f5; padding-left:10px;
    margin:1.2rem 0 .6rem;
  }
  .badge-green { background:#1a3a2a; color:#4eca7f; border-radius:6px;
                 padding:2px 8px; font-size:.78rem; font-weight:600; }
  .badge-red   { background:#3a1a1a; color:#e05c5c; border-radius:6px;
                 padding:2px 8px; font-size:.78rem; font-weight:600; }
  .badge-blue  { background:#1a2a3a; color:#5cadec; border-radius:6px;
                 padding:2px 8px; font-size:.78rem; font-weight:600; }

  div[data-testid="stSelectbox"] label,
  div[data-testid="stSlider"]    label { color:#9095b0 !important; font-size:.82rem; }

  .stTabs [data-baseweb="tab-list"]      { gap:8px; }
  .stTabs [data-baseweb="tab"]           { background:#1e2235; border-radius:8px 8px 0 0;
                                           color:#9095b0; padding:8px 20px; }
  .stTabs [aria-selected="true"]         { background:#7c83f5; color:#fff; }
</style>
""", unsafe_allow_html=True)

# ── Load / train model ────────────────────────────────────────────────────────
DATA_PATH = os.path.join(_APP_DIR, "data", "Nassau_Candy_Distributor.csv")
MODEL_PKL = os.path.join(_APP_DIR, "models", "best_model.pkl")

@st.cache_resource(show_spinner="🔧 Training ML models…")
def get_model_and_data():
    if os.path.exists(MODEL_PKL):
        try:
            model, encoders, best_name, all_results, all_models = load_model()
        except Exception:
            model, encoders, all_results, best_name, _ = train_and_evaluate(DATA_PATH)
            _, _, _, _, all_models = load_model()
    else:
        model, encoders, all_results, best_name, _ = train_and_evaluate(DATA_PATH)
        _, _, _, _, all_models = load_model()
    df = load_and_prepare(DATA_PATH)
    return model, encoders, best_name, all_results, all_models, df

model, encoders, best_name, all_results, all_models, df = get_model_and_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍬 Nassau Candy")
    st.markdown("**Factory Optimizer**")
    st.divider()

    st.markdown("### 🎯 Filters")
    sel_product   = st.selectbox("Product",   sorted(PRODUCT_FACTORY.keys()))
    sel_region    = st.selectbox("Region",    list(REGION_COORDS.keys()))
    sel_ship_mode = st.selectbox("Ship Mode", ["Standard Class","First Class","Second Class","Same Day"])
    opt_priority  = st.slider("Optimization Priority", 0, 100, 50,
                               help="0 = maximize profit, 100 = minimize lead time")
    st.divider()
    st.markdown("### ℹ️ Dataset")
    st.metric("Total Orders",   f"{len(df):,}")
    st.metric("Products",       len(PRODUCT_FACTORY))
    st.metric("Factories",      len(FACTORIES))
    st.metric("Regions",        len(REGION_COORDS))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏭 Factory Reallocation & Shipping Optimizer")
st.markdown(
    f"*Best model: **{best_name}** · R²={all_results[best_name]['R2']} · "
    f"MAE={all_results[best_name]['MAE']} days*"
)

# ── KPI Row ───────────────────────────────────────────────────────────────────
avg_lt      = df["Lead Time"].mean()
avg_margin  = df["Profit_Margin"].mean()
total_profit= df["Gross Profit"].sum()
avg_dist    = df["Distance_km"].mean()

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl, delta in [
    (c1, f"{avg_lt:.1f} days",   "Avg Lead Time",      "⏱ All orders"),
    (c2, f"{avg_margin:.1f}%",   "Avg Profit Margin",  "💰 Gross profit"),
    (c3, f"${total_profit:,.0f}","Total Gross Profit",  "📦 Full dataset"),
    (c4, f"{avg_dist:.0f} km",   "Avg Factory Distance","📍 To region"),
]:
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-val">{val}</div>
      <div class="kpi-label">{lbl}</div>
      <div class="kpi-delta" style="color:#9095b0">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏭 Factory Simulator",
    "🔄 What-If Analysis",
    "🏆 Recommendations",
    "📊 EDA & Insights",
    "🤖 Model Evaluation",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Factory Simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Factory Performance Simulator</div>', unsafe_allow_html=True)
    st.caption(f"Showing predictions for: **{sel_product}** → **{sel_region}** via **{sel_ship_mode}**")

    sim_df = simulate_factory_reassignment(model, encoders, df, sel_product, sel_region, sel_ship_mode)
    current_factory = PRODUCT_FACTORY[sel_product]

    # ── Gauge cards per factory
    cols = st.columns(len(FACTORIES))
    for i, (_, row) in enumerate(sim_df.iterrows()):
        is_cur  = row["Is Current"]
        is_best = (i == 0)
        with cols[i]:
            border_col = "#4eca7f" if is_best and not is_cur else ("#7c83f5" if is_cur else "#3a3f5c")
            tag = ""
            if is_cur:  tag = '<span class="badge-blue">CURRENT</span>'
            if is_best and not is_cur: tag = '<span class="badge-green">BEST</span>'
            improv = row["LT Improvement"]
            delta_html = (
                f'<span class="badge-green">▼ {improv:.1f}d faster</span>' if improv > 0
                else f'<span class="badge-red">▲ {-improv:.1f}d slower</span>' if improv < 0
                else '<span class="badge-blue">No change</span>'
            )
            st.markdown(f"""
            <div style="background:#1e2235;border:1.5px solid {border_col};border-radius:12px;
                        padding:16px;text-align:center;margin-bottom:8px">
              <div style="font-size:.75rem;color:#9095b0;margin-bottom:4px">{tag}</div>
              <div style="font-size:.95rem;font-weight:600;color:#c8caf0">{row['Factory']}</div>
              <div style="font-size:1.7rem;font-weight:700;color:#7c83f5;margin:6px 0">{row['Predicted LT']:.1f}d</div>
              <div style="font-size:.78rem;color:#9095b0">Pred. Lead Time</div>
              <div style="margin-top:8px">{delta_html}</div>
              <div style="font-size:.74rem;color:#9095b0;margin-top:6px">
                📍 {row['Distance km']:.0f} km &nbsp;|&nbsp; 💰 ${row['Avg Gross Profit']:.2f}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart
    fig = px.bar(
        sim_df, x="Factory", y="Predicted LT",
        color="Is Current",
        color_discrete_map={True: "#7c83f5", False: "#4eca7f"},
        text="Predicted LT",
        title=f"Predicted Lead Time by Factory · {sel_product}",
        labels={"Predicted LT": "Lead Time (days)", "Is Current": "Current Factory"},
        template="plotly_dark",
    )
    fig.update_traces(texttemplate="%{text:.1f}d", textposition="outside")
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        showlegend=True, height=380,
        font=dict(color="#c8caf0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Radar / spider chart (multi-metric)
    categories = ["Speed", "Profit", "Low Distance", "Margin", "Confidence"]
    fig2 = go.Figure()
    for _, row in sim_df.iterrows():
        lt_score    = max(0, 100 - row["Predicted LT"] * 10)
        dist_score  = max(0, 100 - row["Distance km"] / 50)
        profit_score= min(100, row["Avg Gross Profit"] * 3)
        margin_score= row["Avg Margin %"]
        conf_score  = row["Confidence"]
        vals = [lt_score, profit_score, dist_score, margin_score, conf_score]
        fig2.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=row["Factory"], opacity=0.7,
        ))
    fig2.update_layout(
        polar=dict(bgcolor="#1e2235",
                   radialaxis=dict(visible=True, range=[0,100], color="#9095b0"),
                   angularaxis=dict(color="#c8caf0")),
        paper_bgcolor="#0f1117", font=dict(color="#c8caf0"),
        title="Multi-Metric Factory Comparison", height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – What-If Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">What-If Scenario Analysis</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        wif_product  = st.selectbox("Product", sorted(PRODUCT_FACTORY.keys()), key="wif_p")
        wif_ship     = st.selectbox("Ship Mode",
                                     ["Standard Class","First Class","Second Class","Same Day"], key="wif_s")
    with col_r:
        wif_region   = st.selectbox("Region", list(REGION_COORDS.keys()), key="wif_r")
        wif_factory  = st.selectbox("Reassign to Factory", list(FACTORIES.keys()), key="wif_f")

    wif_units = st.slider("Units in Order", 1, 14, 3, key="wif_u")
    wif_cost  = st.slider("Cost per Unit ($)", 0.5, 10.0, 3.5, step=0.1, key="wif_c")

    # Current vs proposed
    current_f  = PRODUCT_FACTORY[wif_product]
    lt_current = predict_lead_time(model, encoders, wif_product, wif_region,
                                    current_f, wif_ship, wif_units, wif_cost)
    lt_new     = predict_lead_time(model, encoders, wif_product, wif_region,
                                    wif_factory, wif_ship, wif_units, wif_cost)

    f_cur = FACTORIES[current_f]; f_new = FACTORIES[wif_factory]
    r_inf = REGION_COORDS[wif_region]
    dist_cur = haversine_km(f_cur["lat"], f_cur["lon"], r_inf["lat"], r_inf["lon"])
    dist_new = haversine_km(f_new["lat"], f_new["lon"], r_inf["lat"], r_inf["lon"])

    improvement    = lt_current - lt_new
    improvement_pct= improvement / max(lt_current, 0.01) * 100

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Lead Time",  f"{lt_current:.1f} days", delta=None)
    m2.metric("Proposed Lead Time", f"{lt_new:.1f} days",
              delta=f"{-improvement:.1f} days", delta_color="inverse")
    m3.metric("Distance Change",
              f"{dist_new:.0f} km",
              delta=f"{dist_new-dist_cur:.0f} km", delta_color="inverse")
    m4.metric("LT Improvement",
              f"{improvement_pct:.1f}%",
              delta="faster" if improvement > 0 else "slower",
              delta_color="normal" if improvement > 0 else "inverse")

    # Visual comparison bar
    comp_df = pd.DataFrame({
        "Scenario": [f"Current\n({current_f})", f"Proposed\n({wif_factory})"],
        "Lead Time": [lt_current, lt_new],
        "Distance":  [dist_cur,   dist_new],
        "Scenario_Color": ["Current", "Proposed"],
    })
    fig3 = make_subplots(rows=1, cols=2,
                         subplot_titles=("Lead Time (days)", "Distance to Region (km)"))
    for idx, (col_name, color) in enumerate([("Lead Time","#7c83f5"),("Distance","#f5a742")]):
        for _, r in comp_df.iterrows():
            c = "#4eca7f" if r["Scenario_Color"] == "Proposed" else color
            fig3.add_trace(
                go.Bar(name=r["Scenario"], x=[r["Scenario"]], y=[r[col_name]],
                       marker_color=c, showlegend=idx==0),
                row=1, col=idx+1
            )
    fig3.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#c8caf0"), height=360, barmode="group",
        title_text="Current vs Proposed Scenario Comparison",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Risk assessment
    st.markdown('<div class="section-title">Risk & Impact Assessment</div>', unsafe_allow_html=True)
    risk_cols = st.columns(3)
    with risk_cols[0]:
        risk_level = "Low" if improvement > 0 and dist_new < 3000 else \
                     "Medium" if improvement >= 0 else "High"
        color_map  = {"Low":"#4eca7f","Medium":"#f5a742","High":"#e05c5c"}
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val" style="color:{color_map[risk_level]}">{risk_level}</div>
          <div class="kpi-label">Reassignment Risk</div>
        </div>""", unsafe_allow_html=True)
    with risk_cols[1]:
        # Average profit for this product
        sub  = df[df["Product Name"] == wif_product]
        avg_gp = sub["Gross Profit"].mean()
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val">${avg_gp:.2f}</div>
          <div class="kpi-label">Avg Gross Profit / Order</div>
        </div>""", unsafe_allow_html=True)
    with risk_cols[2]:
        same_factory = current_f == wif_factory
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val" style="color:{'#9095b0' if same_factory else '#4eca7f'}">
            {'No Change' if same_factory else 'Reassign'}
          </div>
          <div class="kpi-label">Recommendation Action</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Top Factory Reassignment Recommendations</div>',
                unsafe_allow_html=True)

    top_n = st.slider("Number of recommendations", 5, 20, 10, key="top_n")

    with st.spinner("⚙️ Generating optimized recommendations…"):
        recs = generate_top_recommendations(model, encoders, df, top_n=top_n)

    if recs.empty:
        st.info("No beneficial reassignments found for current configuration.")
    else:
        # Priority-weighted score
        speed_w  = opt_priority / 100
        profit_w = 1 - speed_w
        recs["Score"] = (speed_w  * recs["LT Reduction %"].rank(pct=True) * 100 +
                         profit_w * recs["Avg Profit"].rank(pct=True)     * 100).round(1)
        recs = recs.sort_values("Score", ascending=False)

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommendations Found", len(recs))
        c2.metric("Avg LT Reduction", f"{recs['LT Reduction %'].mean():.1f}%")
        c3.metric("Avg Confidence",   f"{recs['Confidence'].mean():.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Styled table
        styled = recs[[
            "Product","Region","Ship Mode","Current Factory","Recommended",
            "LT Reduction %","Avg Profit","Confidence","Score"
        ]].reset_index(drop=True)
        styled.index += 1

        def color_row(val):
            if isinstance(val, float):
                if val > 15: return "color:#4eca7f;font-weight:600"
                if val > 5:  return "color:#f5a742"
            return ""

        st.dataframe(
            styled.style.applymap(color_row, subset=["LT Reduction %"]),
            use_container_width=True, height=420,
        )

        # Bubble chart
        fig4 = px.scatter(
            recs, x="LT Reduction %", y="Avg Profit",
            size="Confidence", color="Region",
            hover_name="Product",
            hover_data={"Current Factory":True,"Recommended":True,"Ship Mode":True},
            title="Recommendation Space: Lead Time Savings vs Profit",
            labels={"LT Reduction %":"Lead Time Reduction (%)","Avg Profit":"Avg Gross Profit ($)"},
            template="plotly_dark", size_max=30,
        )
        fig4.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
            font=dict(color="#c8caf0"), height=420,
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Factory flow (sankey-style bar)
        flow = recs.groupby(["Current Factory","Recommended"]).size().reset_index(name="Count")
        fig5 = px.bar(flow, x="Current Factory", y="Count", color="Recommended",
                      title="Recommended Factory Reassignment Flows",
                      template="plotly_dark", barmode="stack")
        fig5.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                           font=dict(color="#c8caf0"), height=340)
        st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – EDA & Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    eda_row1_l, eda_row1_r = st.columns(2)

    with eda_row1_l:
        lt_product = df.groupby("Product Name")["Lead Time"].mean().reset_index().sort_values("Lead Time")
        fig_lt = px.bar(lt_product, x="Lead Time", y="Product Name", orientation="h",
                        title="Avg Lead Time by Product",
                        color="Lead Time", color_continuous_scale="blues",
                        template="plotly_dark")
        fig_lt.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                              coloraxis_showscale=False, font=dict(color="#c8caf0"), height=400)
        st.plotly_chart(fig_lt, use_container_width=True)

    with eda_row1_r:
        margin_prod = df.groupby("Product Name")["Profit_Margin"].mean().reset_index().sort_values("Profit_Margin", ascending=False)
        fig_mg = px.bar(margin_prod, x="Product Name", y="Profit_Margin",
                        title="Avg Profit Margin by Product (%)",
                        color="Profit_Margin", color_continuous_scale="greens",
                        template="plotly_dark")
        fig_mg.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                              coloraxis_showscale=False, font=dict(color="#c8caf0"),
                              xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_mg, use_container_width=True)

    eda_row2_l, eda_row2_r = st.columns(2)

    with eda_row2_l:
        lt_region = df.groupby(["Region","Ship Mode"])["Lead Time"].mean().reset_index()
        fig_reg = px.bar(lt_region, x="Region", y="Lead Time", color="Ship Mode",
                         barmode="group", title="Lead Time by Region & Ship Mode",
                         template="plotly_dark")
        fig_reg.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                               font=dict(color="#c8caf0"), height=380)
        st.plotly_chart(fig_reg, use_container_width=True)

    with eda_row2_r:
        div_profit = df.groupby("Division")[["Gross Profit","Sales","Cost"]].sum().reset_index()
        fig_div = px.pie(div_profit, names="Division", values="Gross Profit",
                         title="Gross Profit Split by Division",
                         template="plotly_dark", hole=0.4,
                         color_discrete_sequence=["#7c83f5","#4eca7f","#f5a742"])
        fig_div.update_layout(paper_bgcolor="#0f1117", font=dict(color="#c8caf0"), height=380)
        st.plotly_chart(fig_div, use_container_width=True)

    # Monthly trend
    monthly = df.groupby("Order_Month").agg(
        Gross_Profit=("Gross Profit","sum"),
        Orders=("Row ID","count"),
        Avg_Lead_Time=("Lead Time","mean"),
    ).reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["Month"] = monthly["Order_Month"].map(month_names)

    fig_trend = make_subplots(specs=[[{"secondary_y":True}]])
    fig_trend.add_trace(go.Bar(x=monthly["Month"], y=monthly["Gross_Profit"],
                                name="Gross Profit ($)", marker_color="#7c83f5"), secondary_y=False)
    fig_trend.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Avg_Lead_Time"],
                                    name="Avg Lead Time (days)", line=dict(color="#4eca7f",width=2),
                                    mode="lines+markers"), secondary_y=True)
    fig_trend.update_layout(
        title="Monthly Gross Profit vs Average Lead Time",
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
        font=dict(color="#c8caf0"), height=380,
        legend=dict(bgcolor="#1e2235"),
    )
    fig_trend.update_yaxes(title_text="Gross Profit ($)", secondary_y=False)
    fig_trend.update_yaxes(title_text="Avg Lead Time (days)", secondary_y=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Distance vs Lead Time scatter
    fig_scat = px.scatter(
        df.sample(min(2000, len(df)), random_state=42),
        x="Distance_km", y="Lead Time", color="Ship Mode",
        hover_data=["Product Name","Region","Factory"],
        title="Distance vs Lead Time (sample of 2,000 orders)",
        template="plotly_dark", opacity=0.6,
    )
    fig_scat.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                            font=dict(color="#c8caf0"), height=380)
    st.plotly_chart(fig_scat, use_container_width=True)

    # Heatmap: Factory × Region avg lead time
    heat_data = df.groupby(["Factory","Region"])["Lead Time"].mean().unstack()
    fig_heat = px.imshow(
        heat_data, text_auto=".1f",
        title="Avg Lead Time Heatmap: Factory × Region",
        color_continuous_scale="RdYlGn_r",
        template="plotly_dark",
        aspect="auto",
    )
    fig_heat.update_layout(paper_bgcolor="#0f1117", font=dict(color="#c8caf0"), height=350)
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)

    metrics_df = pd.DataFrame(all_results).T.reset_index().rename(columns={"index":"Model"})
    metrics_df["Best"] = metrics_df["Model"] == best_name

    c1, c2, c3 = st.columns(3)
    for col, metric, better in [(c1,"RMSE","lower"),(c2,"MAE","lower"),(c3,"R2","higher")]:
        fig_m = px.bar(
            metrics_df, x="Model", y=metric,
            color="Best", color_discrete_map={True:"#4eca7f",False:"#7c83f5"},
            title=f"{metric} ({better} is better)", template="plotly_dark",
            text=metric,
        )
        fig_m.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_m.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                             font=dict(color="#c8caf0"), showlegend=False, height=320)
        col.plotly_chart(fig_m, use_container_width=True)

    # Feature importance (RF)
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    rf_model = all_models.get("Random Forest")
    if rf_model and hasattr(rf_model, "feature_importances_"):
        feat_imp = pd.DataFrame({
            "Feature":    encoders["feature_cols"],
            "Importance": rf_model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                        title="Random Forest Feature Importance",
                        color="Importance", color_continuous_scale="viridis",
                        template="plotly_dark")
        fig_fi.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                              coloraxis_showscale=False, font=dict(color="#c8caf0"), height=380)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Residual / predicted distribution
    st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)
    from data_prep import build_feature_matrix
    X_all, y_all, _ = build_feature_matrix(df)
    y_pred = rf_model.predict(X_all) if rf_model else np.zeros(len(df))
    residuals = y_all - y_pred

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        fig_res = px.histogram(residuals, nbins=60,
                               title="Residual Distribution (Actual − Predicted)",
                               template="plotly_dark", color_discrete_sequence=["#7c83f5"])
        fig_res.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                               font=dict(color="#c8caf0"), height=340,
                               xaxis_title="Residual (days)", yaxis_title="Count")
        st.plotly_chart(fig_res, use_container_width=True)

    with res_col2:
        fig_act = px.scatter(x=y_pred, y=y_all, opacity=0.3,
                             title="Actual vs Predicted Lead Time",
                             labels={"x":"Predicted (days)","y":"Actual (days)"},
                             template="plotly_dark", color_discrete_sequence=["#4eca7f"])
        lo, hi = float(min(y_all.min(), y_pred.min())), float(max(y_all.max(), y_pred.max()))
        fig_act.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi],
                                      mode="lines", line=dict(color="#e05c5c",dash="dash"),
                                      name="Perfect prediction"))
        fig_act.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235",
                               font=dict(color="#c8caf0"), height=340)
        st.plotly_chart(fig_act, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-title">Full Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(metrics_df[["Model","RMSE","MAE","R2","Best"]].set_index("Model"),
                 use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#4a4f6a;font-size:.78rem'>"
    "Nassau Candy Distributor · Factory Reallocation System · "
    "Powered by Random Forest ML · Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True,
)