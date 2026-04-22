import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from translations import TRANSLATIONS
except:
    TRANSLATIONS = {"English": {
        "title":"🌍 Global Temperature Intelligence System",
        "subtitle":"146 Years REAL NASA Data · 20 Countries · ML Risk · Predictions · 8 Languages",
        "tab1":"🌡️ 146 Years","tab2":"🌐 Countries","tab3":"🔮 Predictions",
        "tab4":"🤖 ML Model","tab5":"🌱 Solutions","tab6":"📊 AI Insights",
        "select_year":"Select Year:","select_country":"Choose Country:",
        "predict_btn":"🔍 Predict Risk","generate_btn":"📊 Generate AI Insights",
        "hottest_decade":"2020s are HOTTEST decade in 146 years!",
        "paris_warning":"Paris 1.5°C limit breach around",
        "critical":"🔴 CRITICAL","at_risk":"🟡 AT RISK","stable":"🟢 STABLE",
        "coldest":"Coldest Year","hottest":"Hottest Year","total_rise":"Total Rise",
        "select_lang":"🌐 Language",
        "footer":"Real Data: NASA GISS · Our World in Data · IEA | Python · Streamlit · ML",
        "verdict_critical":"🚨 VERDICT: CRITICAL — Emergency action required NOW!",
        "verdict_risk":"⚠️ VERDICT: AT RISK — Urgent policy changes needed!",
        "verdict_stable":"✅ VERDICT: STABLE — Climate leader. Help others transition!",
        "metric_temp":"🌡️ Temp 2024 NASA Real",
        "metric_co2":"🏭 CO2 2024 NOAA",
        "metric_pred":"🔮 Predicted 2050",
        "metric_worst":"🇸🇦 Worst CO2 Saudi Arabia",
        "metric_best":"🇳🇴 Best Norway Renewable",
        "select_pred_country":"Country for Prediction:",
        "ranking_2030":"2030 Country Risk Rankings",
        "proven":"PROVEN: High renewable countries emit",
        "less_co2":"LESS CO2 than fossil countries!",
        "milestone":"Key Temperature Milestones",
        "temp_forecast":"Global Temperature Forecast 2025–2060",
    }}

st.set_page_config(
    page_title="🌍 Global Temperature Intelligence System",
    page_icon="🌍", layout="wide"
)

st.markdown("""<style>
.main-title{font-size:2rem;font-weight:700;color:#E74C3C;text-align:center;padding:.8rem 0}
.subtitle{font-size:.95rem;color:#7F8C8D;text-align:center;margin-top:-8px;margin-bottom:16px}
.mc{background:#1a1a2e;border-radius:10px;padding:16px;text-align:center;border:1px solid #E74C3C}
.mv{font-size:1.8rem;font-weight:700;color:#E74C3C}
.ml{font-size:.8rem;color:#BDC3C7;margin-top:3px}
.ib{background:#0f3460;border-radius:8px;padding:16px;border-left:4px solid #27AE60;color:white;margin:8px 0}
.rb{background:#1a0a0a;border-radius:8px;padding:16px;border-left:4px solid #E74C3C;color:white;margin:8px 0}
.wb{background:#1a1a00;border-radius:8px;padding:16px;border-left:4px solid #F39C12;color:white;margin:8px 0}
.insight-section{background:#0d1117;border-radius:10px;padding:20px;margin:10px 0;border:1px solid #30363d}
.insight-title{font-size:1.1rem;font-weight:600;color:#58a6ff;margin-bottom:12px}
.insight-item{background:#161b22;border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9;font-size:0.9rem;border-left:3px solid #238636}
.warn-item{background:#161b22;border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9;font-size:0.9rem;border-left:3px solid #f85149}
.action-item{background:#161b22;border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9;font-size:0.9rem;border-left:3px solid #d29922}
</style>""", unsafe_allow_html=True)

# ── Language Selector ─────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    lang = st.selectbox("🌐 Language / भाषा / भाषा / Idioma",
                        list(TRANSLATIONS.keys()), index=0)
    st.markdown("---")
    st.markdown("**📊 Data Sources:**")
    st.markdown("- 🌡️ NASA GISS (1880–2025)")
    st.markdown("- 🌍 Our World in Data (OWID)")
    st.markdown("- ⚡ IEA Renewable Energy")
    st.markdown("---")
    st.markdown("**🛠️ Tech Stack:**")
    st.markdown("Python · Pandas · XGBoost")
    st.markdown("SHAP · FastAPI · Streamlit")
    st.markdown("Plotly · Scikit-Learn")

T = TRANSLATIONS[lang]

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(BASE, 'data')
    df_g = pd.read_csv(os.path.join(d, 'global_temperature_REAL.csv'))
    df_c = pd.read_csv(os.path.join(d, 'country_climate_REAL.csv'))
    df_c['Fossil_Dependency']   = 100 - df_c['Renewable_Energy_pct'].fillna(50)
    df_c['Carbon_Efficiency']   = df_c['CO2_per_capita_tonnes'].fillna(0) / (df_c['Fossil_Dependency'] + 1)
    df_c['Vulnerability_Index'] = (
        df_c['Temp_Anomaly_C'].fillna(0) * 0.3 +
        df_c['Extreme_Weather_Events'].fillna(10) * 0.1 +
        (100 - df_c['Forest_Cover_pct'].fillna(25)) * 0.01 +
        df_c['Fossil_Dependency'] * 0.01
    )
    return df_g, df_c

@st.cache_data
def build_models(df_g):
    X = df_g['Year'].values.reshape(-1, 1)
    y = df_g['Temp_Anomaly_C'].values
    poly = PolynomialFeatures(degree=2)
    Xp = poly.fit_transform(X)
    m = LinearRegression()
    m.fit(Xp, y)
    return m, poly

@st.cache_data
def country_preds(df_c):
    res = []
    for c in df_c['Country'].unique():
        dfc = df_c[df_c['Country'] == c].copy()
        if len(dfc) < 5:
            continue
        X = dfc['Year'].values.reshape(-1, 1)
        cm = LinearRegression().fit(X, dfc['CO2_per_capita_tonnes'].ffill().values)
        rm = LinearRegression().fit(X, dfc['Renewable_Energy_pct'].ffill().values)
        for yr in range(2025, 2041):
            co2p = max(0, float(cm.predict([[yr]])[0]))
            renp = min(100, max(0, float(rm.predict([[yr]])[0])))
            res.append({
                'Country': c, 'Year': yr,
                'Predicted_CO2': round(co2p, 2),
                'Predicted_Renewable': round(renp, 1),
                'Predicted_Risk': round(min(100, max(0, co2p*3.5 + (100-renp)*0.3 - renp*0.25)), 1)
            })
    return pd.DataFrame(res)

# ── Smart AI Insights Generator (No External API) ─────────────
def generate_smart_insights(country, row, pred_2030=None):
    """
    Generates data-driven insights using real climate data.
    Rule-based intelligence — no external API needed.
    """
    co2    = row['CO2_per_capita_tonnes']
    ren    = row['Renewable_Energy_pct']
    temp   = row['Temp_Anomaly_C']
    risk   = row['Climate_Risk_Score']
    forest = row['Forest_Cover_pct']
    fossil = row['Fossil_Dependency']
    events = row['Extreme_Weather_Events']
    yr     = int(row['Year'])

    # ── CURRENT SITUATION ─────────────────────────────────────
    situation = []

    if risk >= 55:
        situation.append(f"🔴 {country} is in CRITICAL climate risk category with a risk score of {risk:.1f}/100 — one of the highest globally.")
    elif risk >= 28:
        situation.append(f"🟡 {country} is classified as AT RISK with a climate score of {risk:.1f}/100, indicating urgent action is needed.")
    else:
        situation.append(f"🟢 {country} is in the STABLE category with a risk score of {risk:.1f}/100 — among the world's climate leaders.")

    situation.append(f"📈 Temperature anomaly stands at +{temp:.2f}°C above the 20th century baseline (NASA GISS {yr} data).")

    if co2 > 15:
        situation.append(f"🏭 CO2 emissions at {co2:.1f} tonnes per person per year — critically high, 7x the global average of 4.7t.")
    elif co2 > 8:
        situation.append(f"🏭 CO2 emissions at {co2:.1f} tonnes per person per year — above global average of 4.7t, needs reduction.")
    elif co2 > 3:
        situation.append(f"🏭 CO2 emissions at {co2:.1f} tonnes per person per year — near global average, manageable with policy action.")
    else:
        situation.append(f"🏭 CO2 emissions at only {co2:.1f} tonnes per person — well below global average. Responsible emitter.")

    situation.append(f"⛈️ {events} extreme weather events per year — directly linked to rising temperatures.")

    # ── ROOT CAUSES ───────────────────────────────────────────
    causes = []

    if fossil > 80:
        causes.append(f"🔥 CRITICAL: {fossil:.0f}% fossil fuel dependency — the primary driver of high emissions. Near-total reliance on coal, oil, and gas.")
    elif fossil > 60:
        causes.append(f"⚠️ HIGH: {fossil:.0f}% fossil fuel dependency — significant structural dependence on non-renewable energy sources.")
    elif fossil > 40:
        causes.append(f"📊 MODERATE: {fossil:.0f}% fossil fuel dependency — transition underway but still majority fossil-based energy system.")
    else:
        causes.append(f"✅ LOW: Only {fossil:.0f}% fossil dependency — strong renewable energy infrastructure in place.")

    if ren < 10:
        causes.append(f"⚡ Renewable energy at just {ren:.1f}% — virtually no green energy transition has begun.")
    elif ren < 25:
        causes.append(f"⚡ Renewable energy at {ren:.1f}% — early stage transition, far from climate targets.")
    elif ren < 50:
        causes.append(f"⚡ Renewable energy at {ren:.1f}% — mid-transition phase, meaningful progress being made.")
    elif ren < 80:
        causes.append(f"⚡ Renewable energy at {ren:.1f}% — advanced transition, approaching climate leadership territory.")
    else:
        causes.append(f"⚡ Renewable energy at {ren:.1f}% — world-class green energy system. Climate leader.")

    if forest < 10:
        causes.append(f"🌲 Forest cover at only {forest:.1f}% — critically low carbon absorption capacity, accelerating warming.")
    elif forest < 25:
        causes.append(f"🌲 Forest cover at {forest:.1f}% — below optimal levels for carbon sink effectiveness.")
    elif forest < 50:
        causes.append(f"🌲 Forest cover at {forest:.1f}% — moderate carbon sink, important to protect and expand.")
    else:
        causes.append(f"🌲 Forest cover at {forest:.1f}% — excellent carbon sink, a major natural climate asset.")

    # ── URGENT ACTIONS ────────────────────────────────────────
    actions = []

    target_ren = min(100, round(ren + 30))
    if ren < 30:
        actions.append(f"🚨 ACTION 1: Emergency renewable energy mandate — target {target_ren}% by 2030. Invest minimum 5% of GDP in solar, wind, and hydro infrastructure immediately.")
    elif ren < 60:
        actions.append(f"⚡ ACTION 1: Accelerate renewable transition — target {target_ren}% by 2030. Current trajectory needs to double in speed.")
    else:
        actions.append(f"✅ ACTION 1: Maintain renewable leadership at {ren:.0f}%. Export green technology and financing to developing nations.")

    if co2 > 10:
        actions.append(f"🏭 ACTION 2: Implement carbon pricing at $50/tonne effective 2025, rising to $150/tonne by 2030. Use revenue to fund community energy transition programs and retrain fossil fuel workers.")
    elif co2 > 4:
        actions.append(f"📊 ACTION 2: Strengthen carbon pricing mechanisms. Set sector-specific emission reduction targets — industry, transport, and agriculture each need dedicated roadmaps.")
    else:
        actions.append(f"🌍 ACTION 2: Share climate financing with developing nations. {country}'s low emissions position it as a credible climate leader in international negotiations.")

    if forest < 20:
        actions.append(f"🌲 ACTION 3: National reforestation emergency program — plant 3 million hectares annually for 10 years. Halt all deforestation with immediate legislation and enforcement.")
    elif forest < 40:
        actions.append(f"🌲 ACTION 3: Forest protection legislation — ban primary forest clearing, incentivize private landowner conservation, target adding 1 million hectares of new forest cover per decade.")
    else:
        actions.append(f"🌳 ACTION 3: Expand forest corridors for biodiversity. Fund community-based forest management. {country}'s forest assets are a global climate service worth protecting.")

    # ── 2030 PREDICTION INSIGHT ───────────────────────────────
    pred_insight = None
    if pred_2030 is not None:
        r30 = pred_2030['Predicted_Risk']
        c30 = pred_2030['Predicted_CO2']
        ren30 = pred_2030['Predicted_Renewable']
        if r30 >= 55:
            pred_insight = f"⚠️ Without policy change, {country} is projected to remain CRITICAL by 2030 (Risk: {r30:.1f}/100, CO2: {c30:.2f}t, Renewable: {ren30:.1f}%)"
        elif r30 >= 28:
            pred_insight = f"📊 By 2030, {country} is projected AT RISK (Risk: {r30:.1f}/100, CO2: {c30:.2f}t, Renewable: {ren30:.1f}%) — urgent policy acceleration needed."
        else:
            pred_insight = f"✅ By 2030, {country} is projected STABLE (Risk: {r30:.1f}/100, CO2: {c30:.2f}t, Renewable: {ren30:.1f}%) — current trajectory is sustainable."

    return situation, causes, actions, pred_insight

df_global, df_country = load_data()
tm, pf = build_models(df_global)
df_cpred = country_preds(df_country)

# ── Header ────────────────────────────────────────────────────
st.markdown(f'<div class="main-title">{T["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{T["subtitle"]}</div>', unsafe_allow_html=True)
st.markdown("---")

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl in [
    (c1, "+1.28°C", T["metric_temp"]),
    (c2, "422.9 ppm", T["metric_co2"]),
    (c3, "+1.63°C", T["metric_pred"]),
    (c4, "20.4 t", T["metric_worst"]),
    (c5, "99%", T["metric_best"])
]:
    with col:
        st.markdown(f'<div class="mc"><div class="mv">{val}</div><div class="ml">{lbl}</div></div>',
                    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tabs = st.tabs([T["tab1"], T["tab2"], T["tab3"], T["tab4"], T["tab5"], T["tab6"]])

# ─── TAB 1 — 146 Years ────────────────────────────────────────
with tabs[0]:
    st.subheader("🌡️ Real NASA GISS Temperature Data — 1880 to 2025")

    clrs = ['#E74C3C' if v > 0 else '#3498DB' for v in df_global['Temp_Anomaly_C']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_global['Year'], y=df_global['Temp_Anomaly_C'],
        marker_color=clrs, name='Anomaly',
        hovertemplate='Year: %{x}<br>Anomaly: %{y}°C<extra></extra>'
    ))
    fig.add_hline(y=0, line_color='white', line_width=1)
    fig.add_hline(y=1.5, line_dash='dash', line_color='orange', annotation_text='Paris 1.5°C Limit')
    fig.add_hline(y=1.28, line_dash='dot', line_color='red', annotation_text='2024: +1.28°C (NASA)')
    fig.update_layout(
        title='🌡️ 146 Years Real NASA GISS Temperature Anomaly (1880–2025)',
        xaxis_title='Year', yaxis_title='Temperature Anomaly (°C)',
        template='plotly_dark', height=480
    )
    st.plotly_chart(fig, use_container_width=True)

    a, b, c, d = st.columns(4)
    a.metric(T["coldest"], str(int(df_global.loc[df_global['Temp_Anomaly_C'].idxmin(), 'Year'])),
             f"{df_global['Temp_Anomaly_C'].min():.2f}°C")
    b.metric(T["hottest"], str(int(df_global.loc[df_global['Temp_Anomaly_C'].idxmax(), 'Year'])),
             f"{df_global['Temp_Anomaly_C'].max():.2f}°C")
    c.metric("2024 (NASA)", "+1.28°C", "Record High")
    d.metric(T["total_rise"],
             f"{df_global['Temp_Anomaly_C'].max()-df_global['Temp_Anomaly_C'].min():.2f}°C", "Since 1880")

    st.markdown("---")
    df_global['Decade'] = (df_global['Year'] // 10) * 10
    dec = df_global.groupby('Decade')['Temp_Anomaly_C'].mean().reset_index()
    fig2 = px.bar(dec, x='Decade', y='Temp_Anomaly_C',
                  color='Temp_Anomaly_C', color_continuous_scale='RdYlGn_r',
                  title='Average Temperature Anomaly by Decade',
                  template='plotly_dark', text=dec['Temp_Anomaly_C'].round(2))
    fig2.update_traces(textposition='outside')
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)
    st.success(f"📊 {T['hottest_decade']}")

    st.markdown("---")
    corr_c = ['CO2_per_capita_tonnes', 'Renewable_Energy_pct', 'Temp_Anomaly_C',
              'Extreme_Weather_Events', 'Fossil_Dependency', 'Climate_Risk_Score']
    corr = df_country[corr_c].corr()
    fig3, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title('Correlation Matrix — Real OWID Data', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ─── TAB 2 — Country Analysis ─────────────────────────────────
with tabs[1]:
    st.subheader("🌐 Country-Level Climate Intelligence — Real OWID Data")

    my = int(df_country['Year'].max())
    ys = st.slider(T["select_year"], 2000, my, my)
    dy = df_country[df_country['Year'] == ys].copy()

    ca, cb = st.columns(2)
    with ca:
        fig_c = px.bar(dy.sort_values('CO2_per_capita_tonnes', ascending=False),
                       x='Country', y='CO2_per_capita_tonnes',
                       color='CO2_per_capita_tonnes', color_continuous_scale='RdYlGn_r',
                       template='plotly_dark', title=f'CO2 per Capita — {ys}')
        fig_c.update_layout(height=370, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_c, use_container_width=True)
    with cb:
        fig_r = px.bar(dy.sort_values('Renewable_Energy_pct', ascending=False),
                       x='Country', y='Renewable_Energy_pct',
                       color='Renewable_Energy_pct', color_continuous_scale='RdYlGn',
                       template='plotly_dark', title=f'Renewable Energy — {ys}')
        fig_r.update_layout(height=370, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    ct = st.selectbox(T["select_country"], sorted(df_country['Country'].unique()))
    dct = df_country[df_country['Country'] == ct]
    ft = go.Figure()
    ft.add_trace(go.Scatter(x=dct['Year'], y=dct['Temp_Anomaly_C'],
                            name='Temp (°C)', line=dict(color='red', width=2.5)))
    ft.add_trace(go.Scatter(x=dct['Year'], y=dct['Renewable_Energy_pct'],
                            name='Renewable %', line=dict(color='green', width=2.5), yaxis='y2'))
    ft.add_trace(go.Scatter(x=dct['Year'], y=dct['CO2_per_capita_tonnes'],
                            name='CO2/capita', line=dict(color='orange', width=2.5), yaxis='y3'))
    ft.update_layout(
        title=f'{ct} — Climate Trends 2000–{my}',
        template='plotly_dark', height=420,
        yaxis=dict(title='Temp (°C)', title_font=dict(color='red')),
        yaxis2=dict(title='Renewable %', title_font=dict(color='green'),
                    overlaying='y', side='right'),
        yaxis3=dict(title='CO2/capita', title_font=dict(color='orange'),
                    overlaying='y', side='right', position=0.93),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(ft, use_container_width=True)

    lr = dct[dct['Year'] == my].iloc[0]
    p, q, r, s = st.columns(4)
    p.metric("🌡️", f"+{lr['Temp_Anomaly_C']:.2f}°C")
    q.metric("🏭", f"{lr['CO2_per_capita_tonnes']:.2f}t")
    r.metric("⚡", f"{lr['Renewable_Energy_pct']:.1f}%")
    s.metric("⚠️ Risk", f"{lr['Climate_Risk_Score']:.1f}/100")

# ─── TAB 3 — Future Predictions ───────────────────────────────
with tabs[2]:
    st.subheader(f"🔮 {T['temp_forecast']}")
    st.info("Polynomial Regression trained on 146 years of real NASA GISS data")

    hy = df_global['Year'].values
    ht = df_global['Temp_Anomaly_C'].values
    fy = np.arange(2025, 2061)
    fp = tm.predict(pf.transform(fy.reshape(-1, 1)))

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=hy, y=ht, name='Historical (NASA Real)',
                               line=dict(color='#3498DB', width=2)))
    fig_p.add_trace(go.Scatter(x=fy, y=fp, name='Predicted (ML Model)',
                               line=dict(color='#E74C3C', width=2.5, dash='dot')))
    fig_p.add_trace(go.Scatter(
        x=np.concatenate([fy, fy[::-1]]),
        y=np.concatenate([fp+0.15, (fp-0.15)[::-1]]),
        fill='toself', fillcolor='rgba(231,76,60,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='Uncertainty Range'
    ))
    fig_p.add_hline(y=1.5, line_dash='dash', line_color='orange', annotation_text='Paris 1.5°C Limit')
    fig_p.add_hline(y=2.0, line_dash='dash', line_color='red', annotation_text='Danger Zone: 2.0°C')
    fig_p.update_layout(
        title='Global Temperature Prediction 2025–2060 (Polynomial Regression on NASA Data)',
        xaxis_title='Year', yaxis_title='Temperature Anomaly (°C)',
        template='plotly_dark', height=500
    )
    st.plotly_chart(fig_p, use_container_width=True)

    ms = []
    for yr in [2025, 2027, 2030, 2033, 2035, 2040, 2045, 2050, 2055, 2060]:
        p2 = fp[yr-2025]
        ms.append({
            'Year': yr,
            'Predicted (°C)': f"+{p2:.3f}",
            'Range': f"+{p2-0.15:.2f} to +{p2+0.15:.2f}",
            'Status': ("🔴 CRITICAL" if p2 >= 2.0 else "🟠 DANGER" if p2 >= 1.5
                       else "🟡 WARNING" if p2 >= 1.2 else "🟢 SAFE")
        })
    st.markdown(f"### 📊 {T['milestone']}")
    st.dataframe(pd.DataFrame(ms), use_container_width=True, hide_index=True)

    for yr, pp in zip(fy, fp):
        if pp >= 1.5:
            st.error(f"⚠️ {T['paris_warning']} **{yr}** — predicted {pp:.2f}°C if current trend continues!")
            break

    st.markdown("---")
    pc = st.selectbox(T.get("select_pred_country", "Country for Prediction:"),
                      sorted(df_country['Country'].unique()), key='pc')
    dh = df_country[df_country['Country'] == pc][['Year', 'CO2_per_capita_tonnes', 'Renewable_Energy_pct']]
    df_ = df_cpred[df_cpred['Country'] == pc]

    col1, col2 = st.columns(2)
    with col1:
        fc = go.Figure()
        fc.add_trace(go.Scatter(x=dh['Year'], y=dh['CO2_per_capita_tonnes'],
                               name='Historical (OWID)', line=dict(color='#3498DB', width=2)))
        fc.add_trace(go.Scatter(x=df_['Year'], y=df_['Predicted_CO2'],
                               name='Predicted', line=dict(color='#E74C3C', width=2, dash='dot')))
        fc.update_layout(template='plotly_dark', height=320,
                         title=f'{pc} CO2 Forecast', yaxis_title='CO2/capita (t)')
        st.plotly_chart(fc, use_container_width=True)

    with col2:
        fr = go.Figure()
        fr.add_trace(go.Scatter(x=dh['Year'], y=dh['Renewable_Energy_pct'],
                               name='Historical (OWID)', line=dict(color='#27AE60', width=2)))
        fr.add_trace(go.Scatter(x=df_['Year'], y=df_['Predicted_Renewable'],
                               name='Predicted', line=dict(color='#F39C12', width=2, dash='dot')))
        fr.add_hline(y=50, line_dash='dash', line_color='white', annotation_text='50% Target')
        fr.update_layout(template='plotly_dark', height=320,
                         title=f'{pc} Renewable Forecast', yaxis_title='Renewable (%)')
        st.plotly_chart(fr, use_container_width=True)

    st.markdown(f"### 🏆 {T['ranking_2030']}")
    r30all = df_cpred[df_cpred['Year'] == 2030].sort_values('Predicted_Risk', ascending=False).copy()
    r30all['Status'] = r30all['Predicted_Risk'].apply(
        lambda x: "🔴 CRITICAL" if x >= 55 else "🟡 AT RISK" if x >= 28 else "🟢 STABLE")
    st.dataframe(
        r30all[['Country', 'Predicted_CO2', 'Predicted_Renewable', 'Predicted_Risk', 'Status']].reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

# ─── TAB 4 — ML Risk Model ────────────────────────────────────
with tabs[3]:
    st.subheader("🤖 ML Climate Risk Classifier — Real-Time Prediction")
    st.info("Adjust sliders → Click Predict → Get instant climate risk + compare with real countries!")

    x1, x2, x3 = st.columns(3)
    with x1:
        ci = st.slider("CO2 per capita (tonnes)", 0.1, 22.0, 5.0, 0.1)
        ri = st.slider("Renewable Energy (%)", 0, 100, 30)
        ti = st.slider("Temperature Anomaly (°C)", 0.0, 2.0, 0.9, 0.05)
    with x2:
        ei = st.slider("Extreme Weather Events/yr", 0, 60, 20)
        fi = st.slider("Forest Cover (%)", 1, 90, 30)
    with x3:
        fd = 100 - ri
        ce = ci / (fd + 1)
        vi = ti*0.3 + ei*0.1 + (100-fi)*0.01 + fd*0.01
        st.metric("🏭 Fossil Dependency", f"{fd:.0f}%")
        st.metric("⚗️ Carbon Efficiency", f"{ce:.3f}")
        st.metric("⚠️ Vulnerability Index", f"{vi:.2f}")

    if st.button(T["predict_btn"], type="primary", use_container_width=True):
        rs = min(100, max(0, ci*3.5 + fd*0.3 + ti*15 + ei*0.4 - ri*0.25 - fi*0.1))

        if rs >= 55:
            st.error(f"🔴 CRITICAL RISK — Score: {rs:.1f}/100\n\n🚨 Emergency renewable investment | 🌲 Stop deforestation | 🏭 Carbon tax NOW")
        elif rs >= 28:
            st.warning(f"🟡 AT RISK — Score: {rs:.1f}/100\n\n⚡ Increase renewable to 60% | 🚗 EV incentives | 📊 Annual reporting")
        else:
            st.success(f"🟢 STABLE — Score: {rs:.1f}/100\n\n✅ Climate leader! Share technology with developing nations")

        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rs,
            title={'text': "Climate Risk Score / 100"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if rs >= 55 else "orange" if rs >= 28 else "green"},
                'steps': [
                    {'range': [0, 28], 'color': '#27AE60'},
                    {'range': [28, 55], 'color': '#F39C12'},
                    {'range': [55, 100], 'color': '#E74C3C'}
                ]
            }
        ))
        fg.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fg, use_container_width=True)

        lat = df_country[df_country['Year'] == df_country['Year'].max()]
        cm2 = lat[['Country', 'CO2_per_capita_tonnes', 'Renewable_Energy_pct', 'Climate_Risk_Score']].copy()
        my2 = pd.DataFrame([{
            'Country': '⭐ Your Input',
            'CO2_per_capita_tonnes': ci,
            'Renewable_Energy_pct': ri,
            'Climate_Risk_Score': rs
        }])
        cm2 = pd.concat([cm2, my2]).sort_values('Climate_Risk_Score', ascending=False)
        fc2 = px.bar(cm2, x='Country', y='Climate_Risk_Score',
                     color='Climate_Risk_Score', color_continuous_scale='RdYlGn_r',
                     template='plotly_dark', title='Your Input vs 20 Real Countries')
        fc2.update_layout(height=370, xaxis_tickangle=-45)
        st.plotly_chart(fc2, use_container_width=True)

# ─── TAB 5 — Solutions ────────────────────────────────────────
with tabs[4]:
    st.subheader(f"🌱 {T['proven']} {T['less_co2']}")

    hc = df_country[df_country['Renewable_Energy_pct'] > 50]['Country'].unique()
    lc = df_country[df_country['Renewable_Energy_pct'] < 20]['Country'].unique()
    ht2 = df_country[df_country['Country'].isin(hc)].groupby('Year')['CO2_per_capita_tonnes'].mean()
    lt2 = df_country[df_country['Country'].isin(lc)].groupby('Year')['CO2_per_capita_tonnes'].mean()

    fs = go.Figure()
    fs.add_trace(go.Scatter(x=lt2.index, y=lt2.values,
                            name='🔴 Low Renewable (<20%)', line=dict(color='red', width=3)))
    fs.add_trace(go.Scatter(x=ht2.index, y=ht2.values,
                            name='🟢 High Renewable (>50%)', line=dict(color='green', width=3),
                            fill='tonexty', fillcolor='rgba(39,174,96,0.1)'))
    fs.update_layout(
        title='CO2 Emissions: High Renewable vs Low Renewable Countries (Real OWID Data)',
        template='plotly_dark', height=420, yaxis_title='Avg CO2 per Capita (tonnes)'
    )
    st.plotly_chart(fs, use_container_width=True)

    latest = df_country[df_country['Year'] == df_country['Year'].max()]
    ha = latest[latest['Country'].isin(hc)]['CO2_per_capita_tonnes'].mean()
    la = latest[latest['Country'].isin(lc)]['CO2_per_capita_tonnes'].mean()
    st.success(f"✅ Real Data Proof: High renewable countries emit **{((la-ha)/la*100):.0f}% LESS CO2** than fossil-dependent countries!")

    n1, n2, n3 = st.columns(3)
    with n1: st.success("**🇳🇴 Norway**\n\n99% Renewable\nCO2: 6.7t\n🟢 World Leader")
    with n2: st.warning("**🇮🇳 India**\n\n43.2% Renewable (2024)\nCO2: 2.2t\n🟡 Great Progress!")
    with n3: st.error("**🇸🇦 Saudi Arabia**\n\n4.5% Renewable\nCO2: 20.4t\n🔴 Highest Risk")

    kc = ['India', 'China', 'USA', 'Germany', 'Norway', 'Saudi Arabia', 'Brazil']
    fig_rt = px.line(df_country[df_country['Country'].isin(kc)],
                     x='Year', y='Renewable_Energy_pct', color='Country',
                     title='Renewable Energy Adoption 2000–2024 (Real OWID Data)',
                     template='plotly_dark', markers=True)
    fig_rt.update_layout(height=400)
    st.plotly_chart(fig_rt, use_container_width=True)

# ─── TAB 6 — AI Insights (No External API) ────────────────────
with tabs[5]:
    st.subheader("📊 AI Climate Insights Generator")
    st.info("Select any country → Click Generate → Get data-driven climate analysis powered by ML logic!")

    ai_ctry = st.selectbox("Select Country for Analysis:",
                           sorted(df_country['Country'].unique()), key='ai_s')
    max_yr = int(df_country['Year'].max())
    row = df_country[(df_country['Country'] == ai_ctry) & (df_country['Year'] == max_yr)].iloc[0]
    fut_row = df_cpred[(df_cpred['Country'] == ai_ctry) & (df_cpred['Year'] == 2030)]

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("🌡️ Temp Anomaly",  f"+{row['Temp_Anomaly_C']:.2f}°C")
    g2.metric("🏭 CO2/capita",    f"{row['CO2_per_capita_tonnes']:.2f}t")
    g3.metric("⚡ Renewable",     f"{row['Renewable_Energy_pct']:.1f}%")
    g4.metric("⚠️ Risk Score",    f"{row['Climate_Risk_Score']:.1f}/100")

    if not fut_row.empty:
        f2 = fut_row.iloc[0]
        st.warning(f"🔮 2030 ML Prediction: CO2={f2['Predicted_CO2']:.2f}t | Renewable={f2['Predicted_Renewable']:.1f}% | Risk={f2['Predicted_Risk']:.1f}/100")

    if st.button("📊 Generate AI Climate Insights", type="primary", use_container_width=True):

        pred_data = fut_row.iloc[0] if not fut_row.empty else None
        situation, causes, actions, pred_insight = generate_smart_insights(
            ai_ctry, row, pred_data
        )

        st.markdown(f"### 🌍 Climate Analysis Report: **{ai_ctry}** ({max_yr})")
        st.markdown("---")

        # Current Situation
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">📍 CURRENT SITUATION</div>', unsafe_allow_html=True)
        for point in situation:
            st.markdown(f'<div class="insight-item">{point}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Root Causes
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">🔍 ROOT CAUSES</div>', unsafe_allow_html=True)
        for point in causes:
            st.markdown(f'<div class="warn-item">{point}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Urgent Actions
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">🎯 URGENT ACTIONS</div>', unsafe_allow_html=True)
        for point in actions:
            st.markdown(f'<div class="action-item">{point}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 2030 Prediction
        if pred_insight:
            st.markdown("---")
            if "CRITICAL" in pred_insight or "AT RISK" in pred_insight:
                st.markdown(f'<div class="rb">🔮 2030 FORECAST: {pred_insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ib">🔮 2030 FORECAST: {pred_insight}</div>', unsafe_allow_html=True)

        # Final Verdict
        st.markdown("---")
        if row['Climate_Risk_Score'] >= 55:
            st.error(T["verdict_critical"])
        elif row['Climate_Risk_Score'] >= 28:
            st.warning(T["verdict_risk"])
        else:
            st.success(T["verdict_stable"])

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:#7F8C8D;font-size:12px;padding:8px'>
🌍 Global Temperature Intelligence System |
{T["footer"]}
</div>
""", unsafe_allow_html=True)
