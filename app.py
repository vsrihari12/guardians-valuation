import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Guardians Valuation Tool", layout="wide")
NAVY, RED, ICE = "#00385D", "#CE1141", "#C2D4E8"


@st.cache_data
def load_data():
    """Real model output if present, otherwise clearly-labeled sample data.
    Real file needs columns: player, pos, age, years, aav, war_p10, war_p50, war_p90, injury_flag."""
    path = "data/processed/projections.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path), False
    rng = np.random.default_rng(7)
    n = 40
    df = pd.DataFrame({
        "player": [f"Player {i + 1:02d}" for i in range(n)],
        "pos": rng.choice(["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "SP", "RP"], n),
        "age": rng.integers(24, 36, n),
        "years": rng.integers(1, 7, n),
        "war_p50": np.round(np.clip(rng.normal(2.5, 1.3, n), 0.3, 6.5), 1),
    })
    df["war_p10"] = np.round(df.war_p50 * 0.65, 1)
    df["war_p90"] = np.round(df.war_p50 * 1.3, 1)
    df["aav"] = np.round(df.war_p50 * rng.uniform(5.5, 11, n), 1)
    df["injury_flag"] = np.where(df.age >= 33, "High", np.where(df.age >= 30, "Medium", "Low"))
    return df, True


def schedule(row, price, growth, disc):
    """Year-by-year projected WAR, dollar value, and discount factors for one contract."""
    t = np.arange(1, int(row.years) + 1)
    drop = 0.15 + 0.05 * max(row.age - 28, 0)
    war = np.clip(row.war_p50 - drop * (t - 1), 0, None)
    widen = 1 + 0.25 * (t - 1)
    lo = np.clip(war - (row.war_p50 - row.war_p10) * widen, 0, None)
    hi = war + (row.war_p90 - row.war_p50) * widen
    fv = war * price * (1 + growth) ** (t - 1)
    return t, war, lo, hi, fv, 1 / (1 + disc) ** t


def value(row, price, growth, disc):
    t, war, lo, hi, fv, d = schedule(row, price, growth, disc)
    return pd.Series({"surplus": (fv * d).sum() - (row.aav * d).sum(), "breakeven": (fv * d).sum() / d.sum()})


df, is_sample = load_data()

with st.sidebar:
    st.markdown("### BSN x Cleveland Guardians")
    st.markdown("**Player Valuation & Roster Decision Model**")
    positions = st.multiselect("Position", sorted(df.pos.unique()), default=sorted(df.pos.unique()))
    st.markdown("---")
    st.markdown("**Assumptions**")
    price = st.slider("Market price ($M per WAR)", 5.0, 14.0, 9.0, 0.5)
    growth = st.slider("Annual price growth", 0.0, 0.08, 0.03, 0.005, format="%.3f")
    disc = st.slider("Discount rate", 0.0, 0.12, 0.08, 0.005, format="%.3f")

df = pd.concat([df, df.apply(value, axis=1, args=(price, growth, disc))], axis=1)
view = df[df.pos.isin(positions)].sort_values("surplus", ascending=False)

if is_sample:
    st.warning("SAMPLE DATA: player names and numbers are illustrative, not real projections.")

tab1, tab2, tab3 = st.tabs(["Valuation", "FA Screener", "Scenario Builder"])

with tab1:
    name = st.selectbox("Player", view.player)
    r = df[df.player == name].iloc[0]
    st.subheader(f"{r.player}  |  {r.pos}  |  Age {r.age}  |  {int(r.years)}-year deal, ${r.aav:.1f}M AAV")
    c = st.columns(4)
    c[0].metric("Proj. WAR (yr 1)", f"{r.war_p50:.1f}", f"range {r.war_p10:.1f} to {r.war_p90:.1f}", delta_color="off")
    c[1].metric("PV surplus", f"${r.surplus:,.1f}M")
    c[2].metric("Breakeven AAV", f"${r.breakeven:,.1f}M")
    c[3].metric("Injury flag", r.injury_flag)
    t, war, lo, hi, fv, d = schedule(r, price, growth, disc)
    yrs = [f"Year {i}" for i in t]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs + yrs[::-1], y=list(hi) + list(lo[::-1]), fill="toself",
                             fillcolor="rgba(194,212,232,0.6)", line=dict(width=0), name="10th to 90th pct"))
    fig.add_trace(go.Scatter(x=yrs, y=war, mode="lines+markers", line=dict(color=NAVY, width=3),
                             marker=dict(color=RED, size=8), name="Median WAR"))
    fig.update_layout(title="Projected WAR over the contract", yaxis_title="WAR", height=380, margin=dict(t=50, b=20))
    st.plotly_chart(fig, width='stretch')
    sched = pd.DataFrame({"Year": yrs, "Proj. WAR": war.round(1), "Fair value ($M)": fv.round(1),
                          "Salary ($M)": r.aav, "Surplus ($M)": (fv - r.aav).round(1)})
    st.dataframe(sched, hide_index=True, width='stretch')

with tab2:
    st.subheader("Free-Agent Screener")
    min_war = st.slider("Minimum projected WAR", 0.0, 6.0, 0.0, 0.1)
    out = view[view.war_p50 >= min_war][["player", "pos", "age", "years", "war_p10", "war_p50", "war_p90",
                                         "aav", "breakeven", "surplus", "injury_flag"]]
    st.dataframe(out, hide_index=True, width='stretch', column_config={
        "aav": st.column_config.NumberColumn("Ask AAV", format="$%.1fM"),
        "breakeven": st.column_config.NumberColumn("Breakeven AAV", format="$%.1fM"),
        "surplus": st.column_config.NumberColumn("PV surplus", format="$%.1fM"),
        "war_p10": "WAR p10", "war_p50": "WAR p50", "war_p90": "WAR p90"})
    st.download_button("Export CSV", out.to_csv(index=False), "screener.csv", "text/csv")
    st.caption("Injury flag is shown beside the dollar value, not folded into it.")

with tab3:
    st.subheader("Scenario Builder")
    base = st.number_input("Current payroll ($M)", value=168.0, step=1.0)
    adds = st.multiselect("Players to add", df.player)
    removed = st.number_input("Salary removed ($M)", value=0.0, step=1.0)
    a = df[df.player.isin(adds)]
    new = base + a.aav.sum() - removed
    m = st.columns(3)
    m[0].metric("Payroll change", f"${new - base:+,.1f}M")
    m[1].metric("Proj. WAR added (yr 1)", f"{a.war_p50.sum():+.1f}")
    m[2].metric("PV surplus added", f"${a.surplus.sum():+,.1f}M")
    bar = go.Figure(go.Bar(x=["Current", "Scenario"], y=[base, new], marker_color=[ICE, RED], text=[f"${base:,.0f}M", f"${new:,.0f}M"]))
    bar.update_layout(title="Payroll", yaxis_title="$M", height=320, margin=dict(t=50, b=20))
    st.plotly_chart(bar, width='stretch')
