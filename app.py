"""
app.py  -  Step 6: the SkillCompass India website (Streamlit, six tabs).

    streamlit run app.py

Every number on screen comes from a SQL query on warehouse.duckdb or from a model in mining.py.
If the warehouse does not exist yet, it is built automatically on first launch.
"""
import os

import duckdb
import pandas as pd
import streamlit as st

import etl_warehouse
import mining as M

st.set_page_config(page_title="SkillCompass India", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Building the warehouse (first launch only)...")
def get_con():
    if not os.path.exists(etl_warehouse.DB_PATH):
        data_dir = etl_warehouse.DATA_DIR
        have_data = os.path.isdir(data_dir) and (
            os.path.exists(os.path.join(data_dir, "jobs.csv"))
            or any(f.endswith(".xlsx") and not f.startswith("~$") for f in os.listdir(data_dir)))
        if not have_data:
            import generate_data
            generate_data.main()
        etl_warehouse.build(force=True)
    return duckdb.connect(etl_warehouse.DB_PATH, read_only=True)


def db():
    """A cursor of the shared connection. DuckDB connections are not thread-safe and Streamlit
    runs each session in its own thread, so every use gets its own cursor."""
    return get_con().cursor()


@st.cache_resource(show_spinner="Training the salary model...")
def get_salary_model():
    return M.train_salary_model(db())


@st.cache_data(show_spinner=False)
def q(sql, params=None):
    return db().execute(sql, params or []).df()


@st.cache_data(show_spinner="Mining skill bundles...")
def cached_rules(role, support, confidence, top_n):
    return M.skill_rules(db(), None if role == "All roles" else role, support, confidence, top_n)


@st.cache_data(show_spinner="Computing trends...")
def cached_trends():
    return M.skill_trends(db())


@st.cache_data(show_spinner="Running K-Means for K = 3..7 ...")
def cached_segments():
    return M.market_segments(db())


@st.cache_data(show_spinner=False)
def cached_gap(role, skills):
    return M.skill_gap(db(), role, list(skills))


@st.cache_data(show_spinner=False)
def cached_fill_summary(_bundle):
    filled = M.fill_undisclosed(db(), _bundle)
    if filled.empty:
        return 0, pd.DataFrame()
    return len(filled), filled.groupby("role")["estimated_salary_lpa"].agg(["count", "mean", "median"]).round(2)


ROLES = q("SELECT role_category FROM dim_role ORDER BY 1")["role_category"].tolist()
CITIES = q("SELECT DISTINCT city FROM dim_location ORDER BY 1")["city"].tolist()
MODES = q("SELECT DISTINCT work_mode FROM fact_job_posting ORDER BY 1")["work_mode"].tolist()
TIERS = q("SELECT tier FROM dim_experience WHERE experience_id > 0 ORDER BY experience_id")["tier"].tolist()


def in_list(col, values):
    """Build 'col IN (?, ?, ...)' with parameters; empty selection = no filter."""
    if not values:
        return "TRUE", []
    return f"{col} IN ({', '.join('?' * len(values))})", list(values)


# ---------------------------------------------------------------- header
st.title("🧭 SkillCompass India")
st.caption("Job-market skill demand mining and career intelligence · DuckDB star schema · "
           "Apriori · K-Means · Gradient Boosting")

tabs = st.tabs(["📊 Overview", "🔎 Market Explorer", "🧩 Skill Bundles",
                "🎯 Skill-Gap Analyzer", "💰 Salary Predictor", "📈 Trends & Segments"])

# ================================================================ 1. OVERVIEW
with tabs[0]:
    k = q("""SELECT COUNT(*) postings, COUNT(salary_lpa) with_salary, ROUND(AVG(salary_lpa), 2) avg_sal,
                    (SELECT COUNT(*) FROM dim_skill) skills, (SELECT COUNT(*) FROM dim_role) roles,
                    (SELECT COUNT(DISTINCT city) FROM dim_location) cities,
                    (SELECT COUNT(*) FROM dim_company) companies
             FROM fact_job_posting""").iloc[0]
    c = st.columns(6)
    c[0].metric("Postings", f"{int(k.postings):,}")
    c[1].metric("Distinct skills", int(k.skills))
    c[2].metric("Roles", int(k.roles))
    c[3].metric("Cities", int(k.cities))
    c[4].metric("Companies", int(k.companies))
    c[5].metric("Salary disclosed", f"{k.with_salary / k.postings:.0%}", help=f"{int(k.with_salary):,} postings")

    left, right = st.columns(2)
    with left:
        st.subheader("Top 20 demanded skills")
        top = q("""SELECT s.skill_name AS skill, COUNT(*) AS postings
                   FROM fact_job_skill JOIN dim_skill s USING (skill_id)
                   GROUP BY 1 ORDER BY 2 DESC LIMIT 20""")
        st.bar_chart(top, x="skill", y="postings", horizontal=True, sort="-postings")
    with right:
        st.subheader("Postings by role")
        by_role = q("""SELECT r.role_category AS role, COUNT(*) AS postings, ROUND(AVG(salary_lpa), 2) AS avg_salary_lpa
                       FROM fact_job_posting JOIN dim_role r USING (role_id) GROUP BY 1 ORDER BY 2 DESC""")
        st.bar_chart(by_role, x="role", y="postings", horizontal=True, sort="-postings")

    with st.expander("Warehouse schema (star schema) and table sizes"):
        st.code("""
                 dim_company      dim_location      dim_role
                        \\              |              /
     dim_date ──  fact_job_posting (one row per posting)  ── dim_experience
                                       |
                              fact_job_skill (bridge) ── dim_skill
""", language="text")
        sizes = pd.DataFrame([(t, db().execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in
                              ["fact_job_posting", "fact_job_skill", "dim_company", "dim_location",
                               "dim_role", "dim_experience", "dim_date", "dim_skill"]],
                             columns=["table", "rows"])
        st.dataframe(sizes, hide_index=True)

# ================================================================ 2. MARKET EXPLORER (OLAP)
with tabs[1]:
    st.subheader("OLAP on the star schema")
    f1, f2, f3 = st.columns(3)
    sel_roles = f1.multiselect("Role (slice)", ROLES, default=["Data Scientist"])
    sel_modes = f2.multiselect("Work mode (dice)", MODES, default=["Hybrid"])
    sel_tiers = f3.multiselect("Experience tier (dice)", TIERS)

    w1, p1 = in_list("r.role_category", sel_roles)
    w2, p2 = in_list("f.work_mode", sel_modes)
    w3, p3 = in_list("e.tier", sel_tiers)
    where, params = f"{w1} AND {w2} AND {w3}", p1 + p2 + p3
    base = f"""FROM fact_job_posting f
               JOIN dim_role r USING (role_id) JOIN dim_location l USING (location_id)
               JOIN dim_experience e USING (experience_id)
               WHERE {where}"""

    s = q(f"SELECT COUNT(*) n, ROUND(AVG(salary_lpa),2) avg, ROUND(MEDIAN(salary_lpa),2) med {base}", params).iloc[0]
    m = st.columns(3)
    m[0].metric("Postings in this slice", f"{int(s.n):,}")
    m[1].metric("Avg salary (LPA)", s.avg if pd.notna(s.avg) else "–")
    m[2].metric("Median salary (LPA)", s.med if pd.notna(s.med) else "–")
    if int(s.n) == 0:
        st.warning("No postings match these filters.")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown("**Roll-up: postings by city** (raw location strings rolled up to city)")
            st.bar_chart(q(f"SELECT l.city, COUNT(*) postings {base} GROUP BY 1 ORDER BY 2 DESC", params),
                         x="city", y="postings", horizontal=True, sort="-postings")
            st.markdown("**Drill-down: top skills in this slice** (via the bridge table)")
            drill = q(f"""SELECT sk.skill_name AS skill, COUNT(*) AS postings
                          FROM fact_job_posting f
                          JOIN fact_job_skill fs USING (job_id) JOIN dim_skill sk USING (skill_id)
                          JOIN dim_role r USING (role_id) JOIN dim_experience e USING (experience_id)
                          WHERE {where}
                          GROUP BY 1 ORDER BY 2 DESC LIMIT 15""", params)
            drill["pct_of_slice"] = (100 * drill.postings / s.n).round(1)
            st.dataframe(drill, hide_index=True, width="stretch")
        with b:
            st.markdown("**Pivot: average salary (LPA)**, experience tier × work mode (role filter only)")
            pv = q(f"""PIVOT (SELECT e.tier, e.experience_id, f.work_mode, f.salary_lpa
                              FROM fact_job_posting f JOIN dim_role r USING (role_id)
                              JOIN dim_experience e USING (experience_id) WHERE {w1})
                       ON work_mode IN ({", ".join("'" + m.replace("'", "''") + "'" for m in MODES)}) USING ROUND(AVG(salary_lpa), 2) GROUP BY tier, experience_id
                       ORDER BY experience_id""", p1).drop(columns="experience_id").set_index("tier")
            st.dataframe(pv, width="stretch")
            st.markdown("**Drill across: top hiring companies in this slice**")
            st.dataframe(q(f"""SELECT c.company, COUNT(*) postings, ROUND(AVG(f.salary_lpa),2) avg_salary
                               {base.replace('FROM fact_job_posting f', 'FROM fact_job_posting f JOIN dim_company c USING (company_id)', 1)}
                               GROUP BY 1 ORDER BY 2 DESC LIMIT 10""", params),
                         hide_index=True, width="stretch")

# ================================================================ 3. SKILL BUNDLES (Apriori)
with tabs[2]:
    st.subheader("Apriori: which skills employers ask for together")
    c = st.columns(4)
    role = c[0].selectbox("Role", ["All roles"] + ROLES, index=1 + ROLES.index("ML Engineer") if "ML Engineer" in ROLES else 0)
    sup = c[1].slider("Min support", 0.02, 0.30, 0.05, 0.01)
    conf = c[2].slider("Min confidence", 0.30, 0.95, 0.50, 0.05)
    top_n = c[3].slider("Skills considered", 20, 60, 45, 5)
    rules, itemsets, n = cached_rules(role, sup, conf, top_n)
    st.caption(f"{n:,} postings · {len(itemsets)} frequent itemsets · {len(rules)} rules. "
               "Support = share of postings with the whole set; confidence = P(B | A); "
               "lift = confidence / P(B), above 1 means they co-occur more than by chance.")
    if rules.empty:
        st.info("No rules at these thresholds. Lower the support or confidence.")
    else:
        only_strong = st.checkbox("Only lift > 1.1", value=True)
        show = rules[rules.lift > 1.1] if only_strong else rules
        st.dataframe(show.head(200).style.format({"support": "{:.3f}", "confidence": "{:.2f}", "lift": "{:.2f}"}),
                     hide_index=True, width="stretch")
        st.markdown("**Ask the rules:** if I know…")
        skill = st.selectbox("Skill", sorted({s for a in rules.if_you_know for s in a.split(", ")}))
        hits = rules[rules.if_you_know == skill].head(10)
        for _, r in hits.iterrows():
            st.write(f"→ **{r.employers_also_ask}**: {r.confidence:.0%} of those postings also ask for it (lift {r.lift:.2f})")
    with st.expander("Frequent itemsets"):
        st.dataframe(itemsets, hide_index=True, width="stretch")

# ================================================================ 4. SKILL-GAP ANALYZER
with tabs[3]:
    st.subheader("How ready am I, and what should I learn next?")
    target = st.selectbox("Target role", ROLES, index=ROLES.index("Data Scientist") if "Data Scientist" in ROLES else 0,
                          key="gap_role")
    role_skills = q("""SELECT s.skill_name FROM fact_job_skill fs
                       JOIN fact_job_posting f USING (job_id) JOIN dim_role r USING (role_id)
                       JOIN dim_skill s USING (skill_id)
                       WHERE r.role_category = ? GROUP BY 1 ORDER BY COUNT(*) DESC""", [target])["skill_name"].tolist()
    default = [s for s in ["python", "sql", "pandas"] if s in role_skills]
    mine = st.multiselect("Skills you already have", sorted(q("SELECT skill_name FROM dim_skill")["skill_name"]),
                          default=default)
    g = cached_gap(target, tuple(sorted(mine)))

    a, b = st.columns([1, 2])
    with a:
        st.metric("Match score", f"{g['match_score']:.0%}",
                  help="Demand-weighted share of the role's 15 most demanded skills that you have")
        st.progress(min(g["match_score"], 1.0))
        st.caption(f"Based on {g['n_postings']:,} {target} postings · role avg salary "
                   f"{g['role_avg_salary']:.2f} LPA")
        st.markdown("**Top 15 skills for this role**")
        tt = g["top_skills"].copy()
        tt["demand"] = (tt.demand * 100).round(1).astype(str) + "%"
        tt["you_have_it"] = tt.you_have_it.map({True: "✅", False: "—"})
        st.dataframe(tt, hide_index=True, width="stretch")
    with b:
        st.markdown(f"**Learn next, ranked** · priority = {M.W_DEMAND} × demand + {M.W_FIT} × fit "
                    f"+ {M.W_PREMIUM} × salary premium (scaled)")
        ln = g["learn_next"].head(15)
        if ln.empty:
            st.success("You already cover every skill with meaningful demand for this role.")
        else:
            st.dataframe(ln.style.format({"demand": "{:.0%}", "fit": "{:.0%}", "salary_premium_lpa": "{:+.2f}",
                                          "premium_scaled": "{:.2f}", "priority": "{:.3f}"}),
                         width="stretch")
            st.caption("Fit = among postings that contain at least one of your skills, the share that also "
                       "list this one. Salary premium is correlational (postings with the skill pay more on "
                       "average), not proof that the skill causes higher pay. Weights are a design choice.")

# ================================================================ 5. SALARY PREDICTOR
with tabs[4]:
    st.subheader("Expected salary for a role, city, experience and skill set")
    bundle = get_salary_model()
    mt = bundle["metrics"]
    c = st.columns(4)
    c[0].metric("Model MAE (LPA)", f"{mt['mae']:.2f}")
    c[1].metric("Baseline MAE (LPA)", f"{mt['baseline_mae']:.2f}", help="Always predicting the mean salary")
    c[2].metric("R²", f"{mt['r2']:.2f}")
    c[3].metric("Train / test", f"{mt['n_train']:,} / {mt['n_test']:,}")

    a, b = st.columns([1, 1])
    with a:
        p_role = st.selectbox("Role", ROLES, key="p_role")
        p_city = st.selectbox("City", CITIES, index=CITIES.index("Pune") if "Pune" in CITIES else 0)
        p_mode = st.radio("Work mode", MODES, horizontal=True)
        p_exp = st.slider("Experience (years)", 0.0, 15.0, 2.0, 0.5)
        p_sk = st.multiselect("Your skills", bundle["flag_skills"], default=[s for s in ["python", "sql"]
                                                                           if s in bundle["flag_skills"]],
                              help="The model has yes/no flags for the 30 most demanded skills")
        extra = st.number_input("Other skills not in the list (count)", 0, 20, 2)
    with b:
        est = M.predict_salary(bundle, p_role, p_city, p_mode, p_exp, p_sk + [f"_other{i}" for i in range(extra)])
        st.metric("Estimated salary", f"{est:.2f} LPA")
        curve = pd.DataFrame({"experience_years": [x / 2 for x in range(0, 31)]})
        curve["estimated_lpa"] = M.predict_salary_curve(bundle, p_role, p_city, p_mode,
                                                        curve.experience_years, p_sk + [f"_o{i}" for i in range(extra)])
        st.line_chart(curve, x="experience_years", y="estimated_lpa")
        st.markdown("**What drives the prediction** (feature importance)")
        imp = bundle["importance"].head(10)
        st.bar_chart(pd.DataFrame({"feature": imp.index, "importance": imp.values}),
                     x="feature", y="importance", horizontal=True, sort="-importance")
    st.caption("Gradient Boosting Regressor (250 trees, depth 4, learning rate 0.08), 80/20 split on postings that "
               "disclose salary. No label leakage: salary cannot be derived from any single input column. "
               "On synthetic data R² is high because salaries come from a formula; expect lower on real data.")
    with st.expander("Fill undisclosed salaries with the model"):
        n_hidden, by_role_est = cached_fill_summary(bundle)
        st.write(f"{n_hidden:,} postings hid their salary. Model estimates by role:")
        st.dataframe(by_role_est, width="stretch")

# ================================================================ 6. TRENDS & SEGMENTS
with tabs[5]:
    st.subheader("Rising and falling skills")
    wide, trend = cached_trends()
    if trend.empty:
        st.info("Not enough dated postings (need at least 6 months) for trend mining.")
    else:
        st.caption("growth = (avg monthly share, last 3 months + 0.1) / (avg share, first 3 months + 0.1) − 1 · "
                   "skills with < 100 postings ignored")
        fmt = {"first_3m_share": "{:.1%}", "last_3m_share": "{:.1%}", "growth": "{:+.1%}"}
        a, b = st.columns(2)
        a.markdown("**🔺 Rising**")
        a.dataframe(trend.head(8).style.format(fmt), width="stretch")
        b.markdown("**🔻 Falling**")
        b.dataframe(trend.tail(8).iloc[::-1].style.format(fmt), width="stretch")
        pick = st.multiselect("Plot monthly share", list(wide.columns),
                              default=[s for s in ["llm", "generative ai", "hadoop", "jquery"] if s in wide.columns])
        if pick:
            st.line_chart(wide[pick] * 100, y_label="% of month's postings")

    st.divider()
    st.subheader("Market segments (K-Means)")
    st.caption("Each posting is a 0/1 vector over the 40 most demanded skills. K = 3..7 scored by silhouette "
               "s = (b − a) / max(a, b) on a 3,000-posting sample; best K refit on all postings.")
    if st.button("Run clustering") or st.session_state.get("clustered"):
        st.session_state["clustered"] = True
        seg = cached_segments()
        a, b = st.columns([1, 3])
        a.metric("Best K", seg["best_k"], help=f"silhouette {seg['silhouette'].max():.3f}")
        a.dataframe(seg["silhouette"].round(3), width="stretch")
        b.dataframe(seg["profiles"], hide_index=True, width="stretch")
        st.caption("Silhouette is low (roughly 0.15–0.25) because roles share many skills, so these are broad "
                   "overlapping segments, not sharply separated groups.")
