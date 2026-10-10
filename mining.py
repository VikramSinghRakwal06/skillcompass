"""
mining.py  -  Step 5: the mining layer. Pure functions, no UI, so each can be tested alone.

    apriori / skill_rules   which skills go together           (association rules)
    skill_gap               how ready am I, what to learn next (scoring over the warehouse)
    train_salary_model      salary prediction                  (Gradient Boosting regression)
    skill_trends            rising / falling skills            (time-series over dim_date)
    market_segments         market segments                    (K-Means + silhouette)

Every function takes an open DuckDB connection and reads the star schema with SQL.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42


# ====================================================================== helpers
def postings(con, role=None):
    """One row per posting with role, city, mode, salary and its skills as a Python list."""
    sql = """
        SELECT f.job_id, r.role_category AS role, l.city, f.work_mode, f.experience_years,
               f.skills_count, f.salary_lpa, list(s.skill_name ORDER BY s.skill_name) AS skills
        FROM fact_job_posting f
        JOIN dim_role r     USING (role_id)
        JOIN dim_location l USING (location_id)
        LEFT JOIN fact_job_skill fs USING (job_id)
        LEFT JOIN dim_skill s       USING (skill_id)
        {where}
        GROUP BY ALL
        ORDER BY f.job_id
    """
    if role:
        return con.execute(sql.format(where="WHERE r.role_category = ?"), [role]).df()
    return con.execute(sql.format(where="")).df()


def skill_matrix(df, skills):
    """Boolean matrix: rows = postings, columns = skills (True if the posting lists it)."""
    idx = {s: i for i, s in enumerate(skills)}
    m = np.zeros((len(df), len(skills)), dtype=bool)
    for r, lst in enumerate(df["skills"]):
        for s in lst if lst is not None else []:
            j = idx.get(s)
            if j is not None:
                m[r, j] = True
    return m


def top_skills(df, n):
    counts = pd.Series([s for lst in df["skills"] if lst is not None for s in lst]).value_counts()
    return list(counts.index[:n])


# ====================================================================== 7.1 Apriori
def apriori(matrix, items, min_support=0.05, max_len=3):
    """
    Level-wise Apriori written from scratch.
    matrix: bool array (transactions x items). Returns {frozenset(itemset): support}.

    Level k candidates are made by joining frequent (k-1)-itemsets that share their first k-2
    items; a candidate is PRUNED if any (k-1)-subset is not frequent (the Apriori property),
    so its support is never counted.
    """
    n = matrix.shape[0]
    frequent = {}
    # level 1
    sup1 = matrix.mean(axis=0)
    level = [(i,) for i in range(len(items)) if sup1[i] >= min_support]
    for (i,) in level:
        frequent[(i,)] = sup1[i]

    k = 2
    while level and k <= max_len:
        prev = set(level)
        candidates = []
        for a, b in combinations(sorted(level), 2):
            if a[:-1] == b[:-1]:                         # join step
                cand = a + (b[-1],)
                if all(sub in prev for sub in combinations(cand, k - 1)):   # prune step
                    candidates.append(cand)
        level = []
        for cand in candidates:                          # count support with a bitwise AND
            sup = matrix[:, cand].all(axis=1).sum() / n
            if sup >= min_support:
                frequent[cand] = sup
                level.append(cand)
        k += 1
    return {frozenset(items[i] for i in key): sup for key, sup in frequent.items()}


def association_rules(freq, min_confidence=0.5):
    """All rules A -> B from each frequent itemset, with support, confidence and lift."""
    rows = []
    for itemset, sup in freq.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ante in combinations(sorted(itemset), r):
                a = frozenset(ante)
                b = itemset - a
                conf = sup / freq[a]           # P(B | A)
                if conf >= min_confidence:
                    lift = conf / freq[b]      # P(B | A) / P(B)
                    rows.append({"if_you_know": ", ".join(sorted(a)), "employers_also_ask": ", ".join(sorted(b)),
                                 "support": sup, "confidence": conf, "lift": lift})
    return pd.DataFrame(rows, columns=["if_you_know", "employers_also_ask", "support", "confidence", "lift"])


def skill_rules(con, role=None, min_support=0.05, min_confidence=0.5, top_n=45, max_len=3):
    """Apriori over the postings of one role (or all), restricted to its top_n skills."""
    df = postings(con, role)
    items = top_skills(df, top_n)
    freq = apriori(skill_matrix(df, items), items, min_support, max_len)
    rules = association_rules(freq, min_confidence)
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    itemsets = pd.DataFrame([{"itemset": ", ".join(sorted(k)), "size": len(k), "support": v}
                             for k, v in freq.items()]).sort_values(["size", "support"], ascending=[True, False])
    return rules, itemsets, len(df)


# ====================================================================== 7.2 Skill gap
W_DEMAND, W_FIT, W_PREMIUM = 0.5, 0.3, 0.2


def skill_gap(con, role, user_skills, top_k=15, min_demand=0.03):
    """
    match_score = demand-weighted share of the role's top_k skills the user already has.
    For each missing skill:
        demand  = share of role postings listing it
        fit     = among role postings that contain >=1 of the user's skills, share that list it
        premium = avg salary of role postings with it - role avg salary  (scaled 0..1 for scoring)
        priority = 0.5*demand + 0.3*fit + 0.2*premium_scaled
    """
    user = {s.strip().lower() for s in user_skills if s.strip()}
    df = postings(con, role)
    skills = top_skills(df, 200)
    m = skill_matrix(df, skills)
    demand = pd.Series(m.mean(axis=0), index=skills)

    top = demand.head(top_k)
    have = [s for s in top.index if s in user]
    match = float(top[have].sum() / top.sum()) if len(top) else 0.0

    owned_cols = [i for i, s in enumerate(skills) if s in user]
    near = m[:, owned_cols].any(axis=1) if owned_cols else np.ones(len(df), bool)
    fit = pd.Series(m[near].mean(axis=0) if near.any() else np.zeros(len(skills)), index=skills)

    sal = df["salary_lpa"].to_numpy(float)
    has_sal = ~np.isnan(sal)
    role_avg = float(np.nanmean(sal)) if has_sal.any() else float("nan")
    premium = {}
    for j, s in enumerate(skills):
        sel = m[:, j] & has_sal
        premium[s] = sal[sel].mean() - role_avg if sel.sum() >= 20 and has_sal.any() else 0.0
    premium = pd.Series(premium)

    cand = [s for s in skills if s not in user and demand[s] >= min_demand]
    out = pd.DataFrame({"skill": cand, "demand": demand[cand].values, "fit": fit[cand].values,
                        "salary_premium_lpa": premium[cand].values})
    if not out.empty:
        p = out["salary_premium_lpa"]
        out["premium_scaled"] = (p - p.min()) / (p.max() - p.min()) if p.max() > p.min() else 0.0
        out["priority"] = W_DEMAND * out.demand + W_FIT * out.fit + W_PREMIUM * out.premium_scaled
        out = out.sort_values("priority", ascending=False).reset_index(drop=True)
        out.index += 1

    top_table = pd.DataFrame({"skill": top.index, "demand": top.values,
                              "you_have_it": [s in user for s in top.index]})
    return {"match_score": match, "n_postings": len(df), "role_avg_salary": role_avg,
            "top_skills": top_table, "learn_next": out}


# ====================================================================== 7.3 Salary model
N_SKILL_FLAGS = 30
CAT = ["role", "city", "work_mode"]
NUM = ["experience_years", "skills_count"]


def _features(df, flag_skills):
    X = df[CAT + NUM].copy()
    sets = df["skills"].map(lambda l: set(l) if l is not None else set())
    for s in flag_skills:
        X["has_" + s] = sets.map(lambda st: int(s in st))
    return X


def train_salary_model(con):
    """
    Gradient Boosting regressor on postings that disclose salary. 80/20 split.
    Reported against a naive baseline that always predicts the training mean.
    Target (salary) is not a function of any single feature column -> no label leakage.
    """
    df = postings(con)
    flag_skills = top_skills(df, N_SKILL_FLAGS)
    known = df[df["salary_lpa"].notna() & df["experience_years"].notna()].reset_index(drop=True)
    X, y = _features(known, flag_skills), known["salary_lpa"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")
    model = Pipeline([("prep", pre), ("gbr", GradientBoostingRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.08, random_state=SEED))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    metrics = {"n_train": len(X_tr), "n_test": len(X_te),
               "mae": mean_absolute_error(y_te, pred),
               "baseline_mae": mean_absolute_error(y_te, np.full_like(y_te, y_tr.mean())),
               "r2": r2_score(y_te, pred)}

    # which inputs mattered: sum one-hot importances back to their source column
    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.named_steps["gbr"].feature_importances_, index=names)
    imp.index = [n.split("__", 1)[1] for n in imp.index]
    grouped = {}
    for n, v in imp.items():
        key = next((c for c in CAT if n.startswith(c + "_")), n)
        grouped[key] = grouped.get(key, 0) + v
    importance = pd.Series(grouped).sort_values(ascending=False)

    return {"model": model, "flag_skills": flag_skills, "metrics": metrics, "importance": importance}


def predict_salary(bundle, role, city, work_mode, experience_years, skills):
    skills = [s.strip().lower() for s in skills]
    row = pd.DataFrame([{"role": role, "city": city, "work_mode": work_mode,
                         "experience_years": float(experience_years),
                         "skills_count": len(skills), "skills": skills}])
    return float(bundle["model"].predict(_features(row, bundle["flag_skills"]))[0])


def predict_salary_curve(bundle, role, city, work_mode, years, skills):
    """predict_salary for many experience values in one model call."""
    skills = [s.strip().lower() for s in skills]
    rows = pd.DataFrame({"role": role, "city": city, "work_mode": work_mode,
                         "experience_years": [float(y) for y in years], "skills_count": len(skills)})
    rows["skills"] = [skills] * len(rows)
    return bundle["model"].predict(_features(rows, bundle["flag_skills"]))


def fill_undisclosed(con, bundle):
    """Estimate salary for postings that hid it (the model's practical use for the warehouse)."""
    df = postings(con)
    hidden = df[df["salary_lpa"].isna() & df["experience_years"].notna()].reset_index(drop=True)
    if hidden.empty:
        return hidden
    hidden["estimated_salary_lpa"] = bundle["model"].predict(_features(hidden, bundle["flag_skills"]))
    return hidden


# ====================================================================== 7.4 Trends
def monthly_skill_share(con):
    """Share of each month's postings that list each skill (SQL over dim_date)."""
    return con.execute("""
        WITH month_total AS (
            SELECT d.year_month, COUNT(*) AS n
            FROM fact_job_posting f JOIN dim_date d USING (date_id)
            WHERE f.date_id <> 0 GROUP BY 1
        )
        SELECT d.year_month, s.skill_name, COUNT(*)::DOUBLE / mt.n AS share
        FROM fact_job_skill fs
        JOIN fact_job_posting f USING (job_id)
        JOIN dim_date d         USING (date_id)
        JOIN dim_skill s        USING (skill_id)
        JOIN month_total mt     USING (year_month)
        WHERE f.date_id <> 0
        GROUP BY d.year_month, s.skill_name, mt.n
        ORDER BY 1
    """).df()


def skill_trends(con, min_postings=100):
    """growth = (avg share last 3 months + 0.1) / (avg share first 3 months + 0.1) - 1"""
    share = monthly_skill_share(con)
    wide = share.pivot(index="year_month", columns="skill_name", values="share").fillna(0).sort_index()
    counts = con.execute("""SELECT s.skill_name, COUNT(*) n FROM fact_job_skill
                            JOIN dim_skill s USING (skill_id) GROUP BY 1""").df().set_index("skill_name")["n"]
    wide = wide[[c for c in wide.columns if counts.get(c, 0) >= min_postings]]
    if len(wide) < 6:
        return wide, pd.DataFrame()
    first, last = wide.head(3).mean(), wide.tail(3).mean()
    growth = (last + 0.1) / (first + 0.1) - 1
    table = pd.DataFrame({"first_3m_share": first, "last_3m_share": last, "growth": growth,
                          "postings": counts.reindex(wide.columns)}).sort_values("growth", ascending=False)
    return wide, table


# ====================================================================== 7.5 K-Means
def market_segments(con, k_range=range(3, 8), n_skills=40, sample=3000):
    """
    Encode postings as 0/1 vectors over the top n_skills. For each K, score silhouette
    s = (b - a) / max(a, b) on a random sample; refit the best K on all postings and profile it.
    """
    df = postings(con)
    skills = top_skills(df, n_skills)
    X = skill_matrix(df, skills).astype(float)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X[idx])
        scores[k] = silhouette_score(X[idx], km.labels_)
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(X)

    overall = X.mean(axis=0)
    profiles = []
    for c in range(best_k):
        sel = labels == c
        share = X[sel].mean(axis=0)
        ratio = np.where(overall > 0, share / overall, 0)
        signature = [skills[j] for j in np.argsort(-ratio) if share[j] >= 0.25][:6]
        roles = df.loc[sel, "role"].value_counts(normalize=True).head(3)
        profiles.append({
            "segment": c + 1, "postings": int(sel.sum()),
            "main_roles": ", ".join(f"{r} ({p:.0%})" for r, p in roles.items()),
            "avg_salary_lpa": round(float(df.loc[sel, "salary_lpa"].mean()), 2),
            "signature_skills": ", ".join(signature),
        })
    return {"silhouette": pd.Series(scores, name="silhouette"), "best_k": best_k,
            "profiles": pd.DataFrame(profiles).sort_values("postings", ascending=False)}
