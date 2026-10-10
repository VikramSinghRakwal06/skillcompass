"""
test_pipeline.py  -  Step 7: tests for every layer.

    python test_pipeline.py        (or: pytest -q test_pipeline.py)

The ETL tests use a temporary warehouse. The model tests read warehouse.duckdb (built if missing).
"""
import os
import tempfile

import duckdb
import numpy as np
import pandas as pd

import etl_warehouse as E
import mining as M


def _tiny_raw():
    return pd.DataFrame({
        "Job ID": ["J1", "J2", "J2", "J3", "J4"],                       # J2 duplicated
        "Title": ["DS", "DS", "DS", "BE", "BE"],
        "Role": ["Data Scientist", "Data Scientist", "Data Scientist", "Backend Developer", "Backend Developer"],
        "Company Name": ["A", "B", "B", "A", "C"],
        "Job Location": ["Hinjewadi, Pune", "Bangalore", "Bangalore", "Gurgaon, Haryana", "Delhi NCR"],
        "Work Mode": ["hybrid", "Remote", "Remote", "onsite", "On-site"],
        "Key Skills": ["Python, SQL , python", "python;Pandas", "python;Pandas", "Java|Spring Boot", ""],
        "Experience": ["2-4 Yrs", "0", "0", 7, None],
        "Salary": ["12 LPA", None, None, "20", "12,00,000"],
        "Date Posted": ["2026-01-05", "2026-02-10", "2026-02-10", "bad date", "2026-03-01"],
    })


def test_transform_cleans_and_maps_aliases():
    t = E.transform(_tiny_raw())
    assert len(t) == 4                                        # duplicate dropped
    j1 = t.set_index("job_id").loc["J1"]
    assert j1.skills_norm == "python,sql" and j1.skills_count == 2   # lowercase, trim, de-dup
    assert j1.city == "Pune" and j1.experience_years == 3.0 and j1.experience_tier == "Mid"
    assert t.set_index("job_id").loc["J2"].city == "Bengaluru"
    assert t.set_index("job_id").loc["J3"].city == "Gurugram"
    assert t.set_index("job_id").loc["J3"].work_mode == "On-site"
    assert t.set_index("job_id").loc["J4"].city == "Delhi"
    assert t.set_index("job_id").loc["J4"].salary_lpa == 12.0        # rupees per annum -> LPA
    assert E._num(pd.Series(["10-15 LPA", "-5"])).tolist() == [12.5, -5.0] and E._tier(-1) == "Unknown"
    assert t.set_index("job_id").loc["J3"].posted_date is None or pd.isna(t.set_index("job_id").loc["J3"].posted_date)


def test_load_builds_star_schema():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "w.duckdb")
        E.load(E.transform(_tiny_raw()), db)
        con = duckdb.connect(db, read_only=True)
        assert con.execute("SELECT COUNT(*) FROM fact_job_posting").fetchone()[0] == 4
        assert con.execute("SELECT COUNT(*) FROM fact_job_skill").fetchone()[0] == 6   # 2+2+2+0
        assert con.execute("SELECT date_id FROM fact_job_posting WHERE job_id='J3'").fetchone()[0] == 0
        con.close()


def test_apriori_matches_hand_count():
    items = ["a", "b", "c"]
    m = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]], bool)
    freq = M.apriori(m, items, min_support=0.5)
    assert freq[frozenset("a")] == 0.75 and freq[frozenset("ab")] == 0.5
    assert frozenset("abc") not in freq                    # support 0.25 < 0.5
    rules = M.association_rules(freq, 0.6).set_index(["if_you_know", "employers_also_ask"])
    r = rules.loc[("a", "b")]
    assert abs(r.confidence - 2 / 3) < 1e-9 and abs(r.lift - (2 / 3) / 0.75) < 1e-9


def _warehouse():
    if not os.path.exists(E.DB_PATH):
        if not os.path.exists(os.path.join(E.DATA_DIR, "jobs.csv")):
            import generate_data
            generate_data.main()
        E.build(force=True)
    return duckdb.connect(E.DB_PATH, read_only=True)


def test_gap_salary_trends_clusters_on_warehouse():
    con = _warehouse()
    g = M.skill_gap(con, "Data Scientist", ["python", "sql", "pandas"])
    assert 0 < g["match_score"] < 1
    assert not g["learn_next"].empty and "python" not in set(g["learn_next"].skill)
    assert g["learn_next"].priority.is_monotonic_decreasing

    b = M.train_salary_model(con)
    assert b["metrics"]["mae"] < b["metrics"]["baseline_mae"]
    lo = M.predict_salary(b, "Data Scientist", "Pune", "Hybrid", 2, ["python"])
    hi = M.predict_salary(b, "Data Scientist", "Pune", "Hybrid", 8, ["python"])
    assert hi > lo

    _, trend = M.skill_trends(con)
    if not trend.empty:
        assert trend.growth.is_monotonic_decreasing

    seg = M.market_segments(con, k_range=range(3, 5), sample=1000)
    total = con.execute("SELECT COUNT(*) FROM fact_job_posting").fetchone()[0]
    assert seg["best_k"] in (3, 4) and seg["profiles"].postings.sum() == total
    con.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
