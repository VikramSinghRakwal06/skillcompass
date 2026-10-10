# SkillCompass India

Job-market skill demand mining and career intelligence system (DWM PBL, 2026-27).
Raw postings → ETL → DuckDB star schema → OLAP + Apriori + K-Means + Gradient Boosting → six-tab Streamlit site.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch the app generates the data and builds `warehouse.duckdb` by itself (about 10 seconds).
To run the steps by hand:

```bash
python generate_data.py           # -> data/jobs.csv (12,000 postings + 40 duplicates)
python etl_warehouse.py --rebuild # -> warehouse.duckdb (star schema)
python test_pipeline.py           # 4 tests, all layers
```

## Using your real dataset

Put your `.xlsx` (e.g. `job_market_pbl_FINAL_CLEANED.xlsx`) in `data/`, delete `warehouse.duckdb`, and rerun.
The ETL picks the first `.xlsx` in `data/` over `jobs.csv`. Required columns (common alternative names like
`Key Skills`, `Company Name`, `Job Location` are mapped automatically):
`job_id, job_title, role_category, company, location, work_mode, skills_required`.
Optional: `experience_years, salary_lpa, posted_date, rating` (trends need `posted_date`; the salary model needs salary + experience).

## Files

| File | Layer | What it does |
|---|---|---|
| `generate_data.py` | Source | Synthetic postings, seed 42: 12 roles, 12 cities, 200 companies, Oct 2025–Sep 2026, rising/falling skills, ~60% salary disclosure |
| `etl_warehouse.py` | ETL + load | De-dup, clean, normalise skills, derive city and experience tier, build 6 dimensions + fact + bridge with SQL |
| `mining.py` | Mining | Apriori (written from scratch), skill-gap scoring, salary regression, trend mining, K-Means |
| `app.py` | Presentation | Six tabs: Overview, Market Explorer (OLAP), Skill Bundles, Skill-Gap Analyzer, Salary Predictor, Trends & Segments |
| `test_pipeline.py` | Tests | Cleaning rules, star-schema load, Apriori vs hand count, all models on the warehouse |

## Actual results from this build (use these in the report)

| Item | Value |
|---|---|
| Postings / distinct skills / roles / cities / companies | 12,000 / 124 / 12 / 12 / 200 |
| Duplicates removed by ETL | 40 |
| Bridge table `fact_job_skill` | 96,629 rows |
| Postings with disclosed salary | 7,298 (61%) |
| Apriori, all roles (support 0.05, confidence 0.5, top 40 skills) | 390 rules in ~0.2 s, max lift ≈ 9.3 |
| Salary model (GBR 250 trees, depth 4, lr 0.08) | MAE 1.23 LPA vs baseline 4.55 LPA, R² 0.92 |
| K-Means (K = 3..7, silhouette on 3,000 sample) | best K = 7, silhouette 0.241 |
| Rising skills | llm, generative ai, mlops, langchain, vector databases |
| Falling skills | hive, hadoop, jquery, manual testing |
