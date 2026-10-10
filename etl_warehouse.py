"""
etl_warehouse.py  -  Steps 3 and 4: Extract, Transform, Load into a DuckDB star schema.

    python etl_warehouse.py            # build warehouse.duckdb (skips if it exists)
    python etl_warehouse.py --rebuild  # delete and rebuild

Source: the first .xlsx in data/ if one exists (your real dataset), otherwise data/jobs.csv.

Star schema produced
    dim_company, dim_location, dim_role, dim_experience, dim_date, dim_skill   (dimensions)
    fact_job_posting   one row per posting
    fact_job_skill     bridge table, one row per (posting, skill) pair
    cleaned_jobs       flat cleaned copy (handy for ML and debugging)
"""
import glob
import os
import re
import sys

import duckdb
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
DB_PATH = os.path.join(BASE, "warehouse.duckdb")

REQUIRED = ["job_id", "job_title", "role_category", "company", "location", "work_mode", "skills_required"]
OPTIONAL = ["experience_years", "salary_lpa", "posted_date", "rating"]

# other column names real exports often use -> our name
ALIASES = {
    "skills": "skills_required", "key_skills": "skills_required", "skill": "skills_required",
    "title": "job_title", "role": "role_category", "category": "role_category",
    "company_name": "company", "city": "location", "job_location": "location",
    "workmode": "work_mode", "mode": "work_mode", "experience": "experience_years",
    "exp_years": "experience_years", "salary": "salary_lpa", "salary_lpa_avg": "salary_lpa",
    "date_posted": "posted_date", "posted_on": "posted_date", "company_rating": "rating",
}
CITY_FIX = {"bangalore": "Bengaluru", "bengaluru": "Bengaluru", "gurgaon": "Gurugram",
            "bombay": "Mumbai", "new delhi": "Delhi", "madras": "Chennai", "calcutta": "Kolkata",
            "cochin": "Kochi"}

# (experience_id, tier, min_years, max_years)  max is exclusive
TIERS = [(0, "Unknown", None, None), (1, "Fresher", 0, 1), (2, "Junior", 1, 3),
         (3, "Mid", 3, 6), (4, "Senior", 6, 10), (5, "Lead", 10, 99)]


# ---------------------------------------------------------------- EXTRACT
def extract():
    xlsx = sorted(f for f in glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
                  if not os.path.basename(f).startswith("~$"))   # skip Excel lock files
    if xlsx:
        src = xlsx[0]
        df = pd.read_excel(src)
    else:
        src = os.path.join(DATA_DIR, "jobs.csv")
        if not os.path.exists(src):
            raise FileNotFoundError("No data found. Run `python generate_data.py` or put your .xlsx in data/.")
        df = pd.read_csv(src)
    print(f"[extract] {os.path.basename(src)}: {len(df):,} rows, {df.shape[1]} columns")
    return df, src


# ---------------------------------------------------------------- TRANSFORM
def _num(series):
    """'12.5 LPA', '3-5 yrs', '4' -> float. A range becomes its midpoint.
    Thousands separators are dropped and a rupee amount (>= 10,000, e.g. '12,00,000') becomes LPA."""
    def one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        text = re.sub(r"(?<=\d),(?=\d)", "", str(v))
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if not nums:
            return np.nan
        vals = [float(x) / 1e5 if float(x) >= 1e4 else float(x) for x in nums[:2]]
        if text.lstrip().startswith("-"):      # "-5" is negative, not the range "?-5"
            vals[0] = -vals[0]
        return sum(vals) / len(vals)
    return series.map(one)


KNOWN_CITIES = {"pune", "mumbai", "hyderabad", "chennai", "noida", "delhi", "kolkata", "ahmedabad",
                "kochi", "jaipur", "gurugram", "bengaluru", "remote", "indore", "coimbatore", "chandigarh"}


def _city(loc):
    if pd.isna(loc):
        return "Unknown"
    parts = [p.strip() for p in str(loc).split(",") if p.strip()]
    if not parts:
        return "Unknown"
    for p in reversed(parts):  # "Hinjewadi, Pune" -> Pune ; "Pune, Maharashtra" -> Pune (if known)
        if p.lower() in CITY_FIX:
            return CITY_FIX[p.lower()]
    for p in parts:
        if p.lower() in KNOWN_CITIES:
            return p.title()
    for word in re.findall(r"[a-z]+", str(loc).lower()):   # "Bangalore Urban, Karnataka", "Delhi NCR"
        if word in CITY_FIX:
            return CITY_FIX[word]
        if word in KNOWN_CITIES:
            return word.title()
    return parts[-1].title() if len(parts) > 1 and len(parts[-1]) > 2 else parts[0].title()


def _tier(y):
    if pd.isna(y) or y < 0:
        return "Unknown"
    for _, name, lo, hi in TIERS[1:]:
        if lo <= y < hi:
            return name
    return "Lead"


def _skills(v):
    """Lowercase, trim, drop empties and repeats inside one posting. Keeps original order."""
    if pd.isna(v):
        return ""
    seen, out = set(), []
    for s in re.split(r"[,;|]", str(v)):
        s = re.sub(r"\s+", " ", s).strip().lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return ",".join(out)


def transform(df):
    df = df.copy()
    df.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Source is missing required columns: {missing}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan

    n0 = len(df)
    df["job_id"] = df["job_id"].astype(str).str.strip()
    blank = df["job_id"].isin(["", "nan", "None"])          # keep rows without an id instead of merging them
    df.loc[blank, "job_id"] = [f"ROW{i:07d}" for i in df.index[blank]]
    df = df.drop_duplicates(subset="job_id", keep="first")
    print(f"[transform] dropped {n0 - len(df)} duplicate job_id rows")

    for c in ["job_title", "role_category", "company", "location", "work_mode"]:
        df[c] = df[c].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})
    df["work_mode"] = df["work_mode"].str.title().replace({"Onsite": "On-site", "On-Site": "On-site",
                                                           "Work From Home": "Remote", "Wfh": "Remote"})
    df["city"] = df["location"].map(_city)

    df["experience_years"] = _num(df["experience_years"])
    df["experience_tier"] = df["experience_years"].map(_tier)

    df["salary_lpa"] = _num(df["salary_lpa"])
    if "salary_disclosed" in df.columns:  # undisclosed -> NULL, so averages are not distorted
        hidden = df["salary_disclosed"].astype(str).str.lower().isin(["no", "false", "0", "not disclosed"])
        df.loc[hidden, "salary_lpa"] = np.nan
    df.loc[(df["salary_lpa"] <= 0) | (df["salary_lpa"] > 200), "salary_lpa"] = np.nan

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce").dt.date

    df["skills_norm"] = df["skills_required"].map(_skills)
    df["skills_count"] = df["skills_norm"].map(lambda s: len(s.split(",")) if s else 0)

    keep = ["job_id", "job_title", "role_category", "company", "location", "city", "work_mode",
            "experience_years", "experience_tier", "salary_lpa", "rating", "posted_date",
            "skills_norm", "skills_count"]
    out = df[keep].reset_index(drop=True)
    print(f"[transform] {len(out):,} clean rows | salary disclosed: {out.salary_lpa.notna().sum():,} "
          f"| dated: {out.posted_date.notna().sum():,}")
    return out


# ---------------------------------------------------------------- LOAD
LOAD_SQL = """
CREATE TABLE dim_company AS
    SELECT ROW_NUMBER() OVER (ORDER BY company) AS company_id, company
    FROM (SELECT DISTINCT company FROM cleaned_jobs);

CREATE TABLE dim_location AS
    SELECT ROW_NUMBER() OVER (ORDER BY city, location) AS location_id, city, location
    FROM (SELECT DISTINCT city, location FROM cleaned_jobs);

CREATE TABLE dim_role AS
    SELECT ROW_NUMBER() OVER (ORDER BY role_category) AS role_id, role_category
    FROM (SELECT DISTINCT role_category FROM cleaned_jobs);

CREATE TABLE dim_date AS
    SELECT 0 AS date_id, NULL::DATE AS full_date, 'Unknown' AS year_month, NULL::INT AS year, NULL::INT AS quarter
    UNION ALL
    SELECT CAST(strftime(d, '%Y%m%d') AS INT), d, strftime(d, '%Y-%m'), year(d), quarter(d)
    FROM (SELECT DISTINCT posted_date AS d FROM cleaned_jobs WHERE posted_date IS NOT NULL);

-- split the comma list into one row per skill (many-to-many)
CREATE TABLE _job_skill_raw AS
    SELECT DISTINCT job_id, skill FROM (
        SELECT job_id, UNNEST(STRING_SPLIT(skills_norm, ',')) AS skill
        FROM cleaned_jobs WHERE skills_norm <> ''
    ) WHERE skill <> '';

CREATE TABLE dim_skill AS
    SELECT ROW_NUMBER() OVER (ORDER BY skill) AS skill_id, skill AS skill_name
    FROM (SELECT DISTINCT skill FROM _job_skill_raw);

CREATE TABLE fact_job_skill AS
    SELECT r.job_id, s.skill_id
    FROM _job_skill_raw r JOIN dim_skill s ON s.skill_name = r.skill;
DROP TABLE _job_skill_raw;

CREATE TABLE fact_job_posting AS
    SELECT  j.job_id,
            c.company_id, l.location_id, r.role_id, e.experience_id,
            COALESCE(CAST(strftime(j.posted_date, '%Y%m%d') AS INT), 0) AS date_id,
            j.work_mode, j.salary_lpa, j.experience_years, j.skills_count, j.rating
    FROM cleaned_jobs j
    JOIN dim_company    c ON c.company = j.company
    JOIN dim_location   l ON l.city = j.city AND l.location = j.location
    JOIN dim_role       r ON r.role_category = j.role_category
    JOIN dim_experience e ON e.tier = j.experience_tier;
"""


def load(clean, db_path=DB_PATH):
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.register("clean_df", clean)
    con.execute("CREATE TABLE cleaned_jobs AS SELECT * FROM clean_df")
    con.unregister("clean_df")
    con.execute("CREATE TABLE dim_experience (experience_id INT, tier VARCHAR, min_years INT, max_years INT)")
    con.executemany("INSERT INTO dim_experience VALUES (?, ?, ?, ?)", TIERS)
    con.execute(LOAD_SQL)
    for t in ["dim_company", "dim_location", "dim_role", "dim_experience", "dim_date", "dim_skill",
              "fact_job_posting", "fact_job_skill", "cleaned_jobs"]:
        print(f"[load] {t:18s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>8,} rows")
    con.close()


def build(force=False):
    if os.path.exists(DB_PATH) and not force:
        print(f"{DB_PATH} already exists (use --rebuild to rebuild).")
        return
    raw, _ = extract()
    load(transform(raw))
    print(f"Warehouse ready: {DB_PATH}")


if __name__ == "__main__":
    build(force="--rebuild" in sys.argv)
