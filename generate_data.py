"""
generate_data.py  -  Step 2 of the build: create the synthetic job-posting dataset.

Produces data/jobs.csv with 12,000 postings and 16 columns.
A fixed seed (42) makes every run identical, so every number in the report is reproducible.

Realistic patterns built in:
  * each role has CORE skills (asked ~78% of the time) and OPTIONAL skills (asked ~25%)
  * some skills RISE over the 12 months (llm, generative ai, mlops, langchain ...)
    and some FALL (hadoop, hive, jquery ...)  -> gives the Trends tab something to find
  * salary = role base + experience + city + work mode + skill premiums + noise
  * only ~60% of postings disclose salary, like real portals
"""
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 12_000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jobs.csv")

# role -> (core skills, optional skills, base salary LPA at 0 yrs, LPA added per year)
ROLES = {
    "Data Scientist": (
        ["python", "sql", "machine learning", "statistics", "pandas", "numpy", "scikit-learn"],
        ["deep learning", "tensorflow", "pytorch", "nlp", "tableau", "power bi", "r", "spark", "llm",
         "generative ai", "computer vision", "data visualization", "a/b testing", "aws", "matplotlib",
         "hadoop", "hive"], 6.5, 1.55),
    "Data Analyst": (
        ["sql", "excel", "power bi", "tableau", "data visualization"],
        ["python", "pandas", "statistics", "r", "communication", "looker", "google analytics",
         "a/b testing", "etl", "hive", "snowflake"], 4.0, 0.95),
    "ML Engineer": (
        ["python", "machine learning", "deep learning", "pytorch", "tensorflow", "mlops"],
        ["docker", "kubernetes", "aws", "llm", "generative ai", "nlp", "computer vision", "spark",
         "scikit-learn", "fastapi", "sql", "langchain", "vector databases", "airflow"], 7.5, 1.7),
    "Data Engineer": (
        ["python", "sql", "spark", "etl", "airflow", "aws"],
        ["kafka", "hadoop", "hive", "snowflake", "databricks", "scala", "docker", "kubernetes",
         "data modeling", "azure", "gcp", "bigquery", "dbt", "postgresql"], 6.0, 1.45),
    "Backend Developer": (
        ["java", "spring boot", "rest api", "sql", "microservices"],
        ["node.js", "python", "django", "postgresql", "mysql", "mongodb", "redis", "kafka", "docker",
         "kubernetes", "aws", "system design", "go", "git", "hibernate", "rabbitmq"], 5.5, 1.4),
    "Frontend Developer": (
        ["javascript", "react", "html", "css", "typescript"],
        ["next.js", "redux", "tailwind css", "angular", "vue.js", "jquery", "figma", "webpack", "git",
         "jest", "bootstrap", "graphql"], 4.5, 1.15),
    "Full Stack Developer": (
        ["javascript", "react", "node.js", "html", "css", "rest api"],
        ["typescript", "next.js", "express.js", "mongodb", "postgresql", "mysql", "docker", "aws", "git",
         "redux", "graphql", "java", "spring boot", "jquery", "tailwind css", "system design"], 5.0, 1.3),
    "DevOps Engineer": (
        ["docker", "kubernetes", "linux", "ci/cd", "aws", "terraform"],
        ["jenkins", "ansible", "git", "python", "bash", "prometheus", "grafana", "azure", "gcp", "helm",
         "github actions", "mlops"], 5.5, 1.45),
    "Cloud Engineer": (
        ["aws", "azure", "gcp", "terraform", "linux", "networking"],
        ["kubernetes", "docker", "python", "bash", "cloud security", "iam", "ci/cd", "ansible",
         "serverless", "cost optimization"], 5.5, 1.4),
    "Mobile Developer": (
        ["android", "kotlin", "java", "rest api"],
        ["swift", "ios", "flutter", "dart", "react native", "firebase", "git", "jetpack compose",
         "sqlite", "figma"], 4.5, 1.2),
    "QA Engineer": (
        ["manual testing", "selenium", "test automation", "java", "sql"],
        ["python", "testng", "jira", "api testing", "postman", "cypress", "jmeter", "appium", "ci/cd",
         "agile", "cucumber"], 3.5, 0.9),
    "Cybersecurity Analyst": (
        ["network security", "siem", "vulnerability assessment", "linux", "penetration testing"],
        ["python", "firewalls", "cloud security", "iam", "incident response", "owasp", "splunk",
         "iso 27001", "networking", "bash"], 5.0, 1.3),
}
ROLE_WEIGHTS = [11, 12, 7, 9, 13, 9, 11, 7, 5, 6, 6, 4]

# generic skills any posting may add
GENERIC = ["communication", "problem solving", "agile", "git", "jira"]

# skill -> salary premium in LPA (positive = pays more, negative = pays less)
PREMIUM = {
    "llm": 2.5, "generative ai": 2.0, "mlops": 1.8, "langchain": 1.6, "vector databases": 1.4,
    "kubernetes": 1.5, "system design": 2.0, "go": 1.5, "rust": 1.8, "kafka": 1.3, "spark": 1.2,
    "pytorch": 1.2, "databricks": 1.1, "terraform": 1.2, "microservices": 1.0, "aws": 1.0,
    "snowflake": 0.9, "tensorflow": 0.8, "next.js": 0.7, "typescript": 0.6, "cloud security": 1.2,
    "penetration testing": 1.0, "scala": 0.9, "graphql": 0.5,
    "jquery": -0.8, "manual testing": -0.8, "excel": -0.5, "bootstrap": -0.4, "hadoop": -0.3,
}

# skill -> (multiplier at month 0, multiplier at month 11): linear trend over the year
TREND = {
    "llm": (0.35, 1.9), "generative ai": (0.4, 1.8), "mlops": (0.6, 1.5), "langchain": (0.3, 1.9),
    "vector databases": (0.3, 1.8), "next.js": (0.75, 1.3), "databricks": (0.75, 1.3),
    "hadoop": (1.6, 0.45), "hive": (1.5, 0.5), "jquery": (1.6, 0.4), "angular": (1.2, 0.8),
    "manual testing": (1.3, 0.7),
}

CITIES = {  # city -> (salary adjustment LPA, weight, local areas for raw location strings)
    "Bengaluru": (2.0, 20, ["Whitefield", "Koramangala", "Electronic City", "HSR Layout"]),
    "Hyderabad": (1.4, 14, ["HITEC City", "Gachibowli", "Madhapur"]),
    "Pune": (0.9, 12, ["Hinjewadi", "Kharadi", "Baner", "Magarpatta"]),
    "Mumbai": (1.5, 10, ["Andheri", "Powai", "BKC", "Navi Mumbai"]),
    "Chennai": (0.6, 9, ["OMR", "Guindy", "Taramani"]),
    "Gurugram": (1.6, 8, ["Cyber City", "Sohna Road", "Golf Course Road"]),
    "Noida": (0.8, 7, ["Sector 62", "Sector 135"]),
    "Delhi": (0.9, 5, ["Connaught Place", "Okhla"]),
    "Kolkata": (-0.6, 5, ["Salt Lake", "New Town"]),
    "Ahmedabad": (-0.4, 4, ["SG Highway", "Prahlad Nagar"]),
    "Kochi": (-0.5, 3, ["Kakkanad", "Infopark"]),
    "Jaipur": (-0.8, 3, ["Malviya Nagar", "Sitapura"]),
}
WORK_MODES = {"On-site": (0.0, 55), "Hybrid": (0.6, 30), "Remote": (1.0, 15)}

START = date(2025, 10, 1)
END = date(2026, 9, 30)


def make_companies(rng, n=200):
    a = ["Info", "Tech", "Data", "Cloud", "Quantum", "Nova", "Bright", "Apex", "Blue", "Zen",
         "Vertex", "Pixel", "Cyber", "Neo", "Smart", "Prime", "Hyper", "Next", "Core", "Sky"]
    b = ["Systems", "Labs", "Solutions", "Analytics", "Technologies", "Works", "Soft", "Digital",
         "Networks", "AI", "Logic", "Minds", "Bytes", "Ventures", "Global"]
    names = sorted({f"{x}{y.lower() if y in ('Soft', 'AI') else ' ' + y}" for x in a for y in b})
    rng.shuffle(names)
    return names[:n]


def exp_label(y):
    lo = int(y)
    return f"{lo}-{lo + 3} Yrs" if lo > 0 else "0-2 Yrs"


def title_for(role, years):
    if years < 1:
        return f"Junior {role}" if random.random() < 0.5 else f"{role} (Fresher)"
    if years >= 10:
        return random.choice([f"Lead {role}", f"Principal {role}"])
    if years >= 6:
        return f"Senior {role}"
    return role


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    companies = make_companies(random)
    company_rating = {c: round(float(rng.uniform(3.0, 4.7)), 1) for c in companies}

    roles = list(ROLES)
    cities = list(CITIES)
    city_w = np.array([CITIES[c][1] for c in cities], float)
    modes = list(WORK_MODES)
    mode_w = np.array([WORK_MODES[m][1] for m in modes], float)
    n_days = (END - START).days

    rows = []
    for i in range(N_ROWS):
        role = random.choices(roles, weights=ROLE_WEIGHTS)[0]
        core, optional, base, per_year = ROLES[role]
        posted = START + timedelta(days=int(rng.integers(0, n_days + 1)))
        t = ((posted.year - START.year) * 12 + posted.month - START.month) / 11.0  # 0 .. 1

        def p(skill, prob):
            if skill in TREND:
                a, b = TREND[skill]
                prob *= a + (b - a) * t
            return random.random() < min(prob, 0.97)

        skills = [s for s in core if p(s, 0.78)]
        skills += [s for s in optional if p(s, 0.25)]
        skills += [s for s in GENERIC if s not in skills and random.random() < 0.08]
        if not skills:
            skills = [core[0]]
        random.shuffle(skills)

        years = float(np.clip(rng.gamma(2.0, 2.2), 0, 18))
        years = round(years * 2) / 2  # half-year steps
        city = cities[rng.choice(len(cities), p=city_w / city_w.sum())]
        mode = modes[rng.choice(len(modes), p=mode_w / mode_w.sum())]

        salary = (base + per_year * years + CITIES[city][0] + WORK_MODES[mode][0]
                  + sum(PREMIUM.get(s, 0) for s in skills))
        salary = salary * (1 + rng.normal(0, 0.06)) + rng.normal(0, 0.8)
        salary = round(max(salary, 2.4), 2)
        disclosed = random.random() < 0.60

        company = random.choice(companies)
        rows.append({
            "job_id": f"JOB{i + 1:06d}",
            "job_title": title_for(role, years),
            "role_category": role,
            "company": company,
            "location": f"{random.choice(CITIES[city][2])}, {city}",
            "work_mode": mode,
            "experience_required": exp_label(years),
            "experience_years": years,
            "salary_lpa": salary if disclosed else None,
            "salary_disclosed": "Yes" if disclosed else "No",
            # mixed case + extra spaces on purpose, so the ETL has real cleaning to do
            "skills_required": ", ".join(s.title() if random.random() < 0.3 else s for s in skills),
            "posted_date": posted.isoformat(),
            "rating": company_rating[company],
            "education": random.choice(["B.Tech/B.E.", "B.Tech/B.E.", "Any Graduate", "M.Tech", "MCA", "B.Sc"]),
            "employment_type": random.choices(["Full-time", "Contract", "Internship"], weights=[88, 9, 3])[0],
            "job_description": f"{company} is hiring a {role} in {city} ({mode}). "
                               f"Key skills: {', '.join(skills[:5])}.",
        })

    df = pd.DataFrame(rows)
    # a few exact duplicates, so "drop duplicate job_id" in the ETL is not a no-op
    df = pd.concat([df, df.sample(40, random_state=SEED)], ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    n_skills = len({s.strip().lower() for v in df.skills_required for s in v.split(",")})
    print(f"Wrote {OUT}: {len(df):,} rows (incl. 40 duplicates), {df.shape[1]} columns, "
          f"{n_skills} distinct skills, {df.salary_lpa.notna().mean():.0%} disclose salary")


if __name__ == "__main__":
    main()
