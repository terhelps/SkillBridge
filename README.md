# SkillBridge

A hiring analysis pipeline that identifies in-demand skills from job descriptions and predicts hiring outcomes by course performance — built to surface actionable recruitment and curriculum insights.

---

## Overview

SkillBridge combines NLP-based job analysis with logistic regression to answer two questions:

1. **What skills are employers actually asking for?** — `course_model.py` scrapes job descriptions, extracts skills via an LLM, clusters them, and suggests course names for each cluster.
2. **Which course scores actually lead to getting hired?** — `regression_model.py` trains a logistic regression model per role track and ranks each course by its hiring signal strength.

---

## Project Structure

```
SkillBridge/
├── data/
│   ├── jobscope.csv          # Job postings with descriptions
│   └── placement_data.csv    # Candidate scores and hiring outcomes
├── models/
│   ├── course_model.py       # Skill extraction and clustering
│   └── regression_model.py   # Logistic regression per role track
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/SkillBridge.git
cd SkillBridge
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and run Ollama (required for `course_model.py`)

Ollama is a local LLM runner — it is not pip-installable. Install it from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3
ollama serve
```

Leave `ollama serve` running in a terminal before executing `course_model.py`.

---

## Usage

### Model 1 — Skill Clustering (`course_model.py`)

Extracts skills from job descriptions and groups them into suggested course themes.

```bash
python models/course_model.py
```

**What it does:**
- Sends each job description to `llama3` to extract skills as a comma-separated list
- Vectorises the extracted skills with TF-IDF
- Uses the elbow/silhouette method to find the optimal number of KMeans clusters
- Asks `llama3` to label each cluster with a short course name

**Sample output:**
```
Optimal number of clusters: 4

Cluster 0: SQL and Data Pipeline Engineering
Cluster 1: Financial Modelling and Reporting
Cluster 2: Business Intelligence and Visualisation
Cluster 3: Compliance and Regulatory Operations

── Suggested Courses ──────────────────────
 cluster  course_name                              job_count
       0  SQL and Data Pipeline Engineering               18
       1  Financial Modelling and Reporting               12
       ...
```

> Note: This model calls the LLM once per job description. Expect ~5 minutes runtime depending on hardware.

---

### Model 2 — Hiring Signal Analysis (`regression_model.py`)

Trains a logistic regression model per role track and ranks each course by its influence on hiring probability.

```bash
python models/regression_model.py
```

**What it does:**
- Filters `placement_data.csv` by role track (e.g. Data Analytics, Finance Operations)
- Trains a separate logistic regression model per track using only the courses relevant to that track
- Outputs coefficients ranked by hiring signal strength

**How to read the output:**

| Coefficient | Interpretation |
|---|---|
| > 0.3 | Strong — employers value this highly |
| 0.1 – 0.3 | Moderate — some hiring impact |
| -0.1 – 0.1 | Weak — little hiring impact |
| < -0.1 | Negative — review this course |

**Sample output:**
```
TRACK: Data Analytics
Candidates in track : 80
Hired               : 56 (70.0%)

  Course                          Coefficient  Interpretation
  -----------------------------------------------------------------
  capstone_score                       1.0247  Strong — employers value this highly
  sql_score                            1.0137  Strong — employers value this highly
  python_score                         0.7707  Strong — employers value this highly
```

---

## Data

| File | Description |
|---|---|
| `jobscope.csv` | Job postings with columns: `id`, `company`, `position`, `role_track`, `location`, `industry`, `Job_Description` |
| `placement_data.csv` | Candidate records with course scores (`sql_score`, `python_score`, etc.), `hired` label (1/0), and post-placement performance ratings |

> `placement_data.csv` contains synthetic candidate data generated for this project.

---

## Requirements

```
pandas
scikit-learn
ollama
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Tech Stack

- **Python** — data processing and modelling
- **Pandas** — data wrangling
- **Scikit-learn** — TF-IDF vectorisation, KMeans clustering, logistic regression
- **Ollama / LLaMA 3** — local LLM for skill extraction and cluster labelling

---
