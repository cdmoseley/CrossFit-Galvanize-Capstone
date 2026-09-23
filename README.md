# CrossFit Open → Quarterfinals Predictors (Galvanize Capstone)

**Predicting Quarterfinals qualification from 2024 CrossFit Open leaderboard + athlete benchmark PRs,** with recommendations for Army Holistic Health and Fitness (H2F) programming.

| | |
|---|---|
| **Role** | End-to-end data project: scrape → clean → EDA → hypothesis tests → ML |
| **Outcome** | Logistic regression ~80% test accuracy predicting Top 25% / Quarterfinals (80/20 split) |
| **Key finding** | Clean & Jerk and Snatch consistently rank among top performance predictors (men & women) |
| **Stack** | Python, pandas, BeautifulSoup/aiohttp, scikit-learn, XGBoost, SciPy, Folium |
| **Artifacts** | Notebooks · cleaned CSVs · [US affiliate map](./crossfit_affiliates_with_stats_bold.html) |

---

## Demo

- **Interactive map:** [US CrossFit affiliates with summary stats](./crossfit_affiliates_with_stats_bold.html) — open in a browser to explore affiliate locations and baseline statistics.
- **Key charts** (hosted assets from the original capstone write-up):

<p align="center">
  <img width="435" alt="2024 Open athlete gender split: about 55% men and 45% women" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/e6919c2c-7331-4176-9cb6-7ee14ce6e703">
  <img width="447" alt="Regional mix of Open athletes: North America largest, then Europe and South America" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/f52c566a-37d8-4e4d-ad21-f2296d580775">
</p>

*Figure: Open participation by gender and region (~300,000+ athletes; North America–heavy).*

<p align="center">
  <img width="440" alt="XGBoost feature importance: Clean and Jerk and Snatch among top predictors for men" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/032996b1-1be9-4dd4-8a85-9239189b1124">
</p>

*Figure: XGBoost feature importance — Olympic lifts again near the top of the ranking.*

<p align="center">
  <img width="307" alt="Modeled Quarterfinals probability near mean benchmarks about 22 percent" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/bbc956e7-293b-4771-8e25-0ae3d0927ac7">
  <img width="333" alt="Modeled Quarterfinals probability at 75th percentile benchmarks about 83 percent" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/bb404f80-e615-4fdd-aa47-639384cea2a2">
</p>

*Figure: Logistic scenarios — mean benchmarks ≈ 22% modeled Quarterfinals probability; 75th percentile across benchmarks ≈ 83%.*

---

## Motivation

In January 2023, the Data Warfare Company from Fort Liberty visited Fort Stewart for a readiness and lethality study tied to Army Holistic Health and Fitness (H2F). That work is still early: trainers introduce functional movements and nutrition, and the program continues to grow.

I analyzed 2024 CrossFit Open data because H2F-style programming already looks a lot like CrossFit (minus specialty gymnastics), CrossFit has deep roots with military and first responders, and — uniquely — **sharing scores is built into the culture**. Scores on the whiteboard and the Games app create a large observational dataset you can actually scrape and model.

The Army does not need to become the world’s biggest CrossFit affiliate. The goal is transferable lessons: which capacity markers associate with high overall performance, and how that thinking can inform programming while H2F’s own longitudinal data matures.

---

## Data

**Source.** 2024 CrossFit Open leaderboard scores plus linked athlete profiles, collected with a custom Python scraper (`Crossfit_Webscrape.ipynb`). CrossFit data is used here for personal / educational analysis only.

**Leaderboard fields.** Overall ranking and per-workout ranks/scores for Open workouts 24.1–24.3.

**Profile fields.** Age, height, weight, gender, region/affiliate, and self-reported benchmark Personal Records (lifts, Fran, 5K, etc.). BMI is derived from height and weight.

**Scale.** Just over 300,000 athletes competed in the Open (~55% men / ~45% women). Analysis-ready tables in this repo:

| File | Rows (approx.) | Notes |
|---|---|---|
| `Data/Analysis_Data/Crossfit_Men.csv` | 176,260 | Primary modeling set used in the logistic / XGBoost notebooks |
| `Data/Cleaning_Data/Pre-Clean_Men_Crossfit.csv` | 176,260 | Pre-clean men’s extract |
| `Data/Cleaning_Data/Pre-Clean_Women_Crossfit.csv` | 124,958 | Pre-clean women’s extract |
| `Data/Analysis_Data/Victory_*.xlsx` | small | Supporting workbook artifacts |

Women’s parallel analysis lives in the notebooks; a separate `Crossfit_Women.csv` analysis export may need to be regenerated from the cleaning notebook if not present locally.

**Caveat (important).** Validated Open scores are judged; biometric and benchmark PR fields are **self-reported**, so selection and measurement bias apply. With large *n* for both men and women, the patterns are still useful — but they are observational, not causal.

---

## Methods

1. **Target.** Binary label: athlete in the **Top 25%** overall (2024 Quarterfinals cutoff).
2. **Features.** Age / height / weight / BMI plus self-reported benchmark PRs (Back Squat, Deadlift, Clean & Jerk, Snatch, Fran, 5K, and related benchmarks).
3. **EDA.** Regional mix; biometric distributions vs Army-relevant ranges; BMI bands vs lift and run performance.
4. **Four complementary views of “what predicts excellence”:**
   - Among athletes in the top 5% on a given benchmark, what % reached Quarterfinals?
   - Hypothesis tests comparing Open rank for top 25% vs bottom 75% on each benchmark
   - XGBoost feature importance
   - Logistic regression coefficients + predicted-probability scenarios
5. **Model detail (logistic).** Benchmark features only; mean imputation; `StandardScaler`; `train_test_split` **80/20**, `random_state=42`. ROC curve is plotted in the notebook; headline metric reported below is held-out accuracy.

---

## Results

- **Convergent signal.** Clean & Jerk, Snatch (± Deadlift) rose to the top across the %→Quarterfinals view, hypothesis tests, XGBoost importance, and logistic coefficients for both men and women.
- **Secondary predictors.** Fran and the 5K run also ranked highly in the ML models.
- **Model.** Logistic regression predicting Quarterfinals (Top 25%) — **~80% training / ~80% test accuracy** on the men’s analysis set (README narrative previously cited ~78%; notebook outputs are ~0.80).
- **Scenarios.** Mean benchmarks ≈ **22%** modeled Quarterfinals probability; **75th percentile** across benchmarks ≈ **83%**. Average scores above roughly the **58th percentile** across benchmarks → >50% modeled chance; above roughly the **71st percentile** → >75% modeled chance.
- **BMI (exploratory).** Men near mean BMI (~27) associated with stronger overall ranks; heavier athletes stronger on squat/deadlift, lighter faster on runs. **BMI ≠ body composition** — needs body-fat data before strong claims.

<p align="center">
  <img width="149" alt="Share of top 5 percent performers on each benchmark who reached Quarterfinals, men" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/71ae6d4b-3fbf-4a98-9672-569f98495e3f">
  <img width="157" alt="Share of top 5 percent performers on each benchmark who reached Quarterfinals, women" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/242d6a5a-a848-455f-8e86-4b4d76487887">
</p>

*Figure: Top-5% benchmark → Quarterfinals conversion (sensitive to who logged a PR for that lift).*

<p align="center">
  <img width="322" alt="Hypothesis-test ranking of benchmarks by Open-rank separation, men" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/6db4ebde-d80a-44e2-ad56-a144a73d6109">
  <img width="329" alt="Hypothesis-test ranking of benchmarks by Open-rank separation, women" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/c0a03339-5290-497a-922d-c6d362c78e28">
</p>

*Figure: Hypothesis-test rankings — Clean & Jerk / Snatch / Deadlift lead for men and women.*

<p align="center">
  <img width="476" alt="Logistic regression coefficients highlighting Olympic lifts, Fran, and 5K" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/76119c1d-5704-4e33-afe5-6314f6d37f26">
</p>

*Figure: Logistic coefficients — Olympic lifts again among the largest effects; Fran and 5K also matter.*

<p align="center">
  <img width="497" alt="Athlete benchmark Personal Records distributions used as model features" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/debdb4c5-16b1-435e-b46f-a60404a6e04a">
</p>

*Figure: Benchmark Personal Record identifiers used as features.*

---

## Recommendations & limitations

**Data-informed suggestions for programming discussion — not medical, coaching certification, or Army doctrine.**

1. **Olympic lifts (Clean & Jerk / Snatch)** are the strongest recurring predictors. Prefer adding them only with trained coaching; otherwise use lower-skill power alternatives (DB clean & press, DB snatch, kettlebell swing, med-ball clean, overhead press).
2. **Supplement with Barbell Thrusters (Fran), Deadlift, and 5K running** — also high-signal in the models and aligned with many H2F-style templates.
3. **BMI near the male Open mean (~27)** tracked with better overall ranks in this sample, but treat as exploratory until body-fat % (or better composition metrics) exist.
4. For CrossFit athletes tracking Games-app percentiles: staying above roughly the **58th percentile** across benchmarks was associated with a >50% modeled Quarterfinals chance in this logistic setup.

**Limitations.** Self-reported PRs; missingness / who chooses to log a lift; observational associations only; Top-25% is a large minority class — accuracy alone is a soft metric (precision/recall/F1 and a majority-class baseline would strengthen follow-up work); women’s analysis CSV may need regeneration from cleaning notebooks.

---

## Tech stack

- **Language:** Python 3.11 (developed in Anaconda / Jupyter)
- **Data:** pandas, numpy, openpyxl
- **Scraping:** BeautifulSoup, requests, aiohttp, tqdm
- **Stats / ML:** SciPy, scikit-learn, XGBoost
- **Viz / maps:** matplotlib, seaborn, Folium, geopandas, shapely

---

## Project structure

```text
.
├── Crossfit_Webscrape.ipynb              # Scrape Open leaderboard + athlete profiles
├── Female_Cleaning.ipynb                 # Clean / join women’s extracts → analysis-ready tables
├── Crossfit_Analysis.ipynb               # EDA, tests, XGBoost, logistic scenarios, Folium map
├── crossfit_affiliates_with_stats_bold.html  # Interactive US affiliate map (demo)
├── Data/
│   ├── Cleaning_Data/                    # Pre-clean men’s & women’s CSVs
│   └── Analysis_Data/                    # Analysis CSVs + Victory workbooks
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/cdmoseley/CrossFit-Galvanize-Capstone.git
cd CrossFit-Galvanize-Capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Recommended run order**

1. `Crossfit_Webscrape.ipynb` — scrape leaderboard + profiles *(optional / large; raw scraped files are gitignored)*
2. `Female_Cleaning.ipynb` — produce analysis-ready women’s CSVs (men’s parallel cleaning lives in the analysis workflow / pre-clean extracts)
3. `Crossfit_Analysis.ipynb` — EDA, hypothesis tests, XGBoost, logistic regression
4. Open `crossfit_affiliates_with_stats_bold.html` in a browser for the US affiliate map

**Data included:** `Data/Analysis_Data/Crossfit_Men.csv`, Victory workbooks, and pre-clean CSVs under `Data/Cleaning_Data/`. Paths inside notebooks may still point at a local Mac absolute path — update to relative `Data/...` paths when re-running.

---

## Future work

- Re-run the pipeline on H2F / unit test data once sample sizes support stable conclusions
- Add body-fat % (or better composition measures) before leaning on BMI findings
- Study transfer from these Open/benchmark predictors to ACFT (or similar standardized military tests), with clearer class-balance metrics (F1, ROC-AUC, baselines)

---

## Author

**Chase Moseley** — Galvanize / Data Science Immersive capstone  
GitHub: [cdmoseley](https://github.com/cdmoseley)
