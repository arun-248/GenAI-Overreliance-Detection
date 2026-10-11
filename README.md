# 🧠 AI Overreliance Research Framework — Detecting Generative AI Overreliance and Providing Adaptive Interventions

<div align="center">

[![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white&style=for-the-badge)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-7C5CFC?style=for-the-badge)](https://shap.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white&style=for-the-badge)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Hosted%20on-Render-46E3B7?logo=render&logoColor=black&style=for-the-badge)](https://genai-overreliance-detection.onrender.com/)

**🎓 An explainable machine learning research prototype for estimating students' AI reliance, understanding the behaviours behind it, and recommending adaptive learning activities**

**Detect · Understand · Support · Learn**

<a href="https://genai-overreliance-detection.onrender.com/"><strong>🚀 Live Demo</strong></a>
·
<a href="https://github.com/arun-248/GenAI-Overreliance-Detection"><strong>📂 Source Code</strong></a>

</div>

---

## 🌟 Project Highlights

> **Not "you use AI too much."** The framework estimates an **AI Reliance Score (1–7)** from self-reported behaviour, shows **which behaviours influenced the estimate**, and offers a **hands-on activity** that helps a student move from passive AI dependence toward independent reasoning.

**🎯 What makes this special:**
- **Random Forest regression** estimates a 1–7 AI Reliance Score from 22 questionnaire-derived features
- **Explainable AI with SHAP**: a plain-language "Why this result?" page with a clear contribution chart
- **Rule-based adaptive interventions** that follow *Hint → Question → Verification → Analysis → Independent Answer*
- **Two datasets kept analytically separate**: exploratory K-Means profiling (AIBT-50K) and supervised prediction (Student Awareness Dataset)
- **Scientific honesty built in**: GAN augmentation and noise experiments are reported even where they did *not* help
- **Research dashboard** with model comparison, cross-validation, test cases, SDG 4 relevance and limitations

> ⚠️ **Research prototype.** Output is an *estimate* from self-reported answers. It is a decision-support and self-reflection tool, **not** a clinical, psychological or diagnostic assessment.

---

## 🚀 Key Features

**📊 AI Reliance Prediction**
* Guided five-section assessment with a final review before submitting
* Predicted score shown as **X / 7** with a Low / Moderate / High *relative intervention level*
* Clear reminder that model performance is population-level and an individual result is an estimate

**🔎 Explainable AI ("Why this result?")**
* Baseline score, predicted score and the difference between them
* Top factors **raising** and **lowering** the predicted reliance, in simple language
* SHAP bar chart with legend, plus an expandable technical explanation for researchers
* If the `shap` package cannot be imported on a machine, the app falls back to a built-in TreeSHAP implementation

**🎯 Adaptive Intervention**
* Behaviour-specific activity chosen by rules, not the same advice for everyone
* Interactive steps, a space for the student's own answer, and a reflection
* Progression view: Hint → Question → Verification → Analysis → Independent Answer

**🧭 Student Profile**
* Behaviour summary cards and radar chart built only from the student's own answers

**🔬 Research & Evaluation**
* Model comparison, cross-validation, feature importance, decision layer
* Dataset, methodology, GAN experiment, noise robustness, **SDG 4 relevance**, limitations, future work
* Testing page with test cases, requirement coverage and an evidence inventory

---

## 🖼️ Application Preview

<!--
Add screenshots to a "screenshots" folder, then delete the opening and closing comment markers to show them:

<div align="center">

### 🏠 Home
<img src="screenshots/home.png" alt="Home" width="800"/>

### 📝 Assessment
<img src="screenshots/assess.png" alt="Assessment" width="800"/>

### 📈 Prediction
<img src="screenshots/prediction.png" alt="Prediction" width="800"/>

### 🔎 Why This Result?
<img src="screenshots/why.png" alt="Why this result" width="800"/>

### 🎯 Intervention
<img src="screenshots/intervention.png" alt="Intervention" width="800"/>

</div>
-->

---

## 🏗️ System Architecture

The two datasets are **analytically separate** and meet only at the **conceptual** decision / intervention layer. They are **never merged row by row**, and no student is linked across them.

```
 BRANCH A · Exploratory behavioural profiling
 ┌──────────────┐   ┌─────────────────────┐   ┌──────────────────┐   ┌────────────────────┐
 │  AIBT-50K    │──▶│ Behavioural feature │──▶│ K-Means (K = 2)  │──▶│ Behavioural        │
 │ 50,000 × 56  │   │ construction (9)    │   │ exploratory      │   │ profiles (weak     │
 └──────────────┘   └─────────────────────┘   └──────────────────┘   │ separation)        │
                                                                     └─────────┬──────────┘
                                                                               │
                                                                               ▼
                                                              ┌──────────────────────────────┐
                                                              │  DECISION / INTERVENTION     │
                                                              │  LAYER  (conceptual meeting  │
                                                              │  point, no row-level merge)  │
                                                              └──────────────▲───────────────┘
                                                                             │
 BRANCH B · Supervised AI reliance prediction                                │
 ┌───────────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌───────┴────────┐
 │ Student Awareness │─▶│ Preprocessing │─▶│ Random Forest        │─▶│ AI Reliance    │
 │ Dataset (394)     │  │ + 22 features │  │ Regression           │  │ Score (1–7)    │
 └───────────────────┘  └───────────────┘  └──────────────────────┘  └───────┬────────┘
                                                                             ▼
                                              Risk level → SHAP explanation → Adaptive intervention → Student reflection
```

---

## 📊 How the System Works

**1. 📝 Student Assessment**
* Five grouped sections: Profile, AI Behaviour, Independent Thinking, Learning, Cognitive Factors
* Every section must be confirmed, then a final review is shown before prediction

**2. ⚙️ Feature Processing**
* The saved pipeline scales numeric features (`StandardScaler`) and encodes categorical ones (`OneHotEncoder`, unknown categories ignored)

**3. 🌲 ML Prediction**
* `RandomForestRegressor` (`n_estimators = 300`, `random_state = 42`) estimates the AI Reliance Score
* The saved pipeline is **loaded, never retrained**, inside the app

**4. 🎚️ Risk Level**
* The score is mapped to a relative intervention level using training-set tertiles

| Predicted score | Level shown |
|---|---|
| below 3.5 | Low Reliance |
| 3.5 to below 5.0 | Moderate Reliance |
| 5.0 and above | High Reliance |

> These are **relative intervention levels**, not clinical or diagnostic labels.

**5. 🔎 Explainable Analysis**
* SHAP shows which behaviours pushed the estimate up or down, relative to the model's baseline

**6. 🎯 Adaptive Intervention**
* A rule-based engine selects an activity (rules below)

**7. 🪞 Reflection**
* The student writes their own answer and reflects; nothing is graded

---

## 🎯 Adaptive Intervention Rules

| Level | Behaviour trigger | Intervention |
|---|---|---|
| High | Accept Without Verification ≥ 5 | Verification Challenge |
| High | Overreliance in Problem Solving ≥ 5 | Independent Problem-Solving Task |
| High | Independent Confidence ≤ 3 | Guided Independent Reasoning |
| High | otherwise | Critical Analysis Task |
| Moderate | Cross Check ≤ 3 | AI Output Verification Task |
| Moderate | Independent Confidence ≤ 3 | Guided Reasoning Question |
| Moderate | Own Ideas ≤ 3 | Independent Idea Generation |
| Moderate | otherwise | Reflective AI-Use Question |
| Low | any | Responsible AI Use Guidance |

Rules are checked top to bottom within each level. They are **prototype recommendations**; whether they improve learning has **not** been tested.

---

## 🗂️ Datasets

### Student Awareness Dataset (supervised prediction)

| Property | Value |
|---|---|
| Raw size | 557 rows × 26 columns |
| After cleaning | **394 unique records** |
| Input features | **22** (18 numeric + 4 categorical) |
| Target | `AI_Reliance_Score = (Decision_Reliance + Independent_Reliance) / 2` |
| Target range | 1 – 7 (mean ≈ 4.204, SD ≈ 1.362) |
| Target reliability | Cronbach's α ≈ 0.715 |

**Cleaning:** duplicates removed, empty `Score` column dropped, mostly-missing `Comments` dropped, five missing `Remorse` values median-imputed, malformed CGPA `3:08` read as 3.08 (raw value preserved).

**Leakage control:** `Decision_Reliance` and `Independent_Reliance` define the target, so they are **not** used as inputs.

**Features**
* Numeric: Own Ideas, Cross Check, Accept Without Verification, Fear of Independent Thinking, Recall Reduction Awareness, Overreliance in Problem Solving, Remorse, Regret, Task Regret, Time Saving, Independent Confidence, Learning Efficiency, Learning Satisfaction, Collaborative Learning, Positive Emotion, Low Mental Effort, Cognitive Burden Reduction, CGPA
* Categorical: Gender, Field of Study, AI Experience, AI Usage Frequency

### AIBT-50K (exploratory behavioural profiling)

| Property | Value |
|---|---|
| Size | 50,000 rows × 56 columns (55 questionnaire items + `Student_ID`) |
| Structure | 11 constructs × 5 items each |
| Clustering inputs | 9 constructs: Access, Understand, Apply, Analyze, Evaluate, Co-Create, Govern, AI Trust, Metacognition |
| Excluded | AI Dependency and Learning Performance |
| Method | K-Means, K = 2 |
| Limitation | **Weak cluster separation** |

K-Means is **exploratory profiling only**. It is not an overreliance classifier, and a student's own answers are never assigned to a cluster.

> 📌 Check the original dataset licences before publishing any dataset files. Do not commit private or identifiable student responses.

---

## 📈 Model Performance & Metrics

> Reported from the project's own experiments on the Student Awareness Dataset (held-out test set).

### Model comparison

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Ridge Regression | 0.8364 | 1.1002 | 0.1828 |
| **Random Forest (selected)** | **0.7694** | **1.0509** | **0.2544** |
| Gradient Boosting | 0.7972 | 1.0546 | 0.2491 |
| XGBoost | 0.7965 | 1.0674 | 0.2308 |
| MLP Regressor | 1.0050 | 1.2393 | −0.0368 |

**5-fold cross-validation (Random Forest):** MAE ≈ 0.842 · RMSE ≈ 1.142 · R² ≈ 0.266

### How to read these numbers
* **R² is not accuracy.** The Random Forest explains roughly **25%** of the variance on held-out data: there is useful signal, but the model is far from perfect.
* **MAE 0.769** means predictions are off by about 0.77 points on average on the 1–7 scale.
* The task is **regression**. Low / Moderate / High is a post-hoc layer on top of it.

### Post-hoc level metrics

| Level | Precision | Recall |
|---|---:|---:|
| Low | 0.55 | 0.55 |
| Medium | 0.61 | 0.86 |
| High | 0.57 | **0.16** |

Overall categorical accuracy is about 0.59. **Recall for the High level is limited**, so this is **not** presented as a strong three-class classifier.

### Most influential predictors
Accept Without Verification, Own Ideas, CGPA, Remorse, Learning Satisfaction, Independent Confidence, Cross Check. These are **influential predictors**, not proven causes.

---

## 🧪 Experiments

### 🎲 Generative data augmentation (CTGAN)

CTGAN generated 315 synthetic rows to accompany 315 real training rows (630 total), with values constrained to valid ranges (Likert 1–7, CGPA 2.31–4.00).

| Training data | MAE | RMSE | R² |
|---|---:|---:|---:|
| Real only (Random Forest) | 0.7694 | 1.0509 | 0.2544 |
| Real + CTGAN synthetic | 0.7703 | 1.0853 | 0.2048 |

**Result:** augmentation did **not** improve the model, so the **real-data-only Random Forest remains the deployed model**. The GAN is a documented negative result.

### 🔊 Noise robustness

Training questionnaire values were perturbed by ±1 within the 1–7 range.

| Noise level | MAE | RMSE | R² |
|---:|---:|---:|---:|
| 0% | 0.7694 | 1.0509 | 0.2544 |
| 5% | 0.7734 | 1.0532 | 0.2512 |
| 10% | 0.7802 | 1.0713 | 0.2251 |
| 15% | 0.7692 | 1.0726 | 0.2233 |

The model is fairly stable at 5% noise, while RMSE rises and R² falls at 10–15%. It is **not** noise-proof.

---

## 🧭 Application Pages

| Journey | Page | What it shows |
|---|---|---|
| — | **Home** | Overview, two journeys, how it works, research highlights |
| 🎓 Student | **Assess** | Five-section assessment and final review |
| 🎓 Student | **Prediction** | Radial score, level, scale position, limitations |
| 🎓 Student | **Why This Result?** | Baseline, factors up/down, SHAP chart, technical explanation |
| 🎓 Student | **Intervention** | Level → trigger → intervention, interactive activity, progression |
| 🎓 Student | **Student Profile** | Behaviour cards and radar chart |
| 🔬 Research | **Model Insights** | Model comparison, cross-validation, feature importance, SHAP summary, residuals, distributions, decision layer, deep-learning comparison |
| 🔬 Research | **Research** | Dataset, methodology, ML, XAI, GAN, noise, limitations, future work, **SDG 4 relevance** |
| 🔬 Research | **Testing** | Test cases, requirement coverage, evidence inventory |
| 🔬 Research | **About** | Project, team contributions, disclaimer |

---

## 🌍 SDG 4 — Quality Education

The framework is designed to contribute to **UN SDG 4 (Quality Education)** by supporting responsible, AI-assisted learning that keeps the student's own reasoning at the centre: verification habits, independent problem solving and reflective use of AI.

> This is the **intended contribution**, not demonstrated impact. The prototype has not been tested in classrooms.

---

## 🛠️ Technologies Used

### 🤖 Machine Learning & Explainability
- **scikit-learn** – preprocessing pipeline and Random Forest regression
- **SHAP** – model explanations (with a built-in TreeSHAP fallback)
- **XGBoost / Gradient Boosting / Ridge / MLP** – comparison models in the experiments
- **CTGAN** – synthetic-data augmentation experiment

### 🌐 Web Application
- **Streamlit** – multi-page interactive interface with custom CSS
- **Matplotlib** – themed charts
- **Pandas / NumPy** – data handling
- **Joblib** – loading the saved pipeline

### 🔧 Infrastructure
- **Docker** – reproducible container
- **Render** – cloud hosting
- **Git / GitHub** – version control

---

## 📁 Project Structure

```
GenAI-Overreliance-Detection/
│
├── app.py                      # Streamlit application (all pages and logic)
├── final_rf_pipeline.pkl       # Saved preprocessing + Random Forest pipeline
├── model_metadata.pkl          # Feature lists and target name
├── risk_thresholds.pkl         # Risk thresholds (3.5 / 5.0)
├── research_results.json       # Reported research results shown on the website
├── requirements.txt            # Pinned Python dependencies
├── Dockerfile                  # Container build for deployment
├── .dockerignore               # Files excluded from the Docker build
├── .gitignore                  # Files excluded from Git
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
└── README.md                   # This file
```

### Optional files the app can use
These are **not included** in the repository yet. When present next to `app.py`, the matching views switch on. Nothing is invented when they are missing.

| File | Enables |
|---|---|
| `evaluation_predictions.csv` (columns `Actual`, `Predicted`) | Actual-vs-predicted, residual plot, risk distribution |
| `reference_data.csv` (the 22 feature columns) | SHAP summary across students, risk and intervention distributions |
| `deep_learning_comparison.csv` (`Member, Dataset, Model, MAE, RMSE, R2`) | Per-member deep-learning comparison table |
| `test_results.json` (produced by running `test_app.py`) | Test cases with expected / observed / status on the Testing page |

---

## ⚖️ Responsible AI & Limitations

1. **Self-reported data.** The framework never reads students' real AI conversations or browsing history.
2. **Moderate predictive performance.** R² ≈ 0.25, so a result is an *estimate*, not a judgement.
3. **Limited High-level recall (≈ 0.16).** Do not treat the levels as a strong classifier.
4. **Exploratory K-Means.** Weak separation, so the profiles are not validated student types.
5. **GAN did not help.** Real-only Random Forest stays the deployed model.
6. **Noise sensitivity.** Performance degrades at higher noise levels.
7. **Rule-based interventions.** Their effect on learning has not been measured; longitudinal studies are needed.
8. **No causality.** Feature importance and SHAP describe model behaviour, not causes.
9. **Small sample (394).** External validation across institutions is needed.
10. **Not for high-stakes use.** Never use it for grading, discipline or ranking students.

---

## 🔄 Future Enhancements

<details>
<summary><strong>🎯 Near-term</strong></summary>

- [ ] Add held-out evidence files (`evaluation_predictions.csv`, `reference_data.csv`) for residual and distribution views
- [ ] Add the per-member deep-learning comparison across datasets
- [ ] Commit an automated test script and results for the Testing page
- [ ] Add screenshots to this README

</details>

<details>
<summary><strong>🌟 Research roadmap</strong></summary>

- [ ] Larger, more diverse datasets and external validation
- [ ] Better recall for the High level and tested alternative thresholds
- [ ] Prediction uncertainty and calibration
- [ ] Longitudinal study of whether interventions change verification and independent reasoning
- [ ] Shared instrument so profiling and prediction can be linked at the individual level
- [ ] Fairness and privacy evaluation before any real educational use

</details>

---

## 👥 Team

| Role | Focus |
|---|---|
| Member 1 | AI Dependency, Cognitive Behaviour and Overreliance Analysis |
| Member 2 | AI Trust, Learning Behaviour and Psychological Factors |
| **Member 3 — C Arun** | **Machine Learning, Risk Prediction and Adaptive Intervention** (supervised learning, Random Forest, SHAP, adaptive interventions, CTGAN experiment, noise robustness, Streamlit deployment) |

*M.Tech Artificial Intelligence · Research Methodology Project*

---

## 🔗 Links

* **Live application:** https://genai-overreliance-detection.onrender.com/
* **Repository:** https://github.com/arun-248/GenAI-Overreliance-Detection

---

## 📄 License

```
MIT License

Copyright (c) 2026 Arun Chinthalapally

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

The software is MIT-licensed. The original datasets are **not** covered by this licence and keep their own terms.

---

<div align="center">

## 🎓 Promoting responsible AI-assisted learning

**We are not trying to remove AI from education. We are helping students use AI as a learning assistant without letting it replace independent thinking.**

---

**Made with ❤️ by [Arun Chinthalapally](https://github.com/arun-248)**

[⬆ Back to Top](#-ai-overreliance-research-framework--detecting-generative-ai-overreliance-and-providing-adaptive-interventions)

</div>
