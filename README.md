# Stack Overflow Developer Survey 2025: A Multi-Method Machine Learning Analysis

## Project Overview

This project applies seven distinct machine learning and statistical methodologies to the **Stack Overflow Annual Developer Survey 2025** to answer publication-ready research questions about AI tool adoption, developer compensation, career satisfaction, technology trends, and role classification.

Each notebook is fully self-contained: it reads the raw CSV from `/kaggle/input/`, performs all preprocessing, trains models, and saves all figures (PDF) and tables (CSV) to `/kaggle/working/`.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | Stack Overflow Annual Developer Survey 2025 |
| **Source** | [Kaggle — Edoardo Galli](https://www.kaggle.com/datasets/edoardogalli/stack-overflow-annual-developer-survey-2025) |
| **Stack Overflow** | [Developer Survey](https://survey.stackoverflow.co/) |
| **File** | `survey_results_public.csv` |
| **Rows** | ~52,845 developer responses |
| **Columns** | 172 |
| **Description** | Comprehensive global survey covering developer demographics, technology stacks, AI adoption, compensation, job satisfaction, and more |

### Key Columns

| Column | Description |
|---|---|
| `YearsCode` | Total years coding (including education) — string, convert to numeric |
| `WorkExp` | Professional work experience in years — string, convert to numeric |
| `EdLevel` | Highest level of formal education |
| `Employment` | Employment status (full-time, contractor, student, etc.) |
| `RemoteWork` | Work situation (Remote, In-person, Hybrid) |
| `OrgSize` | Approximate size of employer |
| `DevType` | Developer role(s) — semicolon-separated multi-choice |
| `ICorPM` | Individual contributor or people manager |
| `Country` | Country of residence |
| `ConvertedCompYearly` | Annual compensation converted to USD |
| `Currency` | Day-to-day currency |
| `AISelect` | Whether developer currently uses AI coding tools |
| `AISent` | Favorability stance toward AI tools |
| `AIAcc` | Trust in accuracy of AI tool output |
| `JobSat` | Overall job satisfaction score |
| `JobSatPoints_1..16` | Ranked job satisfaction attribute importance |
| `LanguageHaveWorkedWith` | Languages used in past year — semicolon-separated |
| `LanguageAdmired` | Languages admired — semicolon-separated |
| `LanguageWantToWorkWith` | Languages wanted for next year — semicolon-separated |
| `WebframeHaveWorkedWith` | Web frameworks used — semicolon-separated |
| `DatabaseHaveWorkedWith` | Databases used — semicolon-separated |

---

## Research Questions

| # | Research Question | Task | Target | Models |
|---|---|---|---|---|
| **RQ1** | Can ML predict whether a developer currently uses AI coding tools from their background and tech stack? | Binary Classification | `AISelect` (uses AI vs. does not plan to) | Logistic Regression, Random Forest, XGBoost |
| **RQ2** | Which developer characteristics most accurately predict annual compensation? | Regression | `log1p(ConvertedCompYearly)` (USD respondents) | Ridge (RidgeCV), Random Forest, XGBoost |
| **RQ3** | What distinct developer archetypes emerge from technology preferences and work characteristics? | Clustering | — (unsupervised) | K-Means, Agglomerative Hierarchical, DBSCAN |
| **RQ4** | What professional and AI-adoption factors most drive developer job satisfaction? | Regression + SHAP | `JobSat` (satisfaction score) | Random Forest, XGBoost, Gradient Boosting + SHAP |
| **RQ5** | Do developers' AI sentiments differ significantly across experience levels and employment types? | Statistical Testing | `AISent`, `AIAcc`, `JobSat` by group | ANOVA, Kruskal-Wallis, Mann-Whitney U (Bonferroni) |
| **RQ6** | Which languages have the largest gap between admiration and adoption rates, and how does this vary by role? | Trend Analysis | `LanguageAdmired` vs `LanguageHaveWorkedWith` rates | Polynomial Regression (degree 2 & 3) |
| **RQ7** | Can ML classify a developer's role (IC vs Manager; role category) from their tech stack and work patterns? | Multi-Class Classification | `ICorPM` (binary) + `DevType` simplified (8 classes) | Logistic Regression, Random Forest, LightGBM |

---

## Repository Structure

```
ue-ml-marketing-and-product-perfomance/
├── README.md
├── requirements.txt
└── notebooks/
    ├── RQ1_Classification_AIAdoption.ipynb
    ├── RQ2_Regression_SalaryPrediction.ipynb
    ├── RQ3_Clustering_DeveloperArchetypes.ipynb
    ├── RQ4_FeatureImportance_SHAP_JobSatisfaction.ipynb
    ├── RQ5_Statistical_AISentimentComparison.ipynb
    ├── RQ6_TechAdmiration_AdoptionGap.ipynb
    └── RQ7_Multiclass_DeveloperRoleClassification.ipynb
```

---

## Expected Outputs

All outputs are saved to `/kaggle/working/` when run on Kaggle Notebooks.

| Notebook | PDF Figures | CSV Tables |
|---|---|---|
| **RQ1** | `rq1_class_distribution.pdf`, `rq1_feature_profile_boxplots.pdf`, `rq1_roc_curves.pdf`, `rq1_confusion_matrix_best_model.pdf` | `rq1_model_comparison.csv` |
| **RQ2** | `rq2_salary_distribution.pdf`, `rq2_feature_salary_correlation.pdf`, `rq2_actual_vs_predicted.pdf`, `rq2_residual_plot.pdf` | `rq2_model_comparison.csv` |
| **RQ3** | `rq3_optimal_k_selection.pdf`, `rq3_dendrogram.pdf`, `rq3_pca_cluster_scatter.pdf`, `rq3_clustering_method_comparison.pdf` | `rq3_cluster_profiles.csv` |
| **RQ4** | `rq4_jobsat_distribution.pdf`, `rq4_feature_importance_grouped_bar.pdf`, `rq4_shap_beeswarm.pdf`, `rq4_shap_waterfall_high_low.pdf` | `rq4_feature_importance_comparison.csv`, `rq4_importance_rank_correlation.csv` |
| **RQ5** | `rq5_violin_plots_experience.pdf`, `rq5_violin_plots_employment.pdf`, `rq5_posthoc_pvalue_heatmap.pdf` | `rq5_descriptive_stats.csv`, `rq5_posthoc_pairwise_tests.csv` |
| **RQ6** | `rq6_admiration_adoption_bubble.pdf`, `rq6_gap_score_bar.pdf`, `rq6_gap_by_devtype_heatmap.pdf` | `rq6_technology_rates.csv`, `rq6_gap_by_devtype.csv` |
| **RQ7** | `rq7_role_distribution.pdf`, `rq7_confusion_matrix_normalized.pdf`, `rq7_multiclass_roc_curves.pdf` | `rq7_model_comparison.csv`, `rq7_per_class_metrics.csv` |
| **Total** | **25 PDF figures** | **11 CSV tables** |

---

## How to Run on Kaggle

1. **Upload the dataset** — Add `survey_results_public.csv` as a Kaggle dataset input to your notebook (it will appear under `/kaggle/input/`) or use "Add Input" button to find [Kaggle Dataset — Edoardo Galli](https://www.kaggle.com/datasets/edoardogalli/stack-overflow-annual-developer-survey-2025).

2. **Open each notebook** — Upload the `.ipynb` files from the `notebooks/` folder to a Kaggle Notebook environment. Select Python 3 kernel.

3. **Run all cells** — Click "Run All". Each notebook auto-discovers the CSV, runs all preprocessing and model training, and saves all output files to `/kaggle/working/`. Download them from the Output tab.

> **Note:** RQ4 installs `shap` (`!pip install shap --quiet`) and RQ7 installs `lightgbm` (`!pip install lightgbm --quiet`) at the start of their first cell. All other libraries (pandas, numpy, scikit-learn, xgboost, scipy, statsmodels, seaborn, matplotlib) are pre-installed on Kaggle.

---

## ML Models & Evaluation Metrics

| Task | Models | Metrics |
|---|---|---|
| **Binary Classification (RQ1)** | Logistic Regression, Random Forest, XGBoost | Accuracy, Precision, Recall, F1, ROC-AUC, Cohen's κ |
| **Regression (RQ2, RQ4)** | Ridge, Random Forest, XGBoost, Gradient Boosting | RMSE, MAE, R², Adjusted R², MAPE |
| **Clustering (RQ3)** | K-Means, Agglomerative, DBSCAN | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| **Explainability (RQ4)** | SHAP TreeExplainer | Beeswarm, Waterfall, Spearman ρ rank correlation |
| **Statistical (RQ5)** | ANOVA, Kruskal-Wallis, Mann-Whitney U | F-stat, η², H-stat, ε², Cohen's d, Bonferroni-corrected p |
| **Trend Analysis (RQ6)** | Polynomial Regression (degree 2 & 3) | Admiration rate, Adoption rate, Gap score, R² |
| **Multi-Class Classification (RQ7)** | Logistic Regression, Random Forest, LightGBM | Macro-F1, Weighted-F1, OVR-AUC, Cohen's κ |

---

## Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework applied to a large-scale developer survey:

- **Business Understanding** — Each RQ is framed as a concrete, publication-ready research question.
- **Data Understanding** — EDA figures and descriptive statistics are produced for every RQ.
- **Data Preparation** — Consistent pipeline: median imputation → StandardScaler for numeric features; most-frequent imputation → OneHotEncoder (drop='first') for categorical. Multi-choice (semicolon-separated) columns are parsed into binary multi-hot features.
- **Modeling** — At least 3 algorithms per task; 5-fold cross-validation with `RANDOM_STATE=42` for reproducibility.
- **Evaluation** — Task-appropriate metrics; statistical corrections for multiple comparisons (Bonferroni); effect sizes reported alongside p-values.
- **Deployment** — All figures saved as publication-ready PDFs (300 DPI, tight layout); all tables as structured CSVs.

---

## Dependencies

```
pandas>=2.1.4
numpy>=1.26.2
matplotlib>=3.8.2
seaborn>=0.13.0
scikit-learn>=1.3.2
xgboost>=2.0.3
lightgbm>=4.1.0
statsmodels>=0.14.1
scipy>=1.11.4
shap>=0.44.0
```

All dependencies are pre-installed on Kaggle except `shap` (auto-installed by RQ4) and `lightgbm` (auto-installed by RQ7).
