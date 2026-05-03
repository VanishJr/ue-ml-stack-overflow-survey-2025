# Data-Driven Marketing Intelligence: A Multi-Method Machine Learning Analysis of Campaign Effectiveness and Product Performance

## Project Overview

This project applies seven distinct machine learning methodologies to the **Marketing and Product Performance Dataset** to answer critical business questions about what drives campaign ROI, revenue, customer satisfaction, and optimal discount strategies.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | Marketing and Product Performance Dataset |
| **Source** | [Kaggle — imranalishahh](https://www.kaggle.com/datasets/imranalishahh/marketing-and-product-performance-dataset) |
| **Type** | Synthetic |
| **Columns** | 16 |
| **Description** | Analyze Marketing Campaigns and Product Metrics with Synthetic Data |

### Columns

| Column | Description |
|---|---|
| Campaign_ID | Unique campaign identifier |
| Product_ID | Unique product identifier |
| Budget | Campaign budget in USD ($500–$50,000) |
| Clicks | Number of clicks generated |
| Conversions | Number of successful conversions |
| Revenue_Generated | Total revenue in USD |
| ROI | Return on Investment |
| Customer_ID | Unique customer identifier |
| Subscription_Tier | Basic / Standard / Premium |
| Subscription_Length | Subscription length in months (1–36) |
| Flash_Sale_ID | Flash sale identifier (null = no flash sale) |
| Discount_Level | Discount % offered (10–70%) |
| Units_Sold | Total units sold |
| Bundle_ID | Bundle identifier (null = no bundle) |
| Bundle_Price | Bundle price in USD ($50–$500) |
| Customer_Satisfaction_Post_Refund | Satisfaction score 1–5 |

---

## Problem Statement

Marketing campaigns vary widely in effectiveness. This project identifies which campaign attributes, customer subscription behaviors, and discount strategies drive ROI, revenue, and customer satisfaction — enabling data-driven marketing budget allocation.

---

## Research Questions

| # | Research Question | Task | Target |
|---|---|---|---|
| RQ1 | Can ML predict whether a campaign achieves above-median ROI? | Binary Classification | High_ROI |
| RQ2 | Which features best predict Revenue_Generated? | Regression | Revenue_Generated |
| RQ3 | What natural campaign-customer performance clusters exist? | Clustering | — |
| RQ4 | What drives customer satisfaction post-refund? | Regression + SHAP XAI | Customer_Satisfaction_Post_Refund |
| RQ5 | Do subscription tiers differ significantly in ROI and revenue? | Statistical Testing | ROI / Revenue / Units_Sold |
| RQ6 | What discount range maximizes ROI while maintaining satisfaction? | Polynomial Optimization | ROI & Satisfaction |
| RQ7 | Can ML classify a customer's subscription tier from behavior? | Multi-Class Classification | Subscription_Tier |

---

## Repository Structure

```
notebooks/
  RQ1_Classification_HighROI.ipynb
  RQ2_Regression_RevenuePrediction.ipynb
  RQ3_Clustering_CampaignSegmentation.ipynb
  RQ4_FeatureImportance_SHAP_Satisfaction.ipynb
  RQ5_Statistical_SubscriptionTierComparison.ipynb
  RQ6_DiscountOptimization.ipynb
  RQ7_Multiclass_SubscriptionTierClassification.ipynb
README.md
requirements.txt
```

---

## ML Models Used

| RQ | Models |
|---|---|
| RQ1 | Logistic Regression, Random Forest, XGBoost |
| RQ2 | Ridge (RidgeCV), Random Forest Regressor, XGBoost Regressor |
| RQ3 | K-Means, Agglomerative Hierarchical (Ward), DBSCAN |
| RQ4 | Random Forest, XGBoost, Gradient Boosting + SHAP TreeExplainer |
| RQ5 | ANOVA + Tukey HSD, Kruskal-Wallis + Mann-Whitney U + Bonferroni |
| RQ6 | Polynomial Regression (degree 3), Binned Analysis |
| RQ7 | Multinomial Logistic Regression, Random Forest, LightGBM |

---

## Evaluation Metrics

| RQ | Metrics |
|---|---|
| RQ1 | Accuracy, Precision, Recall, F1, ROC-AUC, Cohen's Kappa |
| RQ2 | RMSE, MAE, R², Adjusted R², MAPE |
| RQ3 | Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index |
| RQ4 | R², RMSE, Spearman rank correlation of importance rankings |
| RQ5 | F/H statistic, p-values (Bonferroni-adjusted), Cohen's d, η²/ε² |
| RQ6 | R² per segment, polynomial peak (argmax), RMSE |
| RQ7 | Macro F1, Weighted F1, OVR ROC-AUC, Cohen's Kappa, per-class metrics |

---

## How to Run on Kaggle

### Step 1 — Upload Dataset
1. Download the Excel file from the [Kaggle dataset page](https://www.kaggle.com/datasets/imranalishahh/marketing-and-product-performance-dataset)
2. Go to [Kaggle — Create New Dataset](https://www.kaggle.com/datasets/new) and upload it
3. Name the dataset (e.g. `marketing-product-performance`)

### Step 2 — Create a Notebook
1. Go to [Kaggle — Create New Notebook](https://www.kaggle.com/code)
2. Click **Add Data** → search for your dataset → add it
3. Upload the `.ipynb` file for the research question you want to run

### Step 3 — Run
1. Click **Run All** (Session → Run All)
2. After completion, go to the **Output** tab to download PDF figures and CSV tables
3. **If the Output tab looks empty or shows only system files (`.virtual_documents`, `__notebook_source__.ipynb`), click the refresh icon in the Output panel** — Kaggle does not auto-refresh the file list after a run

### Notes
- Each notebook auto-discovers the Excel file from `/kaggle/input/` — no hardcoded paths needed
- RQ4 runs `!pip install shap --quiet` automatically
- RQ7 runs `!pip install lightgbm --quiet` automatically
- All other libraries are pre-installed on Kaggle

---

## Expected Outputs

| Notebook | PDF Figures | CSV Tables |
|---|---|---|
| RQ1 | rq1_roc_curves.pdf, rq1_confusion_matrix_best_model.pdf, rq1_feature_boxplots.pdf, rq1_correlation_heatmap.pdf | rq1_model_comparison.csv |
| RQ2 | rq2_revenue_distribution.pdf, rq2_feature_revenue_correlation.pdf, rq2_actual_vs_predicted.pdf, rq2_residual_plot.pdf | rq2_model_comparison.csv |
| RQ3 | rq3_optimal_k_selection.pdf, rq3_dendrogram.pdf, rq3_pca_cluster_scatter.pdf, rq3_clustering_method_comparison.pdf | rq3_cluster_profiles.csv |
| RQ4 | rq4_shap_beeswarm.pdf, rq4_shap_waterfall_high_low_satisfaction.pdf, rq4_feature_importance_grouped_bar.pdf | rq4_feature_importance_comparison.csv, rq4_importance_rank_correlation.csv |
| RQ5 | rq5_violin_plots.pdf, rq5_posthoc_pvalue_heatmap.pdf | rq5_descriptive_stats.csv, rq5_posthoc_pairwise_tests.csv |
| RQ6 | rq6_discount_roi_polynomial.pdf, rq6_discount_satisfaction_scatter.pdf, rq6_optimal_discount_by_tier.pdf | rq6_discount_performance_summary.csv, rq6_optimal_discount_ranges.csv |
| RQ7 | rq7_confusion_matrix_normalized.pdf, rq7_multiclass_roc_curves.pdf, rq7_metrics_by_tier_boxplots.pdf | rq7_per_class_metrics.csv, rq7_model_comparison.csv |

**Total: 23 PDF figures + 11 CSV tables**

---

## Dependencies

See `requirements.txt`. All libraries are pre-installed on Kaggle kernels (Python 3.10) except `shap` (auto-installed in RQ4) and `lightgbm` (auto-installed in RQ7).

---

## Methodology

This project follows the **CRISP-DM** framework:

1. **Business Understanding** — 7 research questions spanning core ML paradigms
2. **Data Understanding** — EDA, correlation analysis, class balance checks
3. **Data Preparation** — Feature engineering (Has_Flash_Sale, Has_Bundle, Conversion_Rate), imputation, scaling, encoding; leakage-safe column dropping
4. **Modeling** — 2–3 algorithms per RQ with cross-validation (StratifiedKFold / KFold, k=5)
5. **Evaluation** — Task-appropriate metrics, statistical significance tests, SHAP explainability
6. **Reporting** — Publication-ready PDF figures, structured CSV tables, written conclusions per notebook