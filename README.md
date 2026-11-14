# Household Energy Consumption Prediction

**DS605 — Machine Learning Project | MSc Data Science**

Predict hourly household electricity consumption using indoor temperature, humidity, outdoor weather measurements, and machine learning.

**[Try the Streamlit web application](https://machine-learning-project-ds-605-fmf8iap32v5fn8hsgy3kphrere.streamlit.app/)** · **[GitHub repository](https://github.com/Paresh-ds/Machine-Learning-Project-DS-605)** · **[Dataset source (UCI)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)**

## Project overview

This project studies how household conditions relate to electricity consumption. We use the **UCI Appliances Energy Prediction** dataset, which contains measurements recorded inside a house and weather data collected over approximately four and a half months in 2016.

The goal is both **prediction** and **learning the machine learning workflow**: data preparation, exploratory data analysis (EDA), outlier investigation, statistical testing, feature engineering, principal component analysis (PCA), regression algorithms, time-aware evaluation, hyperparameter tuning, and deployment.

Our final deployed model predicts **total hourly energy consumption** in watt-hours (Wh) from 24 temperature, humidity, and weather inputs. The web application automatically applies feature scaling and PCA before making a Ridge Regression prediction.

> **Important:** The application provides a statistical estimate, not a direct electricity-meter reading. The selected model achieved a test R² of approximately **0.14**, so predictions can have substantial error.

## Problem statement and objectives

**Problem:** Estimate household electricity consumption based on indoor environmental conditions and outdoor weather measurements.

**Objectives:**

1. Understand and clean real-world sensor observations.
2. Examine energy-consumption patterns across hours, days, and weather conditions.
3. Identify unusual consumption and distinguish possible anomalies from plausible high-use periods.
4. Engineer features and reduce dimensionality using PCA.
5. Compare regression algorithms with chronological training/testing and time-series cross-validation.
6. Tune promising algorithms and compare MAE, RMSE, and R².
7. Deploy a user-friendly prediction application using Streamlit.

## Dataset

**Source:** [UCI Machine Learning Repository — Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

| Property | Description |
| --- | --- |
| Original sampling frequency | Every 10 minutes |
| Original observations | 19,735 |
| Collection period | January 11 – May 27, 2016 |
| Project analysis frequency | Hourly (aggregated from 10-minute observations) |
| Hourly observations | **3,289** |
| Prediction target | `Total_Energy = Appliances + lights` |
| Target unit | **Wh per hour** |
| Final deployment inputs | **24** environmental/weather variables |
| Final modeling split | Chronological: 2,631 training / 658 testing rows |

### Model input variables

The deployed model uses these **24 original variables**:

- **9 indoor temperatures:** `T1`–`T9`
- **9 indoor relative humidities:** `RH_1`–`RH_9`
- **6 outdoor/weather variables:** `T_out`, `RH_out`, `Press_mm_hg`, `Windspeed`, `Visibility`, `Tdewpoint`

`date` is used to preserve chronological order, but it is **not** a prediction input. Calendar features were tested in earlier experiments but are **not used in the final deployed pipeline**.

## Project workflow

### 1. Data preparation and EDA

- Loaded and inspected the original 10-minute sensor dataset.
- Examined column types, missing values, distributions, and descriptive statistics.
- Combined `Appliances` and `lights` into the target `Total_Energy`.
- Aggregated the observations to **1-hour intervals** for analysis and modeling.
- Studied energy usage over time and its relationship with temperature and humidity.
- Generated visual analyses and saved plots in `EDA_graph/`.

### 2. Outlier investigation and data mining

- Used an interquartile-range (IQR) rule to flag unusually high hourly energy observations.
- Hourly `Total_Energy`: mean **608.88 Wh**, median **400 Wh**, maximum **3,740 Wh**.
- Upper IQR threshold: **1,310 Wh**; **310** hourly rows exceeded it.
- Examined high-use cases by time of day, weekend status, and weather conditions.
- Conducted hypothesis tests on proportions of flagged high-energy observations.
- Distinguished **flagged observations** from **confirmed errors**: high electricity usage is not automatically an invalid record.

### 3. Feature engineering

Experimental features included:

- Cyclical hour representation: `hour_sin` and `hour_cos`.
- Weekend and afternoon peak indicators.
- Month indicators.
- Composite first-floor and second-floor temperature/humidity features.
- Temperature change and simultaneous warming indicators.

These were investigated in the exploratory/model-comparison workflow. The production application instead accepts the **24 original environmental variables** directly.

### 4. Principal component analysis (PCA)

PCA was used to reduce correlated sensor/weather variables to a smaller number of components.

| Dataset | PCA components | Calendar features in later PCA-only comparison |
| --- | ---: | ---: |
| **Model A** | 11 | 0 |
| **Model B** | 6 | 0 |

Earlier experiments also considered **Model B with calendar variables** and **Kernel PCA plus K-Means cluster features**. These are research experiments, not the deployed configuration.

### 5. Machine learning algorithms

The project explored:

- K-Nearest Neighbors (KNN) Regression
- Linear Regression and polynomial feature experiments
- Ridge and Lasso Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Bagging Regression
- Random Forest Regression
- Gradient Boosting Regression

For evaluation we used a **chronological 80/20 holdout split**, and for tuning we used **5-fold `TimeSeriesSplit`** on training data. Model selection was based primarily on cross-validation RMSE; held-out test results are reported for reference.

## Model comparison

Selected **tuned** results from the PCA-only experiments:

| Algorithm | PCA model | CV RMSE (Wh) | Test RMSE (Wh) | Test R² |
| --- | --- | ---: | ---: | ---: |
| Ridge Regression | A — 11 PCs | 463.113 | **395.387** | **0.140** |
| SVR (RBF) | B — 6 PCs | **462.792** | 396.485 | 0.135 |
| Bagging Regression | A — 11 PCs | 464.677 | 405.113 | 0.097 |
| Random Forest Regression | A — 11 PCs | 464.519 | 408.410 | 0.082 |
| Gradient Boosting Regression | A — 11 PCs | 464.400 | 414.148 | 0.056 |

**Selected deployed model:** **Ridge Regression**, with **11 PCA components** and `alpha=100` (equivalent to polynomial degree 1 in the tuning experiment).

- Test **MAE:** 270.970 Wh
- Test **RMSE:** 395.387 Wh
- Test **R²:** 0.140

**Interpretation:** SVR Model B had the lowest reported cross-validation RMSE among the tuned configurations above, while Ridge Model A had the lowest observed test RMSE. Their scores are very close, and neither result proves superiority on future households or time periods.

## Deployment pipeline

The production model is trained in `scripts/train_deployment_model.py` and saved as `models/energy_pipeline.joblib`.

```text
24 original sensor/weather inputs
              |
              v
     StandardScaler
              |
              v
      PCA (11 components)
              |
              v
     StandardScaler
              |
              v
   Ridge Regression (alpha=100)
              |
              v
 Predicted hourly energy (Wh)
```

All preprocessing is fitted **on the training portion only**, then included in the saved pipeline so the web app uses the same transformations for new inputs.

## Project structure

```text
Machine-Learning-Project-DS-605/
├── EDA_graph/                     # EDA figures
├── algorithm/                     # Regression experiment notebooks
│   ├── KNN_Regression.ipynb
│   └── Linear_Regression.ipynb
├── app/
│   └── streamlit_app.py           # Streamlit frontend
├── data/                          # Raw and processed datasets
│   ├── energy_1hour_fe_ready.csv
│   ├── energy_model_A_pca11.csv
│   ├── energy_model_B_pca6_calendar.csv
│   └── energy_model_B_pca6_only.csv
├── house_map/                     # Household sensor/location references
├── models/
│   └── energy_pipeline.joblib     # Serialized prediction pipeline
├── notebook/                      # EDA, data mining, FE, PCA notebooks
├── scripts/
│   └── train_deployment_model.py  # Production model training
├── txt_file/                      # Project progress notes
├── requirements.txt
└── README.md
```

*The tree highlights relevant files; it is not an exhaustive directory listing.*

## Run the project locally

### Prerequisites

- Python **3.13**, matching the environment used to train the saved model
- Git (optional, to clone the repository)

### 1. Clone the repository

```bash
git clone https://github.com/Paresh-ds/Machine-Learning-Project-DS-605.git
cd Machine-Learning-Project-DS-605
```

### 2. Create a virtual environment and install packages

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### 4. Retrain the deployment pipeline (optional)

```bash
python scripts/train_deployment_model.py
```

This reads `data/energy_1hour_fe_ready.csv`, performs a chronological train/test split, evaluates the model, and saves the pipeline to `models/energy_pipeline.joblib`.

> **Security note:** Only load `.joblib` model files from trusted sources. Python model serialization can execute code when loaded. Use compatible Python and scikit-learn versions.

## Web app usage

1. Enter **9 indoor temperature** readings and **9 indoor humidity** readings.
2. Enter **6 outdoor weather** measurements.
3. Click **Predict Energy Consumption**.
4. View the predicted hourly total in **Wh** and **kWh**.

**Live app:** https://machine-learning-project-ds-605-fmf8iap32v5fn8hsgy3kphrere.streamlit.app/

## Limitations and future work

- Test R² near **0.14** indicates limited predictive accuracy; the model should not be used for billing, safety, or automated high-stakes decisions.
- The dataset covers one specific house and a limited time period; performance on other houses, climates, and seasons is unknown.
- High-energy outliers may reflect legitimate use, not measurement errors.
- Tests and repeated examination of the same holdout can create optimistic selection bias; a fresh untouched evaluation period would improve final assessment.
- Potential next steps: additional time-series features (lags and rolling statistics), time-aware validation on later periods, improved feature selection, stronger models, input-range checks, and model interpretability.

## Contributors

- **[Paresh-ds](https://github.com/Paresh-ds)** — project contributor
- **[jayeshedaoo](https://github.com/jayeshedaoo)** — project contributor

Both contributors are credited on the GitHub repository. Roles can be detailed further as the project evolves.

## Acknowledgment

- **Dataset:** UCI Machine Learning Repository, *Appliances Energy Prediction*.
- **Tools:** Python, Jupyter Notebook, pandas, NumPy, scikit-learn, joblib, Streamlit, and GitHub.

---

*Academic project for DS605, MSc Data Science.*
