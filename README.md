# Inventory Reorder & Stockout Risk Predictor

> An end-to-end machine learning system for retail inventory planning that predicts product demand, calculates data-driven reorder points and safety stock, and identifies potential stockout risk at SKU level.

## 📌 Project Overview

Retail businesses need to maintain enough inventory to satisfy customer demand while avoiding unnecessary excess stock.

This project builds an ML-driven inventory decision system around two complementary predictive approaches:

1. **Demand Regression** — predicts expected daily product demand (`Units Sold`).
2. **Stockout Classification** — predicts whether a stockout is likely to occur.

The predicted demand is then combined with lead time and recent prediction-error variability to calculate a **Reorder Point** and **Safety Stock**.

The system finally compares current inventory with the reorder point and generates an actionable business flag:

- 🔴 **Understocked (High Risk)**
- 🟢 **Optimal**
- 🟠 **Overstocked**

A Streamlit application is included for one-off SKU audits and instant inventory-risk assessment.

---

## 🎯 Business Problem

Traditional inventory decisions often rely on fixed reorder levels or manual judgement. These approaches may fail when demand changes because of:

- Pricing
- Discounts
- Promotions
- Weather
- Seasonality
- Store and regional differences
- Competitor pricing
- Product/category differences

This project uses machine learning to estimate demand from these operational factors and convert the prediction into an inventory-management recommendation.

The overall business objective is:

> **Predict demand → estimate inventory uncertainty → calculate reorder point → identify stockout risk → support replenishment decisions.**

---

## 🚀 Key Features

- Retail demand prediction using multiple ML regression models
- Direct stockout-risk classification
- Feature engineering from date and operational variables
- Categorical feature encoding
- Feature scaling
- Multicollinearity analysis using VIF
- Multiple regression model comparison
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Classification evaluation using Accuracy, Precision, Recall, F1 and ROC-AUC
- Data-driven safety-stock calculation
- Data-driven reorder-point calculation
- Understocked / Optimal / Overstocked business classification
- Comparison between business-rule risk and direct classifier prediction
- Manual single-SKU audit
- Automatically generated project conclusion
- Saved Joblib model artifact
- Deployable Streamlit application

---

## 📊 Dataset

The project uses the **Retail Store Inventory Forecasting Dataset** from Kaggle.

Dataset:
https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset

The notebook loads the dataset through `kagglehub`.

It also includes a synthetic-data fallback with the same schema so the workflow can still run if the Kaggle download is unavailable.

### Dataset Size

The recorded notebook run loaded:

- **73,100 rows**
- **15 columns**
- Daily retail records covering approximately 2022–2024
- 5 stores
- 20 products
- Multiple categories and regions

### Main Columns

| Column | Description |
|---|---|
| `Date` | Daily record date |
| `Store ID` | Store identifier |
| `Product ID` | Product/SKU identifier |
| `Category` | Product category |
| `Region` | Store region |
| `Inventory Level` | Current inventory |
| `Units Sold` | Actual daily units sold / regression target |
| `Units Ordered` | Restock quantity |
| `Demand Forecast` | Existing demand forecast |
| `Price` | Product price |
| `Discount` | Discount percentage |
| `Weather Condition` | Weather category |
| `Holiday/Promotion` | Promotion/holiday indicator |
| `Competitor Pricing` | Competitor price |
| `Seasonality` | Seasonal category |

---

# 🧠 Machine Learning Workflow

```text
                    Retail Dataset
                         │
                         ▼
               Data Loading & Validation
                         │
                         ▼
                 Exploratory Data Analysis
                         │
                         ▼
              Feature Engineering / Cleaning
                         │
                         ▼
                 Categorical Encoding
                         │
                         ▼
                 Train/Test Split (80/20)
                         │
                         ▼
                    Standard Scaling
                         │
                         ▼
              Multicollinearity / VIF Check
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Demand Regression       Stockout Classification
             │                       │
             ▼                       ▼
       Model Comparison         Model Comparison
             │                       │
             ▼                       ▼
    Hyperparameter Tuning       Hyperparameter Tuning
             │                       │
             ▼                       ▼
     Predicted Demand          Stockout Probability
             │                       │
             └───────────┬───────────┘
                         ▼
              Reorder Point Calculation
                         │
                         ▼
                Business Risk Flag
                         │
                         ▼
                  Streamlit App
```

---

# 🔧 Data Preprocessing

The notebook performs preprocessing and feature engineering before model training.

### Date Features

Date information is converted into useful model features such as:

- Month
- Day of Week
- Weekend indicator

### Lead Time

A per-product lead time is assigned because supplier lead time is not directly available in the raw dataset.

### Leakage Prevention

`Demand Forecast` is excluded from the model inputs because it is itself a forecast of the target (`Units Sold`). Including it would leak information about the target into the model.

### Categorical Encoding

Categorical variables such as:

- Store
- Product
- Category
- Region
- Weather
- Seasonality

are encoded into numerical features.

### Scaling

The encoded feature matrix is standardized using `StandardScaler`.

### Train/Test Split

The dataset is divided into:

- **80% training**
- **20% testing**

with a fixed `random_state=42`.

---

# 🔍 Exploratory Data Analysis

The notebook performs exploratory analysis of:

- Demand distribution
- Raw stockout rate
- Inventory distribution
- Demand by category
- Demand by region
- Price relationships
- Discount relationships
- Promotion effects
- Weather effects
- Seasonal patterns
- Inventory patterns

These analyses are used to understand the factors that may influence demand and inventory risk.

---

# 📐 Multicollinearity Analysis

Before fitting linear models, the project checks feature redundancy using **Variance Inflation Factor (VIF)**.

High VIF values can indicate strong multicollinearity between features and can make linear-model coefficients less reliable.

The VIF analysis is therefore included as part of the model-diagnostics workflow.

---

# 🤖 Regression Models

The project compares multiple approaches for predicting daily demand.

### Models Evaluated

1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression — Degree 2
4. Ridge Regression
5. Lasso Regression
6. ElasticNet
7. KNN Regressor
8. Decision Tree Regressor
9. Random Forest Regressor
10. XGBoost Regressor

### Regression Metrics

Models are evaluated using:

- **R²**
- **RMSE**
- **MAE**

### Hyperparameter Tuning

The notebook uses:

- `GridSearchCV` for Ridge and Lasso
- `RandomizedSearchCV` for Random Forest
- `RandomizedSearchCV` for XGBoost

Cross-validation and parallel processing are used during tuning.

---

# 🎯 Stockout Classification

Regression answers:

> **How much is expected to sell?**

The project also asks a second business question:

> **Is a stockout likely to happen at all?**

A direct classification layer is therefore trained using the available information, including current inventory.

### Classification Models

- Logistic Regression
- KNN Classifier
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

Because stockouts are highly imbalanced, the notebook uses class-balancing techniques such as:

- `class_weight="balanced"`
- `scale_pos_weight` for XGBoost

### Classification Metrics

The classifiers are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

---

# 📦 Reorder Point & Safety Stock

This is the main business-logic layer of the project.

The predicted daily demand is converted into an inventory recommendation.

### Safety Stock

The notebook uses:

```text
Safety Stock = Z × Residual Standard Deviation × √Lead Time
```

where:

- `Z` represents the selected service-level factor
- `Residual Standard Deviation` represents demand-prediction uncertainty
- `Lead Time` represents the supplier lead time

### Reorder Point

```text
Reorder Point =
(Predicted Daily Demand × Lead Time) + Safety Stock
```

This creates a demand-driven inventory threshold instead of relying only on a fixed inventory value.

---

# 🚦 Business Risk Classification

The current inventory level is compared with the calculated reorder point.

The resulting deviation is used to generate one of three business states:

### 🔴 Understocked (High Risk)

Inventory is significantly below the calculated reorder point.

**Business action:** Prioritize replenishment.

### 🟢 Optimal

Inventory is within the acceptable range around the reorder point.

**Business action:** Continue normal inventory management.

### 🟠 Overstocked

Inventory is significantly above the calculated reorder point.

**Business action:** Review excess inventory and potential capital tied up in stock.

---

# 🔄 Rule vs. Classifier

The project intentionally uses two independent risk signals:

### Business-Rule Signal

```text
Current Inventory
        ↓
Compare with Reorder Point
        ↓
Understocked / Optimal / Overstocked
```

### ML Classification Signal

```text
SKU Features + Inventory
        ↓
Stockout Classifier
        ↓
Stockout Probability
```

The notebook compares how often these two approaches agree.

This makes the project more useful from a business perspective because the final decision does not depend on only one model.

---

# 📈 Recorded Model Results

The following results are from the recorded notebook execution.

### Best Regression Model

**Polynomial Regression (Degree 2)**

- R²: **0.345**
- RMSE: **88.0 units**

### Best Direct Classifier

**Tuned XGBoost Classifier**

- ROC-AUC: **0.689**
- Recall: **0.534**

### Other Recorded Business Results

- Raw actual stockout rate: **0.5%**
- Model-flagged high-risk rate: **100.0%**
- Agreement between reorder-point rule and classifier: **27.6%**
- Highest-risk category in the recorded run: **Clothing**
- Recorded overstocked share: **0.0%**
- Recorded understocked share: **100.0%**

> These values are specific to the recorded notebook run. They may change if the dataset, environment, preprocessing, random seed, or model configuration changes.

---

# 🧪 Manual SKU Audit

The notebook includes a manual-entry section for testing a single store-product combination.

The user can enter factors such as:

- Category
- Region
- Weather
- Seasonality
- Promotion
- Current inventory
- Price
- Discount
- Competitor pricing
- Lead time
- Units ordered

The system then calculates:

- Predicted daily demand
- Safety stock
- Reorder point
- Inventory deviation
- Stockout probability
- Final business-rule flag

This makes the model easier to demonstrate to a business user or examiner.

---

# 🌐 Streamlit Application

The project includes a deployable Streamlit application called:

**Inventory Reorder & Stockout Risk Checker**

The application loads the saved model artifact and allows a user to perform an instant inventory-risk check.

### Application Outputs

The app displays:

- Predicted Daily Demand
- Reorder Point
- Deviation from Reorder Point
- Classifier Stockout Probability
- Business-Rule Flag

If the classifier predicts high stockout probability while the business-rule flag does not indicate understocking, the application displays a warning for manual review.

---

# 💾 Saved Model Artifact

The notebook saves the trained model as:

```text
inventory_model.joblib
```

The artifact contains:

- Trained demand model
- Demand model name
- Trained stockout classifier
- Classifier name
- Standard scaler
- Feature columns
- Categorical columns
- Strongest feature
- Residual standard deviation
- Z/service-level value
- Reorder deviation threshold

The Streamlit app loads this artifact for inference.

---

# 📁 Recommended Repository Structure

```text
inventory-reorder-stockout-risk-predictor/
│
├── Inventory_Reorder_Stockout_Risk_Main.ipynb
├── app.py
├── inventory_model.joblib
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    ├── eda.png
    ├── model-results.png
    └── streamlit-app.png
```

### Important

Do not upload the raw Kaggle dataset to GitHub unless its license/terms allow redistribution.

The notebook downloads the dataset through KaggleHub and can generate a synthetic fallback dataset.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/inventory-reorder-stockout-risk-predictor.git
cd inventory-reorder-stockout-risk-predictor
```

Replace `YOUR_USERNAME` with your GitHub username.

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Notebook

Open the notebook using Jupyter:

```bash
jupyter notebook
```

or open it in:

- JupyterLab
- Google Colab
- VS Code

Run the notebook from beginning to end.

The notebook will:

1. Load the dataset
2. Perform EDA
3. Preprocess the data
4. Train regression models
5. Tune selected models
6. Train classification models
7. Calculate reorder points
8. Evaluate stockout risk
9. Save `inventory_model.joblib`
10. Generate the Streamlit application

---

# ▶️ Running the Streamlit App

Make sure these files are in the same directory:

```text
app.py
inventory_model.joblib
```

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📦 Requirements

Main dependencies include:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
xgboost
statsmodels
kagglehub
joblib
streamlit
```

See `requirements.txt` for the complete environment dependency list.

---

# 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy
- SciPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Statistical Analysis

- Statsmodels

### Model Persistence

- Joblib

### Dataset Access

- KaggleHub

### Deployment

- Streamlit

### Development Environment

- Jupyter Notebook
- Google Colab

---

# 💼 Business Applications

This system can be adapted for retail and supply-chain use cases such as:

- Inventory replenishment
- SKU-level demand planning
- Stockout prevention
- Safety-stock planning
- Store-level inventory monitoring
- Promotional demand planning
- Excess-inventory identification
- Replenishment prioritization

A production implementation could connect the model to a retailer's inventory database and continuously score active SKUs.

---

# ⚠️ Limitations

1. The project uses the Kaggle Retail Store Inventory Forecasting Dataset, which is synthetic.
2. The dataset may not represent the behavior of a real retailer.
3. Supplier lead time is not directly available in the raw dataset and is assigned per product.
4. The model is trained on the available historical features and may not generalize to every business.
5. The Streamlit demo uses default Store ID/Product ID and fixed date-derived values for the manual prediction interface.
6. The recorded model performance should not be interpreted as production-level performance.
7. The business-rule classifier and direct ML classifier can disagree, so human review may still be required.
8. A production system would require real-time inventory, supplier, purchase-order and sales data.

---

# 🔮 Future Improvements

Potential future improvements include:

- Connect to a real inventory database
- Use real company sales and purchase-order data
- Add supplier-specific lead times
- Add supplier reliability
- Include carrying cost
- Include stockout cost
- Include lost-sales estimates
- Add real-time inventory updates
- Add SKU-level dashboards
- Add automated model retraining
- Add time-series lag and rolling features
- Compare against dedicated time-series models
- Add automated replenishment recommendations
- Add email/notification alerts for high-risk SKUs
- Deploy the application to a cloud platform

---

# 📚 Project Sections

The notebook is organized into the following major sections:

1. Setup & Load Data
2. Exploratory Data Analysis
3. Preprocessing
4. Train-Test Split & Scaling
5. VIF — Multicollinearity Check
6. Baseline Regression Models
7. Hyperparameter Tuning
8. Evaluation Plots
9. Classification Models — Predicting Stockout Risk Directly
10. Reorder Point & Stockout Risk Detection
11. Manual Entry — Audit a Specific SKU
12. Auto-Generated Conclusion
13. Save Model Artifact
14. Deployable Streamlit App

---

# 👨‍💻 Author

**Karan Patil**

Final Year BSc Computer Science — Artificial Intelligence & Machine Learning

This project was developed as an academic/portfolio machine learning project focused on applying predictive analytics to a practical retail business problem.

---

# 📄 License

This project is intended for educational and portfolio purposes.

The dataset is provided by Kaggle. Refer to the original dataset page for its applicable license and usage terms before redistributing the dataset.

---

## ⭐ If you find this project useful

Feel free to fork the repository, experiment with different models, improve the inventory logic, and adapt the system for real-world retail data.

Dataset
