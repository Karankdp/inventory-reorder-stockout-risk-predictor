# Inventory Reorder & Stockout Risk Predictor

An end-to-end machine learning project for **retail inventory planning**. The system predicts daily product demand and converts that prediction into a data-driven **Reorder Point**, **Safety Stock**, and **stockout-risk flag** for individual SKUs.

It also trains a separate classification model to estimate the probability of a stockout, allowing the business-rule recommendation and direct ML prediction to be compared.

## Project Overview

Retail businesses need to keep enough inventory to satisfy demand without tying up too much capital in excess stock.

This project addresses that problem using two complementary ML approaches:

1. **Regression** — predicts expected daily demand (`Units Sold`).
2. **Classification** — predicts whether a stockout is likely.

The predicted demand is then combined with lead time and safety-stock logic to calculate a reorder point:

```text
Reorder Point = Predicted Demand × Lead Time + Safety Stock
```

The application compares the current inventory level with the calculated reorder point and labels the SKU as:

- **Understocked (High Risk)**
- **Optimal**
- **Overstocked**

## Dataset

The project uses the Kaggle **Retail Store Inventory Forecasting Dataset**.

Dataset:
https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset

The notebook loads the Kaggle dataset through `kagglehub`. It also contains a synthetic-data fallback with the same schema so the workflow can continue if the Kaggle download fails.

The loaded dataset contains approximately **73,100 daily records** and 15 columns.

Important fields include:

- `Date`
- `Store ID`
- `Product ID`
- `Category`
- `Region`
- `Inventory Level`
- `Units Sold`
- `Units Ordered`
- `Demand Forecast`
- `Price`
- `Discount`
- `Weather Condition`
- `Holiday/Promotion`
- `Competitor Pricing`
- `Seasonality`

## Machine Learning Workflow

```text
Dataset
