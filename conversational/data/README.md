<div align="center">

# 📊 Sample Datasets
### *Ready-to-Use Data for Testing*

</div>

This directory contains pre-validated datasets to test the pipeline's capabilities.

---

## 📁 Available Datasets

| File | Description | Shape | Type |
|---|---|---|---|
| `insurance[1].csv` | **Medical Insurance Charges** | (1338, 7) | Regression |
| `heart.csv` | **Heart Disease Classification** | (303, 14) | Classification |
| `advertising.csv` | **Ad Spend vs Sales** | (200, 4) | Regression |
| `Wholesale-Mandi...csv` | **Commodity Prices** | (~5000, 6) | Time Series |

---

## 🎯 Usage

Simply drop your files here! 

1.  **Place File**: Add your `.csv` or `.xlsx` file to this folder.
2.  **Auto-Detect**: The system detects new files on startup/refresh.
3.  **Analyze**: "Refresh Data" in the UI to generate summaries.

> [!TIP]
> **Best Practices for Custom Data:**
> - Clean header names (no special chars like `$` or `#`).
> - Consistent types in columns.
> - Handle missing values (empty or `NaN`) if possible (though the pipeline handles imputation).

---

## 🔄 How to Refresh

Adding data while the server is running?

1.  Go to the **Chat Interface**.
2.  Click the **"Refresh Data"** button.
3.  *Or* restart the server.
