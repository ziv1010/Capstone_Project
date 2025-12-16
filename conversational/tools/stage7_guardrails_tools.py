"""
Stage 7 Tools: Guardrails Validation

Tools for validating model predictions using statistical tests:
- Correlation analysis
- Propensity score analysis
- Inverse propensity weighting
- Residual analysis
- Feature importance validation
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE7_OUT_DIR, STAGE3B_OUT_DIR,
    DataPassingManager, logger
)


# ============================================================================
# DATA LOADING TOOLS
# ============================================================================

@tool
def load_predictions_for_validation(plan_id: str) -> str:
    """
    Load model predictions and prepared data for validation.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Summary of loaded data for validation
    """
    try:
        # Load predictions
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        if not predictions_path.exists():
            return f"Predictions not found at: {predictions_path}"

        df = pd.read_parquet(predictions_path)

        # Load prepared data
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        prepared_df = None
        if prepared_path.exists():
            prepared_df = pd.read_parquet(prepared_path)

        result = ["=== PREDICTIONS LOADED FOR VALIDATION ===\n"]
        result.append(f"Predictions shape: {df.shape}")
        result.append(f"Columns: {list(df.columns)}")

        # Find prediction/actual columns
        pred_cols = [c for c in df.columns if 'predict' in c.lower() or 'forecast' in c.lower()]
        actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'target' in c.lower()]

        if pred_cols and actual_cols:
            result.append(f"\nPrediction column: {pred_cols[0]}")
            result.append(f"Actual column: {actual_cols[0]}")
            result.append(f"Sample predictions: {df[pred_cols[0]].head(5).tolist()}")
            result.append(f"Sample actuals: {df[actual_cols[0]].head(5).tolist()}")

        if prepared_df is not None:
            result.append(f"\nPrepared data shape: {prepared_df.shape}")
            result.append(f"Feature columns: {[c for c in prepared_df.columns if c not in ['target', 'output', 'predicted', 'actual']][:10]}")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading predictions: {e}"


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

@tool
def run_correlation_analysis(plan_id: str) -> str:
    """
    Run correlation analysis between features, predictions, and actuals.
    
    Checks:
    1. Feature-target correlations
    2. Feature-prediction correlations
    3. Prediction-target correlation
    4. Flags if correlations are suspiciously different

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Correlation analysis results with pass/fail assessment
    """
    try:
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"

        if not predictions_path.exists():
            return "Predictions not found for correlation analysis"

        df = pd.read_parquet(predictions_path)
        
        # Find columns
        pred_col = next((c for c in df.columns if 'predict' in c.lower()), None)
        actual_col = next((c for c in df.columns if 'actual' in c.lower()), None)

        if not pred_col or not actual_col:
            return "Could not identify prediction and actual columns"

        result = ["=== CORRELATION ANALYSIS ===\n"]
        
        # 1. Prediction-Actual correlation
        pred_actual_corr = df[pred_col].corr(df[actual_col])
        result.append(f"Prediction-Actual Correlation: {pred_actual_corr:.4f}")
        
        # Assess correlation quality with detailed reasons
        if pred_actual_corr >= 0.8:
            result.append("  ✅ PASS: Strong positive correlation")
            corr_status = "PASS"
            reason = f"Correlation of {pred_actual_corr:.4f} is >= 0.8 threshold, indicating strong linear relationship between predictions and actual values."
        elif pred_actual_corr >= 0.5:
            result.append("  ⚠️ WARNING: Moderate correlation - review model")
            corr_status = "WARNING"
            reason = f"Correlation of {pred_actual_corr:.4f} is between 0.5-0.8. The model captures some patterns but may miss important relationships. Consider feature engineering or alternative models."
        else:
            result.append("  ❌ FAIL: Weak correlation - model may not be capturing patterns")
            corr_status = "FAIL"
            reason = f"Correlation of {pred_actual_corr:.4f} is below 0.5 threshold. The model is not effectively predicting the target. Investigate data quality, feature selection, or model choice."

        # 2. Feature correlations if prepared data exists
        if prepared_path.exists():
            prepared_df = pd.read_parquet(prepared_path)
            numeric_cols = prepared_df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = next((c for c in ['target', 'output'] if c in numeric_cols), None)
            
            if target_col:
                feature_cols = [c for c in numeric_cols if c != target_col][:10]  # Top 10 features
                result.append(f"\nFeature-Target Correlations (top {len(feature_cols)}):")
                
                for col in feature_cols:
                    try:
                        corr = prepared_df[col].corr(prepared_df[target_col])
                        if not np.isnan(corr):
                            result.append(f"  {col}: {corr:.4f}")
                    except:
                        pass

        # Save correlation results
        corr_data = {
            "test_name": "correlation_analysis",
            "prediction_actual_correlation": float(pred_actual_corr),
            "threshold_pass": 0.8,
            "threshold_warning": 0.5,
            "status": corr_status,
            "reason": reason,
            "interpretation": "Measures linear relationship between predictions and actuals. Higher correlation indicates better model fit."
        }
        
        output_path = STAGE7_OUT_DIR / f"{plan_id}_correlation.json"
        DataPassingManager.save_artifact(corr_data, STAGE7_OUT_DIR, f"{plan_id}_correlation.json")
        
        result.append(f"\nResults saved to: {output_path}")
        return "\n".join(result)

    except Exception as e:
        return f"Error in correlation analysis: {e}"


# ============================================================================
# PROPENSITY SCORE ANALYSIS
# ============================================================================

@tool
def run_propensity_score_analysis(plan_id: str) -> str:
    """
    Run propensity score analysis to check for selection bias.
    
    For regression/forecasting, this checks if certain feature combinations
    are over-represented in predictions vs actuals (covariate balance).

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Propensity score analysis results
    """
    try:
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"

        if not predictions_path.exists() or not prepared_path.exists():
            return "Required data not found for propensity score analysis"

        df = pd.read_parquet(predictions_path)
        prepared_df = pd.read_parquet(prepared_path)

        result = ["=== PROPENSITY SCORE ANALYSIS ===\n"]
        result.append("Checking for covariate balance and selection bias...\n")

        # Find prediction column
        pred_col = next((c for c in df.columns if 'predict' in c.lower()), None)
        actual_col = next((c for c in df.columns if 'actual' in c.lower()), None)

        if not pred_col or not actual_col:
            return "Could not identify prediction and actual columns"

        # Calculate residuals
        residuals = df[actual_col] - df[pred_col]
        
        # Create binary treatment: is residual above median?
        median_residual = residuals.median()
        treatment = (residuals > median_residual).astype(int)
        
        # Estimate propensity scores using logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Get numeric features from prepared data
        numeric_cols = prepared_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ['target', 'output', 'predicted', 'actual']][:15]
        
        if len(feature_cols) < 2:
            return "Not enough features for propensity score analysis"

        # Align data
        n_samples = min(len(prepared_df), len(treatment))
        X = prepared_df[feature_cols].iloc[:n_samples].fillna(0)
        y = treatment[:n_samples].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit propensity model
        ps_model = LogisticRegression(max_iter=500, random_state=42)
        ps_model.fit(X_scaled, y)
        propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
        
        # Analyze propensity score distribution
        result.append(f"Propensity Score Statistics:")
        result.append(f"  Mean: {propensity_scores.mean():.4f}")
        result.append(f"  Std: {propensity_scores.std():.4f}")
        result.append(f"  Min: {propensity_scores.min():.4f}")
        result.append(f"  Max: {propensity_scores.max():.4f}")
        
        # Check for extreme propensity scores (indicates imbalance)
        extreme_low = (propensity_scores < 0.1).sum() / len(propensity_scores)
        extreme_high = (propensity_scores > 0.9).sum() / len(propensity_scores)
        
        result.append(f"\nExtreme Propensity Scores:")
        result.append(f"  < 0.1: {extreme_low*100:.1f}%")
        result.append(f"  > 0.9: {extreme_high*100:.1f}%")
        
        # Assess with detailed reasons
        extreme_total = extreme_low + extreme_high
        if extreme_total < 0.2:
            result.append("\n✅ PASS: Good covariate balance - propensity scores well distributed")
            status = "PASS"
            reason = f"Only {extreme_total*100:.1f}% of samples have extreme propensity scores (< 0.1 or > 0.9), which is below the 20% threshold. This indicates good covariate balance and no significant selection bias."
        elif extreme_total < 0.4:
            result.append("\n⚠️ WARNING: Some covariate imbalance detected")
            status = "WARNING"
            reason = f"{extreme_total*100:.1f}% of samples have extreme propensity scores, which is between 20-40%. Some feature combinations may be over-represented in prediction errors. Review feature distributions."
        else:
            result.append("\n❌ FAIL: Significant covariate imbalance - model may have selection bias")
            status = "FAIL"
            reason = f"{extreme_total*100:.1f}% of samples have extreme propensity scores, exceeding the 40% threshold. The model shows significant selection bias - certain data subgroups are poorly handled. Consider rebalancing training data or stratified sampling."

        # Save results
        ps_data = {
            "test_name": "propensity_score_analysis",
            "propensity_mean": float(propensity_scores.mean()),
            "propensity_std": float(propensity_scores.std()),
            "extreme_low_pct": float(extreme_low),
            "extreme_high_pct": float(extreme_high),
            "extreme_total_pct": float(extreme_total),
            "threshold_pass": 0.2,
            "threshold_warning": 0.4,
            "status": status,
            "reason": reason,
            "interpretation": "Propensity scores measure the probability of belonging to high/low error groups. Extreme scores indicate certain feature combinations are over-represented in errors."
        }
        DataPassingManager.save_artifact(ps_data, STAGE7_OUT_DIR, f"{plan_id}_propensity.json")

        return "\n".join(result)

    except Exception as e:
        return f"Error in propensity score analysis: {e}"


# ============================================================================
# INVERSE PROPENSITY WEIGHTING
# ============================================================================

@tool
def run_inverse_propensity_weighting(plan_id: str) -> str:
    """
    Run Inverse Propensity Weighting (IPW) validation.
    
    IPW reweights predictions to check if the model performs
    consistently across different data subgroups.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        IPW validation results
    """
    try:
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"

        if not predictions_path.exists() or not prepared_path.exists():
            return "Required data not found for IPW analysis"

        df = pd.read_parquet(predictions_path)
        prepared_df = pd.read_parquet(prepared_path)

        result = ["=== INVERSE PROPENSITY WEIGHTING (IPW) VALIDATION ===\n"]

        # Find columns
        pred_col = next((c for c in df.columns if 'predict' in c.lower()), None)
        actual_col = next((c for c in df.columns if 'actual' in c.lower()), None)

        if not pred_col or not actual_col:
            return "Could not identify prediction and actual columns"

        # Calculate residuals and squared errors
        residuals = df[actual_col] - df[pred_col]
        squared_errors = residuals ** 2
        
        # Create quintile-based groups
        n_samples = len(df)
        quintile_size = n_samples // 5
        
        result.append("Performance by Prediction Quintile:")
        result.append("-" * 40)
        
        sorted_indices = df[pred_col].argsort()
        quintile_maes = []
        
        for i in range(5):
            start_idx = i * quintile_size
            end_idx = (i + 1) * quintile_size if i < 4 else n_samples
            quintile_indices = sorted_indices[start_idx:end_idx]
            
            quintile_mae = residuals.iloc[quintile_indices].abs().mean()
            quintile_maes.append(quintile_mae)
            
            result.append(f"  Q{i+1}: MAE = {quintile_mae:.4f}")
        
        # Check consistency across quintiles
        mae_range = max(quintile_maes) - min(quintile_maes)
        mae_cv = np.std(quintile_maes) / np.mean(quintile_maes)
        
        result.append(f"\nMAE Range across quintiles: {mae_range:.4f}")
        result.append(f"MAE Coefficient of Variation: {mae_cv:.4f}")
        
        # IPW Assessment with detailed reasons
        if mae_cv < 0.15:
            result.append("\n✅ PASS: Model performs consistently across prediction ranges")
            status = "PASS"
            reason = f"MAE coefficient of variation is {mae_cv:.4f}, below the 0.15 threshold. The model's error is consistent across all prediction quintiles, indicating stable performance regardless of prediction magnitude."
        elif mae_cv < 0.30:
            result.append("\n⚠️ WARNING: Some inconsistency in model performance across ranges")
            status = "WARNING"
            reason = f"MAE CV of {mae_cv:.4f} is between 0.15-0.30. The model shows moderate variation in error across prediction quintiles (range: {mae_range:.4f}). Consider investigating specific quintiles with higher errors."
        else:
            result.append("\n❌ FAIL: Model performance varies significantly across prediction ranges")
            status = "FAIL"
            reason = f"MAE CV of {mae_cv:.4f} exceeds 0.30 threshold. Error varies significantly across quintiles (range: {mae_range:.4f}). The model may be overfitting to certain prediction ranges or missing patterns in others."
        
        # IPW-adjusted overall MAE
        weights = 1.0 / (np.abs(residuals) + 0.01)  # Avoid division by zero
        weights = weights / weights.sum()
        ipw_mae = (residuals.abs() * weights * len(residuals)).sum()
        
        result.append(f"\nIPW-adjusted MAE: {ipw_mae:.4f}")
        result.append(f"Standard MAE: {residuals.abs().mean():.4f}")

        # Save results
        ipw_data = {
            "test_name": "inverse_propensity_weighting",
            "quintile_maes": [float(m) for m in quintile_maes],
            "mae_range": float(mae_range),
            "mae_cv": float(mae_cv),
            "threshold_pass": 0.15,
            "threshold_warning": 0.30,
            "ipw_adjusted_mae": float(ipw_mae),
            "standard_mae": float(residuals.abs().mean()),
            "status": status,
            "reason": reason,
            "interpretation": "IPW validates that the model performs consistently across different prediction ranges. High variation suggests the model may be unreliable for certain value ranges."
        }
        DataPassingManager.save_artifact(ipw_data, STAGE7_OUT_DIR, f"{plan_id}_ipw.json")

        return "\n".join(result)

    except Exception as e:
        return f"Error in IPW analysis: {e}"


# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================

@tool
def run_residual_analysis(plan_id: str) -> str:
    """
    Run comprehensive residual analysis.
    
    Checks:
    1. Residual normality
    2. Heteroscedasticity
    3. Autocorrelation (for time series)
    4. Outlier detection

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Residual analysis results
    """
    try:
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"

        if not predictions_path.exists():
            return "Predictions not found for residual analysis"

        df = pd.read_parquet(predictions_path)

        pred_col = next((c for c in df.columns if 'predict' in c.lower()), None)
        actual_col = next((c for c in df.columns if 'actual' in c.lower()), None)

        if not pred_col or not actual_col:
            return "Could not identify prediction and actual columns"

        result = ["=== RESIDUAL ANALYSIS ===\n"]
        
        residuals = df[actual_col] - df[pred_col]
        
        # Basic statistics
        result.append("Residual Statistics:")
        result.append(f"  Mean: {residuals.mean():.4f}")
        result.append(f"  Std: {residuals.std():.4f}")
        result.append(f"  Skewness: {residuals.skew():.4f}")
        result.append(f"  Kurtosis: {residuals.kurtosis():.4f}")
        
        # 1. Check for zero mean (unbiasedness)
        mean_residual = residuals.mean()
        if abs(mean_residual) < 0.05 * residuals.std():
            result.append("\n✅ Residual mean near zero: Predictions are unbiased")
            bias_status = "PASS"
        else:
            result.append(f"\n⚠️ Non-zero residual mean ({mean_residual:.4f}): Systematic bias detected")
            bias_status = "WARNING"
        
        # 2. Check for normality (using skewness and kurtosis)
        skew = residuals.skew()
        kurt = residuals.kurtosis()
        if abs(skew) < 1 and abs(kurt) < 3:
            result.append("✅ Residuals approximately normal")
            normality_status = "PASS"
        else:
            result.append(f"⚠️ Non-normal residuals (skew={skew:.2f}, kurtosis={kurt:.2f})")
            normality_status = "WARNING"
        
        # 3. Outlier detection (residuals > 3 std)
        std_residuals = (residuals - residuals.mean()) / residuals.std()
        outliers = (np.abs(std_residuals) > 3).sum()
        outlier_pct = outliers / len(residuals) * 100
        
        result.append(f"\nOutliers (|z| > 3): {outliers} ({outlier_pct:.1f}%)")
        if outlier_pct < 1:
            result.append("✅ Acceptable outlier level")
            outlier_status = "PASS"
        elif outlier_pct < 5:
            result.append("⚠️ Moderate outlier presence")
            outlier_status = "WARNING"
        else:
            result.append("❌ High outlier presence - investigate data quality")
            outlier_status = "FAIL"
        
        # Overall status with detailed reason
        statuses = [bias_status, normality_status, outlier_status]
        reasons = []
        if all(s == "PASS" for s in statuses):
            overall_status = "PASS"
            overall_reason = "All residual checks passed: unbiased predictions, approximately normal distribution, and acceptable outlier levels."
        elif any(s == "FAIL" for s in statuses):
            overall_status = "FAIL"
            if outlier_status == "FAIL":
                reasons.append(f"High outlier presence ({outlier_pct:.1f}% > 5%)")
            overall_reason = "Residual analysis failed. Issues: " + "; ".join(reasons) if reasons else "One or more sub-checks failed."
        else:
            overall_status = "WARNING"
            if bias_status == "WARNING":
                reasons.append(f"systematic bias detected (mean={mean_residual:.4f})")
            if normality_status == "WARNING":
                reasons.append(f"non-normal distribution (skew={skew:.2f}, kurtosis={kurt:.2f})")
            if outlier_status == "WARNING":
                reasons.append(f"moderate outliers ({outlier_pct:.1f}%)")
            overall_reason = "Residual analysis has warnings: " + "; ".join(reasons)
        
        result.append(f"\nOverall Residual Analysis: {overall_status}")

        # Save results
        res_data = {
            "test_name": "residual_analysis",
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "skewness": float(skew),
            "kurtosis": float(kurt),
            "outlier_count": int(outliers),
            "outlier_pct": float(outlier_pct),
            "bias_status": bias_status,
            "bias_reason": f"Residual mean of {mean_residual:.4f} is {'< 5% of std (unbiased)' if bias_status == 'PASS' else 'non-zero indicating systematic prediction bias'}",
            "normality_status": normality_status,
            "normality_reason": f"Skewness={skew:.2f}, Kurtosis={kurt:.2f}. {'Within acceptable range (-1 to 1 skew, -3 to 3 kurtosis)' if normality_status == 'PASS' else 'Outside normal range, residuals are not normally distributed'}",
            "outlier_status": outlier_status,
            "outlier_reason": f"{outliers} outliers ({outlier_pct:.1f}%). {'Acceptable (<1%)' if outlier_status == 'PASS' else 'Moderate (1-5%)' if outlier_status == 'WARNING' else 'High (>5%) - investigate data quality'}",
            "status": overall_status,
            "reason": overall_reason,
            "interpretation": "Residual analysis validates regression assumptions: zero-mean residuals (unbiased), normality (valid confidence intervals), and few outliers (data quality)."
        }
        DataPassingManager.save_artifact(res_data, STAGE7_OUT_DIR, f"{plan_id}_residuals.json")

        return "\n".join(result)

    except Exception as e:
        return f"Error in residual analysis: {e}"


# ============================================================================
# VISUALIZATION
# ============================================================================

@tool
def create_guardrails_visualization(plan_id: str, viz_type: str) -> str:
    """
    Create guardrails validation visualizations.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        viz_type: Type of visualization (correlation_heatmap, propensity_dist, residual_qq, error_by_quintile)

    Returns:
        Path to saved visualization
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        df = pd.read_parquet(predictions_path)
        
        pred_col = next((c for c in df.columns if 'predict' in c.lower()), None)
        actual_col = next((c for c in df.columns if 'actual' in c.lower()), None)
        
        if not pred_col or not actual_col:
            return "Could not identify prediction and actual columns"
        
        residuals = df[actual_col] - df[pred_col]
        task_id = plan_id.replace("PLAN-", "")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "residual_distribution":
            sns.histplot(residuals, kde=True, ax=ax, color='steelblue')
            ax.axvline(x=0, color='red', linestyle='--', label='Zero')
            ax.axvline(x=residuals.mean(), color='orange', linestyle='-', label=f'Mean: {residuals.mean():.3f}')
            ax.set_xlabel('Residual (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.set_title('Guardrails: Residual Distribution')
            ax.legend()
            filename = f"{task_id}_guardrails_residual_dist.png"
            
        elif viz_type == "prediction_scatter":
            ax.scatter(df[actual_col], df[pred_col], alpha=0.6, c='steelblue')
            min_val = min(df[actual_col].min(), df[pred_col].min())
            max_val = max(df[actual_col].max(), df[pred_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Guardrails: Actual vs Predicted')
            ax.legend()
            filename = f"{task_id}_guardrails_scatter.png"
            
        elif viz_type == "error_by_quintile":
            n = len(df)
            quintile_size = n // 5
            sorted_indices = df[pred_col].argsort()
            
            quintiles = []
            maes = []
            for i in range(5):
                start_idx = i * quintile_size
                end_idx = (i + 1) * quintile_size if i < 4 else n
                quintile_indices = sorted_indices[start_idx:end_idx]
                quintiles.append(f'Q{i+1}')
                maes.append(residuals.iloc[quintile_indices].abs().mean())
            
            bars = ax.bar(quintiles, maes, color='steelblue')
            ax.axhline(y=np.mean(maes), color='red', linestyle='--', label=f'Mean MAE: {np.mean(maes):.3f}')
            ax.set_xlabel('Prediction Quintile')
            ax.set_ylabel('Mean Absolute Error')
            ax.set_title('Guardrails: Error by Prediction Quintile (IPW Check)')
            ax.legend()
            filename = f"{task_id}_guardrails_quintile.png"
            
        else:
            return f"Unknown visualization type: {viz_type}"
        
        plt.tight_layout()
        output_path = STAGE7_OUT_DIR / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"Visualization saved to: {output_path}"

    except Exception as e:
        return f"Error creating visualization: {e}"


# ============================================================================
# FINAL REPORT
# ============================================================================

@tool
def save_guardrails_report(
    plan_id: str,
    correlation_result: str,
    propensity_result: str,
    ipw_result: str,
    residual_result: str,
    overall_assessment: str
) -> str:
    """
    Save the final guardrails validation report.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        correlation_result: Pass/Fail/Warning for correlation test
        propensity_result: Pass/Fail/Warning for propensity score test
        ipw_result: Pass/Fail/Warning for IPW test
        residual_result: Pass/Fail/Warning for residual analysis
        overall_assessment: Overall validity assessment with explanation

    Returns:
        Path to saved report
    """
    try:
        task_id = plan_id.replace("PLAN-", "")
        
        # Load individual test results if they exist
        tests = {}
        for test_name in ['correlation', 'propensity', 'ipw', 'residuals']:
            test_path = STAGE7_OUT_DIR / f"{plan_id}_{test_name}.json"
            if test_path.exists():
                tests[test_name] = DataPassingManager.load_artifact(test_path)
        
        # Collect visualization paths
        viz_files = list(STAGE7_OUT_DIR.glob(f"{task_id}_guardrails_*.png"))
        visualizations = [{"filename": f.name, "api_url": f"/api/guardrails/{task_id}/image/{f.name}"} for f in viz_files]
        
        # Calculate validity percentage (each test is worth 25%)
        results = [correlation_result.upper(), propensity_result.upper(), ipw_result.upper(), residual_result.upper()]
        
        def score_result(result):
            if result == "PASS":
                return 25.0
            elif result == "WARNING":
                return 12.5
            else:
                return 0.0
        
        validity_score = sum(score_result(r) for r in results)
        
        # Determine validity label based on score
        if validity_score >= 75:
            validity_label = "HIGH"
            validity_icon = "✅"
            validity_color = "#10b981"
        elif validity_score >= 50:
            validity_label = "MEDIUM"
            validity_icon = "⚠️"
            validity_color = "#f59e0b"
        else:
            validity_label = "LOW"
            validity_icon = "❌"
            validity_color = "#ef4444"
        
        # Individual test scores for display
        test_scores = {
            "correlation": score_result(correlation_result.upper()),
            "propensity": score_result(propensity_result.upper()),
            "ipw": score_result(ipw_result.upper()),
            "residual": score_result(residual_result.upper())
        }
        
        report = {
            "task_id": task_id,
            "plan_id": plan_id,
            "validity_score": validity_score,
            "validity_label": validity_label,
            "validity_icon": validity_icon,
            "validity_color": validity_color,
            "test_scores": test_scores,
            "tests": {
                "correlation_analysis": {
                    "status": correlation_result,
                    "score": test_scores["correlation"],
                    "details": tests.get("correlation", {})
                },
                "propensity_score_analysis": {
                    "status": propensity_result,
                    "score": test_scores["propensity"],
                    "details": tests.get("propensity", {})
                },
                "inverse_propensity_weighting": {
                    "status": ipw_result,
                    "score": test_scores["ipw"],
                    "details": tests.get("ipw", {})
                },
                "residual_analysis": {
                    "status": residual_result,
                    "score": test_scores["residual"],
                    "details": tests.get("residuals", {})
                }
            },
            "overall_assessment": overall_assessment,
            "visualizations": visualizations,
            "generated_at": pd.Timestamp.now().isoformat()
        }
        
        output_path = DataPassingManager.save_artifact(
            report,
            STAGE7_OUT_DIR,
            f"{task_id}_guardrails_report.json",
            metadata={"stage": "stage7", "type": "guardrails_report"}
        )
        
        logger.info(f"Guardrails report saved to {output_path}")
        
        return f"Guardrails report saved: {output_path}\nValidity: {validity_icon} {validity}"

    except Exception as e:
        return f"Error saving guardrails report: {e}"


# Export tools list
STAGE7_GUARDRAILS_TOOLS = [
    load_predictions_for_validation,
    run_correlation_analysis,
    run_propensity_score_analysis,
    run_inverse_propensity_weighting,
    run_residual_analysis,
    create_guardrails_visualization,
    save_guardrails_report,
]
