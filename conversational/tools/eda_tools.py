"""
EDA Agent Tools: Exploratory Data Analysis

Tools for the EDA agent to explore, analyze, and visualize datasets.
The agent uses these tools to intelligently answer user queries about data.
"""

import json
import sys
import io
import time
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, SUMMARIES_DIR, EDA_OUT_DIR, EDA_WORKSPACE,
    DataPassingManager, logger
)
from code.utils import list_data_files, list_summary_files


# ============================================================================
# DATASET DISCOVERY TOOLS
# ============================================================================

@tool
def list_all_datasets() -> str:
    """
    List all available datasets with their status (summarized or new).
    
    Returns a comprehensive list of datasets showing:
    - Which datasets have summaries (from Stage 1)
    - Which datasets are new and need summarization
    - Basic file information
    """
    try:
        # Get all data files
        all_files = list_data_files(DATA_DIR)
        
        # Get existing summaries
        summary_files = list_summary_files(SUMMARIES_DIR)
        summarized_stems = {
            Path(sf).stem.replace('.summary', '') 
            for sf in summary_files
        }
        
        result = ["=== Available Datasets ===\n"]
        summarized = []
        new_datasets = []
        
        for f in all_files:
            filepath = DATA_DIR / f
            stem = Path(f).stem
            size_mb = filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0
            
            if stem in summarized_stems:
                summarized.append(f"  ✅ {f} ({size_mb:.2f} MB) - Summarized")
            else:
                new_datasets.append(f"  🆕 {f} ({size_mb:.2f} MB) - NEW (needs summarization)")
        
        if summarized:
            result.append("Summarized Datasets:")
            result.extend(summarized)
            result.append("")
        
        if new_datasets:
            result.append("New Datasets (not yet summarized):")
            result.extend(new_datasets)
            result.append("\n⚠️ New datasets detected! Ask user if they want to summarize them.")
        
        result.append(f"\nTotal: {len(all_files)} datasets ({len(summarized)} summarized, {len(new_datasets)} new)")
        return "\n".join(result)
        
    except Exception as e:
        return f"Error listing datasets: {e}"


@tool
def get_dataset_info(dataset_name: str) -> str:
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset file (e.g., 'sales_data.csv')
    
    Returns:
        Comprehensive dataset information including columns, types, statistics,
        and sample data. If a summary exists, uses that; otherwise reads the file directly.
    """
    try:
        # Find the file
        dataset_path = DATA_DIR / dataset_name
        if not dataset_path.exists():
            # Try to find by partial name
            matches = list(DATA_DIR.glob(f"*{dataset_name}*"))
            if matches:
                dataset_path = matches[0]
                dataset_name = dataset_path.name
            else:
                return f"Dataset '{dataset_name}' not found. Use list_all_datasets to see available datasets."
        
        result = [f"=== Dataset Info: {dataset_name} ===\n"]
        
        # Check for existing summary
        summary_path = SUMMARIES_DIR / f"{dataset_path.stem}.summary.json"
        if summary_path.exists():
            summary = DataPassingManager.load_artifact(summary_path)
            data = summary.get('data', summary) if isinstance(summary, dict) else summary
            
            result.append(f"Rows: {data.get('n_rows', 'Unknown')}")
            result.append(f"Columns: {data.get('n_cols', 'Unknown')}")
            result.append(f"Has DateTime: {data.get('has_datetime_column', False)}")
            result.append(f"Quality Score: {data.get('data_quality_score', 'N/A')}")
            result.append("\nColumns:")
            
            for col in data.get('columns', [])[:20]:
                col_type = col.get('logical_type', col.get('dtype', 'unknown'))
                null_pct = col.get('null_fraction', 0) * 100
                result.append(f"  - {col.get('name')}: {col_type} (nulls: {null_pct:.1f}%)")
            
            if len(data.get('columns', [])) > 20:
                result.append(f"  ... and {len(data.get('columns', [])) - 20} more columns")
        else:
            # Read file directly
            df = pd.read_csv(dataset_path, nrows=1000)
            
            result.append(f"Rows (sample): {len(df)} (file may have more)")
            result.append(f"Columns: {len(df.columns)}")
            result.append("\nColumn Details:")
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_pct = df[col].isnull().mean() * 100
                n_unique = df[col].nunique()
                result.append(f"  - {col}: {dtype} (nulls: {null_pct:.1f}%, unique: {n_unique})")
            
            result.append("\nSample Data (first 3 rows):")
            result.append(df.head(3).to_string())
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error getting dataset info: {e}"


@tool
def check_for_new_datasets() -> str:
    """
    Check for new datasets that haven't been summarized yet.
    
    This tool detects new CSV/TSV/Parquet files in the data directory
    that don't have corresponding summaries.
    
    Returns information about new datasets and asks if user wants to summarize them.
    """
    try:
        # Get all data files
        all_files = list_data_files(DATA_DIR)
        
        # Get existing summaries
        summary_files = list_summary_files(SUMMARIES_DIR)
        summarized_stems = {
            Path(sf).stem.replace('.summary', '') 
            for sf in summary_files
        }
        
        # Find new files
        new_files = []
        for f in all_files:
            stem = Path(f).stem
            if stem not in summarized_stems:
                filepath = DATA_DIR / f
                size_mb = filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0
                new_files.append({
                    'name': f,
                    'size_mb': size_mb,
                    'path': str(filepath)
                })
        
        if not new_files:
            return "✅ No new datasets detected. All datasets have been summarized."
        
        result = [f"🆕 Found {len(new_files)} new dataset(s):\n"]
        for nf in new_files:
            result.append(f"  - {nf['name']} ({nf['size_mb']:.2f} MB)")
        
        result.append("\n" + "="*50)
        result.append("❓ Would you like me to summarize these new datasets?")
        result.append("   (This will run Stage 1 data profiling)")
        result.append("   Please ask the user for confirmation before proceeding.")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error checking for new datasets: {e}"


# ============================================================================
# CODE EXECUTION TOOL
# ============================================================================

@tool
def execute_analysis_code(code: str, description: str = "") -> str:
    """
    Execute Python code for data analysis.
    
    The code runs in an environment with access to:
    - pandas as pd
    - numpy as np
    - matplotlib.pyplot as plt
    - seaborn as sns
    - scipy.stats as stats
    - All datasets via: df = pd.read_csv(DATA_DIR / 'filename.csv')
    
    The code should:
    1. Load data using pd.read_csv(DATA_DIR / 'filename.csv')
    2. Perform analysis
    3. Print results or save plots to EDA_WORKSPACE
    
    Args:
        code: Python code to execute
        description: Brief description of what the code does
    
    Returns:
        Code execution result including stdout output and any generated files
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        
        # Create execution namespace with useful imports
        namespace = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'stats': stats,
            'DATA_DIR': DATA_DIR,
            'EDA_WORKSPACE': EDA_WORKSPACE,
            'EDA_OUT_DIR': EDA_OUT_DIR,
            'Path': Path,
        }
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        start_time = time.time()
        artifacts = []
        
        try:
            # Execute the code
            exec(code, namespace)
            
            # Check for any figures and save them with descriptive names
            if plt.get_fignums():
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Create descriptive base name from description
                desc_slug = description[:30].lower().replace(' ', '_').replace('/', '_') if description else 'analysis'
                desc_slug = ''.join(c for c in desc_slug if c.isalnum() or c == '_')
                
                for fig_num in plt.get_fignums():
                    plot_path = EDA_OUT_DIR / f"{desc_slug}_{timestamp}_{fig_num}.png"
                    plt.figure(fig_num).savefig(plot_path, dpi=150, bbox_inches='tight')
                    artifacts.append(str(plot_path))
                plt.close('all')
            
            output = sys.stdout.getvalue()
            execution_time = time.time() - start_time
            
            result = [f"=== Code Execution Result ==="]
            if description:
                result.append(f"Description: {description}")
            result.append(f"Execution time: {execution_time:.2f}s")
            result.append(f"\nOutput:\n{output if output else '(No printed output)'}")
            
            if artifacts:
                result.append(f"\nGenerated files:")
                for a in artifacts:
                    result.append(f"  - {a}")
            
            return "\n".join(result)
            
        except Exception as e:
            output = sys.stdout.getvalue()
            error_msg = traceback.format_exc()
            return f"❌ Code execution error:\n{error_msg}\n\nPartial output:\n{output}"
        finally:
            sys.stdout = old_stdout
            plt.close('all')
            
    except Exception as e:
        return f"Error setting up execution environment: {e}"


# ============================================================================
# ANALYSIS TOOLS
# ============================================================================

@tool
def compute_statistics(dataset_name: str, columns: str = None) -> str:
    """
    Compute descriptive statistics for specified columns in a dataset.
    
    Args:
        dataset_name: Name of the dataset file
        columns: Comma-separated column names (optional, all numeric if not specified)
    
    Returns:
        Descriptive statistics including mean, median, std, min, max, quartiles
    """
    try:
        dataset_path = DATA_DIR / dataset_name
        if not dataset_path.exists():
            matches = list(DATA_DIR.glob(f"*{dataset_name}*"))
            if matches:
                dataset_path = matches[0]
            else:
                return f"Dataset '{dataset_name}' not found."
        
        df = pd.read_csv(dataset_path)
        
        if columns:
            col_list = [c.strip() for c in columns.split(',')]
            # Validate columns exist
            valid_cols = [c for c in col_list if c in df.columns]
            if not valid_cols:
                return f"None of the specified columns found. Available: {list(df.columns)}"
            df = df[valid_cols]
        
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return "No numeric columns found in the dataset or selection."
        
        stats_df = numeric_df.describe()
        
        result = [f"=== Statistics for {dataset_path.name} ===\n"]
        result.append(stats_df.to_string())
        
        # Add additional stats
        result.append("\n\nAdditional Statistics:")
        for col in numeric_df.columns:
            result.append(f"\n{col}:")
            result.append(f"  Skewness: {numeric_df[col].skew():.4f}")
            result.append(f"  Kurtosis: {numeric_df[col].kurtosis():.4f}")
            result.append(f"  Missing: {numeric_df[col].isnull().sum()} ({numeric_df[col].isnull().mean()*100:.1f}%)")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error computing statistics: {e}"


@tool  
def compute_correlation(dataset_name: str, columns: str = None, method: str = "pearson") -> str:
    """
    Compute correlation matrix for numeric columns in a dataset.
    
    Args:
        dataset_name: Name of the dataset file
        columns: Comma-separated column names (optional)
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
    
    Returns:
        Correlation matrix and key insights about relationships
    """
    try:
        dataset_path = DATA_DIR / dataset_name
        if not dataset_path.exists():
            matches = list(DATA_DIR.glob(f"*{dataset_name}*"))
            if matches:
                dataset_path = matches[0]
            else:
                return f"Dataset '{dataset_name}' not found."
        
        df = pd.read_csv(dataset_path)
        
        if columns:
            col_list = [c.strip() for c in columns.split(',')]
            valid_cols = [c for c in col_list if c in df.columns]
            if valid_cols:
                df = df[valid_cols]
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return "Need at least 2 numeric columns for correlation analysis."
        
        # Compute correlation
        corr_matrix = numeric_df.corr(method=method)
        
        result = [f"=== Correlation Matrix ({method}) ===\n"]
        result.append(corr_matrix.to_string())
        
        # Find strongest correlations
        result.append("\n\nTop Correlations (excluding self-correlation):")
        corr_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if not np.isnan(corr_val):
                    corr_pairs.append((col1, col2, corr_val))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, val in corr_pairs[:10]:
            strength = "Strong" if abs(val) > 0.7 else "Moderate" if abs(val) > 0.4 else "Weak"
            direction = "positive" if val > 0 else "negative"
            result.append(f"  {col1} ↔ {col2}: {val:.4f} ({strength} {direction})")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error computing correlation: {e}"


@tool
def find_patterns(dataset_name: str, column: str) -> str:
    """
    Find patterns, trends, or anomalies in a specific column.
    
    Args:
        dataset_name: Name of the dataset file
        column: Column name to analyze
    
    Returns:
        Pattern analysis including distribution, outliers, trends, and anomalies
    """
    try:
        dataset_path = DATA_DIR / dataset_name
        if not dataset_path.exists():
            matches = list(DATA_DIR.glob(f"*{dataset_name}*"))
            if matches:
                dataset_path = matches[0]
            else:
                return f"Dataset '{dataset_name}' not found."
        
        df = pd.read_csv(dataset_path)
        
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"
        
        series = df[column]
        result = [f"=== Pattern Analysis: {column} ===\n"]
        
        # Basic info
        result.append(f"Data type: {series.dtype}")
        result.append(f"Non-null values: {series.notna().sum()} ({series.notna().mean()*100:.1f}%)")
        result.append(f"Unique values: {series.nunique()}")
        
        if pd.api.types.is_numeric_dtype(series):
            # Numeric analysis
            result.append(f"\nNumeric Statistics:")
            result.append(f"  Mean: {series.mean():.4f}")
            result.append(f"  Median: {series.median():.4f}")
            result.append(f"  Std: {series.std():.4f}")
            result.append(f"  Range: [{series.min():.4f}, {series.max():.4f}]")
            
            # Outlier detection using IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers_low = series[series < (Q1 - 1.5 * IQR)]
            outliers_high = series[series > (Q3 + 1.5 * IQR)]
            
            result.append(f"\nOutlier Analysis (IQR method):")
            result.append(f"  Low outliers: {len(outliers_low)} values below {Q1 - 1.5 * IQR:.4f}")
            result.append(f"  High outliers: {len(outliers_high)} values above {Q3 + 1.5 * IQR:.4f}")
            
            # Distribution shape
            skew = series.skew()
            result.append(f"\nDistribution:")
            result.append(f"  Skewness: {skew:.4f} ({'right-skewed' if skew > 0.5 else 'left-skewed' if skew < -0.5 else 'symmetric'})")
            
        else:
            # Categorical analysis
            result.append(f"\nValue Frequencies (top 10):")
            value_counts = series.value_counts()
            for val, count in value_counts.head(10).items():
                pct = count / len(series) * 100
                result.append(f"  {val}: {count} ({pct:.1f}%)")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error finding patterns: {e}"


@tool
def compare_datasets(dataset1: str, dataset2: str, join_key: str = None) -> str:
    """
    Compare two datasets and find potential relationships.
    
    Args:
        dataset1: First dataset name
        dataset2: Second dataset name
        join_key: Optional common column to join on
    
    Returns:
        Comparison including structure, common columns, and potential relationships
    """
    try:
        # Load datasets
        path1 = DATA_DIR / dataset1
        path2 = DATA_DIR / dataset2
        
        if not path1.exists():
            matches = list(DATA_DIR.glob(f"*{dataset1}*"))
            if matches:
                path1 = matches[0]
            else:
                return f"Dataset '{dataset1}' not found."
        
        if not path2.exists():
            matches = list(DATA_DIR.glob(f"*{dataset2}*"))
            if matches:
                path2 = matches[0]
            else:
                return f"Dataset '{dataset2}' not found."
        
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        
        result = [f"=== Dataset Comparison ===\n"]
        
        # Structure comparison
        result.append(f"{path1.name}:")
        result.append(f"  Rows: {len(df1)}, Columns: {len(df1.columns)}")
        result.append(f"  Columns: {list(df1.columns)[:10]}...")
        
        result.append(f"\n{path2.name}:")
        result.append(f"  Rows: {len(df2)}, Columns: {len(df2.columns)}")
        result.append(f"  Columns: {list(df2.columns)[:10]}...")
        
        # Find common columns
        common_cols = set(df1.columns) & set(df2.columns)
        if common_cols:
            result.append(f"\nCommon Columns: {list(common_cols)}")
        else:
            result.append("\nNo common columns found (different structures)")
        
        # Check for potential join keys
        result.append("\nPotential Join Keys:")
        for col in common_cols:
            vals1 = set(df1[col].dropna().astype(str).unique())
            vals2 = set(df2[col].dropna().astype(str).unique())
            overlap = len(vals1 & vals2)
            if overlap > 0:
                result.append(f"  {col}: {overlap} matching values ({overlap/max(len(vals1), len(vals2))*100:.1f}% overlap)")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error comparing datasets: {e}"


# ============================================================================
# VISUALIZATION TOOL
# ============================================================================

@tool
def create_visualization(
    dataset_name: str,
    plot_type: str,
    x_column: str = None,
    y_column: str = None,
    title: str = None,
    hue_column: str = None
) -> str:
    """
    Create a visualization and save it to the EDA output directory.
    
    Args:
        dataset_name: Name of the dataset to visualize
        plot_type: Type of plot (bar, line, scatter, histogram, box, heatmap, pie)
        x_column: Column for x-axis (optional for some plots)
        y_column: Column for y-axis (optional for some plots)
        title: Plot title (auto-generated if not provided)
        hue_column: Column for color grouping (optional)
    
    Returns:
        Path to the generated visualization
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Find the dataset
        dataset_path = DATA_DIR / dataset_name
        if not dataset_path.exists():
            matches = list(DATA_DIR.glob(f"*{dataset_name}*"))
            if matches:
                dataset_path = matches[0]
            else:
                return f"Dataset '{dataset_name}' not found."
        
        df = pd.read_csv(dataset_path)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate title if not provided
        if not title:
            title = f"{plot_type.capitalize()} Plot: {dataset_path.stem}"
        
        # Create the appropriate plot
        plot_type = plot_type.lower()
        
        if plot_type == 'histogram':
            if y_column:
                sns.histplot(data=df, x=y_column, hue=hue_column if hue_column and hue_column in df.columns else None, ax=ax)
            else:
                # Plot first numeric column
                numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                sns.histplot(data=df, x=numeric_col, ax=ax)
                title = f"Histogram: {numeric_col}"
                
        elif plot_type == 'bar':
            if x_column and y_column and x_column in df.columns and y_column in df.columns:
                sns.barplot(data=df, x=x_column, y=y_column, hue=hue_column if hue_column and hue_column in df.columns else None, ax=ax)
            else:
                return "Bar plot requires valid x_column and y_column."
                
        elif plot_type == 'line':
            if x_column and y_column and x_column in df.columns and y_column in df.columns:
                sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column if hue_column and hue_column in df.columns else None, ax=ax)
            else:
                return "Line plot requires valid x_column and y_column."
                
        elif plot_type == 'scatter':
            if x_column and y_column and x_column in df.columns and y_column in df.columns:
                sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column if hue_column and hue_column in df.columns else None, ax=ax)
            else:
                return "Scatter plot requires valid x_column and y_column."
                
        elif plot_type == 'box':
            if y_column and y_column in df.columns:
                sns.boxplot(data=df, x=x_column if x_column and x_column in df.columns else None, y=y_column, ax=ax)
            else:
                return "Box plot requires valid y_column."
                
        elif plot_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                return "Heatmap requires at least 2 numeric columns."
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            title = f"Correlation Heatmap: {dataset_path.stem}"
            
        elif plot_type == 'pie':
            if x_column and x_column in df.columns:
                counts = df[x_column].value_counts().head(10)
                ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
                title = f"Distribution: {x_column}"
            else:
                return "Pie chart requires valid x_column."
        else:
            return f"Unknown plot type: {plot_type}. Use: bar, line, scatter, histogram, box, heatmap, pie"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot with descriptive name
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_slug = dataset_path.stem[:20].lower().replace(' ', '_')
        col_slug = (y_column or x_column or 'data')[:15].lower().replace(' ', '_')
        plot_filename = f"{dataset_slug}_{plot_type}_{col_slug}_{timestamp}.png"
        plot_path = EDA_OUT_DIR / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"✅ Visualization created:\n  Path: {plot_path}\n  Type: {plot_type}\n  Title: {title}"
        
    except Exception as e:
        plt.close('all')
        return f"Error creating visualization: {e}"


# ============================================================================
# REPORT SAVING TOOL
# ============================================================================

@tool
def save_eda_report(
    query: str,
    answer: str,
    insights: str,
    visualizations: str = None,
    datasets_used: str = None
) -> str:
    """
    Save the EDA analysis results as a report.
    
    Args:
        query: The original user query
        answer: The answer/response to the query
        insights: Comma-separated key insights from the analysis
        visualizations: Comma-separated paths to generated visualizations
        datasets_used: Comma-separated names of datasets analyzed
    
    Returns:
        Confirmation of saved report
    """
    try:
        import time
        
        report_id = f"EDA-{int(time.time())}"
        
        report = {
            "report_id": report_id,
            "query": query,
            "answer": answer,
            "insights": [i.strip() for i in insights.split(',')] if insights else [],
            "visualizations": [v.strip() for v in visualizations.split(',')] if visualizations else [],
            "datasets_used": [d.strip() for d in datasets_used.split(',')] if datasets_used else [],
            "created_at": DataPassingManager.generate_artifact_id("eda")
        }
        
        DataPassingManager.save_artifact(
            data=report,
            output_dir=EDA_OUT_DIR,
            filename=f"eda_report_{report_id}.json",
            metadata={"stage": "eda", "type": "eda_report"}
        )
        
        return f"✅ EDA Report saved:\n  Report ID: {report_id}\n  Path: {EDA_OUT_DIR / f'eda_report_{report_id}.json'}"
        
    except Exception as e:
        return f"Error saving report: {e}"


@tool
def summarize_new_dataset(dataset_name: str) -> str:
    """
    Summarize a new dataset by running data profiling.
    
    Only call this after getting user confirmation!
    
    Args:
        dataset_name: Name of the dataset to summarize
    
    Returns:
        Summary of the newly profiled dataset
    """
    try:
        from code.utils import profile_csv
        
        dataset_path = DATA_DIR / dataset_name
        if not dataset_path.exists():
            matches = list(DATA_DIR.glob(f"*{dataset_name}*"))
            if matches:
                dataset_path = matches[0]
            else:
                return f"Dataset '{dataset_name}' not found."
        
        # Profile the dataset
        summary = profile_csv(dataset_path)
        
        # Save the summary
        summary_dict = summary.model_dump()
        output_name = f"{dataset_path.stem}.summary.json"
        
        DataPassingManager.save_artifact(
            data=summary_dict,
            output_dir=SUMMARIES_DIR,
            filename=output_name,
            metadata={"stage": "stage1", "type": "dataset_summary", "source": "eda_agent"}
        )
        
        result = [f"✅ Dataset summarized: {dataset_name}\n"]
        result.append(f"Rows: {summary.n_rows}")
        result.append(f"Columns: {summary.n_cols}")
        result.append(f"Has DateTime: {summary.has_datetime_column}")
        result.append(f"Quality Score: {summary.data_quality_score}")
        result.append(f"\nSummary saved to: {SUMMARIES_DIR / output_name}")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error summarizing dataset: {e}"


# ============================================================================
# TOOL EXPORTS
# ============================================================================

EDA_TOOLS = [
    list_all_datasets,
    get_dataset_info,
    check_for_new_datasets,
    execute_analysis_code,
    compute_statistics,
    compute_correlation,
    find_patterns,
    compare_datasets,
    create_visualization,
    save_eda_report,
    summarize_new_dataset,
]
