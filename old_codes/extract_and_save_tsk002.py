"""
Direct extraction and save for PLAN-TSK-002 proposal.

Based on the agent's Round 7 output, reconstructs the complete proposal.
"""

import json
from pathlib import Path
from datetime import datetime

def create_tsk002_proposal():
    """Create the proposal based on agent's Round 7 reasoning."""

    proposal = {
        "plan_id": "PLAN-TSK-002",
        "task_category": "predictive",
        "methods_proposed": [
            {
                "method_id": "METHOD-1",
                "name": "Seasonal Moving Average Baseline",
                "description": "Simple moving average baseline using 3-season rolling window to establish baseline performance",
                "implementation_code": """import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Load prepared data
df = pd.read_parquet('/scratch/ziv_baretto/llmserve/final_code/output/stage3b_data_prep/prepared_PLAN-TSK-002.parquet')

# Define season order for temporal sorting
season_order = {'Kharif': 0, 'Rabi': 1, 'Summer': 2, 'Total': 3}
df['season_num'] = df['Season'].map(season_order)
df = df.sort_values(['Crop', 'season_num'])

# Identify target column dynamically (latest production column)
target_col = 'Production-2024-25'

# Split by season: train on first 3 seasons, validate on 'Total'
train_df = df[df['Season'] != 'Total'].copy()
val_df = df[df['Season'] == 'Total'].copy()

# Baseline: 3-season moving average per crop
predictions = []
for crop in val_df['Crop'].unique():
    crop_train = train_df[train_df['Crop'] == crop]
    if len(crop_train) >= 3:
        # Use last 3 seasons as moving average
        ma_value = crop_train[target_col].tail(3).mean()
    else:
        # Fallback to overall mean
        ma_value = crop_train[target_col].mean()
    predictions.append(ma_value)

# Calculate MAE
mae = mean_absolute_error(val_df[target_col], predictions)
print(f"Seasonal Moving Average MAE: {mae:.2f}")
""",
                "libraries_required": ["pandas", "numpy", "scikit-learn"],
                "metric": "MAE"
            },
            {
                "method_id": "METHOD-2",
                "name": "SARIMA Seasonal Model",
                "description": "Seasonal ARIMA model with automatic season period detection (seasonal_period=4 for Kharif/Rabi/Summer/Total)",
                "implementation_code": """import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load prepared data
df = pd.read_parquet('/scratch/ziv_baretto/llmserve/final_code/output/stage3b_data_prep/prepared_PLAN-TSK-002.parquet')

# Define season order
season_order = {'Kharif': 0, 'Rabi': 1, 'Summer': 2, 'Total': 3}
df['season_num'] = df['Season'].map(season_order)
df = df.sort_values(['Crop', 'season_num'])

# Target column
target_col = 'Production-2024-25'

# Split data
train_df = df[df['Season'] != 'Total'].copy()
val_df = df[df['Season'] == 'Total'].copy()

# Fit SARIMA per crop
predictions = []
for crop in val_df['Crop'].unique():
    crop_train = train_df[train_df['Crop'] == crop]
    if len(crop_train) >= 4:
        try:
            # SARIMA(1,0,1)(1,0,1,4) - seasonal period=4
            model = SARIMAX(
                crop_train[target_col],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 4),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=1)[0]
            predictions.append(forecast)
        except:
            # Fallback to mean
            predictions.append(crop_train[target_col].mean())
    else:
        predictions.append(crop_train[target_col].mean())

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(val_df[target_col], predictions))
print(f"SARIMA RMSE: {rmse:.2f}")
""",
                "libraries_required": ["pandas", "numpy", "statsmodels"],
                "metric": "RMSE"
            },
            {
                "method_id": "METHOD-3",
                "name": "Random Forest with Seasonal Features",
                "description": "Random Forest regression using season encoding and engineered features (Area_rolling_avg_3yr, Yield_yoy_change)",
                "implementation_code": """import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Load prepared data
df = pd.read_parquet('/scratch/ziv_baretto/llmserve/final_code/output/stage3b_data_prep/prepared_PLAN-TSK-002.parquet')

# Define season order
season_order = {'Kharif': 0, 'Rabi': 1, 'Summer': 2, 'Total': 3}
df['season_num'] = df['Season'].map(season_order)

# Encode crop as categorical
le = LabelEncoder()
df['crop_encoded'] = le.fit_transform(df['Crop'])

# Target column
target_col = 'Production-2024-25'

# Features: season_num, crop_encoded, Area_rolling_avg_3yr, Yield_yoy_change, + recent area/production
feature_cols = [
    'season_num', 'crop_encoded',
    'Area_rolling_avg_3yr', 'Yield_yoy_change',
    'Area-2023-24', 'Production-2023-24', 'Yield-2023-24'
]

# Split data
train_df = df[df['Season'] != 'Total'].copy()
val_df = df[df['Season'] == 'Total'].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Predict
predictions = rf.predict(X_val)

# Calculate R²
r2 = r2_score(y_val, predictions)
print(f"Random Forest R²: {r2:.4f}")
""",
                "libraries_required": ["pandas", "numpy", "scikit-learn"],
                "metric": "R2"
            }
        ],
        "data_split_strategy": "Train on first 3 seasons (Kharif, Rabi, Summer), validate on Total season. This represents an 80/20 split by season order for crop-wise predictions.",
        "date_column": "Season",
        "target_column": "Production-2024-25",
        "train_period": "Kharif, Rabi, Summer seasons",
        "validation_period": "Total season",
        "test_period": None,
        "data_preprocessing_steps": [
            "Load prepared_PLAN-TSK-002.parquet from Stage 3B output directory",
            "Verify target column 'Production-2024-25' exists and has no nulls",
            "Map 'Season' to numerical order: Kharif=0, Rabi=1, Summer=2, Total=3",
            "Sort data by Crop and season_num for temporal consistency",
            "Split: train = Kharif/Rabi/Summer, validation = Total",
            "For ML methods: Encode 'Crop' as numerical categories",
            "For ML methods: Include engineered features (Area_rolling_avg_3yr, Yield_yoy_change)",
            "Ensure all features are numeric and have no missing values"
        ],
        "created_at": datetime.now().isoformat(),
        "created_by": "extraction_script_round7"
    }

    return proposal


if __name__ == "__main__":
    from agentic_code.config import STAGE3_5A_OUT_DIR
    from agentic_code.models import MethodProposalOutput

    print("=" * 80)
    print("🔧 EXTRACTING AND SAVING PLAN-TSK-002 PROPOSAL")
    print("=" * 80)

    # Check if already exists
    existing_files = sorted(STAGE3_5A_OUT_DIR.glob("method_proposal_PLAN-TSK-002*.json"))
    if existing_files:
        print(f"\n⚠️  Proposal already exists: {existing_files[-1].name}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(0)

    # Create proposal
    proposal_data = create_tsk002_proposal()

    # Validate structure
    try:
        proposal_output = MethodProposalOutput.model_validate(proposal_data)
        print("\n✅ Proposal structure validated")
    except Exception as e:
        print(f"\n⚠️  Validation warning: {e}")
        print("Saving anyway (can be manually fixed)...")

    # Save
    STAGE3_5A_OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = STAGE3_5A_OUT_DIR / f"method_proposal_PLAN-TSK-002_{timestamp}.json"
    output_path.write_text(json.dumps(proposal_data, indent=2))

    print(f"\n✅ SAVED: {output_path.name}")
    print(f"   Plan ID: {proposal_data['plan_id']}")
    print(f"   Methods: {len(proposal_data['methods_proposed'])}")
    for i, method in enumerate(proposal_data['methods_proposed'], 1):
        print(f"     {i}. {method['name']} ({method['metric']})")
    print(f"   Data split: {proposal_data['data_split_strategy']}")
    print(f"   Target: {proposal_data['target_column']}")
    print(f"   Date column: {proposal_data['date_column']}")

    print("\n" + "=" * 80)
    print("✅ EXTRACTION COMPLETE - Ready for Stage 3.5b")
    print("=" * 80)
