"""Central configuration: paths and constants shared by every module."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The dataset location can be overridden (useful on Kaggle):  CHURN_DATA_PATH=/path/to/file.csv
DATA_PATH = Path(os.environ.get("CHURN_DATA_PATH", PROJECT_ROOT / "data" / "customer_data.csv"))
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "Churn"
ID_COL = "customerID"

RANDOM_STATE = 42      # every random operation uses this seed -> reproducible results
TEST_SIZE = 0.20
CV_FOLDS = 5

# Business rule: missing a churner (false negative) is costlier than sending a retention
# offer to a loyal customer (false positive), so we tune the decision threshold with F-beta,
# beta=1.5 (recall counts more than precision).
BETA = 1.5
