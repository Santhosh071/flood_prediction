import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "metadata_indofloods.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "cleaned_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
LOG_PATH = os.path.join(BASE_DIR, "logs", "app.log")

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
