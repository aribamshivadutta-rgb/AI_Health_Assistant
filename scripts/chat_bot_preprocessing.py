import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import re
import json
import shutil

# =======================
# CONFIGURATION
# =======================
# 1. Automatic Relative Path Detection
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # .../AI_Health_Assistant
BASE_DIR = os.path.join(PROJECT_ROOT, "data")           # .../AI_Health_Assistant/data

print(f"[System] Detected relative path: {BASE_DIR}")

RAW_DIR = os.path.join(BASE_DIR, "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")

# NEW: Output Path - ALL processed data goes to 'chat_bot_clean'
UNIFIED_CLEAN_DIR = os.path.join(CLEAN_DIR, "chat_bot_clean")
CLEAN_CHATBOT_DIR = UNIFIED_CLEAN_DIR
CLEAN_ML_DIR = UNIFIED_CLEAN_DIR

# Specific folder for Chatbot/User input files (Raw Source)
CHATBOT_RAW_INPUT_DIR = os.path.join(RAW_DIR, "Chat_bot_verified_vocab_and_user_input")

# Files to Process
MAIN_TRAIN_FILE = os.path.join(RAW_DIR, "FInal_Train_Data.csv")

# Logic to find learned data (Check specific folder first, then raw)
LEARNED_DATA_FILE = os.path.join(CHATBOT_RAW_INPUT_DIR, "learned_user_data.csv")
if not os.path.exists(LEARNED_DATA_FILE):
    # Fallback if not found in the new folder
    LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")

# Create Output Folders
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(UNIFIED_CLEAN_DIR, exist_ok=True)

# =======================
# PART 1: PROCESS MACHINE LEARNING DATA
# =======================
print("--- Processing ML Training Data ---")

# 1. Initialize Learned Data if missing
if not os.path.exists(LEARNED_DATA_FILE):
    os.makedirs(os.path.dirname(LEARNED_DATA_FILE), exist_ok=True)
    print(f" -> Initializing {LEARNED_DATA_FILE}...")
    with open(LEARNED_DATA_FILE, 'w', newline='', encoding='utf-8') as f:
        f.write("prognosis,fever,cough\n")

# 2. Load Main Data
if not os.path.exists(MAIN_TRAIN_FILE):
    raise FileNotFoundError(f"Missing main training file: {MAIN_TRAIN_FILE}")

print(f" -> Loading main dataset...")
df_main = pd.read_csv(MAIN_TRAIN_FILE)

# 3. Load & Merge Learned Data
try:
    print(f" -> Checking for user learned data at: {LEARNED_DATA_FILE}")
    df_learned = pd.read_csv(LEARNED_DATA_FILE)
    if not df_learned.empty and len(df_learned) > 0:
        print(f" -> Merging {len(df_learned)} new verified samples.")
        # Ensure columns match before merging
        df = pd.concat([df_main, df_learned], ignore_index=True)
    else:
        df = df_main
except Exception as e:
    print(f" -> Warning reading learned data: {e}")
    df = df_main

# 4. Clean Data (Robust Cleaning)
# Remove duplicate columns
duplicate_cols = df.columns[df.columns.duplicated()].tolist()
if duplicate_cols:
    print(f" -> Removing duplicate columns: {duplicate_cols}")
df = df.loc[:, ~df.columns.duplicated()].copy()

# Drop unnamed index columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Clean column names (strip spaces, handle non-breaking spaces)
df.columns = df.columns.str.replace('\xa0', ' ', regex=False).str.strip()

# Clean prognosis column text
if 'prognosis' in df.columns and df['prognosis'].dtype == object:
    df['prognosis'] = df['prognosis'].str.replace('\xa0', ' ', regex=False).str.strip()

# Identify symptom columns (all cols except prognosis)
symptom_columns = [col for col in df.columns if col != 'prognosis']

# Fill NaNs with 0
df[symptom_columns] = df[symptom_columns].fillna(0)

# 5. Filter Data (Quality Control)
print(" -> Filtering low-quality data...")

# Filter rows with fewer than MIN_SYMPTOMS
symptom_count = df[symptom_columns].sum(axis=1)
MIN_SYMPTOMS = 3
df = df[symptom_count >= MIN_SYMPTOMS]

# Filter diseases with fewer than MIN_ROWS examples
MIN_ROWS = 3
valid_diseases = df['prognosis'].value_counts()
valid_diseases = valid_diseases[valid_diseases >= MIN_ROWS].index
df = df[df['prognosis'].isin(valid_diseases)]

# Reset index to avoid alignment issues later
df.reset_index(drop=True, inplace=True)


# 6. Sanitize Features
def sanitize_column(name):
    return re.sub(r'\W+', '_', name).lower()


df.columns = [sanitize_column(col) for col in df.columns]
symptom_columns = [sanitize_column(col) for col in symptom_columns]

# 7. Encode Labels
le = LabelEncoder()
prognosis_encoded = le.fit_transform(df['prognosis'])

# Add Encoded Prognosis to DataFrame
df['prognosis_encoded'] = prognosis_encoded

# 8. Save Processed Files (For Prediction Scripts)
clean_data_path = os.path.join(CLEAN_ML_DIR, "preprocessed_data.csv")
df.to_csv(clean_data_path, index=False)

# Separate features and labels for saving
X = df[symptom_columns]
y = df['prognosis_encoded']

X_path = os.path.join(CLEAN_ML_DIR, "X_preprocessed.csv")
y_path = os.path.join(CLEAN_ML_DIR, "y_preprocessed.csv")
le_path = os.path.join(CLEAN_ML_DIR, "label_encoder.pkl")

X.to_csv(X_path, index=False)
y.to_csv(y_path, index=False)
joblib.dump(le, le_path)

# Save Label Map JSON (Useful for Chatbot)
label_map = {int(i): disease for i, disease in enumerate(le.classes_)}
with open(os.path.join(CLEAN_ML_DIR, "disease_label_map.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print(f" -> Full preprocessed data saved to: {clean_data_path}")
print(f" -> Features saved to: {X_path}")
print(f" -> Labels saved to: {y_path}")
print(f" -> Label encoder saved to: {le_path}")

# 9. Split & Save Train/Test (Required for Training Scripts)
print(" -> Splitting Train/Test sets...")
# Stratify ensures we have examples of every disease in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Recombine for saving standard train/test files used by train_model.py
train_df = X_train.copy();
train_df['disease_id'] = y_train
test_df = X_test.copy();
test_df['disease_id'] = y_test

train_df.to_csv(os.path.join(CLEAN_ML_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(CLEAN_ML_DIR, "test.csv"), index=False)

# Process Metadata (Verified Disease Sources)
print(" -> Processing Disease Metadata...")
metadata_file = "verified_disease_sources.csv"
raw_meta_path = os.path.join(CHATBOT_RAW_INPUT_DIR, metadata_file)
if not os.path.exists(raw_meta_path):
    raw_meta_path = os.path.join(RAW_DIR, metadata_file)

if os.path.exists(raw_meta_path):
    try:
        meta_df = pd.read_csv(raw_meta_path)
        meta_df.drop_duplicates(inplace=True)
        meta_clean_path = os.path.join(CLEAN_ML_DIR, metadata_file)
        meta_df.to_csv(meta_clean_path, index=False)
        print(f"    Saved metadata to: {meta_clean_path}")
    except Exception as e:
        print(f"    Error processing metadata: {e}")
else:
    print(f"    Warning: {metadata_file} not found.")

print(f" -> Train/Test split saved to: {CLEAN_ML_DIR}")

# =======================
# PART 2: PROCESS CHATBOT DATA
# =======================
print("\n--- Processing Chatbot Knowledge Base ---")
print(f" -> Looking for files in: {CHATBOT_RAW_INPUT_DIR}")

chatbot_files = [
    ("verified_vocab.csv", "Vocabulary"),
    ("chatbot_intents.csv", "Intents"),
]

for filename, desc in chatbot_files:
    # 1. Try the new specific folder
    raw_path = os.path.join(CHATBOT_RAW_INPUT_DIR, filename)

    # 2. Fallback to main RAW folder if not found in specific folder
    if not os.path.exists(raw_path):
        raw_path = os.path.join(RAW_DIR, filename)

    # OUTPUT: Uses the new UNIFIED path
    clean_path = os.path.join(CLEAN_CHATBOT_DIR, filename)

    if os.path.exists(raw_path):
        try:
            cb_df = pd.read_csv(raw_path)
            initial_count = len(cb_df)
            cb_df.drop_duplicates(inplace=True)
            cb_df.to_csv(clean_path, index=False)
            print(f" -> {desc}: Processed {initial_count} raw -> {len(cb_df)} clean rows.")
            print(f"    Source: {raw_path}")
            print(f"    Saved to: {clean_path}")
        except Exception as e:
            print(f" -> Error processing {desc}: {e}")
    else:
        print(f" -> Warning: {filename} not found in {CHATBOT_RAW_INPUT_DIR} or {RAW_DIR}.")

print("-" * 30)
print("Preprocessing Complete!")
print(f"All data saved to: {UNIFIED_CLEAN_DIR}")
print("-" * 30)