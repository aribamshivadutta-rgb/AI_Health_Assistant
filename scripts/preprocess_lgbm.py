import os
import pandas as pd
import re
import joblib
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =======================
# CONFIGURATION
# =======================
# 1. Dynamic Path Detection
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "FInal_Train_Data.csv")
LEARNED_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "learned_user_data.csv")

# [CRITICAL UPDATE] Output to 'chat_bot_clean' to match other scripts
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SETTINGS
MIN_SAMPLES_PER_DISEASE = 10  # Ensure every disease has at least 10 examples


# =======================
# HELPER FUNCTIONS
# =======================
def sanitize_column(name):
    return re.sub(r'\W+', '_', name).lower().strip()


def augment_rare_diseases(df, min_samples):
    """
    Finds diseases with few examples and creates synthetic copies
    by randomly dropping symptoms. This makes the AI robust.
    """
    print(f" -> Running Data Augmentation (Target: {min_samples} samples/disease)...")

    counts = df['prognosis'].value_counts()
    rare_diseases = counts[counts < min_samples].index

    if len(rare_diseases) == 0:
        print("    No rare diseases found. Skipping augmentation.")
        return df

    new_rows = []
    symptom_cols = [c for c in df.columns if c != 'prognosis']

    for disease in rare_diseases:
        # Get the original rows for this disease
        original_rows = df[df['prognosis'] == disease]
        current_count = len(original_rows)
        needed = min_samples - current_count

        print(f"    - Augmenting '{disease}': Creating {needed} synthetic variations...")

        for _ in range(needed):
            # Pick a random source row to clone
            source = original_rows.sample(1).iloc[0].copy()

            # Identify active symptoms (value = 1)
            active_symptoms = [c for c in symptom_cols if source[c] == 1]

            # Randomly drop 1 or 2 symptoms (set to 0) to simulate vague cases
            if len(active_symptoms) > 2:
                drop_count = random.randint(1, 2)
                to_drop = random.sample(active_symptoms, drop_count)
                for sym in to_drop:
                    source[sym] = 0
            elif len(active_symptoms) == 2:
                to_drop = random.choice(active_symptoms)
                source[to_drop] = 0

            new_rows.append(source)

    if new_rows:
        augmented_df = pd.DataFrame(new_rows)
        df_final = pd.concat([df, augmented_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
        return df_final

    return df


def preprocess():
    print("========================================")
    print("   PRE-PROCESSING + AUGMENTATION SYSTEM ")
    print("========================================")

    # --- 1. Load Main Data ---
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"CRITICAL ERROR: Missing raw file: {RAW_FILE}")

    df_main = pd.read_csv(RAW_FILE)
    print(f" -> Loaded Main Dataset: {len(df_main)} rows")

    # --- 2. Load Learned Data ---
    df_learned = pd.DataFrame()
    if os.path.exists(LEARNED_FILE):
        try:
            df_learned = pd.read_csv(LEARNED_FILE)
            if not df_learned.empty:
                print(f" -> Found New Knowledge: {len(df_learned)} rows")
            else:
                print(" -> Learned file exists but is empty.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load learned data: {e}")

    # --- 3. Normalize Columns ---
    df_main.columns = [sanitize_column(c) for c in df_main.columns]

    if not df_learned.empty:
        df_learned.columns = [sanitize_column(c) for c in df_learned.columns]
        df = pd.concat([df_main, df_learned], ignore_index=True, sort=False)
        print(f" -> Merged successfully. Total rows: {len(df)}")
    else:
        df = df_main

    # --- 4. Basic Cleaning ---
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, ~df.columns.str.contains('^unnamed')]

    if 'prognosis' not in df.columns:
        raise ValueError("CRITICAL: Column 'prognosis' not found!")

    df['prognosis'] = df['prognosis'].str.replace(r'\xa0', ' ', regex=True).str.strip().str.title()

    # --- 5. Handle Symptoms ---
    symptom_cols = [col for col in df.columns if col != 'prognosis']
    df[symptom_cols] = df[symptom_cols].fillna(0)

    # --- 6. Filtering ---
    df = df[df[symptom_cols].sum(axis=1) >= 1]

    # --- 7. DATA AUGMENTATION ---
    df = augment_rare_diseases(df, min_samples=MIN_SAMPLES_PER_DISEASE)
    print(f" -> Augmentation Complete. Final Dataset Size: {len(df)} rows")

    # --- 8. Encoding ---
    print(" -> Encoding Targets...")
    le = LabelEncoder()
    df['prognosis_encoded'] = le.fit_transform(df['prognosis'])
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    # Save Features
    X = df[symptom_cols]
    X.to_csv(os.path.join(OUTPUT_DIR, "X_preprocessed.csv"), index=False)

    y = df['prognosis_encoded']

    # --- 9. Splitting ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # --- 10. Saving ---
    train_df = X_train.copy();
    train_df['disease_id'] = y_train
    test_df = X_test.copy();
    test_df['disease_id'] = y_test

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    df.to_csv(os.path.join(OUTPUT_DIR, "preprocessed_data.csv"), index=False)

    print(f"✅ Preprocessing Complete! Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess()