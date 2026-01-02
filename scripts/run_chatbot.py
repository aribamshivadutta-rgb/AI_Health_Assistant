import os
import sys
import pandas as pd
import joblib
import re
import difflib
import csv
import requests
import subprocess
from datetime import datetime
from bs4 import BeautifulSoup

# =======================
# CONFIGURATION
# =======================
# 1. Dynamic Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

# [CRITICAL UPDATE] Pointing to the new unified data folder
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp")

# Files
REQUESTS_FILE = os.path.join(TEMP_DIR, "unverified_diseases.csv")
ADVICE_DB_PATH = os.path.join(RAW_DIR, "who_data_clean.csv")
LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")

# Model Paths
LGBM_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

# Auto-detect Model
if os.path.exists(SVM_PATH):
    MODEL_PATH = SVM_PATH
    MODEL_TYPE = "SVM"
elif os.path.exists(LGBM_PATH):
    MODEL_PATH = LGBM_PATH
    MODEL_TYPE = "LightGBM"
else:
    MODEL_PATH = None

LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")

# [CRITICAL FIX] Ensure we look for the full dataset in the correct folder
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")

# Training Scripts
PREPROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "chat_bot_preprocessing.py")
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_lgbm.py")


class HealthChatbot:
    def __init__(self):
        self.model = None
        self.le = None
        self.known_symptoms = []
        self.known_diseases = []
        self.df_full = None

        self.load_resources()

    def check_and_fix_csv(self):
        """Ensures the requests CSV exists and has correct headers."""
        required_columns = ["timestamp", "source_url", "proposed_disease", "symptoms", "status"]
        reset_needed = False

        if not os.path.exists(REQUESTS_FILE):
            reset_needed = True
        else:
            try:
                df = pd.read_csv(REQUESTS_FILE, nrows=0)
                if not all(col in df.columns for col in required_columns):
                    reset_needed = True
            except:
                reset_needed = True

        if reset_needed:
            os.makedirs(TEMP_DIR, exist_ok=True)
            with open(REQUESTS_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(required_columns)

    def load_resources(self):
        print("\n--- Initializing AI Health Assistant ---")

        # 1. Safety Checks
        self.check_and_fix_csv()
        if not os.path.exists(ADVICE_DB_PATH):
            os.makedirs(RAW_DIR, exist_ok=True)
            with open(ADVICE_DB_PATH, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(["Disease", "Source", "Advice_1", "Advice_2", "Advice_3"])

        # 2. Load Model & Encoders
        if not MODEL_PATH or not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model missing at {MODEL_PATH}")
            print("        Please run 'train_lgbm.py' first.")
            return

        try:
            self.model = joblib.load(MODEL_PATH)
            self.le = joblib.load(LE_PATH)
            self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
            self.known_diseases = [d.lower() for d in self.le.classes_]

            if os.path.exists(FULL_DATA_PATH):
                self.df_full = pd.read_csv(FULL_DATA_PATH)
                print(f" -> Symptom Database loaded from: {FULL_DATA_PATH}")
            else:
                print(f" [WARNING] Symptom database missing at: {FULL_DATA_PATH}")
                print("           'Reverse lookup' (Disease -> Symptoms) will not work.")

            print(f" -> Model loaded ({MODEL_TYPE}). Knows {len(self.known_diseases)} diseases.")
            print(f" -> Symptom Features: {len(self.known_symptoms)} symptoms ready.")
        except Exception as e:
            print(f"[ERROR] Failed to load AI brain: {e}")

    # =========================================
    #  PART 1: DATA FETCHER
    # =========================================
    def get_advice(self, disease_name):
        clean_name = disease_name.lower().strip()
        if os.path.exists(ADVICE_DB_PATH):
            try:
                df = pd.read_csv(ADVICE_DB_PATH)
                match = df[df['Disease'].str.lower() == clean_name]
                if not match.empty:
                    row = match.iloc[0]
                    return [str(row[c]) for c in ["Advice_1", "Advice_2", "Advice_3"] if pd.notna(row[c])], "Local DB"
            except:
                pass

        print(f"     (Fetching advice for {disease_name}...)")
        self.fetch_advice(disease_name)

        if os.path.exists(ADVICE_DB_PATH):
            try:
                df = pd.read_csv(ADVICE_DB_PATH)
                match = df[df['Disease'].str.lower() == clean_name]
                if not match.empty:
                    row = match.iloc[0]
                    return [str(row[c]) for c in ["Advice_1", "Advice_2", "Advice_3"] if pd.notna(row[c])], "Downloaded"
            except:
                pass

        return ["Information available after update."], "Pending"

    def log_request(self, disease_name):
        try:
            self.check_and_fix_csv()
            with open(REQUESTS_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "User Chat",
                    disease_name,
                    "Pending",
                    "Pending"
                ])
            return True
        except:
            return False

    def get_symptoms_for_disease(self, disease_name):
        """Looks up symptoms for a given disease using the full dataset."""
        if self.df_full is None:
            return []

        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty:
            return []

        # Get the first matching row
        row = subset.iloc[0]

        # Find all columns where value is 1 (Active Symptom)
        # We check if col is in known_symptoms to avoid 'prognosis' column or indices
        active_symptoms = [col.replace("_", " ") for col in self.known_symptoms if col in row and row[col] == 1]
        return active_symptoms

    # =========================================
    #  PART 2: PARSING & LEARNING
    # =========================================
    def smart_parse(self, user_input):
        cleaned_input = re.sub(r'\b(and|or|I have|feeling|my|is|very|severe|mild)\b', '', user_input,
                               flags=re.IGNORECASE)
        raw_tokens = [s.strip().replace(" ", "_").lower() for s in cleaned_input.split(",")]
        input_dict = {col: 0 for col in self.known_symptoms}
        matched_log = []

        for token in raw_tokens:
            if not token: continue
            if token in self.known_symptoms:
                input_dict[token] = 1;
                matched_log.append(token);
                continue
            user_words = set(token.split('_'))
            reorder_match = None
            for db_sym in self.known_symptoms:
                if user_words.issubset(set(db_sym.split('_'))) and len(user_words) >= 1:
                    if reorder_match is None or len(db_sym) < len(reorder_match): reorder_match = db_sym
            if reorder_match:
                input_dict[reorder_match] = 1;
                matched_log.append(reorder_match);
                continue
            matches = difflib.get_close_matches(token, self.known_symptoms, n=1, cutoff=0.7)
            if matches:
                input_dict[matches[0]] = 1;
                matched_log.append(matches[0]);
                continue
            for db_sym in self.known_symptoms:
                if token in db_sym: input_dict[db_sym] = 1; matched_log.append(db_sym); break

        return pd.DataFrame([input_dict]), matched_log

    def verify_and_extract(self, disease_name):
        """Returns: (list_of_symptoms, actual_url)"""
        found_symptoms = []
        searchable_symptoms = [(s.replace("_", " "), s) for s in self.known_symptoms]

        # 1. CHECK WHO
        print(f"   🔎 Checking WHO for '{disease_name}'...")
        try:
            url = f"https://www.who.int/news-room/fact-sheets/detail/{disease_name.replace(' ', '-').lower()}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                text = BeautifulSoup(resp.text, 'html.parser').get_text().lower()
                for clean_sym, original_sym in searchable_symptoms:
                    parts = clean_sym.split()
                    if all(p in text for p in parts): found_symptoms.append(original_sym)

                if len(found_symptoms) >= 1:
                    print(f"     (Found in WHO: {len(found_symptoms)} matches)")
                    return list(set(found_symptoms)), url
        except:
            pass

        # 2. CHECK WIKIPEDIA
        print(f"   🔎 Checking Wikipedia (Fallback)...")
        try:
            url = f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                text = BeautifulSoup(resp.text, 'html.parser').get_text().lower()
                for clean_sym, original_sym in searchable_symptoms:
                    parts = clean_sym.split()
                    if all(p in text for p in parts): found_symptoms.append(original_sym)

                if len(found_symptoms) >= 1:
                    print(f"     (Found in Wikipedia: {len(found_symptoms)} matches)")
                    return list(set(found_symptoms)), url
        except:
            pass

        return None, None

    def fetch_advice(self, disease_name):
        if os.path.exists(ADVICE_DB_PATH):
            try:
                if disease_name.lower() in pd.read_csv(ADVICE_DB_PATH)['Disease'].str.lower().values: return
            except:
                pass

        tips = [];
        source = "N/A"
        try:
            url = f"https://www.who.int/news-room/fact-sheets/detail/{disease_name.replace(' ', '-').lower()}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for h in soup.find_all(['h2', 'h3']):
                    if any(x in h.get_text() for x in ["Prevention", "Treatment"]):
                        for t in h.find_next_siblings(['p', 'ul'])[:3]:
                            tips.append(t.get_text().strip().replace("\n", " "))
                        if tips: source = "WHO"; break
        except:
            pass

        if not tips:
            try:
                url = f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}"
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    for p in soup.find_all('p'):
                        if len(p.get_text()) > 50:
                            tips.append(f"Summary: {p.get_text().strip()[:200]}...");
                            source = "Wikipedia";
                            break
            except:
                pass

        if tips:
            while len(tips) < 3: tips.append("")
            with open(ADVICE_DB_PATH, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([disease_name.lower(), source, tips[0], tips[1], tips[2]])

    def execute_learning(self):
        print("\n[ADMIN] ⚙️  Processing Updates...")
        update_needed = False
        self.check_and_fix_csv()

        if os.path.exists(LEARNED_DATA_FILE):
            try:
                pd.read_csv(LEARNED_DATA_FILE)
            except Exception:
                print(" [SYSTEM] ⚠️ Corrupted memory detected. Resetting learned data file...")
                os.remove(LEARNED_DATA_FILE)

        if os.path.exists(REQUESTS_FILE) and os.path.exists(FEAT_PATH):
            df = pd.read_csv(REQUESTS_FILE)
            if 'status' not in df.columns: df['status'] = 'Pending'

            pending = df[df['status'] == 'Pending']
            if not pending.empty:
                print(f" [ADMIN] Found {len(pending)} pending requests.")
                unique_requests = pending['proposed_disease'].unique()

                existing_diseases = []
                if os.path.exists(LEARNED_DATA_FILE):
                    try:
                        df_learned = pd.read_csv(LEARNED_DATA_FILE)
                        if 'prognosis' in df_learned.columns:
                            existing_diseases = df_learned['prognosis'].str.lower().unique().tolist()
                    except:
                        pass

                new_entries = []
                for d_name in unique_requests:
                    clean_name = d_name.strip().lower()

                    symptoms, found_url = self.verify_and_extract(d_name)

                    mask = df['proposed_disease'] == d_name
                    if found_url: df.loc[mask, 'source_url'] = found_url

                    if clean_name in existing_diseases:
                        print(f"   ⚠️ SKIPPING '{d_name}': Already in database (URL updated).")
                        df.loc[mask, 'status'] = 'Duplicate'
                        continue

                    if symptoms:
                        print(f"   ✅ VERIFIED: '{d_name}' via {found_url}")
                        self.fetch_advice(d_name)
                        entry = {col: 0 for col in self.known_symptoms}
                        entry['prognosis'] = d_name.title()
                        for s in symptoms: entry[s] = 1
                        new_entries.append(entry)

                        df.loc[mask, 'status'] = 'Approved'
                        df.loc[mask, 'symptoms'] = ", ".join(symptoms)
                        update_needed = True
                    else:
                        print(f"   ❌ REJECTED: '{d_name}' (Verification Failed)")
                        df.loc[mask, 'status'] = 'Rejected'

                df.to_csv(REQUESTS_FILE, index=False)
                if new_entries:
                    df_new = pd.DataFrame(new_entries)
                    header = not os.path.exists(LEARNED_DATA_FILE)
                    df_new.to_csv(LEARNED_DATA_FILE, mode='a', header=header, index=False)

        if update_needed:
            print("\n [ADMIN] 🧠 Retraining Neural Model...")
            try:
                subprocess.run([sys.executable, PREPROCESS_SCRIPT], cwd=PROJECT_ROOT, check=True)
                subprocess.run([sys.executable, TRAIN_SCRIPT], cwd=PROJECT_ROOT, check=True)
                print(" [ADMIN] ✅ Update Complete.")
                return True
            except Exception as e:
                print(f" [ERROR] Training failed: {e}")
                return False
        else:
            print(" [ADMIN] No valid updates found.")
            return False

    # =========================================
    #  PART 3: MAIN LOOP
    # =========================================
    def start_chat(self):
        if not self.model:
            print("System unavailable. Please run preprocessing and training first.")
            return

        print("\n" + "=" * 50)
        print(" 🏥 AI HEALTH CHATBOT")
        print(" - Type symptoms (e.g. 'fever, rash') -> Diagnosis.")
        print(" - Type a disease (e.g. 'Malaria') -> Symptoms info.")
        print(" - Ask 'Do you know [Disease]?' -> Teach me.")
        print(" - Type 'Learn now' -> Update memory.")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input: continue
                if user_input.lower() in ['quit', 'exit', 'bye']: print("Bot: Goodbye."); break

                # A. ADMIN
                if user_input.lower() == "learn now":
                    print("\nBot: ⚙️  Learning...")
                    if self.execute_learning():
                        print("Bot: 🧠 Reloading...");
                        self.load_resources();
                        print("Bot: ✅ Ready.")
                    else:
                        print("Bot: No new data.");
                        continue

                # B. REVERSE LOOKUP (DISEASE -> SYMPTOMS)
                matches = difflib.get_close_matches(user_input.lower(), self.known_diseases, n=1, cutoff=0.9)

                # Also check exact match (case-insensitive) if difflib misses it
                if not matches and user_input.lower() in self.known_diseases:
                    matches = [user_input.lower()]

                if matches:
                    d_name = matches[0].title()
                    print(f"Bot: Found info for **{d_name}**.")

                    # Fetch symptoms
                    sym = self.get_symptoms_for_disease(d_name)
                    if sym:
                        print(f"     Symptoms: {', '.join(sym[:8])}")
                    else:
                        print("     (No symptom data found in full database for this disease)")

                    # Fetch Advice
                    adv, src = self.get_advice(d_name)
                    print(f"     Advice ({src}):");
                    [print(f"      - {a}") for a in adv]
                    continue

                # C. TEACHING
                if user_input.lower().startswith("do you know "):
                    query = user_input[12:].replace("?", "").strip()
                    if self.log_request(query):
                        print(f"Bot: Request logged. Type 'Learn now' to teach me.");
                        continue

                # D. SYMPTOM PREDICTION
                input_df, matched = self.smart_parse(user_input)
                if matched:
                    pred_id = self.model.predict(input_df)[0]
                    conf = self.model.predict_proba(input_df)[0][pred_id] * 100
                    disease = self.le.inverse_transform([pred_id])[0]
                    print(
                        f"Bot: Based on ({', '.join(matched).replace('_', ' ')}), I suspect: {disease.upper()} ({conf:.1f}%)")
                    adv, src = self.get_advice(disease)
                    print(f"     Advice ({src}):");
                    [print(f"      - {a}") for a in adv]
                else:
                    if len(user_input.split()) <= 4:
                        print(f"Bot: I don't know '{user_input}'. Type 'Do you know {user_input}?' to teach me.")
                    else:
                        print("Bot: I didn't recognize those symptoms. Try simpler terms.")

            except KeyboardInterrupt:
                print("\nBot: Exiting...");
                break
            except Exception as e:
                print(f"Bot: Error: {e}")


if __name__ == "__main__":
    HealthChatbot().start_chat()