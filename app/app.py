import streamlit as st
import os
import sys
import pandas as pd
import joblib
import re
import difflib
import requests
import csv
import subprocess
from datetime import datetime
from bs4 import BeautifulSoup

# =======================
# 1. CONFIGURATION (PORTABLE PATHS)
# =======================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# Paths for Models & Data
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp")

# Specific Files
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
REQUESTS_FILE = os.path.join(TEMP_DIR, "unverified_diseases.csv")
LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")

# Scripts to Trigger
PREPROCESS_SCRIPT = os.path.join(CURRENT_SCRIPT_DIR, "chat_bot_preprocessing.py")
TRAIN_SCRIPT = os.path.join(CURRENT_SCRIPT_DIR, "train_lgbm.py")

APP_DATA_DIR = os.path.join(CURRENT_SCRIPT_DIR, "app_data")
INFO_DB_PATH = os.path.join(APP_DATA_DIR, "who_data_clean.csv")

os.makedirs(APP_DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

DISEASE_ALIASES = {
    "common cold": "upper respiratory infection",
    "cold": "upper respiratory infection",
    "flu": "influenza",
    "sugar": "diabetes",
    "bp": "hypertension",
    "heart attack": "myocardial infarction",
    "brain stroke": "cerebrovascular accident"
}


# =======================
# 2. BACKEND LOGIC CLASS
# =======================
class MedicalAI:
    def __init__(self):
        self.model = None
        self.le = None
        self.known_symptoms = []
        self.known_diseases = []
        self.df_full = None
        self.load_resources()

    def load_resources(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.le = joblib.load(LE_PATH)
                self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
                self.known_diseases = [d.lower() for d in self.le.classes_]
                if os.path.exists(FULL_DATA_PATH):
                    self.df_full = pd.read_csv(FULL_DATA_PATH)
            except Exception as e:
                st.error(f"Error loading model files: {e}")
        else:
            st.error(
                f"⚠️ Model not found at: {MODEL_PATH}\nPlease verify your folder structure and run 'train_lgbm.py'.")

    # --- LOGGING REQUESTS ---
    def log_learning_request(self, disease_name):
        required_columns = ["timestamp", "source_url", "proposed_disease", "symptoms", "status"]
        if not os.path.exists(REQUESTS_FILE):
            with open(REQUESTS_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(required_columns)
        try:
            with open(REQUESTS_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(
                    [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "User App", disease_name, "Pending", "Pending"])
            return True
        except:
            return False

    # --- VERIFICATION & LEARNING LOGIC ---
    def verify_and_extract(self, disease_name):
        """Checks if disease exists online and has matching symptoms."""
        found_symptoms = []
        # Pre-calculate clean versions of known symptoms for matching
        searchable_symptoms = [(s.replace("_", " "), s) for s in self.known_symptoms]

        print(f"Checking '{disease_name}' online...")  # Console log

        # 1. Try WHO
        try:
            url = f"https://www.who.int/news-room/fact-sheets/detail/{disease_name.replace(' ', '-').lower()}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                text = BeautifulSoup(resp.text, 'html.parser').get_text().lower()
                for clean_sym, original_sym in searchable_symptoms:
                    parts = clean_sym.split()
                    if all(p in text for p in parts): found_symptoms.append(original_sym)
                if len(found_symptoms) >= 1: return list(set(found_symptoms)), url
        except:
            pass

        # 2. Try Wikipedia
        try:
            url = f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                text = BeautifulSoup(resp.text, 'html.parser').get_text().lower()
                for clean_sym, original_sym in searchable_symptoms:
                    parts = clean_sym.split()
                    if all(p in text for p in parts): found_symptoms.append(original_sym)
                if len(found_symptoms) >= 1: return list(set(found_symptoms)), url
        except:
            pass

        return None, None

    def execute_verification_cycle(self):
        """Processes pending requests, updates data, and runs training."""
        if not os.path.exists(REQUESTS_FILE): return False, "No requests file found."

        df = pd.read_csv(REQUESTS_FILE)
        if 'status' not in df.columns: df['status'] = 'Pending'

        pending = df[df['status'] == 'Pending']
        if pending.empty: return False, "No pending requests to verify."

        update_needed = False
        new_entries = []

        # Load existing learned data to prevent duplicates
        existing_diseases = []
        if os.path.exists(LEARNED_DATA_FILE):
            try:
                temp_df = pd.read_csv(LEARNED_DATA_FILE)
                if 'prognosis' in temp_df.columns:
                    existing_diseases = temp_df['prognosis'].str.lower().unique().tolist()
            except:
                pass

        progress_bar = st.progress(0, text="Verifying diseases...")
        total = len(pending)

        for i, (index, row) in enumerate(pending.iterrows()):
            d_name = row['proposed_disease']
            clean_name = d_name.strip().lower()

            # Check Duplicates
            if clean_name in existing_diseases:
                df.at[index, 'status'] = 'Duplicate'
                continue

            # Verify Online
            symptoms, url = self.verify_and_extract(d_name)

            if symptoms:
                # Add to new entries
                entry = {col: 0 for col in self.known_symptoms}
                entry['prognosis'] = d_name.title()
                for s in symptoms: entry[s] = 1
                new_entries.append(entry)

                # Update Status
                df.at[index, 'status'] = 'Approved'
                df.at[index, 'source_url'] = url
                df.at[index, 'symptoms'] = ", ".join(symptoms)
                update_needed = True
                self.get_advice(d_name)  # Fetch advice while we are here
            else:
                df.at[index, 'status'] = 'Rejected'

            progress_bar.progress((i + 1) / total, text=f"Checked: {d_name}")

        df.to_csv(REQUESTS_FILE, index=False)

        if update_needed and new_entries:
            # Append new data to learned file
            df_new = pd.DataFrame(new_entries)
            header = not os.path.exists(LEARNED_DATA_FILE)
            df_new.to_csv(LEARNED_DATA_FILE, mode='a', header=header, index=False)

            # Trigger Training
            progress_bar.progress(0.8, text="🧠 Retraining Neural Network...")
            try:
                subprocess.run([sys.executable, PREPROCESS_SCRIPT], cwd=PROJECT_ROOT, check=True)
                subprocess.run([sys.executable, TRAIN_SCRIPT], cwd=PROJECT_ROOT, check=True)
                progress_bar.empty()
                return True, "✅ Update Complete! I have learned the new diseases."
            except Exception as e:
                return False, f"Training Failed: {e}"

        progress_bar.empty()
        return False, "Verification done, but no valid new data was found."

    # --- STANDARD LOOKUPS ---
    def get_symptoms(self, disease_name):
        if self.df_full is None: return []
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty: return []
        row = subset.iloc[0]
        active_symptoms = [col.replace("_", " ") for col in self.known_symptoms if col in row and row[col] == 1]
        return active_symptoms

    def scrape_wikipedia(self, disease_name):
        slug = disease_name.strip().replace(" ", "_").title()
        url = f"https://en.wikipedia.org/wiki/{slug}"
        found_data = []
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for p in soup.find_all('p'):
                    if len(p.get_text()) > 50:
                        clean_text = re.sub(r'\[\d+\]', '', p.get_text().strip())
                        found_data.append(f"**Summary:** {clean_text[:300]}...")
                        break
                for h in soup.find_all(['h2', 'h3']):
                    if any(t in h.get_text() for t in ["Prevention", "Management", "Treatment"]):
                        ul = h.find_next('ul')
                        if ul:
                            for li in ul.find_all('li')[:3]:
                                found_data.append(re.sub(r'\[\d+\]', '', li.get_text().strip()))
                        break
                return found_data, url
        except:
            pass
        return [], "None"

    def get_advice(self, disease_name):
        clean_name = disease_name.lower().strip()
        if os.path.exists(INFO_DB_PATH):
            try:
                df = pd.read_csv(INFO_DB_PATH)
                match = df[df['Disease'].str.lower() == clean_name]
                if not match.empty:
                    row = match.iloc[0]
                    tips = [row[f"Precaution_{i}"] for i in range(1, 6) if pd.notna(row.get(f"Precaution_{i}"))]
                    return tips, row.get('Source', 'Local')
            except:
                pass

        found_text = []
        source = "WHO"
        slug = clean_name.replace(" ", "-")
        try:
            resp = requests.get(f"https://www.who.int/news-room/fact-sheets/detail/{slug}",
                                headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for h in soup.find_all(['h2', 'h3']):
                    if any(k in h.get_text() for k in ["Prevention", "Treatment", "Key facts"]):
                        for tag in h.find_next_siblings(['p', 'ul'])[:4]:
                            txt = tag.get_text().strip().replace('\n', ' ')
                            if len(txt) > 20: found_text.append(txt)
                        if found_text: break
        except:
            pass

        if not found_text: found_text, source = self.scrape_wikipedia(clean_name)

        if found_text:
            new_row = {"Disease": clean_name, "Source": source}
            for i, tip in enumerate(found_text[:5]): new_row[f"Precaution_{i + 1}"] = tip
            if os.path.exists(INFO_DB_PATH):
                df = pd.read_csv(INFO_DB_PATH)
            else:
                df = pd.DataFrame(
                    columns=["Disease", "Source", "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4",
                             "Precaution_5"])
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(INFO_DB_PATH, index=False)
        return found_text, source

    def predict(self, user_input):
        cleaned = re.sub(r'\b(and|or|I have|feeling|my|is)\b', '', user_input, flags=re.IGNORECASE)
        tokens = [s.strip().replace(" ", "_").lower() for s in cleaned.split(",")]
        input_dict = {col: 0 for col in self.known_symptoms}
        matched = []
        for t in tokens:
            matches = difflib.get_close_matches(t, self.known_symptoms, n=1, cutoff=0.7)
            if matches:
                input_dict[matches[0]] = 1
                matched.append(matches[0])

        if not matched: return None, [], 0
        input_df = pd.DataFrame([input_dict])
        pred_id = self.model.predict(input_df)[0]
        conf = self.model.predict_proba(input_df)[0][pred_id] * 100
        disease = self.le.inverse_transform([pred_id])[0]
        return disease, matched, conf


# =======================
# 3. STREAMLIT CHAT UI
# =======================
def main():
    st.set_page_config(page_title="Medical AI Chat", page_icon="💬", layout="centered")
    st.title("💬 AI Health Assistant")
    st.caption("Describe your symptoms (e.g., 'fever, headache') or ask about a disease.")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalAI()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Health Assistant. How can I help you today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your symptoms here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        bot = st.session_state.bot
        query_lower = prompt.lower().strip()

        # --- 1. HANDLE "VERIFY NOW" (ADMIN COMMAND) ---
        if query_lower == "verify now":
            with st.spinner("⚙️ Running verification & training pipeline..."):
                success, msg = bot.execute_verification_cycle()

            if success:
                st.success(msg)
                # Reload resources to capture new knowledge
                bot.load_resources()
                response_text = "I have updated my memory. I am ready to answer questions about the new diseases!"
            else:
                st.warning(msg)
                response_text = f"System Update: {msg}"

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()

        # --- 2. HANDLE "DO YOU KNOW" TRIGGER ---
        elif query_lower.startswith("do you know "):
            disease_request = query_lower[12:].strip("?., ")
            if disease_request:
                if bot.log_learning_request(disease_request):
                    response_text = f"📝 **Request Logged:** I have added **{disease_request}** to my queue.\n\nType **'verify now'** to verify and learn it immediately."
                else:
                    response_text = "❌ Error: I couldn't log that request right now."
            else:
                response_text = "Please specify a disease name. (e.g., 'Do you know Malaria?')"

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()

        # --- 3. STANDARD CHAT LOGIC ---
        else:
            search_term = DISEASE_ALIASES.get(query_lower, query_lower)
            response_text = ""
            disease_found = None

            matches = difflib.get_close_matches(search_term, bot.known_diseases, n=1, cutoff=0.85)
            if not matches:
                matches = [d for d in bot.known_diseases if search_term in d]

            if matches:
                disease_found = matches[0]
                response_text = f"✅ **Identification:** I found information for **{disease_found.title()}**.\n"
            else:
                disease, matched, conf = bot.predict(query_lower)
                if matched:
                    response_text = f"**Based on symptoms** ({', '.join(matched).replace('_', ' ')}), I suspect **{disease.upper()}** (Confidence: {conf:.1f}%).\n"
                    disease_found = disease
                else:
                    response_text = "❌ I couldn't recognize those symptoms. Please try using standard medical terms.\n\n*Tip: If you want to teach me a new disease, type 'Do you know [Disease]?'*"

            if disease_found:
                symptoms = bot.get_symptoms(disease_found)
                if symptoms:
                    response_text += f"\n\n**🩺 Typical Symptoms:**\n"
                    response_text += f"Common indications include: {', '.join(symptoms[:8])}.\n"

                advice, source = bot.get_advice(disease_found)
                response_text += f"\n\n--- \n**🛡️ Recommended Advice** *(Source: {source})*:\n"
                if advice:
                    for item in advice:
                        response_text += f"- {item}\n"
                else:
                    response_text += "- No specific online advice found."

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()


if __name__ == "__main__":
    main()