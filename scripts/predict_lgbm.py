import os
import pandas as pd
import joblib
import re
import difflib
import sys
import requests
from bs4 import BeautifulSoup
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# =======================
# 1. CONFIGURATION
# =======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# [CRITICAL UPDATE] Point to 'chat_bot_clean'
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
INFO_DB_PATH = os.path.join(DATA_DIR, "who_data_clean.csv")

sys.path.append(SCRIPT_DIR)

# 🧠 ALIAS DICTIONARY
DISEASE_ALIASES = {
    "tb": "tuberculosis",
    "common cold": "upper respiratory infection",
    "cold": "upper respiratory infection",
    "flu": "influenza",
    "piles": "hemorrhoids",
    "sugar": "diabetes",
    "heart attack": "myocardial infarction",
    "brain stroke": "cerebrovascular accident"
}


class MedicalAI:
    def __init__(self):
        self.model = None
        self.le = None
        self.known_symptoms = []
        self.known_diseases = []
        self.load_resources()

    def load_resources(self):
        print("\n--- Initializing Medical AI ---")
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model missing at {MODEL_PATH}")
            return
        try:
            self.model = joblib.load(MODEL_PATH)
            self.le = joblib.load(LE_PATH)
            self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
            self.known_diseases = [d.lower() for d in self.le.classes_]
            print(f" -> Model loaded. Knows {len(self.known_diseases)} diseases.")
        except Exception as e:
            print(f"[ERROR] Failed to load AI brain: {e}")

    # ... (Scraping logic is identical to original, just ensuring paths above are correct) ...
    
    def scrape_wikipedia(self, disease_name):
        # (Same Wikipedia logic)
        variations = [
            disease_name.strip().replace(" ", "_").title(),
            disease_name.strip().replace(" ", "_").title() + "_(disease)"
        ]
        found_data = []
        target_headers = ["Prevention", "Treatment", "Management", "Mitigation", "Therapy", "Self-care"]
        try:
            for slug in variations:
                url = f"https://en.wikipedia.org/wiki/{slug}"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        clean_text = re.sub(r'\[.*?\]', '', text)
                        if len(clean_text) > 50 and "refer to" not in clean_text:
                            found_data.append(f"SUMMARY: {clean_text[:250]}...")
                            break
                    for h in soup.find_all(['h2', 'h3']):
                        header_text = h.get_text().strip()
                        if any(t in header_text for t in target_headers):
                            curr_elem = h.find_next_sibling()
                            while curr_elem and curr_elem.name not in ['h2', 'h3']:
                                if curr_elem.name == 'p':
                                    text = re.sub(r'\[.*?\]', '', curr_elem.get_text().strip())
                                    if len(text) > 40: found_data.append(text)
                                elif curr_elem.name == 'ul':
                                    for li in curr_elem.find_all('li'):
                                        text = re.sub(r'\[.*?\]', '', li.get_text().strip())
                                        if len(text) > 10: found_data.append(text)
                                curr_elem = curr_elem.find_next_sibling()
                                if len(found_data) > 6: break
                    if found_data: return found_data, url
        except Exception: pass
        return [], "None"

    def get_disease_info(self, disease_name):
        clean_name = disease_name.lower().strip()
        if os.path.exists(INFO_DB_PATH):
            try:
                df = pd.read_csv(INFO_DB_PATH)
                match = df[df['Disease'].str.lower() == clean_name]
                if not match.empty:
                    row = match.iloc[0]
                    precautions = [row[f"Precaution_{i}"] for i in range(1, 6) if pd.notna(row.get(f"Precaution_{i}"))]
                    return precautions, row.get('Source', 'Local')
            except: pass

        found_text, source = self.scrape_wikipedia(clean_name)
        if found_text:
            new_row = {"Disease": clean_name, "Source": source}
            for i, tip in enumerate(found_text[:5]): new_row[f"Precaution_{i + 1}"] = tip
            if os.path.exists(INFO_DB_PATH): df = pd.read_csv(INFO_DB_PATH)
            else: df = pd.DataFrame(columns=["Disease", "Source", "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4", "Precaution_5"])
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(INFO_DB_PATH, index=False)
        return found_text, source

    def smart_parse(self, user_input):
        cleaned_input = re.sub(r'\b(and|or|I have|feeling|my|is|very|severe)\b', '', user_input, flags=re.IGNORECASE)
        raw_tokens = [s.strip().replace(" ", "_").lower() for s in cleaned_input.split(",")]
        input_dict = {col: 0 for col in self.known_symptoms}
        matched_log = []
        for token in raw_tokens:
            if not token: continue
            matches = difflib.get_close_matches(token, self.known_symptoms, n=1, cutoff=0.7)
            if matches:
                input_dict[matches[0]] = 1
                matched_log.append(matches[0])
        return pd.DataFrame([input_dict]), matched_log

    def start_chat(self):
        if not self.model: return
        print("\n🏥 MEDICAL AI ASSISTANT (WHO + Wiki Enabled)")
        while True:
            user_input = input("\nPatient: ").strip().lower()
            if user_input in ['quit', 'exit']: break
            
            search_term = DISEASE_ALIASES.get(user_input, user_input)
            matches = difflib.get_close_matches(search_term, self.known_diseases, n=1, cutoff=0.85)
            
            if matches:
                self.show_details(matches[0])
            else:
                input_df, matched = self.smart_parse(user_input)
                if matched:
                    pred_id = self.model.predict(input_df)[0]
                    conf = self.model.predict_proba(input_df)[0][pred_id] * 100
                    disease = self.le.inverse_transform([pred_id])[0]
                    print(f"Bot: I suspect: {disease.upper()} ({conf:.1f}%)")
                    self.show_details(disease)
                else:
                    print("Bot: Symptoms not recognized.")

    def show_details(self, disease):
        advice, source = self.get_disease_info(disease)
        print("-" * 40)
        if advice:
            print(f"🛡️ ADVICE FOR {disease.upper()} (Source: {source}):")
            for item in advice: print(f"   • {item}")
        else:
            print("   Please consult a doctor.")
        print("-" * 40)

if __name__ == "__main__":
    bot = MedicalAI()
    bot.start_chat()