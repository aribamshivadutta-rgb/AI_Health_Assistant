import pandas as pd
import os
import datetime
import csv
import requests
import subprocess
import time
import math
from bs4 import BeautifulSoup
from datetime import timedelta
import re

# =======================
# CONFIGURATION
# =======================
# 1. Determine Paths Dynamically
SCRIPT_LOCATION = os.path.dirname(os.path.abspath(__file__)) # .../scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_LOCATION)              # .../AI_Health_Assistant

# 2. Define Directories relative to root
BASE_DIR = os.path.join(PROJECT_ROOT, "data")
SCRIPTS_DIR = SCRIPT_LOCATION 

RAW_DIR = os.path.join(BASE_DIR, "raw")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# SPECIFIC FOLDER FOR CHATBOT MEMORY
CHATBOT_RAW_DIR = os.path.join(RAW_DIR, "Chat_bot_verified_vocab_and_user_input")

# FILES
UNVERIFIED_DISEASE_FILE = os.path.join(TEMP_DIR, "unverified_diseases.csv")
UNVERIFIED_VOCAB_FILE = os.path.join(TEMP_DIR, "unverified_vocab.csv")
UNVERIFIED_INTENT_FILE = os.path.join(TEMP_DIR, "unverified_intents.csv")

# PATHS FOR SAVED DATA
LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")
VERIFIED_VOCAB_FILE = os.path.join(CHATBOT_RAW_DIR, "verified_vocab.csv")
VERIFIED_INTENT_FILE = os.path.join(CHATBOT_RAW_DIR, "chatbot_intents.csv")
VERIFIED_METADATA_FILE = os.path.join(CHATBOT_RAW_DIR, "verified_disease_sources.csv")
CHAT_LOG_FILE = os.path.join(LOGS_DIR, "chat_history.csv")

# SETTINGS
DISEASE_VERIFY_DELAY_MINUTES = 30
CHAT_VERIFY_DELAY_MINUTES = 2
TRIGGER_PHRASE = "do you know"

# SCRIPT NAMES
PREPROCESS_SCRIPT_NAME = "preprocess_lgbm.py"
TRAIN_SCRIPT_NAME = "train_lgbm.py"

# EXPANDED STOP WORDS
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it",
    "they", "what", "who", "this", "that", "am", "is", "are", "was", "were",
    "be", "have", "has", "do", "does", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "at", "by", "for", "with", "about", "to", "from",
    "in", "out", "on", "off", "up", "down", "when", "where", "why", "how",
    "all", "any", "no", "not", "so", "very", "can", "will", "just", "hello", "hi",
    "please", "verify", "now", "help", "thanks",
    "feeling", "feel", "feels", "suffering", "suffer", "got", "get",
    "severe", "mild", "acute", "chronic", "high", "low", "bad", "terrible",
    "strong", "weak", "sudden", "constant", "lot", "little", "much",
    "often", "usually", "sometimes", "always", "never", "rarely",
    "day", "night", "morning", "evening", "today", "yesterday",
    "pain", "ache", "hurts", "sore",
    "head", "stomach", "tummy", "belly", "chest", "back", "leg", "arm",
    "muscle", "joint", "bone", "skin", "body", "throat", "nose", "eye", "eyes",
    "behind", "front", "left", "right", "upper", "lower", "inside"
}

COMMON_SYMPTOMS_LIST = [
    "fever", "cough", "fatigue", "headache", "nausea", "vomiting", "diarrhea",
    "pain", "rash", "shortness of breath", "dizziness", "chills", "sore throat",
    "runny nose", "congestion", "sneezing", "muscle ache", "confusion",
    "chest pain", "tremor", "sweating", "swelling", "itching", "redness"
]

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CHATBOT_RAW_DIR, exist_ok=True)


# =======================
# PIPELINE TRIGGER
# =======================
def run_full_training_pipeline():
    print("\n[Pipeline] 🚀 New Data Verified! Initiating Auto-Training Sequence...")
    p_script = os.path.join(SCRIPTS_DIR, PREPROCESS_SCRIPT_NAME)
    t_script = os.path.join(SCRIPTS_DIR, TRAIN_SCRIPT_NAME)

    if os.path.exists(p_script):
        try:
            subprocess.run(["python", p_script], check=True)
            print("   -> [Pipeline] Pre-processing Done.")
        except subprocess.CalledProcessError:
            print(f"   -> [ERROR] Execution failed for {PREPROCESS_SCRIPT_NAME}")
            return False
    else:
        print(f"   -> [WARNING] File not found: {p_script}")
        return False

    if os.path.exists(t_script):
        try:
            subprocess.run(["python", t_script], check=True)
            print("   -> [Pipeline] Training Done. The Bot is now smarter! 🧠")
            return True
        except subprocess.CalledProcessError:
            print(f"   -> [ERROR] Execution failed for {TRAIN_SCRIPT_NAME}")
            return False
    else:
        print(f"   -> [WARNING] File not found: {t_script}")
        return False


# =======================
# TRIGGER HANDLER
# =======================
def check_trigger_and_process(user_input):
    clean_input = user_input.strip().lower()

    if clean_input == "verify now":
        print("[Manual Override] User requested immediate verification.")
        was_updated = process_pending_verifications(force=True)
        if was_updated:
            return True, "✅ **Update Complete!** I have verified the new data, learned the patterns, and retrained my model.\n\nI am ready to help! Please tell me your symptoms."
        else:
            return True, "ℹ️ I checked the queue, but there was no new data to verify. I am ready to help you with existing knowledge."

    if clean_input.startswith(TRIGGER_PHRASE):
        potential_disease = user_input[len(TRIGGER_PHRASE):].strip(" ?.,")
        if potential_disease:
            submit_unverified_disease(potential_disease)
            return True, f"I see you're asking about '{potential_disease}'. Added to queue.\nType **'verify now'** to learn it immediately."

    if " means " in clean_input:
        parts = clean_input.split(" means ")
        if len(parts) == 2:
            submit_unverified_vocabulary(parts[0].strip(), parts[1].strip())
            return True, f"Thanks! I'll remember that '{parts[0].strip()}' means '{parts[1].strip()}'."

    if " reply with " in clean_input:
        parts = clean_input.split(" reply with ")
        if len(parts) == 2:
            submit_unverified_intent(parts[0].strip(), parts[1].strip())
            return True, f"Got it! When you say '{parts[0].strip()}', I will reply: '{parts[1].strip()}'."

    words = re.findall(r'\b\w+\b', clean_input)
    potential_new_words = []

    for w in words:
        if w in STOP_WORDS: continue
        if len(w) <= 3: continue
        if w in COMMON_SYMPTOMS_LIST: continue
        is_part_of_symptom = False
        for known_symptom in COMMON_SYMPTOMS_LIST:
            if w in known_symptom.split():
                is_part_of_symptom = True
                break
        if is_part_of_symptom: continue
        potential_new_words.append(w)

    if potential_new_words:
        candidate_word = max(potential_new_words, key=len)
        found, definition = _auto_define_word(candidate_word)
        if found:
            submit_unverified_vocabulary(candidate_word, definition)
            return True, f"I noticed a new word '{candidate_word}'. I looked it up: '{definition}'. Saved to memory."

    return False, None


# =======================
# HELPERS
# =======================
def _auto_define_word(word):
    print(f"[Auto-Learn] Trying to define: '{word}'...")
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            try:
                definition = data[0]['meanings'][0]['definitions'][0]['definition']
                return True, definition
            except (KeyError, IndexError):
                return False, ""
    except Exception:
        return False, ""
    return False, ""


def log_interaction(user_input_symptoms, predicted_disease, confidence, user_verification):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp, "user_input": user_input_symptoms,
        "ai_prediction": predicted_disease, "confidence": confidence, "status": user_verification
    }
    _append_to_csv(CHAT_LOG_FILE, log_entry)


def submit_unverified_disease(disease_name, symptoms_list=None):
    print(f"\n[Staging] Submitting unverified disease: {disease_name}")
    symptoms_str = "PENDING_EXTRACTION" if symptoms_list is None else "|".join(symptoms_list)
    entry = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'source_url_or_note': "Pending Auto-Discovery",
        'proposed_disease': disease_name,
        'symptoms_list': symptoms_str,
        'status': 'Pending'
    }
    _append_to_csv(UNVERIFIED_DISEASE_FILE, entry)
    print(f"[Success] Queued. Waiting for auto-check or 'verify now' command.")


def submit_unverified_vocabulary(word, meaning):
    print(f"\n[Staging] Submitting new vocabulary: {word}")
    entry = {'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             'word': word, 'meaning_or_synonym': meaning}
    _append_to_csv(UNVERIFIED_VOCAB_FILE, entry)
    print(f"[Success] Queued.")


def submit_unverified_intent(trigger, response):
    print(f"\n[Staging] Submitting new intent: {trigger} -> {response}")
    entry = {'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             'intent_trigger': trigger, 'bot_response': response}
    _append_to_csv(UNVERIFIED_INTENT_FILE, entry)
    print(f"[Success] Queued.")


def _search_and_verify_disease(disease_name):
    print(f"[Auto-Verify] Searching internet for: '{disease_name}'...")
    headers = {'User-Agent': 'HealthAssistantBot/1.0 (Research Project)'}
    slug = disease_name.lower().strip().replace(' ', '-')

    try:
        who_resp = requests.get(f"https://www.who.int/health-topics/{slug}", headers=headers, timeout=5)
        if who_resp.status_code == 200:
            symptoms = _extract_symptoms_from_html(who_resp.text)
            if symptoms: return True, f"https://www.who.int/health-topics/{slug}", "Verified on WHO", symptoms
    except:
        pass

    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": disease_name, "limit": 1, "namespace": 0, "format": "json"}
        resp = requests.get(search_url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if data[1]:
            best_url = data[3][0]
            page_resp = requests.get(best_url, headers=headers, timeout=10)
            if page_resp.status_code == 200:
                symptoms = _extract_symptoms_from_html(page_resp.text)
                if symptoms: return True, best_url, "Verified on Wikipedia", symptoms
    except Exception as e:
        return False, "", f"Error: {e}", []
    return False, "", "No matching reputable sources found", []


def _extract_symptoms_from_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text().lower()
    found = []
    for s in COMMON_SYMPTOMS_LIST:
        if s in text:
            found.append(s)
    return list(set(found))


# =======================
# AUTO-VERIFICATION LOGIC
# =======================
def process_pending_verifications(force=False):
    print("\n[System] Checking for pending auto-verifications...")
    now = datetime.datetime.now()
    if force:
        print("[System] ⚡ FORCE MODE: Ignoring time delays.")
        disease_threshold = now
        chat_threshold = now
    else:
        disease_threshold = now - timedelta(minutes=DISEASE_VERIFY_DELAY_MINUTES)
        chat_threshold = now - timedelta(minutes=CHAT_VERIFY_DELAY_MINUTES)

    data_changed = False

    if os.path.exists(UNVERIFIED_DISEASE_FILE):
        if _process_diseases_log_mode(UNVERIFIED_DISEASE_FILE, disease_threshold):
            data_changed = True
    if os.path.exists(UNVERIFIED_VOCAB_FILE):
        _process_generic(UNVERIFIED_VOCAB_FILE, chat_threshold, _promote_vocab)
    if os.path.exists(UNVERIFIED_INTENT_FILE):
        _process_generic(UNVERIFIED_INTENT_FILE, chat_threshold, _promote_intent)

    if data_changed:
        return run_full_training_pipeline()
    else:
        print("[System] No new valid diseases found. No training needed.")
        return False


def _process_diseases_log_mode(source_file, threshold_time):
    try:
        df = pd.read_csv(source_file)
        if df.empty: return False

        if 'status' not in df.columns:
            df['status'] = 'Pending'

        promoted_any = False
        for index, row in df.iterrows():
            if row['status'] in ['Verified', 'Failed']: continue

            timestamp_dt = pd.to_datetime(row['timestamp'])
            if timestamp_dt < threshold_time:
                verified, url, msg, symptoms = _search_and_verify_disease(row['proposed_disease'])

                if verified:
                    df.at[index, 'status'] = 'Verified'
                    df.at[index, 'source_url_or_note'] = url
                    df.at[index, 'symptoms_list'] = "|".join(symptoms)

                    promo_data = row.to_dict()
                    promo_data['symptoms_list'] = "|".join(symptoms)
                    promo_data['source_url_or_note'] = url
                    _promote_disease(pd.Series(promo_data))
                    promoted_any = True
                    print(f"   -> Verified '{row['proposed_disease']}'. Log updated.")
                else:
                    df.at[index, 'status'] = 'Failed'
                    df.at[index, 'source_url_or_note'] = msg
                    print(f"   -> Failed '{row['proposed_disease']}'. Log updated.")

        df.to_csv(source_file, index=False)
        return promoted_any
    except Exception as e:
        print(f"[Error] Disease process log mode: {e}")
        return False


def _process_generic(source_file, threshold_time, processor_func):
    try:
        df = pd.read_csv(source_file)
        if df.empty: return
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        ready = df[df['timestamp_dt'] < threshold_time]
        pending = df[df['timestamp_dt'] >= threshold_time]
        if not ready.empty:
            for _, row in ready.iterrows(): processor_func(row)
        pending.drop(columns=['timestamp_dt'], errors='ignore').to_csv(source_file, index=False)
    except Exception as e:
        print(f"[Error] Generic process: {e}")


def _promote_disease(row):
    disease = row['proposed_disease']
    symptoms = row['symptoms_list'].split("|") if row['symptoms_list'] else []
    new_entry = {'prognosis': disease}
    for s in symptoms:
        clean_symptom = s.strip().lower().replace(' ', '_')
        new_entry[clean_symptom] = 1
    _append_to_csv_smart(LEARNED_DATA_FILE, new_entry)
    _append_to_csv(VERIFIED_METADATA_FILE, {'disease': disease, 'source_url': row['source_url_or_note'],
                                            'verified_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


def _promote_vocab(row):
    _append_to_csv(VERIFIED_VOCAB_FILE, {'word': row['word'], 'meaning_or_synonym': row['meaning_or_synonym'],
                                         'verified_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


def _promote_intent(row):
    _append_to_csv(VERIFIED_INTENT_FILE, {'intent_trigger': row['intent_trigger'], 'bot_response': row['bot_response'],
                                          'verified_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


def _append_to_csv(filepath, dict_data):
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=dict_data.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(dict_data)


def _append_to_csv_smart(filepath, new_row_dict):
    new_df = pd.DataFrame([new_row_dict])
    if os.path.exists(filepath):
        try:
            combined = pd.concat([pd.read_csv(filepath), new_df], ignore_index=True)
        except:
            combined = new_df
    else:
        combined = new_df
    combined.fillna(0, inplace=True)
    combined.to_csv(filepath, index=False)


def generate_starter_data():
    if not os.path.exists(VERIFIED_VOCAB_FILE):
        with open(VERIFIED_VOCAB_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=['word', 'meaning_or_synonym', 'verified_at']).writeheader()
    if not os.path.exists(VERIFIED_INTENT_FILE):
        with open(VERIFIED_INTENT_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=['intent_trigger', 'bot_response', 'verified_at']).writeheader()
    if not os.path.exists(UNVERIFIED_DISEASE_FILE):
        with open(UNVERIFIED_DISEASE_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=['timestamp', 'source_url_or_note', 'proposed_disease', 'symptoms_list',
                                          'status']).writeheader()


# =======================
# EXECUTION (DEEP SLEEP MONITOR)
# =======================
if __name__ == "__main__":
    print("=============================================")
    print("   AI HEALTH ASSISTANT: DEEP SLEEP MONITOR   ")
    print("=============================================")
    print(f"[*] Monitor Mode: ACTIVE")
    print(f"[*] Cycle Time: {DISEASE_VERIFY_DELAY_MINUTES} minutes")

    generate_starter_data()

    while True:
        try:
            # 1. Do the work
            process_pending_verifications(force=False)

            # 2. Calculate Sleep Time
            wait_seconds = DISEASE_VERIFY_DELAY_MINUTES * 60

            if os.path.exists(UNVERIFIED_DISEASE_FILE):
                try:
                    df = pd.read_csv(UNVERIFIED_DISEASE_FILE)
                    if 'status' in df.columns:
                        pending_df = df[df['status'] == 'Pending']
                    else:
                        pending_df = df

                    if not pending_df.empty:
                        pending_df = pending_df.copy()
                        pending_df['timestamp_dt'] = pd.to_datetime(pending_df['timestamp'])
                        oldest_entry = pending_df['timestamp_dt'].min()
                        target_wake_time = oldest_entry + timedelta(minutes=DISEASE_VERIFY_DELAY_MINUTES)
                        now = datetime.datetime.now()
                        remaining = (target_wake_time - now).total_seconds()

                        if remaining > 0:
                            wait_seconds = remaining
                            print(f"[System] Next item ready at {target_wake_time.strftime('%H:%M:%S')}")
                        else:
                            wait_seconds = 1
                except Exception as e:
                    print(f"[Error] Scheduling check: {e}")

            # 3. Deep Sleep
            minutes_sleep = math.ceil(wait_seconds / 60)
            print(f"[System] 💤 Logger falling asleep for {minutes_sleep} minutes...")
            time.sleep(wait_seconds)

        except KeyboardInterrupt:
            print("\n[System] Waking up to stop. Goodbye!")
            break
        except Exception as e:
            print(f"[Critical Error] {e}")
            time.sleep(60)