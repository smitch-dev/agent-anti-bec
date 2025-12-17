from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import torch
import time
import os
import csv
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. CONFIGURATION ET INITIALISATION ---
app = FastAPI(
    title="Anti-BEC AI Agent API",
    description="Système industriel de détection de fraudes au virement (BEC)",
    version="1.1.0"
)

# Chemins des fichiers
METADATA_MODEL = 'metadata_model.joblib'
PREPROCESSOR = 'metadata_preprocessor.joblib'
NLP_MODEL_PATH = './results/checkpoint-200'
LOG_FILE = "detection_history.csv"

# Dictionnaire global pour garder les modèles en mémoire vive (RAM)
models = {}

# Création du fichier de logs avec en-têtes s'il n'existe pas
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "sender", "subject", "score_rf", "score_nlp", "final_score", "decision"])

# --- 2. CHARGEMENT DES MODÈLES AU DÉMARRAGE ---
@app.on_event("startup")
def load_models():
    """Charge les modèles une seule fois pour garantir des performances optimales."""
    print("🚀 Chargement des cerveaux de l'IA en cours...")
    try:
        # Chargement Module 1 (Métadonnées)
        models['rf'] = joblib.load(METADATA_MODEL)
        models['preprocessor'] = joblib.load(PREPROCESSOR)
        
        # Chargement Module 2 (NLP)
        models['tokenizer'] = AutoTokenizer.from_pretrained("xlm-roberta-base")
        models['nlp'] = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
        
        print("✅ Tous les systèmes sont opérationnels (RF + NLP + Logs) !")
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE lors du chargement : {e}")

# --- 3. STRUCTURE DES DONNÉES D'ENTRÉE ---
class EmailInput(BaseModel):
    subject: str
    body: str
    sender_email: str
    hour: int

# --- 4. LOGIQUE D'INGÉNIERIE DES CARACTÉRISTIQUES ---
def feature_engineering(data: EmailInput):
    """Transforme l'e-mail brut en données compréhensibles par le module ML."""
    df = pd.DataFrame([data.dict()])
    df['body_length'] = df['body'].apply(len)
    df['has_attachment'] = df['body'].apply(lambda x: 1 if 'pièce jointe' in x.lower() or 'attached' in x.lower() else 0)
    df['sender_domain'] = df['sender_email'].apply(lambda x: x.split('@')[-1])
    df['sender_role'] = 'employee' # Valeur par défaut pour la simulation
    return df

# --- 5. POINT D'ENTRÉE DE L'ANALYSE (L'API) ---
@app.post("/analyze")
async def analyze_email(email: EmailInput):
    """
    Analyse un email, prend une décision et enregistre l'historique.
    """
    start_time = time.time()
    
    try:
        # A. Inférence Module Métadonnées (RF)
        df_meta = feature_engineering(email)
        X_processed = models['preprocessor'].transform(df_meta)
        proba_rf = models['rf'].predict_proba(X_processed)[:, 1][0]
        
        # B. Inférence Module Contenu (NLP)
        text = f"{email.subject} [SEP] {email.body}"
        inputs = models['tokenizer'](text, return_tensors="pt", truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = models['nlp'](**inputs)
        proba_nlp = torch.softmax(outputs.logits, dim=1)[0][1].item()
        
        # C. Calcul du Score Final et Décision
        # On peut ajuster la pondération ici (ex: 0.4*RF + 0.6*NLP)
        final_score = (proba_rf * 0.5) + (proba_nlp * 0.5)
        decision = "BEC_ALERT" if final_score >= 0.5 else "LEGITIMATE"
        
        # D. Enregistrement automatique dans les Logs
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                email.sender_email,
                email.subject,
                round(proba_rf, 4),
                round(proba_nlp, 4),
                round(final_score, 4),
                decision
            ])
            
        # E. Réponse renvoyée au client (Serveur de mail / Dashboard)
        return {
            "status": "success",
            "prediction": decision,
            "risk_score": round(final_score, 4),
            "details": {
                "metadata_score": round(proba_rf, 4),
                "nlp_score": round(proba_nlp, 4)
            },
            "execution_time_sec": round(time.time() - start_time, 3)
        }
        
    except Exception as e:
        # En cas de crash, on renvoie l'erreur 500 pour que le SI sache qu'il y a un souci
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# Route de test de santé
@app.get("/")
def home():
    return {"status": "online", "agent": "Anti-BEC AI Agent v1.1.0"}