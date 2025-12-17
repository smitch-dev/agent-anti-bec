# ==============================================================================
# SCRIPT PYTHON : final_agent.py
# Agent IA Anti-BEC v1.0
# Intégration Finale des Modules Métadonnées (RF) et Contenu (XLM-RoBERTa)
# ==============================================================================

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import time
import os # Utile pour vérifier les chemins

# --- 0. FONCTIONS D'INGÉNIERIE DES CARACTÉRISTIQUES ---

def clean_domain(email):
    """ Extrait le domaine racine de l'email pour le ColumnTransformer. """
    try:
        # Sépare à l'arobase et prend le dernier élément (le domaine)
        return email.split('@')[-1]
    except:
        return 'unknown'

def feature_engineering(df):
    """ Calcule toutes les caractéristiques attendues par le pré-processeur du Module Métadonnées. """
    
    # 1. Calcul de la longueur du corps
    df['body_length'] = df['body'].apply(len)
    
    # 2. Détection de pièce jointe (simple vérification de mot-clé)
    df['has_attachment'] = df['body'].apply(lambda x: 1 if 'pièce jointe' in x.lower() or 'attached' in x.lower() or 'ci-joint' in x.lower() else 0)
    
    # 3. Extraction et nettoyage du domaine
    df['sender_domain'] = df['sender_email'].apply(clean_domain)
    
    # 4. Assignation d'un rôle (Simulé : l'entraînement du Module 2 supposait cette colonne)
    # Dans le cadre de notre simulation, tout email provient d'un 'employee' ou 'CEO' par défaut.
    # Le pré-processeur gérera cette colonne.
    df['sender_role'] = 'employee' 
    
    return df

# --- 1. CHARGEMENT DES COMPOSANTS ENTRAÎNÉS ---

# 1.1 Modèle du Module Métadonnées (Random Forest)
try:
    # Les fichiers ont été sauvegardés après correction
    model_metadata = joblib.load('metadata_model.joblib')
    print("Module Métadonnées (RF) chargé.")
except FileNotFoundError:
    print("FATAL ERROR: Le fichier 'metadata_model.joblib' est introuvable.")
    exit()

# 1.2 Pré-processeur (ColumnTransformer) du Module Métadonnées
try:
    # Le modèle a été sauvegardé avec le nom de fichier 'metadata_preprocessor.joblib'
    # La variable dans la mémoire de l'Agent sera nommée 'preprocessor'
    preprocessor = joblib.load('metadata_preprocessor.joblib') 
    print("Pré-processeur Métadonnées chargé.")
except FileNotFoundError:
    print("FATAL ERROR: Le fichier 'metadata_preprocessor.joblib' est introuvable.")
    exit()

# 1.3 Modèle et Tokenizer du Module NLP (XLM-RoBERTa)
MODEL_NAME = "xlm-roberta-base"
# ATTENTION : Chemin Corrigé vers le seul checkpoint existant !
BEST_CHECKPOINT_PATH = "./results/checkpoint-200"

try:
    if not os.path.isdir(BEST_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Dossier non trouvé: {BEST_CHECKPOINT_PATH}")
        
    tokenizer_nlp = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Nous chargeons le modèle entraîné depuis le dossier de sauvegarde
    model_nlp = AutoModelForSequenceClassification.from_pretrained(BEST_CHECKPOINT_PATH, num_labels=2)
    print(f"Module NLP (XLM-RoBERTa) chargé depuis {BEST_CHECKPOINT_PATH}.")
except Exception as e:
    print(f"FATAL ERROR: Erreur lors du chargement du Module NLP. Détails: {e}")
    exit()

# --- 2. FONCTIONS DE DÉTECTION ---

def run_metadata_module(email_df):
    """ Exécute l'inférence sur le Module Métadonnées. """
    # Utilisation de la variable correcte 'preprocessor'
    X_processed = preprocessor.transform(email_df) 
    proba = model_metadata.predict_proba(X_processed)[:, 1] # Probabilité de classe 1 (BEC)
    return proba[0]

def run_nlp_module(email_df):
    """ Exécute l'inférence sur le Module NLP. """
    # Concaténation du sujet et du corps avec le séparateur spécial [SEP]
    text = email_df.iloc[0]['subject'] + " [SEP] " + email_df.iloc[0]['body']
    
    # Tokenisation
    inputs = tokenizer_nlp(text, return_tensors="pt", truncation=True, padding="max_length")
    
    # Inférence sur GPU/CPU
    with torch.no_grad():
        outputs = model_nlp(**inputs)
    
    # Softmax pour obtenir la probabilité de chaque classe
    proba = torch.softmax(outputs.logits, dim=1)[0]
    return proba[1].item() # Probabilité de classe 1 (BEC)

def run_ai_agent(email_data):
    """ Fonction principale de l'Agent IA. """
    # Convertir les données en DataFrame
    email_df = pd.DataFrame([email_data])
    
    # 1. ÉTAPE D'INGÉNIERIE DES CARACTÉRISTIQUES (Résout le ValueError)
    email_df = feature_engineering(email_df)
    
    print("\n--- DÉMARRAGE DE L'AGENT ANTI-BEC ---")
    start_time = time.time()

    # 2. Exécution du Module Métadonnées
    proba_metadata = run_metadata_module(email_df)
    
    # 3. Exécution du Module NLP
    proba_nlp = run_nlp_module(email_df)
    
    # 4. Combinaison des Résultats (Moyenne pondérée simple)
    # Nous donnons une importance égale aux deux modules (50/50).
    final_proba = (proba_metadata * 0.5) + (proba_nlp * 0.5)
    
    # 5. Décision Finale
    threshold = 0.5
    is_bec = final_proba >= threshold
    
    end_time = time.time()
    
    print("-" * 50)
    print(f"Probabilité Métadonnées (RF): {proba_metadata:.4f}")
    print(f"Probabilité Contenu (NLP):   {proba_nlp:.4f}")
    print(f"Probabilité Finale Agent:    {final_proba:.4f}")
    print("-" * 50)
    
    status = "ALERTE BEC : BLOCAGE" if is_bec else "LÉGITIME : AUTORISÉ"
    
    print(f"STATUT FINAL: {status}")
    print(f"Temps de Traitement: {end_time - start_time:.3f} secondes")
    print("-" * 50)
    
    return status

# --- 3. SCÉNARIOS DE DÉMONSTRATION ---

# Scénario 1 : BEC Flagrant (Heure anormale + Domaine usurpé + Texte d'urgence)
bec_flagrant = {
    'subject': "URGENT - TRANSFERT IMMÉDIAT A VALIDER",
    'body': "Je suis en réunion, transférez les 30k EUR immédiatement sur le compte ci-joint. C'est le nouveau fournisseur, ne posez pas de questions. Dépêchez-vous! Je vous envoie les coordonnées du fournisseur en pièce jointe.",
    'sender_email': 'jean.dupont@compagnie-xyz.com', # Domaine usurpé
    'hour': 22 # Heure anormale (nuit)
}

# Scénario 2 : Légitime (Domaine OK + Heure OK + Texte normal)
legitime = {
    'subject': "Rapport mensuel de vente",
    'body': "Bonjour, veuillez trouver ci-joint le rapport de performance du mois d'octobre. N'hésitez pas si vous avez des questions. Cordialement, Jean Dupont.",
    'sender_email': 'jean.dupont@compagniexyz.com', # Domaine OK
    'hour': 10 # Heure OK (matin)
}

# Scénario 3 : BEC Sophistiqué (Domaine OK + Heure OK, mais Contenu Urgent)
bec_sophistique = {
    'subject': "Demande de virement en attente - Confidentialité !",
    'body': "Bonjour, en tant que PDG, je demande un virement urgent de 25,000 $ vers le compte bancaire ci-dessous. C'est une acquisition confidentielle. Ne contactez personne. Veuillez procéder immédiatement et confirmer l'exécution. Merci.",
    'sender_email': 'jean.dupont@compagniexyz.com', # Domaine OK
    'hour': 11 # Heure OK
}

# Scénario 4 : 
bec_copilot = {
    'subject': "Virement urgent requis",
    'body': "Je suis en réunion avec nos partenaires à l’étranger. Il est impératif que tu procèdes immédiatement au virement de 48 500 € vers le compte suivant :  IBAN : FR76 3000 6000 1234 5678 9012 345 Ceci est confidentiel, ne préviens personne. Merci pour ta réactivité.",
    'sender_email': 'jean.dupont@compagniexyz.com', # Domaine OK
    'hour': 20 # Heure OK
}

# Nouvel e-mail à tester :
nouvel_email = {
    'subject': "IMPORTANT : VÉRIFICATION D'UN PAIEMENT URGENT",
    'body': "Je suis sur mon téléphone. Veuillez procéder immédiatement au changement de coordonnées bancaires pour le fournisseur Alpha. C'est pour un nouveau compte confidentiel. Confirmez par retour de mail. Merci.",
    'sender_email': 'marie.legrand@votresociete.co', # Remarquez l'erreur de domaine (.co au lieu de .com)
    'hour': 18 # 18h00 (Fin de journée)
}


# --- 4. EXÉCUTION ---

if __name__ == '__main__':
    print("--- DÉMARRAGE DE LA DÉMONSTRATION DE L'AGENT BEC ---")
    
    # Test 1
    print("\n>>> TEST 1 : BEC FLAGRANT (DOMAINE/HEURE/TEXTE HORS NORME) <<<")
    run_ai_agent(bec_flagrant)
    
    # Test 2
    print("\n>>> TEST 2 : E-MAIL LÉGITIME (TOUT EST NORMAL) <<<")
    run_ai_agent(legitime)
    
    # Test 3
    print("\n>>> TEST 3 : BEC SOPHISTIQUÉ (SEUL LE TEXTE EST ANORMAL) <<<")
    run_ai_agent(bec_sophistique)

    # Test 4
    print("\n>>> TEST SCENARIO 4 <<<")
    run_ai_agent(nouvel_email) 

    # Test nouvel email
    print("\n>>> TEST NOUVEL EMAIL INCONNU <<<")
    run_ai_agent(nouvel_email) 
    