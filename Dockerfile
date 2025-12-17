# 1. Utiliser une image Python légère
FROM python:3.9-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 3. Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier le fichier des dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout le projet (Code + Modèles)
COPY . .

# 6. Exposer le port que FastAPI utilise
EXPOSE 8000

# 7. Commande pour lancer l'API au démarrage du conteneur
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]