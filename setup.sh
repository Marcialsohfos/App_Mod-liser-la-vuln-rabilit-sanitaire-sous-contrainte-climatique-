#!/bin/bash

# Installation pour Streamlit Cloud

echo "🔧 Installation de l'application..."

# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances
pip install -r requirements.txt

# Créer les dossiers nécessaires
mkdir -p data
mkdir -p ml_model
mkdir -p uploads
mkdir -p .streamlit

echo "✅ Installation terminée"
echo ""
echo "🚀 Pour démarrer l'application:"
echo "   streamlit run app.py"