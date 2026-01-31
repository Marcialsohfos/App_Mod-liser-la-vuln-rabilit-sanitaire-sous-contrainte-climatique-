"""
Script pour créer un modèle initial
"""

import sys
from pathlib import Path

# Créer le dossier ml_model s'il n'existe pas
Path('ml_model').mkdir(exist_ok=True)

# Vérifier si les données existent
data_path = Path('data/bdpoche_prec.xlsx')
if not data_path.exists():
    print("❌ Veuillez d'abord placer bdpoche_prec.xlsx dans le dossier data/")
    sys.exit(1)

# Importer et exécuter l'entraînement
from ml_model.train_model import train_streamlit_model

print("🤖 Création du modèle initial...")
model = train_streamlit_model()

if model:
    print("✅ Modèle créé avec succès!")
    print("\n📁 Fichiers créés:")
    print("  - ml_model/model_latest.pkl")
    print("  - ml_model/preprocessing_latest.pkl")
else:
    print("❌ Échec de la création du modèle")