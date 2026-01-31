#!/usr/bin/env python3
"""
Script d'initialisation de l'application
"""

import os
import sys
import shutil
from pathlib import Path

def create_structure():
    """Crée la structure de dossiers nécessaire"""
    
    print("🏗️  Création de la structure de dossiers...")
    
    directories = [
        "data",
        "ml_model",
        "uploads",
        "utils",
        ".streamlit"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    # Créer le fichier de données par défaut si absent
    data_file = Path("data/bdpoche_prec.xlsx")
    if not data_file.exists():
        print("⚠️  Attention: Aucun fichier de données trouvé dans data/")
        print("   Veuillez placer votre fichier bdpoche_prec.xlsx dans le dossier data/")
    
    print("\n✅ Structure créée avec succès!")

def check_dependencies():
    """Vérifie les dépendances"""
    
    print("\n🔍 Vérification des dépendances...")
    
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import sklearn
        import xgboost
        
        print("✅ Toutes les dépendances sont installées")
        
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("\n📦 Installation des dépendances...")
        
        # Essayer d'installer les dépendances
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dépendances installées")
        except:
            print("❌ Impossible d'installer les dépendances automatiquement")
            print("   Veuillez exécuter: pip install -r requirements.txt")

def create_sample_data():
    """Crée des données d'exemple si nécessaire"""
    
    print("\n📊 Création de données d'exemple...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Créer un DataFrame d'exemple
        data = {
            'id_poche': [f'POCHE_{i:03d}' for i in range(1, 51)],
            'quartier': ['Quartier_' + str((i % 10) + 1) for i in range(50)],
            'commune': ['Yaoundé ' + str((i % 7) + 1) for i in range(50)],
            'dens_log': np.random.randint(50, 300, 50),
            'larg_voiri': np.random.uniform(2.0, 8.0, 50),
            'mat_mur': np.random.choice(['Parpaing', 'Brique', 'Terre', 'Bois'], 50),
            'mat_toit': np.random.choice(['Tôle', 'Tuile', 'Chaume'], 50),
            'eau_bois': np.random.choice(['Réseau', 'Forage', 'Puits', 'Source'], 50),
            'elec': np.random.choice(['Oui', 'Non', 'Partiel'], 50),
            'evac_eau': np.random.choice(['Réseau', 'Fossé', 'Naturel'], 50),
            'evac_ord': np.random.choice(['Collecte', 'Dépôt', 'Brûlage'], 50),
            'risq_nat': ['Inondation, Glissement' if i % 3 == 0 else 'Aucun' for i in range(50)],
            'risq_artif': ['Haute tension' if i % 4 == 0 else 'Aucun' for i in range(50)],
            'dist_sant': np.random.uniform(0.5, 5.0, 50),
            'nbre_sant': np.random.randint(0, 3, 50),
            'dist_ecole': np.random.uniform(0.2, 3.0, 50),
            'nbre_ecole': np.random.randint(0, 2, 50)
        }
        
        df = pd.DataFrame(data)
        
        # Sauvegarder
        sample_path = Path("data/bdpoche_exemple.xlsx")
        df.to_excel(sample_path, index=False)
        
        print(f"✅ Données d'exemple créées: {sample_path}")
        print("   Vous pouvez utiliser ce fichier pour tester l'application")
        
    except Exception as e:
        print(f"❌ Erreur création données: {e}")

def main():
    """Fonction principale"""
    
    print("="*60)
    print("INITIALISATION DE L'APPLICATION STREAMLIT")
    print("="*60)
    
    # Créer la structure
    create_structure()
    
    # Vérifier les dépendances
    check_dependencies()
    
    # Créer des données d'exemple
    create_sample_data()
    
    print("\n" + "="*60)
    print("🎉 INITIALISATION TERMINÉE!")
    print("\n📋 Prochaines étapes:")
    print("1. Placez votre fichier Excel dans le dossier data/")
    print("2. Lancez l'application: streamlit run app.py")
    print("3. Allez dans Configuration > Modèle IA pour entraîner le modèle")
    print("4. Explorez les différentes pages de l'application")
    print("="*60)

if __name__ == "__main__":
    main()