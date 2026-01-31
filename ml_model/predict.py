"""
Module de prédiction pour Streamlit
Version simplifiée
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class StreamlitPredictor:
    """Prédicteur simplifié pour Streamlit"""
    
    def __init__(self, model_path='ml_model/model_latest.pkl'):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Charge le modèle"""
        try:
            # Charger le modèle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Charger le préprocessing
            preprocessing_path = model_path.replace('model_', 'preprocessing_')
            if Path(preprocessing_path).exists():
                with open(preprocessing_path, 'rb') as f:
                    preprocessing = pickle.load(f)
                    self.scaler = preprocessing.get('scaler')
                    self.label_encoders = preprocessing.get('label_encoders', {})
                    self.feature_names = preprocessing.get('feature_names', [])
            
            print(f"✅ Modèle chargé: {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"⚠️  Erreur chargement modèle: {e}")
            print("⚠️  Création d'un modèle par défaut...")
            self._create_default_model()
    
    def _create_default_model(self):
        """Crée un modèle par défaut"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'densite_logements', 'largeur_voirie', 
            'materiaux_murs_encoded', 'materiaux_toit_encoded'
        ]
    
    def preprocess_input(self, input_df):
        """Prétraite les données d'entrée"""
        df = input_df.copy()
        
        # Standardisation des noms de colonnes
        df.columns = df.columns.str.strip().str.lower()
        
        # Mapping des colonnes
        column_mapping = {
            'dens_log': 'densite_logements',
            'larg_voiri': 'largeur_voirie',
            'mat_mur': 'materiaux_murs',
            'mat_toit': 'materiaux_toit',
            'eau_bois': 'source_eau',
            'elec': 'electricite',
            'evac_eau': 'evacuation_eaux',
            'evac_ord': 'evacuation_ordures',
            'risq_nat': 'risque_naturel',
            'risq_artif': 'risque_artificiel',
            'dist_sant': 'distance_sante',
            'nbre_sant': 'nombre_sante',
            'dist_ecole': 'distance_ecole',
            'nbre_ecole': 'nombre_ecole'
        }
        
        # Appliquer le mapping
        for old, new in column_mapping.items():
            if old in df.columns:
                df[new] = df[old]
        
        # Encoder les variables catégorielles
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Préparer les valeurs
                values = df[col].fillna('Inconnu').astype(str)
                
                # Gérer les valeurs nouvelles
                known_classes = set(encoder.classes_)
                values = values.apply(lambda x: x if x in known_classes else 'Inconnu')
                
                # Encoder
                try:
                    df[f'{col}_encoded'] = encoder.transform(values)
                except:
                    df[f'{col}_encoded'] = 0
        
        # S'assurer que toutes les features sont présentes
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Sélectionner les features dans le bon ordre
        if self.feature_names:
            df = df[self.feature_names]
        
        return df
    
    def predict(self, input_df):
        """Fait une prédiction"""
        try:
            # Prétraitement
            processed_data = self.preprocess_input(input_df)
            
            # Vérifier le nombre de features
            if len(processed_data.columns) != len(self.feature_names):
                print(f"⚠️  Mismatch features: {len(processed_data.columns)} vs {len(self.feature_names)}")
            
            # Normalisation
            if self.scaler is not None:
                processed_scaled = self.scaler.transform(processed_data)
            else:
                processed_scaled = processed_data.values
            
            # Prédiction
            prediction = self.model.predict(processed_scaled)[0]
            
            # S'assurer que la prédiction est dans [0, 100]
            prediction = max(0, min(100, prediction))
            
            return {
                'success': True,
                'prediction': float(prediction),
                'niveau': self._get_level(prediction),
                'confidence': self._calculate_confidence(prediction)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'niveau': 'Inconnu'
            }
    
    def _get_level(self, score):
        """Détermine le niveau de vulnérabilité"""
        if score <= 25:
            return "Faible"
        elif score <= 50:
            return "Modérée"
        elif score <= 75:
            return "Élevée"
        else:
            return "Critique"
    
    def _calculate_confidence(self, score):
        """Calcule la confiance (simplifiée)"""
        # Plus le score est extrême, plus la confiance est élevée
        if score < 20 or score > 80:
            return 0.95
        elif score < 30 or score > 70:
            return 0.85
        else:
            return 0.75
    
    def predict_batch(self, df):
        """Prédit un lot de données"""
        results = []
        
        for idx, row in df.iterrows():
            row_df = pd.DataFrame([row])
            pred = self.predict(row_df)
            
            if pred['success']:
                results.append({
                    'index': idx,
                    'icv': pred['prediction'],
                    'niveau': pred['niveau'],
                    'confidence': pred['confidence']
                })
        
        return pd.DataFrame(results)

# Instance globale pour Streamlit
predictor_instance = None

def get_predictor():
    """Retourne l'instance du prédicteur (singleton)"""
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = StreamlitPredictor()
    return predictor_instance