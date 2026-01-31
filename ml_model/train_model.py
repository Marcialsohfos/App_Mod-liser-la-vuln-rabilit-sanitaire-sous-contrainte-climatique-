"""
Script d'entraînement du modèle ML pour Streamlit
Version adaptée et simplifiée
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

class StreamlitVulnerabilityModel:
    """Version simplifiée pour Streamlit"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.metrics = {}
        self.feature_names = []
        
    def load_data(self, filepath):
        """Charge les données Excel"""
        print(f"📂 Chargement de {filepath}")
        
        try:
            df = pd.read_excel(filepath)
            print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def preprocess_data(self, df):
        """Prétraitement des données"""
        # Copie
        df_processed = df.copy()
        
        # Standardisation des noms de colonnes
        df_processed.columns = df_processed.columns.str.strip().str.lower()
        
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
            'nbre_ecole': 'nombre_ecole',
            'etat_voir': 'etat_voirie'
        }
        
        # Appliquer le mapping
        for old, new in column_mapping.items():
            if old in df_processed.columns:
                df_processed[new] = df_processed[old]
        
        # Calculer un score de vulnérabilité simplifié
        df_processed = self.calculate_vulnerability_score(df_processed)
        
        return df_processed
    
    def calculate_vulnerability_score(self, df):
        """Calcule un score de vulnérabilité simplifié"""
        
        # Initialiser le score
        df['score_vulnerabilite'] = 0
        
        # 1. Facteur habitat (0-30 points)
        if 'materiaux_murs' in df.columns:
            df['score_murs'] = df['materiaux_murs'].apply(self._score_materiaux_murs)
            df['score_vulnerabilite'] += df['score_murs'] * 15
        
        if 'materiaux_toit' in df.columns:
            df['score_toit'] = df['materiaux_toit'].apply(self._score_materiaux_toit)
            df['score_vulnerabilite'] += df['score_toit'] * 10
        
        if 'densite_logements' in df.columns:
            densite_norm = (df['densite_logements'] - df['densite_logements'].min()) / \
                          (df['densite_logements'].max() - df['densite_logements'].min() + 1e-10)
            df['score_vulnerabilite'] += densite_norm * 5
        
        # 2. Facteur infrastructure (0-40 points)
        if 'source_eau' in df.columns:
            df['score_eau'] = df['source_eau'].apply(self._score_source_eau)
            df['score_vulnerabilite'] += df['score_eau'] * 15
        
        if 'evacuation_eaux' in df.columns:
            df['score_drainage'] = df['evacuation_eaux'].apply(self._score_drainage)
            df['score_vulnerabilite'] += df['score_drainage'] * 10
        
        if 'evacuation_ordures' in df.columns:
            df['score_dechets'] = df['evacuation_ordures'].apply(self._score_dechets)
            df['score_vulnerabilite'] += df['score_dechets'] * 10
        
        if 'electricite' in df.columns:
            df['score_electricite'] = df['electricite'].apply(self._score_electricite)
            df['score_vulnerabilite'] += df['score_electricite'] * 5
        
        # 3. Facteur risques (0-20 points)
        if 'risque_naturel' in df.columns:
            df['score_risque_nat'] = df['risque_naturel'].apply(self._score_risque_naturel)
            df['score_vulnerabilite'] += df['score_risque_nat'] * 10
        
        if 'risque_artificiel' in df.columns:
            df['score_risque_artif'] = df['risque_artificiel'].apply(self._score_risque_artificiel)
            df['score_vulnerabilite'] += df['score_risque_artif'] * 10
        
        # 4. Facteur accès (0-10 points)
        if 'distance_sante' in df.columns:
            dist_norm = 1 - (df['distance_sante'] - df['distance_sante'].min()) / \
                       (df['distance_sante'].max() - df['distance_sante'].min() + 1e-10)
            df['score_vulnerabilite'] += dist_norm * 5
        
        if 'distance_ecole' in df.columns:
            dist_ecole_norm = 1 - (df['distance_ecole'] - df['distance_ecole'].min()) / \
                            (df['distance_ecole'].max() - df['distance_ecole'].min() + 1e-10)
            df['score_vulnerabilite'] += dist_ecole_norm * 5
        
        # Normaliser sur 100
        df['icv_normalise'] = (df['score_vulnerabilite'] - df['score_vulnerabilite'].min()) / \
                             (df['score_vulnerabilite'].max() - df['score_vulnerabilite'].min() + 1e-10) * 100
        
        # Catégoriser
        df['niveau_vulnerabilite'] = pd.cut(df['icv_normalise'], 
                                           bins=[0, 25, 50, 75, 100],
                                           labels=['Faible', 'Modérée', 'Élevée', 'Critique'])
        
        return df
    
    def _score_materiaux_murs(self, x):
        """Score pour les matériaux des murs"""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        if any(word in x_str for word in ['béton', 'parpaing', 'ciment']):
            return 3
        elif any(word in x_str for word in ['brique', 'pierre']):
            return 2
        elif any(word in x_str for word in ['terre', 'bois']):
            return 1
        else:
            return 0
    
    def _score_materiaux_toit(self, x):
        """Score pour les matériaux de toit"""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        if 'tôle' in x_str:
            return 3
        elif 'tuile' in x_str or 'ciment' in x_str:
            return 2
        elif any(word in x_str for word in ['chaume', 'paille']):
            return 1
        else:
            return 0
    
    def _score_source_eau(self, x):
        """Score pour la source d'eau"""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        if any(word in x_str for word in ['camwater', 'réseau', 'adduction']):
            return 3
        elif any(word in x_str for word in ['forage', 'pompe']):
            return 2
        elif any(word in x_str for word in ['puit', 'source']):
            return 1
        else:
            return 0
    
    def _score_drainage(self, x):
        """Score pour le drainage"""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        if any(word in x_str for word in ['réseau', 'collectif', 'tout-à-l\'égout']):
            return 3
        elif any(word in x_str for word in ['fossé', 'canal']):
            return 2
        elif any(word in x_str for word in ['naturel', 'ruisseau']):
            return 1
        else:
            return 0
    
    def _score_dechets(self, x):
        """Score pour la gestion des déchets"""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        if any(word in x_str for word in ['collecte', 'ramassage', 'hysacam']):
            return 3
        elif any(word in x_str for word in ['dépôt', 'sauvage']):
            return 1
        else:
            return 0
    
    def _score_electricite(self, x):
        """Score pour l'électricité"""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        if any(word in x_str for word in ['oui', 'yes', 'connecté']):
            return 3
        elif any(word in x_str for word in ['partiel', 'intermittent']):
            return 2
        elif any(word in x_str for word in ['non', 'no']):
            return 0
        else:
            return 0
    
    def _score_risque_naturel(self, x):
        """Score pour les risques naturels"""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        score = 0
        if 'inondation' in x_str:
            score += 2
        if 'glissement' in x_str:
            score += 2
        if 'érosion' in x_str:
            score += 1
        return min(score, 3)
    
    def _score_risque_artificiel(self, x):
        """Score pour les risques artificiels"""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        score = 0
        if 'haute tension' in x_str or 'électrique' in x_str:
            score += 2
        if 'décharge' in x_str or 'pollution' in x_str:
            score += 2
        return min(score, 3)
    
    def prepare_features(self, df):
        """Prépare les features pour l'entraînement"""
        # Colonnes numériques
        numeric_features = []
        if 'densite_logements' in df.columns:
            numeric_features.append('densite_logements')
        if 'largeur_voirie' in df.columns:
            numeric_features.append('largeur_voirie')
        if 'distance_sante' in df.columns:
            numeric_features.append('distance_sante')
        if 'distance_ecole' in df.columns:
            numeric_features.append('distance_ecole')
        
        # Colonnes catégorielles à encoder
        categorical_features = []
        for col in ['materiaux_murs', 'materiaux_toit', 'source_eau', 
                   'evacuation_eaux', 'evacuation_ordures', 'risque_naturel', 
                   'risque_artificiel', 'electricite']:
            if col in df.columns:
                categorical_features.append(col)
        
        # Encodage des variables catégorielles
        for col in categorical_features:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Inconnu').astype(str))
            self.label_encoders[col] = le
        
        # Features finales
        encoded_features = [f'{col}_encoded' for col in categorical_features]
        self.feature_names = numeric_features + encoded_features
        
        X = df[self.feature_names].fillna(0)
        y = df['icv_normalise'].fillna(0)
        
        return X, y
    
    def train_model(self, X, y):
        """Entraîne le modèle"""
        print("🎯 Entraînement du modèle...")
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entraîner plusieurs modèles
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        best_score = -np.inf
        best_model = None
        
        for name, model in models.items():
            print(f"  - {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
        
        self.model = best_model
        
        # Métriques
        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'best_model': best_model_name,
            'n_features': len(self.feature_names),
            'n_samples': len(X)
        }
        
        # Importance des features
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"✅ Modèle entraîné: {best_model_name} (R²: {self.metrics['r2']:.3f})")
        return self.model
    
    def save_model(self, output_dir='ml_model'):
        """Sauvegarde le modèle"""
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sauvegarde du modèle
        model_path = f'{output_dir}/model_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Sauvegarde des objets de préprocessing
        preprocessing_path = f'{output_dir}/preprocessing_{timestamp}.pkl'
        with open(preprocessing_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance
            }, f)
        
        # Sauvegarde des métriques
        metrics_path = f'{output_dir}/metrics_{timestamp}.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        # Créer un lien vers la dernière version
        latest_model = f'{output_dir}/model_latest.pkl'
        latest_preprocessing = f'{output_dir}/preprocessing_latest.pkl'
        
        try:
            # Supprimer les anciens liens
            for file in [latest_model, latest_preprocessing]:
                if Path(file).exists():
                    Path(file).unlink()
            
            # Créer les nouveaux liens
            Path(model_path).link_to(latest_model)
            Path(preprocessing_path).link_to(latest_preprocessing)
        except:
            # Pour Windows, copier les fichiers
            import shutil
            shutil.copy2(model_path, latest_model)
            shutil.copy2(preprocessing_path, latest_preprocessing)
        
        print(f"💾 Modèle sauvegardé dans {output_dir}/")
        return model_path

def train_streamlit_model():
    """Fonction principale pour l'entraînement dans Streamlit"""
    print("="*60)
    print("ENTRAÎNEMENT DU MODÈLE POUR STREAMLIT")
    print("="*60)
    
    # Initialiser le modèle
    model = StreamlitVulnerabilityModel()
    
    # Charger les données
    df = model.load_data('data/bdpoche_prec.xlsx')
    if df is None:
        return None
    
    # Prétraiter les données
    df_processed = model.preprocess_data(df)
    
    # Préparer les features
    X, y = model.prepare_features(df_processed)
    
    # Entraîner le modèle
    model.train_model(X, y)
    
    # Sauvegarder le modèle
    model.save_model()
    
    # Afficher les résultats
    print("\n📊 RÉSULTATS:")
    print(f"  - Modèle: {model.metrics.get('best_model', 'Inconnu')}")
    print(f"  - R²: {model.metrics.get('r2', 0):.3f}")
    print(f"  - RMSE: {model.metrics.get('rmse', 0):.2f}")
    print(f"  - MAE: {model.metrics.get('mae', 0):.2f}")
    print(f"  - Features: {model.metrics.get('n_features', 0)}")
    print(f"  - Échantillons: {model.metrics.get('n_samples', 0)}")
    
    if model.feature_importance is not None:
        print("\n🔝 TOP 5 FEATURES:")
        for idx, row in model.feature_importance.head().iterrows():
            print(f"  - {row['feature']}: {row['importance']:.3f}")
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    
    return model

if __name__ == "__main__":
    train_streamlit_model()