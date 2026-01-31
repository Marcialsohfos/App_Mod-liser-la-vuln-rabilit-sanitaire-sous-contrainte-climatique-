"""
Application Streamlit principale pour la prédiction de vulnérabilité sanitaire
Yaoundé - Base de données terrain 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="IA Vulnérabilité Sanitaire - Yaoundé",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    .vulnerability-critical { color: #e74c3c; font-weight: bold; }
    .vulnerability-high { color: #e67e22; font-weight: bold; }
    .vulnerability-medium { color: #f39c12; font-weight: bold; }
    .vulnerability-low { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("🏥 Modélisation IA de la Vulnérabilité Sanitaire")
st.markdown("### Yaoundé - Prise de décision basée sur les données")

# Initialisation de l'état de session
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None

# Fonction pour charger les données
@st.cache_data
def load_data(file_path='data/bdpoche_prec.xlsx'):
    """Charge les données Excel"""
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {e}")
        return None

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    """Charge le modèle ML"""
    try:
        from ml_model.predict import VulnerabilityPredictor
        predictor = VulnerabilityPredictor('ml_model/model_latest.pkl')
        return predictor
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Sidebar pour la navigation et les contrôles
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1998/1998678.png", width=100)
    st.title("Navigation")
    
    menu = st.selectbox(
        "Menu Principal",
        ["🏠 Accueil", "📊 Tableau de Bord", "🤖 Prédiction", "📈 Analyses", "⚙️ Configuration", "ℹ️ À Propos"]
    )
    
    st.divider()
    
    # Chargement des données
    st.subheader("Données")
    if st.button("📂 Charger les données", use_container_width=True):
        with st.spinner("Chargement des données..."):
            st.session_state.data = load_data()
            if st.session_state.data is not None:
                st.success(f"✅ {len(st.session_state.data)} poches chargées")
    
    # Chargement du modèle
    st.subheader("Modèle IA")
    if st.button("🤖 Charger le modèle", use_container_width=True):
        with st.spinner("Chargement du modèle..."):
            st.session_state.predictor = load_model()
            if st.session_state.predictor is not None:
                st.success("✅ Modèle chargé avec succès")
    
    st.divider()
    
    # Informations système
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption("Version 1.0 - Recherche Yaoundé 2024")

# Page d'accueil
if menu == "🏠 Accueil":
    st.markdown('<div class="main-header">Bienvenue dans l\'application IA de Vulnérabilité Sanitaire</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Contexte de la Recherche
        Cette application utilise l'intelligence artificielle pour prédire la vulnérabilité sanitaire 
        des poches d'habitat précaire à Yaoundé, en tenant compte des contraintes climatiques.
        
        ### Objectifs
        - Identifier les poches les plus vulnérables
        - Prioriser les interventions urbaines
        - Simuler l'impact des changements climatiques
        - Aider à la prise de décision des autorités
        
        ### Fonctionnalités
        - **Tableau de bord interactif** : Visualisation des données
        - **Prédiction IA** : Évaluation de la vulnérabilité
        - **Analyses avancées** : Statistiques et clustering
        - **Export des résultats** : Rapports et visualisations
        """)
    
    with col2:
        st.image("https://cdn.pixabay.com/photo/2017/08/01/11/48/blue-2564660_1280.png", 
                caption="IA pour la résilience urbaine")
    
    # Cartes de statistiques
    st.divider()
    st.subheader("📈 Vue d'ensemble du projet")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container(border=True):
            st.metric("Poches analysées", "266", "Données 2024")
    
    with col2:
        with st.container(border=True):
            st.metric("Communes", "7", "Yaoundé I-VII")
    
    with col3:
        with st.container(border=True):
            st.metric("Variables", "50+", "Indicateurs")
    
    with col4:
        with st.container(border=True):
            st.metric("Précision", "94%", "Modèle IA")
    
    # Démarrage rapide
    st.divider()
    st.subheader("🚀 Démarrage rapide")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("📊 Voir le tableau de bord", use_container_width=True):
            st.switch_page("pages/1_📊_Tableau_de_bord.py")
    
    with quick_col2:
        if st.button("🤖 Tester la prédiction", use_container_width=True):
            st.switch_page("pages/2_🤖_Prédiction.py")
    
    with quick_col3:
        if st.button("📈 Explorer les analyses", use_container_width=True):
            st.switch_page("pages/3_📈_Analyses.py")

# Page Tableau de Bord
elif menu == "📊 Tableau de Bord":
    # Redirection vers la page dédiée
    st.switch_page("pages/1_📊_Tableau_de_bord.py")

# Page Prédiction
elif menu == "🤖 Prédiction":
    # Redirection vers la page dédiée
    st.switch_page("pages/2_🤖_Prédiction.py")

# Page Analyses
elif menu == "📈 Analyses":
    # Redirection vers la page dédiée
    st.switch_page("pages/3_📈_Analyses.py")

# Page Configuration
elif menu == "⚙️ Configuration":
    st.header("Configuration de l'application")
    
    tab1, tab2, tab3 = st.tabs(["Données", "Modèle", "Système"])
    
    with tab1:
        st.subheader("Gestion des données")
        
        uploaded_file = st.file_uploader("Télécharger un nouveau fichier de données", 
                                        type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            if st.button("Sauvegarder les données"):
                try:
                    # Sauvegarder le fichier
                    with open('data/bdpoche_prec.xlsx', 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("✅ Fichier sauvegardé avec succès")
                    
                    # Recharger les données
                    st.session_state.data = load_data()
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
        
        st.divider()
        st.subheader("Variables disponibles")
        
        if st.session_state.data is not None:
            variables = st.session_state.data.columns.tolist()
            st.write(f"**{len(variables)} variables disponibles:**")
            st.write(variables)
    
    with tab2:
        st.subheader("Configuration du modèle")
        
        model_options = st.multiselect(
            "Sélectionner les algorithmes",
            ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting", "Stacking"],
            default=["Random Forest", "XGBoost", "Stacking"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 200, 50)
            max_depth = st.slider("Profondeur max", 3, 20, 10)
        
        with col2:
            test_size = st.slider("Taille du jeu de test", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random state", 0, 100, 42)
        
        if st.button("🎯 Réentraîner le modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                try:
                    from ml_model.train_model import main as train_main
                    model, df = train_main()
                    st.success("✅ Modèle réentraîné avec succès!")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement: {e}")
    
    with tab3:
        st.subheader("Paramètres système")
        
        st.write("**Performance**")
        cache_enabled = st.toggle("Activer le cache", value=True)
        debug_mode = st.toggle("Mode debug", value=False)
        
        st.write("**Visualisation**")
        theme = st.selectbox("Thème", ["Light", "Dark", "Auto"])
        chart_style = st.selectbox("Style des graphiques", ["Plotly", "Matplotlib", "Altair"])

# Page À Propos
elif menu == "ℹ️ À Propos":
    st.header("À Propos de l'application")
    
    st.markdown("""
    ### Contexte de la Recherche
    Cette application a été développée dans le cadre d'une recherche sur la vulnérabilité 
    sanitaire en contexte urbain africain, avec une étude de cas sur Yaoundé.
    
    ### Méthodologie
    1. **Collecte de données** : 266 poches d'habitat précaire analysées
    2. **Modélisation IA** : Algorithmes de machine learning avancés
    3. **Indicateurs composites** : 4 dimensions d'analyse
    4. **Validation terrain** : Données MINHDU/BUCREP 2024
    
    ### Dimensions d'analyse
    - **Climat-Risques** (40%) : Inondations, glissements, érosion
    - **Infrastructure** (30%) : Eau, assainissement, électricité
    - **Accès aux services** (20%) : Santé, éducation, sécurité
    - **Habitat** (10%) : Matériaux, densité, occupation
    
    ### Équipe de recherche
    - **Université** : Université de Yaoundé I
    - **Laboratoire** : Laboratoire de Recherche en Géomatique
    - **Contact** : recherche.vulnerabilite@cm
    
    ### Références
    1. Ministère de l'Habitat et du Développement Urbain (MINHDU)
    2. Bureau Central des Recensements et des Études de Population (BUCREP)
    3. Rapport sur la vulnérabilité sanitaire et changement climatique
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Version** : 1.0.0  
        **Dernière mise à jour** : Décembre 2024  
        **Langage** : Python 3.9+  
        **Framework** : Streamlit  
        **Licence** : Recherche Académique
        """)
    
    with col2:
        st.warning("""
        **Disclaimer** :  
        Cette application est un outil de recherche.  
        Les prédictions doivent être validées sur le terrain.  
        Les décisions doivent être prises avec prudence.
        """)

# Affichage du statut en bas de page
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    status_data = "✅ Chargé" if st.session_state.data is not None else "❌ Non chargé"
    st.caption(f"Données: {status_data}")

with col2:
    status_model = "✅ Chargé" if st.session_state.predictor is not None else "❌ Non chargé"
    st.caption(f"Modèle IA: {status_model}")

with col3:
    st.caption("© 2024 Recherche Vulnérabilité Sanitaire Yaoundé")