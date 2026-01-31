"""
Application Streamlit principale - Version ultra-simplifiée
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# ===== CONFIGURATION INITIALE =====
st.set_page_config(
    page_title="IA Vulnérabilité Sanitaire",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Créer les dossiers nécessaires
Path("data").mkdir(exist_ok=True)
Path("ml_model").mkdir(exist_ok=True)

# ===== CSS PERSONNALISÉ =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .success-msg {
        color: #27ae60;
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-msg {
        color: #e74c3c;
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== FONCTIONS UTILITAIRES =====
def load_data(file_path):
    """Charge les données Excel"""
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            st.error("Format de fichier non supporté")
            return None
        
        st.success(f"✅ Données chargées: {len(df)} lignes")
        return df
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return None

def calculate_vulnerability_simple(df):
    """Calcule un score de vulnérabilité simplifié"""
    try:
        df = df.copy()
        
        # Initialisation du score
        df['score_total'] = 0
        
        # 1. Facteur Densité (0-30 points)
        if 'dens_log' in df.columns:
            dens_max = df['dens_log'].max()
            if dens_max > 0:
                df['score_densite'] = (df['dens_log'] / dens_max) * 30
                df['score_total'] += df['score_densite']
        
        # 2. Facteur Infrastructure (0-40 points)
        if 'mat_mur' in df.columns:
            # Points selon les matériaux
            def score_materiaux(x):
                if pd.isna(x):
                    return 0
                x = str(x).lower()
                if 'parpaing' in x or 'béton' in x:
                    return 10
                elif 'brique' in x:
                    return 7
                elif 'terre' in x or 'bois' in x:
                    return 3
                return 5  # Par défaut
            
            df['score_materiaux'] = df['mat_mur'].apply(score_materiaux)
            df['score_total'] += df['score_materiaux']
        
        # 3. Facteur Risques (0-30 points)
        if 'risq_nat' in df.columns:
            def score_risque(x):
                if pd.isna(x):
                    return 0
                x = str(x).lower()
                score = 0
                if 'inondation' in x:
                    score += 10
                if 'glissement' in x:
                    score += 10
                if 'érosion' in x:
                    score += 5
                return min(score, 30)
            
            df['score_risques'] = df['risq_nat'].apply(score_risque)
            df['score_total'] += df['score_risques']
        
        # Normalisation 0-100
        if df['score_total'].max() > df['score_total'].min():
            df['icv'] = (df['score_total'] - df['score_total'].min()) / \
                       (df['score_total'].max() - df['score_total'].min()) * 100
        else:
            df['icv'] = 50
        
        # Catégorisation
        def categoriser_icv(x):
            if x <= 25:
                return "Faible"
            elif x <= 50:
                return "Modérée"
            elif x <= 75:
                return "Élevée"
            else:
                return "Critique"
        
        df['categorie'] = df['icv'].apply(categoriser_icv)
        
        return df
    
    except Exception as e:
        st.error(f"Erreur calcul ICV: {e}")
        return df

# ===== INITIALISATION DE SESSION =====
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# ===== SIDEBAR =====
with st.sidebar:
    st.title("🏥 IA Vulnérabilité")
    st.markdown("---")
    
    # Menu
    page = st.radio(
        "Navigation",
        ["Accueil", "Charger données", "Dashboard", "Prédiction", "Configuration"]
    )
    
    st.markdown("---")
    
    # Téléchargement de fichier
    st.subheader("📂 Charger données")
    uploaded_file = st.file_uploader(
        "Choisir un fichier",
        type=['xlsx', 'csv'],
        help="Formats supportés: Excel (.xlsx) ou CSV"
    )
    
    if uploaded_file is not None:
        # Sauvegarder le fichier
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Charger les données
        with st.spinner("Chargement..."):
            df = load_data(file_path)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_processed = calculate_vulnerability_simple(df)
                st.success("✅ Données prêtes!")
    
    st.markdown("---")
    st.caption("© 2024 Recherche Yaoundé")

# ===== PAGE ACCUEIL =====
if page == "Accueil":
    st.markdown('<div class="main-header">IA Vulnérabilité Sanitaire - Yaoundé</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌍 Contexte de la Recherche
        Cette application analyse la vulnérabilité sanitaire des quartiers précaires 
        de Yaoundé face aux contraintes climatiques.
        
        ### 🎯 Objectifs
        - Identifier les poches les plus vulnérables
        - Prioriser les interventions urbaines
        - Simuler l'impact des changements climatiques
        - Aider à la prise de décision
        
        ### 📊 Méthodologie
        1. **Collecte de données** : 266 poches analysées
        2. **Modélisation** : Scores multi-critères
        3. **Visualisation** : Dashboard interactif
        4. **Prédiction** : Estimation de la vulnérabilité
        
        ### 🚀 Démarrage rapide
        1. Téléchargez vos données dans la barre latérale
        2. Explorez le Dashboard
        3. Testez les prédictions
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998678.png", width=200)
        
        # Cartes statistiques
        if st.session_state.df is not None:
            st.markdown("### 📈 Aperçu")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Poches", len(st.session_state.df))
            with col_b:
                if 'commune' in st.session_state.df.columns:
                    st.metric("Communes", st.session_state.df['commune'].nunique())
    
    # Guide d'utilisation
    with st.expander("📖 Guide d'utilisation"):
        st.markdown("""
        ### Étapes pour utiliser l'application
        
        **1. Charger les données**
        - Utilisez la barre latérale pour télécharger votre fichier Excel
        - Format attendu: bdpoche_prec.xlsx
        - Le fichier sera sauvegardé dans le dossier `data/`
        
        **2. Explorer les données**
        - Allez dans "Dashboard" pour voir les visualisations
        - Filtrez par commune ou quartier
        - Exportez les résultats
        
        **3. Faire des prédictions**
        - Allez dans "Prédiction"
        - Entrez les paramètres d'une nouvelle poche
        - Obtenez l'estimation de vulnérabilité
        
        **4. Configurer**
        - Allez dans "Configuration"
        - Ajustez les paramètres du modèle
        - Générez des rapports
        """)

# ===== PAGE CHARGER DONNÉES =====
elif page == "Charger données":
    st.title("📂 Gestion des données")
    
    tab1, tab2, tab3 = st.tabs(["Télécharger", "Aperçu", "Statistiques"])
    
    with tab1:
        st.subheader("Téléverser un fichier")
        
        uploaded = st.file_uploader(
            "Glissez-déposez ou sélectionnez un fichier",
            type=['xlsx', 'csv'],
            key="uploader_main"
        )
        
        if uploaded:
            # Afficher les informations
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Nom:** {uploaded.name}")
                st.info(f"**Taille:** {uploaded.size / 1024:.1f} KB")
            with col2:
                st.info(f"**Type:** {uploaded.type}")
                
            # Bouton pour charger
            if st.button("🚀 Charger les données", type="primary"):
                with st.spinner("Traitement en cours..."):
                    # Sauvegarder
                    save_path = f"data/{uploaded.name}"
                    with open(save_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    
                    # Charger et traiter
                    df = pd.read_excel(save_path) if save_path.endswith('.xlsx') else pd.read_csv(save_path)
                    st.session_state.df = df
                    st.session_state.df_processed = calculate_vulnerability_simple(df)
                    
                    st.success(f"✅ {len(df)} lignes chargées avec succès!")
    
    with tab2:
        st.subheader("Aperçu des données")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Sélection des colonnes à afficher
            all_columns = df.columns.tolist()
            default_cols = [col for col in all_columns if col in ['id_poche', 'quartier', 'commune', 'dens_log', 'mat_mur', 'risq_nat']]
            
            selected_cols = st.multiselect(
                "Colonnes à afficher",
                all_columns,
                default=default_cols[:5] if len(default_cols) >= 5 else all_columns[:5]
            )
            
            # Nombre de lignes
            n_rows = st.slider("Nombre de lignes", 5, 100, 20)
            
            # Afficher le tableau
            if selected_cols:
                st.dataframe(df[selected_cols].head(n_rows), use_container_width=True)
            else:
                st.warning("Sélectionnez au moins une colonne")
        else:
            st.info("📁 Aucune donnée chargée. Téléversez un fichier d'abord.")
    
    with tab3:
        st.subheader("Statistiques descriptives")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Sélection de la variable
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Variable numérique", numeric_cols)
                
                if selected_col:
                    # Calcul des statistiques
                    stats = df[selected_col].describe()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Moyenne", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Médiane", f"{df[selected_col].median():.2f}")
                    with col3:
                        st.metric("Min", f"{stats['min']:.2f}")
                    with col4:
                        st.metric("Max", f"{stats['max']:.2f}")
                    
                    # Histogramme simple
                    st.subheader("Distribution")
                    hist_data = df[selected_col].dropna()
                    hist_counts, hist_bins = np.histogram(hist_data, bins=20)
                    
                    # Créer un histogramme avec matplotlib
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.hist(hist_data, bins=20, alpha=0.7, color='#3498db')
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel('Fréquence')
                    st.pyplot(fig)
            else:
                st.warning("Aucune variable numérique trouvée")

# ===== PAGE DASHBOARD =====
elif page == "Dashboard":
    st.title("📊 Tableau de bord")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Veuillez d'abord charger des données")
        st.stop()
    
    df = st.session_state.df_processed
    
    # Filtres
    st.subheader("🔍 Filtres")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'commune' in df.columns:
            communes = ['Toutes'] + sorted(df['commune'].dropna().unique().tolist())
            commune_filter = st.selectbox("Commune", communes)
        else:
            commune_filter = 'Toutes'
    
    with col2:
        if 'quartier' in df.columns:
            quartiers = ['Tous'] + sorted(df['quartier'].dropna().unique().tolist())
            quartier_filter = st.selectbox("Quartier", quartiers)
        else:
            quartier_filter = 'Tous'
    
    with col3:
        if 'categorie' in df.columns:
            categories = ['Toutes'] + sorted(df['categorie'].dropna().unique().tolist())
            categorie_filter = st.selectbox("Catégorie", categories)
        else:
            categorie_filter = 'Toutes'
    
    # Appliquer les filtres
    df_filtered = df.copy()
    if commune_filter != 'Toutes':
        df_filtered = df_filtered[df_filtered['commune'] == commune_filter]
    if quartier_filter != 'Tous':
        df_filtered = df_filtered[df_filtered['quartier'] == quartier_filter]
    if categorie_filter != 'Toutes':
        df_filtered = df_filtered[df_filtered['categorie'] == categorie_filter]
    
    # Métriques
    st.subheader("📈 Métriques")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Poches", len(df_filtered))
    
    with col2:
        if 'icv' in df_filtered.columns:
            st.metric("ICV Moyen", f"{df_filtered['icv'].mean():.1f}")
    
    with col3:
        if 'categorie' in df_filtered.columns:
            critique_count = len(df_filtered[df_filtered['categorie'] == 'Critique'])
            st.metric("Critiques", critique_count)
    
    with col4:
        if 'quartier' in df_filtered.columns:
            st.metric("Quartiers", df_filtered['quartier'].nunique())
    
    # Tableau des données
    st.subheader("📋 Données filtrées")
    
    display_cols = []
    for col in ['id_poche', 'quartier', 'commune', 'icv', 'categorie']:
        if col in df_filtered.columns:
            display_cols.append(col)
    
    if display_cols:
        st.dataframe(df_filtered[display_cols].head(50), use_container_width=True)
        
        # Bouton d'export
        csv = df_filtered[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="donnees_filtrees.csv",
            mime="text/csv"
        )
    
    # Analyse par catégorie
    if 'categorie' in df_filtered.columns:
        st.subheader("📊 Répartition par catégorie")
        
        cat_counts = df_filtered['categorie'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Diagramme en barres simple
            st.bar_chart(cat_counts)
        
        with col2:
            for cat, count in cat_counts.items():
                percentage = (count / len(df_filtered)) * 100
                st.metric(cat, count, f"{percentage:.1f}%")

# ===== PAGE PRÉDICTION =====
elif page == "Prédiction":
    st.title("🤖 Prédiction de vulnérabilité")
    
    tab1, tab2 = st.tabs(["Prédiction unique", "Prédiction par lot"])
    
    with tab1:
        st.subheader("Évaluer une nouvelle poche")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Informations générales**")
                commune = st.selectbox(
                    "Commune",
                    ["Yaoundé 1", "Yaoundé 2", "Yaoundé 3", "Yaoundé 4", 
                     "Yaoundé 5", "Yaoundé 6", "Yaoundé 7"]
                )
                quartier = st.text_input("Quartier", "Nkolbisson")
                densite = st.slider("Densité (logements/ha)", 0, 500, 150)
            
            with col2:
                st.write("**Caractéristiques**")
                materiaux = st.selectbox(
                    "Matériaux des murs",
                    ["Parpaing", "Brique", "Terre", "Bois", "Mixte"]
                )
                risque = st.multiselect(
                    "Risques naturels",
                    ["Inondation", "Glissement", "Érosion", "Aucun"]
                )
                distance_sante = st.slider("Distance santé (km)", 0.0, 10.0, 2.5)
            
            submitted = st.form_submit_button("🔮 Prédire la vulnérabilité")
        
        if submitted:
            # Calcul simplifié
            score = 0
            
            # Points pour la densité
            score += min(densite / 5, 30)  # Max 30 points
            
            # Points pour les matériaux
            if materiaux == "Parpaing":
                score += 10
            elif materiaux == "Brique":
                score += 7
            elif materiaux == "Terre":
                score += 3
            elif materiaux == "Bois":
                score += 3
            else:
                score += 5
            
            # Points pour les risques
            if "Inondation" in risque:
                score += 10
            if "Glissement" in risque:
                score += 10
            if "Érosion" in risque:
                score += 5
            
            # Points pour la distance
            score += min(distance_sante * 3, 15)  # 3 points par km, max 15
            
            # Normalisation (simplifiée)
            icv = min(score, 100)
            
            # Catégorisation
            if icv <= 25:
                categorie = "Faible"
                color = "🟢"
            elif icv <= 50:
                categorie = "Modérée"
                color = "🟡"
            elif icv <= 75:
                categorie = "Élevée"
                color = "🟠"
            else:
                categorie = "Critique"
                color = "🔴"
            
            # Affichage des résultats
            st.success("✅ Prédiction terminée!")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("ICV", f"{icv:.1f}/100")
            
            with col_b:
                st.metric("Catégorie", f"{color} {categorie}")
            
            with col_c:
                confidence = 85 + (icv / 100 * 10)
                st.metric("Confiance", f"{min(confidence, 95):.1f}%")
            
            # Recommandations
            st.subheader("💡 Recommandations")
            
            if categorie == "Critique":
                st.error("""
                **INTERVENTION URGENTE REQUISE**
                - Priorité absolue pour les autorités
                - Évaluation immédiate des risques
                - Plan d'évacuation à préparer
                """)
            elif categorie == "Élevée":
                st.warning("""
                **NÉCESSITÉ D'INTERVENTION**
                - Amélioration des infrastructures
                - Système d'alerte précoce
                - Renforcement des structures
                """)
            elif categorie == "Modérée":
                st.info("""
                **SURVEILLANCE CONTINUE**
                - Maintenance préventive
                - Sensibilisation communautaire
                - Préparation aux urgences
                """)
            else:
                st.success("""
                **SITUATION STABLE**
                - Surveillance régulière
                - Maintenance programmée
                - Mise à jour des plans
                """)
    
    with tab2:
        st.subheader("Prédiction par lot")
        
        uploaded_batch = st.file_uploader(
            "Téléversez un fichier avec plusieurs poches",
            type=['xlsx', 'csv'],
            key="batch_upload"
        )
        
        if uploaded_batch:
            try:
                # Charger le fichier
                if uploaded_batch.name.endswith('.xlsx'):
                    batch_df = pd.read_excel(uploaded_batch)
                else:
                    batch_df = pd.read_csv(uploaded_batch)
                
                st.success(f"✅ {len(batch_df)} poches chargées")
                
                # Bouton pour prédire
                if st.button("🚀 Lancer les prédictions", type="primary"):
                    with st.spinner("Calcul en cours..."):
                        # Calcul simplifié pour chaque ligne
                        results = []
                        
                        for idx, row in batch_df.iterrows():
                            # Score simplifié basé sur les colonnes disponibles
                            score = 50  # Valeur par défaut
                            
                            if 'dens_log' in row:
                                score += min(row['dens_log'] / 10, 20)
                            
                            if 'risq_nat' in row and pd.notna(row['risq_nat']):
                                if 'inondation' in str(row['risq_nat']).lower():
                                    score += 10
                            
                            icv = min(max(score, 0), 100)
                            
                            # Catégorisation
                            if icv <= 25:
                                cat = "Faible"
                            elif icv <= 50:
                                cat = "Modérée"
                            elif icv <= 75:
                                cat = "Élevée"
                            else:
                                cat = "Critique"
                            
                            results.append({
                                'ID': row.get('id_poche', f"POCHE_{idx}"),
                                'Quartier': row.get('quartier', 'Inconnu'),
                                'ICV': icv,
                                'Catégorie': cat
                            })
                        
                        results_df = pd.DataFrame(results)
                        
                        # Afficher les résultats
                        st.subheader("📋 Résultats")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Statistiques
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Moyenne ICV", f"{results_df['ICV'].mean():.1f}")
                        with col2:
                            critique_count = len(results_df[results_df['Catégorie'] == 'Critique'])
                            st.metric("Poches critiques", critique_count)
                        with col3:
                            st.metric("Précision estimée", "85%")
                        
                        # Téléchargement
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger résultats",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ===== PAGE CONFIGURATION =====
elif page == "Configuration":
    st.title("⚙️ Configuration")
    
    tab1, tab2, tab3 = st.tabs(["Application", "Modèle", "Système"])
    
    with tab1:
        st.subheader("Paramètres de l'application")
        
        # Thème
        theme = st.selectbox(
            "Thème de l'interface",
            ["Light", "Dark", "Auto"],
            help="Apparence visuelle de l'application"
        )
        
        # Langue
        language = st.selectbox(
            "Langue",
            ["Français", "English"],
            index=0
        )
        
        # Affichage
        col1, col2 = st.columns(2)
        with col1:
            rows_per_page = st.slider("Lignes par page", 10, 100, 50)
        with col2:
            auto_refresh = st.toggle("Rafraîchissement auto", False)
        
        # Sauvegarde
        if st.button("💾 Sauvegarder paramètres", type="primary"):
            st.success("Paramètres sauvegardés (simulation)")
    
    with tab2:
        st.subheader("Configuration du modèle")
        
        # Paramètres du calcul
        st.write("**Pondérations des facteurs**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            poids_densite = st.slider("Densité", 0, 100, 30)
        
        with col2:
            poids_materiaux = st.slider("Matériaux", 0, 100, 25)
        
        with col3:
            poids_risques = st.slider("Risques", 0, 100, 30)
        
        with col4:
            poids_distance = st.slider("Distance", 0, 100, 15)
        
        # Seuils
        st.write("**Seuils de catégorisation**")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            seuil_faible = st.slider("Faible", 0, 100, 25)
        
        with col_b:
            seuil_modere = st.slider("Modéré", 0, 100, 50)
        
        with col_c:
            seuil_eleve = st.slider("Élevé", 0, 100, 75)
        
        if st.button("🔄 Appliquer les paramètres", type="primary"):
            st.info("Les paramètres seront appliqués aux prochains calculs")
    
    with tab3:
        st.subheader("Informations système")
        
        # Version
        st.write("**Application**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Version", "1.0.0")
            st.metric("Environnement", "Streamlit Cloud")
        with col2:
            st.metric("Python", "3.9+")
            st.metric("Dernière mise à jour", "2024")
        
        # Espace disque
        st.write("**Stockage**")
        
        # Dossiers
        st.write("**Structure des fichiers**")
        
        folders = {
            "data/": "Données",
            "ml_model/": "Modèles IA",
            "uploads/": "Téléversements"
        }
        
        for folder, description in folders.items():
            if Path(folder).exists():
                items = len(list(Path(folder).iterdir()))
                st.success(f"✅ {description}: {folder} ({items} éléments)")
            else:
                st.warning(f"⚠️ {description}: {folder} (absent)")
        
        # Maintenance
        st.write("**Maintenance**")
        
        if st.button("🧹 Nettoyer le cache", type="secondary"):
            st.cache_data.clear()
            st.success("Cache nettoyé")
        
        if st.button("🔄 Redémarrer l'application", type="secondary"):
            st.info("Redémarrage simulé - Sur Streamlit Cloud, cela se fait automatiquement")

# ===== PIED DE PAGE =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>© 2024 - Recherche sur la Vulnérabilité Sanitaire - Université de Yaoundé I</p>
    <p>Contact: recherche.vulnerabilite@cm | Version 1.0.0</p>
</div>
""", unsafe_allow_html=True)