"""
Page de configuration et administration
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import shutil
from datetime import datetime

# Ajouter le chemin parent
sys.path.append(str(Path(__file__).parent.parent))

from ml_model.train_model import train_streamlit_model
from utils.helpers import validate_file_upload, convert_df_to_csv, convert_df_to_excel

st.set_page_config(page_title="Configuration", page_icon="⚙️")

st.title("⚙️ Configuration et Administration")

st.markdown("""
Cette page permet de configurer l'application, gérer les données et administrer le modèle IA.
""")

# Onglets de configuration
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Données", 
    "🤖 Modèle IA", 
    "🎨 Interface",
    "📁 Système",
    "🔄 Maintenance"
])

with tab1:
    st.header("Gestion des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Données actuelles")
        
        data_path = Path("data/bdpoche_prec.xlsx")
        if data_path.exists():
            try:
                df = pd.read_excel(data_path, nrows=5)
                st.success(f"✅ Fichier trouvé: {data_path}")
                
                # Informations sur le fichier
                file_size = data_path.stat().st_size / 1024 / 1024
                st.metric("Taille", f"{file_size:.2f} MB")
                st.metric("Lignes", "Voir aperçu")
                st.metric("Colonnes", len(df.columns))
                
                # Aperçu
                with st.expander("👁️ Aperçu des données (5 premières lignes)"):
                    st.dataframe(df)
                    
            except Exception as e:
                st.error(f"❌ Erreur de lecture: {e}")
        else:
            st.warning("⚠️ Aucun fichier de données trouvé")
    
    with col2:
        st.subheader("Téléverser de nouvelles données")
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier Excel",
            type=['xlsx', 'xls', 'csv'],
            help="Formats acceptés: Excel (.xlsx, .xls) ou CSV"
        )
        
        if uploaded_file is not None:
            # Validation
            is_valid, message = validate_file_upload(uploaded_file)
            
            if is_valid:
                st.success(f"✅ {message}")
                
                # Options de sauvegarde
                backup_option = st.radio(
                    "Options de sauvegarde",
                    ["Remplacer le fichier actuel", "Créer une sauvegarde d'abord"]
                )
                
                if st.button("💾 Sauvegarder les données", type="primary"):
                    try:
                        # Créer une sauvegarde si demandé
                        if backup_option == "Créer une sauvegarde d'abord" and data_path.exists():
                            backup_dir = Path("data/backups")
                            backup_dir.mkdir(exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            backup_path = backup_dir / f"bdpoche_prec_backup_{timestamp}.xlsx"
                            shutil.copy2(data_path, backup_path)
                            st.info(f"📦 Sauvegarde créée: {backup_path.name}")
                        
                        # Sauvegarder le nouveau fichier
                        with open(data_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.success("✅ Données sauvegardées avec succès!")
                        
                        # Recharger les données
                        if 'data' in st.session_state:
                            del st.session_state.data
                        
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
            else:
                st.error(f"❌ {message}")
    
    st.divider()
    
    # Export des données
    st.subheader("Export des données")
    
    if data_path.exists():
        export_format = st.selectbox(
            "Format d'export",
            ["CSV", "Excel", "JSON"]
        )
        
        if st.button("📤 Exporter toutes les données"):
            try:
                df = pd.read_excel(data_path)
                
                if export_format == "CSV":
                    csv = convert_df_to_csv(df)
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f"donnees_vulnerabilite_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    excel = convert_df_to_excel(df)
                    st.download_button(
                        label="📥 Télécharger Excel",
                        data=excel,
                        file_name=f"donnees_vulnerabilite_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif export_format == "JSON":
                    json_str = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Télécharger JSON",
                        data=json_str,
                        file_name=f"donnees_vulnerabilite_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                    
            except Exception as e:
                st.error(f"❌ Erreur: {e}")

with tab2:
    st.header("Gestion du modèle IA")
    
    # État actuel du modèle
    st.subheader("État du modèle")
    
    model_path = Path("ml_model/model_latest.pkl")
    preprocessing_path = Path("ml_model/preprocessing_latest.pkl")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model_path.exists():
            model_size = model_path.stat().st_size / 1024
            st.metric("Modèle principal", f"{model_size:.1f} KB", "✅ Présent")
        else:
            st.metric("Modèle principal", "❌ Absent")
    
    with col2:
        if preprocessing_path.exists():
            prep_size = preprocessing_path.stat().st_size / 1024
            st.metric("Préprocessing", f"{prep_size:.1f} KB", "✅ Présent")
        else:
            st.metric("Préprocessing", "❌ Absent")
    
    with col3:
        metrics_path = list(Path("ml_model").glob("metrics_*.json"))
        if metrics_path:
            st.metric("Métriques", f"{len(metrics_path)}", "✅ Disponibles")
        else:
            st.metric("Métriques", "❌ Absentes")
    
    # Entraînement du modèle
    st.divider()
    st.subheader("Entraînement du modèle")
    
    st.info("""
    L'entraînement crée un nouveau modèle basé sur les données actuelles.
    Cette opération peut prendre plusieurs minutes.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider(
            "Nombre d'arbres (Random Forest)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Plus d'arbres = meilleure précision mais plus lent"
        )
    
    with col2:
        test_size = st.slider(
            "Taille du jeu de test",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Pourcentage de données pour le test"
        )
    
    if st.button("🎯 Lancer l'entraînement", type="primary"):
        if not data_path.exists():
            st.error("❌ Aucune donnée disponible pour l'entraînement")
        else:
            with st.spinner("Entraînement en cours... Cela peut prendre quelques minutes."):
                try:
                    # Sauvegarde de l'ancien modèle
                    if model_path.exists():
                        backup_dir = Path("ml_model/backups")
                        backup_dir.mkdir(exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Sauvegarder l'ancien modèle
                        old_model = backup_dir / f"model_backup_{timestamp}.pkl"
                        shutil.copy2(model_path, old_model)
                        
                        # Sauvegarder l'ancien préprocessing
                        if preprocessing_path.exists():
                            old_prep = backup_dir / f"preprocessing_backup_{timestamp}.pkl"
                            shutil.copy2(preprocessing_path, old_prep)
                        
                        st.info(f"📦 Ancien modèle sauvegardé: {old_model.name}")
                    
                    # Entraîner le nouveau modèle
                    model = train_streamlit_model()
                    
                    if model:
                        st.success("✅ Modèle entraîné avec succès!")
                        
                        # Afficher les métriques
                        if hasattr(model, 'metrics'):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R²", f"{model.metrics.get('r2', 0):.3f}")
                            with col2:
                                st.metric("RMSE", f"{model.metrics.get('rmse', 0):.2f}")
                            with col3:
                                st.metric("Modèle", model.metrics.get('best_model', 'Inconnu'))
                        
                        # Mettre à jour le session state
                        if 'predictor' in st.session_state:
                            del st.session_state.predictor
                            
                    else:
                        st.error("❌ L'entraînement a échoué")
                        
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Gestion des sauvegardes
    st.divider()
    st.subheader("Sauvegardes du modèle")
    
    backup_dir = Path("ml_model/backups")
    if backup_dir.exists():
        backups = list(backup_dir.glob("model_backup_*.pkl"))
        
        if backups:
            st.write(f"📦 {len(backups)} sauvegardes disponibles:")
            
            for backup in sorted(backups, reverse=True)[:5]:  # 5 dernières
                backup_size = backup.stat().st_size / 1024
                backup_date = backup.stem.replace("model_backup_", "")
                
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"`{backup.name}`")
                with col2:
                    st.write(f"{backup_size:.1f} KB")
                with col3:
                    if st.button("🔄", key=f"restore_{backup.name}"):
                        try:
                            shutil.copy2(backup, model_path)
                            # Essayer de copier le préprocessing correspondant
                            prep_backup = backup_dir / backup.name.replace("model_backup_", "preprocessing_backup_")
                            if prep_backup.exists():
                                shutil.copy2(prep_backup, preprocessing_path)
                            st.success(f"✅ Modèle restauré: {backup.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Erreur: {e}")

with tab3:
    st.header("Configuration de l'interface")
    
    # Thème
    st.subheader("Thème")
    
    theme = st.selectbox(
        "Thème de l'interface",
        ["Light", "Dark", "Auto"],
        index=0,
        help="Light: clair, Dark: sombre, Auto: suit les préférences système"
    )
    
    # Langue
    st.subheader("Langue")
    
    language = st.selectbox(
        "Langue de l'interface",
        ["Français", "English", "Español"],
        index=0
    )
    
    # Affichage des données
    st.subheader("Affichage des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_rows = st.slider(
            "Lignes par défaut dans les tableaux",
            min_value=10,
            max_value=100,
            value=50,
            step=5
        )
    
    with col2:
        auto_refresh = st.toggle(
            "Rafraîchissement automatique",
            value=False,
            help="Rafraîchir automatiquement les données"
        )
    
    # Notifications
    st.subheader("Notifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_success = st.toggle("Messages de succès", value=True)
    
    with col2:
        show_warnings = st.toggle("Avertissements", value=True)
    
    with col3:
        show_errors = st.toggle("Messages d'erreur", value=True)
    
    # Sauvegarde des paramètres
    if st.button("💾 Sauvegarder la configuration", type="primary"):
        st.success("✅ Configuration sauvegardée (simulation)")
        st.info("Dans une version future, ces paramètres seront persistants.")

with tab4:
    st.header("Informations système")
    
    # Informations générales
    st.subheader("Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Version", "1.0.0")
        st.metric("Environnement", "Streamlit Cloud" if "STREAMLIT_SHARING" in os.environ else "Local")
        st.metric("Python", sys.version.split()[0])
    
    with col2:
        st.metric("Développeur", "Équipe Recherche Yaoundé")
        st.metric("Année", "2024")
        st.metric("Contact", "recherche.vulnerabilite@cm")
    
    # Utilisation des ressources
    st.subheader("Ressources")
    
    import psutil
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_percent = psutil.cpu_percent()
        st.metric("CPU", f"{cpu_percent}%")
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("Mémoire", f"{memory.percent}%")
    
    with col3:
        disk = psutil.disk_usage('/')
        st.metric("Disque", f"{disk.percent}%")
    
    # Fichiers et dossiers
    st.subheader("Structure des fichiers")
    
    directories = [
        ("data/", "Données"),
        ("ml_model/", "Modèles IA"),
        ("uploads/", "Téléversements"),
        (".streamlit/", "Configuration Streamlit"),
        ("pages/", "Pages de l'application")
    ]
    
    for dir_path, description in directories:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                items = len(list(path.iterdir()))
                st.success(f"✅ {description}: {dir_path} ({items} éléments)")
            else:
                st.info(f"📄 {description}: {dir_path}")
        else:
            st.warning(f"⚠️ {description}: {dir_path} (absent)")

with tab5:
    st.header("Maintenance et nettoyage")
    
    st.warning("""
    ⚠️ Attention: Ces opérations sont irréversibles.
    Assurez-vous d'avoir fait des sauvegardes si nécessaire.
    """)
    
    # Nettoyage des données temporaires
    st.subheader("Nettoyage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Vider le cache", type="secondary"):
            st.cache_data.clear()
            st.success("✅ Cache vidé")
    
    with col2:
        if st.button("🧹 Nettoyer uploads", type="secondary"):
            uploads_dir = Path("uploads")
            if uploads_dir.exists():
                for file in uploads_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                st.success("✅ Dossier uploads nettoyé")
            else:
                st.info("📁 Dossier uploads déjà vide")
    
    with col3:
        if st.button("📊 Réinitialiser données", type="secondary"):
            if st.checkbox("Je confirme vouloir réinitialiser toutes les données"):
                # Supprimer le fichier de données
                data_path.unlink()
                # Vider le session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Données réinitialisées")
                st.rerun()
    
    # Export complet
    st.divider()
    st.subheader("Export complet")
    
    st.info("""
    Crée une archive de tous les fichiers de l'application.
    Utile pour les sauvegardes ou le partage.
    """)
    
    if st.button("📦 Créer une archive complète", type="primary"):
        try:
            import zipfile
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"backup_complet_{timestamp}.zip"
            
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Ajouter tous les fichiers importants
                for root, dirs, files in os.walk("."):
                    # Exclure certains dossiers
                    exclude_dirs = {'.git', '__pycache__', '.venv', 'venv'}
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        if not file.endswith(('.pyc', '.tmp', '.log')):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, ".")
                            zipf.write(file_path, arcname)
            
            # Proposer le téléchargement
            with open(zip_filename, "rb") as f:
                st.download_button(
                    label="📥 Télécharger l'archive",
                    data=f,
                    file_name=zip_filename,
                    mime="application/zip"
                )
            
            # Supprimer le fichier temporaire
            Path(zip_filename).unlink()
            
        except Exception as e:
            st.error(f"❌ Erreur: {e}")

# Pied de page
st.divider()
st.markdown("""
**Application développée par** l'Équipe de Recherche sur la Vulnérabilité Sanitaire - Université de Yaoundé I - 2024

Pour le support technique, contactez: `recherche.vulnerabilite@cm`
""")