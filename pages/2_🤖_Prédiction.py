"""
Page de prédiction IA pour la vulnérabilité sanitaire
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Prédiction IA", page_icon="🤖")

st.title("🤖 Prédiction de Vulnérabilité IA")

# Vérifier que le modèle est chargé
if 'predictor' not in st.session_state or st.session_state.predictor is None:
    st.warning("⚠️ Veuillez d'abord charger le modèle depuis la page d'accueil")
    
    if st.button("🔌 Charger le modèle maintenant"):
        from app import load_model
        st.session_state.predictor = load_model()
        if st.session_state.predictor is not None:
            st.success("✅ Modèle chargé avec succès!")
            st.rerun()
    st.stop()

predictor = st.session_state.predictor

# Onglets pour différents modes de prédiction
tab1, tab2, tab3 = st.tabs(["🔍 Prédiction unique", "📁 Prédiction par lot", "🎯 Simulation"])

with tab1:
    st.header("Prédiction pour une poche individuelle")
    
    # Formulaire de saisie
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informations générales")
            commune = st.selectbox("Commune", 
                                 ["Yaoundé 1", "Yaoundé 2", "Yaoundé 3", 
                                  "Yaoundé 4", "Yaoundé 5", "Yaoundé 6", "Yaoundé 7"])
            quartier = st.text_input("Quartier", "Nkolbisson")
            id_poche = st.text_input("ID Poche", "POCHE_001")
        
        with col2:
            st.subheader("Caractéristiques de l'habitat")
            densite_logements = st.slider("Densité de logements", 0, 500, 150, 10)
            largeur_voirie = st.slider("Largeur de la voirie (m)", 0.0, 10.0, 4.5, 0.5)
            materiaux_murs = st.selectbox("Matériaux des murs", 
                                         ["Parpaing", "Brique", "Terre", "Bois", "Tôle", "Mixte"])
            materiaux_toit = st.selectbox("Matériaux de toit", 
                                         ["Tôle", "Tuile", "Chaume", "Béton"])
        
        st.divider()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Services et infrastructures")
            source_eau = st.selectbox("Source d'eau", 
                                     ["Réseau CAMWATER", "Forage", "Puits", "Source/Rivière"])
            acces_electricite = st.selectbox("Accès à l'électricité", 
                                           ["Oui", "Non", "Partiel"])
            evacuation_eaux = st.selectbox("Évacuation des eaux", 
                                         ["Réseau collectif", "Fossé", "Naturel", "Aucun"])
        
        with col4:
            st.subheader("Risques")
            risque_naturel = st.multiselect("Risques naturels", 
                                           ["Inondation", "Glissement", "Érosion", 
                                            "Chute de pierres", "Aucun"])
            risque_artificiel = st.multiselect("Risques artificiels", 
                                             ["Haute tension", "Décharge", "Pollution", "Aucun"])
            distance_sante = st.slider("Distance au centre de santé (km)", 0.0, 10.0, 2.5, 0.5)
        
        submitted = st.form_submit_button("🔮 Lancer la prédiction", type="primary")
    
    if submitted:
        with st.spinner("Analyse en cours..."):
            # Préparer les données d'entrée
            input_data = {
                'commune': commune,
                'quartier': quartier,
                'id_poche': id_poche,
                'dens_log': densite_logements,
                'larg_voiri': largeur_voirie,
                'mat_mur': materiaux_murs,
                'mat_toit': materiaux_toit,
                'eau_bois': source_eau,
                'elec': acces_electricite,
                'evac_eau': evacuation_eaux,
                'risq_nat': ', '.join(risque_naturel) if risque_naturel else 'Aucun',
                'risq_artif': ', '.join(risque_artificiel) if risque_artificiel else 'Aucun',
                'dist_sant': distance_sante
            }
            
            # Convertir en DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Faire la prédiction
            try:
                result = predictor.predict_single(input_df)
                
                if result['success']:
                    # Afficher les résultats
                    st.success("✅ Prédiction terminée avec succès!")
                    
                    # Métriques principales
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Indice de Vulnérabilité", 
                                 f"{result['prediction']:.1f}/100")
                    
                    with col2:
                        niveau = result['niveau_vulnerabilite']
                        couleur = {
                            'Critique': '🔴',
                            'Élevée': '🟠', 
                            'Modérée': '🟡',
                            'Faible': '🟢'
                        }.get(niveau, '⚪')
                        st.metric("Niveau", f"{couleur} {niveau}")
                    
                    with col3:
                        st.metric("Confiance", f"{result['confidence']*100:.1f}%")
                    
                    # Facteurs clés
                    st.subheader("🔍 Facteurs clés influençant la vulnérabilité")
                    
                    if result['factors']:
                        for factor in result['factors'][:5]:
                            st.write(f"• {factor}")
                    else:
                        st.info("Aucun facteur spécifique identifié")
                    
                    # Recommandations
                    st.subheader("💡 Recommandations")
                    
                    for i, recommendation in enumerate(result['recommendations'][:5], 1):
                        st.write(f"{i}. {recommendation}")
                    
                    # Visualisation de la prédiction
                    st.subheader("📊 Visualisation")
                    
                    # Diagramme radar (simplifié)
                    scores = {
                        'Climat-Risques': min(result['prediction'] / 100 * 0.4 * 100, 100),
                        'Infrastructure': min(result['prediction'] / 100 * 0.3 * 100, 100),
                        'Accès Services': min(result['prediction'] / 100 * 0.2 * 100, 100),
                        'Habitat': min(result['prediction'] / 100 * 0.1 * 100, 100)
                    }
                    
                    fig = px.line_polar(
                        r=list(scores.values()) + [list(scores.values())[0]],
                        theta=list(scores.keys()) + [list(scores.keys())[0]],
                        line_close=True,
                        title="Répartition par dimension"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction: {str(e)}")

with tab2:
    st.header("Prédiction par lot")
    
    st.info("""
    Téléchargez un fichier Excel contenant les données des poches à analyser.
    Le fichier doit contenir les colonnes requises pour la prédiction.
    """)
    
    uploaded_file = st.file_uploader("Choisir un fichier Excel", 
                                    type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Lire le fichier
            df_upload = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Fichier chargé: {len(df_upload)} poches")
            
            # Aperçu
            with st.expander("👁️ Aperçu des données"):
                st.dataframe(df_upload.head())
            
            # Options de prédiction
            st.subheader("Paramètres de prédiction")
            batch_size = st.slider("Taille du lot", 10, 1000, 100, 10)
            
            if st.button("🚀 Lancer les prédictions sur tout le lot", type="primary"):
                with st.spinner(f"Prédiction en cours pour {len(df_upload)} poches..."):
                    # Faire les prédictions (en lot ou une par une selon la taille)
                    results = []
                    
                    for i in range(0, min(len(df_upload), batch_size)):
                        row_df = pd.DataFrame([df_upload.iloc[i]])
                        result = predictor.predict_single(row_df)
                        
                        if result['success']:
                            results.append({
                                'ID': row_df.iloc[0].get('id_poche', f"POCHE_{i}"),
                                'Quartier': row_df.iloc[0].get('quartier', 'Inconnu'),
                                'ICV': result['prediction'],
                                'Niveau': result['niveau_vulnerabilite'],
                                'Confidence': f"{result['confidence']*100:.1f}%"
                            })
                    
                    if results:
                        # Créer un DataFrame de résultats
                        results_df = pd.DataFrame(results)
                        
                        st.success(f"✅ {len(results_df)} prédictions terminées")
                        
                        # Afficher les résultats
                        st.subheader("📋 Résultats des prédictions")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Statistiques
                        st.subheader("📈 Statistiques du lot")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Moyenne ICV", f"{results_df['ICV'].mean():.1f}")
                        
                        with col2:
                            critical = len(results_df[results_df['Niveau'] == 'Critique'])
                            st.metric("Poches critiques", critical)
                        
                        with col3:
                            high_confidence = len(results_df[results_df['Confidence'].str.contains('8[0-9]|9[0-9]|100')])
                            st.metric("Haute confiance", f"{high_confidence}")
                        
                        # Distribution des niveaux
                        niveau_counts = results_df['Niveau'].value_counts()
                        fig = px.pie(values=niveau_counts.values, 
                                    names=niveau_counts.index,
                                    title="Distribution des niveaux de vulnérabilité")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Téléchargement des résultats
                        st.subheader("📥 Export des résultats")
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Télécharger en CSV",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

with tab3:
    st.header("Simulation de scénarios")
    
    st.info("""
    Simulez l'impact de différentes interventions sur la vulnérabilité.
    """)
    
    # Scénarios prédéfinis
    scenario = st.selectbox(
        "Choisir un scénario",
        ["Amélioration infrastructure", "Réduction risques", 
         "Amélioration habitat", "Scénario personnalisé"]
    )
    
    if scenario == "Amélioration infrastructure":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amelioration_eau = st.slider("Amélioration eau (%)", 0, 100, 50)
        
        with col2:
            amelioration_drainage = st.slider("Amélioration drainage (%)", 0, 100, 50)
        
        with col3:
            amelioration_dechets = st.slider("Amélioration déchets (%)", 0, 100, 50)
    
    elif scenario == "Réduction risques":
        col1, col2 = st.columns(2)
        
        with col1:
            reduction_inondation = st.slider("Réduction inondations (%)", 0, 100, 30)
        
        with col2:
            reduction_glissement = st.slider("Réduction glissements (%)", 0, 100, 40)
    
    elif scenario == "Amélioration habitat":
        col1, col2 = st.columns(2)
        
        with col1:
            amelioration_materiaux = st.slider("Amélioration matériaux (%)", 0, 100, 60)
        
        with col2:
            reduction_densite = st.slider("Réduction densité (%)", 0, 50, 20)
    
    else:  # Scénario personnalisé
        st.write("Ajustez tous les paramètres:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            score_climat = st.slider("Score climat-risques", 0, 100, 70)
            score_infra = st.slider("Score infrastructure", 0, 100, 50)
        
        with col2:
            score_acces = st.slider("Score accès services", 0, 100, 60)
            score_habitat = st.slider("Score habitat", 0, 100, 40)
    
    if st.button("🎯 Simuler l'impact", type="primary"):
        # Calculer l'impact (simulation)
        icv_base = 65.0  # Valeur de base simulée
        
        if scenario == "Amélioration infrastructure":
            impact = (amelioration_eau * 0.3 + amelioration_drainage * 0.25 + 
                     amelioration_dechets * 0.2) / 100 * 30
            icv_nouveau = max(0, icv_base - impact)
        
        elif scenario == "Réduction risques":
            impact = (reduction_inondation * 0.6 + reduction_glissement * 0.4) / 100 * 40
            icv_nouveau = max(0, icv_base - impact)
        
        elif scenario == "Amélioration habitat":
            impact = (amelioration_materiaux * 0.7 + reduction_densite * 0.3) / 100 * 10
            icv_nouveau = max(0, icv_base - impact)
        
        else:
            icv_nouveau = (score_climat * 0.4 + score_infra * 0.3 + 
                          score_acces * 0.2 + score_habitat * 0.1)
        
        # Afficher les résultats
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ICV initial", f"{icv_base:.1f}/100")
        
        with col2:
            reduction = icv_base - icv_nouveau
            st.metric("ICV après intervention", f"{icv_nouveau:.1f}/100", 
                     f"{reduction:.1f} points")
        
        # Visualisation
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Avant', 'Après'],
            y=[icv_base, icv_nouveau],
            marker_color=['#e74c3c', '#27ae60'],
            text=[f"{icv_base:.1f}", f"{icv_nouveau:.1f}"],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Impact de l'intervention",
            yaxis_title="ICV",
            yaxis_range=[0, 100]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interprétation
        st.info(f"""
        **Interprétation:**  
        L'intervention permettrait de réduire l'ICV de **{reduction:.1f} points**.
        
        **Impact estimé:**  
        - Réduction de la vulnérabilité: **{(reduction/icv_base*100):.1f}%**  
        - Changement de catégorie: {'Oui' if (icv_base > 75 and icv_nouveau <= 75) or (icv_base > 50 and icv_nouveau <= 50) else 'Non'}
        """)