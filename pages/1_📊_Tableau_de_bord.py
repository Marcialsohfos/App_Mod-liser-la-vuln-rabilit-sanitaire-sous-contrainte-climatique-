"""
Tableau de bord interactif pour la visualisation des données
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Tableau de Bord", page_icon="📊")

st.title("📊 Tableau de Bord Interactif")

# Charger les données si nécessaire
if 'data' not in st.session_state or st.session_state.data is None:
    from app import load_data
    st.session_state.data = load_data()

if st.session_state.data is None:
    st.error("❌ Veuillez d'abord charger les données depuis la page d'accueil")
    st.stop()

df = st.session_state.data

# Sidebar pour les filtres
with st.sidebar:
    st.header("Filtres")
    
    # Filtre par commune
    if 'commune' in df.columns:
        communes = ['Toutes'] + sorted(df['commune'].dropna().unique().tolist())
        commune_selected = st.selectbox("Commune", communes)
    
    # Filtre par quartier
    if 'quartier' in df.columns:
        if commune_selected != 'Toutes':
            quartiers = ['Tous'] + sorted(df[df['commune'] == commune_selected]['quartier'].dropna().unique().tolist())
        else:
            quartiers = ['Tous'] + sorted(df['quartier'].dropna().unique().tolist())
        quartier_selected = st.selectbox("Quartier", quartiers)
    
    # Filtre par niveau de vulnérabilité
    if 'niveau_vulnerabilite' in df.columns:
        niveaux = ['Tous'] + sorted(df['niveau_vulnerabilite'].dropna().unique().tolist())
        niveau_selected = st.selectbox("Niveau de vulnérabilité", niveaux)

# Appliquer les filtres
df_filtered = df.copy()
if 'commune' in df.columns and commune_selected != 'Toutes':
    df_filtered = df_filtered[df_filtered['commune'] == commune_selected]
if 'quartier' in df.columns and quartier_selected != 'Tous':
    df_filtered = df_filtered[df_filtered['quartier'] == quartier_selected]
if 'niveau_vulnerabilite' in df.columns and niveau_selected != 'Tous':
    df_filtered = df_filtered[df_filtered['niveau_vulnerabilite'] == niveau_selected]

# Métriques principales
st.subheader("📈 Métriques Globales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Poches analysées", len(df_filtered))

with col2:
    if 'icv_normalise' in df_filtered.columns:
        st.metric("Vulnérabilité moyenne", f"{df_filtered['icv_normalise'].mean():.1f}/100")

with col3:
    if 'niveau_vulnerabilite' in df_filtered.columns:
        critical = len(df_filtered[df_filtered['niveau_vulnerabilite'] == 'Critique'])
        st.metric("Poches critiques", critical)

with col4:
    if 'quartier' in df_filtered.columns:
        st.metric("Quartiers uniques", df_filtered['quartier'].nunique())

# Onglets pour différentes visualisations
tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Géographique", "Comparaisons", "Données"])

with tab1:
    # Distribution de la vulnérabilité
    if 'icv_normalise' in df_filtered.columns:
        fig = px.histogram(df_filtered, x='icv_normalise', 
                         nbins=30,
                         title="Distribution de l'Indice de Vulnérabilité (ICV)",
                         labels={'icv_normalise': 'ICV (0-100)'},
                         color_discrete_sequence=['#3498db'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    # Répartition par niveau
    if 'niveau_vulnerabilite' in df_filtered.columns:
        niveau_counts = df_filtered['niveau_vulnerabilite'].value_counts().reset_index()
        niveau_counts.columns = ['Niveau', 'Nombre de poches']
        
        colors = {'Critique': '#e74c3c', 'Élevée': '#e67e22', 
                 'Modérée': '#f39c12', 'Faible': '#27ae60'}
        
        fig = px.bar(niveau_counts, x='Niveau', y='Nombre de poches',
                    title="Répartition par niveau de vulnérabilité",
                    color='Niveau',
                    color_discrete_map=colors)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Visualisation par commune
    if 'commune' in df_filtered.columns and 'icv_normalise' in df_filtered.columns:
        commune_stats = df_filtered.groupby('commune')['icv_normalise'].agg(['mean', 'count', 'std']).reset_index()
        
        fig = px.bar(commune_stats, x='commune', y='mean',
                    title="Vulnérabilité moyenne par commune",
                    labels={'mean': 'ICV moyen', 'commune': 'Commune'},
                    color='mean',
                    color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    
    # Carte thermique des quartiers (simulée)
    if 'quartier' in df_filtered.columns and 'icv_normalise' in df_filtered.columns:
        st.subheader("Top 10 quartiers les plus vulnérables")
        top_quartiers = df_filtered.nlargest(10, 'icv_normalise')[['quartier', 'icv_normalise']]
        fig = px.bar(top_quartiers, x='icv_normalise', y='quartier',
                    orientation='h',
                    title="Top 10 des quartiers les plus vulnérables",
                    color='icv_normalise',
                    color_continuous_scale='RdYlGn_r')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Matrice de corrélation
    score_cols = [col for col in df_filtered.columns if col.startswith('score_')]
    if len(score_cols) >= 3:
        st.subheader("Corrélations entre les dimensions")
        corr_matrix = df_filtered[score_cols].corr()
        
        fig = px.imshow(corr_matrix,
                       title="Matrice de corrélation entre les dimensions",
                       color_continuous_scale='RdBu',
                       zmin=-1, zmax=1,
                       labels=dict(color="Corrélation"))
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plots comparatifs
    if 'commune' in df_filtered.columns and 'icv_normalise' in df_filtered.columns:
        fig = px.box(df_filtered, x='commune', y='icv_normalise',
                    title="Distribution de la vulnérabilité par commune",
                    points="all")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Tableau des données
    st.subheader("Données détaillées")
    
    # Sélection des colonnes à afficher
    default_cols = ['id_poche', 'quartier', 'commune', 'icv_normalise', 'niveau_vulnerabilite']
    available_cols = [col for col in default_cols if col in df_filtered.columns]
    
    # Options d'affichage
    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.slider("Nombre de lignes à afficher", 10, 100, 20)
    with col2:
        show_all_cols = st.toggle("Afficher toutes les colonnes", False)
    
    if show_all_cols:
        columns_to_show = df_filtered.columns.tolist()
    else:
        columns_to_show = available_cols
    
    # Affichage du dataframe
    st.dataframe(df_filtered[columns_to_show].head(n_rows), use_container_width=True)
    
    # Options d'export
    st.divider()
    st.subheader("Export des données")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Exporter en CSV"):
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"donnees_vulnerabilite_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Exporter en Excel"):
            excel_buffer = io.BytesIO()
            df_filtered.to_excel(excel_buffer, index=False)
            st.download_button(
                label="Télécharger Excel",
                data=excel_buffer.getvalue(),
                file_name=f"donnees_vulnerabilite_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("📋 Copier le tableau"):
            st.info("Utilisez Ctrl+C pour copier les données du tableau ci-dessus")

# Statistiques avancées
with st.expander("📊 Statistiques avancées"):
    if 'icv_normalise' in df_filtered.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Statistiques descriptives:**")
            stats = df_filtered['icv_normalise'].describe()
            st.dataframe(pd.DataFrame(stats).T)
        
        with col2:
            st.write("**Distribution par percentile:**")
            percentiles = np.percentile(df_filtered['icv_normalise'].dropna(), 
                                      [10, 25, 50, 75, 90])
            percentiles_df = pd.DataFrame({
                'Percentile': ['10%', '25%', '50%', '75%', '90%'],
                'Valeur ICV': percentiles
            })
            st.dataframe(percentiles_df)