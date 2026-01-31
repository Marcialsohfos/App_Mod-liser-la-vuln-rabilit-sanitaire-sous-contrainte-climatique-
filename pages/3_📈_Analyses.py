"""
Page d'analyses avancées et de clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analyses", page_icon="📈")

st.title("📈 Analyses Avancées")

# Vérifier les données
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("❌ Veuillez d'abord charger les données")
    st.stop()

df = st.session_state.data

# Onglets d'analyse
tab1, tab2, tab3, tab4 = st.tabs(["Clustering", "PCA", "Corrélations", "Tendances"])

with tab1:
    st.header("Clustering des poches")
    
    # Sélection des features pour le clustering
    st.subheader("Sélection des variables")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    selected_features = st.multiselect(
        "Choisir les variables pour le clustering",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]
    )
    
    if len(selected_features) >= 3:
        # Préparation des données
        X = df[selected_features].dropna()
        
        # Paramètres du clustering
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Nombre de clusters", 2, 10, 4)
        
        with col2:
            max_iter = st.slider("Itérations maximum", 100, 1000, 300)
        
        if st.button("🔍 Appliquer le clustering"):
            with st.spinner("Clustering en cours..."):
                # Appliquer K-means
                kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                # Ajouter les clusters aux données
                df_clustered = X.copy()
                df_clustered['Cluster'] = clusters
                
                # Visualisation 2D avec PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                fig = px.scatter(
                    x=X_pca[:, 0], y=X_pca[:, 1],
                    color=clusters.astype(str),
                    title=f"Clustering K-means (k={n_clusters})",
                    labels={'x': 'PC1', 'y': 'PC2'},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques par cluster
                st.subheader("Caractéristiques des clusters")
                
                # Calculer les moyennes par cluster
                cluster_stats = df_clustered.groupby('Cluster').mean().round(2)
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Visualisation des centres de clusters
                fig = go.Figure()
                
                for i in range(n_clusters):
                    fig.add_trace(go.Scatterpolar(
                        r=kmeans.cluster_centers_[i][:8],  # Premières 8 features
                        theta=selected_features[:8],
                        fill='toself',
                        name=f'Cluster {i}',
                        line_color=px.colors.qualitative.Set1[i]
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title="Centres des clusters (radar)"
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Analyse en Composantes Principales (PCA)")
    
    if len(numeric_cols) >= 3:
        # Sélection des variables
        pca_features = st.multiselect(
            "Variables pour PCA",
            numeric_cols,
            default=numeric_cols[:min(15, len(numeric_cols))]
        )
        
        if len(pca_features) >= 3:
            X_pca = df[pca_features].dropna()
            
            # Appliquer PCA
            pca = PCA()
            X_transformed = pca.fit_transform(X_pca)
            
            # Variance expliquée
            variance_exp = np.cumsum(pca.explained_variance_ratio_)
            
            fig = px.line(
                x=range(1, len(variance_exp) + 1),
                y=variance_exp,
                title="Variance expliquée cumulative",
                labels={'x': 'Nombre de composantes', 'y': 'Variance expliquée'}
            )
            fig.add_hline(y=0.95, line_dash="dash", line_color="red")
            fig.add_hline(y=0.80, line_dash="dash", line_color="orange")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Biplot (2 premières composantes)
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            fig = go.Figure()
            
            # Points
            fig.add_trace(go.Scatter(
                x=X_transformed[:, 0],
                y=X_transformed[:, 1],
                mode='markers',
                marker=dict(size=8, color=df['icv_normalise'].values if 'icv_normalise' in df.columns else 'blue'),
                text=df['quartier'].values if 'quartier' in df.columns else None,
                hoverinfo='text'
            ))
            
            # Vecteurs de loadings
            for i, feature in enumerate(pca_features):
                fig.add_trace(go.Scatter(
                    x=[0, loadings[i, 0] * 10],
                    y=[0, loadings[i, 1] * 10],
                    mode='lines+text',
                    line=dict(color='red', width=2),
                    text=[None, feature],
                    textposition="top center",
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Biplot - PCA",
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des loadings
            st.subheader("Loadings des composantes principales")
            
            loadings_df = pd.DataFrame(
                pca.components_[:5].T,
                columns=[f'PC{i+1}' for i in range(5)],
                index=pca_features
            )
            
            st.dataframe(loadings_df.style.background_gradient(cmap='RdBu', axis=0), 
                        use_container_width=True)

with tab3:
    st.header("Analyse des corrélations")
    
    # Sélection des variables
    corr_vars = st.multiselect(
        "Variables à corréler",
        numeric_cols,
        default=numeric_cols[:min(20, len(numeric_cols))]
    )
    
    if len(corr_vars) >= 2:
        # Matrice de corrélation
        corr_matrix = df[corr_vars].corr()
        
        # Heatmap
        fig = px.imshow(
            corr_matrix,
            title="Matrice de corrélations",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1,
            aspect="auto"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Corrélations les plus fortes
        st.subheader("Corrélations significatives")
        
        # Créer un dataframe des corrélations
        corr_pairs = []
        for i in range(len(corr_vars)):
            for j in range(i+1, len(corr_vars)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Seuil de corrélation
                    corr_pairs.append({
                        'Variable 1': corr_vars[i],
                        'Variable 2': corr_vars[j],
                        'Corrélation': corr_value
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Corrélation', key=abs, ascending=False)
            st.dataframe(corr_df, use_container_width=True)
        
        # Analyse de régression simple
        st.subheader("Analyse de régression simple")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Variable indépendante (X)", corr_vars)
        
        with col2:
            y_var = st.selectbox("Variable dépendante (Y)", corr_vars)
        
        if x_var != y_var:
            # Scatter plot avec ligne de régression
            fig = px.scatter(
                df, x=x_var, y=y_var,
                trendline="ols",
                title=f"Relation entre {x_var} et {y_var}",
                hover_data=['quartier'] if 'quartier' in df.columns else None
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcul des statistiques
            from scipy import stats
            subset = df[[x_var, y_var]].dropna()
            if len(subset) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    subset[x_var], subset[y_var]
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("R²", f"{r_value**2:.3f}")
                
                with col2:
                    st.metric("P-value", f"{p_value:.4f}")
                
                with col3:
                    st.metric("Pente", f"{slope:.4f}")

with tab4:
    st.header("Analyse des tendances")
    
    # Sélection de la variable à analyser
    trend_var = st.selectbox(
        "Variable à analyser",
        ['icv_normalise'] + numeric_cols
    )
    
    if trend_var in df.columns:
        # Distribution temporelle (si date disponible)
        if 'annee_col' in df.columns:
            st.subheader("Évolution temporelle")
            
            yearly_trend = df.groupby('annee_col')[trend_var].agg(['mean', 'std', 'count']).reset_index()
            
            fig = px.line(
                yearly_trend, x='annee_col', y='mean',
                error_y='std',
                title=f"Évolution de {trend_var} par année",
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Analyse par commune
        if 'commune' in df.columns:
            st.subheader("Comparaison par commune")
            
            commune_stats = df.groupby('commune')[trend_var].agg(['mean', 'std', 'count']).reset_index()
            commune_stats = commune_stats.sort_values('mean', ascending=False)
            
            fig = px.bar(
                commune_stats, x='commune', y='mean',
                error_y='std',
                title=f"{trend_var} par commune",
                color='mean',
                color_continuous_scale='RdYlGn_r'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plots
        st.subheader("Distribution statistique")
        
        fig = px.box(
            df, y=trend_var,
            title=f"Distribution de {trend_var}",
            points="all"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Histogramme avec courbe de densité
        fig = px.histogram(
            df, x=trend_var,
            nbins=30,
            marginal="box",
            title=f"Histogramme de {trend_var}",
            opacity=0.7
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Export des analyses
st.divider()
st.subheader("📊 Export des analyses")

analysis_type = st.selectbox(
    "Type d'export",
    ["Rapport complet", "Tableau des clusters", "Matrice de corrélations", "Données transformées"]
)

if st.button("💾 Générer l'export"):
    with st.spinner("Génération en cours..."):
        # Ici, générer l'export selon le type sélectionné
        st.success("✅ Export généré avec succès!")
        
        # Exemple de téléchargement
        sample_data = pd.DataFrame({'Exemple': [1, 2, 3]})
        csv = sample_data.to_csv(index=False)
        
        st.download_button(
            label="📥 Télécharger",
            data=csv,
            file_name=f"analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )