"""
Constantes de l'application
"""

# Seuils de vulnérabilité
VULNERABILITY_THRESHOLDS = {
    'FAIBLE': 25,
    'MODEREE': 50,
    'ELEVEE': 75,
    'CRITIQUE': 100
}

VULNERABILITY_LABELS = {
    'FAIBLE': 'Faible',
    'MODEREE': 'Modérée',
    'ELEVEE': 'Élevée',
    'CRITIQUE': 'Critique'
}

# Couleurs associées aux niveaux
VULNERABILITY_COLORS = {
    'FAIBLE': '#27ae60',    # Vert
    'MODEREE': '#f39c12',   # Orange clair
    'ELEVEE': '#e67e22',    # Orange
    'CRITIQUE': '#e74c3c',  # Rouge
    'DEFAULT': '#95a5a6'    # Gris
}

# Poids des dimensions (pour le calcul de l'ICV)
DIMENSION_WEIGHTS = {
    'CLIMAT_RISQUES': 0.40,
    'INFRASTRUCTURE': 0.30,
    'ACCES_SERVICES': 0.20,
    'HABITAT': 0.10
}

# Communes de Yaoundé
COMMUNES_YAOUNDE = [
    "Yaoundé I",
    "Yaoundé II", 
    "Yaoundé III",
    "Yaoundé IV",
    "Yaoundé V",
    "Yaoundé VI",
    "Yaoundé VII"
]

# Quartiers prioritaires (exemple)
PRIORITY_QUARTERS = [
    "Briqueterie",
    "Mvog-Ada",
    "Nkolbisson",
    "Mimboman",
    "Mvog-Mbi",
    "Elig-Edzoa",
    "Nkol-Eton",
    "Ekounou",
    "Mvog-Betsi",
    "Odza"
]

# Variables essentielles pour le modèle
ESSENTIAL_FEATURES = [
    'dens_log',           # Densité logements
    'larg_voiri',         # Largeur voirie
    'mat_mur',           # Matériaux murs
    'mat_toit',          # Matériaux toit
    'eau_bois',          # Source eau
    'elec',              # Électricité
    'evac_eau',          # Évacuation eaux
    'evac_ord',          # Évacuation ordures
    'risq_nat',          # Risques naturels
    'risq_artif',        # Risques artificiels
    'dist_sant',         # Distance santé
    'nbre_sant',         # Nombre centres santé
    'dist_ecole',        # Distance écoles
    'nbre_ecole',        # Nombre écoles
    'etat_voir',         # État voirie
    'sec_occup',         # Secteur occupation
    'stat_occup'         # Statut occupation
]

# Recommandations par niveau
RECOMMENDATIONS = {
    'FAIBLE': [
        "Maintenance préventive des infrastructures",
        "Surveillance continue des risques",
        "Sensibilisation communautaire"
    ],
    'MODEREE': [
        "Amélioration des infrastructures de base",
        "Plan de gestion des risques",
        "Renforcement des capacités communautaires"
    ],
    'ELEVEE': [
        "Interventions prioritaires d'aménagement",
        "Systèmes d'alerte précoce",
        "Relocalisation partielle si nécessaire"
    ],
    'CRITIQUE': [
        "INTERVENTION URGENTE REQUISE",
        "Évacuation préventive recommandée",
        "Plan d'urgence immédiat",
        "Reconstruction des infrastructures"
    ]
}

# Scénarios climatiques
CLIMATE_SCENARIOS = {
    'INONDATION': {
        'name': 'Inondation majeure',
        'description': 'Précipitations exceptionnelles (+40%) pendant 72h',
        'impact': {
            'risque_naturel': +2.0,
            'infrastructure': -1.5,
            'acces_services': -1.0
        }
    },
    'SECHERESSE': {
        'name': 'Sécheresse prolongée',
        'description': 'Période de sécheresse de 3 mois',
        'impact': {
            'acces_eau': -2.0,
            'sante': -1.0,
            'agriculture': -1.5
        }
    },
    'TEMPETE': {
        'name': 'Tempête tropicale',
        'description': 'Vents violents et pluies intenses',
        'impact': {
            'risque_naturel': +2.5,
            'habitat': -2.0,
            'infrastructure': -1.5
        }
    }
}

# Configuration de l'application
APP_CONFIG = {
    'NAME': 'IA Vulnérabilité Sanitaire - Yaoundé',
    'VERSION': '1.0.0',
    'AUTHOR': 'Équipe de Recherche Université de Yaoundé I',
    'YEAR': '2024',
    'CONTACT': 'recherche.vulnerabilite@cm',
    'DATA_SOURCE': 'MINHDU/BUCREP 2024',
    'MAX_FILE_SIZE_MB': 100,
    'SUPPORTED_FORMATS': ['.xlsx', '.xls', '.csv']
}

# Messages d'erreur
ERROR_MESSAGES = {
    'NO_DATA': "❌ Aucune donnée disponible. Veuillez charger les données d'abord.",
    'NO_MODEL': "❌ Modèle non chargé. Veuillez entraîner ou charger un modèle.",
    'FILE_NOT_FOUND': "❌ Fichier non trouvé: {filename}",
    'INVALID_FORMAT': "❌ Format de fichier non supporté.",
    'UPLOAD_ERROR': "❌ Erreur lors du téléversement du fichier.",
    'PREDICTION_ERROR': "❌ Erreur lors de la prédiction: {error}"
}

# Messages de succès
SUCCESS_MESSAGES = {
    'DATA_LOADED': "✅ Données chargées avec succès: {rows} lignes, {cols} colonnes",
    'MODEL_LOADED': "✅ Modèle chargé avec succès",
    'MODEL_TRAINED': "✅ Modèle entraîné avec succès",
    'PREDICTION_DONE': "✅ Prédiction terminée",
    'EXPORT_DONE': "✅ Export terminé"
}