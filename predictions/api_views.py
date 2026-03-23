"""
api_views.py - VERSION AVEC MODÈLE NCA INTÉGRÉ
Zones diagnostiques + Prédiction NCA avec gestion des NaN
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import joblib
import os

from .centile_curves import (
    calculate_centile_curves_lms,
    calculate_patient_centile_lms
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "predictions" / "Data_NCA_exposome.xlsx"
NCA_MODEL_PATH = BASE_DIR / "predictions" / "nca_models_with_nan" / "LGBM_with_nan.sav"


# Définition des 34 features (dans le bon ordre !)
FEATURES_ALL_PLUS_PLUS = [
    # Obligatoires (7)
    'age',
    'sex',
    'education',
    'language',
    'fluency_score',
    'moca',
    'ravlt_imm',
    
    # Optionnels cognitifs (6)
    'handedness',
    'nb_language',
    'hearing',
    'ravlt_delay',
    'logic_imm',
    'logic_delay',
    
    # Facteurs de risque (21)
    'hist_demence_fam',
    'hist_demence_parent',
    'living_alone',
    'income',
    'retired',
    'stroke',
    'tbi',
    'hta',
    'diab_type2',
    'obesity',
    'depression',
    'anxiety',
    'smoking',
    'alcohol',
    'poly_pharm5',
    'physical_activity',
    'social_life',
    'cognitive_activities',
    'nutrition_score',
    'sleep_deprivation',
]

# Charger le modèle au démarrage (singleton)
_NCA_MODEL = None

def load_nca_model():
    """Charge le modèle NCA (une seule fois)"""
    global _NCA_MODEL
    
    if _NCA_MODEL is None:
        if not os.path.exists(NCA_MODEL_PATH):
            raise FileNotFoundError(f"Modèle NCA non trouvé : {NCA_MODEL_PATH}")
        
        print(f"📦 Chargement du modèle NCA : {NCA_MODEL_PATH}")
        _NCA_MODEL = joblib.load(NCA_MODEL_PATH)
        print(f"✅ Modèle NCA chargé avec succès")
    
    return _NCA_MODEL


def prepare_nca_features(patient_data):
    """
    Prépare les 34 features pour la prédiction NCA
    
    IMPORTANT : Garde np.nan pour les valeurs manquantes
    Le modèle LightGBM gère automatiquement les NaN
    
    Args:
        patient_data: dict avec les données du patient
        
    Returns:
        list de 34 valeurs (avec np.nan pour les manquants)
    """
    features = []
    
    for feature_name in FEATURES_ALL_PLUS_PLUS:
        value = patient_data.get(feature_name)
        
        # Si la valeur n'existe pas ou est vide → np.nan
        if value is None or value == '' or value == 'null':
            features.append(np.nan)
        
        # Si c'est déjà un NaN
        elif isinstance(value, float) and np.isnan(value):
            features.append(np.nan)
        
        # Sinon, convertir en float
        else:
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                # Si conversion impossible → np.nan
                features.append(np.nan)
    
    return features


def predict_nca(patient_data):
    """
    Prédiction du NCA (Neurocognitive Age)
    
    Args:
        patient_data: dict avec les données du patient
        
    Returns:
        dict avec :
            - nca_predicted: Âge neurocognitif prédit
            - delta_nca: Différence (NCA - âge chronologique)
            - n_features_used: Nombre de features utilisées
            - completeness: Pourcentage de complétude
            - reliability: Niveau de fiabilité
    """
    # Charger le modèle
    model = load_nca_model()
    
    # Vérifier les champs obligatoires
    required_fields = ['age', 'sex', 'education', 'language', 'fluency_score', 'moca']
    missing_required = [f for f in required_fields if f not in patient_data or patient_data[f] is None]
    
    if missing_required:
        raise ValueError(f"Champs obligatoires manquants : {missing_required}")
    
    # Préparer les features (avec np.nan pour manquants)
    features = prepare_nca_features(patient_data)
    
    # Compter les features utilisées (non-NaN)
    n_features_used = sum(1 for f in features if not (isinstance(f, float) and np.isnan(f)))
    completeness = (n_features_used / len(FEATURES_ALL_PLUS_PLUS)) * 100
    
    # Déterminer la fiabilité
    if completeness >= 90:
        reliability = 'Élevée'
        reliability_stars = '⭐⭐⭐⭐⭐'
    elif completeness >= 70:
        reliability = 'Bonne'
        reliability_stars = '⭐⭐⭐⭐'
    elif completeness >= 50:
        reliability = 'Acceptable'
        reliability_stars = '⭐⭐⭐'
    else:
        reliability = 'Limitée'
        reliability_stars = '⭐⭐'
    
    # Prédire (le modèle gère automatiquement les NaN !)
    nca_predicted = model.predict([features])[0]
    
    # Calculer delta
    age_chronologique = float(patient_data['age'])
    delta_nca = nca_predicted - age_chronologique
    
    # Log
    print(f"\n🔮 Prédiction NCA :")
    print(f"   Âge chronologique : {age_chronologique:.1f} ans")
    print(f"   NCA prédit : {nca_predicted:.1f} ans")
    print(f"   Delta NCA : {delta_nca:+.1f} ans")
    print(f"   Features utilisées : {n_features_used}/{len(FEATURES_ALL_PLUS_PLUS)} ({completeness:.0f}%)")
    print(f"   Fiabilité : {reliability} {reliability_stars}")
    
    return {
        'nca_predicted': float(nca_predicted),
        'delta_nca': float(delta_nca),
        'age_chronologique': float(age_chronologique),
        'n_features_used': int(n_features_used),
        'n_features_total': len(FEATURES_ALL_PLUS_PLUS),
        'completeness': float(completeness),
        'reliability': reliability,
        'reliability_stars': reliability_stars,
        'features_detail': {
            'obligatoires': n_features_used >= 7,  # Au minimum les 7 obligatoires
            'cognitifs': sum(1 for i, f in enumerate(features[7:13]) if not (isinstance(f, float) and np.isnan(f))),
            'risques': sum(1 for i, f in enumerate(features[13:]) if not (isinstance(f, float) and np.isnan(f))),
        }
    }


# ========== UTILITAIRES DE NETTOYAGE JSON ==========

def clean_numeric_value(value):
    """Nettoie une valeur numérique pour JSON"""
    if value is None:
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return value
    
    if isinstance(value, (np.integer, np.floating)):
        value = float(value)
    
    if np.isnan(value) or np.isinf(value):
        return None
    
    return float(value)


def clean_dict_for_json(data):
    """Nettoie récursivement pour JSON"""
    if isinstance(data, dict):
        return {k: clean_dict_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_dict_for_json(item) for item in data]
    elif isinstance(data, (np.integer, np.floating, int, float)):
        return clean_numeric_value(data)
    else:
        return data


# ========== CALCUL ZONES DIAGNOSTIQUES CORRIGÉ ==========

def calculate_diagnostic_zones_CORRECT(df, sex_value, age_col='age', value_col='delta_NCA', diagnosis_col='diagnosis'):
    """
    Calcule les zones diagnostiques CORRECTEMENT
    Chaque zone est basée UNIQUEMENT sur son diagnostic
    
    Args:
        df: DataFrame avec colonnes age, sex, delta_NCA, diagnosis
        sex_value: 0 (femme) ou 1 (homme)
        
    Returns:
        dict avec zones par âge + statistiques globales
    """
    # Vérifier que la colonne diagnosis existe
    if diagnosis_col not in df.columns:
        print(f"⚠️ Colonne '{diagnosis_col}' non trouvée. Colonnes disponibles : {df.columns.tolist()}")
        # Essayer des noms alternatifs
        for alt_name in ['dementia_dx_code', 'diagnostic', 'dx', 'label']:
            if alt_name in df.columns:
                diagnosis_col = alt_name
                print(f"✅ Utilisation de la colonne '{alt_name}' à la place")
                break
    
    # Filtrer par sexe
    data = df[df['sex'] == sex_value].copy()
    
    # Séparer par diagnostic
    con_data = data[data[diagnosis_col] == 'CON'].copy()
    mci_data = data[data[diagnosis_col] == 'MCI'].copy()
    ad_data = data[data[diagnosis_col] == 'AD'].copy()
    
    print(f"\n📊 Zones diagnostiques pour sexe={sex_value} ({'Homme' if sex_value == 1 else 'Femme'})")
    print(f"CON: {len(con_data)} sujets, delta_NCA: [{con_data[value_col].min():.2f}, {con_data[value_col].max():.2f}]")
    print(f"MCI: {len(mci_data)} sujets, delta_NCA: [{mci_data[value_col].min():.2f}, {mci_data[value_col].max():.2f}]")
    print(f"AD:  {len(ad_data)} sujets, delta_NCA: [{ad_data[value_col].min():.2f}, {ad_data[value_col].max():.2f}]")
    
    # Statistiques globales par diagnostic
    global_stats = {
        'CON': {
            'min': float(con_data[value_col].min()),
            'max': float(con_data[value_col].max()),
            'mean': float(con_data[value_col].mean()),
            'n': len(con_data)
        },
        'MCI': {
            'min': float(mci_data[value_col].min()),
            'max': float(mci_data[value_col].max()),
            'mean': float(mci_data[value_col].mean()),
            'n': len(mci_data)
        },
        'AD': {
            'min': float(ad_data[value_col].min()),
            'max': float(ad_data[value_col].max()),
            'mean': float(ad_data[value_col].mean()),
            'n': len(ad_data)
        }
    }
    
    # Calculer limites entre zones (moyenne entre max du groupe inférieur et min du groupe supérieur)
    limit_normal_mci = (global_stats['CON']['max'] + global_stats['MCI']['min']) / 2
    limit_mci_ad = (global_stats['MCI']['max'] + global_stats['AD']['min']) / 2
    
    print(f"\n🔍 Limites calculées:")
    print(f"Limite Normale/MCI: {limit_normal_mci:.2f} (entre max CON={global_stats['CON']['max']:.2f} et min MCI={global_stats['MCI']['min']:.2f})")
    print(f"Limite MCI/AD: {limit_mci_ad:.2f} (entre max MCI={global_stats['MCI']['max']:.2f} et min AD={global_stats['AD']['min']:.2f})")
    
    # Calculer zones pour chaque âge
    zones = []
    ages = range(50, 91)
    
    for age in ages:
        age_min = age - 3
        age_max = age + 3
        
        # Filtrer CON pour cet âge
        con_subset = con_data[(con_data[age_col] >= age_min) & (con_data[age_col] <= age_max)]
        
        # Si pas assez de données, utiliser limites globales
        if len(con_subset) < 10:
            green_bottom = global_stats['CON']['min'] - 2
            green_top = limit_normal_mci
        else:
            # Utiliser percentiles des CON uniquement
            green_bottom = float(np.percentile(con_subset[value_col], 5))  # 5e percentile
            green_top = limit_normal_mci  # Limite fixe
        
        zones.append({
            'age': int(age),
            'green_bottom': clean_numeric_value(green_bottom),
            'green_blue': clean_numeric_value(limit_normal_mci),  # Limite Normale/MCI
            'blue_red': clean_numeric_value(limit_mci_ad),        # Limite MCI/AD
            'red_top': clean_numeric_value(global_stats['AD']['max'] + 2)
        })
    
    return {
        'zones': zones,
        'global_stats': global_stats,
        'limits': {
            'normal_mci': clean_numeric_value(limit_normal_mci),
            'mci_ad': clean_numeric_value(limit_mci_ad)
        }
    }


# ========== API ENDPOINTS ==========

@api_view(['POST'])
def predict_api(request):
    """
    API principale avec prédiction NCA intégrée
    
    Body JSON attendu :
    {
        # Obligatoires
        "age": 68,
        "sex": 0,  // 0 = femme, 1 = homme
        "education": 12,
        "language": 1,  // 1 = français, 0 = anglais
        "fluency_score": 45,
        "moca": 24,
        
        # Optionnels cognitifs
        "ravlt_imm": 8,
        "ravlt_delay": 6,
        "logic_imm": 10,
        "logic_delay": 8,
        "handedness": 0,
        "nb_language": 2,
        "hearing": 0,
        
        # Facteurs de risque (0 ou 1)
        "hta": 1,
        "diab_type2": 1,
        "depression": 0,
        ...
    }
    """
    try:
        input_data = request.data
        print("📥 Requête reçue :", input_data.keys())
        
        # ========== 1. CHARGER LES DONNÉES DE RÉFÉRENCE ==========
        data_path = DATA_PATH
        if not os.path.exists(data_path):
            return Response(
                {'error': f'Fichier non trouvé: {data_path}'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        df_exposome = pd.read_excel(data_path)
        print(f"📂 Données de référence chargées : {len(df_exposome)} lignes")
        
        # ========== 2. PRÉDICTION DU NCA (NOUVEAU !) ==========
        print("\n🔮 Prédiction du NCA avec modèle LightGBM...")
        
        try:
            nca_result = predict_nca(input_data)
            predicted_delta_nca = nca_result['delta_nca']
            patient_age = nca_result['age_chronologique']
            
        except ValueError as e:
            return Response(
                {'error': f'Erreur de validation : {str(e)}', 'success': False},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Erreur lors de la prédiction NCA : {str(e)}', 'success': False},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # ========== 3. EXTRAIRE SEXE PATIENT ==========
        patient_sex = int(input_data.get('sex', 1))
        
        # ========== 4. CALCUL DES ZONES DIAGNOSTIQUES (CORRIGÉ) ==========
        print("\n🎯 Calcul des zones diagnostiques (CORRIGÉ)...")
        
        zones_male_result = calculate_diagnostic_zones_CORRECT(df_exposome, sex_value=1)
        zones_female_result = calculate_diagnostic_zones_CORRECT(df_exposome, sex_value=0)
        
        # ========== 5. CALCUL DES COURBES LMS (UNIQUEMENT SUR CON) ==========
        print("\n📊 Calcul des courbes LMS (UNIQUEMENT CON)...")
        
        # Filtrer UNIQUEMENT les CON pour les courbes de centiles
        df_con_only = df_exposome[df_exposome['diagnosis'] == 'CON'].copy()
        
        lms_result_male = calculate_centile_curves_lms(
            df=df_con_only,
            sex_value=1,
            age_col='age',
            value_col='delta_NCA',
            centiles=[3, 10, 25, 50, 75, 90, 97],
            window=3
        )
        
        lms_result_female = calculate_centile_curves_lms(
            df=df_con_only,
            sex_value=0,
            age_col='age',
            value_col='delta_NCA',
            centiles=[3, 10, 25, 50, 75, 90, 97],
            window=3
        )
        
        print(f"✅ Courbes LMS (CON uniquement) : {len(lms_result_male['curves'])} points (H), {len(lms_result_female['curves'])} points (F)")
        
        # ========== 6. POSITION DU PATIENT ==========
        patient_lms_params = (
            lms_result_male['lms_parameters'] if patient_sex == 1 
            else lms_result_female['lms_parameters']
        )
        
        patient_centile_info = calculate_patient_centile_lms(
            age=patient_age,
            value=predicted_delta_nca,
            lms_parameters=patient_lms_params
        )
        
        # Déterminer la zone du patient
        limits = zones_male_result['limits'] if patient_sex == 1 else zones_female_result['limits']
        
        if predicted_delta_nca < limits['normal_mci']:
            patient_zone = 'Normale'
        elif predicted_delta_nca < limits['mci_ad']:
            patient_zone = 'MCI'
        else:
            patient_zone = 'Pathologique'
        
        print(f"📈 Patient : Centile {patient_centile_info['centile']:.1f}e, Zone {patient_zone}")
        
        # ========== 7. CONSTRUIRE LA RÉPONSE ==========
        result = {
            # ========== PRÉDICTION NCA (NOUVEAU !) ==========
            'nca_prediction': {
                'nca_predicted': clean_numeric_value(nca_result['nca_predicted']),
                'delta_nca': clean_numeric_value(predicted_delta_nca),
                'age_chronologique': clean_numeric_value(patient_age),
                'interpretation': (
                    'Vieillissement cognitif accéléré' if predicted_delta_nca > 0 
                    else 'Vieillissement cognitif ralenti'
                ),
                'features_used': nca_result['n_features_used'],
                'features_total': nca_result['n_features_total'],
                'completeness': clean_numeric_value(nca_result['completeness']),
                'reliability': nca_result['reliability'],
                'reliability_stars': nca_result['reliability_stars'],
                'features_detail': nca_result['features_detail']
            },
            
            # Patient info
            'patient_age': clean_numeric_value(patient_age),
            'patient_sex': int(patient_sex),
            'patient_zone': patient_zone,
            
            # Zones diagnostiques (CORRIGÉES)
            'zone_boundaries': {
                'male': zones_male_result['zones'],
                'female': zones_female_result['zones'],
                'patient_sex': int(patient_sex),
                'limits': zones_male_result['limits'] if patient_sex == 1 else zones_female_result['limits'],
                'stats': zones_male_result['global_stats'] if patient_sex == 1 else zones_female_result['global_stats']
            },
            
            # Courbes LMS (UNIQUEMENT CON)
            'centile_curves': {
                'male': lms_result_male['curves'],
                'female': lms_result_female['curves'],
                'patient_point': {
                    'age': clean_numeric_value(patient_age),
                    'delta_nca': clean_numeric_value(predicted_delta_nca),
                    'centile': clean_numeric_value(patient_centile_info['centile']),
                    'z_score': clean_numeric_value(patient_centile_info['z_score']),
                    'interpretation': patient_centile_info['interpretation'],
                    'zone': patient_zone,
                    'lms_parameters': {
                        'L': clean_numeric_value(patient_centile_info['lms_parameters']['L']),
                        'M': clean_numeric_value(patient_centile_info['lms_parameters']['M']),
                        'S': clean_numeric_value(patient_centile_info['lms_parameters']['S']),
                    }
                },
                'patient_sex': int(patient_sex),
                'age_range': [50, 90],
                'method': 'LMS (Box-Cox) - CON uniquement',
                'axis_domain': [-15, 25]
            },
            
            'metadata': {
                'n_samples_total': len(df_exposome),
                'n_con': len(df_con_only),
                'n_samples_male': len(df_exposome[df_exposome['sex'] == 1]),
                'n_samples_female': len(df_exposome[df_exposome['sex'] == 0]),
                'nca_model': 'LGBM_with_nan',
                'nca_model_path': NCA_MODEL_PATH
            },
            
            'success': True
        }
        
        # Nettoyage final
        result = clean_dict_for_json(result)
        
        print("✅ Réponse nettoyée et prête")
        return Response(result, status=status.HTTP_200_OK)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("❌ ERREUR:", str(e))
        print(error_trace)
        return Response(
            {'error': str(e), 'traceback': error_trace, 'success': False},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def model_info_api(request):
    """Informations sur le modèle"""
    try:
        # Vérifier si le modèle existe
        model_exists = os.path.exists(NCA_MODEL_PATH)
        
        # Charger le modèle pour info
        if model_exists:
            model = load_nca_model()
            model_info = {
                'loaded': True,
                'type': str(type(model).__name__),
                'n_features': len(FEATURES_ALL_PLUS_PLUS),
            }
        else:
            model_info = {
                'loaded': False,
                'error': f'Modèle non trouvé : {NCA_MODEL_PATH}'
            }
        
        return Response({
            'model_type': 'Cognitive Aging - NCA avec gestion NaN',
            'nca_model': model_info,
            'features': {
                'total': len(FEATURES_ALL_PLUS_PLUS),
                'obligatoires': 7,
                'cognitifs_optionnels': 6,
                'facteurs_risque': 21,
                'list': FEATURES_ALL_PLUS_PLUS
            },
            'method': 'LightGBM avec gestion native des NaN',
            'zones_method': 'LMS (OMS) sur CON + Zones séparées par diagnostic',
            'centiles': [3, 10, 25, 50, 75, 90, 97],
            'zones': {
                'Normale': 'Basée sur CON uniquement',
                'MCI': 'Limite entre max(CON) et min(MCI)',
                'Pathologique': 'Limite entre max(MCI) et min(AD)'
            },
            'success': True
        })
    except Exception as e:
        return Response({
            'error': str(e),
            'success': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def health_check(request):
    """Health check"""
    try:
        data_path = DATA_PATH
        data_exists = os.path.exists(data_path)
        model_exists = os.path.exists(NCA_MODEL_PATH)
        
        # Tester le chargement du modèle
        model_loaded = False
        model_error = None
        try:
            model = load_nca_model()
            model_loaded = True
        except Exception as e:
            model_error = str(e)
        
        return Response({
            'status': 'healthy' if (data_exists and model_loaded) else 'warning',
            'data_file_exists': data_exists,
            'nca_model_exists': model_exists,
            'nca_model_loaded': model_loaded,
            'nca_model_error': model_error,
            'lms_module_loaded': True,
            'zones_method': 'CORRIGÉ - Séparées par diagnostic',
            'nca_model_path': NCA_MODEL_PATH,
            'success': True
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'error': str(e),
            'success': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)