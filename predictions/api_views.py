"""
api_views.py - VERSION FINALE POUR PRODUCTION RAILWAY
Zones diagnostiques + Prédiction NCA avec gestion des NaN + Cohorte de référence
Chemins corrigés: medicalapp/predictions/data/, predictions/nca_models_with_nan/
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from .centile_curves import (
    calculate_centile_curves_lms,
    calculate_patient_centile_lms
)

# ========== CHEMINS CORRIGÉS POUR PRODUCTION ==========

# BASE_DIR pointe vers le dossier predictions/
BASE_DIR = Path(__file__).resolve().parent

# UN SEUL fichier de données dans predictions/data/
DATA_PATH = BASE_DIR / "data" / "Example_database_withoutrois.xlsx"

# Modèle NCA dans predictions/nca_models_with_nan/
NCA_MODEL_PATH = BASE_DIR / "nca_models_with_nan" / "LGBM_with_nan.sav"

# Modèles de régression dans predictions/models/ (si utilisés)
MODELS_DIR = BASE_DIR / "models"

# Log des chemins au démarrage
print(f"\n📂 Configuration des chemins :")
print(f"   BASE_DIR: {BASE_DIR}")
print(f"   DATA_PATH: {DATA_PATH}")
print(f"   NCA_MODEL_PATH: {NCA_MODEL_PATH}")


# ========== DÉFINITION DES 34 FEATURES ==========

FEATURES_ALL_PLUS_PLUS = [
    # Obligatoires (7)
    'age', 'sex', 'education', 'language', 'fluency_score', 'moca', 'ravlt_imm',
    
    # Optionnels cognitifs (6)
    'handedness', 'nb_language', 'hearing', 'ravlt_delay', 'logic_imm', 'logic_delay',
    
    # Facteurs de risque (20)
    'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income', 'retired',
    'stroke', 'tbi', 'hta', 'diab_type2', 'obesity', 'depression', 'anxiety',
    'smoking', 'alcohol', 'poly_pharm5', 'physical_activity', 'social_life',
    'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
]


# ========== CHARGEMENT DU MODÈLE NCA (SINGLETON) ==========

_NCA_MODEL = None

def load_nca_model():
    """Charge le modèle NCA (une seule fois)"""
    global _NCA_MODEL
    
    if _NCA_MODEL is None:
        if not NCA_MODEL_PATH.exists():
            error_msg = f"❌ Modèle NCA non trouvé : {NCA_MODEL_PATH}"
            print(error_msg)
            print(f"\n📂 Contenu de {BASE_DIR}:")
            if BASE_DIR.exists():
                for item in BASE_DIR.iterdir():
                    print(f"   {'📁' if item.is_dir() else '📄'} {item.name}")
            raise FileNotFoundError(error_msg)
        
        print(f"\n📦 Chargement modèle NCA : {NCA_MODEL_PATH}")
        print(f"   Taille : {NCA_MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        
        try:
            _NCA_MODEL = joblib.load(NCA_MODEL_PATH)
            print(f"✅ Modèle chargé : {type(_NCA_MODEL).__name__}")
        except Exception as e:
            print(f"❌ Erreur : {e}")
            raise
    
    return _NCA_MODEL


def prepare_nca_features(patient_data):
    """Prépare 34 features avec np.nan pour manquants"""
    features = []
    for feature_name in FEATURES_ALL_PLUS_PLUS:
        value = patient_data.get(feature_name)
        if value is None or value == '' or value == 'null':
            features.append(np.nan)
        elif isinstance(value, float) and np.isnan(value):
            features.append(np.nan)
        else:
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(np.nan)
    return features


def predict_nca(patient_data):
    """Prédiction NCA avec LightGBM"""
    try:
        model = load_nca_model()
    except FileNotFoundError:
        age = float(patient_data.get('age', 65))
        return {
            'nca_predicted': age + 5.0,
            'delta_nca': 5.0,
            'age_chronologique': age,
            'n_features_used': 0,
            'n_features_total': 34,
            'completeness': 0.0,
            'reliability': 'Non disponible',
            'reliability_stars': '',
            'features_detail': {'obligatoires': False, 'cognitifs': 0, 'risques': 0},
            'interpretation': 'Estimation par défaut'
        }
    
    required_fields = ['age', 'sex', 'education', 'language', 'fluency_score', 'moca', 'ravlt_imm']
    missing_required = [f for f in required_fields if f not in patient_data or patient_data[f] is None]
    if missing_required:
        raise ValueError(f"Champs obligatoires manquants : {missing_required}")
    
    features = prepare_nca_features(patient_data)
    n_features_used = sum(1 for f in features if not (isinstance(f, float) and np.isnan(f)))
    completeness = (n_features_used / len(FEATURES_ALL_PLUS_PLUS)) * 100
    
    if completeness >= 90:
        reliability, reliability_stars = 'Élevée', '⭐⭐⭐⭐⭐'
    elif completeness >= 70:
        reliability, reliability_stars = 'Bonne', '⭐⭐⭐⭐'
    elif completeness >= 50:
        reliability, reliability_stars = 'Acceptable', '⭐⭐⭐'
    else:
        reliability, reliability_stars = 'Limitée', '⭐⭐'
    
    nca_predicted = model.predict([features])[0]
    age_chronologique = float(patient_data['age'])
    delta_nca = nca_predicted - age_chronologique
    
    if abs(delta_nca) < 2:
        interpretation = "Vieillissement cognitif normal"
    elif delta_nca > 0:
        interpretation = "Vieillissement cognitif accéléré"
    else:
        interpretation = "Vieillissement cognitif ralenti"
    
    print(f"\n🔮 NCA : {age_chronologique:.1f} → {nca_predicted:.1f} (Δ{delta_nca:+.1f}) | {completeness:.0f}% | {reliability}")
    
    return {
        'nca_predicted': float(nca_predicted),
        'delta_nca': float(delta_nca),
        'age_chronologique': float(age_chronologique),
        'interpretation': interpretation,
        'n_features_used': int(n_features_used),
        'n_features_total': len(FEATURES_ALL_PLUS_PLUS),
        'completeness': float(completeness),
        'reliability': reliability,
        'reliability_stars': reliability_stars,
        'features_detail': {
            'obligatoires': n_features_used >= 7,
            'cognitifs': sum(1 for f in features[7:13] if not (isinstance(f, float) and np.isnan(f))),
            'risques': sum(1 for f in features[13:] if not (isinstance(f, float) and np.isnan(f))),
        }
    }


def clean_numeric_value(value):
    """Nettoie les valeurs numériques pour JSON"""
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
    """Nettoie récursivement un dictionnaire pour JSON"""
    if isinstance(data, dict):
        return {k: clean_dict_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_dict_for_json(item) for item in data]
    elif isinstance(data, (np.integer, np.floating, int, float)):
        return clean_numeric_value(data)
    else:
        return data


def calculate_diagnostic_zones_CORRECT(df, sex_value, age_col='age', value_col='delta_NCA', diagnosis_col='diagnosis'):
    """Calcule les zones diagnostiques globales (pas par âge)"""
    if diagnosis_col not in df.columns:
        for alt_name in ['dementia_dx_code', 'diagnostic', 'dx', 'label']:
            if alt_name in df.columns:
                diagnosis_col = alt_name
                break
    
    data = df[df['sex'] == sex_value].copy()
    con_data = data[data[diagnosis_col] == 'CON'].copy()
    mci_data = data[data[diagnosis_col] == 'MCI'].copy()
    ad_data = data[data[diagnosis_col] == 'AD'].copy()
    
    global_stats = {
        'CON': {'min': float(con_data[value_col].min()), 'max': float(con_data[value_col].max()), 
                'mean': float(con_data[value_col].mean()), 'n': len(con_data)},
        'MCI': {'min': float(mci_data[value_col].min()), 'max': float(mci_data[value_col].max()), 
                'mean': float(mci_data[value_col].mean()), 'n': len(mci_data)},
        'AD': {'min': float(ad_data[value_col].min()), 'max': float(ad_data[value_col].max()), 
               'mean': float(ad_data[value_col].mean()), 'n': len(ad_data)}
    }
    
    limit_normal_mci = (global_stats['CON']['max'] + global_stats['MCI']['min']) / 2
    limit_mci_ad = (global_stats['MCI']['max'] + global_stats['AD']['min']) / 2
    
    zones = []
    for age in range(50, 91):
        zones.append({
            'age': int(age),
            'green_bottom': clean_numeric_value(global_stats['CON']['min'] - 2),
            'green_blue': clean_numeric_value(limit_normal_mci),
            'blue_red': clean_numeric_value(limit_mci_ad),
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


@api_view(['POST'])
def predict_api(request):
    """API principale de prédiction"""
    try:
        input_data = request.data
        print(f"📥 Requête : {list(input_data.keys())[:5]}...")
        
        # Vérifier que le fichier de données existe
        if not DATA_PATH.exists():
            return Response({'error': f'Fichier non trouvé: {DATA_PATH}'}, status=status.HTTP_404_NOT_FOUND)
        
        # Charger les données
        df_raw = pd.read_excel(DATA_PATH)
        print(f"📂 Données : {len(df_raw)} lignes")
        print(f"📊 Colonnes : {list(df_raw.columns)[:10]}")
        
        # ========== NORMALISER LES NOMS DE COLONNES ==========
        # Détecter et renommer les colonnes automatiquement
        df = df_raw.copy()
        
        # Age
        for col in ['age', 'Age', 'AGE']:
            if col in df.columns:
                df.rename(columns={col: 'age'}, inplace=True)
                break
        
        # Sex
        for col in ['sex', 'Sex', 'SEX']:
            if col in df.columns:
                df.rename(columns={col: 'sex'}, inplace=True)
                break
        
        # Diagnosis
        for col in ['diagnosis', 'Diagnosis', 'dementia_dx_code', 'Dementia_Dx_Code', 'DIAGNOSIS']:
            if col in df.columns:
                df.rename(columns={col: 'diagnosis'}, inplace=True)
                break
        
        # Neurocog age
        for col in ['neurocog_age_flu_weight', 'Neurocog_Age_Flu_Weight', 'NCA', 'neurocog_age']:
            if col in df.columns:
                df.rename(columns={col: 'neurocog_age_flu_weight'}, inplace=True)
                break
        
        # Education
        for col in ['education', 'Education', 'EDUCATION', 'educ', 'scolarite']:
            if col in df.columns:
                df.rename(columns={col: 'education'}, inplace=True)
                break
        
        # Calculer delta_NCA si nécessaire
        if 'delta_NCA' not in df.columns and 'neurocog_age_flu_weight' in df.columns:
            df['delta_NCA'] = df['neurocog_age_flu_weight'] - df['age']
            print(f"✅ delta_NCA calculé : {df['delta_NCA'].mean():.2f} (moyenne)")
        
        print(f"✅ Colonnes normalisées : {list(df.columns)[:10]}")
        
        # ========== FILTRER LES NaN ==========
        # Colonnes essentielles qui ne doivent pas avoir de NaN
        required_cols = ['age', 'sex', 'diagnosis']
        available_required = [col for col in required_cols if col in df.columns]
        
        df_before = len(df)
        df = df.dropna(subset=available_required).copy()
        df_after = len(df)
        
        print(f"📊 Dataset avant filtrage : {df_before} patients")
        print(f"📊 Dataset après filtrage : {df_after} patients")
        if df_before != df_after:
            print(f"⚠️ {df_before - df_after} lignes supprimées (NaN dans {', '.join(available_required)})")
        
        # Prédiction NCA
        try:
            nca_result = predict_nca(input_data)
            predicted_delta_nca = nca_result['delta_nca']
            patient_age = nca_result['age_chronologique']
        except ValueError as e:
            return Response({'error': f'Validation : {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f'Erreur NCA : {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        patient_sex = int(input_data.get('sex', 1))
        
        # Calculer les zones diagnostiques
        zones_male = calculate_diagnostic_zones_CORRECT(df, sex_value=1)
        zones_female = calculate_diagnostic_zones_CORRECT(df, sex_value=0)
        
        # Calculer les courbes de centiles (CON uniquement)
        df_con = df[df['diagnosis'] == 'CON'].copy()
        
        print(f"📊 CON data : {len(df_con)} patients")
        print(f"📊 delta_NCA range : [{df_con['delta_NCA'].min():.1f}, {df_con['delta_NCA'].max():.1f}]")
        
        try:
            lms_male = calculate_centile_curves_lms(df=df_con, sex_value=1, age_col='age', value_col='delta_NCA',
                                                     centiles=[3, 10, 25, 50, 75, 90, 97], window=3)
            lms_female = calculate_centile_curves_lms(df=df_con, sex_value=0, age_col='age', value_col='delta_NCA',
                                                       centiles=[3, 10, 25, 50, 75, 90, 97], window=3)
            print(f"✅ Courbes LMS calculées")
        except Exception as e:
            print(f"⚠️ Erreur calcul LMS : {e}")
            # Fallback : courbes simples basées sur les statistiques
            lms_male = {'curves': [], 'lms_parameters': {}}
            lms_female = {'curves': [], 'lms_parameters': {}}
        
        # Calculer le centile du patient
        if lms_male.get('lms_parameters') and lms_female.get('lms_parameters'):
            patient_lms = lms_male['lms_parameters'] if patient_sex == 1 else lms_female['lms_parameters']
            patient_centile = calculate_patient_centile_lms(age=patient_age, value=predicted_delta_nca,
                                                             lms_parameters=patient_lms)
        else:
            patient_centile = None
        
        # ✅ Vérifier que le calcul a réussi
        if patient_centile is None:
            print(f"⚠️ Calcul centile échoué pour âge={patient_age}, delta={predicted_delta_nca}")
            # Fallback : créer un centile par défaut
            patient_centile = {
                'centile': 50,
                'z_score': 0,
                'interpretation': 'Calcul non disponible',
                'lms_parameters': {'L': 0, 'M': predicted_delta_nca, 'S': 1}
            }
        
        # Déterminer la zone du patient
        limits = zones_male['limits'] if patient_sex == 1 else zones_female['limits']
        if predicted_delta_nca < limits['normal_mci']:
            patient_zone = 'Normale'
        elif predicted_delta_nca < limits['mci_ad']:
            patient_zone = 'MCI'
        else:
            patient_zone = 'Pathologique'
        
        # ========== ✅ COHORTE DE RÉFÉRENCE (TOUS LES PATIENTS VALIDES) ==========
        reference_cohort = []
        
        # Utiliser toutes les données valides (déjà filtrées)
        df_sample = df
        
        print(f"📊 Cohorte de référence : {len(df_sample)} patients")
        
        def convert_education_years_to_group(years):
            """Convertit les années d'éducation en groupe (0-3)"""
            if pd.isna(years):
                return 0
            years = float(years)
            if years <= 12:
                return 0  # Secondaire ou moins
            elif years <= 15:
                return 1  # Collégial/Technique
            elif years <= 20:
                return 2  # Universitaire 1er cycle
            else:
                return 3  # Universitaire cycles supérieurs
        
        for _, row in df_sample.iterrows():
            # NCA peut être déjà calculé ou doit l'être
            if 'neurocog_age_flu_weight' in row and not pd.isna(row['neurocog_age_flu_weight']):
                nca_value = float(row['neurocog_age_flu_weight'])
            elif 'delta_NCA' in row and not pd.isna(row['delta_NCA']):
                nca_value = float(row['age']) + float(row['delta_NCA'])
            else:
                nca_value = float(row['age'])  # Fallback
            
            education_years = row.get('education', 0)
            education_group = convert_education_years_to_group(education_years)
            
            # Gérer les NaN potentiels (même si on a filtré, sécurité supplémentaire)
            try:
                sex_value = int(row['sex']) if not pd.isna(row['sex']) else 0
                diagnosis_value = str(row['diagnosis']) if not pd.isna(row['diagnosis']) else 'CON'
            except (ValueError, TypeError):
                sex_value = 0
                diagnosis_value = 'CON'
            
            reference_cohort.append({
                'age': clean_numeric_value(row['age']),
                'neurocog_age_flu_weight': clean_numeric_value(nca_value),
                'sex': sex_value,
                'dementia_dx_code': diagnosis_value,
                'education_group': int(education_group)
            })
        
        print(f"✅ Cohorte : {len(reference_cohort)} participants")
        
        # Distribution des groupes d'éducation
        groups_dist = {}
        for p in reference_cohort:
            g = p['education_group']
            groups_dist[g] = groups_dist.get(g, 0) + 1
        print(f"📊 Distribution éducation : {groups_dist}")
        
        # ========== CONSTRUCTION DU RÉSULTAT ==========
        result = {
            'nca_prediction': {
                'nca_predicted': clean_numeric_value(nca_result['nca_predicted']),
                'delta_nca': clean_numeric_value(predicted_delta_nca),
                'age_chronologique': clean_numeric_value(patient_age),
                'interpretation': nca_result['interpretation'],
                'features_used': nca_result['n_features_used'],
                'features_total': nca_result['n_features_total'],
                'completeness': clean_numeric_value(nca_result['completeness']),
                'reliability': nca_result['reliability'],
                'reliability_stars': nca_result['reliability_stars'],
                'features_detail': nca_result['features_detail']
            },
            'patient_age': clean_numeric_value(patient_age),
            'patient_sex': int(patient_sex),
            'patient_zone': patient_zone,
            'zone_boundaries': {
                'male': zones_male['zones'],
                'female': zones_female['zones'],
                'patient_sex': int(patient_sex),
                'limits': zones_male['limits'] if patient_sex == 1 else zones_female['limits'],
                'stats': zones_male['global_stats'] if patient_sex == 1 else zones_female['global_stats']
            },
            'centile_curves': {
                'male': lms_male.get('curves', []),
                'female': lms_female.get('curves', []),
                'patient_point': {
                    'age': clean_numeric_value(patient_age),
                    'delta_nca': clean_numeric_value(predicted_delta_nca),
                    'centile': clean_numeric_value(patient_centile['centile']),
                    'z_score': clean_numeric_value(patient_centile['z_score']),
                    'interpretation': patient_centile['interpretation'],
                    'zone': patient_zone,
                    'lms_parameters': {
                        'L': clean_numeric_value(patient_centile['lms_parameters']['L']),
                        'M': clean_numeric_value(patient_centile['lms_parameters']['M']),
                        'S': clean_numeric_value(patient_centile['lms_parameters']['S']),
                    }
                },
                'patient_sex': int(patient_sex),
                'age_range': [50, 90],
                'method': 'LMS (Box-Cox) - CON uniquement',
                'axis_domain': [-15, 25]
            },
            'metadata': {
                'n_samples_total': len(df),
                'n_con': len(df_con),
                'nca_model': 'LGBM_with_nan',
                'data_file': 'Example_database_withoutrois.xlsx'
            },
            'reference_cohort': reference_cohort,  # ✅ CORRECT
            'success': True
        }
        
        result = clean_dict_for_json(result)
        print("✅ Réponse prête")
        return Response(result, status=status.HTTP_200_OK)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ ERREUR: {str(e)}")
        print(error_trace)
        return Response({'error': str(e), 'success': False}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def health_check(request):
    """Vérification de santé du service"""
    try:
        data_exists = DATA_PATH.exists()
        model_exists = NCA_MODEL_PATH.exists()
        model_loaded = False
        model_error = None
        try:
            model = load_nca_model()
            model_loaded = True
        except Exception as e:
            model_error = str(e)
        
        available_files = {}
        if BASE_DIR.exists():
            for subdir in ['data', 'models', 'nca_models_with_nan']:
                dir_path = BASE_DIR / subdir
                if dir_path.exists():
                    available_files[subdir] = [f.name for f in dir_path.iterdir() if f.is_file()]
        
        return Response({
            'status': 'healthy' if (data_exists and model_loaded) else 'warning',
            'data_file_exists': data_exists,
            'data_file_path': str(DATA_PATH),
            'nca_model_exists': model_exists,
            'nca_model_path': str(NCA_MODEL_PATH),
            'nca_model_loaded': model_loaded,
            'nca_model_error': model_error,
            'available_files': available_files,
            'base_dir': str(BASE_DIR),
            'success': True
        })
    except Exception as e:
        return Response({'status': 'error', 'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def model_info_api(request):
    """Informations sur le modèle"""
    try:
        model_exists = NCA_MODEL_PATH.exists()
        if model_exists:
            model = load_nca_model()
            model_info = {'loaded': True, 'type': str(type(model).__name__), 'n_features': len(FEATURES_ALL_PLUS_PLUS)}
        else:
            model_info = {'loaded': False, 'error': f'Modèle non trouvé : {NCA_MODEL_PATH}'}
        
        return Response({
            'model_type': 'Cognitive Aging - NCA avec gestion NaN',
            'nca_model': model_info,
            'features': {
                'total': len(FEATURES_ALL_PLUS_PLUS),
                'obligatoires': 7,
                'cognitifs_optionnels': 6,
                'facteurs_risque': 20,
                'list': FEATURES_ALL_PLUS_PLUS
            },
            'method': 'LightGBM avec gestion native des NaN',
            'success': True
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)