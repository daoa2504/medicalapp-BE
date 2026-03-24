"""
api_views.py - VERSION FINALE POUR PRODUCTION RAILWAY
Zones diagnostiques + Prédiction NCA avec gestion des NaN + Cohorte de référence + Risques ML
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

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Example_database_withoutrois.xlsx"
NCA_MODEL_PATH = BASE_DIR / "nca_models_with_nan" / "LGBM_with_nan.sav"
MODELS_DIR = BASE_DIR / "models"

# Modèles de risque
RISK_DEMENTIA_MODEL_PATH = BASE_DIR / "risk_dementia" / "LGBM_reg_all_plus_plus.sav"
RISK_HANDICAP_MODEL_PATH = BASE_DIR / "risk_handicap" / "LGBM_reg_all_plus_plus.sav"

print(f"\n📂 Configuration des chemins :")
print(f"   BASE_DIR: {BASE_DIR}")
print(f"   DATA_PATH: {DATA_PATH}")
print(f"   NCA_MODEL_PATH: {NCA_MODEL_PATH}")
print(f"   RISK_DEMENTIA_MODEL_PATH: {RISK_DEMENTIA_MODEL_PATH}")
print(f"   RISK_HANDICAP_MODEL_PATH: {RISK_HANDICAP_MODEL_PATH}")


# ========== DÉFINITION DES 33 FEATURES ==========

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
            raise FileNotFoundError(error_msg)
        
        print(f"\n📦 Chargement modèle NCA : {NCA_MODEL_PATH}")
        _NCA_MODEL = joblib.load(NCA_MODEL_PATH)
        print(f"✅ Modèle chargé : {type(_NCA_MODEL).__name__}")
    
    return _NCA_MODEL


def prepare_nca_features(patient_data):
    """Prépare 33 features avec np.nan pour manquants"""
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
            'n_features_total': 33,
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


# ========== MODÈLES DE RISQUE ==========

_risk_models_cache = {
    'dementia': None,
    'handicap': None
}


def load_risk_model(model_type='dementia'):
    """Charge le modèle de risque (démence ou handicap) avec cache"""
    global _risk_models_cache
    
    if _risk_models_cache[model_type] is not None:
        return _risk_models_cache[model_type]
    
    model_path = RISK_DEMENTIA_MODEL_PATH if model_type == 'dementia' else RISK_HANDICAP_MODEL_PATH
    
    if not model_path.exists():
        print(f"⚠️ Modèle de risque {model_type} non trouvé : {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        _risk_models_cache[model_type] = model
        print(f"✅ Modèle de risque {model_type} chargé : {model_path.name}")
        return model
    except Exception as e:
        print(f"❌ Erreur chargement modèle {model_type} : {e}")
        return None
def safe_float(value, default=np.nan):
    """Convertit en float; None, '', 'null', NaN invalides -> default"""
    if value is None:
        return default
    if value == '' or value == 'null':
        return default
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default

def prepare_risk_features(input_data):
    """Prépare les features pour les modèles de risque (33 features)"""
    features_dict = {}

    # Obligatoires (7)
    features_dict['age'] = safe_float(input_data.get('age'), 0)
    features_dict['sex'] = safe_float(input_data.get('sex'), 0)
    features_dict['education'] = safe_float(input_data.get('education'), 0)
    features_dict['language'] = safe_float(input_data.get('language'), 0)
    features_dict['fluency_score'] = safe_float(input_data.get('fluency_score'), 0)
    features_dict['moca'] = safe_float(input_data.get('moca'), 0)
    features_dict['ravlt_imm'] = safe_float(input_data.get('ravlt_imm'), 0)

    # Optionnels cognitifs (6)
    features_dict['handedness'] = safe_float(input_data.get('handedness'))
    features_dict['nb_language'] = safe_float(input_data.get('nb_language'))
    features_dict['hearing'] = safe_float(input_data.get('hearing'))
    features_dict['ravlt_delay'] = safe_float(input_data.get('ravlt_delay'))
    features_dict['logic_imm'] = safe_float(input_data.get('logic_imm'))
    features_dict['logic_delay'] = safe_float(input_data.get('logic_delay'))

    # Facteurs de risque (20)
    features_dict['hist_demence_fam'] = safe_float(input_data.get('hist_demence_fam'))
    features_dict['hist_demence_parent'] = safe_float(input_data.get('hist_demence_parent'))
    features_dict['living_alone'] = safe_float(input_data.get('living_alone'))
    features_dict['income'] = safe_float(input_data.get('income'))
    features_dict['retired'] = safe_float(input_data.get('retired'))
    features_dict['stroke'] = safe_float(input_data.get('stroke'))
    features_dict['tbi'] = safe_float(input_data.get('tbi'))
    features_dict['hta'] = safe_float(input_data.get('hta'))
    features_dict['diab_type2'] = safe_float(input_data.get('diab_type2'))
    features_dict['obesity'] = safe_float(input_data.get('obesity'))
    features_dict['depression'] = safe_float(input_data.get('depression'))
    features_dict['anxiety'] = safe_float(input_data.get('anxiety'))
    features_dict['smoking'] = safe_float(input_data.get('smoking'))
    features_dict['alcohol'] = safe_float(input_data.get('alcohol'))
    features_dict['poly_pharm5'] = safe_float(input_data.get('poly_pharm5'))
    features_dict['physical_activity'] = safe_float(input_data.get('physical_activity'))
    features_dict['social_life'] = safe_float(input_data.get('social_life'))
    features_dict['cognitive_activities'] = safe_float(input_data.get('cognitive_activities'))
    features_dict['nutrition_score'] = safe_float(input_data.get('nutrition_score'))
    features_dict['sleep_deprivation'] = safe_float(input_data.get('sleep_deprivation'))

    features_df = pd.DataFrame([features_dict], columns=FEATURES_ALL_PLUS_PLUS)
    return features_df


def predict_risk_scores(input_data):
    """Prédit les scores de risque avec les modèles ML (0-100%)"""
    
    features_df = prepare_risk_features(input_data)
    print("DEBUG risk features:")
    print(features_df.T)
    print(features_df.dtypes)
    model_dementia = load_risk_model('dementia')
    model_handicap = load_risk_model('handicap')

    
    print("TYPE model_dementia:", type(model_dementia))
    print("TYPE model_handicap:", type(model_handicap))
    print("HAS predict dementia:", hasattr(model_dementia, "predict"))
    print("HAS predict handicap:", hasattr(model_handicap, "predict"))

    pred = model_dementia.predict(features_df)
    print("RAW pred dementia repr:", repr(pred))


    risk_dementia = 56.0
    risk_handicap = 50.0
    
    if model_dementia is not None:
        try:
            pred = model_dementia.predict(features_df)
            print("DEBUG dementia predict raw:", pred, type(pred))

            if pred is None:
                raise ValueError("model_dementia.predict(...) returned None")

            pred_value = pred[0] if hasattr(pred, "__len__") else pred
            print("DEBUG dementia pred_value:", pred_value, type(pred_value))

            if pred_value is None:
                raise ValueError("pred_value is None")

            risk_dementia = float(pred_value)
            risk_dementia = max(0.0, min(100.0, risk_dementia))

        except Exception as e:
            print(f"⚠️ Erreur prédiction risque démence : {e}")
            risk_dementia = 50.0
    
    if model_handicap is not None:
        try:
            pred = model_handicap.predict(features_df)
            print("DEBUG handicap predict raw:", pred, type(pred))

            if pred is None:
                raise ValueError("model_handicap.predict(...) returned None")

            pred_value = pred[0] if hasattr(pred, "__len__") else pred
            print("DEBUG handicap pred_value:", pred_value, type(pred_value))

            if pred_value is None:
                raise ValueError("pred_value is None")

            risk_handicap = float(pred_value)
            risk_handicap = max(0.0, min(100.0, risk_handicap))

        except Exception as e:
            print(f"⚠️ Erreur prédiction risque handicap : {e}")
            risk_handicap = 50.0
    
    return {
        'risk_dementia': risk_dementia,
        'risk_handicap': risk_handicap
    }


# ========== UTILITAIRES ==========

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
    """Calcule les zones diagnostiques globales"""
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


# ========== API ENDPOINTS ==========

@api_view(['POST'])
def predict_api(request):
    """API principale de prédiction"""
    try:
        input_data = request.data
        print(f"📥 Requête : {list(input_data.keys())[:5]}...")
        
        if not DATA_PATH.exists():
            return Response({'error': f'Fichier non trouvé: {DATA_PATH}'}, status=status.HTTP_404_NOT_FOUND)
        
        df_raw = pd.read_excel(DATA_PATH)
        print(f"📂 Données : {len(df_raw)} lignes")
        
        # Normaliser les noms de colonnes
        df = df_raw.copy()
        
        for col in ['age', 'Age', 'AGE']:
            if col in df.columns:
                df.rename(columns={col: 'age'}, inplace=True)
                break
        
        for col in ['sex', 'Sex', 'SEX']:
            if col in df.columns:
                df.rename(columns={col: 'sex'}, inplace=True)
                break
        
        for col in ['diagnosis', 'Diagnosis', 'dementia_dx_code', 'Dementia_Dx_Code']:
            if col in df.columns:
                df.rename(columns={col: 'diagnosis'}, inplace=True)
                break
        
        for col in ['neurocog_age_flu_weight', 'Neurocog_Age_Flu_Weight', 'NCA']:
            if col in df.columns:
                df.rename(columns={col: 'neurocog_age_flu_weight'}, inplace=True)
                break
        
        for col in ['education', 'Education', 'educ']:
            if col in df.columns:
                df.rename(columns={col: 'education'}, inplace=True)
                break
        
        if 'delta_NCA' not in df.columns and 'neurocog_age_flu_weight' in df.columns:
            df['delta_NCA'] = df['neurocog_age_flu_weight'] - df['age']
            print(f"✅ delta_NCA calculé : {df['delta_NCA'].mean():.2f}")
        
        # Filtrer les NaN
        required_cols = ['age', 'sex', 'diagnosis']
        available_required = [col for col in required_cols if col in df.columns]
        
        df_before = len(df)
        df = df.dropna(subset=available_required).copy()
        df_after = len(df)
        
        print(f"📊 Dataset avant filtrage : {df_before} patients")
        print(f"📊 Dataset après filtrage : {df_after} patients")
        if df_before != df_after:
            print(f"⚠️ {df_before - df_after} lignes supprimées (NaN)")
        
        # Prédiction NCA
        nca_result = predict_nca(input_data)
        predicted_delta_nca = nca_result['delta_nca']
        patient_age = nca_result['age_chronologique']
        patient_sex = int(input_data.get('sex', 1))
        
        # Zones diagnostiques
        zones_male = calculate_diagnostic_zones_CORRECT(df, sex_value=1)
        zones_female = calculate_diagnostic_zones_CORRECT(df, sex_value=0)
        
        # Courbes de centiles
        df_con = df[df['diagnosis'] == 'CON'].copy()
        
        try:
            lms_male = calculate_centile_curves_lms(df=df_con, sex_value=1, age_col='age', value_col='delta_NCA',
                                                     centiles=[3, 10, 25, 50, 75, 90, 97], window=3)
            lms_female = calculate_centile_curves_lms(df=df_con, sex_value=0, age_col='age', value_col='delta_NCA',
                                                       centiles=[3, 10, 25, 50, 75, 90, 97], window=3)
        except Exception as e:
            print(f"⚠️ Erreur calcul LMS : {e}")
            lms_male = {'curves': [], 'lms_parameters': {}}
            lms_female = {'curves': [], 'lms_parameters': {}}
        
        # Centile patient
        if lms_male.get('lms_parameters') and lms_female.get('lms_parameters'):
            patient_lms = lms_male['lms_parameters'] if patient_sex == 1 else lms_female['lms_parameters']
            patient_centile = calculate_patient_centile_lms(age=patient_age, value=predicted_delta_nca,
                                                             lms_parameters=patient_lms)
        else:
            patient_centile = None
        
        if patient_centile is None:
            patient_centile = {
                'centile': 50,
                'z_score': 0,
                'interpretation': 'Calcul non disponible',
                'lms_parameters': {'L': 0, 'M': predicted_delta_nca, 'S': 1}
            }
        
        # Zone patient
        limits = zones_male['limits'] if patient_sex == 1 else zones_female['limits']
        if predicted_delta_nca < limits['normal_mci']:
            patient_zone = 'Normale'
        elif predicted_delta_nca < limits['mci_ad']:
            patient_zone = 'MCI'
        else:
            patient_zone = 'Pathologique'
        
        # Cohorte de référence
        reference_cohort = []
        df_sample = df
        
        def convert_education_years_to_group(years):
            if pd.isna(years):
                return 0
            years = float(years)
            if years <= 12:
                return 0
            elif years <= 15:
                return 1
            elif years <= 20:
                return 2
            else:
                return 3
        
        for _, row in df_sample.iterrows():
            if 'neurocog_age_flu_weight' in row and not pd.isna(row['neurocog_age_flu_weight']):
                nca_value = float(row['neurocog_age_flu_weight'])
            elif 'delta_NCA' in row and not pd.isna(row['delta_NCA']):
                nca_value = float(row['age']) + float(row['delta_NCA'])
            else:
                nca_value = float(row['age'])
            
            education_group = convert_education_years_to_group(row.get('education', 0))
            
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
        
        # Calcul des risques
        risk_scores = predict_risk_scores(input_data=input_data)
        
        print(f"📊 Risque démence : {risk_scores['risk_dementia']:.1f}%")
        print(f"📊 Risque handicap : {risk_scores['risk_handicap']:.1f}%")
        
        # Résultat
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
            'risk_scores': {
                'risk_dementia': clean_numeric_value(risk_scores['risk_dementia']),
                'risk_handicap': clean_numeric_value(risk_scores['risk_handicap'])
            },
            'reference_cohort': reference_cohort,
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
        risk_dementia_exists = RISK_DEMENTIA_MODEL_PATH.exists()
        risk_handicap_exists = RISK_HANDICAP_MODEL_PATH.exists()
        
        model_loaded = False
        try:
            model = load_nca_model()
            model_loaded = True
        except:
            pass
        
        return Response({
            'status': 'healthy' if (data_exists and model_loaded) else 'warning',
            'data_file_exists': data_exists,
            'nca_model_exists': model_exists,
            'nca_model_loaded': model_loaded,
            'risk_dementia_model_exists': risk_dementia_exists,
            'risk_handicap_model_exists': risk_handicap_exists,
            'success': True
        })
    except Exception as e:
        return Response({'status': 'error', 'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def model_info_api(request):
    """Informations sur le modèle"""
    try:
        model = load_nca_model()
        
        return Response({
            'model_type': 'Cognitive Aging - NCA avec gestion NaN',
            'nca_model': {'loaded': True, 'type': str(type(model).__name__)},
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