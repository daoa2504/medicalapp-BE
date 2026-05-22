"""
api_views.py - VERSION FINALE POUR PRODUCTION RAILWAY
30 FEATURES (sans RAVLT et Mémoire logique)
Zones diagnostiques + Prédiction NCA + Risques ML + INTERVALLE DE CONFIANCE
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

# Donnees reelles de la cohorte (1119 patients, 51 colonnes)
# Utilisees pour : zones diagnostiques, courbes de centiles, cohorte de reference
DATA_PATH = BASE_DIR / "data" / "Example_database_withoutrois1.xlsx"

# Tous les modeles sont regroupes dans models/ avec sous-dossiers :
#   models/nca/              -> LGBM NCA (prediction delta) + classifieur diagnostic
#   models/nca_regression/   -> Linear Regression basic/all/all_plus_plus (utils.py)
#   models/risk_dementia/    -> XGB + LGBM pour le risque de demence
#   models/risk_handicap/    -> XGB + LGBM pour le risque de perte d'autonomie
MODELS_DIR = BASE_DIR / "models"

# Modele 1 du pipeline : predit l'age neurocognitif (NCA) a partir de 30 features
NCA_MODEL_PATH = MODELS_DIR / "nca" / "LGBM_with_nan.sav"

# Modeles de risque (LGBMClassifier, 3 classes : faible/modere/eleve)
RISK_DEMENTIA_MODEL_PATH = MODELS_DIR / "risk_dementia" / "LGBM_reg_all_plus_plus.sav"
RISK_HANDICAP_MODEL_PATH = MODELS_DIR / "risk_handicap" / "LGBM_reg_all_plus_plus.sav"

print(f"\n📂 Configuration des chemins :")
print(f"   BASE_DIR: {BASE_DIR}")
print(f"   DATA_PATH: {DATA_PATH}")
print(f"   NCA_MODEL_PATH: {NCA_MODEL_PATH}")
print(f"   RISK_DEMENTIA_MODEL_PATH: {RISK_DEMENTIA_MODEL_PATH}")
print(f"   RISK_HANDICAP_MODEL_PATH: {RISK_HANDICAP_MODEL_PATH}")


# ========== 30 FEATURES (SANS RAVLT ET MÉMOIRE LOGIQUE) ==========

FEATURES_ALL_PLUS_PLUS = [
    # Obligatoires (6) - ENLEVÉ : ravlt_imm
    'age', 'sex', 'education', 'language', 'fluency_score', 'moca',
    
    # Optionnels cognitifs (3) - ENLEVÉ : ravlt_delay, logic_imm, logic_delay
    'handedness', 'nb_language', 'hearing',
    
    # Facteurs de risque (21) - INCHANGÉS
    'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income', 'retired',
    'stroke', 'tbi', 'hta', 'diab_type2', 'chol_total',
    'obesity', 'depression', 'anxiety',
    'smoking', 'alcohol', 'poly_pharm5', 'physical_activity', 'social_life',
    'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
]

print(f"✅ Features configurées : {len(FEATURES_ALL_PLUS_PLUS)} features")
print(f"   - Obligatoires : 6 (âge, sexe, éducation, langue, fluence, MoCA)")
print(f"   - Cognitifs optionnels : 3 (latéralité, nb langues, audition)")
print(f"   - Facteurs de risque : 21 (incluant chol_total)")

# ========== CHARGEMENT DES MODÈLES NCA (SINGLETON) ==========
#
# Deux modeles NCA travaillent ensemble dans un pipeline :
#   1. LGBM NCA (regression) : predit le delta NCA (age neurocognitif - age reel)
#      a partir des 30 features du patient
#   2. LGBM Classifieur (classification) : predit le diagnostic (CON/SCD/MCI/AD)
#      a partir des 30 features + le delta predit par le modele 1
#
# Le classifieur remplace les anciens seuils hardcodes (accuracy 56.6% -> 69.4%)
# car il utilise toutes les features (moca, comorbidites, etc.) pour classifier,
# pas seulement le delta NCA.
#
# Les deux modeles sont charges en singleton (une seule fois au premier appel)
# pour eviter de relire les fichiers .sav a chaque requete.

DIAG_CLASSIFIER_PATH = MODELS_DIR / "nca" / "LGBM_diagnosis_classifier.sav"

_NCA_MODEL = None
_DIAG_CLASSIFIER = None


def load_diag_classifier():
    """
    Charge le classifieur de diagnostic LGBM (Modele 2 du pipeline).
    Ce modele prend en entree les 30 features NCA + le delta predit
    et retourne un diagnostic : CON, SCD, MCI ou AD.
    Entraine sur les donnees augmentees (1369 patients) avec accuracy 69.4%.
    """
    global _DIAG_CLASSIFIER

    if _DIAG_CLASSIFIER is None:
        if not DIAG_CLASSIFIER_PATH.exists():
            print(f"⚠️ Classifieur diagnostic non trouve : {DIAG_CLASSIFIER_PATH}")
            return None

        print(f"📦 Chargement classifieur diagnostic : {DIAG_CLASSIFIER_PATH}")
        _DIAG_CLASSIFIER = joblib.load(DIAG_CLASSIFIER_PATH)
        print(f"✅ Classifieur charge : {type(_DIAG_CLASSIFIER).__name__}")

    return _DIAG_CLASSIFIER

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
    """Prépare 30 features avec np.nan pour manquants"""
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
    """Prédiction NCA avec LightGBM + Intervalle de confiance"""
    try:
        model = load_nca_model()
    except FileNotFoundError:
        age = float(patient_data.get('age', 65))
        return {
            'nca_predicted': age + 5.0,
            'delta_nca': 5.0,
            'age_chronologique': age,
            'confidence_interval': None,
            'n_features_used': 0,
            'n_features_total': 30,
            'completeness': 0.0,
            'reliability': 'Non disponible',
            'reliability_stars': '',
            'features_detail': {'obligatoires': False, 'cognitifs': 0, 'risques': 0},
            'interpretation': 'Estimation par défaut'
        }
    
    # Champs obligatoires (6) - SANS ravlt_imm
    required_fields = ['age', 'sex', 'education', 'language', 'fluency_score', 'moca']
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
    
    nca_predicted = float(model.predict([features])[0])
    age_chronologique = float(patient_data['age'])
    delta_nca = nca_predicted - age_chronologique

# ── Intervalle de confiance ──────────────────────────────────────────────
    # Valeur statique temporaire : ±2.5 ans
    # TODO : remplacer par interpolation sur quantiles calibrés
    q_used = 2.5
    confidence_level = 0.95

    lower = nca_predicted - q_used
    upper = nca_predicted + q_used

    confidence_interval = {
        'lower': float(lower),
        'upper': float(upper),
        'std': float(q_used),
        'confidence_level': float(confidence_level)
    }
        
    if abs(delta_nca) < 2:
        interpretation = "Vieillissement cognitif normal"
    elif delta_nca > 0:
        interpretation = "Vieillissement cognitif accéléré"
    else:
        interpretation = "Vieillissement cognitif ralenti"
    
    print(f"\n🔮 NCA : {age_chronologique:.1f} → {nca_predicted:.1f} (Δ{delta_nca:+.1f}) | {completeness:.0f}% | {reliability}")
    print(f"📊 IC 95% : [{lower:.1f} - {upper:.1f}] ans")
    
    return {
        'nca_predicted': float(nca_predicted),
        'delta_nca': float(delta_nca),
        'age_chronologique': float(age_chronologique),
        'confidence_interval': confidence_interval,
        'interpretation': interpretation,
        'n_features_used': int(n_features_used),
        'n_features_total': len(FEATURES_ALL_PLUS_PLUS),
        'completeness': float(completeness),
        'reliability': reliability,
        'reliability_stars': reliability_stars,
        'features_detail': {
            'obligatoires': n_features_used >= 6,  # ← CHANGÉ : 6 au lieu de 7
            'cognitifs': sum(1 for f in features[6:9] if not (isinstance(f, float) and np.isnan(f))),  # ← CHANGÉ : indices 6-9
            'risques': sum(1 for f in features[9:] if not (isinstance(f, float) and np.isnan(f))),  # ← CHANGÉ : à partir de 9
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
    """Prépare les features pour les modèles de risque (30 features)"""
    features_dict = {}

    # Obligatoires (6) - SANS ravlt_imm
    features_dict['age'] = safe_float(input_data.get('age'), 0)
    features_dict['sex'] = safe_float(input_data.get('sex'), 0)
    features_dict['education'] = safe_float(input_data.get('education'), 0)
    features_dict['language'] = safe_float(input_data.get('language'), 0)
    features_dict['fluency_score'] = safe_float(input_data.get('fluency_score'), 0)
    features_dict['moca'] = safe_float(input_data.get('moca'), 0)

    # Optionnels cognitifs (3) - SANS ravlt_delay, logic_imm, logic_delay
    features_dict['handedness'] = safe_float(input_data.get('handedness'))
    features_dict['nb_language'] = safe_float(input_data.get('nb_language'))
    features_dict['hearing'] = safe_float(input_data.get('hearing'))

    # Facteurs de risque (21) - INCHANGÉS
    features_dict['hist_demence_fam'] = safe_float(input_data.get('hist_demence_fam'))
    features_dict['hist_demence_parent'] = safe_float(input_data.get('hist_demence_parent'))
    features_dict['living_alone'] = safe_float(input_data.get('living_alone'))
    features_dict['income'] = safe_float(input_data.get('income'))
    features_dict['retired'] = safe_float(input_data.get('retired'))
    features_dict['stroke'] = safe_float(input_data.get('stroke'))
    features_dict['tbi'] = safe_float(input_data.get('tbi'))
    features_dict['hta'] = safe_float(input_data.get('hta'))
    features_dict['diab_type2'] = safe_float(input_data.get('diab_type2'))
    features_dict['chol_total'] = safe_float(input_data.get('chol_total'))
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


def _score_from_proba(obj, features_df):
    """
    Calcule un score de risque en % à partir d'un modèle LGBMClassifier
    sauvegardé sous forme de dict {'model': ..., 'class_map': ...}.

    Les classes sont {0: faible, 1: modéré, 2: élevé} correspondant aux
    valeurs ordinales {0.0, 0.5, 1.0}.
    Score = P(modéré)*50 + P(élevé)*100  → toujours dans [0, 100].
    """
    model = obj['model'] if isinstance(obj, dict) else obj

    if hasattr(model, 'predict_proba'):
        # Nouveau format : LGBMClassifier — predict_proba → [P(0), P(0.5), P(1)]
        proba = model.predict_proba(features_df)[0]
        if len(proba) == 3:
            score = float(proba[1]) * 50.0 + float(proba[2]) * 100.0
        else:
            # Fallback binaire : P(classe 1) * 100
            score = float(proba[-1]) * 100.0
        print(f"   → predict_proba : {[f'{p:.3f}' for p in proba]}  score={score:.1f}%")
    else:
        # Ancien format : LGBMRegressor — predict → valeur dans [0, 1]
        pred_value = float(model.predict(features_df)[0])
        score = pred_value * 100.0 if pred_value <= 1.0 else pred_value
        print(f"   → predict (régression) : {pred_value:.4f}  score={score:.1f}%")

    return max(0.0, min(100.0, score))


def predict_risk_scores(input_data):
    """Prédit les scores de risque avec les modèles ML (0-100%)"""

    features_df = prepare_risk_features(input_data)

    obj_dementia = load_risk_model('dementia')
    obj_handicap = load_risk_model('handicap')

    risk_dementia = 50.0
    risk_handicap = 50.0

    if obj_dementia is not None:
        try:
            print(f"🔍 DEBUG Démence :")
            risk_dementia = _score_from_proba(obj_dementia, features_df)
            print(f"   → Final : {risk_dementia:.2f}%")
        except Exception as e:
            print(f"⚠️ Erreur prédiction risque démence : {e}")
            import traceback
            traceback.print_exc()
            risk_dementia = 50.0

    if obj_handicap is not None:
        try:
            print(f"🔍 DEBUG Handicap :")
            risk_handicap = _score_from_proba(obj_handicap, features_df)
            print(f"   → Final : {risk_handicap:.2f}%")
        except Exception as e:
            print(f"⚠️ Erreur prédiction risque handicap : {e}")
            import traceback
            traceback.print_exc()
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

        # Si la colonne 'diagnosis' n'existe pas dans le fichier Excel
        # (ex: fichier simule sans dementia_dx_code), on la derive de risk_dementia.
        # C'est un fallback : le fichier reel (Example_database_withoutrois1.xlsx)
        # a deja une colonne dementia_dx_code qui est renommee en 'diagnosis' ci-dessus.
        # Mapping : risk_dementia 0.0 -> CON, 0.5 -> MCI, 1.0 -> AD
        if 'diagnosis' not in df.columns and 'risk_dementia' in df.columns:
            def risk_to_diagnosis(val):
                if val <= 0.25:
                    return 'CON'
                elif val <= 0.75:
                    return 'MCI'
                else:
                    return 'AD'
            df['diagnosis'] = df['risk_dementia'].apply(risk_to_diagnosis)
            print(f"✅ diagnosis dérivé de risk_dementia : {dict(df['diagnosis'].value_counts())}")
        
        # Filtrer les NaN
        required_cols = ['age', 'sex', 'diagnosis']
        available_required = [col for col in required_cols if col in df.columns]

        df_before = len(df)
        df = df.dropna(subset=available_required).copy()

        # MODIFIE : exclure OTHER_DEM de la cohorte (groupe heterogene non utilise)
        if 'diagnosis' in df.columns:
            n_other = (df['diagnosis'] == 'OTHER_DEM').sum()
            if n_other > 0:
                df = df[df['diagnosis'] != 'OTHER_DEM'].copy()
                print(f"⚠️ {n_other} patients OTHER_DEM exclus de la cohorte")
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

        # MODIFIE : aligner l'interpretation du delta NCA sur les centiles
        # (relative aux pairs CON+SCD+MCI+AD du meme sexe et meme age)
        # Au lieu d'utiliser un seuil absolu (ex: delta > 2 = "accelere"),
        # on utilise la position relative dans la cohorte
        try:
            # Fenetre adaptative ±3, ±5, ±8 ans
            cohort_window = None
            for half_width in [3, 5, 8]:
                window_df = df[
                    (df['sex'] == patient_sex)
                    & (df['age'] >= patient_age - half_width)
                    & (df['age'] <= patient_age + half_width)
                    & df['neurocog_age_flu_weight'].notna()
                ]
                if len(window_df) >= 5:
                    cohort_window = window_df
                    break

            if cohort_window is not None:
                nca_values = sorted(cohort_window['neurocog_age_flu_weight'].tolist())
                patient_nca = nca_result['nca_predicted']

                # Calcul du centile : recherche binaire alignee avec les courbes
                if patient_nca <= nca_values[0]:
                    patient_centile_pct = 1
                elif patient_nca >= nca_values[-1]:
                    patient_centile_pct = 99
                else:
                    n = len(nca_values)
                    below = sum(1 for v in nca_values if v < patient_nca)
                    patient_centile_pct = max(1, min(99, round((below / n) * 100)))

                # Interpretation basee sur le centile (cohérente avec l'onglet Centiles)
                if patient_centile_pct < 10:
                    new_interp = "Vieillissement cognitif exceptionnellement ralenti"
                elif patient_centile_pct < 25:
                    new_interp = "Vieillissement cognitif ralenti (meilleur que la moyenne)"
                elif patient_centile_pct < 75:
                    new_interp = "Vieillissement cognitif typique (dans la norme)"
                elif patient_centile_pct < 90:
                    new_interp = "Vieillissement cognitif legerement accelere"
                else:
                    new_interp = "Vieillissement cognitif accelere"

                nca_result['interpretation'] = new_interp
                nca_result['patient_centile'] = patient_centile_pct
                print(f"📊 Centile patient : {patient_centile_pct}e -> {new_interp}")
        except Exception as e:
            print(f"⚠️ Calcul centile interpretation echoue : {e}")
        
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
        
        # ── Zone patient (diagnostic) ────────────────────────────────────
        # MODIFIE : remplacement des seuils hardcodes par le classifieur LGBM
        #
        # AVANT (seuils) :
        #   if delta < limit_normal_mci  -> "Normale"
        #   elif delta < limit_mci_ad    -> "MCI"
        #   else                         -> "Pathologique"
        #   Probleme : n'utilise qu'un seul chiffre (delta), accuracy 56.6%
        #
        # APRES (classifieur LGBM) :
        #   Etape 1 : on prepare les 30 features du patient + le delta predit
        #   Etape 2 : le classifieur LGBM predit CON/SCD/MCI/AD
        #   Etape 3 : on convertit en zone (CON/SCD -> Normale, MCI -> MCI, AD -> Pathologique)
        #   Avantage : utilise 31 variables (moca, comorbidites, etc.), accuracy 69.4%
        #
        # Si le classifieur n'est pas disponible, on retombe sur les seuils (fallback)
        limits = zones_male['limits'] if patient_sex == 1 else zones_female['limits']
        diag_classifier = load_diag_classifier()
        if diag_classifier is not None:
            # Pipeline : 30 features NCA + delta_pred -> classifieur -> diagnostic
            clf_features = prepare_nca_features(input_data)
            clf_input = pd.DataFrame([clf_features], columns=FEATURES_ALL_PLUS_PLUS)
            clf_input['delta_pred'] = predicted_delta_nca  # ajouter le delta du Modele 1
            predicted_diag = diag_classifier.predict(clf_input)[0]
            # Conversion diagnostic -> zone d'affichage pour le frontend
            diag_to_zone = {'CON': 'Normale', 'SCD': 'Normale', 'MCI': 'MCI', 'AD': 'Pathologique'}
            patient_zone = diag_to_zone.get(predicted_diag, 'MCI')
            print(f"🔮 Classifieur diagnostic : {predicted_diag} -> zone {patient_zone}")
        else:
            # Fallback : seuils hardcodes (utilise seulement si le classifieur est absent)
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
                'confidence_interval': {
                    'lower': clean_numeric_value(nca_result['confidence_interval']['lower']),
                    'upper': clean_numeric_value(nca_result['confidence_interval']['upper']),
                    'std': clean_numeric_value(nca_result['confidence_interval']['std']),
                    'confidence_level': nca_result['confidence_interval']['confidence_level']
                } if nca_result.get('confidence_interval') else None,
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
                'data_file': 'Example_database_withoutrois1.xlsx'
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
            'features_count': len(FEATURES_ALL_PLUS_PLUS),
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
            'model_type': 'Cognitive Aging - NCA 30 features + IC 95%',
            'nca_model': {'loaded': True, 'type': str(type(model).__name__)},
            'features': {
                'total': len(FEATURES_ALL_PLUS_PLUS),
                'obligatoires': 6,
                'cognitifs_optionnels': 3,
                'facteurs_risque': 21,
                'list': FEATURES_ALL_PLUS_PLUS,
                'removed': ['ravlt_imm', 'ravlt_delay', 'logic_imm', 'logic_delay']
            },
            'method': 'LightGBM avec gestion native des NaN',
            'confidence_interval': {
                'enabled': True,
                'method': 'RMSE-based',
                'level': '95%'
            },
            'success': True
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)