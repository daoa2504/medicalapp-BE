"""
utils.py - VERSION CORRIGÉE POUR PRODUCTION
Chemins relatifs + Lazy loading des modèles
"""

import pickle
import joblib
from pathlib import Path

# ========== CONFIGURATION DES CHEMINS ==========

# BASE_DIR pointe vers predictions/
BASE_DIR = Path(__file__).resolve().parent

# Dossier des modèles
MODEL_DIR = BASE_DIR / 'models'

print(f"📂 utils.py - MODEL_DIR: {MODEL_DIR}")


# ========== FONCTIONS UTILITAIRES ==========

def load_model(model_path):
    """Charge un modèle depuis un fichier .sav"""
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
    return joblib.load(model_path)


def pred_to_proba(y_pred):
    """Convertit les prédictions en probabilités (0-1)"""
    predictions = []
    for value in y_pred:
        value = int(value * 100)
        if value < 0:
            value = 0
        elif value > 100:
            value = 100
        predictions.append(value / 100)
    return predictions


# ========== FONCTIONS DE PRÉDICTION ==========

def predict_with_model_1(data):
    """
    Prédiction avec modèle basique (5 features)
    Features: age, sex, education, language, fluency_score
    """
    # Extraire les champs nécessaires
    necessary_fields = {
        'age': data['age'],
        'sex': data['sex'],
        'education': data['education'],
        'language': data['language'],
        'fluency_score': data['fluency_score']
    }

    # MODIFIE : chemin reorganise dans models/nca_regression/
    model1 = load_model(MODEL_DIR / 'nca_regression' / 'Linear_reg_basic.sav')
    prediction = model1.predict([list(necessary_fields.values())])

    # Calculer les métriques
    age = float(data['age'])
    delta_neurocogage_flu_weight = prediction - age
    necessary_fields['neurocog_age_flu_weight'] = prediction[0]

    # Modèles de risque
    model2 = load_model(MODEL_DIR / 'risk_dementia' / 'XGB_reg_basic.sav')
    model3 = load_model(MODEL_DIR / 'risk_handicap' / 'XGB_reg_basic.sav')

    # Prédictions de risque
    risk_dementia = pred_to_proba(model2.predict([list(necessary_fields.values())]))
    risk_handicap = pred_to_proba(model3.predict([list(necessary_fields.values())]))

    return {
        'neurocog_age_flu_weight': int(prediction[0] * 100) / 100,
        'delta_neurocogage_flu_weight': int(delta_neurocogage_flu_weight[0] * 100) / 100,
        'risk_dementia': risk_dementia[0],
        'risk_handicap': risk_handicap[0]
    }


def predict_with_model_2(data):
    """
    Prédiction avec modèle complet (13 features)
    Features: age, sex, education, language, fluency_score,
              handedness, nb_language, hearing, moca, ravlt_imm,
              ravlt_delay, logic_imm, logic_delay
    """
    # Extraire tous les champs
    all_fields = {
        'age': data['age'],
        'sex': data['sex'],
        'education': data['education'],
        'language': data['language'],
        'fluency_score': data['fluency_score'],
        'handedness': data['handedness'],
        'nb_language': data['nb_language'],
        'hearing': data['hearing'],
        'moca': data['moca'],
        'ravlt_imm': data['ravlt_imm'],
        'ravlt_delay': data['ravlt_delay'],
        'logic_imm': data['logic_imm'],
        'logic_delay': data['logic_delay']
    }

    # MODIFIE : chemin reorganise dans models/nca_regression/
    model1 = load_model(MODEL_DIR / 'nca_regression' / 'Linear_reg_all.sav')
    prediction = model1.predict([list(all_fields.values())])

    # Calculer les métriques
    age = float(data['age'])
    delta_neurocogage_flu_weight = prediction - age
    all_fields['neurocog_age_flu_weight'] = prediction[0]

    # Modèles de risque
    model2 = load_model(MODEL_DIR / 'risk_dementia' / 'XGB_reg_all.sav')
    model3 = load_model(MODEL_DIR / 'risk_handicap' / 'XGB_reg_all.sav')

    # Prédictions de risque
    risk_dementia = pred_to_proba(model2.predict([list(all_fields.values())]))
    risk_handicap = pred_to_proba(model3.predict([list(all_fields.values())]))

    return {
        'neurocog_age_flu_weight': int(prediction[0] * 100) / 100,
        'delta_neurocogage_flu_weight': int(delta_neurocogage_flu_weight[0] * 100) / 100,
        'risk_dementia': risk_dementia[0],
        'risk_handicap': risk_handicap[0]
    }


def predict_with_model_3(data):
    """
    Prédiction avec modèle avancé (34 features)
    Inclut tous les champs + facteurs de risque
    """
    # Extraire tous les champs plus facteurs de risque
    all_fields_plus_plus = {
        'age': data['age'],
        'sex': data['sex'],
        'education': data['education'],
        'language': data['language'],
        'fluency_score': data['fluency_score'],
        'handedness': data['handedness'],
        'nb_language': data['nb_language'],
        'hearing': data['hearing'],
        'moca': data['moca'],
        'ravlt_imm': data['ravlt_imm'],
        'ravlt_delay': data['ravlt_delay'],
        'logic_imm': data['logic_imm'],
        'logic_delay': data['logic_delay'],
        'hist_demence_fam': data['hist_demence_fam'],
        'hist_demence_parent': data['hist_demence_parent'],
        'living_alone': data['living_alone'],
        'income': data['income'],
        'retired': data['retired'],
        'stroke': data['stroke'],
        'tbi': data['tbi'],
        'hta': data['hta'],
        'diab_type2': data['diab_type2'],
        'obesity': data['obesity'],
        'depression': data['depression'],
        'anxiety': data['anxiety'],
        'smoking': data['smoking'],
        'alcohol': data['alcohol'],
        'poly_pharm5': data['poly_pharm5'],
        'physical_activity': data['physical_activity'],
        'social_life': data['social_life'],
        'cognitive_activities': data['cognitive_activities'],
        'nutrition_score': data['nutrition_score'],
        'sleep_deprivation': data['sleep_deprivation']
    }

    # MODIFIE : chemin reorganise dans models/nca_regression/
    model1 = load_model(MODEL_DIR / 'nca_regression' / 'Linear_reg_all_plus_plus.sav')
    prediction = model1.predict([list(all_fields_plus_plus.values())])

    # Calculer les métriques
    age = float(data['age'])
    delta_neurocogage_flu_weight = prediction - age
    all_fields_plus_plus['neurocog_age_flu_weight'] = prediction[0]

    # Modèles de risque
    model2 = load_model(MODEL_DIR / 'risk_dementia' / 'XGB_reg_all_plus_plus.sav')
    model3 = load_model(MODEL_DIR / 'risk_handicap' / 'XGB_reg_all_plus_plus.sav')

    # Prédictions de risque
    risk_dementia = pred_to_proba(model2.predict([list(all_fields_plus_plus.values())]))
    risk_handicap = pred_to_proba(model3.predict([list(all_fields_plus_plus.values())]))

    return {
        'neurocog_age_flu_weight': int(prediction[0] * 100) / 100,
        'delta_neurocogage_flu_weight': int(delta_neurocogage_flu_weight[0] * 100) / 100,
        'risk_dementia': risk_dementia[0],
        'risk_handicap': risk_handicap[0]
    }