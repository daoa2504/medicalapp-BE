"""
fine_analysis.py - Analyse fine des performances du modele NCA par sous-population
==================================================================================
Objectif : identifier les points faibles du modele en segmentant les resultats par :
  - Age (< 65, 65-75, 75+)
  - Sexe (Femme, Homme)
  - Education (Secondaire, Collegial, Universitaire 1er, Universitaire sup.)

Filtrage : on exclut OTHER_DEM et AD pour se concentrer sur CON / SCD / MCI
(les 3 categories du depistage precoce).

Sortie : matrices de confusion par sous-groupe + metriques (accuracy, recall MCI).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict

# ========== CONFIG ==========

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Example_database_augmented.xlsx"
NCA_MODEL_PATH = BASE_DIR / "models" / "nca" / "LGBM_with_nan.sav"
DIAG_CLASSIFIER_PATH = BASE_DIR / "models" / "nca" / "LGBM_diagnosis_classifier.sav"

# Diagnostics inclus (on exclut AD et OTHER_DEM)
INCLUDED_DIAGNOSES = ['CON', 'SCD', 'MCI']

# Tranches d'age
AGE_GROUPS = [
    ('< 65 ans', lambda a: a < 65),
    ('65-75 ans', lambda a: 65 <= a < 75),
    ('75+ ans', lambda a: a >= 75),
]

# Sexe
SEX_GROUPS = [
    ('Femme', 0),
    ('Homme', 1),
]

# Education (education_group)
EDUC_GROUPS = [
    ('Secondaire', [0]),
    ('Collegial', [1]),
    ('Univ. 1er cycle', [2]),
    ('Univ. superieur', [3, 4]),
]

FEATURES_NCA_30 = [
    'age', 'sex', 'education', 'language', 'fluency_score', 'moca',
    'handedness', 'nb_language', 'hearing',
    'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income',
    'retired', 'stroke', 'tbi', 'hta', 'diab_type2', 'chol_total',
    'obesity', 'depression', 'anxiety', 'smoking', 'alcohol',
    'poly_pharm5', 'physical_activity', 'social_life',
    'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
]


def predict_diagnosis_from_delta(delta_nca, moca):
    """
    Predit le diagnostic (CON/SCD/MCI) a partir du delta NCA et du MoCA.
    Seuils calibres pour la tranche depistage precoce (3 classes : CON/SCD/MCI uniquement).

    Logique :
      - delta < -3 ans  : CON (cerveau plus jeune)
      - delta >= -3 et delta < 1 : SCD (proche de la norme)
      - delta >= 1 : MCI (vieillissement accelere)
    """
    if delta_nca < -3:
        return 'CON'
    elif delta_nca < 1:
        return 'SCD'
    else:
        return 'MCI'


def predict_with_classifier(X_test, delta_pred, clf_model):
    """
    Predit le diagnostic avec le classifieur LGBM (Modele 2).
    Retourne uniquement CON/SCD/MCI (mappe AD vers MCI si predit).
    """
    X_clf = X_test.copy()
    X_clf['delta_pred'] = delta_pred
    raw = clf_model.predict(X_clf)
    # Mapper AD -> MCI pour rester dans nos 3 classes
    mapped = ['MCI' if p == 'AD' else p for p in raw]
    return mapped


def compute_metrics(y_true, y_pred, labels=None):
    """Calcule les metriques de performance"""
    if labels is None:
        labels = INCLUDED_DIAGNOSES
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    # Metriques pour chaque classe
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        'n': len(y_true),
        'accuracy': acc,
        'confusion_matrix': cm,
        'mci_recall': report['MCI']['recall'] if 'MCI' in report else 0,
        'mci_precision': report['MCI']['precision'] if 'MCI' in report else 0,
        'mci_f1': report['MCI']['f1-score'] if 'MCI' in report else 0,
        'con_recall': report['CON']['recall'] if 'CON' in report else 0,
        'scd_recall': report['SCD']['recall'] if 'SCD' in report else 0,
        'report_dict': report,
    }


def print_confusion_matrix(cm, labels, indent=4):
    """Affiche une matrice de confusion lisible"""
    pad = ' ' * indent
    header = f"{pad}{'':<14}" + "".join([f"{'Pred '+l:>10}" for l in labels])
    print(header)
    print(f"{pad}{'-'*60}")
    for i, label in enumerate(labels):
        row = f"{pad}{'Vrai '+label:<14}"
        for j in range(len(labels)):
            val = cm[i, j]
            row += f"{val:>10}"
        print(row)


def analyze_subgroup(df_sub, name, _unused, model_nca, clf_model, available_features):
    """
    Analyse un sous-groupe :
      - Predit delta NCA via modele 1
      - Predit diagnostic via 2 methodes (seuils + classifieur)
      - Calcule les metriques
    """
    if len(df_sub) < 5:
        return None

    X = df_sub[available_features]
    y_true = df_sub['dementia_dx_code'].values
    age = df_sub['age'].values
    moca = df_sub['moca'].values

    # Predire delta NCA
    nca_pred = model_nca.predict(X)
    delta_pred = nca_pred - age

    # Methode 1 : seuils delta
    y_pred_thresh = [predict_diagnosis_from_delta(d, m) for d, m in zip(delta_pred, moca)]
    metrics_thresh = compute_metrics(y_true, y_pred_thresh)

    # Methode 2 : classifieur LGBM
    metrics_clf = None
    if clf_model is not None:
        y_pred_clf = predict_with_classifier(X, delta_pred, clf_model)
        metrics_clf = compute_metrics(y_true, y_pred_clf)

    return {
        'name': name,
        'n': len(df_sub),
        'distribution': dict(pd.Series(y_true).value_counts()),
        'thresh': metrics_thresh,
        'clf': metrics_clf,
        'delta_mean_per_dx': {
            dx: delta_pred[y_true == dx].mean() if (y_true == dx).any() else None
            for dx in INCLUDED_DIAGNOSES
        }
    }


def print_analysis(result):
    """Affiche les resultats d'une analyse de sous-groupe"""
    if result is None:
        return

    print(f"\n  {'='*65}")
    print(f"  Sous-groupe : {result['name']}")
    print(f"  N = {result['n']} | Distribution : {result['distribution']}")
    print(f"  {'='*65}")

    # Methode seuils
    print(f"\n  --- Methode SEUILS (delta NCA -> diagnostic) ---")
    print(f"  Accuracy : {result['thresh']['accuracy']:.1%}")
    print(f"  MCI Recall : {result['thresh']['mci_recall']:.1%}  (capacite a detecter les MCI)")
    print(f"  MCI Precision : {result['thresh']['mci_precision']:.1%}")
    print(f"  MCI F1-score : {result['thresh']['mci_f1']:.1%}")
    print()
    print_confusion_matrix(result['thresh']['confusion_matrix'], INCLUDED_DIAGNOSES)

    # Methode classifieur
    if result['clf']:
        print(f"\n  --- Methode CLASSIFIEUR LGBM ---")
        print(f"  Accuracy : {result['clf']['accuracy']:.1%}")
        print(f"  MCI Recall : {result['clf']['mci_recall']:.1%}")
        print(f"  MCI Precision : {result['clf']['mci_precision']:.1%}")
        print(f"  MCI F1-score : {result['clf']['mci_f1']:.1%}")
        print()
        print_confusion_matrix(result['clf']['confusion_matrix'], INCLUDED_DIAGNOSES)

    # Delta moyen par diagnostic
    print(f"\n  Delta NCA predit moyen par diagnostic :")
    for dx, mean in result['delta_mean_per_dx'].items():
        if mean is not None:
            print(f"    {dx:<5} : {mean:+.2f} ans")


def main():
    print("=" * 70)
    print("  ANALYSE FINE DES PERFORMANCES NCA PAR SOUS-POPULATION")
    print("=" * 70)

    # 1. Charger donnees + modeles
    print(f"\n  Chargement donnees : {DATA_PATH.name}")
    df = pd.read_excel(DATA_PATH)
    print(f"  Donnees brutes : {len(df)} patients")

    # Filtrer : CON / SCD / MCI seulement
    df = df[df['dementia_dx_code'].isin(INCLUDED_DIAGNOSES)].copy()
    df = df.dropna(subset=['neurocog_age_flu_weight', 'age', 'moca'])
    print(f"  Apres filtrage CON/SCD/MCI : {len(df)} patients")
    print(f"  Distribution : {dict(df['dementia_dx_code'].value_counts())}")

    print(f"\n  Chargement modeles...")
    model_nca = joblib.load(NCA_MODEL_PATH)
    clf_model = None
    if DIAG_CLASSIFIER_PATH.exists():
        clf_model = joblib.load(DIAG_CLASSIFIER_PATH)
        print(f"  Classifieur LGBM charge")
    else:
        print(f"  ATTENTION : classifieur non trouve, methode SEUILS uniquement")

    available_features = [f for f in FEATURES_NCA_30 if f in df.columns]

    # 2. Analyse globale (tous patients confondus)
    print("\n\n" + "=" * 70)
    print("  ANALYSE GLOBALE (tous patients CON/SCD/MCI)")
    print("=" * 70)
    global_result = analyze_subgroup(df, "GLOBAL", None, model_nca, clf_model, available_features)
    print_analysis(global_result)

    # 3. Analyse par tranche d'age
    print("\n\n" + "=" * 70)
    print("  ANALYSE PAR TRANCHE D'AGE")
    print("=" * 70)
    for age_label, age_filter in AGE_GROUPS:
        df_sub = df[df['age'].apply(age_filter)].copy()
        result = analyze_subgroup(df_sub, f"Age : {age_label}", None, model_nca, clf_model, available_features)
        print_analysis(result)

    # 4. Analyse par sexe
    print("\n\n" + "=" * 70)
    print("  ANALYSE PAR SEXE")
    print("=" * 70)
    for sex_label, sex_value in SEX_GROUPS:
        df_sub = df[df['sex'] == sex_value].copy()
        result = analyze_subgroup(df_sub, f"Sexe : {sex_label}", None, model_nca, clf_model, available_features)
        print_analysis(result)

    # 5. Analyse par education
    print("\n\n" + "=" * 70)
    print("  ANALYSE PAR NIVEAU D'EDUCATION")
    print("=" * 70)
    for educ_label, educ_values in EDUC_GROUPS:
        df_sub = df[df['education_group'].isin(educ_values)].copy()
        result = analyze_subgroup(df_sub, f"Educ : {educ_label}", None, model_nca, clf_model, available_features)
        print_analysis(result)

    # 6. Analyse croisee (age x sexe) pour reperer les sous-groupes a faible perf
    print("\n\n" + "=" * 70)
    print("  ANALYSE CROISEE (Age x Sexe) -- Synthese MCI Recall")
    print("=" * 70)
    print(f"\n  {'Age x Sexe':<25} {'N':>5} {'Acc.':>8} {'MCI Recall':>12} {'MCI Prec.':>12}")
    print(f"  {'-'*65}")
    for age_label, age_filter in AGE_GROUPS:
        for sex_label, sex_value in SEX_GROUPS:
            df_sub = df[df['age'].apply(age_filter) & (df['sex'] == sex_value)].copy()
            if len(df_sub) < 5:
                continue
            result = analyze_subgroup(df_sub, f"{age_label} x {sex_label}",
                                      None, model_nca, clf_model, available_features)
            if result:
                m = result['clf'] if result['clf'] else result['thresh']
                print(f"  {age_label+' x '+sex_label:<25} {result['n']:>5} {m['accuracy']:>8.1%} {m['mci_recall']:>12.1%} {m['mci_precision']:>12.1%}")

    print("\n\n" + "=" * 70)
    print("  ANALYSE TERMINEE")
    print("=" * 70)


if __name__ == '__main__':
    main()
