"""
evaluate_models.py - Evaluation des performances des modeles au demarrage
=========================================================================
Affiche R2, MAE, RMSE pour les modeles de regression NCA
et matrice de confusion pour les modeles de classification risk
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Example_database_withoutrois1.xlsx"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features
FEATURES_BASIC = ['age', 'sex', 'education', 'language', 'fluency_score']
FEATURES_ALL = [
    'age', 'sex', 'education', 'language', 'fluency_score',
    'handedness', 'nb_language', 'hearing', 'moca',
    'ravlt_imm', 'ravlt_delay', 'logic_imm', 'logic_delay'
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


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def evaluate_nca_regression(df):
    """Evalue les modeles de regression NCA (R2, MAE, RMSE)"""
    print_separator("EVALUATION NCA REGRESSION (neurocog_age_flu_weight)")

    target = 'neurocog_age_flu_weight'
    configs = [
        ('basic', FEATURES_BASIC, 'Linear_reg_basic.sav'),
        ('all', FEATURES_ALL, 'Linear_reg_all.sav'),
    ]

    for level, features, filename in configs:
        model_path = MODELS_DIR / 'nca_regression' / filename
        if not model_path.exists():
            print(f"  [{level}] Modele non trouve: {model_path}")
            continue

        df_clean = df[features + [target]].dropna()
        X = df_clean[features]
        y = df_clean[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        obj = joblib.load(model_path)
        model = obj['model']
        scaler = obj['scaler']

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n  [{level.upper()}] Linear Regression ({len(features)} features, n={len(df_clean)})")
        print(f"  {'─'*50}")
        print(f"  R2 Score   : {r2:.4f}")
        print(f"  MAE        : {mae:.2f} ans")
        print(f"  RMSE       : {rmse:.2f} ans")

    # NCA LightGBM
    lgbm_path = MODELS_DIR / 'nca' / 'LGBM_with_nan.sav'
    if lgbm_path.exists():
        available_features = [f for f in FEATURES_NCA_30 if f in df.columns]
        df_nca = df[available_features + [target]].dropna(subset=[target])
        X = df_nca[available_features]
        y = df_nca[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        model = joblib.load(lgbm_path)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n  [NCA LGBM] LightGBM ({len(available_features)} features, n={len(df_nca)})")
        print(f"  {'─'*50}")
        print(f"  R2 Score   : {r2:.4f}")
        print(f"  MAE        : {mae:.2f} ans")
        print(f"  RMSE       : {rmse:.2f} ans")


def evaluate_risk_classification(df):
    """Evalue les modeles LGBM Classifier (matrice de confusion)"""
    print_separator("EVALUATION RISK CLASSIFIERS (matrice de confusion)")

    dx_map = {'CON': 0.0, 'SCD': 0.0, 'MCI': 0.5, 'AD': 1.0, 'OTHER_DEM': 1.0}
    label_names = ['Faible (0)', 'Modere (0.5)', 'Eleve (1.0)']

    targets = [
        ('risk_dementia', 'Risque de Demence'),
        ('risk_handicap', 'Risque de Perte d\'Autonomie'),
    ]

    available_features = [f for f in FEATURES_NCA_30 if f in df.columns]

    for target_name, display_name in targets:
        model_path = MODELS_DIR / target_name / 'LGBM_reg_all_plus_plus.sav'
        if not model_path.exists():
            print(f"  [{display_name}] Modele non trouve: {model_path}")
            continue

        df_clean = df[available_features + [target_name]].dropna()
        X = df_clean[available_features]
        y = df_clean[target_name]

        # Convertir en classes ordinales (meme logique que l'entrainement)
        def to_ordinal(val):
            if val <= 0.25:
                return 0
            elif val <= 0.75:
                return 1
            else:
                return 2

        y_classes = y.apply(to_ordinal)

        min_class_count = y_classes.value_counts().min()
        stratify_param = y_classes if min_class_count >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_classes, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=stratify_param
        )

        obj = joblib.load(model_path)
        model = obj['model'] if isinstance(obj, dict) else obj

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

        print(f"\n  [{display_name}] LGBMClassifier ({len(available_features)} features, n={len(df_clean)})")
        print(f"  {'─'*50}")
        print(f"  Accuracy   : {acc:.2%}")
        print(f"\n  Matrice de confusion :")
        print(f"  {'':>15} {'Pred Faible':>12} {'Pred Modere':>12} {'Pred Eleve':>12}")
        for i, row_label in enumerate(['Vrai Faible', 'Vrai Modere', 'Vrai Eleve']):
            print(f"  {row_label:>15} {cm[i,0]:>12} {cm[i,1]:>12} {cm[i,2]:>12}")

        print(f"\n  Rapport de classification :")
        report = classification_report(y_test, y_pred, labels=[0, 1, 2],
                                       target_names=label_names, zero_division=0)
        for line in report.split('\n'):
            print(f"  {line}")


def evaluate_diagnosis_coherence(df):
    """Verifie la coherence entre delta NCA et diagnostic"""
    print_separator("COHERENCE DELTA NCA vs DIAGNOSTIC")

    if 'dementia_dx_code' not in df.columns:
        print("  Colonne dementia_dx_code absente")
        return

    df_clean = df.dropna(subset=['neurocog_age_flu_weight', 'age', 'dementia_dx_code'])
    df_clean = df_clean.copy()
    df_clean['delta'] = df_clean['neurocog_age_flu_weight'] - df_clean['age']

    print(f"\n  {'Diagnostic':<12} {'N':>5} {'Delta moyen':>12} {'Delta min':>10} {'Delta max':>10}")
    print(f"  {'─'*55}")
    for dx in ['CON', 'SCD', 'MCI', 'AD']:
        sub = df_clean[df_clean['dementia_dx_code'] == dx]
        if len(sub) > 0:
            print(f"  {dx:<12} {len(sub):>5} {sub['delta'].mean():>12.1f} {sub['delta'].min():>10.1f} {sub['delta'].max():>10.1f}")


def evaluate_population_diagnostic(df):
    """Affiche le N de la population par combinaisons de filtres (memes groupes que le frontend)"""
    print_separator("DIAGNOSTIC POPULATION (N par sous-groupe)")

    df_pop = df.dropna(subset=['age', 'sex']).copy()

    # Groupes d'age identiques au frontend : < 60, 60-80, > 80
    def age_group(age):
        if age < 60:
            return '< 60 ans'
        elif age <= 80:
            return '60-80 ans'
        else:
            return '> 80 ans'

    df_pop['groupe_age'] = df_pop['age'].apply(age_group)

    # Sexe identique au frontend
    df_pop['sexe'] = df_pop['sex'].map({0.0: 'Femme', 1.0: 'Homme'})

    # Education identique au frontend
    educ_labels = {
        0.0: 'Secondaire',
        1.0: 'Collegial',
        2.0: 'Univ. 1er cycle',
        3.0: 'Univ. sup.',
        4.0: 'Univ. sup.'
    }
    df_pop['education_label'] = df_pop['education_group'].map(educ_labels) if 'education_group' in df_pop.columns else 'N/A'

    # Diagnostic identique au frontend (OTHER_DEM exclu de toutes les analyses)
    diag_col = 'dementia_dx_code' if 'dementia_dx_code' in df_pop.columns else None
    diag_order = ['CON', 'SCD', 'MCI', 'AD']
    if diag_col:
        df_pop = df_pop[df_pop[diag_col] != 'OTHER_DEM'].copy()

    # ── Totaux par filtre individuel ──
    print(f"\n  Population totale : {len(df_pop)} patients\n")

    # Par age
    age_order = ['< 60 ans', '60-80 ans', '> 80 ans']
    age_counts = df_pop['groupe_age'].value_counts()
    parts = [f"{k}: {age_counts.get(k, 0)}" for k in age_order]
    print(f"  Par Age       : {' | '.join(parts)}")

    # Par sexe
    sex_counts = df_pop['sexe'].value_counts()
    print(f"  Par Sexe      : Femme: {sex_counts.get('Femme', 0)} | Homme: {sex_counts.get('Homme', 0)}")

    # Par education
    educ_order = ['Secondaire', 'Collegial', 'Univ. 1er cycle', 'Univ. sup.']
    educ_counts = df_pop['education_label'].value_counts()
    parts = [f"{k}: {educ_counts.get(k, 0)}" for k in educ_order]
    print(f"  Par Education : {' | '.join(parts)}")

    # Par diagnostic
    if diag_col:
        diag_counts = df_pop[diag_col].value_counts()
        parts = [f"{dx}: {diag_counts.get(dx, 0)}" for dx in diag_order]
        print(f"  Par Diagnostic: {' | '.join(parts)}")

    # ── Combinaisons detaillees ──
    if diag_col:
        group_cols = ['groupe_age', 'sexe', 'education_label', diag_col]
        col_headers = ['Age', 'Sexe', 'Education', 'Diagnostic', 'N']
    else:
        group_cols = ['groupe_age', 'sexe', 'education_label']
        col_headers = ['Age', 'Sexe', 'Education', 'N']

    grouped = df_pop.groupby(group_cols, observed=True).size().reset_index(name='N')

    # Trier par ordre logique
    age_sort = {'< 60 ans': 0, '60-80 ans': 1, '> 80 ans': 2}
    sex_sort = {'Femme': 0, 'Homme': 1}
    educ_sort = {'Secondaire': 0, 'Collegial': 1, 'Univ. 1er cycle': 2, 'Univ. sup.': 3}
    diag_sort = {'CON': 0, 'SCD': 1, 'MCI': 2, 'AD': 3}

    grouped['_age_sort'] = grouped['groupe_age'].map(age_sort)
    grouped['_sex_sort'] = grouped['sexe'].map(sex_sort)
    grouped['_educ_sort'] = grouped['education_label'].map(educ_sort)
    if diag_col:
        grouped['_diag_sort'] = grouped[diag_col].map(diag_sort)
        grouped = grouped.sort_values(['_age_sort', '_sex_sort', '_educ_sort', '_diag_sort'])
    else:
        grouped = grouped.sort_values(['_age_sort', '_sex_sort', '_educ_sort'])

    fiables = grouped[grouped['N'] >= 5]
    non_fiables = grouped[grouped['N'] < 5]

    print(f"\n  Combinaisons avec N >= 5 : {len(fiables)} sous-groupes")
    print(f"  {'─'*75}")
    if diag_col:
        print(f"  {'Age':<12} {'Sexe':<8} {'Education':<18} {'Diagnostic':<12} {'N':>5}")
    else:
        print(f"  {'Age':<12} {'Sexe':<8} {'Education':<18} {'N':>5}")
    print(f"  {'─'*75}")

    for _, row in fiables.iterrows():
        if diag_col:
            print(f"  {row['groupe_age']:<12} {row['sexe']:<8} {row['education_label']:<18} {row[diag_col]:<12} {row['N']:>5}")
        else:
            print(f"  {row['groupe_age']:<12} {row['sexe']:<8} {row['education_label']:<18} {row['N']:>5}")

    print(f"\n  Sous-groupes non fiables (N < 5) : {len(non_fiables)} combinaisons")
    if len(non_fiables) > 0:
        total_patients_non_fiables = non_fiables['N'].sum()
        print(f"  ({total_patients_non_fiables} patients au total dans ces sous-groupes)")


def run_evaluation():
    """Lance l'evaluation complete des modeles"""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "EVALUATION DES MODELES" + " " * 27 + "#")
    print("#" + f"  Donnees: {DATA_PATH.name}" + " " * (70 - len(f"  Donnees: {DATA_PATH.name}") - 2) + "#")
    print("#" * 70)

    if not DATA_PATH.exists():
        print(f"  ERREUR: Fichier non trouve: {DATA_PATH}")
        return

    df = pd.read_excel(DATA_PATH)
    print(f"\n  Donnees chargees: {df.shape[0]} patients, {df.shape[1]} colonnes")

    evaluate_nca_regression(df)
    evaluate_risk_classification(df)
    evaluate_diagnosis_coherence(df)
    evaluate_population_diagnostic(df)

    print(f"\n{'#'*70}")
    print(f"#{'EVALUATION TERMINEE':^68}#")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    run_evaluation()
