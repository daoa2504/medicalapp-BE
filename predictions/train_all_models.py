"""
train_all_models.py - Script unifie d'entrainement de tous les modeles
=====================================================================
Entraine et sauvegarde tous les modeles depuis les donnees augmentees :
  - NCA regression (Linear) : basic / all / all_plus_plus  (pour utils.py)
  - NCA LightGBM - Modele 1 : predit le delta NCA  (pour api_views.py)
  - Classifieur diagnostic - Modele 2 : predit CON/SCD/MCI/AD  (pour api_views.py)
  - Risk dementia : XGB (pour utils.py) + LGBM classifier (pour api_views.py)
  - Risk handicap : XGB (pour utils.py) + LGBM classifier (pour api_views.py)

Pipeline dans api_views.py :
  Patient -> [30 features] -> Modele 1 (NCA) -> delta
                                                   |
              [30 features + delta] -> Modele 2 (Classifieur) -> diagnostic

Usage : py train_all_models.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier

# ========== CONFIGURATION ==========

BASE_DIR = Path(__file__).resolve().parent
# Donnees augmentees (1119 reels + 250 synthetiques depistage precoce)
DATA_PATH = BASE_DIR / "data" / "Example_database_augmented.xlsx"
MODELS_DIR = BASE_DIR / "models"

# Sous-dossiers de sortie
NCA_DIR = MODELS_DIR / "nca"
NCA_REG_DIR = MODELS_DIR / "nca_regression"
RISK_DEM_DIR = MODELS_DIR / "risk_dementia"
RISK_HAN_DIR = MODELS_DIR / "risk_handicap"

# Parametres d'entrainement
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ========== DEFINITION DES FEATURES ==========

# Features pour les 3 niveaux de modeles (utils.py)
FEATURES_BASIC = ['age', 'sex', 'education', 'language', 'fluency_score']

FEATURES_ALL = [
    'age', 'sex', 'education', 'language', 'fluency_score',
    'handedness', 'nb_language', 'hearing', 'moca',
    'ravlt_imm', 'ravlt_delay', 'logic_imm', 'logic_delay'
]

FEATURES_ALL_PLUS_PLUS = [
    'age', 'sex', 'education', 'language', 'fluency_score',
    'handedness', 'nb_language', 'hearing', 'moca',
    'ravlt_imm', 'ravlt_delay', 'logic_imm', 'logic_delay',
    'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income',
    'retired', 'stroke', 'tbi', 'hta', 'diab_type2', 'obesity',
    'depression', 'anxiety', 'smoking', 'alcohol', 'poly_pharm5',
    'physical_activity', 'social_life', 'cognitive_activities',
    'nutrition_score', 'sleep_deprivation'
]

# Features pour le modele NCA LightGBM 30 features (api_views.py)
FEATURES_NCA_30 = [
    'age', 'sex', 'education', 'language', 'fluency_score', 'moca',
    'handedness', 'nb_language', 'hearing',
    'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income',
    'retired', 'stroke', 'tbi', 'hta', 'diab_type2', 'chol_total',
    'obesity', 'depression', 'anxiety', 'smoking', 'alcohol',
    'poly_pharm5', 'physical_activity', 'social_life',
    'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
]


# ========== FONCTIONS UTILITAIRES ==========

def print_metrics(name, y_true, y_pred):
    """Affiche les metriques d'erreur"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    print(f"  {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MedAE={medae:.2f}")


def pred_to_proba(y_pred):
    """Convertit les predictions en probabilites clippees [0, 1]"""
    return [max(0, min(1, int(v * 100) / 100)) for v in y_pred]


# ========== ENTRAINEMENT NCA REGRESSION (utils.py) ==========

def train_nca_regression(df):
    """Entraine les 3 modeles Linear Regression pour neurocog_age_flu_weight"""
    print("\n" + "=" * 60)
    print("ENTRAINEMENT NCA REGRESSION (Linear)")
    print("=" * 60)

    target = 'neurocog_age_flu_weight'
    configs = [
        ('basic', FEATURES_BASIC, 'Linear_reg_basic.sav'),
        ('all', FEATURES_ALL, 'Linear_reg_all.sav'),
        ('all_plus_plus', FEATURES_ALL_PLUS_PLUS, 'Linear_reg_all_plus_plus.sav'),
    ]

    for level, features, filename in configs:
        print(f"\n--- Niveau: {level} ({len(features)} features) ---")
        df_clean = df[features + [target]].dropna()
        print(f"  Donnees: {len(df_clean)} lignes apres dropna")

        X = df_clean[features]
        y = df_clean[target]

        # Split 80/20 avec random_state pour reproductibilite
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Normalisation (important pour la regression lineaire)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrainement
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = model.predict(X_test_scaled)
        print_metrics("Test", y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            model, scaler.transform(X), y,
            cv=5, scoring='neg_mean_absolute_error'
        )
        print(f"  CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

        # Sauvegarde du modele ET du scaler ensemble
        output_path = NCA_REG_DIR / filename
        joblib.dump({'model': model, 'scaler': scaler, 'features': features}, output_path)
        print(f"  Sauvegarde: {output_path}")


# ========== ENTRAINEMENT NCA LGBM (api_views.py) ==========
#
# Ce modele est le MODELE 1 du pipeline de prediction dans api_views.py
# Il predit l'age neurocognitif (neurocog_age_flu_weight) a partir de 30 features.
# Le delta NCA = age_predit - age_reel indique le vieillissement cognitif.
#
# Pourquoi LightGBM ?
#   - Gere nativement les valeurs manquantes (NaN) sans imputation
#   - Performant sur des datasets moyens (~1000 lignes)
#   - Plus rapide que XGBoost a entrainer
#
# Le modele est sauvegarde dans models/nca/LGBM_with_nan.sav
# et charge en singleton dans api_views.py via load_nca_model()

def train_nca_lgbm(df):
    """
    Entraine le modele LightGBM NCA (Modele 1 du pipeline).
    Input : 30 features du patient (age, sex, moca, comorbidites, etc.)
    Output : age neurocognitif predit (regression continue)
    Le delta = output - age_reel donne le vieillissement cognitif.
    """
    print("\n" + "=" * 60)
    print("ENTRAINEMENT NCA LightGBM (30 features, gestion NaN)")
    print("=" * 60)

    target = 'neurocog_age_flu_weight'

    # On garde les NaN dans les features (LightGBM les gere nativement)
    # mais on supprime les lignes ou la target elle-meme est NaN
    available_features = [f for f in FEATURES_NCA_30 if f in df.columns]
    print(f"  Features disponibles: {len(available_features)}/{len(FEATURES_NCA_30)}")

    df_nca = df[available_features + [target]].dropna(subset=[target]).copy()
    print(f"  Donnees: {len(df_nca)} lignes apres suppression NaN dans target")

    X = df_nca[available_features]
    y = df_nca[target]

    # Split 80/20 avec seed fixe pour reproductibilite
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Hyperparametres choisis pour eviter le surapprentissage sur ~1000 lignes :
    #   - n_estimators=300 : assez d'arbres pour capturer le signal
    #   - max_depth=6 : limite la profondeur pour generaliser
    #   - min_child_samples=20 : chaque feuille doit avoir au moins 20 exemples
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # Evaluation sur le test set (20% des donnees jamais vues)
    y_pred = model.predict(X_test)
    print_metrics("Test", y_test, y_pred)

    # Cross-validation 5 folds pour verifier la stabilite
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_absolute_error'
    )
    print(f"  CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    output_path = NCA_DIR / "LGBM_with_nan.sav"
    joblib.dump(model, output_path)
    print(f"  Sauvegarde: {output_path}")


# ========== ENTRAINEMENT RISK XGB (utils.py) ==========

def train_risk_xgb(df):
    """Entraine les modeles XGBRegressor pour risk_dementia et risk_handicap"""
    print("\n" + "=" * 60)
    print("ENTRAINEMENT RISK XGBoost (pour utils.py)")
    print("=" * 60)

    targets = [
        ('risk_dementia', RISK_DEM_DIR),
        ('risk_handicap', RISK_HAN_DIR),
    ]

    configs = [
        ('basic', FEATURES_BASIC, 'XGB_reg_basic.sav'),
        ('all', FEATURES_ALL, 'XGB_reg_all.sav'),
        ('all_plus_plus', FEATURES_ALL_PLUS_PLUS, 'XGB_reg_all_plus_plus.sav'),
    ]

    for target_name, output_dir in targets:
        print(f"\n>>> Target: {target_name}")

        for level, features, filename in configs:
            # Les features incluent neurocog_age_flu_weight comme calculee par le modele NCA
            risk_features = features + ['neurocog_age_flu_weight']
            df_clean = df[risk_features + [target_name]].dropna()
            print(f"\n  --- {level} ({len(risk_features)} features, {len(df_clean)} lignes) ---")

            X = df_clean[risk_features]
            y = df_clean[target_name]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            model = XGBRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                verbosity=0
            )
            model.fit(X_train, y_train)

            y_pred = pred_to_proba(model.predict(X_test))
            print_metrics("Test", y_test, y_pred)

            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring='neg_mean_absolute_error'
            )
            print(f"  CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

            output_path = output_dir / filename
            joblib.dump(model, output_path)
            print(f"  Sauvegarde: {output_path}")


# ========== ENTRAINEMENT RISK LGBM CLASSIFIER (api_views.py) ==========
#
# Ces modeles predisent les scores de risque affiches dans les jauges du frontend :
#   - Risque de trouble neurocognitif (risk_dementia) : base sur le profil cognitif
#   - Risque de perte d'autonomie (risk_handicap) : base sur les facteurs fonctionnels
#
# Fonctionnement :
#   1. La target continue (0.0, 0.5, 1.0) est convertie en 3 classes ordinales
#   2. Le LGBMClassifier apprend a predire la classe
#   3. predict_proba() donne les probabilites de chaque classe
#   4. Le score affiche = P(modere)*50 + P(eleve)*100  (entre 0% et 100%)
#
# Le risque handicap utilise is_unbalance=True car la classe 0 (faible) est
# sur-representee (78% des patients) -- cela force le modele a mieux apprendre
# les classes rares (modere et eleve).
#
# Les modeles sont sauvegardes comme dictionnaires {model, class_map, features, ...}
# pour que api_views.py puisse recuperer les metadonnees.

def train_risk_lgbm(df):
    """
    Entraine les modeles LGBMClassifier pour les jauges de risque du frontend.
    Chaque modele predit 3 classes : faible (0), modere (1), eleve (2).
    Le score de risque est calcule a partir des probabilites de chaque classe.
    """
    print("\n" + "=" * 60)
    print("ENTRAINEMENT RISK LGBMClassifier (pour api_views.py)")
    print("=" * 60)

    targets = [
        ('risk_dementia', RISK_DEM_DIR, False),    # distribution equilibree
        ('risk_handicap', RISK_HAN_DIR, True),     # is_unbalance=True (78% classe 0)
    ]

    available_features = [f for f in FEATURES_NCA_30 if f in df.columns]

    for target_name, output_dir, is_unbalance in targets:
        print(f"\n>>> Target: {target_name} (is_unbalance={is_unbalance})")

        df_clean = df[available_features + [target_name]].dropna()
        X = df_clean[available_features]
        y = df_clean[target_name]

        # Convertir les valeurs continues en 3 classes ordinales :
        #   0 = faible  (risk <= 0.25)  -> score proche de 0%
        #   1 = modere  (0.25 < risk <= 0.75) -> score autour de 50%
        #   2 = eleve   (risk > 0.75)  -> score proche de 100%
        def to_ordinal(val):
            if val <= 0.25:
                return 0
            elif val <= 0.75:
                return 1
            else:
                return 2

        y_classes = y.apply(to_ordinal)
        print(f"  Distribution des classes: {dict(y_classes.value_counts().sort_index())}")

        # Stratifier le split pour garder les proportions de classes dans train/test
        # Sauf si une classe a moins de 2 membres (impossible a stratifier)
        min_class_count = y_classes.value_counts().min()
        stratify_param = y_classes if min_class_count >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_classes, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=stratify_param
        )

        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            is_unbalance=is_unbalance,  # compense les classes desequilibrees
            random_state=RANDOM_STATE,
            verbose=-1
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"  Accuracy: {accuracy:.2%}")

        # Calcul du score de risque (meme formule que dans api_views.py/_score_from_proba)
        # Score = P(classe 1) * 50 + P(classe 2) * 100
        # Resultat toujours entre 0% (tout faible) et 100% (tout eleve)
        proba_test = model.predict_proba(X_test)
        if proba_test.shape[1] == 3:
            scores = proba_test[:, 1] * 50.0 + proba_test[:, 2] * 100.0
        else:
            scores = proba_test[:, -1] * 100.0
        print(f"  Score moyen: {scores.mean():.1f}%, min={scores.min():.1f}%, max={scores.max():.1f}%")

        # Sauvegarde au format dict attendu par api_views.py
        # api_views.py utilise obj['model'] pour acceder au classifieur
        class_map = {0: 0, 1: 1, 2: 2}  # 0=faible, 1=modere, 2=eleve
        output = {
            'model': model,
            'class_map': class_map,
            'class_map_inv': {v: k for k, v in class_map.items()},
            'features': available_features,
            'target': target_name,
            'is_unbalance': is_unbalance
        }

        output_path = output_dir / "LGBM_reg_all_plus_plus.sav"
        joblib.dump(output, output_path)
        print(f"  Sauvegarde: {output_path}")


# ========== ENTRAINEMENT CLASSIFIEUR DE DIAGNOSTIC (api_views.py) ==========
#
# Ce modele est le MODELE 2 du pipeline de prediction dans api_views.py.
# Il remplace les anciens seuils hardcodes pour determiner la zone diagnostique
# du patient (Normale / MCI / Pathologique).
#
# AVANT (seuils hardcodes, accuracy 56.6%) :
#   delta < seuil1 -> "Normale"
#   delta < seuil2 -> "MCI"
#   delta >= seuil2 -> "Pathologique"
#   Probleme : utilise un seul chiffre (delta) pour classifier
#
# APRES (classifieur LGBM, accuracy 69.4%) :
#   [30 features + delta_pred] -> LGBM -> CON / SCD / MCI / AD
#   Avantage : utilise toutes les features (moca, comorbidites, etc.)
#   pour distinguer les diagnostics, pas seulement le delta
#
# Le pipeline complet dans api_views.py :
#   Patient -> [30 features] -> Modele 1 (NCA LGBM) -> delta_pred
#                                                          |
#              [30 features + delta_pred] -> Modele 2 (ce classifieur) -> "MCI"
#                                                          |
#              Conversion : CON/SCD -> "Normale", MCI -> "MCI", AD -> "Pathologique"
#
# Le classifieur est sauvegarde dans models/nca/LGBM_diagnosis_classifier.sav
# et charge en singleton dans api_views.py via load_diag_classifier()

def train_diagnosis_classifier(df):
    """
    Entraine le classifieur de diagnostic LGBM (Modele 2 du pipeline).
    Input : 30 features NCA + delta_pred (calcule par le Modele 1)
    Output : diagnostic predit (CON, SCD, MCI ou AD)

    Le delta_pred est calcule en chargeant le Modele 1 (LGBM NCA) et en
    predisant l'age neurocognitif sur le train set, pour que le classifieur
    apprenne a utiliser le delta comme feature supplementaire.
    """
    print("\n" + "=" * 60)
    print("ENTRAINEMENT CLASSIFIEUR DIAGNOSTIC (Modele 2 du pipeline)")
    print("=" * 60)

    # On exclut OTHER_DEM car c'est un groupe heterogene
    # (inclut des demences non-Alzheimer avec des profils tres varies)
    diag_col = 'dementia_dx_code' if 'dementia_dx_code' in df.columns else None
    if diag_col is None:
        print("  Colonne dementia_dx_code absente - classifieur non entraine")
        return

    available_features = [f for f in FEATURES_NCA_30 if f in df.columns]

    # Filtrer : garder seulement CON, SCD, MCI, AD et les lignes avec NCA
    df_clf = df[df[diag_col].isin(['CON', 'SCD', 'MCI', 'AD'])].copy()
    df_clf = df_clf.dropna(subset=['neurocog_age_flu_weight', 'age'])
    print(f"  Donnees : {len(df_clf)} patients (CON/SCD/MCI/AD)")
    print(f"  Distribution : {dict(df_clf[diag_col].value_counts())}")

    X = df_clf[available_features].copy()
    y = df_clf[diag_col]

    # Split stratifie pour garder les proportions de chaque diagnostic
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Calculer le delta predit par le Modele 1 (NCA LGBM) pour chaque patient
    # C'est la feature supplementaire qui donne au classifieur l'info du delta
    nca_model_path = NCA_DIR / "LGBM_with_nan.sav"
    if nca_model_path.exists():
        nca_model = joblib.load(nca_model_path)
        # delta = age_neurocognitif_predit - age_reel
        X_train['delta_pred'] = nca_model.predict(X_train[available_features]) - X_train['age'].values
        X_test['delta_pred'] = nca_model.predict(X_test[available_features]) - X_test['age'].values
        print(f"  Feature delta_pred ajoutee (Modele 1 NCA)")
    else:
        # Fallback : utiliser le delta reel si le Modele 1 n'existe pas encore
        X_train['delta_pred'] = df_clf.loc[X_train.index, 'neurocog_age_flu_weight'] - X_train['age']
        X_test['delta_pred'] = df_clf.loc[X_test.index, 'neurocog_age_flu_weight'] - X_test['age']
        print(f"  Feature delta_pred calculee depuis donnees reelles (Modele 1 non disponible)")

    # Entrainer le classifieur avec 31 features (30 NCA + delta_pred)
    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {accuracy:.1%}")

    # Matrice de confusion
    from sklearn.metrics import confusion_matrix
    labels = ['CON', 'SCD', 'MCI', 'AD']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"\n  {'':>12} {'Pred CON':>10} {'Pred SCD':>10} {'Pred MCI':>10} {'Pred AD':>10}")
    print(f"  {'─'*55}")
    for i, label in enumerate(labels):
        row = f"  {'Vrai '+label:>12}"
        for j in range(4):
            row += f" {cm[i,j]:>10}"
        print(row)

    # Sauvegarder
    output_path = NCA_DIR / "LGBM_diagnosis_classifier.sav"
    joblib.dump(clf, output_path)
    print(f"\n  Sauvegarde : {output_path}")


# ========== MAIN ==========

def main():
    print("=" * 60)
    print("ENTRAINEMENT DE TOUS LES MODELES")
    print(f"Donnees: {DATA_PATH}")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERREUR: Fichier non trouve: {DATA_PATH}")
        return

    df = pd.read_excel(DATA_PATH)
    print(f"Donnees chargees: {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # MODIFIE : exclure OTHER_DEM de TOUS les modeles (groupe heterogene)
    if 'dementia_dx_code' in df.columns:
        n_other = (df['dementia_dx_code'] == 'OTHER_DEM').sum()
        if n_other > 0:
            df = df[df['dementia_dx_code'] != 'OTHER_DEM'].copy()
            print(f"⚠️ {n_other} patients OTHER_DEM exclus de l'entrainement")
            print(f"Donnees apres exclusion : {df.shape[0]} lignes")

    # Entrainer tous les modeles dans l'ordre du pipeline :
    # 1. NCA regression (Linear) - pour utils.py (formulaire Django)
    train_nca_regression(df)
    # 2. NCA LGBM (Modele 1) - predit le delta NCA pour api_views.py
    train_nca_lgbm(df)
    # 3. Risk XGB - pour utils.py (formulaire Django)
    train_risk_xgb(df)
    # 4. Risk LGBM classifiers - pour les jauges de risque dans api_views.py
    train_risk_lgbm(df)
    # 5. Classifieur diagnostic (Modele 2) - doit etre entraine APRES le Modele 1
    #    car il utilise le Modele 1 pour calculer delta_pred
    train_diagnosis_classifier(df)

    # Verification finale : lister tous les .sav generes
    print("\n" + "=" * 60)
    print("VERIFICATION FINALE")
    print("=" * 60)
    for sav_file in sorted(MODELS_DIR.rglob("*.sav")):
        rel = sav_file.relative_to(MODELS_DIR)
        size_kb = sav_file.stat().st_size / 1024
        print(f"  {rel} ({size_kb:.1f} KB)")

    print("\nTermine avec succes!")


if __name__ == '__main__':
    main()
