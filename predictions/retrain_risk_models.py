"""
Script de réentraînement des modèles de risque (Démence et Handicap)
VERSION CORRIGÉE :
  - LGBMClassifier (ordinal 3 classes : 0.0 / 0.5 / 1.0)
  - is_unbalance=True pour risk_handicap (78% de zéros)
  - predict_proba → score de risque pondéré entre 0 et 100
  - Sauvegarde joblib
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ========== CONFIGURATION ==========

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Example_database_withoutrois1.xlsx"

OUTPUT_DIR_DEMENTIA = BASE_DIR / "risk_dementia"
OUTPUT_DIR_HANDICAP = BASE_DIR / "risk_handicap"

OUTPUT_DIR_DEMENTIA.mkdir(exist_ok=True)
OUTPUT_DIR_HANDICAP.mkdir(exist_ok=True)

# ========== FEATURES ==========

FEATURES_ALL_PLUS_PLUS = [
    'age', 'sex', 'education', 'language', 'fluency_score', 'moca',
    'handedness', 'nb_language', 'hearing',
    'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income', 'retired',
    'stroke', 'tbi', 'hta', 'diab_type2', 'chol_total',
    'obesity', 'depression', 'anxiety',
    'smoking', 'alcohol', 'poly_pharm5', 'physical_activity', 'social_life',
    'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
]

print(f"✅ {len(FEATURES_ALL_PLUS_PLUS)} features")

# ========== CHARGEMENT ==========

try:
    from lightgbm import LGBMClassifier
    import joblib
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as e:
    print(f"❌ Import manquant : {e}")
    print("   Installez : pip install lightgbm scikit-learn joblib")
    raise

print(f"\n📂 Chargement : {DATA_PATH}")
df = pd.read_excel(DATA_PATH)
print(f"✅ {len(df)} lignes, {len(df.columns)} colonnes")

# ========== VÉRIFICATIONS ==========

for col in ['risk_dementia', 'risk_handicap']:
    vals = sorted(df[col].dropna().unique())
    print(f"\n{col} : {df[col].value_counts().sort_index().to_dict()}")
    print(f"  Valeurs : {vals}")
    if not all(v in [0.0, 0.5, 1.0] for v in vals):
        print(f"  ⚠️ Valeurs inattendues — vérifier les données")

# ========== FONCTION D'ENTRAÎNEMENT ==========

def train_and_save(target_col: str, output_path: Path, is_unbalanced: bool = False):
    """
    Entraîne un LGBMClassifier sur une cible ordinale {0.0, 0.5, 1.0}.

    Le modèle sauvegardé expose predict_proba(X) → shape (n, 3).
    Le score de risque en % est calculé côté API par :
        score = proba[0]*0 + proba[1]*50 + proba[2]*100
    """
    print(f"\n{'='*60}")
    print(f"MODÈLE : {target_col.upper()}")
    print(f"{'='*60}")

    # Préparer X, y
    df_sub = df[FEATURES_ALL_PLUS_PLUS + [target_col]].dropna(subset=[target_col])
    X = df_sub[FEATURES_ALL_PLUS_PLUS]
    y = df_sub[target_col]   # valeurs {0.0, 0.5, 1.0}

    print(f"✅ {len(df_sub)} lignes utilisées")
    print(f"   Distribution :")
    for v, c in y.value_counts().sort_index().items():
        print(f"     {v} → {c} ({c/len(y)*100:.1f}%)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✅ Train : {len(X_train)}  |  Test : {len(X_test)}")

    # Modèle
    #
    # IMPORTANT : is_unbalance=True est nécessaire pour risk_handicap
    # car 78% des patients sont à 0.0. Sans ça, le modèle prédit
    # presque toujours 0 → score de risque ~0% pour tout le monde.
    #
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        is_unbalance=is_unbalanced,   # True pour risk_handicap
        random_state=42,
        verbose=-1,
    )

    # Encoder y en entiers 0/1/2 pour LGBMClassifier
    class_map = {0.0: 0, 0.5: 1, 1.0: 2}
    y_train_enc = y_train.map(class_map)
    y_test_enc  = y_test.map(class_map)

    model.fit(X_train, y_train_enc)
    print(f"✅ Modèle entraîné")

    # Évaluation
    y_pred = model.predict(X_test)
    print(f"\n📊 Rapport classification (test) :")
    print(classification_report(
        y_test_enc, y_pred,
        target_names=["Faible (0)", "Modéré (0.5)", "Élevé (1)"],
        zero_division=0
    ))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test_enc, y_pred))

    # Vérifier que predict_proba fonctionne et donne 3 colonnes
    proba_sample = model.predict_proba(X_test.iloc[:3])
    print(f"\n✅ predict_proba shape : {proba_sample.shape}")
    print(f"   Exemple (3 premiers patients) :")
    for i, (p0, p1, p2) in enumerate(proba_sample):
        score = p1 * 50 + p2 * 100
        print(f"   Patient {i+1} : P(0)={p0:.2f}  P(0.5)={p1:.2f}  P(1)={p2:.2f}  → Score={score:.1f}%")

    # Sauvegarder le modèle ET le mapping des classes
    save_obj = {
        'model': model,
        'class_map': class_map,           # {0.0: 0, 0.5: 1, 1.0: 2}
        'class_map_inv': {0: 0.0, 1: 0.5, 2: 1.0},
        'features': FEATURES_ALL_PLUS_PLUS,
        'target': target_col,
        'is_unbalance': is_unbalanced,
    }
    joblib.dump(save_obj, output_path)
    print(f"\n💾 Sauvegardé : {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Test rechargement
    loaded = joblib.load(output_path)
    assert loaded['target'] == target_col
    test_pred = loaded['model'].predict_proba(X_test.iloc[:1])
    print(f"✅ Rechargement OK — proba : {test_pred[0]}")

    return model


# ========== ENTRAÎNEMENTS ==========

# risk_dementia : distribution équilibrée → pas besoin de is_unbalance
train_and_save(
    target_col='risk_dementia',
    output_path=OUTPUT_DIR_DEMENTIA / "LGBM_reg_all_plus_plus.sav",
    is_unbalanced=False,
)

# risk_handicap : 78% de zéros → is_unbalance=True OBLIGATOIRE
train_and_save(
    target_col='risk_handicap',
    output_path=OUTPUT_DIR_HANDICAP / "LGBM_reg_all_plus_plus.sav",
    is_unbalanced=True,
)

# ========== RÉSUMÉ ==========

print(f"\n{'='*60}")
print("RÉSUMÉ")
print(f"{'='*60}")
print(f"✅ risk_dementia → {OUTPUT_DIR_DEMENTIA / 'LGBM_reg_all_plus_plus.sav'}")
print(f"✅ risk_handicap → {OUTPUT_DIR_HANDICAP / 'LGBM_reg_all_plus_plus.sav'}")
print(f"""
⚠️  IMPORTANT — Mettre à jour api_views.py :

Remplacer la logique de prédiction par :

    obj = joblib.load(model_path)
    model = obj['model']
    proba = model.predict_proba(X)[0]   # shape (3,) : P(0), P(0.5), P(1)
    risk_score = proba[1] * 50 + proba[2] * 100   # score en %
""")