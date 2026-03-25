"""
Script de réentraînement des modèles de risque (Démence et Handicap)
Corrige le problème de compatibilité LightGBM et utilise joblib au lieu de pickle
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib
from pathlib import Path

# ========== CONFIGURATION ==========

# Chemins (à adapter selon votre configuration)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Example_database_withoutrois1.xlsx"

OUTPUT_DIR_DEMENTIA = Path("risk_dementia")
OUTPUT_DIR_HANDICAP = Path("risk_handicap")

# Créer les dossiers de sortie
OUTPUT_DIR_DEMENTIA.mkdir(exist_ok=True)
OUTPUT_DIR_HANDICAP.mkdir(exist_ok=True)


# ========== FEATURES all_plus_plus ==========

# Features pour le modèle all_plus_plus (sans neurocog_age_flu_weight, sans risk_dementia)
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

print(f"✅ {len(FEATURES_ALL_PLUS_PLUS)} features all_plus_plus")


# ========== CHARGEMENT DES DONNÉES ==========

print(f"\n📂 Chargement des données : {DATA_PATH}")
df = pd.read_excel(DATA_PATH)
print(f"✅ {len(df)} lignes chargées")
print(f"✅ Colonnes : {list(df.columns)[:10]}...")


# ========== VÉRIFICATION DES COLONNES ==========

# Vérifier que les colonnes nécessaires existent
required_columns_dementia = FEATURES_ALL_PLUS_PLUS + ['risk_dementia']
required_columns_handicap = FEATURES_ALL_PLUS_PLUS + ['risk_handicap']

missing_dementia = [col for col in required_columns_dementia if col not in df.columns]
missing_handicap = [col for col in required_columns_handicap if col not in df.columns]

if missing_dementia:
    print(f"⚠️ Colonnes manquantes pour risk_dementia : {missing_dementia}")
    
if missing_handicap:
    print(f"⚠️ Colonnes manquantes pour risk_handicap : {missing_handicap}")


# ========== PRÉPARATION DES DONNÉES - DÉMENCE ==========

print("\n" + "="*60)
print("MODÈLE DE RISQUE DÉMENCE")
print("="*60)

# Sélectionner les colonnes
df_dementia = df[required_columns_dementia].copy()

# Supprimer les lignes avec NaN dans la variable cible
df_dementia = df_dementia.dropna(subset=['risk_dementia'])

print(f"✅ {len(df_dementia)} lignes après suppression NaN dans risk_dementia")

# Séparer X et y
X_dementia = df_dementia[FEATURES_ALL_PLUS_PLUS]
y_dementia = df_dementia['risk_dementia']

print(f"✅ X shape : {X_dementia.shape}")
print(f"✅ y shape : {y_dementia.shape}")
print(f"✅ y range : [{y_dementia.min():.4f}, {y_dementia.max():.4f}]")
print(f"✅ y mean : {y_dementia.mean():.4f}")


# ========== ENTRAÎNEMENT MODÈLE DÉMENCE ==========

print(f"\n🔮 Entraînement du modèle LightGBM pour risk_dementia...")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_dementia, y_dementia, 
    test_size=0.2, 
    random_state=42
)

print(f"✅ Train : {len(X_train)} lignes")
print(f"✅ Test : {len(X_test)} lignes")

# Entraîner le modèle
model_dementia = LGBMRegressor(
    objective='regression',  # IMPORTANT : regression, pas binary
    metric='rmse',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=-1
)

model_dementia.fit(X_train, y_train)

print(f"✅ Modèle entraîné")

# Évaluer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred_train = model_dementia.predict(X_train)
y_pred_test = model_dementia.predict(X_test)

print(f"\n📊 Métriques train :")
print(f"   RMSE : {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"   MAE : {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"   R² : {r2_score(y_train, y_pred_train):.4f}")

print(f"\n📊 Métriques test :")
print(f"   RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"   MAE : {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"   R² : {r2_score(y_test, y_pred_test):.4f}")


# ========== SAUVEGARDE MODÈLE DÉMENCE ==========

output_path_dementia = OUTPUT_DIR_DEMENTIA / "LGBM_reg_all_plus_plus.sav"

print(f"\n💾 Sauvegarde du modèle : {output_path_dementia}")
joblib.dump(model_dementia, output_path_dementia)
print(f"✅ Modèle sauvegardé ({output_path_dementia.stat().st_size / 1024:.1f} KB)")


# ========== TEST DE RECHARGEMENT DÉMENCE ==========

print(f"\n🧪 Test de rechargement du modèle...")
loaded_model_dementia = joblib.load(output_path_dementia)
print(f"✅ Modèle rechargé : {type(loaded_model_dementia).__name__}")

# Tester une prédiction
test_sample = X_test.iloc[0:1]
pred = loaded_model_dementia.predict(test_sample)
print(f"✅ Prédiction test : {pred[0]:.4f}")
print(f"✅ Vraie valeur : {y_test.iloc[0]:.4f}")


# ========== PRÉPARATION DES DONNÉES - HANDICAP ==========

print("\n" + "="*60)
print("MODÈLE DE RISQUE HANDICAP")
print("="*60)

# Sélectionner les colonnes
df_handicap = df[required_columns_handicap].copy()

# Supprimer les lignes avec NaN dans la variable cible
df_handicap = df_handicap.dropna(subset=['risk_handicap'])

print(f"✅ {len(df_handicap)} lignes après suppression NaN dans risk_handicap")

# Séparer X et y
X_handicap = df_handicap[FEATURES_ALL_PLUS_PLUS]
y_handicap = df_handicap['risk_handicap']

print(f"✅ X shape : {X_handicap.shape}")
print(f"✅ y shape : {y_handicap.shape}")
print(f"✅ y range : [{y_handicap.min():.4f}, {y_handicap.max():.4f}]")
print(f"✅ y mean : {y_handicap.mean():.4f}")


# ========== ENTRAÎNEMENT MODÈLE HANDICAP ==========

print(f"\n🔮 Entraînement du modèle LightGBM pour risk_handicap...")

# Split train/test
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_handicap, y_handicap, 
    test_size=0.2, 
    random_state=42
)

print(f"✅ Train : {len(X_train_h)} lignes")
print(f"✅ Test : {len(X_test_h)} lignes")

# Entraîner le modèle
model_handicap = LGBMRegressor(
    objective='regression',
    metric='rmse',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=-1
)

model_handicap.fit(X_train_h, y_train_h)

print(f"✅ Modèle entraîné")

# Évaluer
y_pred_train_h = model_handicap.predict(X_train_h)
y_pred_test_h = model_handicap.predict(X_test_h)

print(f"\n📊 Métriques train :")
print(f"   RMSE : {np.sqrt(mean_squared_error(y_train_h, y_pred_train_h)):.4f}")
print(f"   MAE : {mean_absolute_error(y_train_h, y_pred_train_h):.4f}")
print(f"   R² : {r2_score(y_train_h, y_pred_train_h):.4f}")

print(f"\n📊 Métriques test :")
print(f"   RMSE : {np.sqrt(mean_squared_error(y_test_h, y_pred_test_h)):.4f}")
print(f"   MAE : {mean_absolute_error(y_test_h, y_pred_test_h):.4f}")
print(f"   R² : {r2_score(y_test_h, y_pred_test_h):.4f}")


# ========== SAUVEGARDE MODÈLE HANDICAP ==========

output_path_handicap = OUTPUT_DIR_HANDICAP / "LGBM_reg_all_plus_plus.sav"

print(f"\n💾 Sauvegarde du modèle : {output_path_handicap}")
joblib.dump(model_handicap, output_path_handicap)
print(f"✅ Modèle sauvegardé ({output_path_handicap.stat().st_size / 1024:.1f} KB)")


# ========== TEST DE RECHARGEMENT HANDICAP ==========

print(f"\n🧪 Test de rechargement du modèle...")
loaded_model_handicap = joblib.load(output_path_handicap)
print(f"✅ Modèle rechargé : {type(loaded_model_handicap).__name__}")

# Tester une prédiction
test_sample_h = X_test_h.iloc[0:1]
pred_h = loaded_model_handicap.predict(test_sample_h)
print(f"✅ Prédiction test : {pred_h[0]:.4f}")
print(f"✅ Vraie valeur : {y_test_h.iloc[0]:.4f}")


# ========== RÉSUMÉ FINAL ==========

print("\n" + "="*60)
print("RÉSUMÉ FINAL")
print("="*60)

print(f"\n✅ Modèles sauvegardés dans :")
print(f"   📁 {OUTPUT_DIR_DEMENTIA.absolute()}")
print(f"   📁 {OUTPUT_DIR_HANDICAP.absolute()}")

print(f"\n✅ Fichiers créés :")
print(f"   📄 {output_path_dementia}")
print(f"   📄 {output_path_handicap}")

print(f"\n✅ Tests de rechargement : OK")
print(f"✅ Les modèles sont prêts pour l'utilisation en production !")

print(f"\n📋 Prochaines étapes :")
print(f"   1. Copier risk_dementia/ dans medicalapp/predictions/")
print(f"   2. Copier risk_handicap/ dans medicalapp/predictions/")
print(f"   3. Redémarrer le serveur Django")
print(f"   4. Tester avec une prédiction réelle")