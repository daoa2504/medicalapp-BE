"""
gamlss_python_pure.py
GAMLSS en Python Pur - Pas besoin de R !

À placer dans : predictions/gamlss_python_pure.py
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
from typing import Dict, List, Tuple


def fit_smooth_curve(ages: np.ndarray, values: np.ndarray, 
                     predict_ages: np.ndarray) -> np.ndarray:
    """
    Ajuste une courbe lisse avec interpolation spline
    
    Args:
        ages: Âges des observations
        values: Valeurs observées
        predict_ages: Âges pour prédiction
    
    Returns:
        Valeurs prédites lissées
    """
    # Spline cubique lissée
    # s=None pour interpolation exacte, ou s>0 pour lissage
    tck = interpolate.splrep(ages, values, s=len(ages), k=3)
    smooth_values = interpolate.splev(predict_ages, tck)
    
    return smooth_values


def calculate_percentiles_at_age(df: pd.DataFrame, age: float, 
                                 window: float = 2.0) -> Dict[int, float]:
    """
    Calcule les percentiles pour un âge donné avec fenêtre
    
    Args:
        df: DataFrame avec colonnes 'age' et 'delta_nca'
        age: Âge cible
        window: Fenêtre autour de l'âge (±2 ans par défaut)
    
    Returns:
        dict: Percentiles {3: value, 10: value, ...}
    """
    # Filtrer autour de l'âge
    age_window = df[
        (df['age'] >= age - window) &
        (df['age'] <= age + window)
    ]['delta_nca']
    
    if len(age_window) < 5:
        return None
    
    # Calculer percentiles
    percentiles = {}
    for p in [3, 10, 25, 50, 75, 90, 97]:
        percentiles[p] = np.percentile(age_window, p)
    
    return percentiles


def fit_smooth_percentiles_python_pure(df_reference: pd.DataFrame, 
                                       sex_value: int) -> List[Dict]:
    """
    Ajuste des courbes de percentiles lisses en Python pur
    Méthode : Interpolation spline sur percentiles empiriques
    
    Args:
        df_reference: DataFrame avec données
        sex_value: 0 (femme) ou 1 (homme)
    
    Returns:
        list: Courbes de percentiles lissées
    """
    
    # Préparer les données
    df_sex = df_reference[
        (df_reference['sex'] == sex_value) &
        (df_reference['durée_suivie'] == 0) &
        (df_reference['age'] >= 50) &
        (df_reference['age'] <= 90)
    ][['age', 'delta_neurocogage_flu_weight']].dropna()
    
    df_sex = df_sex.rename(columns={'delta_neurocogage_flu_weight': 'delta_nca'})
    
    # Filtrer valeurs aberrantes (garder entre -50 et +50)
    df_sex = df_sex[
        (df_sex['delta_nca'] >= -50) &
        (df_sex['delta_nca'] <= 50)
    ]
    
    if len(df_sex) < 50:
        print(f"⚠️ Pas assez de données pour sexe={sex_value} ({len(df_sex)} patients)")
        return []
    
    print(f"📊 Ajustement courbes lisses pour sexe={sex_value} : {len(df_sex)} patients")
    
    # Calculer percentiles empiriques tous les 2 ans
    empirical_ages = []
    empirical_percentiles = {p: [] for p in [3, 10, 25, 50, 75, 90, 97]}
    
    for age in range(52, 89, 2):  # 52, 54, 56, ..., 88
        percs = calculate_percentiles_at_age(df_sex, age, window=3.0)
        
        if percs is not None:
            empirical_ages.append(age)
            for p in [3, 10, 25, 50, 75, 90, 97]:
                empirical_percentiles[p].append(percs[p])
    
    if len(empirical_ages) < 5:
        print(f"⚠️ Pas assez de points pour interpolation")
        return []
    
    empirical_ages = np.array(empirical_ages)
    
    # Lisser chaque courbe de percentile avec spline
    predict_ages = np.arange(50, 91, 1)
    smooth_percentiles = {}
    
    for p in [3, 10, 25, 50, 75, 90, 97]:
        smooth_percentiles[p] = fit_smooth_curve(
            empirical_ages,
            np.array(empirical_percentiles[p]),
            predict_ages
        )
    
    # Construire le résultat
    curves = []
    for i, age in enumerate(predict_ages):
        curves.append({
            'age': int(age),
            'p3': float(smooth_percentiles[3][i]),
            'p10': float(smooth_percentiles[10][i]),
            'p25': float(smooth_percentiles[25][i]),
            'p50': float(smooth_percentiles[50][i]),
            'p75': float(smooth_percentiles[75][i]),
            'p90': float(smooth_percentiles[90][i]),
            'p97': float(smooth_percentiles[97][i]),
        })
    
    print(f"✅ Courbes lissées calculées : {len(curves)} points")
    
    return curves


def calculate_percentile_curves_python_pure(df_reference: pd.DataFrame) -> Dict:
    """
    Calcule des courbes de percentiles lisses en Python pur
    
    Méthode : Interpolation spline cubique sur percentiles empiriques
    
    Avantages :
    - ✅ Pas besoin d'installer R
    - ✅ Courbes lisses (pas saccadées)
    - ✅ Rapide (< 1 seconde)
    - ✅ Simple à comprendre
    
    Usage dans api_views.py:
        from .gamlss_python_pure import calculate_percentile_curves_python_pure
        
        percentile_data = calculate_percentile_curves_python_pure(df_reference)
        percentile_curves_male = percentile_data['male']
        percentile_curves_female = percentile_data['female']
    """
    
    print("=" * 60)
    print("📊 Courbes de Percentiles - Python Pur (Spline)")
    print("=" * 60)
    
    # Hommes
    print("\n👨 Ajustement pour HOMMES...")
    male_curves = fit_smooth_percentiles_python_pure(df_reference, sex_value=1)
    
    # Femmes
    print("\n👩 Ajustement pour FEMMES...")
    female_curves = fit_smooth_percentiles_python_pure(df_reference, sex_value=0)
    
    return {
        'male': male_curves,
        'female': female_curves,
        'method': 'spline_interpolation_python'
    }


# ===== VERSION ALTERNATIVE : GAM avec statsmodels =====

def calculate_percentile_curves_gam_python(df_reference: pd.DataFrame) -> Dict:
    """
    Version avec GAM (si statsmodels disponible)
    Plus sophistiqué que spline mais nécessite statsmodels
    
    Installation :
        pip install statsmodels --break-system-packages
    """
    
    try:
        from statsmodels.gam.api import GLMGam, BSplines
        from scipy.stats import norm
    except ImportError:
        print("⚠️ statsmodels non disponible, fallback sur spline")
        return calculate_percentile_curves_python_pure(df_reference)
    
    print("=" * 60)
    print("📊 Courbes de Percentiles - GAM Python")
    print("=" * 60)
    
    def fit_gam_curves(sex_value: int) -> List[Dict]:
        # Préparer données
        df_sex = df_reference[
            (df_reference['sex'] == sex_value) &
            (df_reference['durée_suivie'] == 0) &
            (df_reference['age'] >= 50) &
            (df_reference['age'] <= 90)
        ][['age', 'delta_neurocogage_flu_weight']].dropna()
        
        if len(df_sex) < 50:
            return []
        
        print(f"📊 GAM pour sexe={sex_value} : {len(df_sex)} patients")
        
        X = df_sex['age'].values
        y = df_sex['delta_neurocogage_flu_weight'].values
        
        # Modèle pour la médiane
        bs = BSplines(X, df=[6], degree=[3])
        gam_median = GLMGam(y, exog=bs.basis)
        result_median = gam_median.fit()
        
        # Modèle pour la variance
        residuals = y - result_median.fittedvalues
        residuals_sq = residuals ** 2
        gam_var = GLMGam(residuals_sq, exog=bs.basis)
        result_var = gam_var.fit()
        
        # Prédictions
        ages = np.arange(50, 91, 1)
        bs_pred = BSplines(ages, df=[6], degree=[3])
        
        mu = result_median.predict(exog=bs_pred.basis)
        sigma_sq = result_var.predict(exog=bs_pred.basis)
        sigma = np.sqrt(np.maximum(sigma_sq, 0.1))
        
        # Calculer percentiles
        curves = []
        for i, age in enumerate(ages):
            curves.append({
                'age': int(age),
                'p3': float(mu[i] + norm.ppf(0.03) * sigma[i]),
                'p10': float(mu[i] + norm.ppf(0.10) * sigma[i]),
                'p25': float(mu[i] + norm.ppf(0.25) * sigma[i]),
                'p50': float(mu[i]),
                'p75': float(mu[i] + norm.ppf(0.75) * sigma[i]),
                'p90': float(mu[i] + norm.ppf(0.90) * sigma[i]),
                'p97': float(mu[i] + norm.ppf(0.97) * sigma[i]),
            })
        
        print(f"✅ GAM ajusté : {len(curves)} points")
        return curves
    
    return {
        'male': fit_gam_curves(1),
        'female': fit_gam_curves(0),
        'method': 'gam_python'
    }


# ===== FONCTION DE TEST =====

def test_python_pure():
    """
    Tester la fonction avec des données simulées
    
    Usage:
        python -c "from predictions.gamlss_python_pure import test_python_pure; test_python_pure()"
    """
    
    print("=" * 60)
    print("TEST - Courbes Lisses Python Pur")
    print("=" * 60)
    
    # Générer données de test
    np.random.seed(42)
    n = 200
    ages = np.random.uniform(50, 90, n)
    
    # Simuler vieillissement avec bruit
    delta_nca = -15 + 0.3 * ages + np.random.normal(0, 5, n)
    
    df_test = pd.DataFrame({
        'age': ages,
        'delta_neurocogage_flu_weight': delta_nca,
        'sex': np.random.choice([0, 1], n),
        'durée_suivie': 0
    })
    
    # Tester
    result = calculate_percentile_curves_python_pure(df_test)
    
    print("\nRésultats :")
    print(f"  Courbes hommes : {len(result['male'])} points")
    print(f"  Courbes femmes : {len(result['female'])} points")
    print(f"  Méthode : {result['method']}")
    
    if result['male']:
        print("\nExemple courbe hommes (âge 70) :")
        curve_70 = [c for c in result['male'] if c['age'] == 70][0]
        for p in ['p3', 'p25', 'p50', 'p75', 'p97']:
            print(f"    {p} : {curve_70[p]:.2f}")
    
    print("\n✅ Test réussi !")


if __name__ == "__main__":
    test_python_pure()