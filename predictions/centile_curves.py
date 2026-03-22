"""
centile_curves_LMS.py
Calcul des courbes de centiles avec la méthode LMS (Box-Cox transformation)
Méthode utilisée par l'OMS/CDC pour les courbes de croissance

Formule LMS:
- L = skewness (asymétrie)
- M = median (médiane)
- S = coefficient of variation

Z-score: Z = ((X/M)^L - 1) / (L*S)  si L ≠ 0
         Z = ln(X/M) / S            si L = 0

Centile = Φ(Z) × 100  où Φ est la fonction de répartition normale
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def fit_lms_at_age(data, age, value_col='delta_NCA', window=3):
    """
    Ajuste les paramètres L, M, S pour un âge donné
    
    Args:
        data: DataFrame avec colonnes 'age' et value_col
        age: Âge cible
        value_col: Nom de la colonne de valeurs
        window: Fenêtre ±N ans pour avoir assez de données
    
    Returns:
        dict avec 'L', 'M', 'S', 'n_samples'
    """
    # Filtrer données autour de l'âge
    age_min = age - window
    age_max = age + window
    subset = data[(data['age'] >= age_min) & (data['age'] <= age_max)].copy()
    
    if len(subset) < 20:
        return None
    
    values = subset[value_col].values
    
    # M = médiane
    M = np.median(values)
    
    # S = coefficient de variation (approximation initiale)
    S = np.std(values) / (np.abs(M) + 1e-6)  # Éviter division par zéro
    
    # L = skewness (asymétrie)
    # On utilise une optimisation pour trouver le meilleur L
    
    def negative_log_likelihood(L_val):
        """
        Fonction de vraisemblance négative pour trouver L optimal
        """
        L = L_val[0]
        
        try:
            if abs(L) < 1e-6:  # L ≈ 0
                z_scores = np.log(values / M) / S
            else:
                z_scores = ((values / M) ** L - 1) / (L * S)
            
            # Log-vraisemblance normale
            ll = -np.sum(stats.norm.logpdf(z_scores))
            
            return ll if np.isfinite(ll) else 1e10
        except:
            return 1e10
    
    # Optimisation de L (départ à 1, proche de la normale)
    result = minimize(
        negative_log_likelihood,
        x0=[1.0],
        bounds=[(-3, 3)],  # L typiquement entre -3 et 3
        method='L-BFGS-B'
    )
    
    L_opt = result.x[0] if result.success else 1.0
    
    # Recalculer S avec L optimal
    if abs(L_opt) < 1e-6:
        z_scores = np.log(values / M) / S
    else:
        z_scores = ((values / M) ** L_opt - 1) / (L_opt * S)
    
    S_opt = np.std(z_scores)
    
    return {
        'L': L_opt,
        'M': M,
        'S': S_opt,
        'n_samples': len(subset)
    }


def lms_to_centile(z_score):
    """
    Convertit un z-score en centile (0-100)
    """
    return stats.norm.cdf(z_score) * 100


def centile_to_z(centile):
    """
    Convertit un centile (0-100) en z-score
    """
    return stats.norm.ppf(centile / 100)


def calculate_value_from_lms(L, M, S, centile):
    """
    Calcule la valeur X correspondant à un centile donné
    
    Args:
        L, M, S: Paramètres LMS
        centile: Centile désiré (0-100)
    
    Returns:
        Valeur X correspondant au centile
    """
    z = centile_to_z(centile)
    
    if abs(L) < 1e-6:  # L ≈ 0
        X = M * np.exp(S * z)
    else:
        X = M * (1 + L * S * z) ** (1 / L)
    
    return X


def calculate_centile_curves_lms(
    df,
    sex_value,
    age_col='age',
    value_col='delta_NCA',
    age_min=50,
    age_max=90,
    age_step=1,
    window=3,
    centiles=[3, 10, 25, 50, 75, 90, 97]
):
    """
    Calcule les courbes de centiles avec la méthode LMS
    
    Args:
        df: DataFrame avec données
        sex_value: 0 (femme) ou 1 (homme)
        age_col: nom colonne âge
        value_col: nom colonne valeur (delta_NCA)
        age_min/max: plage d'âges
        age_step: pas d'âge
        window: fenêtre pour fit LMS
        centiles: liste des percentiles à calculer
    
    Returns:
        dict avec 'curves', 'lms_parameters', 'raw_data'
    """
    # Filtrer par sexe
    data = df[df['sex'] == sex_value].copy()
    
    if len(data) < 50:
        return {
            'curves': [],
            'lms_parameters': [],
            'raw_data': [],
            'error': 'Pas assez de données'
        }
    
    ages = np.arange(age_min, age_max + 1, age_step)
    
    # 1. Fitter L, M, S pour chaque âge
    lms_params = []
    
    for age in ages:
        params = fit_lms_at_age(data, age, value_col=value_col, window=window)
        
        if params is not None:
            lms_params.append({
                'age': age,
                **params
            })
    
    if len(lms_params) < 5:
        return {
            'curves': [],
            'lms_parameters': [],
            'raw_data': [],
            'error': 'Pas assez de points LMS ajustés'
        }
    
    # 2. Lisser les paramètres L, M, S avec interpolation
    lms_df = pd.DataFrame(lms_params)
    
    # Interpolation cubique pour lisser
    f_L = interp1d(lms_df['age'], lms_df['L'], kind='cubic', fill_value='extrapolate')
    f_M = interp1d(lms_df['age'], lms_df['M'], kind='cubic', fill_value='extrapolate')
    f_S = interp1d(lms_df['age'], lms_df['S'], kind='cubic', fill_value='extrapolate')
    
    # 3. Générer les courbes de centiles
    curves = []
    
    for age in ages:
        L = float(f_L(age))
        M = float(f_M(age))
        S = float(f_S(age))
        
        # Calculer la valeur pour chaque centile
        centile_values = {}
        for c in centiles:
            value = calculate_value_from_lms(L, M, S, c)
            key = f'p{int(c)}'  # p3, p10, p25, etc.
            centile_values[key] = float(value)
        
        curves.append({
            'age': age,
            **centile_values
        })
    
    # 4. Échantillonner des points bruts pour visualisation
    raw_sample = data.sample(n=min(500, len(data)), random_state=42)
    raw_data = [
        {
            'age': float(row[age_col]),
            'value': float(row[value_col])
        }
        for _, row in raw_sample.iterrows()
    ]
    
    return {
        'curves': curves,
        'lms_parameters': lms_params,
        'raw_data': raw_data
    }


def calculate_patient_centile_lms(age, value, lms_parameters):
    """
    Calcule le centile et z-score d'un patient avec la méthode LMS
    
    Args:
        age: Âge du patient
        value: Valeur mesurée (delta_NCA)
        lms_parameters: Liste des paramètres LMS par âge
    
    Returns:
        dict avec 'centile', 'z_score', 'interpretation'
    """
    if not lms_parameters:
        return None
    
    # Trouver les paramètres LMS pour cet âge (interpolation)
    lms_df = pd.DataFrame(lms_parameters)
    
    if age < lms_df['age'].min() or age > lms_df['age'].max():
        # Extrapolation
        nearest = lms_df.iloc[(lms_df['age'] - age).abs().argsort()[:1]]
        L = float(nearest['L'].values[0])
        M = float(nearest['M'].values[0])
        S = float(nearest['S'].values[0])
    else:
        # Interpolation
        f_L = interp1d(lms_df['age'], lms_df['L'], kind='linear')
        f_M = interp1d(lms_df['age'], lms_df['M'], kind='linear')
        f_S = interp1d(lms_df['age'], lms_df['S'], kind='linear')
        
        L = float(f_L(age))
        M = float(f_M(age))
        S = float(f_S(age))
    
    # Calculer z-score avec formule LMS
    if abs(L) < 1e-6:  # L ≈ 0
        z_score = np.log(value / M) / S
    else:
        z_score = ((value / M) ** L - 1) / (L * S)
    
    # Convertir en centile
    centile = lms_to_centile(z_score)
    
    # Interprétation
    if centile < 2:
        interpretation = "Très inférieur à la norme"
    elif centile < 10:
        interpretation = "Inférieur à la norme"
    elif centile < 25:
        interpretation = "Légèrement inférieur à la moyenne"
    elif centile < 75:
        interpretation = "Dans la norme"
    elif centile < 90:
        interpretation = "Légèrement supérieur à la moyenne"
    elif centile < 98:
        interpretation = "Supérieur à la norme"
    else:
        interpretation = "Très supérieur à la norme"
    
    return {
        'centile': float(centile),
        'z_score': float(z_score),
        'interpretation': interpretation,
        'lms_parameters': {
            'L': float(L),
            'M': float(M),
            'S': float(S)
        }
    }


# ============== EXEMPLE D'UTILISATION ==============

if __name__ == "__main__":
    # Charger données
    df = pd.read_excel("Data_NCA_exposome.xlsx")
    
    # Calculer courbes pour hommes
    result_male = calculate_centile_curves_lms(
        df=df,
        sex_value=1,
        age_col='age',
        value_col='delta_NCA',
        centiles=[3, 10, 25, 50, 75, 90, 97]
    )
    
    print("✅ Courbes LMS calculées")
    print(f"Nombre de points: {len(result_male['curves'])}")
    print(f"Paramètres LMS: {len(result_male['lms_parameters'])} âges")
    
    # Exemple patient
    patient_result = calculate_patient_centile_lms(
        age=65,
        value=5.2,
        lms_parameters=result_male['lms_parameters']
    )
    
    print(f"\n📊 Patient 65 ans, delta_NCA=5.2")
    print(f"Centile: {patient_result['centile']:.1f}e")
    print(f"Z-score: {patient_result['z_score']:.2f}")
    print(f"Interprétation: {patient_result['interpretation']}")
    print(f"Paramètres LMS: L={patient_result['lms_parameters']['L']:.3f}, M={patient_result['lms_parameters']['M']:.3f}, S={patient_result['lms_parameters']['S']:.3f}")