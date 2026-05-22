"""
generate_augmented_data.py - Augmentation de donnees par translation de structure
=================================================================================
Genere 250 profils synthetiques (55-65 ans) pour le depistage precoce :
  - 100 CON (controles sains)
  - 100 SCD (declin subjectif)
  - 50 MCI (trouble cognitif leger)

Methode : Translation de la structure de covariance des 60-65 ans
avec ajustements des priors pour les 55-60 ans.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ========== CONFIG ==========

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Example_database_withoutrois1.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "Example_database_augmented.xlsx"

np.random.seed(42)

# Features continues et binaires utilisees pour la generation
CONTINUOUS_FEATURES = [
    'age', 'education', 'moca', 'fluency_score',
    'nb_language', 'education_group',
    'neurocog_age_flu_weight', 'delta_neurocogage_flu_weight',
]

BINARY_FEATURES = [
    'sex', 'language', 'handedness', 'hearing',
    'hta', 'diab_type2', 'obesity', 'stroke', 'depression', 'anxiety',
    'smoking', 'alcohol', 'poly_pharm5', 'physical_activity', 'social_life',
    'cognitive_activities', 'nutrition_score', 'sleep_deprivation',
    'retired', 'living_alone', 'income',
    'hist_demence_fam', 'hist_demence_parent', 'tbi', 'chol_total',
]

# Lots a generer
LOTS = [
    {'dx': 'CON', 'n': 100},
    {'dx': 'SCD', 'n': 100},
    {'dx': 'MCI', 'n': 50},
]

# ========== AJUSTEMENTS DES PRIORS POUR 55-65 ANS ==========

# Facteurs d'ajustement pour les 55-60 ans par rapport aux 60-65 ans
# < 1.0 = reduire, > 1.0 = augmenter
PRIOR_ADJUSTMENTS = {
    # Age : translater vers 55-65
    'age_shift': -5.0,  # decaler de 5 ans vers le bas

    # Cognition : meilleure chez les plus jeunes
    # A 55-65 ans, les CON doivent etre quasi parfaits (MoCA 29-30)
    # Les MCI jeunes sont des cas SUBTILS (MoCA 22-26), pas severes
    'moca_shift': {
        'CON': +2.0,   # CON 55-65 : MoCA eleve (viser 29-30)
        'SCD': +1.5,   # SCD : quasi normal (viser 28-29)
        'MCI': +2.0,   # MCI jeune : deficit subtil, pas severe (viser 24-26)
    },
    'fluency_shift': {
        'CON': +2.0,   # Fluence plus elevee chez les jeunes sains
        'SCD': +1.5,
        'MCI': +1.0,   # MCI jeune : meilleure fluence que MCI age
    },

    # Risques medicaux : reduits chez les plus jeunes
    'binary_adjustments': {
        'hta': 0.6,          # 40% de reduction
        'obesity': 0.8,      # 20% de reduction
        'stroke': 0.3,       # 70% de reduction (tres rare a 55 ans)
        'poly_pharm5': 0.5,  # 50% de reduction
        'diab_type2': 0.7,   # 30% de reduction
        'retired': 0.0,      # 100% actifs a 55-60 ans
        'depression': 0.8,   # leger reduction
        'anxiety': 0.9,      # tres leger reduction
    },
}


# ========== FONCTIONS ==========

def extract_structure(df_source, dx, features_cont, features_bin):
    """Extrait la moyenne, covariance et proportions d'un sous-groupe diagnostique"""
    sub = df_source[df_source['dementia_dx_code'] == dx].copy()

    if len(sub) < 5:
        raise ValueError(f"Pas assez de donnees pour {dx}: n={len(sub)}")

    # Features continues : moyenne et covariance
    avail_cont = [f for f in features_cont if f in sub.columns]
    sub_cont = sub[avail_cont].dropna()
    mean_cont = sub_cont.mean().values
    cov_cont = sub_cont.cov().values

    # Regulariser la matrice de covariance (eviter les problemes numeriques)
    cov_cont = cov_cont + np.eye(len(avail_cont)) * 1e-6

    # Features binaires : proportions
    bin_probs = {}
    for f in features_bin:
        if f in sub.columns:
            bin_probs[f] = sub[f].dropna().mean()

    return {
        'mean': mean_cont,
        'cov': cov_cont,
        'features_cont': avail_cont,
        'bin_probs': bin_probs,
        'n_source': len(sub),
    }


def apply_priors(structure, dx):
    """Applique les ajustements de priors pour les 55-65 ans"""
    adjusted = {
        'mean': structure['mean'].copy(),
        'cov': structure['cov'].copy(),
        'features_cont': structure['features_cont'].copy(),
        'bin_probs': structure['bin_probs'].copy(),
    }

    feat_list = adjusted['features_cont']

    # Translater l'age
    if 'age' in feat_list:
        idx = feat_list.index('age')
        adjusted['mean'][idx] += PRIOR_ADJUSTMENTS['age_shift']

    # Ajuster le MoCA
    if 'moca' in feat_list and dx in PRIOR_ADJUSTMENTS['moca_shift']:
        idx = feat_list.index('moca')
        adjusted['mean'][idx] += PRIOR_ADJUSTMENTS['moca_shift'][dx]

    # Ajuster la fluence
    if 'fluency_score' in feat_list and dx in PRIOR_ADJUSTMENTS['fluency_shift']:
        idx = feat_list.index('fluency_score')
        adjusted['mean'][idx] += PRIOR_ADJUSTMENTS['fluency_shift'][dx]

    # Translater le NCA coheremment avec l'age
    if 'neurocog_age_flu_weight' in feat_list:
        idx = feat_list.index('neurocog_age_flu_weight')
        adjusted['mean'][idx] += PRIOR_ADJUSTMENTS['age_shift']
    if 'delta_neurocogage_flu_weight' in feat_list:
        # Le delta ne change pas (meme ecart relatif)
        pass

    # Ajuster les probabilites binaires
    for var, factor in PRIOR_ADJUSTMENTS['binary_adjustments'].items():
        if var in adjusted['bin_probs']:
            adjusted['bin_probs'][var] = min(1.0, max(0.0, adjusted['bin_probs'][var] * factor))

    return adjusted


def generate_samples(adjusted_structure, n, dx, max_id):
    """Genere n echantillons synthetiques a partir de la structure ajustee"""
    feat_list = adjusted_structure['features_cont']

    # Generer les features continues par distribution multivariee
    samples_cont = np.random.multivariate_normal(
        adjusted_structure['mean'],
        adjusted_structure['cov'],
        size=n
    )

    df_gen = pd.DataFrame(samples_cont, columns=feat_list)

    # Clipper les valeurs pour rester realiste
    # Les seuils de MoCA sont adaptes par diagnostic pour les 55-65 ans :
    #   CON : 27-30 (quasi parfait a cet age)
    #   SCD : 26-30 (plainte subjective mais scores normaux)
    #   MCI : 22-28 (deficit SUBTIL, pas severe -- c'est le depistage precoce)
    if 'age' in df_gen.columns:
        df_gen['age'] = df_gen['age'].clip(55, 65).round(1)
    if 'moca' in df_gen.columns:
        if dx == 'CON':
            df_gen['moca'] = df_gen['moca'].clip(27, 30).round(0)
        elif dx == 'SCD':
            df_gen['moca'] = df_gen['moca'].clip(26, 30).round(0)
        elif dx == 'MCI':
            df_gen['moca'] = df_gen['moca'].clip(22, 28).round(0)
        else:
            df_gen['moca'] = df_gen['moca'].clip(9, 30).round(0)
    if 'fluency_score' in df_gen.columns:
        df_gen['fluency_score'] = df_gen['fluency_score'].clip(5, 50).round(0)
    if 'education' in df_gen.columns:
        df_gen['education'] = df_gen['education'].clip(3, 30).round(0)
    if 'education_group' in df_gen.columns:
        df_gen['education_group'] = df_gen['education_group'].clip(0, 4).round(0)
    if 'nb_language' in df_gen.columns:
        df_gen['nb_language'] = df_gen['nb_language'].clip(1, 6).round(0)

    # Generer les features binaires
    for var, prob in adjusted_structure['bin_probs'].items():
        df_gen[var] = np.random.binomial(1, prob, size=n).astype(float)

    # Ajouter les colonnes de diagnostic
    df_gen['dementia_dx_code'] = dx

    # risk_dementia coherent avec le diagnostic
    risk_map = {'CON': 0.0, 'SCD': 0.0, 'MCI': 0.5, 'AD': 1.0}
    df_gen['risk_dementia'] = risk_map.get(dx, 0.0)

    # risk_handicap base sur facteurs fonctionnels
    def compute_risk_handicap(row):
        score = 0
        score += row.get('stroke', 0) * 3
        score += row.get('obesity', 0) * 1.5
        score += row.get('hta', 0) * 1
        score += row.get('diab_type2', 0) * 1.5
        score += row.get('living_alone', 0) * 2
        score += (1 - row.get('social_life', 1)) * 1.5
        score += (1 - row.get('physical_activity', 1)) * 2
        score += row.get('depression', 0) * 2
        score += row.get('poly_pharm5', 0) * 1.5
        if score >= 6: return 1.0
        elif score >= 3: return 0.5
        else: return 0.0

    df_gen['risk_handicap'] = df_gen.apply(compute_risk_handicap, axis=1)
    df_gen['dailylife_dependancy'] = df_gen['risk_handicap'].map({0.0: 0, 0.5: 1, 1.0: 2})

    # Calculer delta si NCA present
    if 'neurocog_age_flu_weight' in df_gen.columns and 'age' in df_gen.columns:
        df_gen['delta_neurocogage_flu_weight'] = df_gen['neurocog_age_flu_weight'] - df_gen['age']

    # Identifiants synthetiques
    df_gen['Identifiers'] = [max_id + i + 1 for i in range(n)]

    # Colonnes supplementaires avec valeurs par defaut
    df_gen['hachinski'] = 0.0
    df_gen['depression_score'] = np.where(df_gen['depression'] == 1, np.random.randint(10, 20, n), np.random.randint(0, 5, n)).astype(float)
    df_gen['anxiety_score'] = np.where(df_gen['anxiety'] == 1, np.random.randint(8, 15, n), np.random.randint(0, 5, n)).astype(float)
    df_gen['poly_pharm10'] = 0.0
    df_gen['intrinsic_capacity'] = np.where(df_gen['risk_handicap'] == 0.0, 1.0, 0.0)

    # brain_age et neurocog_age_memory (estimation)
    if 'neurocog_age_flu_weight' in df_gen.columns:
        df_gen['brain_age'] = df_gen['neurocog_age_flu_weight'] + np.random.normal(0, 2, n)
        df_gen['delta_brain_age'] = df_gen['brain_age'] - df_gen['age']
        df_gen['neurocog_age_memory'] = df_gen['neurocog_age_flu_weight'] + np.random.normal(0, 3, n)
        df_gen['delta_neurocog_age_memory'] = df_gen['neurocog_age_memory'] - df_gen['age']

    # RAVLT et logic : scores cognitifs ajustes pour les 55-65 ans
    # Les jeunes ont globalement de meilleurs scores que les ages
    # Les MCI jeunes ont des deficits SUBTILS (pas severes)
    if dx == 'CON':
        df_gen['ravlt_imm'] = np.random.normal(50, 6, n).clip(35, 70).round(0)
        df_gen['ravlt_delay'] = np.random.normal(11, 2, n).clip(7, 15).round(0)
        df_gen['logic_imm'] = np.random.normal(15, 3, n).clip(8, 24).round(0)
        df_gen['logic_delay'] = np.random.normal(14, 3, n).clip(8, 24).round(0)
    elif dx == 'SCD':
        df_gen['ravlt_imm'] = np.random.normal(46, 7, n).clip(30, 70).round(0)
        df_gen['ravlt_delay'] = np.random.normal(10, 2, n).clip(5, 15).round(0)
        df_gen['logic_imm'] = np.random.normal(13, 3, n).clip(6, 24).round(0)
        df_gen['logic_delay'] = np.random.normal(12, 3, n).clip(5, 24).round(0)
    else:  # MCI jeune : deficit subtil, pas severe
        df_gen['ravlt_imm'] = np.random.normal(38, 7, n).clip(25, 55).round(0)
        df_gen['ravlt_delay'] = np.random.normal(7, 2, n).clip(3, 13).round(0)
        df_gen['logic_imm'] = np.random.normal(10, 3, n).clip(5, 20).round(0)
        df_gen['logic_delay'] = np.random.normal(8, 3, n).clip(3, 18).round(0)

    return df_gen


def validate_generated(df_gen, df_source, dx):
    """Valide la coherence des donnees generees"""
    gen = df_gen[df_gen['dementia_dx_code'] == dx]
    src = df_source[(df_source['dementia_dx_code'] == dx) & (df_source['age'] >= 60) & (df_source['age'] < 65)]

    print(f"\n  [{dx}] Generes: {len(gen)} | Source 60-65: {len(src)}")
    for col in ['age', 'moca', 'fluency_score', 'education']:
        if col in gen.columns and col in src.columns:
            g_mean = gen[col].mean()
            s_mean = src[col].dropna().mean()
            print(f"    {col:<18} Gen: {g_mean:>6.1f}  |  Src60-65: {s_mean:>6.1f}  |  Diff: {g_mean - s_mean:>+.1f}")

    for col in ['hta', 'obesity', 'stroke', 'poly_pharm5', 'retired']:
        if col in gen.columns and col in src.columns:
            g_pct = gen[col].mean() * 100
            s_pct = src[col].dropna().mean() * 100
            print(f"    {col:<18} Gen: {g_pct:>5.0f}%  |  Src60-65: {s_pct:>5.0f}%  |  Diff: {g_pct - s_pct:>+.0f}%")


# ========== MAIN ==========

def main():
    print("=" * 70)
    print("  AUGMENTATION DE DONNEES PAR TRANSLATION DE STRUCTURE")
    print(f"  Source : {DATA_PATH.name}")
    print("=" * 70)

    df = pd.read_excel(DATA_PATH)
    print(f"\n  Donnees originales : {len(df)} patients")

    # Filtrer les 60-65 ans comme base de translation
    df60 = df[(df['age'] >= 60) & (df['age'] < 65)].copy()
    print(f"  Base de translation (60-65 ans) : {len(df60)} patients")

    max_id = 9000000  # Identifiants synthetiques a partir de 9M
    all_generated = []

    for lot in LOTS:
        dx = lot['dx']
        n = lot['n']

        print(f"\n{'─'*70}")
        print(f"  Generation : {n} x {dx}")
        print(f"{'─'*70}")

        # 1. Extraire la structure du sous-groupe 60-65 ans
        structure = extract_structure(df60, dx, CONTINUOUS_FEATURES, BINARY_FEATURES)
        print(f"  Structure extraite de {structure['n_source']} patients {dx} (60-65 ans)")

        # 2. Appliquer les priors pour 55-65 ans
        adjusted = apply_priors(structure, dx)
        print(f"  Priors appliques (age shift: {PRIOR_ADJUSTMENTS['age_shift']}, retired: 0%)")

        # 3. Generer les echantillons
        df_gen = generate_samples(adjusted, n, dx, max_id)
        max_id += n

        all_generated.append(df_gen)
        print(f"  {n} profils generes")

    # Combiner
    df_augmented = pd.concat(all_generated, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"  TOTAL GENERE : {len(df_augmented)} profils synthetiques")
    print(f"{'='*70}")

    # Validation
    print(f"\n  VALIDATION (comparaison generes vs source 60-65 ans)")
    for lot in LOTS:
        validate_generated(df_augmented, df, lot['dx'])

    # Fusionner avec les donnees originales
    # Aligner les colonnes
    for col in df.columns:
        if col not in df_augmented.columns:
            df_augmented[col] = np.nan

    df_augmented = df_augmented[df.columns]
    df_final = pd.concat([df, df_augmented], ignore_index=True)

    print(f"\n  Dataset final : {len(df_final)} patients ({len(df)} originaux + {len(df_augmented)} synthetiques)")
    print(f"  Distribution diagnostique finale :")
    print(df_final['dementia_dx_code'].value_counts().to_string())

    # Sauvegarder
    df_final.to_excel(OUTPUT_PATH, index=False)
    print(f"\n  Sauvegarde : {OUTPUT_PATH}")
    print(f"\n{'='*70}")
    print(f"  AUGMENTATION TERMINEE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
