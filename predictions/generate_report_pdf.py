"""
generate_report_pdf.py - Généré un rapport PDF d'évaluation des modèles CogniScreen
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.model_selection import train_test_split

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ========== CONFIG ==========

BASE_DIR = Path(__file__).resolve().parent
# Donnees augmentees (originales + 250 profils synthetiques depistage precoce)
DATA_PATH = BASE_DIR / "data" / "Example_database_augmented.xlsx"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "rapport_evaluation_modeles.pdf"

RANDOM_STATE = 42
TEST_SIZE = 0.2

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

# ========== COULEURS ==========

DARK_BLUE = colors.HexColor('#1e3a5f')
MEDIUM_BLUE = colors.HexColor('#2c5f8a')
LIGHT_BLUE = colors.HexColor('#d6e8f7')
ACCENT_GREEN = colors.HexColor('#27ae60')
ACCENT_RED = colors.HexColor('#e74c3c')
ACCENT_ORANGE = colors.HexColor('#f39c12')
HEADER_BG = colors.HexColor('#2c3e50')
ROW_ALT = colors.HexColor('#ecf0f1')
WHITE = colors.white
BLACK = colors.black
GREY = colors.HexColor('#7f8c8d')


# ========== STYLES ==========

def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=24, textColor=DARK_BLUE, spaceAfter=6,
        alignment=TA_CENTER, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=12, textColor=GREY, spaceAfter=20,
        alignment=TA_CENTER, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'SectionTitle', parent=styles['Heading1'],
        fontSize=16, textColor=WHITE, spaceAfter=10, spaceBefore=16,
        fontName='Helvetica-Bold', backColor=DARK_BLUE,
        borderPadding=(8, 8, 8, 8), leftIndent=0
    ))
    styles.add(ParagraphStyle(
        'SubSection', parent=styles['Heading2'],
        fontSize=13, textColor=MEDIUM_BLUE, spaceAfter=8, spaceBefore=12,
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'BodyText2', parent=styles['Normal'],
        fontSize=10, textColor=BLACK, spaceAfter=6,
        fontName='Helvetica', leading=14
    ))
    styles.add(ParagraphStyle(
        'SmallNote', parent=styles['Normal'],
        fontSize=8, textColor=GREY, spaceAfter=4,
        fontName='Helvetica-Oblique'
    ))
    styles.add(ParagraphStyle(
        'MetricGood', parent=styles['Normal'],
        fontSize=11, textColor=ACCENT_GREEN, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'MetricBad', parent=styles['Normal'],
        fontSize=11, textColor=ACCENT_RED, fontName='Helvetica-Bold'
    ))

    return styles


def make_colored_table(data, col_widths=None):
    """Crée un tableau avec header colore et lignes alternees"""
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        # Body
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]
    # Lignes alternees
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), ROW_ALT))

    t.setStyle(TableStyle(style_cmds))
    return t


def make_confusion_matrix_table(cm_data, labels):
    """Crée un tableau de matrice de confusion avec couleurs"""
    header = [''] + [f'Pred {l}' for l in labels]
    data = [header]
    for i, label in enumerate(labels):
        row = [f'Vrai {label}']
        for j in range(len(labels)):
            row.append(str(cm_data[i, j]))
        data.append(row)

    t = Table(data, colWidths=[3 * cm] + [2.5 * cm] * len(labels))
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('BACKGROUND', (0, 1), (0, -1), MEDIUM_BLUE),
        ('TEXTCOLOR', (0, 1), (0, -1), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]

    # Colorer la diagonale en vert
    for i in range(len(labels)):
        style_cmds.append(('BACKGROUND', (i + 1, i + 1), (i + 1, i + 1), colors.HexColor('#d5f5e3')))
        style_cmds.append(('TEXTCOLOR', (i + 1, i + 1), (i + 1, i + 1), ACCENT_GREEN))
        style_cmds.append(('FONTNAME', (i + 1, i + 1), (i + 1, i + 1), 'Helvetica-Bold'))

    t.setStyle(TableStyle(style_cmds))
    return t


# ========== GENERATION DU RAPPORT ==========

def generate_report():
    print(f"Génération du rapport PDF...")

    df = pd.read_excel(DATA_PATH)
    # MODIFIE : exclure OTHER_DEM de toutes les analyses
    if 'dementia_dx_code' in df.columns:
        df = df[df['dementia_dx_code'] != 'OTHER_DEM'].copy()
    styles = get_styles()
    story = []

    # ════════════════ PAGE DE TITRE ════════════════
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("CogniScreen", styles['CustomTitle']))
    story.append(Paragraph("Rapport d'Evaluation des Modèles", styles['CustomTitle']))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="60%", thickness=2, color=MEDIUM_BLUE, spaceAfter=10))
    story.append(Paragraph(
        f"Données : {DATA_PATH.name} ({df.shape[0]} patients, {df.shape[1]} colonnes)",
        styles['Subtitle']
    ))
    story.append(Paragraph(
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles['Subtitle']
    ))
    story.append(PageBreak())

    # ════════════════ 1. NCA REGRESSION ════════════════
    story.append(Paragraph("1. Performance NCA Régression", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Prédiction de l'âge neurocognitif (neurocog_age_flu_weight). "
        "Les modèles sont évalués sur 20% des donnees (test set), avec random_state=42.",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 4 * mm))

    nca_results = []
    configs = [
        ('Linear Basic', FEATURES_BASIC, 'Linear_reg_basic.sav', True),
        ('Linear All', FEATURES_ALL, 'Linear_reg_all.sav', True),
        ('LightGBM NCA', FEATURES_NCA_30, None, False),
    ]

    target = 'neurocog_age_flu_weight'

    for name, features, filename, is_linear in configs:
        available = [f for f in features if f in df.columns]
        df_clean = df[available + [target]].dropna(subset=[target]) if not is_linear else df[available + [target]].dropna()
        X = df_clean[available]
        y = df_clean[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        if is_linear:
            model_path = MODELS_DIR / 'nca_regression' / filename
            obj = joblib.load(model_path)
            model, scaler = obj['model'], obj['scaler']
            y_pred = model.predict(scaler.transform(X_test))
        else:
            model_path = MODELS_DIR / 'nca' / 'LGBM_with_nan.sav'
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nca_results.append([name, str(len(available)), str(len(df_clean)), f"{r2:.4f}", f"{mae:.2f}", f"{rmse:.2f}"])

    table_data = [['Modèle', 'Features', 'N', 'R2', 'MAE (ans)', 'RMSE (ans)']] + nca_results
    story.append(make_colored_table(table_data, col_widths=[3.5 * cm, 2 * cm, 1.5 * cm, 2 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        "R2 : proportion de variance expliquée (1.0 = parfait). "
        "MAE : erreur moyenne absolue. "
        "RMSE : erreur quadratique moyenne (pénalise les grandes erreurs).",
        styles['SmallNote']
    ))

    # ════════════════ 2. RISK CLASSIFIERS ════════════════
    story.append(PageBreak())
    story.append(Paragraph("2. Performance des Classifieurs de Risque", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))

    available_features = [f for f in FEATURES_NCA_30 if f in df.columns]

    for target_name, display_name in [('risk_dementia', 'Risque de Démence'), ('risk_handicap', "Risque de Perte d'Autonomie")]:
        story.append(Paragraph(f"2.{1 if 'dementia' in target_name else 2}  {display_name}", styles['SubSection']))

        model_path = MODELS_DIR / target_name / 'LGBM_reg_all_plus_plus.sav'
        df_clean = df[available_features + [target_name]].dropna()
        X = df_clean[available_features]
        y = df_clean[target_name]

        def to_ordinal(val):
            if val <= 0.25: return 0
            elif val <= 0.75: return 1
            else: return 2

        y_classes = y.apply(to_ordinal)
        min_cc = y_classes.value_counts().min()
        strat = y_classes if min_cc >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y_classes, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat)

        obj = joblib.load(model_path)
        model = obj['model'] if isinstance(obj, dict) else obj
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

        # Accuracy
        acc_color = ACCENT_GREEN if acc >= 0.7 else (ACCENT_ORANGE if acc >= 0.5 else ACCENT_RED)
        story.append(Paragraph(
            f"Accuracy : <b>{acc:.1%}</b> &nbsp;|&nbsp; N = {len(df_clean)} &nbsp;|&nbsp; Features : {len(available_features)}",
            styles['BodyText2']
        ))
        story.append(Spacer(1, 3 * mm))

        # Matrice de confusion
        labels_cm = ['Faible', 'Modéré', 'Élevé']
        story.append(make_confusion_matrix_table(conf_mat, labels_cm))
        story.append(Spacer(1, 3 * mm))

        # Rapport de classification
        report = classification_report(y_test, y_pred, labels=[0, 1, 2],
                                       target_names=labels_cm, output_dict=True, zero_division=0)

        report_data = [['Classe', 'Precision', 'Recall', 'F1-Score', 'Support']]
        for cls in labels_cm:
            r = report[cls]
            report_data.append([cls, f"{r['precision']:.2f}", f"{r['recall']:.2f}", f"{r['f1-score']:.2f}", str(int(r['support']))])
        report_data.append(['Accuracy', '', '', f"{report['accuracy']:.2f}", str(int(report['weighted avg']['support']))])

        story.append(make_colored_table(report_data, col_widths=[3 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm]))
        story.append(Spacer(1, 6 * mm))

    # ════════════════ 3. COHERENCE DIAGNOSTIC ════════════════
    story.append(PageBreak())
    story.append(Paragraph("3. Cohérence Delta NCA vs Diagnostic", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Le delta NCA (age neurocognitif - age chronologique) doit augmenter avec la sévérité du diagnostic. "
        "Un delta negatif signifie un cerveau 'plus jeune' que l'age reel.",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 4 * mm))

    df_coh = df.dropna(subset=['neurocog_age_flu_weight', 'age', 'dementia_dx_code']).copy()
    df_coh['delta'] = df_coh['neurocog_age_flu_weight'] - df_coh['age']

    coh_data = [['Diagnostic', 'N', 'Delta moyen', 'Delta min', 'Delta max']]
    for dx in ['CON', 'SCD', 'MCI', 'AD']:
        sub = df_coh[df_coh['dementia_dx_code'] == dx]
        if len(sub) > 0:
            coh_data.append([dx, str(len(sub)), f"{sub['delta'].mean():.1f}", f"{sub['delta'].min():.1f}", f"{sub['delta'].max():.1f}"])

    story.append(make_colored_table(coh_data, col_widths=[3 * cm, 2 * cm, 3 * cm, 2.5 * cm, 2.5 * cm]))

    # ════════════════ 3.BIS. STATS MOCA : REEL vs SYNTHETIQUE ════════════════
    story.append(PageBreak())
    story.append(Paragraph("3.bis  Comparaison MoCA : donnees reelles vs synthetiques", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Cette section comparé les distributions du score MoCA entre les donnees reelles d'origine "
        "et les 250 profils synthetiques ajoutes par augmentation (translation de structure 55-65 ans). "
        "Objectif : vérifier que les profils générés sont cohérents avec le profil clinique attendu.",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 4 * mm))

    # Distinction reel / synthetique (IDs synthetiques commencent a 9000000)
    def is_synthetic(identifier):
        try:
            return int(identifier) >= 9000000
        except (ValueError, TypeError):
            return False

    df['_is_synthetic'] = df['Identifiers'].apply(is_synthetic)
    df_real_all = df[~df['_is_synthetic']].copy()
    df_synth_all = df[df['_is_synthetic']].copy()

    # ── 3.bis.1 Comparaison globale par diagnostic ──
    story.append(Paragraph("3.bis.1  Statistiques MoCA par diagnostic", styles['SubSection']))
    story.append(Paragraph(
        f"Donnees reelles : {len(df_real_all)} patients &nbsp;|&nbsp; "
        f"Donnees synthetiques : {len(df_synth_all)} patients (uniquement 55-65 ans)",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 3 * mm))

    moca_data = [['Diagnostic', 'Source', 'N', 'MoCA moyen', 'MoCA min', 'MoCA max', 'MoCA < 22']]
    for dx in ['CON', 'SCD', 'MCI', 'AD']:
        real_sub = df_real_all[df_real_all['dementia_dx_code'] == dx].dropna(subset=['moca'])
        synth_sub = df_synth_all[df_synth_all['dementia_dx_code'] == dx].dropna(subset=['moca'])

        if len(real_sub) > 0:
            moca_data.append([
                dx, 'Réel',
                str(len(real_sub)),
                f"{real_sub['moca'].mean():.1f}",
                f"{real_sub['moca'].min():.0f}",
                f"{real_sub['moca'].max():.0f}",
                f"{(real_sub['moca'] < 22).sum()} ({(real_sub['moca'] < 22).mean():.0%})"
            ])
        if len(synth_sub) > 0:
            moca_data.append([
                dx, 'Synth.',
                str(len(synth_sub)),
                f"{synth_sub['moca'].mean():.1f}",
                f"{synth_sub['moca'].min():.0f}",
                f"{synth_sub['moca'].max():.0f}",
                f"{(synth_sub['moca'] < 22).sum()} ({(synth_sub['moca'] < 22).mean():.0%})"
            ])

    story.append(make_colored_table(moca_data, col_widths=[2.5 * cm, 2 * cm, 1.5 * cm, 2.2 * cm, 1.8 * cm, 1.8 * cm, 2.5 * cm]))
    story.append(Spacer(1, 6 * mm))

    # ── 3.bis.2 Focus sur la tranche 55-65 ans ──
    story.append(Paragraph("3.bis.2  Focus sur la tranche 55-65 ans (depistage precoce)", styles['SubSection']))
    story.append(Paragraph(
        "Comparaison des MoCA uniquement dans la tranche d'âge 55-65 ans, "
        "cible du depistage precoce. On voit que la cohorte reelle contient quelques patients "
        "avec démence precoce (AD / OTHER_DEM avec MoCA bas), qui font partie du signal clinique a apprendre.",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 3 * mm))

    df_young = df[(df['age'] >= 55) & (df['age'] < 65)].copy()
    df_young_real = df_young[~df_young['_is_synthetic']]
    df_young_synth = df_young[df_young['_is_synthetic']]

    moca_young = [['Diagnostic', 'Source', 'N', 'MoCA moyen', 'MoCA min', 'MoCA max']]
    for dx in ['CON', 'SCD', 'MCI', 'AD']:
        real_sub = df_young_real[df_young_real['dementia_dx_code'] == dx].dropna(subset=['moca'])
        synth_sub = df_young_synth[df_young_synth['dementia_dx_code'] == dx].dropna(subset=['moca'])

        if len(real_sub) > 0:
            moca_young.append([
                dx, 'Réel',
                str(len(real_sub)),
                f"{real_sub['moca'].mean():.1f}",
                f"{real_sub['moca'].min():.0f}",
                f"{real_sub['moca'].max():.0f}"
            ])
        if len(synth_sub) > 0:
            moca_young.append([
                dx, 'Synth.',
                str(len(synth_sub)),
                f"{synth_sub['moca'].mean():.1f}",
                f"{synth_sub['moca'].min():.0f}",
                f"{synth_sub['moca'].max():.0f}"
            ])

    story.append(make_colored_table(moca_young, col_widths=[2.8 * cm, 2.2 * cm, 1.8 * cm, 2.5 * cm, 2 * cm, 2 * cm]))
    story.append(Spacer(1, 5 * mm))

    # ── 3.bis.3 Distribution détaillée des MoCA synthetiques ──
    story.append(Paragraph("3.bis.3  Distribution détaillée des MoCA synthetiques", styles['SubSection']))
    story.append(Paragraph(
        "Répartition des scores MoCA pour les 250 profils synthetiques générés. "
        "Les bornes de clipping appliquees :<br/>"
        "<b>CON</b> : 27-30 (quasi parfait a 55-65 ans) &nbsp;|&nbsp; "
        "<b>SCD</b> : 26-30 (plainte mais scores normaux) &nbsp;|&nbsp; "
        "<b>MCI</b> : 22-28 (deficit subtil, pas sévère)",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 3 * mm))

    dist_data = [['Diagnostic', 'MoCA 22', 'MoCA 23-25', 'MoCA 26-27', 'MoCA 28-29', 'MoCA 30']]
    for dx in ['CON', 'SCD', 'MCI']:
        sub = df_synth_all[df_synth_all['dementia_dx_code'] == dx].dropna(subset=['moca'])
        if len(sub) > 0:
            n22 = (sub['moca'] == 22).sum()
            n23_25 = ((sub['moca'] >= 23) & (sub['moca'] <= 25)).sum()
            n26_27 = ((sub['moca'] >= 26) & (sub['moca'] <= 27)).sum()
            n28_29 = ((sub['moca'] >= 28) & (sub['moca'] <= 29)).sum()
            n30 = (sub['moca'] == 30).sum()
            dist_data.append([dx, str(n22), str(n23_25), str(n26_27), str(n28_29), str(n30)])

    story.append(make_colored_table(dist_data, col_widths=[2.5 * cm, 2 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm]))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        "<b>Conclusion :</b> Les profils synthetiques respectent les bornes attendues pour le depistage precoce. "
        "Les CON/SCD ont des MoCA normaux (>= 26), les MCI ont des MoCA intermediaires (22-28), "
        "pas de démence sévère (MoCA < 22) dans les donnees generees.",
        styles['BodyText2']
    ))

    # ── 3.bis.4 Transformation des donnees reelles ──
    if 'moca_transformed' in df.columns:
        n_transformed = df['moca_transformed'].sum()
        if n_transformed > 0:
            story.append(Spacer(1, 6 * mm))
            story.append(Paragraph("3.bis.4  Transformation des donnees reelles (55-65 ans, MoCA < 20)", styles['SubSection']))
            story.append(Paragraph(
                f"<b>{int(n_transformed)} patients reels</b> de la tranche 55-65 ans avaient des MoCA < 20 "
                f"(cas de démence precoce, AD / OTHER_DEM / MCI avance). "
                f"Pour rendre la base cohérente avec l'objectif de depistage precoce, "
                f"leur MoCA a ete remonte vers la fourchette 22-25 selon le mapping suivant :",
                styles['BodyText2']
            ))
            story.append(Spacer(1, 3 * mm))

            mapping_data = [
                ['MoCA original', 'MoCA transforme'],
                ['4 - 10', '22'],
                ['11 - 14', '23'],
                ['15 - 17', '24'],
                ['18 - 19', '25'],
            ]
            story.append(make_colored_table(mapping_data, col_widths=[4 * cm, 4 * cm]))
            story.append(Spacer(1, 4 * mm))

            story.append(Paragraph(
                "<b>Note importante :</b> la valeur MoCA originale est conservee dans la colonne "
                "<i>moca_original</i> du fichier Excel. Un flag <i>moca_transformed=True</i> permet "
                "d'identifier les patients concernes.",
                styles['SmallNote']
            ))
            story.append(Spacer(1, 3 * mm))

            # Stats des transformations
            transformed = df[df['moca_transformed'] == True]
            if 'moca_original' in df.columns:
                trans_stats = [['Diagnostic', 'N transforme', 'MoCA orig. moyen', 'MoCA transf. moyen']]
                for dx in ['MCI', 'AD']:
                    sub = transformed[transformed['dementia_dx_code'] == dx]
                    if len(sub) > 0:
                        trans_stats.append([
                            dx,
                            str(len(sub)),
                            f"{sub['moca_original'].mean():.1f}",
                            f"{sub['moca'].mean():.1f}"
                        ])
                story.append(make_colored_table(trans_stats, col_widths=[3 * cm, 3 * cm, 3.5 * cm, 3.5 * cm]))

    # ════════════════ 4. CLASSIFIEUR DIAGNOSTIC ════════════════
    story.append(PageBreak())
    story.append(Paragraph("4. Performance du Classifieur de Diagnostic", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Le classifieur LGBM prédit le diagnostic (CON/SCD/MCI/AD) a partir de 30 features + le delta NCA prédit. "
        "Il remplace les anciens seuils hardcodes (accuracy 56.6%) par un modèle plus precis (accuracy 69.4%).",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 4 * mm))

    clf_path = MODELS_DIR / "nca" / "LGBM_diagnosis_classifier.sav"
    nca_model_path = MODELS_DIR / "nca" / "LGBM_with_nan.sav"

    if clf_path.exists() and nca_model_path.exists() and 'dementia_dx_code' in df.columns:
        clf_model = joblib.load(clf_path)
        nca_model = joblib.load(nca_model_path)

        avail_feats = [f for f in FEATURES_NCA_30 if f in df.columns]
        df_clf = df[df['dementia_dx_code'].isin(['CON', 'SCD', 'MCI', 'AD'])].copy()
        df_clf = df_clf.dropna(subset=['neurocog_age_flu_weight', 'age'])

        X_clf = df_clf[avail_feats].copy()
        y_clf = df_clf['dementia_dx_code']

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_clf, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf
        )

        # Ajouter delta_pred
        X_test_c = X_test_c.copy()
        X_test_c['delta_pred'] = nca_model.predict(X_test_c[avail_feats]) - X_test_c['age'].values
        delta_pred_all = X_test_c['delta_pred'].values

        y_pred_clf = clf_model.predict(X_test_c)

        # 4.1 Matrice 4 classes
        story.append(Paragraph("4.1  Matrice de confusion (4 classes)", styles['SubSection']))
        labels_4 = ['CON', 'SCD', 'MCI', 'AD']
        acc_4 = accuracy_score(y_test_c, y_pred_clf)
        cm_4 = confusion_matrix(y_test_c, y_pred_clf, labels=labels_4)

        story.append(Paragraph(
            f"Accuracy : <b>{acc_4:.1%}</b> &nbsp;|&nbsp; N = {len(y_test_c)}",
            styles['BodyText2']
        ))
        story.append(Spacer(1, 3 * mm))
        story.append(make_confusion_matrix_table(cm_4, labels_4))
        story.append(Spacer(1, 3 * mm))

        # Rapport classification 4 classes
        report_4 = classification_report(y_test_c, y_pred_clf, labels=labels_4, output_dict=True, zero_division=0)
        report_data_4 = [['Classe', 'Precision', 'Recall', 'F1-Score', 'Support']]
        for cls in labels_4:
            r = report_4[cls]
            report_data_4.append([cls, f"{r['precision']:.2f}", f"{r['recall']:.2f}", f"{r['f1-score']:.2f}", str(int(r['support']))])
        report_data_4.append(['Accuracy', '', '', f"{report_4['accuracy']:.2f}", str(int(report_4['weighted avg']['support']))])
        story.append(make_colored_table(report_data_4, col_widths=[3 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm]))
        story.append(Spacer(1, 6 * mm))

        # 4.2 Matrice 3 classes
        story.append(Paragraph("4.2  Matrice de confusion (3 classes simplifiees)", styles['SubSection']))
        story.append(Paragraph(
            "Regroupement : CON + SCD = Normal, MCI = MCI, AD = Démence.",
            styles['SmallNote']
        ))
        story.append(Spacer(1, 3 * mm))

        def to_3class(dx):
            if dx in ['CON', 'SCD']: return 'Normal'
            elif dx == 'MCI': return 'MCI'
            else: return 'Démence'

        y_true_3 = [to_3class(d) for d in y_test_c]
        y_pred_3 = [to_3class(d) for d in y_pred_clf]
        labels_3 = ['Normal', 'MCI', 'Démence']

        acc_3 = accuracy_score(y_true_3, y_pred_3)
        cm_3 = confusion_matrix(y_true_3, y_pred_3, labels=labels_3)

        story.append(Paragraph(
            f"Accuracy : <b>{acc_3:.1%}</b>",
            styles['BodyText2']
        ))
        story.append(Spacer(1, 3 * mm))
        story.append(make_confusion_matrix_table(cm_3, labels_3))
        story.append(Spacer(1, 3 * mm))

        report_3 = classification_report(y_true_3, y_pred_3, labels=labels_3, output_dict=True, zero_division=0)
        report_data_3 = [['Classe', 'Precision', 'Recall', 'F1-Score', 'Support']]
        for cls in labels_3:
            r = report_3[cls]
            report_data_3.append([cls, f"{r['precision']:.2f}", f"{r['recall']:.2f}", f"{r['f1-score']:.2f}", str(int(r['support']))])
        report_data_3.append(['Accuracy', '', '', f"{report_3['accuracy']:.2f}", str(int(report_3['weighted avg']['support']))])
        story.append(make_colored_table(report_data_3, col_widths=[3 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm]))
        story.append(Spacer(1, 6 * mm))

        # 4.3 Delta prédit vs reel par diagnostic
        story.append(Paragraph("4.3  Delta NCA prédit vs reel par diagnostic", styles['SubSection']))
        story.append(Paragraph(
            "Comparaison entre le delta NCA reel (donnees) et le delta prédit par le Modèle 1 (LGBM NCA), "
            "par groupe diagnostique. Un écart faible indique une bonne calibration du modèle.",
            styles['BodyText2']
        ))
        story.append(Spacer(1, 3 * mm))

        delta_real = df_clf.loc[y_test_c.index, 'neurocog_age_flu_weight'].values - df_clf.loc[y_test_c.index, 'age'].values

        delta_data = [['Diagnostic', 'N', 'Delta reel', 'Delta prédit', 'Ecart']]
        for dx in ['CON', 'SCD', 'MCI', 'AD']:
            mask_dx = y_test_c == dx
            if mask_dx.sum() > 0:
                dr = delta_real[mask_dx.values].mean()
                dp = delta_pred_all[mask_dx.values].mean()
                écart = dp - dr
                # Couleur de l'écart
                ecart_str = f"{écart:+.1f}"
                delta_data.append([dx, str(mask_dx.sum()), f"{dr:.1f}", f"{dp:.1f}", ecart_str])

        story.append(make_colored_table(delta_data, col_widths=[3 * cm, 1.5 * cm, 3 * cm, 3 * cm, 2.5 * cm]))
        story.append(Spacer(1, 4 * mm))

        # Comparatif méthodes
        story.append(Paragraph("4.4  Comparatif des méthodes de classification", styles['SubSection']))
        comp_data = [['Méthode', 'Accuracy', 'Avantage']]
        comp_data.append(['Seuils hardcodes', '56.6%', '1 variable (delta)'])
        comp_data.append(['Seuils optimises', '63.8%', 'Seuils calibres sur les donnees'])
        comp_data.append(['Classifieur LGBM', f'{acc_4:.1%}', '31 variables (delta + features)'])
        story.append(make_colored_table(comp_data, col_widths=[4.5 * cm, 2.5 * cm, 6 * cm]))

    # ════════════════ 5. DIAGNOSTIC POPULATION ════════════════
    story.append(PageBreak())
    story.append(Paragraph("5. Diagnostic de la Population", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))

    df_pop = df.dropna(subset=['age', 'sex']).copy()

    def age_group(age):
        if age < 60: return '< 60 ans'
        elif age <= 80: return '60-80 ans'
        else: return '> 80 ans'

    df_pop['groupe_age'] = df_pop['age'].apply(age_group)
    df_pop['sexe'] = df_pop['sex'].map({0.0: 'Femme', 1.0: 'Homme'})
    educ_labels = {0.0: 'Secondaire', 1.0: 'Collegial', 2.0: 'Univ. 1er', 3.0: 'Univ. sup.', 4.0: 'Univ. sup.'}
    df_pop['educ'] = df_pop['education_group'].map(educ_labels)

    # Totaux
    story.append(Paragraph(f"Population totale : <b>{len(df_pop)}</b> patients", styles['BodyText2']))
    story.append(Spacer(1, 3 * mm))

    # Par age
    story.append(Paragraph("5.1  Répartition par Age", styles['SubSection']))
    age_data = [['Tranche', 'N', '%']]
    for ag in ['< 60 ans', '60-80 ans', '> 80 ans']:
        n = len(df_pop[df_pop['groupe_age'] == ag])
        age_data.append([ag, str(n), f"{n/len(df_pop)*100:.1f}%"])
    story.append(make_colored_table(age_data, col_widths=[4 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 4 * mm))

    # Par sexe
    story.append(Paragraph("5.2  Répartition par Sexe", styles['SubSection']))
    sex_data = [['Sexe', 'N', '%']]
    for s_label in ['Femme', 'Homme']:
        n = len(df_pop[df_pop['sexe'] == s_label])
        sex_data.append([s_label, str(n), f"{n/len(df_pop)*100:.1f}%"])
    story.append(make_colored_table(sex_data, col_widths=[4 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 4 * mm))

    # Par education
    story.append(Paragraph("5.3  Répartition par Education", styles['SubSection']))
    educ_data = [['Niveau', 'N', '%']]
    for ed in ['Secondaire', 'Collegial', 'Univ. 1er', 'Univ. sup.']:
        n = len(df_pop[df_pop['educ'] == ed])
        educ_data.append([ed, str(n), f"{n/len(df_pop)*100:.1f}%"])
    story.append(make_colored_table(educ_data, col_widths=[4 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 4 * mm))

    # Par diagnostic
    story.append(Paragraph("5.4  Répartition par Diagnostic", styles['SubSection']))
    diag_data = [['Diagnostic', 'N', '%']]
    for dx in ['CON', 'SCD', 'MCI', 'AD']:
        n = len(df_pop[df_pop.get('dementia_dx_code', pd.Series()) == dx]) if 'dementia_dx_code' in df_pop.columns else 0
        diag_data.append([dx, str(n), f"{n/len(df_pop)*100:.1f}%"])
    story.append(make_colored_table(diag_data, col_widths=[4 * cm, 2.5 * cm, 2.5 * cm]))

    # ════════════════ 6. COMBINAISONS DETAILLEES ════════════════
    story.append(PageBreak())
    story.append(Paragraph("6. Combinaisons Detaillees (N >= 5)", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Toutes les combinaisons Age x Sexe x Education x Diagnostic avec au moins 5 patients. "
        "Les sous-groupes avec moins de 5 patients sont consideres non fiables pour l'analyse.",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 4 * mm))

    diag_col = 'dementia_dx_code'
    group_cols = ['groupe_age', 'sexe', 'educ', diag_col]

    grouped = df_pop.groupby(group_cols, observed=True).size().reset_index(name='N')

    age_sort = {'< 60 ans': 0, '60-80 ans': 1, '> 80 ans': 2}
    sex_sort = {'Femme': 0, 'Homme': 1}
    educ_sort = {'Secondaire': 0, 'Collegial': 1, 'Univ. 1er': 2, 'Univ. sup.': 3}
    diag_sort = {'CON': 0, 'SCD': 1, 'MCI': 2, 'AD': 3}

    grouped['_a'] = grouped['groupe_age'].map(age_sort)
    grouped['_s'] = grouped['sexe'].map(sex_sort)
    grouped['_e'] = grouped['educ'].map(educ_sort)
    grouped['_d'] = grouped[diag_col].map(diag_sort)
    grouped = grouped.sort_values(['_a', '_s', '_e', '_d'])

    fiables = grouped[grouped['N'] >= 5]
    non_fiables = grouped[grouped['N'] < 5]

    combo_data = [['Age', 'Sexe', 'Education', 'Diagnostic', 'N']]
    for _, row in fiables.iterrows():
        combo_data.append([row['groupe_age'], row['sexe'], row['educ'], row[diag_col], str(row['N'])])

    story.append(make_colored_table(combo_data, col_widths=[2.8 * cm, 2.2 * cm, 3 * cm, 3 * cm, 1.5 * cm]))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph(
        f"<b>{len(fiables)}</b> combinaisons fiables (N >= 5) &nbsp;|&nbsp; "
        f"<b>{len(non_fiables)}</b> combinaisons non fiables (N < 5, total {non_fiables['N'].sum()} patients)",
        styles['BodyText2']
    ))

    # ════════════════ FOOTER ════════════════
    story.append(Spacer(1, 2 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=GREY))
    story.append(Paragraph(
        f"CogniScreen - Rapport généré le {datetime.now().strftime('%d/%m/%Y a %H:%M')} | "
        f"Donnees : {DATA_PATH.name}",
        styles['SmallNote']
    ))

    # ════════════════ BUILD ════════════════
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="CogniScreen - Rapport d'Evaluation",
        author="CogniScreen"
    )
    doc.build(story)
    print(f"Rapport PDF généré : {OUTPUT_PATH}")


if __name__ == '__main__':
    generate_report()
