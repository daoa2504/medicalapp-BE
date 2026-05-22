"""
generate_fine_analysis_pdf.py - Rapport PDF de l'analyse fine par sous-population
==================================================================================
Genere un rapport detaille avec :
  - Matrices de confusion par sous-groupe (age, sexe, education)
  - Metriques par sous-groupe (Accuracy, MCI Recall, F1)
  - Synthese visuelle des points forts/faibles
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER

# Importer les fonctions de fine_analysis
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fine_analysis import (
    DATA_PATH, NCA_MODEL_PATH, DIAG_CLASSIFIER_PATH,
    INCLUDED_DIAGNOSES, AGE_GROUPS, SEX_GROUPS, EDUC_GROUPS,
    FEATURES_NCA_30, analyze_subgroup,
)

OUTPUT_PATH = Path(__file__).resolve().parent / "rapport_analyse_fine.pdf"

# Couleurs
DARK_BLUE = colors.HexColor('#1e3a5f')
MEDIUM_BLUE = colors.HexColor('#2c5f8a')
ACCENT_GREEN = colors.HexColor('#27ae60')
ACCENT_RED = colors.HexColor('#e74c3c')
ACCENT_ORANGE = colors.HexColor('#f39c12')
HEADER_BG = colors.HexColor('#2c3e50')
ROW_ALT = colors.HexColor('#ecf0f1')
WHITE = colors.white
GREY = colors.HexColor('#7f8c8d')


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('CustomTitle', parent=styles['Title'],
        fontSize=22, textColor=DARK_BLUE, spaceAfter=6, alignment=TA_CENTER,
        fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle('Subtitle', parent=styles['Normal'],
        fontSize=11, textColor=GREY, spaceAfter=20, alignment=TA_CENTER))
    styles.add(ParagraphStyle('SectionTitle', parent=styles['Heading1'],
        fontSize=15, textColor=WHITE, spaceAfter=10, spaceBefore=14,
        fontName='Helvetica-Bold', backColor=DARK_BLUE,
        borderPadding=(8, 8, 8, 8)))
    styles.add(ParagraphStyle('SubSection', parent=styles['Heading2'],
        fontSize=12, textColor=MEDIUM_BLUE, spaceAfter=6, spaceBefore=10,
        fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle('BodyText2', parent=styles['Normal'],
        fontSize=10, spaceAfter=6, leading=13))
    styles.add(ParagraphStyle('SmallNote', parent=styles['Normal'],
        fontSize=8, textColor=GREY, spaceAfter=4, fontName='Helvetica-Oblique'))
    return styles


def make_styled_table(data, col_widths=None, header_color=HEADER_BG):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), ROW_ALT))
    t.setStyle(TableStyle(style))
    return t


def make_confusion_matrix_table(conf_mat, labels):
    """Matrice de confusion 3x3 avec diagonale en vert"""
    header = [''] + [f'Pred {l}' for l in labels]
    data = [header]
    for i, label in enumerate(labels):
        row = [f'Vrai {label}'] + [str(int(conf_mat[i, j])) for j in range(len(labels))]
        data.append(row)

    t = Table(data, colWidths=[3 * cm] + [2.5 * cm] * len(labels))
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('BACKGROUND', (0, 1), (0, -1), MEDIUM_BLUE),
        ('TEXTCOLOR', (0, 1), (0, -1), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    # Diagonale en vert
    for i in range(len(labels)):
        style.append(('BACKGROUND', (i + 1, i + 1), (i + 1, i + 1), colors.HexColor('#d5f5e3')))
        style.append(('TEXTCOLOR', (i + 1, i + 1), (i + 1, i + 1), ACCENT_GREEN))
        style.append(('FONTNAME', (i + 1, i + 1), (i + 1, i + 1), 'Helvetica-Bold'))

    t.setStyle(TableStyle(style))
    return t


def color_for_metric(value):
    """Couleur selon la valeur d'une metrique (0-1)"""
    if value >= 0.85:
        return ACCENT_GREEN
    elif value >= 0.70:
        return colors.HexColor('#27ae60')
    elif value >= 0.55:
        return ACCENT_ORANGE
    else:
        return ACCENT_RED


def add_subgroup_section(story, styles, result, title_prefix="Sous-groupe"):
    """Ajoute une section pour un sous-groupe"""
    if result is None:
        return

    story.append(Paragraph(f"{title_prefix} : {result['name']}", styles['SubSection']))

    # Distribution
    dist_str = ", ".join([f"{dx}: {int(n)}" for dx, n in result['distribution'].items()])
    story.append(Paragraph(
        f"<b>N = {result['n']}</b> &nbsp;|&nbsp; Distribution : {dist_str}",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 2 * mm))

    # Tableau metriques (seuils vs classifieur)
    metrics_data = [['Methode', 'Accuracy', 'MCI Recall', 'MCI Precision', 'MCI F1']]
    metrics_data.append([
        'Seuils delta',
        f"{result['thresh']['accuracy']:.1%}",
        f"{result['thresh']['mci_recall']:.1%}",
        f"{result['thresh']['mci_precision']:.1%}",
        f"{result['thresh']['mci_f1']:.1%}",
    ])
    if result['clf']:
        metrics_data.append([
            'Classifieur LGBM',
            f"{result['clf']['accuracy']:.1%}",
            f"{result['clf']['mci_recall']:.1%}",
            f"{result['clf']['mci_precision']:.1%}",
            f"{result['clf']['mci_f1']:.1%}",
        ])

    story.append(make_styled_table(metrics_data,
                                    col_widths=[3.5 * cm, 2.2 * cm, 2.5 * cm, 2.8 * cm, 2 * cm]))
    story.append(Spacer(1, 3 * mm))

    # Matrice de confusion (classifieur si dispo, sinon seuils)
    cm_to_show = result['clf']['confusion_matrix'] if result['clf'] else result['thresh']['confusion_matrix']
    method_label = "Classifieur LGBM" if result['clf'] else "Seuils delta"
    story.append(Paragraph(f"<i>Matrice de confusion ({method_label})</i>", styles['BodyText2']))
    story.append(Spacer(1, 1 * mm))
    story.append(make_confusion_matrix_table(cm_to_show, INCLUDED_DIAGNOSES))
    story.append(Spacer(1, 5 * mm))


def main():
    print("Generation du rapport PDF d'analyse fine...")

    # Charger donnees + modeles
    df = pd.read_excel(DATA_PATH)
    df = df[df['dementia_dx_code'].isin(INCLUDED_DIAGNOSES)].copy()
    df = df.dropna(subset=['neurocog_age_flu_weight', 'age', 'moca'])

    model_nca = joblib.load(NCA_MODEL_PATH)
    clf_model = joblib.load(DIAG_CLASSIFIER_PATH) if DIAG_CLASSIFIER_PATH.exists() else None
    available_features = [f for f in FEATURES_NCA_30 if f in df.columns]

    styles = get_styles()
    story = []

    # ════════════════ TITRE ════════════════
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("CogniScreen", styles['CustomTitle']))
    story.append(Paragraph("Analyse Fine des Performances NCA", styles['CustomTitle']))
    story.append(Paragraph("par Sous-Population", styles['CustomTitle']))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="60%", thickness=2, color=MEDIUM_BLUE, spaceAfter=10))
    story.append(Paragraph(
        f"Donnees : {DATA_PATH.name} ({len(df)} patients CON/SCD/MCI)",
        styles['Subtitle']
    ))
    story.append(Paragraph(
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles['Subtitle']
    ))
    story.append(Spacer(1, 1 * cm))

    # Note methodologique
    story.append(Paragraph("Methodologie", styles['SubSection']))
    story.append(Paragraph(
        "Cette analyse evalue la capacite du modele NCA a classifier correctement les diagnostics "
        "<b>CON / SCD / MCI</b>, en se concentrant sur le depistage precoce. "
        "Les cas <b>AD et OTHER_DEM ont ete exclus</b> car ils representent des stades plus avances "
        "dont la detection n'est pas l'objectif principal du depistage.",
        styles['BodyText2']
    ))
    story.append(Paragraph(
        "Deux methodes de classification sont comparees : <br/>"
        "<b>1. Seuils delta NCA</b> : delta &lt; -3 -&gt; CON ; -3 &le; delta &lt; 1 -&gt; SCD ; delta &ge; 1 -&gt; MCI<br/>"
        "<b>2. Classifieur LGBM</b> : prediction directe a partir de 30 features + delta predit",
        styles['BodyText2']
    ))
    story.append(PageBreak())

    # ════════════════ 1. ANALYSE GLOBALE ════════════════
    story.append(Paragraph("1. Analyse Globale", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    global_result = analyze_subgroup(df, "Tous patients CON/SCD/MCI", None,
                                      model_nca, clf_model, available_features)
    add_subgroup_section(story, styles, global_result, title_prefix="Population")

    # ════════════════ 2. PAR AGE ════════════════
    story.append(PageBreak())
    story.append(Paragraph("2. Analyse par Tranche d'Age", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    for age_label, age_filter in AGE_GROUPS:
        df_sub = df[df['age'].apply(age_filter)].copy()
        result = analyze_subgroup(df_sub, age_label, None, model_nca, clf_model, available_features)
        add_subgroup_section(story, styles, result, title_prefix="Age")

    # ════════════════ 3. PAR SEXE ════════════════
    story.append(PageBreak())
    story.append(Paragraph("3. Analyse par Sexe", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    for sex_label, sex_value in SEX_GROUPS:
        df_sub = df[df['sex'] == sex_value].copy()
        result = analyze_subgroup(df_sub, sex_label, None, model_nca, clf_model, available_features)
        add_subgroup_section(story, styles, result, title_prefix="Sexe")

    # ════════════════ 4. PAR EDUCATION ════════════════
    story.append(PageBreak())
    story.append(Paragraph("4. Analyse par Niveau d'Education", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    for educ_label, educ_values in EDUC_GROUPS:
        df_sub = df[df['education_group'].isin(educ_values)].copy()
        result = analyze_subgroup(df_sub, educ_label, None, model_nca, clf_model, available_features)
        add_subgroup_section(story, styles, result, title_prefix="Education")

    # ════════════════ 5. SYNTHESE ════════════════
    story.append(PageBreak())
    story.append(Paragraph("5. Synthese : Carte des Performances", styles['SectionTitle']))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Tableau synthetique avec les principales metriques par sous-groupe (Classifieur LGBM). "
        "<b>MCI Recall</b> = capacite a detecter les MCI (le plus important pour le depistage).",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 4 * mm))

    # Construire le tableau synthese
    synth_data = [['Sous-groupe', 'N', 'Accuracy', 'MCI Recall', 'MCI Precision', 'MCI F1']]

    # Global
    if global_result and global_result['clf']:
        m = global_result['clf']
        synth_data.append(['GLOBAL', str(global_result['n']),
                          f"{m['accuracy']:.1%}", f"{m['mci_recall']:.1%}",
                          f"{m['mci_precision']:.1%}", f"{m['mci_f1']:.1%}"])

    # Par age
    for age_label, age_filter in AGE_GROUPS:
        df_sub = df[df['age'].apply(age_filter)].copy()
        result = analyze_subgroup(df_sub, age_label, None, model_nca, clf_model, available_features)
        if result and result['clf']:
            m = result['clf']
            synth_data.append([f"Age : {age_label}", str(result['n']),
                              f"{m['accuracy']:.1%}", f"{m['mci_recall']:.1%}",
                              f"{m['mci_precision']:.1%}", f"{m['mci_f1']:.1%}"])

    # Par sexe
    for sex_label, sex_value in SEX_GROUPS:
        df_sub = df[df['sex'] == sex_value].copy()
        result = analyze_subgroup(df_sub, sex_label, None, model_nca, clf_model, available_features)
        if result and result['clf']:
            m = result['clf']
            synth_data.append([f"Sexe : {sex_label}", str(result['n']),
                              f"{m['accuracy']:.1%}", f"{m['mci_recall']:.1%}",
                              f"{m['mci_precision']:.1%}", f"{m['mci_f1']:.1%}"])

    # Par education
    for educ_label, educ_values in EDUC_GROUPS:
        df_sub = df[df['education_group'].isin(educ_values)].copy()
        result = analyze_subgroup(df_sub, educ_label, None, model_nca, clf_model, available_features)
        if result and result['clf']:
            m = result['clf']
            synth_data.append([f"Educ : {educ_label}", str(result['n']),
                              f"{m['accuracy']:.1%}", f"{m['mci_recall']:.1%}",
                              f"{m['mci_precision']:.1%}", f"{m['mci_f1']:.1%}"])

    story.append(make_styled_table(synth_data,
                                   col_widths=[4.5 * cm, 1.5 * cm, 2 * cm, 2.2 * cm, 2.5 * cm, 1.8 * cm]))
    story.append(Spacer(1, 6 * mm))

    # ════════════════ 6. ANALYSE CROISEE ════════════════
    story.append(Paragraph("6. Analyse Croisee : Age x Sexe", styles['SubSection']))
    story.append(Paragraph(
        "Identification des sous-populations a risque de mauvaise classification.",
        styles['BodyText2']
    ))
    story.append(Spacer(1, 3 * mm))

    cross_data = [['Age x Sexe', 'N', 'Accuracy', 'MCI Recall', 'MCI Precision']]
    for age_label, age_filter in AGE_GROUPS:
        for sex_label, sex_value in SEX_GROUPS:
            df_sub = df[df['age'].apply(age_filter) & (df['sex'] == sex_value)].copy()
            if len(df_sub) < 5:
                continue
            result = analyze_subgroup(df_sub, f"{age_label} x {sex_label}", None,
                                       model_nca, clf_model, available_features)
            if result and result['clf']:
                m = result['clf']
                cross_data.append([f"{age_label} x {sex_label}", str(result['n']),
                                  f"{m['accuracy']:.1%}", f"{m['mci_recall']:.1%}",
                                  f"{m['mci_precision']:.1%}"])

    story.append(make_styled_table(cross_data,
                                   col_widths=[5 * cm, 1.5 * cm, 2.2 * cm, 2.5 * cm, 2.8 * cm]))

    # ════════════════ FOOTER ════════════════
    story.append(Spacer(1, 1.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=GREY))
    story.append(Paragraph(
        f"CogniScreen - Rapport d'analyse fine genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}",
        styles['SmallNote']
    ))

    # ════════════════ BUILD ════════════════
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="CogniScreen - Analyse Fine",
    )
    doc.build(story)
    print(f"Rapport PDF genere : {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
