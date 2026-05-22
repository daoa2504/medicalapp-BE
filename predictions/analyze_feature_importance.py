"""
analyze_feature_importance.py - Analyse de l'importance des variables CogniScreen
==================================================================================
Extrait l'importance de chaque feature dans les modeles entraines :
  - LGBM NCA (Modele 1 du pipeline)
  - LGBM Diagnosis Classifier (Modele 2)
  - LGBM Risk Dementia
  - LGBM Risk Handicap

Deux methodes d'importance :
  - SPLIT : nombre de fois ou la feature est utilisee dans les arbres
  - GAIN  : reduction totale d'erreur apportee par la feature
            (plus pertinent que SPLIT pour mesurer l'utilite reelle)

Categorisation :
  - OBLIGATOIRES (6) : variables cles du profil cognitif
  - COGNITIVES (3)   : tests cognitifs supplementaires
  - EXPOSOME (21)    : facteurs de risque et environnement

Sortie :
  - Tableau dans le terminal
  - Graphiques PNG
  - Rapport PDF
"""

import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# ========== CONFIG ==========

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "feature_importance_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Categorisation des features
OBLIGATOIRES = {
    'age', 'sex', 'education', 'language', 'fluency_score', 'moca'
}
COGNITIVES = {
    'handedness', 'nb_language', 'hearing'
}
EXPOSOME = {
    'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income',
    'retired', 'stroke', 'tbi', 'hta', 'diab_type2', 'chol_total',
    'obesity', 'depression', 'anxiety', 'smoking', 'alcohol',
    'poly_pharm5', 'physical_activity', 'social_life',
    'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
}

CATEGORY_COLORS = {
    'OBLIGATOIRE': '#dc2626',
    'COGNITIVE':   '#3b82f6',
    'EXPOSOME':    '#16a34a',
    'AUTRE':       '#9ca3af',
}


def categorize(feature_name):
    if feature_name in OBLIGATOIRES:
        return 'OBLIGATOIRE'
    if feature_name in COGNITIVES:
        return 'COGNITIVE'
    if feature_name in EXPOSOME:
        return 'EXPOSOME'
    return 'AUTRE'


def extract_importance(model, model_name):
    """Extrait split + gain importance d'un modele LightGBM"""
    feat_names = list(model.feature_name_) if hasattr(model, 'feature_name_') else list(model.booster_.feature_name())

    # Split importance (nombre d'utilisations)
    split_imp = list(model.booster_.feature_importance(importance_type='split'))
    # Gain importance (reduction d'erreur totale)
    gain_imp = list(model.booster_.feature_importance(importance_type='gain'))

    total_split = sum(split_imp) if sum(split_imp) > 0 else 1
    total_gain = sum(gain_imp) if sum(gain_imp) > 0 else 1

    items = []
    for name, s, g in zip(feat_names, split_imp, gain_imp):
        items.append({
            'feature': name,
            'category': categorize(name),
            'split': int(s),
            'gain': float(g),
            'split_pct': 100 * s / total_split,
            'gain_pct': 100 * g / total_gain,
        })

    items.sort(key=lambda x: -x['gain_pct'])
    return {
        'model_name': model_name,
        'features': items,
        'total_features': len(items),
    }


def print_table(analysis):
    """Affiche un tableau formate dans le terminal"""
    print(f"\n{'='*85}")
    print(f"  MODELE : {analysis['model_name']}")
    print(f"{'='*85}")
    print(f"  {'Feature':<25} {'Categorie':<13} {'Gain %':>8} {'Split %':>8} {'Bar':<20}")
    print(f"  {'-'*85}")

    max_gain = analysis['features'][0]['gain_pct'] if analysis['features'] else 1

    for f in analysis['features']:
        bar_len = int((f['gain_pct'] / max_gain) * 20)
        bar = '#' * bar_len
        print(f"  {f['feature']:<25} {f['category']:<13} {f['gain_pct']:>7.1f}% {f['split_pct']:>7.1f}% {bar:<20}")


def print_summary_by_category(analyses):
    """Resume par categorie : combien chaque categorie contribue dans chaque modele"""
    print(f"\n{'='*85}")
    print(f"  RESUME PAR CATEGORIE (% du gain total par categorie)")
    print(f"{'='*85}")
    print(f"  {'Modele':<35} {'OBLIG.':>10} {'COGNIT.':>10} {'EXPOSOME':>10} {'AUTRE':>10}")
    print(f"  {'-'*85}")

    for a in analyses:
        cat_sums = {'OBLIGATOIRE': 0.0, 'COGNITIVE': 0.0, 'EXPOSOME': 0.0, 'AUTRE': 0.0}
        for f in a['features']:
            cat_sums[f['category']] += f['gain_pct']
        print(f"  {a['model_name']:<35} {cat_sums['OBLIGATOIRE']:>9.1f}% {cat_sums['COGNITIVE']:>9.1f}% {cat_sums['EXPOSOME']:>9.1f}% {cat_sums['AUTRE']:>9.1f}%")


def detect_noisy_and_predictive(analysis):
    """Identifie les features bruyantes (gain < 1%) et les predicteurs surprenants de l'exposome"""
    noisy = [f for f in analysis['features'] if f['gain_pct'] < 1.0]
    exposome_predictive = [
        f for f in analysis['features']
        if f['category'] == 'EXPOSOME' and f['gain_pct'] > 2.0
    ]
    return noisy, exposome_predictive


def plot_importance(analysis, output_path):
    """Genere un graphique horizontal d'importance par feature"""
    items = analysis['features']
    feature_names = [f['feature'] for f in items]
    gains = [f['gain_pct'] for f in items]
    colors = [CATEGORY_COLORS[f['category']] for f in items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(items) * 0.25)))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, gains, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (% du gain total)', fontsize=10)
    ax.set_title(f"Feature Importance - {analysis['model_name']}", fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Legende des categories
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS['OBLIGATOIRE'], label='Obligatoires'),
        mpatches.Patch(color=CATEGORY_COLORS['COGNITIVE'], label='Cognitives'),
        mpatches.Patch(color=CATEGORY_COLORS['EXPOSOME'], label='Exposome'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_json(analyses, output_path):
    """Sauvegarde toutes les analyses en JSON"""
    out = {
        'generated_at': datetime.now().isoformat(),
        'categorization': {
            'OBLIGATOIRE': sorted(OBLIGATOIRES),
            'COGNITIVE': sorted(COGNITIVES),
            'EXPOSOME': sorted(EXPOSOME),
        },
        'analyses': analyses,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def generate_pdf_report(analyses, output_path):
    """Rapport PDF avec graphiques et tableaux"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=22,
                                  textColor=colors.HexColor('#1e3a5f'), alignment=TA_CENTER)
    section_style = ParagraphStyle('SectionH', parent=styles['Heading1'], fontSize=15,
                                    textColor=colors.white, backColor=colors.HexColor('#1e3a5f'),
                                    borderPadding=(8, 8, 8, 8), spaceAfter=10, spaceBefore=14)
    subsection_style = ParagraphStyle('Sub', parent=styles['Heading2'], fontSize=12,
                                       textColor=colors.HexColor('#2c5f8a'), spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14)

    story = []

    # Page de titre
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("CogniScreen", title_style))
    story.append(Paragraph("Analyse de l'Importance des Variables", title_style))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor('#2c5f8a')))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}",
                            ParagraphStyle('date', parent=body_style, alignment=TA_CENTER, textColor=colors.gray)))
    story.append(PageBreak())

    # Introduction
    story.append(Paragraph("1. Methodologie", section_style))
    story.append(Paragraph(
        "Ce rapport analyse l'importance de chaque variable dans les modeles LightGBM de CogniScreen. "
        "L'importance est mesuree par le <b>gain</b> (reduction totale d'erreur apportee par la feature) "
        "et le <b>split</b> (nombre d'utilisations dans les arbres). Le gain est plus pertinent pour "
        "mesurer l'utilite reelle d'une feature.",
        body_style
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Les 30 variables sont categorisees en :", body_style
    ))
    cat_data = [
        ['Categorie', 'N', 'Variables'],
        ['OBLIGATOIRE', str(len(OBLIGATOIRES)), ', '.join(sorted(OBLIGATOIRES))],
        ['COGNITIVE', str(len(COGNITIVES)), ', '.join(sorted(COGNITIVES))],
        ['EXPOSOME', str(len(EXPOSOME)), ', '.join(sorted(EXPOSOME))],
    ]
    cat_table = Table(cat_data, colWidths=[3 * cm, 1.5 * cm, 12 * cm])
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#fee2e2')),
        ('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#dbeafe')),
        ('BACKGROUND', (0, 3), (0, 3), colors.HexColor('#dcfce7')),
    ]))
    story.append(Spacer(1, 4 * mm))
    story.append(cat_table)

    # Resume par categorie
    story.append(Paragraph("2. Resume par categorie", section_style))
    story.append(Paragraph(
        "Contribution totale de chaque categorie au gain de chaque modele :",
        body_style
    ))
    story.append(Spacer(1, 3 * mm))

    summary_data = [['Modele', 'OBLIGATOIRE', 'COGNITIVE', 'EXPOSOME', 'AUTRE']]
    for a in analyses:
        cat_sums = {'OBLIGATOIRE': 0.0, 'COGNITIVE': 0.0, 'EXPOSOME': 0.0, 'AUTRE': 0.0}
        for f in a['features']:
            cat_sums[f['category']] += f['gain_pct']
        summary_data.append([
            a['model_name'],
            f"{cat_sums['OBLIGATOIRE']:.1f}%",
            f"{cat_sums['COGNITIVE']:.1f}%",
            f"{cat_sums['EXPOSOME']:.1f}%",
            f"{cat_sums['AUTRE']:.1f}%",
        ])

    summary_table = Table(summary_data, colWidths=[5.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#fee2e2')),
        ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#dbeafe')),
        ('BACKGROUND', (3, 1), (3, -1), colors.HexColor('#dcfce7')),
    ]))
    story.append(summary_table)

    # Section par modele
    for a in analyses:
        story.append(PageBreak())
        story.append(Paragraph(f"Modele : {a['model_name']}", section_style))

        # Image du graphique
        img_path = OUTPUT_DIR / f"importance_{a['model_name'].replace(' ', '_').replace('/', '_')}.png"
        if img_path.exists():
            img = RLImage(str(img_path), width=16 * cm, height=12 * cm)
            story.append(img)
            story.append(Spacer(1, 4 * mm))

        # Top 10 features
        story.append(Paragraph(f"Top 10 features par gain :", subsection_style))
        top_data = [['Rang', 'Feature', 'Categorie', 'Gain %', 'Split %']]
        for i, f in enumerate(a['features'][:10], 1):
            top_data.append([
                str(i),
                f['feature'],
                f['category'],
                f"{f['gain_pct']:.2f}%",
                f"{f['split_pct']:.2f}%",
            ])
        top_table = Table(top_data, colWidths=[1.5 * cm, 4.5 * cm, 3 * cm, 2.5 * cm, 2.5 * cm])

        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]
        for i, f in enumerate(a['features'][:10], 1):
            cat = f['category']
            color_map = {'OBLIGATOIRE': '#fee2e2', 'COGNITIVE': '#dbeafe', 'EXPOSOME': '#dcfce7', 'AUTRE': '#f3f4f6'}
            style_cmds.append(('BACKGROUND', (2, i), (2, i), colors.HexColor(color_map[cat])))

        top_table.setStyle(TableStyle(style_cmds))
        story.append(top_table)
        story.append(Spacer(1, 4 * mm))

        # Detection bruyantes / predictives
        noisy, expo_predictive = detect_noisy_and_predictive(a)

        if expo_predictive:
            story.append(Paragraph("Variables exposome surprenamment predictives (gain > 2%) :", subsection_style))
            for f in expo_predictive:
                story.append(Paragraph(
                    f"&bull; <b>{f['feature']}</b> : {f['gain_pct']:.2f}% du gain total",
                    body_style
                ))
            story.append(Spacer(1, 3 * mm))

        if noisy:
            n_noisy = len(noisy)
            noisy_names = ', '.join(f['feature'] for f in noisy[:10])
            story.append(Paragraph(
                f"<b>Features faiblement utilisees (gain < 1%) :</b> {n_noisy} features. "
                f"Premieres : {noisy_names}{'...' if n_noisy > 10 else ''}",
                body_style
            ))

    # Conclusions
    story.append(PageBreak())
    story.append(Paragraph("Conclusions et recommandations", section_style))

    # Identifier les features bruyantes communes a tous les modeles
    all_noisy = {}
    for a in analyses:
        for f in a['features']:
            if f['gain_pct'] < 1.0:
                all_noisy.setdefault(f['feature'], 0)
                all_noisy[f['feature']] += 1

    consistently_noisy = [f for f, count in all_noisy.items() if count == len(analyses)]

    story.append(Paragraph(
        f"<b>Variables systematiquement peu utilisees</b> (faible gain dans tous les modeles) : "
        f"{len(consistently_noisy)} features.",
        body_style
    ))
    if consistently_noisy:
        story.append(Paragraph(
            ', '.join(sorted(consistently_noisy)),
            body_style
        ))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        "<b>Recommandations</b> :",
        subsection_style
    ))
    story.append(Paragraph(
        "1. Les variables systematiquement peu utilisees pourraient etre retirees pour simplifier "
        "le modele sans perte de performance significative.",
        body_style
    ))
    story.append(Paragraph(
        "2. Les variables de l'exposome qui apparaissent dans le top sont des cibles d'intervention "
        "potentielles (modifiables, ex: physical_activity, social_life).",
        body_style
    ))
    story.append(Paragraph(
        "3. Une variable obligatoire avec un gain faible ne signifie pas qu'elle est inutile : "
        "elle peut servir de stratification (ex: sex) plutot que de feature predictive directe.",
        body_style
    ))

    # Build
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="CogniScreen - Analyse Feature Importance",
        author="CogniScreen"
    )
    doc.build(story)


def main():
    print("=" * 85)
    print("  ANALYSE DE L'IMPORTANCE DES VARIABLES (CogniScreen)")
    print("=" * 85)

    models_to_analyze = [
        ('NCA Regression', MODELS_DIR / "nca" / "LGBM_with_nan.sav"),
        ('Diagnosis Classifier', MODELS_DIR / "nca" / "LGBM_diagnosis_classifier.sav"),
        ('Risk Dementia', MODELS_DIR / "risk_dementia" / "LGBM_reg_all_plus_plus.sav"),
        ('Risk Handicap', MODELS_DIR / "risk_handicap" / "LGBM_reg_all_plus_plus.sav"),
    ]

    analyses = []

    for name, path in models_to_analyze:
        if not path.exists():
            print(f"  [WARN] Modele non trouve : {path}")
            continue

        print(f"\n  Analyse : {name}")
        obj = joblib.load(path)
        # Les LGBM Classifier sont sauvegardes en dict {model, features, ...}
        model = obj['model'] if isinstance(obj, dict) else obj

        analysis = extract_importance(model, name)
        analyses.append(analysis)
        print_table(analysis)

        # Generer le graphique
        img_name = f"importance_{name.replace(' ', '_').replace('/', '_')}.png"
        plot_importance(analysis, OUTPUT_DIR / img_name)
        print(f"  Graphique sauve : {img_name}")

    # Resume par categorie
    print_summary_by_category(analyses)

    # Sauvegarde JSON
    json_path = OUTPUT_DIR / "feature_importance.json"
    save_json(analyses, json_path)
    print(f"\n  JSON sauve : {json_path}")

    # Rapport PDF
    pdf_path = OUTPUT_DIR / "rapport_feature_importance.pdf"
    generate_pdf_report(analyses, pdf_path)
    print(f"  PDF sauve  : {pdf_path}")

    # Detection finale : features systematiquement bruyantes
    print(f"\n{'='*85}")
    print(f"  FEATURES SYSTEMATIQUEMENT PEU UTILISEES (gain < 1% dans tous les modeles)")
    print(f"{'='*85}")
    all_noisy = {}
    for a in analyses:
        for f in a['features']:
            if f['gain_pct'] < 1.0:
                all_noisy.setdefault(f['feature'], 0)
                all_noisy[f['feature']] += 1

    consistently_noisy = sorted([f for f, count in all_noisy.items() if count == len(analyses)])
    print(f"  {len(consistently_noisy)} features : {', '.join(consistently_noisy) if consistently_noisy else '(aucune)'}")

    print(f"\n  Termine.")


if __name__ == '__main__':
    main()
