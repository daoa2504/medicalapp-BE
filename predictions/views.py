"""
views.py - VERSION CORRIGÉE POUR PRODUCTION
Chemins relatifs + Lazy loading
"""

import io
import urllib
import base64
import numpy as np
from pathlib import Path
from django.shortcuts import render
from .forms import PredictionForm
from .utils import predict_with_model_1, predict_with_model_2, predict_with_model_3
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ========== CONFIGURATION DES CHEMINS ==========

BASE_DIR = Path(__file__).resolve().parent
# MODIFIE : utilisation des donnees reelles
DATA_PATH = BASE_DIR / "data" / "Example_database_withoutrois1.xlsx"

print(f"📂 views.py - DATA_PATH: {DATA_PATH}")

# ========== LAZY LOADING ==========

_DF_CACHE = None

def get_sample_dataframe():
    """Charge le DataFrame (une seule fois)"""
    global _DF_CACHE
    
    if _DF_CACHE is None:
        if not DATA_PATH.exists():
            print(f"⚠️ Fichier non trouvé : {DATA_PATH}")
            return pd.DataFrame({'neurocog_age_flu_weight': [], 'age': []})
        
        try:
            print(f"📂 Chargement de {DATA_PATH}...")
            _DF_CACHE = pd.read_excel(DATA_PATH)
            print(f"✅ {len(_DF_CACHE)} lignes chargées")
        except Exception as e:
            print(f"❌ Erreur : {e}")
            _DF_CACHE = pd.DataFrame({'neurocog_age_flu_weight': [], 'age': []})
    
    return _DF_CACHE


def home(request):
    return render(request, 'predictions/index.html')


def prediction_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        
        if form.is_valid():
            data = form.cleaned_data
            numeric_data = {
                key: float(value) if value not in [None, ''] else 0 
                for key, value in data.items()
            }
            
            has_optional_fields = any(
                numeric_data[field] not in [None, '', 0] for field in [
                    'handedness', 'nb_language', 'hearing', 'moca', 'ravlt_imm', 
                    'ravlt_delay', 'logic_imm', 'logic_delay'
                ]
            )

            has_optional_fields_plus_plus = any(
                numeric_data[field] in [1, 0] for field in [
                    'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income', 
                    'retired', 'stroke', 'tbi', 'hta', 'diab_type2', 'obesity', 'depression', 
                    'anxiety', 'smoking', 'alcohol', 'poly_pharm5', 'physical_activity', 
                    'social_life', 'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
                ]
            )
            
            if has_optional_fields_plus_plus:
                result = predict_with_model_3(numeric_data)
            elif has_optional_fields:
                result = predict_with_model_2(numeric_data)
            else:
                result = predict_with_model_1(numeric_data)

            result['risk_dementia'] = round(result['risk_dementia'] * 100, 2)
            result['risk_handicap'] = round(result['risk_handicap'] * 100, 2)
            result['risk_dementia_comment'] = f"Risk of Dementia: {result['risk_dementia']}% ±5%"
            result['risk_handicap_comment'] = f"Risk of Handicap: {result['risk_handicap']}% ±5%"
            result['neurocog_age_flu_weight_comment'] = f"Neurocog Age Flu Weight: {result['neurocog_age_flu_weight']} ±1.5"
            result['delta_neurocogage_flu_weight_comment'] = f"Delta Neurocog Age Flu Weight: {result['delta_neurocogage_flu_weight']} ±0.3"

            # ✅ Charger le DataFrame (lazy loading)
            df = get_sample_dataframe()

            scatter_fig = go.Figure()

            if len(df) > 0:
                scatter_fig.add_trace(go.Scatter(
                    x=df['neurocog_age_flu_weight'],
                    y=df['age'],
                    mode='markers',
                    name='Sample Data',
                    marker=dict(color='blue', opacity=0.6)
                ))
                age_min = df['age'].min()
                age_max = df['age'].max()
            else:
                age_min = 50
                age_max = 90

            scatter_fig.add_trace(go.Scatter(
                x=[result['neurocog_age_flu_weight']],
                y=[numeric_data['age']],
                mode='markers',
                name='User Prediction',
                marker=dict(color='red', size=10)
            ))

            scatter_fig.update_layout(
                xaxis=dict(
                    title='Neurocog Age Flu Weight',
                    rangeslider=dict(visible=True),
                    type='linear'
                ),
                yaxis=dict(
                    title='Age',
                    range=[age_min, age_max]
                ),
                updatemenus=[
                    dict(
                        buttons=list([
                            dict(args=["yaxis.range", [age_min, age_max]], label="All Ages", method="relayout"),
                            dict(args=["yaxis.range", [age_min, age_min + 20]], label="0-20", method="relayout"),
                            dict(args=["yaxis.range", [age_min + 20, age_min + 40]], label="20-40", method="relayout"),
                            dict(args=["yaxis.range", [age_min + 40, age_min + 60]], label="40-60", method="relayout"),
                            dict(args=["yaxis.range", [age_min + 60, age_max]], label="60+", method="relayout")
                        ]),
                        direction="down",
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    ),
                ]
            )

            gauge_fig = go.Figure()

            gauge_fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=result['risk_dementia'],
                domain={'x': [0, 0.5], 'y': [0, 1]},
                title={'text': "Risk of Dementia"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 20], 'color': "green"},
                        {'range': [20, 40], 'color': "yellowgreen"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': result['risk_dementia']
                    }
                }
            ))

            gauge_fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=result['risk_handicap'],
                domain={'x': [0.5, 1], 'y': [0, 1]},
                title={'text': "Risk of Handicap"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 20], 'color': "green"},
                        {'range': [20, 40], 'color': "yellowgreen"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': result['risk_handicap']
                    }
                }
            ))

            scatter_json = pio.to_json(scatter_fig)
            gauge_json = pio.to_json(gauge_fig)

            return render(request, 'predictions/result.html', {
                'result': result,
                'scatter_json': scatter_json,
                'gauge_json': gauge_json
            })
    
    else:
        form = PredictionForm()

    return render(request, 'predictions/prediction_form.html', {'form': form})