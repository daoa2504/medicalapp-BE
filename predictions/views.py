import io
import urllib, base64
import numpy as np
from django.shortcuts import render
from .forms import PredictionForm
from .utils import predict_with_model_1, predict_with_model_2,predict_with_model_3
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

df = pd.read_excel('D:\\Projects\\CogniScreen\\CogniScreen\\Moncef_elise_rg order_2\\Example_database_withoutrois.xlsx')
df.head()

def home(request):
    return render(request,'predictions/index.html')


def prediction_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract cleaned data
            data = form.cleaned_data
            numeric_data = {key: float(value) if value not in [None, ''] else 0 for key, value in data.items()}
            
            # Check if any optional fields are provided
            has_optional_fields = any(
                numeric_data[field] not in [None, '', 0] for field in [
                    'handedness', 'nb_language', 'hearing', 'moca', 'ravlt_imm', 
                    'ravlt_delay', 'logic_imm', 'logic_delay'
                ]
            )

            has_optional_fields_plus_plus = any(
                numeric_data[field] in [1, 0] for field in [
                     'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income', 
                     'retired','stroke', 'tbi', 'hta', 'diab_type2', 'obesity', 'depression', 'anxiety', 
                     'smoking', 'alcohol','poly_pharm5', 'physical_activity', 'social_life','cognitive_activities',
                     'nutrition_score','sleep_deprivation'
                ]
            )
            
            if has_optional_fields_plus_plus:
                result = predict_with_model_3(numeric_data)
            elif has_optional_fields:
                result = predict_with_model_2(numeric_data)
            else:
                result = predict_with_model_1(numeric_data)

            # Convert risks to percentages
            result['risk_dementia'] = round(result['risk_dementia'] * 100, 2)
            result['risk_handicap'] = round(result['risk_handicap'] * 100, 2)

            # Define error range comments (example with ±5% error range)
            result['risk_dementia_comment'] = f"Risk of Dementia: {result['risk_dementia']}% ±5% (Estimated Error Range)"
            result['risk_handicap_comment'] = f"Risk of Handicap: {result['risk_handicap']}% ±5% (Estimated Error Range)"
            result['neurocog_age_flu_weight_comment'] = f"Neurocog Age Flu Weight: {result['neurocog_age_flu_weight']} ±1.5 (Estimated Error Range)"
            result['delta_neurocogage_flu_weight_comment'] = f"Delta Neurocog Age Flu Weight: {result['delta_neurocogage_flu_weight']} ±0.3 (Estimated Error Range)"

            # Create the main scatter plot with Plotly
            scatter_fig = go.Figure()

            # Add sample data points
            scatter_fig.add_trace(go.Scatter(
                x=df['neurocog_age_flu_weight'],
                y=df['age'],
                mode='markers',
                name='Sample Data',
                marker=dict(color='blue', opacity=0.6)
            ))

            # Add user prediction point (always visible)
            scatter_fig.add_trace(go.Scatter(
                x=[result['neurocog_age_flu_weight']],
                y=[numeric_data['age']],
                mode='markers',
                name='User Prediction',
                marker=dict(color='red', size=10)
            ))

            # Add range slider for X-axis
            scatter_fig.update_layout(
                xaxis=dict(
                    title='Neurocog Age Flu Weight',
                    rangeslider=dict(visible=True),  # Enable range slider for X-axis
                    type='linear'
                ),
                yaxis=dict(
                    title='Age',
                    range=[df['age'].min(), df['age'].max()]  # Set the default range to the min and max age
                ),
                # Add simulated slider for Y-axis
                updatemenus=[
                    dict(
                        buttons=list([
                            dict(
                                args=["yaxis.range", [df['age'].min(), df['age'].max()]],
                                label="All Ages",
                                method="relayout"
                            ),
                            dict(
                                args=["yaxis.range", [df['age'].min(), df['age'].min() + 20]],
                                label="0-20",
                                method="relayout"
                            ),
                            dict(
                                args=["yaxis.range", [df['age'].min() + 20, df['age'].min() + 40]],
                                label="20-40",
                                method="relayout"
                            ),
                            dict(
                                args=["yaxis.range", [df['age'].min() + 40, df['age'].min() + 60]],
                                label="40-60",
                                method="relayout"
                            ),
                            dict(
                                args=["yaxis.range", [df['age'].min() + 60, df['age'].max()]],
                                label="60+",
                                method="relayout"
                            )
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

            # Create a separate figure for gauges
            gauge_fig = go.Figure()

            # Add gauge for Risk of Dementia
            gauge_fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=result['risk_dementia'],
                domain={'x': [0, 0.5], 'y': [0, 1]},  # Adjust position to left half
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

            # Add gauge for Risk of Handicap
            gauge_fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=result['risk_handicap'],
                domain={'x': [0.5, 1], 'y': [0, 1]},  # Adjust position to right half
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

            # Convert both figures to JSON format for rendering in the template
            scatter_json = pio.to_json(scatter_fig)
            gauge_json = pio.to_json(gauge_fig)

            # Send the plots and result to the template
            return render(request, 'predictions/result.html', {
                'result': result,
                'scatter_json': scatter_json,
                'gauge_json': gauge_json
            })
    else:
        form = PredictionForm()

    return render(request, 'predictions/prediction_form.html', {'form': form})
