import pickle  # or any other method to load your model
from pathlib import Path
import joblib

data_folder = Path("D:/Projects/CogniScreen/CogniScreen/Moncef_elise_rg order_2")


def load_model(model_path):
    return joblib.load(model_path)


def pred_to_proba(y_pred):
    predictions = []
    for value in y_pred:
        value = int(value * 100)
        if value < 0:
            value = 0
        elif value > 100:
            value = 100
        else:
            pass
        predictions.append(value / 100)

    return predictions


# Load your model from the .sav file
model_path = data_folder / 'Linear_reg_basic.sav'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)


def predict_neurocog_age_flu_weight(data):
    # Extract features from data
    features = [data['age'], data['sex'], data['education'], data['language'], data['fluency_score']]

    # Predict using your model
    prediction = model.predict([features])

    return prediction[0]


# Example functions for predictions
def predict_with_model_1(data):
    # Extract only the necessary fields for this model
    necessary_fields = {
        'age': data['age'],
        'sex': data['sex'],
        'education': data['education'],
        'language': data['language'],
        'fluency_score': data['fluency_score']
    }

    # Load and use the model
    model1 = load_model(data_folder / 'Linear_reg_basic.sav')
    prediction = model1.predict([list(necessary_fields.values())])

    # Calculate additional metrics
    age = float(data['age'])
    delta_neurocogage_flu_weight = prediction - age
    necessary_fields['neurocog_age_flu_weight'] = prediction[0]

    model2 = load_model(data_folder / 'risk_dementia' / 'XGB_reg_basic.sav')
    model3 = load_model(data_folder / 'risk_handicap' / 'XGB_reg_basic.sav')

    # Example placeholders for additional predictions
    risk_dementia = pred_to_proba(model2.predict([list(necessary_fields.values())]))
    risk_handicap = pred_to_proba(model3.predict([list(necessary_fields.values())]))

    # Return the results
    return {
        'neurocog_age_flu_weight': int(prediction[0] * 100) / 100,
        'delta_neurocogage_flu_weight': int(delta_neurocogage_flu_weight[0] * 100) / 100,
        'risk_dementia': risk_dementia[0],
        'risk_handicap': risk_handicap[0]
    }


def predict_with_model_2(data):
    # Extract all fields for this model
    all_fields = {
        'age': data['age'],
        'sex': data['sex'],
        'education': data['education'],
        'language': data['language'],
        'fluency_score': data['fluency_score'],
        'handedness': data['handedness'],
        'nb_language': data['nb_language'],
        'hearing': data['hearing'],
        'moca': data['moca'],
        'ravlt_imm': data['ravlt_imm'],
        'ravlt_delay': data['ravlt_delay'],
        'logic_imm': data['logic_imm'],
        'logic_delay': data['logic_delay']
    }

    # Load and use the model
    model1 = load_model(data_folder / 'Linear_reg_all.sav')
    prediction = model1.predict([list(all_fields.values())])

    # Calculate additional metrics
    age = float(data['age'])
    delta_neurocogage_flu_weight = prediction - age
    all_fields['neurocog_age_flu_weight'] = prediction[0]

    model2 = load_model(data_folder / 'risk_dementia' / 'XGB_reg_all.sav')
    model3 = load_model(data_folder / 'risk_handicap' / 'XGB_reg_all.sav')

    # Example placeholders for additional predictions
    risk_dementia = pred_to_proba(model2.predict([list(all_fields.values())]))
    risk_handicap = pred_to_proba(model3.predict([list(all_fields.values())]))

    # Return the results
    return {
        'neurocog_age_flu_weight': int(prediction[0] * 100) / 100,
        'delta_neurocogage_flu_weight': int(delta_neurocogage_flu_weight[0] * 100) / 100,
        'risk_dementia': risk_dementia[0],
        'risk_handicap': risk_handicap[0]
    }


def predict_with_model_3(data):
    # Extract all fields for this model
    all_fields_plus_plus = {
        'age': data['age'],
        'sex': data['sex'],
        'education': data['education'],
        'language': data['language'],
        'fluency_score': data['fluency_score'],
        'handedness': data['handedness'],
        'nb_language': data['nb_language'],
        'hearing': data['hearing'],
        'moca': data['moca'],
        'ravlt_imm': data['ravlt_imm'],
        'ravlt_delay': data['ravlt_delay'],
        'logic_imm': data['logic_imm'],
        'logic_delay': data['logic_delay'],
        'hist_demence_fam': data['hist_demence_fam'],
        'hist_demence_parent': data['hist_demence_parent'],
        'living_alone': data['living_alone'],
        'income': data['income'],
        'retired': data['retired'],
        'stroke': data['stroke'],
        'tbi': data['tbi'],
        'hta': data['hta'],
        'diab_type2': data['diab_type2'],
        'obesity': data['obesity'],
        'depression': data['depression'],
        'anxiety': data['anxiety'],
        'smoking': data['smoking'],
        'alcohol': data['alcohol'],
        'poly_pharm5': data['poly_pharm5'],
        'physical_activity': data['physical_activity'],
        'social_life': data['social_life'],
        'cognitive_activities': data['cognitive_activities'],
        'nutrition_score': data['nutrition_score'],
        'sleep_deprivation': data['sleep_deprivation']
    }

    # Load and use the model
    model1 = load_model(data_folder / 'Linear_reg_all_plus_plus.sav')
    prediction = model1.predict([list(all_fields_plus_plus.values())])

    # Calculate additional metrics
    age = float(data['age'])
    delta_neurocogage_flu_weight = prediction - age
    all_fields_plus_plus['neurocog_age_flu_weight'] = prediction[0]

    model2 = load_model(data_folder / 'risk_dementia' / 'XGB_reg_all_plus_plus.sav')
    model3 = load_model(data_folder / 'risk_handicap' / 'XGB_reg_all_plus_plus.sav')

    # Example placeholders for additional predictions
    risk_dementia = pred_to_proba(model2.predict([list(all_fields_plus_plus.values())]))
    risk_handicap = pred_to_proba(model3.predict([list(all_fields_plus_plus.values())]))

    # Return the results
    return {
        'neurocog_age_flu_weight': int(prediction[0] * 100) / 100,
        'delta_neurocogage_flu_weight': int(delta_neurocogage_flu_weight[0] * 100) / 100,
        'risk_dementia': risk_dementia[0],
        'risk_handicap': risk_handicap[0]
    }