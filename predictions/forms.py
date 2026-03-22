from django import forms
from django.db.models.fields import BLANK_CHOICE_DASH

class PredictionForm(forms.Form):
    # Existing fields
    identifier = forms.CharField(label='Identifiant', max_length=100, required=True)
    age = forms.FloatField(label='Âge', required=True, min_value=0)
    sex = forms.ChoiceField(
        label='Sexe',
        choices=BLANK_CHOICE_DASH + [(0, 'Femme'), (1, 'Homme')],
        required=True
    )
    education = forms.FloatField(label='Scolarité', required=True, min_value=0)
    language = forms.ChoiceField(
        label='Langue maternelle',
        choices=BLANK_CHOICE_DASH + [(1, 'Français'), (0, 'Anglais')],
        required=True
    )
    fluency_score = forms.FloatField(label='Score fluence catégorielle', required=True)

    # Optional fields (existing)
    handedness = forms.ChoiceField(
        label='Latéralité manuelle',
        choices=BLANK_CHOICE_DASH + [(0, 'Droitier'), (1, 'Gaucher ou ambidextre')],
        required=False
    )
    nb_language = forms.FloatField(label='Bi/Multi-linguisme', required=False, min_value=0)
    hearing = forms.ChoiceField(
        label='Audition',
        choices=BLANK_CHOICE_DASH + [(0, 'Non'), (1, 'Oui')],
        required=False
    )
    moca = forms.FloatField(label='Score MoCA', required=False, min_value=0)
    ravlt_imm = forms.FloatField(label='Score RAVLT immédiat', required=False, min_value=0)
    ravlt_delay = forms.FloatField(label='Score RAVLT différé', required=False, min_value=0)
    logic_imm = forms.FloatField(label='Score Mémoire Logique immédiat', required=False, min_value=0)
    logic_delay = forms.FloatField(label='Score Mémoire Logique différé', required=False, min_value=0)

    # New risk factors
    hist_demence_fam = forms.ChoiceField(
        label='Historique de démence familiale',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    hist_demence_parent = forms.ChoiceField(
        label='Historique de démence parentale',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    living_alone = forms.ChoiceField(
        label='Vit seul(e)',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    income = forms.ChoiceField(
        label='Revenu',
        choices=[(0, 'Faible'), (1, 'Élevé')],
        widget=forms.RadioSelect,
        required=False
    )
    retired = forms.ChoiceField(
        label='Retraité(e)',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    stroke = forms.ChoiceField(
        label='Antécédent d\'AVC',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    tbi = forms.ChoiceField(
        label='Antécédent de traumatisme crânien',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    hta = forms.ChoiceField(
        label='Hypertension artérielle',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    diab_type2 = forms.ChoiceField(
        label='Diabète de type 2',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    obesity = forms.ChoiceField(
        label='Obésité',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    depression = forms.ChoiceField(
        label='Dépression',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    anxiety = forms.ChoiceField(
        label='Anxiété',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    smoking = forms.ChoiceField(
        label='Tabagisme',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    alcohol = forms.ChoiceField(
        label='Consommation d\'alcool',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    poly_pharm5 = forms.ChoiceField(
        label='Polypharmacie (≥5 médicaments)',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    physical_activity = forms.ChoiceField(
        label='Activité physique',
        choices=[(0, 'Sédentaire'), (1, 'Active')],
        widget=forms.RadioSelect,
        required=False
    )
    social_life = forms.ChoiceField(
        label='Vie sociale',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    cognitive_activities = forms.ChoiceField(
        label='Activités cognitives',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )
    nutrition_score = forms.ChoiceField(
        label='Score de nutrition',
        choices=[(0, 'Faible'), (1, 'Élevé')],
        widget=forms.RadioSelect,
        required=False
    )
    sleep_deprivation = forms.ChoiceField(
        label='Privation de sommeil',
        choices=[(0, 'Non'), (1, 'Oui')],
        widget=forms.RadioSelect,
        required=False
    )

    def clean(self):
        cleaned_data = super().clean()

        # Handle optional fields and replace None with 0
        optional_fields = [
            'handedness', 'nb_language', 'hearing', 'moca', 'ravlt_imm', 'ravlt_delay', 'logic_imm', 'logic_delay',
            'hist_demence_fam', 'hist_demence_parent', 'living_alone', 'income', 'retired', 'stroke', 'tbi', 'hta',
            'diab_type2', 'obesity', 'depression', 'anxiety', 'smoking', 'alcohol', 'poly_pharm5', 'physical_activity',
            'social_life', 'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
        ]
        for field in optional_fields:
            value = cleaned_data.get(field)
            if value in [None, '']:
                cleaned_data[field] = 0
            elif value == '-':
                cleaned_data[field] = 0

        return cleaned_data
