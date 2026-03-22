from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    """
    Serializer pour les données d'entrée de prédiction
    Basé sur PredictionForm
    """
    # Champs obligatoires
    identifier = serializers.CharField(
        max_length=100,
        required=True,
        help_text="Identifiant du patient"
    )
    age = serializers.FloatField(
        required=True,
        min_value=0,
        help_text="Âge du patient"
    )
    sex = serializers.ChoiceField(
        choices=[(0, 'Femme'), (1, 'Homme')],
        required=True,
        help_text="Sexe (0=Femme, 1=Homme)"
    )
    education = serializers.FloatField(
        required=True,
        min_value=0,
        help_text="Années de scolarité"
    )
    language = serializers.ChoiceField(
        choices=[(0, 'Anglais'), (1, 'Français')],
        required=True,
        help_text="Langue maternelle (0=Anglais, 1=Français)"
    )
    fluency_score = serializers.FloatField(
        required=True,
        help_text="Score de fluence catégorielle"
    )

    # Champs optionnels - Données neuropsychologiques
    handedness = serializers.ChoiceField(
        choices=[(0, 'Droitier'), (1, 'Gaucher ou ambidextre')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Latéralité manuelle (0=Droitier, 1=Gaucher/ambidextre)"
    )
    nb_language = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=0,
        help_text="Nombre de langues parlées"
    )
    hearing = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Problème d'audition (0=Non, 1=Oui)"
    )
    moca = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=0,
        help_text="Score MoCA (Montreal Cognitive Assessment)"
    )
    ravlt_imm = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=0,
        help_text="Score RAVLT immédiat (Rey Auditory Verbal Learning Test)"
    )
    ravlt_delay = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=0,
        help_text="Score RAVLT différé"
    )
    logic_imm = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=0,
        help_text="Score Mémoire Logique immédiat"
    )
    logic_delay = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=0,
        help_text="Score Mémoire Logique différé"
    )

    # Champs optionnels - Facteurs de risque génétiques et démographiques
    hist_demence_fam = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Historique de démence familiale (0=Non, 1=Oui)"
    )
    hist_demence_parent = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Historique de démence parentale (0=Non, 1=Oui)"
    )
    living_alone = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Vit seul(e) (0=Non, 1=Oui)"
    )
    income = serializers.ChoiceField(
        choices=[(0, 'Faible'), (1, 'Élevé')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Niveau de revenu (0=Faible, 1=Élevé)"
    )
    retired = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Retraité(e) (0=Non, 1=Oui)"
    )

    # Champs optionnels - Facteurs de risque médicaux
    stroke = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Antécédent d'AVC (0=Non, 1=Oui)"
    )
    tbi = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Antécédent de traumatisme crânien (0=Non, 1=Oui)"
    )
    hta = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Hypertension artérielle (0=Non, 1=Oui)"
    )
    diab_type2 = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Diabète de type 2 (0=Non, 1=Oui)"
    )
    obesity = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Obésité (0=Non, 1=Oui)"
    )
    depression = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Dépression (0=Non, 1=Oui)"
    )
    anxiety = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Anxiété (0=Non, 1=Oui)"
    )
    smoking = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Tabagisme (0=Non, 1=Oui)"
    )
    alcohol = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Consommation d'alcool (0=Non, 1=Oui)"
    )
    poly_pharm5 = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Polypharmacie ≥5 médicaments (0=Non, 1=Oui)"
    )

    # Champs optionnels - Style de vie
    physical_activity = serializers.ChoiceField(
        choices=[(0, 'Sédentaire'), (1, 'Active')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Niveau d'activité physique (0=Sédentaire, 1=Active)"
    )
    social_life = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Vie sociale active (0=Non, 1=Oui)"
    )
    cognitive_activities = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Activités cognitives régulières (0=Non, 1=Oui)"
    )
    nutrition_score = serializers.ChoiceField(
        choices=[(0, 'Faible'), (1, 'Élevé')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Score de nutrition (0=Faible, 1=Élevé)"
    )
    sleep_deprivation = serializers.ChoiceField(
        choices=[(0, 'Non'), (1, 'Oui')],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Privation de sommeil (0=Non, 1=Oui)"
    )

    def validate(self, data):
        """
        Nettoie les données comme dans PredictionForm.clean()
        Remplace None, '', et '-' par 0 pour les champs optionnels
        """
        optional_fields = [
            'handedness', 'nb_language', 'hearing', 'moca', 'ravlt_imm', 'ravlt_delay',
            'logic_imm', 'logic_delay', 'hist_demence_fam', 'hist_demence_parent',
            'living_alone', 'income', 'retired', 'stroke', 'tbi', 'hta',
            'diab_type2', 'obesity', 'depression', 'anxiety', 'smoking', 'alcohol',
            'poly_pharm5', 'physical_activity', 'social_life', 'cognitive_activities',
            'nutrition_score', 'sleep_deprivation'
        ]

        for field in optional_fields:
            value = data.get(field)
            if value in [None, '', '-']:
                data[field] = 0

        # Convertir les ChoiceFields en entiers
        choice_fields = [
            'sex', 'language', 'handedness', 'hearing', 'hist_demence_fam',
            'hist_demence_parent', 'living_alone', 'income', 'retired', 'stroke',
            'tbi', 'hta', 'diab_type2', 'obesity', 'depression', 'anxiety',
            'smoking', 'alcohol', 'poly_pharm5', 'physical_activity', 'social_life',
            'cognitive_activities', 'nutrition_score', 'sleep_deprivation'
        ]

        for field in choice_fields:
            if field in data and data[field] is not None:
                try:
                    data[field] = int(data[field])
                except (ValueError, TypeError):
                    pass

        return data


class PredictionOutputSerializer(serializers.Serializer):
    """
    Serializer pour les résultats de prédiction
    """
    identifier = serializers.CharField(
        help_text="Identifiant du patient"
    )
    neurocog_age_flu_weight = serializers.FloatField(
        help_text="Âge neurocognitif calculé"
    )
    delta_neurocogage_flu_weight = serializers.FloatField(
        help_text="Delta entre âge neurocognitif et âge réel"
    )
    risk_dementia = serializers.FloatField(
        help_text="Risque de démence (0-1)"
    )
    risk_handicap = serializers.FloatField(
        help_text="Risque de handicap (0-1)"
    )
    model_type = serializers.CharField(
        required=False,
        help_text="Type de modèle utilisé (model_1, model_2, ou model_3)"
    )


class ModelTypeSerializer(serializers.Serializer):
    """
    Serializer pour sélectionner le type de modèle
    """
    model_type = serializers.ChoiceField(
        choices=[
            ('model_1', 'Modèle basique (fluence seulement)'),
            ('model_2', 'Modèle complet (avec tests neuropsychologiques)'),
            ('model_3', 'Modèle avancé (avec facteurs de risque)')
        ],
        default='model_3',
        help_text="Type de modèle à utiliser pour la prédiction"
    )