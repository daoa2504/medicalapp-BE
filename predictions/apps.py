import os
from django.apps import AppConfig


class PredictionsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictions'

    def ready(self):
        # Evaluer les modeles au demarrage (seulement dans le processus principal)
        if os.environ.get('RUN_MAIN') == 'true':
            try:
                from .evaluate_models import run_evaluation
                run_evaluation()
            except Exception as e:
                print(f"Erreur evaluation: {e}")
