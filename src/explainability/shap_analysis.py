import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap
from tensorflow.keras.models import load_model


FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


def _flat(x):
    return np.asarray(x[0] if isinstance(x, list) else x).squeeze()


def _shap(model, background, X, patient):
    for cls in (shap.DeepExplainer, shap.GradientExplainer):
        try:
            explainer = cls(model, background)
            return explainer, _flat(explainer.shap_values(X)), _flat(explainer.shap_values(patient))
        except Exception:
            pass
    explainer = shap.KernelExplainer(lambda x: model.predict(x, verbose=0).ravel(), background)
    return explainer, _flat(explainer.shap_values(X, nsamples=100)), _flat(explainer.shap_values(patient, nsamples=100))


def interpret_shap(shap_values, patient_shap_values, patient, prediction):
    global_top = np.abs(shap_values).mean(0).argsort()[-3:][::-1]
    local_top = np.abs(patient_shap_values).argsort()[-3:][::-1]
    return (
        f"Globally, the ANN is driven most by {', '.join(FEATURES[i] for i in global_top)}. "
        f"For this patient, predicted diabetes risk is {prediction:.3f}; the strongest contributors are "
        f"{', '.join(f'{FEATURES[i]}={patient[i]:.3f} ({patient_shap_values[i]:+.3f})' for i in local_top)}."
    )


def run_shap_analysis():
    os.makedirs('outputs/results', exist_ok=True)
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    model = load_model('outputs/models/trained_model.h5', compile=False)
    background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
    X_global = shap.sample(X_test, min(100, len(X_test)), random_state=42)
    explainer, values, patient_values = _shap(model, background, X_global, X_test[:1])

    shap.summary_plot(values, X_global, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig('outputs/results/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    base = getattr(explainer, 'expected_value', model.predict(background, verbose=0).mean())
    shap.plots.waterfall(shap.Explanation(values=patient_values, base_values=float(_flat(base)), data=X_test[0], feature_names=FEATURES), show=False)
    plt.tight_layout()
    plt.savefig('outputs/results/shap_patient_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()

    return interpret_shap(values, patient_values, X_test[0], float(model.predict(X_test[:1], verbose=0).ravel()[0]))


if __name__ == '__main__':
    run_shap_analysis()
