import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
from src.model.train import train_model
from src.explainability.shap_analysis import run_shap_analysis


def main():
    print("Step 1: Training the model...")
    train_model()

    print("\nStep 2: Running SHAP analysis...")
    explanation = run_shap_analysis()

    print("\nStep 3: Model Explanation:")
    print(explanation)


if __name__ == "__main__":
    main()