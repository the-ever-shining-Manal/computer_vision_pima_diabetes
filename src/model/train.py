import numpy as np
import os
from src.model.ann_model import build_ann_model

def train_model():
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')

    model = build_ann_model(input_dim=X_train.shape[1])

    print("Starting model training...")
    model.fit(X_train, y_train, 
              epochs=50, 
              batch_size=10, 
              validation_split=0.2, 
              verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    os.makedirs('outputs/models', exist_ok=True)
    model.save('outputs/models/trained_model.h5')
    print("Model saved to outputs/models/trained_model.h5")

    os.makedirs('outputs/results', exist_ok=True)
    with open('outputs/results/metrics.txt', 'w') as f:
        f.write(f"Final Test Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Final Test Loss: {loss:.4f}")
    print("Metrics saved to outputs/results/metrics.txt")

if __name__ == "__main__":
    train_model()
