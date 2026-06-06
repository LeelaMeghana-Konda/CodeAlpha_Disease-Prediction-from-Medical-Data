import os
import joblib
import pandas as pd
from data.data_loader import load_all, DATASETS
from models.classifiers import train_all
from models.evaluator import evaluate_all, results_to_dataframe
from utils.preprocessor import split_and_scale

MODEL_DIR = "saved_models"

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Loading datasets...")
    all_data = load_all()

    for ds_key, (X, y, feature_names) in all_data.items():
        print(f"\n--- Processing {ds_key} ---")
        
        # --- FEATURE REDUCTION ---
        # Reduce features to handle user request
        reduce_map = {"heart_disease": 8, "diabetes": 6, "breast_cancer": 10}
        n_top = reduce_map.get(ds_key, len(feature_names))
        
        if n_top < len(feature_names):
            from sklearn.ensemble import RandomForestClassifier
            print(f"Reducing features from {len(feature_names)} to {n_top}...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = importances.argsort()[-n_top:][::-1]
            
            # Update X and feature_names
            X = X.iloc[:, indices]
            feature_names = [feature_names[i] for i in indices]
            print(f"Selected: {', '.join(feature_names)}")

        # 1. Split & scale
        X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

        # 2. Train all models
        fitted_models = train_all(X_train, y_train)

        # 3. Evaluate and save all models
        eval_results = evaluate_all(fitted_models, X_test, y_test)
        
        # Save each model
        for model_name, model in fitted_models.items():
            # Create a safe name for the file
            safe_name = model_name.lower().replace(" ", "_")
            model_path = os.path.join(MODEL_DIR, f"{ds_key}_{safe_name}.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")

        # Save scaler and feature names (common for all models of this disease)
        scaler_path = os.path.join(MODEL_DIR, f"{ds_key}_scaler.joblib")
        features_path = os.path.join(MODEL_DIR, f"{ds_key}_features.json")

        joblib.dump(scaler, scaler_path)
        
        import json
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)

        print(f"Saved scaler to {scaler_path}")
        print(f"Saved features to {features_path}")

    print("\nAll models (SVM, Logistic Regression, Random Forest, XGBoost) trained and saved successfully.")

if __name__ == "__main__":
    main()
