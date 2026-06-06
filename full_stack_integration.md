# Full Stack Integration: Disease Prediction System

The project has been upgraded from a standalone ML pipeline to a full-stack web application. This enables real-time interaction and prediction visualization.

## Architecture Overview

### 1. Backend (Flask)
- **Model Storage**: Models, Scalers, and Feature metadata are stored in `saved_models/`.
- **API Endpoints**:
    - `GET /api/features/<disease>`: Returns the list of features expected by the model.
    - `POST /api/predict/<disease>`: Accepts feature values, scales them, and returns prediction + probabilities.
- **Static Hosting**: Serves the frontend assets directly.

### 2. Frontend (Vanilla JS + CSS)
- **Premium Design**: Dark mode aesthetic with glassmorphism and background animations.
- **Dynamic UX**: Forms are generated on-the-fly based on model requirements.
- **Visualization**: Chart.js is used to display prediction confidence.

## File Structure Changes
- `train_and_save.py`: Utility to prepare models for production.
- `app.py`: Main Flask application entry point.
- `saved_models/`: Serialized artifacts (`.joblib` and `.json`).
- `static/`: Frontend assets (`index.html`, `style.css`, `script.js`).

## How to Run
1. Ensure dependencies are installed: `pip install flask flask-cors joblib scikit-learn xgboost pandas`
2. Run the application: `python app.py`
3. Open in browser: `http://127.0.0.1:5000`

## Implementation Details
- **Scaling Persistence**: Using the *exact* scaler fitted during training is critical for prediction accuracy. These are saved separately for each disease.
- **Dynamic Forms**: The UI doesn't hardcode input fields; it asks the backend what features are needed, making the system flexible for future model updates.
- **Aesthetics**: Custom CSS animations and blur filters provide a modern, medical-tech feel.
