````markdown
# Invent.ai Intelligence Engine – Hybrid Rating Predictor

This repository implements a hybrid machine learning system for predicting e-commerce product ratings from both free-text reviews and structured product/user metadata.

The solution combines:

- Transformer-based text encoders for semantic representation of reviews  
- Gradient boosting models (CatBoost, XGBoost, LightGBM) for tabular features  
- A stacking ensemble with a Ridge regression meta-model

The pipeline covers training, evaluation, REST API serving, and an interactive demo.

---

## 1. Overview

The problem is framed as **multimodal regression**:

- Text reviews are converted into dense semantic vectors using Transformer encoders.
- Structured fields (user_id, product_id, price, discount, time features, etc.) are engineered into numeric features.
- All features are fused into a single matrix and used to train a set of tree-based regressors.
- A stacking layer learns how to combine the base models into the final rating prediction.

Stack components:

- Text encoders: HuggingFace Transformer models (e.g. `all-mpnet-base-v2`, `emotion-english-distilroberta-base`)
- Dimensionality reduction: PCA on text embeddings
- Base regressors: CatBoost, XGBoost, LightGBM
- Meta-model: Ridge regression on out-of-fold base predictions
- Serving: FastAPI
- Demo UI: Streamlit

---

## 2. Repository Structure

Example layout:

```text
invent_solution/
├── data/
│   └── product_user_reviews.csv         # Training data
├── model_artifacts/
│   ├── catboost_tuned.cbm              # Trained CatBoost model
│   ├── xgb_model.pkl                   # Trained XGBoost model
│   ├── lgb_model.pkl                   # Trained LightGBM model
│   ├── stack_cat_model.cbm.pkl         # Stacking CatBoost
│   ├── stack_xgb_model.pkl             # Stacking XGBoost
│   ├── stack_lgb_model.pkl             # Stacking LightGBM
│   ├── stack_meta_ridge.pkl            # Ridge meta-model
│   ├── pca_base_64.pkl                 # PCA for base encoder
│   ├── pca_emotion_64.pkl              # PCA for emotion encoder
│   ├── label_encoders.pkl              # LabelEncoders for categorical features
│   ├── feature_order.pkl               # Column order for inference
│   └── catboost/plots/
│       ├── feature_importance.png
│       └── shap_summary.png
├── models/
│   ├── all-mpnet-base-v2/              # Local HF model files
│   └── emotion-english-distilroberta-base/
├── tests/
│   └── test_app.py                     # API tests
├── preprocessing.py                    # Central preprocessing / feature engineering
├── train_advanced.py                   # Training pipeline
├── app.py                              # FastAPI application
├── demo.py                             # Streamlit demo
├── Dockerfile                          # Docker configuration (API)
├── requirements.txt                    # Python dependencies
└── README.md
````

The training and serving code assumes the presence of:

* `preprocessing.py`
* `train_advanced.py`
* `app.py`
* `demo.py`
* `model_artifacts/`
* `models/`

---

## 3. Data and Feature Engineering

### 3.1 Input Data

The training script expects a CSV file (e.g. `data/product_user_reviews.csv`) with at least:

* `review_text` – raw text of the review
* `rating` – target variable (numeric rating)
* Product metadata (e.g. `product_id`, `product_category`, `price`)
* User metadata (e.g. `user_id`)
* Optional timestamp (e.g. `review_date`) for time-based features

### 3.2 Preprocessing (`preprocessing.py`)

A central function `DataPreprocessor.preprocess_dataframe(df)` is used in both training and inference. It:

* Cleans review text:

  * Lowercasing
  * Removing HTML tags
  * Removing URLs
  * Keeping punctuation relevant for sentiment (e.g. `!`, `?`, `.`)
* Computes text meta-features:

  * `word_count`
  * `char_count`
  * `caps_ratio` (ratio of uppercase characters)
  * `exclamation_count`
* Builds regex-based flags:

  * `has_positive_word`: presence of words like `great`, `excellent`, `amazing`, `love`, `highly recommend`
  * `has_negative_word`: presence of words like `bad`, `terrible`, `awful`, `disappointed`, `useless`, `broke`, `refund`
* Applies numeric transforms:

  * `price` sanitisation (no negatives)
  * `log_price` = `log1p(price)`
  * `discount_rate` clipped to `[0, 1]`
* Encodes temporal features (if `review_date` is available):

  * `review_year`
  * `review_month`
  * `is_weekend`
  * `month_sin`, `month_cos` (cyclic encoding of month)

Feature engineering is centralised so the same logic is reused in `train_advanced.py` and `app.py`, avoiding train–serve skew.

---

## 4. Model Architecture

### 4.1 Text Encoding

Two Transformer encoders are used:

* Base encoder (e.g. `all-mpnet-base-v2`) for general semantics
* Emotion-focused encoder (e.g. `emotion-english-distilroberta-base`) for affective information

For each encoder:

1. Tokenise the review text.
2. Run the model to obtain token-level hidden states.
3. Apply mean pooling over tokens with attention masking.
4. L2-normalise the resulting sentence embedding.

This yields two high-dimensional vectors per review (one per encoder).

### 4.2 Dimensionality Reduction (PCA)

Raw embeddings are high-dimensional, so PCA is applied separately for each encoder:

* `pca_base_64.pkl` – PCA trained on base embeddings
* `pca_emotion_64.pkl` – PCA trained on emotion embeddings

At training time:

* Embeddings are computed for all reviews.
* PCA is fitted for each encoder’s output.
* Reduced embeddings are concatenated (e.g. 64 + 64 = 128-dimensional vector).

The PCA models are stored under `model_artifacts/` and reused at inference.

### 4.3 Tabular Features

In addition to embeddings, the model uses:

* Categorical features:

  * `user_id`
  * `product_id`
  * `main_category`
* Numeric features:

  * `log_price`
  * `discount_rate`
  * `word_count`
  * `char_count`
  * `caps_ratio`
  * `exclamation_count`
  * `has_positive_word`
  * `has_negative_word`
* Date/time features:

  * `review_year`
  * `month_sin`
  * `month_cos`
  * `is_weekend`
* Simple product-level statistic:

  * `product_review_count`: number of reviews seen per product in training
    (For unseen products at inference time, this defaults to 0.)

All of these, together with the PCA-reduced embeddings, form the final feature matrix.

### 4.4 Base Models and Stacking

Three base regressors are trained:

* `CatBoostRegressor`
* `XGBRegressor` (XGBoost)
* `LGBMRegressor` (LightGBM)

An out-of-fold stacking procedure is used:

1. Apply K-Fold split (e.g. K=5) on the training set.
2. For each fold:

   * Train CatBoost, XGBoost, and LightGBM on the train folds.
   * Predict on the validation fold to produce out-of-fold (OOF) predictions.
3. Concatenate OOF predictions into a stacking feature matrix of shape `[n_samples, 3]`.

A **Ridge regression** meta-model is then trained on these OOF predictions to learn how to weight the three base models.

After meta-model training:

* Base models are retrained on the full training set.
* At inference:

  * Each base model predicts a rating.
  * These three predictions are passed to the Ridge meta-model.
  * The meta-model returns the final predicted rating (optionally clipped to `[1.0, 10.0]`).

---

## 5. Training Pipeline (`train_advanced.py`)

The training script follows these steps:

1. **Load and preprocess data**

   * Read `data/product_user_reviews.csv`.
   * Run `DataPreprocessor.preprocess_dataframe`.
   * Compute embeddings using the Transformer models.
   * Fit PCA on both encoder outputs and reduce the embeddings.
   * Concatenate engineered tabular features and reduced embeddings.

2. **Train–test split**

   * Split into train and test sets (e.g. 80/20).
   * Save the feature order (`feature_order.pkl`) to ensure consistent column order at inference.

3. **Product-level statistics**

   * Compute `product_review_count` based only on the training set.
   * Map these counts onto train and test sets without leaking test labels.

4. **Label encoding for XGBoost/LightGBM**

   * Fit `LabelEncoder` on each categorical column in the training set.
   * Transform train and test.
   * Unseen categories in the test set receive a reserved code (e.g. `-1`).
   * Save encoders (`label_encoders.pkl`).

5. **Baseline**

   * Use the mean training rating as a baseline predictor.
   * Evaluate RMSE, MAE, R² on the test set.

6. **Hyperparameter tuning (CatBoost + Optuna)**

   * Run an Optuna study to tune CatBoost hyperparameters.
   * Use an eval set and early stopping based on validation metrics (e.g. MAE).
   * Retrieve the best parameter set.

7. **Final CatBoost model**

   * Train a CatBoostRegressor with the best parameters on the train split.
   * Evaluate on the test split (RMSE, MAE, R²).
   * Save the tuned model (`catboost_tuned.cbm`).

8. **XGBoost and LightGBM**

   * Train XGBoost and LightGBM on the label-encoded features.
   * Evaluate on the test split.
   * Save models as `.pkl` files.

9. **OOF stacking ensemble**

   * Run the OOF stacking procedure on the training portion.
   * Train the Ridge meta-model on OOF predictions.
   * Retrain the base models on the full training data.
   * Evaluate the stacked ensemble on the test set.

10. **Explainability**

    * Plot CatBoost feature importance.
    * Optionally plot learning curves if eval metrics are available.
    * Run SHAP analysis on a sample of the test set and save a summary plot.

11. **Artifact saving**

    * Save:

      * Base models (CatBoost, XGBoost, LightGBM)
      * Stacking base models + Ridge meta-model
      * PCA models
      * Label encoders
      * Feature order
      * Best CatBoost hyperparameters
    * All under `model_artifacts/`.

---

## 6. Serving Architecture (`app.py` – FastAPI)

The FastAPI application loads all required artifacts and serves predictions over HTTP.

### 6.1 Startup

On startup, the app:

* Loads PCA models (`pca_base_64.pkl`, `pca_emotion_64.pkl`)
* Loads feature order (`feature_order.pkl`)
* Loads label encoders (`label_encoders.pkl`)
* Loads stacking base models and the Ridge meta-model:

  * `stack_cat_model.cbm.pkl`
  * `stack_xgb_model.pkl`
  * `stack_lgb_model.pkl`
  * `stack_meta_ridge.pkl`
* Loads the Transformer encoders from `models/`
* Sets an internal flag indicating whether the model stack is ready

### 6.2 Health Endpoint

`GET /health`

Returns a JSON response indicating service status, for example:

```json
{ "status": "healthy" }
```

or

```json
{ "status": "not_ready" }
```

depending on whether the required artifacts were loaded successfully.

### 6.3 Prediction Endpoint

`POST /predict`

Example request:

```json
{
  "review_text": "Excellent quality, fast shipping!",
  "user_id": "A123456789",
  "product_id": "B987654321",
  "product_category": "Electronics",
  "price": 150.0,
  "discount_rate": 0.10
}
```

High-level steps inside `/predict`:

1. Validate and parse the request with Pydantic.
2. If the model is not ready (artifacts missing), return HTTP 503.
3. Construct a one-row DataFrame from the request.
4. Run `DataPreprocessor.preprocess_dataframe`.
5. Compute text embeddings using the Transformer models.
6. Apply PCA transformations and concatenate embeddings with tabular features.
7. Align columns with `feature_order` and fill any missing feature columns with defaults.
8. Apply label encoders to categorical columns.
9. Obtain predictions from each base model.
10. Pass the three base predictions to the Ridge meta-model.
11. Clip the final rating into `[1.0, 10.0]`.
12. Optionally derive a simple sentiment label from the text or rating.
13. Return a JSON response such as:

```json
{
  "predicted_rating": 9.25,
  "sentiment_analysis": "Positive"
}
```

On errors (e.g. missing artifacts), the endpoint returns HTTP 503 with a short message.

---

## 7. Streamlit Demo (`demo.py`)

The Streamlit application provides an interactive UI for testing the model.

Typical behaviour:

* Sidebar inputs:

  * Review text
  * Price
  * Discount rate
  * Product category (select box)
  * Optional user_id and product_id
* On button press:

  * Calls the API health endpoint.
  * Sends a POST request to `/predict`.
  * Displays:

    * Predicted rating as a `st.metric`
    * A sentiment label
    * A progress bar indicating rating magnitude
    * Raw JSON response for debugging under an expander

The API URL can be configured (e.g. via environment variable), defaulting to `http://localhost:8000`.

---

## 8. Running the Project Locally

### 8.1 Prerequisites

* Python 3.9+
* `pip`
* (Optional) Docker

### 8.2 Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 8.3 Train the Models

Run the training pipeline:

```bash
python train_advanced.py
```

After this step, trained models and auxiliary artifacts will be saved under `model_artifacts/`.

### 8.4 Start the API

Start the FastAPI server with Uvicorn:

```bash
uvicorn app:app --reload
```

Useful endpoints:

* API docs (Swagger UI): `http://localhost:8000/docs`
* Health check: `http://localhost:8000/health`

### 8.5 Run the Streamlit Demo

In a separate terminal:

```bash
streamlit run demo.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`) and send example reviews through the UI.

---

## 9. Docker

A minimal Docker configuration for the API:

```dockerfile
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]
```

Build and run:

```bash
docker build -t invent-engine .
docker run -p 8000:8000 \
  -v $(pwd)/model_artifacts:/app/model_artifacts \
  -v $(pwd)/models:/app/models \
  invent-engine
```

This expects the trained artifacts and local Transformer checkpoints to be available under `model_artifacts/` and `models/` respectively.

---

## 10. Testing

The `tests/` folder contains basic tests using `pytest` and FastAPI’s `TestClient`. They cover:

* `/health` endpoint behaviour
* Prediction flow when model artifacts exist
* Proper handling of missing models (HTTP 503)
* Validation errors (e.g. empty review text, negative price)
* Cold-start scenarios (unseen user/product IDs)

Run tests from the project root:

```bash
pytest tests/
```

---

## 11. Possible Extensions

Potential extensions:

* Additional text encoders or domain-specific language models
* More advanced product/user statistics or graph-based features
* Calibration of rating outputs
* Logging and monitoring (latency, error rates, drift indicators)
* Extra endpoints for per-request explainability (e.g. SHAP values)

The current design keeps preprocessing, feature engineering, training, and serving modular, so changes can be made in a targeted way without affecting the entire stack.

```
```
