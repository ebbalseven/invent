import os
import pandas as pd
import numpy as np
import joblib
import logging
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from preprocessing import DataPreprocessor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdvancedTrainer")
ARTIFACTS_DIR = "model_artifacts"
PLOTS_DIR = "model_artifacts/catboost/plots"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1) Base encoder: all-mpnet-base-v2
BASE_MODEL_PATH = "models/all-mpnet-base-v2"
_base_config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
_base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
_base_model = AutoModel.from_pretrained(
    BASE_MODEL_PATH,
    config=_base_config,
    use_safetensors=True,
    trust_remote_code=False
)
_base_model.to(DEVICE)
_base_model.eval()

# 2) Emotion encoder: j-hartmann/emotion-english-distilroberta-base
EMOTION_MODEL_PATH = "models/emotion-english-distilroberta-base"
_emotion_config = AutoConfig.from_pretrained(EMOTION_MODEL_PATH)
_emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
_emotion_model = AutoModel.from_pretrained(
    EMOTION_MODEL_PATH,
    config=_emotion_config,
    use_safetensors=False,
    trust_remote_code=False
)
_emotion_model.to(DEVICE)
_emotion_model.eval()



def oof_stacking_ensemble(X, y, cat_features, n_splits=5, random_state=42):
    """
    CatBoost + XGBoost + LightGBM için 5-fold OOF stacking ensemble.
    Dönüş:
      - base_models: dict (cat, xgb, lgb tam train edilmiş modeller)
      - meta_model: Ridge meta-öğrenici
    """

    logger.info(f"{n_splits}-Fold OOF stacking başlatılıyor...")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_cat = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))

    fold_idx = 0
    for train_idx, valid_idx in kf.split(X):
        fold_idx += 1
        logger.info(f"Fold {fold_idx}/{n_splits}...")

        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=800,
            depth=8,
            learning_rate=0.05,
            loss_function='RMSE',
            eval_metric='RMSE',
            verbose=False,
            allow_writing_files=False,
            cat_features=[],
            task_type='GPU'
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        oof_cat[valid_idx] = cat_model.predict(X_val)

        # XGBoost
        xgb_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[valid_idx] = xgb_model.predict(X_val)

        # LightGBM
        lgb_model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            boosting_type="gbdt",
            random_state=random_state,
            n_jobs=-1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        oof_lgb[valid_idx] = lgb_model.predict(X_val)

    # Meta-model için OOF feature matrisi
    stack_X = np.vstack([oof_cat, oof_xgb, oof_lgb]).T  # [n_samples, 3]
    logger.info("Meta-model (Ridge) eğitiliyor...")
    meta_model = Ridge(alpha=1.0, random_state=random_state)
    meta_model.fit(stack_X, y)

    # Meta-model performansını OOF üzerinde ölçme
    stack_oof_pred = meta_model.predict(stack_X)
    oof_rmse = np.sqrt(mean_squared_error(y, stack_oof_pred))
    oof_mae = mean_absolute_error(y, stack_oof_pred)
    oof_r2 = r2_score(y, stack_oof_pred)
    logger.info(f"OOF Stacking Ensemble -> RMSE={oof_rmse:.4f}, MAE={oof_mae:.4f}, R2={oof_r2:.4f}")


    logger.info("Base modeller full train verisiyle yeniden eğitiliyor (prod için)...")

    # Full CatBoost
    full_cat = CatBoostRegressor(
        iterations=800,
        depth=8,
        learning_rate=0.05,
        loss_function='RMSE',
        eval_metric='RMSE',
        verbose=False,
        allow_writing_files=False,
        cat_features=[],
        task_type='GPU'
    )
    full_cat.fit(X, y)

    # Full XGB
    full_xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )
    full_xgb.fit(X, y)

    # Full LGBM
    full_lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=random_state,
        n_jobs=-1
    )
    full_lgb.fit(X, y)

    base_models = {
        "cat": full_cat,
        "xgb": full_xgb,
        "lgb": full_lgb
    }

    return base_models, meta_model


def _mean_pooling(model_output, attention_mask):
    """
    Sentence-Transformers'in default mean pooling mantığı:
    last_hidden_state -> attention mask'e göre ortalama
    """
    token_embeddings = model_output.last_hidden_state  # [batch, seq, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / sum_mask


def encode_texts_base(texts, batch_size=32):
    """
    Base encoder (all-mpnet-base-v2) için embedding üretir.
    return: np.array, shape = (n_samples, hidden_dim_base)
    """
    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = _base_tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = _base_model(**enc)

        sent_emb = _mean_pooling(out, enc["attention_mask"])
        sent_emb = F.normalize(sent_emb, p=2, dim=1)

        all_embeddings.append(sent_emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings.numpy()


def encode_texts_emotion(texts, batch_size=32):
    """
    Emotion encoder (emotion-english-distilroberta-base) için embedding üretir.
    return: np.array, shape = (n_samples, hidden_dim_emotion)
    """
    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = _emotion_tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = _emotion_model(**enc)

        sent_emb = _mean_pooling(out, enc["attention_mask"])
        sent_emb = F.normalize(sent_emb, p=2, dim=1)

        all_embeddings.append(sent_emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings.numpy()



class ModelEvaluator:
    """Model performansını görselleştiren ve raporlayan sınıf."""

    @staticmethod
    def plot_feature_importance(model, feature_names):
        """Özellik önem düzeylerini görselleştirir."""
        if not hasattr(model, 'get_feature_importance'):
            return

        importance = model.get_feature_importance()

        # En önemli 20 özellik
        indices = np.argsort(importance)[::-1][:20]

        plt.figure(figsize=(12, 8))
        plt.title("Top 20 Feature Importance")
        plt.bar(range(len(indices)), importance[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/feature_importance.png")
        plt.close()
        logger.info(f"Grafik kaydedildi: {PLOTS_DIR}/feature_importance.png")

    @staticmethod
    def plot_learning_curve(model):
        """Eğitim ve Doğrulama hatasının değişimini çizer."""
        if not hasattr(model, 'get_evals_result'):
            return

        results = model.get_evals_result()
        if 'validation' not in results or 'RMSE' not in results['validation']:
            logger.warning("Validation sonuçları bulunamadı, grafik çizilemiyor.")
            return

        epochs = len(results['validation']['RMSE'])
        x_axis = range(0, epochs)

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results['learn']['RMSE'], label='Train')
        plt.plot(x_axis, results['validation']['RMSE'], label='Validation')
        plt.legend()
        plt.ylabel('RMSE')
        plt.title('CatBoost Training Process')
        plt.savefig(f"{PLOTS_DIR}/learning_curve.png")
        plt.close()
        logger.info(f"Grafik kaydedildi: {PLOTS_DIR}/learning_curve.png")

    @staticmethod
    def explain_with_shap(model, X_sample):
        """SHAP analizi ile modelin kararlarını açıklar."""
        logger.info("SHAP analizi çalıştırılıyor...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(f"{PLOTS_DIR}/shap_summary.png")
            plt.close()
            logger.info(f"SHAP grafiği kaydedildi: {PLOTS_DIR}/shap_summary.png")
        except Exception as e:
            logger.warning(f"SHAP analizi sırasında hata: {e}")


def prepare_data():
    logger.info("Veri yükleniyor ve temizleniyor...")
    if not os.path.exists('data/product_user_reviews.csv'):
        raise FileNotFoundError("Data dosyası bulunamadı!")

    df = pd.read_csv('data/product_user_reviews.csv')

    # 1. Preprocess
    df = DataPreprocessor.preprocess_dataframe(df)

    # 2. NLP Embeddings
    logger.info("Metinler vektörleştiriliyor (mpnet + emotion)...")

    texts = df['clean_review_text'].astype(str).tolist()

    # 2.1 Base (mpnet) embedding
    emb_base = encode_texts_base(texts, batch_size=32)        # [N, d1]

    # 2.2 Emotion embedding
    emb_emotion = encode_texts_emotion(texts, batch_size=32)  # [N, d2]

    # 2.3 PCA
    pca_dim = 64

    logger.info(f"PCA fit ediliyor (base/mpnet -> {pca_dim})...")
    pca_base = PCA(n_components=pca_dim, random_state=42)
    emb_base_pca = pca_base.fit_transform(emb_base)           # [N,32]

    logger.info(f"PCA fit ediliyor (emotion -> {pca_dim})...")
    pca_emotion = PCA(n_components=pca_dim, random_state=42)
    emb_emotion_pca = pca_emotion.fit_transform(emb_emotion)  # [N,32]

    joblib.dump(pca_base, os.path.join(ARTIFACTS_DIR, f"pca_base_{pca_dim}.pkl"))
    joblib.dump(pca_emotion, os.path.join(ARTIFACTS_DIR, f"pca_emotion_{pca_dim}.pkl"))

    # 2.4 Birleştir: [N,64]
    emb_combined = np.concatenate([emb_base_pca, emb_emotion_pca], axis=1)

    # 2.5 Embedding sütun isimleri
    emb_cols = [f'emb_base_{i}' for i in range(pca_dim)] + \
               [f'emb_emotion_{i}' for i in range(pca_dim)]

    embedding_df = pd.DataFrame(emb_combined, columns=emb_cols, index=df.index)

    # 4. Kategorik
    cat_cols = ['user_id', 'product_id', 'main_category']

    # 5. Sayısal
    num_cols = [
        'log_price', 'discount_rate', 'word_count', 'char_count',
        'caps_ratio', 'exclamation_count',
        "has_positive_word", "has_negative_word"
    ]

    # 6. Tarihsel
    date_cols = ['review_year', 'month_sin', 'month_cos', 'is_weekend']

    final_features = cat_cols + num_cols + date_cols

    # 7. Tüm feature'ları birleştirme
    X = pd.concat([
        df[final_features],
        embedding_df
    ], axis=1)

    y = df['rating'].astype(float)

    return train_test_split(X, y, test_size=0.2, random_state=42), emb_cols, cat_cols



def objective(trial, X_train, y_train, X_test, y_test, cat_features):
    """Optuna için optimizasyon hedef fonksiyonu."""

    params = {
        'iterations': 750,
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2),
        'depth': trial.suggest_int('depth', 5, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 30),
        'random_strength': trial.suggest_float('random_strength', 0.5, 8),
        'loss_function': 'RMSE',
        'eval_metric': 'MAE',
        'verbose': False,
        'allow_writing_files': False,
        'cat_features': cat_features,
        'task_type': 'GPU'
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)

    return model.get_best_score()['validation']['MAE']


def train_pipeline():
    # 1. Veri Hazırlığı
    (X_train, X_test, y_train, y_test), emb_cols, cat_features = prepare_data()

    logger.info(f"Eğitim başlıyor. Toplam Özellik Sayısı (Encoding öncesi): {X_train.shape[1]}")
    logger.info(f"Kullanılan Kategorik Özellikler: {cat_features}")

    logger.info("Target Encoding uygulanıyor...")

    # Geçici dataframe oluşturup ortalama hesapla
    X_train_temp = X_train.copy()
    X_train_temp['rating'] = y_train

    # İstatistikleri hesapla
    # user_stats = X_train_temp.groupby("user_id")["rating"].mean()
    product_stats = X_train_temp.groupby("product_id")["rating"].agg(["mean", "count"])
    global_mean = y_train.mean()

    # Train setine ekle
    # X_train["user_mean_rating"] = X_train["user_id"].map(user_stats).fillna(global_mean)
    # X_train["product_mean_rating"] = X_train["product_id"].map(product_stats["mean"]).fillna(global_mean)
    X_train["product_review_count"] = X_train["product_id"].map(product_stats["count"]).fillna(0)

    # Test setine ekle
    # X_test["user_mean_rating"] = X_test["user_id"].map(user_stats).fillna(global_mean)
    # X_test["product_mean_rating"] = X_test["product_id"].map(product_stats["mean"]).fillna(global_mean)
    X_test["product_review_count"] = X_test["product_id"].map(product_stats["count"]).fillna(0)

    # Production için istatistikleri kaydet
    # joblib.dump(user_stats, os.path.join(ARTIFACTS_DIR, "user_stats.pkl"))
    # joblib.dump(product_stats, os.path.join(ARTIFACTS_DIR, "product_stats.pkl"))
    joblib.dump(global_mean, os.path.join(ARTIFACTS_DIR, "global_mean.pkl"))

    joblib.dump(X_train.columns.tolist(), os.path.join(ARTIFACTS_DIR, "feature_order.pkl"))

    logger.info("Target Encoding tamamlandı.")
    # -------------------------------------------------- -----

    # ---- XGBoost / LightGBM için kategorikleri label encoding ----
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    le_dict = {}
    for col in cat_features:
        le = LabelEncoder()
        X_train_xgb[col] = le.fit_transform(X_train_xgb[col].astype(str))
        X_test_xgb[col] = X_test_xgb[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        le_dict[col] = le

    joblib.dump(le_dict, os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))

    # -------------------------------------------------------------

    # Baseline (mean predictor)
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    logger.info(f"Baseline (Mean) -> RMSE={baseline_rmse:.4f}, MAE={baseline_mae:.4f}, R2={baseline_r2:.4f}")

    # 2. CatBoost için Hyperparameter Tuning (Optuna)
    logger.info("Optuna ile CatBoost için en iyi parametreler aranıyor...")
    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective(trial, X_train, y_train, X_test, y_test, cat_features)
    study.optimize(func, n_trials=30)

    logger.info(f"En iyi CatBoost parametreleri bulundu: {study.best_params}")

    # 3. Final CatBoost Modeli Eğitme
    final_params = study.best_params.copy()
    final_params.update({
        'iterations': 1500,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': 100,
        'allow_writing_files': False,
        'cat_features': cat_features,
        'task_type': 'GPU'
    })

    logger.info("Final CatBoost modeli (tuned) eğitiliyor...")
    cat_model = CatBoostRegressor(**final_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=30,
        use_best_model=True
    )

    cat_preds = cat_model.predict(X_test)
    cat_rmse = np.sqrt(mean_squared_error(y_test, cat_preds))
    cat_mae = mean_absolute_error(y_test, cat_preds)
    cat_r2 = r2_score(y_test, cat_preds)

    logger.info("-" * 30)
    logger.info(f"CatBoost (tuned) TEST SONUÇLARI:")
    logger.info(f"RMSE : {cat_rmse:.4f}")
    logger.info(f"MAE  : {cat_mae:.4f}")
    logger.info(f"R2   : {cat_r2:.4f}")
    logger.info("-" * 30)

    # 4. XGBoost
    logger.info("XGBoost modeli eğitiliyor...")
    xgb_model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_xgb, y_train)
    xgb_preds = xgb_model.predict(X_test_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)
    logger.info(f"XGBoost -> RMSE={xgb_rmse:.4f}, MAE={xgb_mae:.4f}, R2={xgb_r2:.4f}")

    # 5. LightGBM
    logger.info("LightGBM modeli eğitiliyor...")
    lgb_model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train_xgb, y_train)
    lgb_preds = lgb_model.predict(X_test_xgb)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    lgb_mae = mean_absolute_error(y_test, lgb_preds)
    lgb_r2 = r2_score(y_test, lgb_preds)
    logger.info(f"LightGBM -> RMSE={lgb_rmse:.4f}, MAE={lgb_mae:.4f}, R2={lgb_r2:.4f}")

    # 6. OOF Stacking Ensemble (Cat+XGB+LGBM -> Ridge)
    logger.info("OOF stacking ensemble (CatBoost + XGBoost + LightGBM -> Ridge) eğitiliyor...")
    base_models, meta_model = oof_stacking_ensemble(X_train_xgb, y_train, cat_features=[], n_splits=5,
                                                    random_state=42)

    # Test set üzerinde ensemble performansı
    logger.info("Stacking ensemble test set üzerinde değerlendiriliyor...")
    cat_test = base_models["cat"].predict(X_test_xgb)
    xgb_test = base_models["xgb"].predict(X_test_xgb)
    lgb_test = base_models["lgb"].predict(X_test_xgb)

    stack_X_test = np.vstack([cat_test, xgb_test, lgb_test]).T  # [n_samples, 3]
    stack_preds = meta_model.predict(stack_X_test)
    stack_rmse = np.sqrt(mean_squared_error(y_test, stack_preds))
    stack_mae = mean_absolute_error(y_test, stack_preds)
    stack_r2 = r2_score(y_test, stack_preds)
    logger.info(f"Stacked Ensemble -> RMSE={stack_rmse:.4f}, MAE={stack_mae:.4f}, R2={stack_r2:.4f}")

    # 7. Model karşılaştırmaları
    logger.info("-" * 50)
    logger.info("MODEL KARŞILAŞTIRMA (MAE):")
    logger.info(f"Baseline (Mean)    : {baseline_mae:.4f}")
    logger.info(f"CatBoost (tuned)   : {cat_mae:.4f}")
    logger.info(f"XGBoost            : {xgb_mae:.4f}")
    logger.info(f"LightGBM           : {lgb_mae:.4f}")
    logger.info(f"Stacked Ensemble   : {stack_mae:.4f}")
    logger.info("-" * 50)

    # 8. Görselleştirme
    logger.info("Grafikler oluşturuluyor (CatBoost için)...")
    ModelEvaluator.plot_learning_curve(cat_model)
    ModelEvaluator.plot_feature_importance(cat_model, X_train.columns.tolist())
    X_sample = X_test.iloc[:200]
    ModelEvaluator.explain_with_shap(cat_model, X_sample)

    # 9. Kaydetme
    logger.info("Modeller kaydediliyor...")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # CatBoost tuned
    cat_model_path = os.path.join(ARTIFACTS_DIR, "catboost_tuned.cbm")
    cat_model.save_model(cat_model_path)

    # Diğer tek modeller
    joblib.dump(xgb_model, os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
    joblib.dump(lgb_model, os.path.join(ARTIFACTS_DIR, "lgb_model.pkl"))

    # Stacking base modelleri ve meta model
    joblib.dump(base_models["cat"], os.path.join(ARTIFACTS_DIR, "stack_cat_model.cbm.pkl"))
    joblib.dump(base_models["xgb"], os.path.join(ARTIFACTS_DIR, "stack_xgb_model.pkl"))
    joblib.dump(base_models["lgb"], os.path.join(ARTIFACTS_DIR, "stack_lgb_model.pkl"))
    joblib.dump(meta_model, os.path.join(ARTIFACTS_DIR, "stack_meta_ridge.pkl"))

    # Optuna en iyi parametreler
    joblib.dump(study.best_params, os.path.join(ARTIFACTS_DIR, "best_params_catboost.pkl"))

    logger.info(f"Model ve raporlar {ARTIFACTS_DIR} altına kaydedildi.")


if __name__ == "__main__":
    train_pipeline()
