from fastapi.testclient import TestClient
from app import app
import pytest

client = TestClient(app)


def test_health_check_initial():
    """Sistem başlatıldığında sağlık durumu kontrolü."""
    response = client.get("/health")
    assert response.status_code == 200
    # Model yüklü değilse 'not_ready', yüklüyse 'healthy' döner.
    # Bu testin geçmesi için sunucunun ayakta olması yeterli.
    assert response.json()["status"] in ["healthy", "not_ready"]


def test_prediction_flow_success():
    """Geçerli bir istek başarılı dönmeli."""
    # Not: Bu testin geçmesi için 'model_artifacts' klasöründe eğitilmiş model olmalı.
    # Eğer yoksa test fail edebilir

    payload = {
        "review_text": "Excellent quality, I love it!",
        "price": 49.99,
        "discount_rate": 0.0,
        "product_category": "Home",
        "user_id": "TEST_USER",
        "product_id": "TEST_PRODUCT"
    }

    response = client.post("/predict", json=payload)

    if response.status_code == 503:
        pytest.skip("Model dosyaları bulunamadı, tahmin testi atlanıyor.")

    assert response.status_code == 200
    data = response.json()
    assert "predicted_rating" in data
    assert 1.0 <= data["predicted_rating"] <= 10.0
    assert data["sentiment_analysis"] == "Positive"


def test_validation_error_empty_text():
    """Boş yorum gönderilirse Pydantic 422 hatası vermeli."""
    payload = {
        "review_text": "",  # Hatalı
        "price": 100
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_price_validation():
    """Negatif fiyat gönderilirse Pydantic 422 hatası vermeli."""
    payload = {
        "review_text": "Good",
        "price": -10.0  # Hatalı
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_cold_start_handling():
    """Bilinmeyen kullanıcı/ürün durumunda sistem çökmeksizin yanıt vermeli."""
    payload = {
        "review_text": "Meh, it's okay.",
        "user_id": "BRAND_NEW_USER_9999",
        "product_id": "NEVER_SEEN_PRODUCT_8888",
        "price": 25.0
    }
    response = client.post("/predict", json=payload)

    if response.status_code == 503:
        pytest.skip("Model yok.")

    assert response.status_code == 200
    assert "predicted_rating" in response.json()