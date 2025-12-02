import os
import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Invent.ai Intelligence Dashboard",
    page_icon="",
    layout="wide"
)

st.title("Invent.ai: Intelligent Rating Predictor")
st.markdown("""
Bu dashboard, **SOTA Hybrid Architecture (LLM + CatBoost)** modelini kullanarak ürün yorumlarını ve özelliklerini analiz eder.
Model; metin duygusu, fiyat psikolojisi (log-price) ve kullanıcı geçmişi var ise birleştirerek 1-10 arası puan tahmini yapar.
""")

st.markdown("---")

API_URL = os.getenv("API_URL", "http://localhost:8000")


with st.sidebar:
    st.header("Ürün Detayları")

    review_text = st.text_area(
        "Müşteri Yorumu",
        "Absolutely fantastic product. Build quality is amazing and it feels premium in every detail. Worth every penny.",
        height=250,
        help="Müşterinin yazdığı yorumu buraya yapıştırın."
    )

    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Fiyat ($)", value=49.99, min_value=0.0, step=1.0)
    with col2:
        discount = st.number_input("İndirim Oranı  (0-1)", value=0.0, min_value=0.0, max_value=1.0, step=0.05)

    category = st.selectbox(
        "Kategori",
        ["Electronics", "Sports & Outdoors", "Home", "Clothing", "Health & Personal Care", "Other"]
    )

    with st.expander("Gelişmiş Ayarlar (ID)"):
        user_id = st.text_input(
            "User ID",
            "UNKNOWN_USER",
            help="Model bu kullanıcıyı daha önce gördüyse davranışını hatırlar."
        )
        product_id = st.text_input("Product ID", "UNKNOWN_PRODUCT")

    predict_btn = st.button("Analiz Et ve Puanla", type="primary", use_container_width=True)

if predict_btn:

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.sidebar.write(f"API durumu: {health.get('status', 'unknown')}")
    except Exception:
        st.sidebar.write("API durumu: erişilemiyor")

    payload = {
        "review_text": review_text,
        "user_id": user_id,
        "product_id": product_id,
        "product_category": category,
        "price": price,
        "discount_rate": discount
    }

    with st.spinner("Yapay Zeka (LLM + CatBoost) veriyi işliyor..."):
        try:
            start_time = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            end_time = time.time()
            latency = round((end_time - start_time) * 1000, 2)

            st.metric(label="Servis Gecikmesi", value=f"{latency} ms")



            if response.status_code == 200:
                result = response.json()

                if "prediction_details" in result and isinstance(result["prediction_details"], dict):
                    details = result["prediction_details"]
                else:
                    details = result

                rating = details.get("predicted_rating")
                sentiment = details.get("sentiment_analysis", "-")

                if rating is None:
                    st.error(f"Beklenmeyen response formatı: {result}")
                else:
                    st.success(f"Analiz Tamamlandı ({latency}ms)")

                    col_res1, col_res2, col_res3 = st.columns([1, 1, 2])

                    with col_res1:
                        st.metric(
                            label="Tahmini Puan",
                            value=f"{rating:.2f}/10",
                            delta=("Yüksek" if rating > 7 else "Nötr" if rating >= 4 else "Düşük")
                        )

                    with col_res2:
                        st.metric(label="Duygu Durumu", value=sentiment)

                    with col_res3:
                        st.write("Puan Ölçeği:")
                        st.progress(min(max(rating / 10, 0), 1))
                        if rating > 8:
                            st.caption("🌟 Bu ürün müşteriyi çok mutlu etmiş görünüyor!")
                        elif rating < 4:
                            st.caption("⚠️ Müşteri memnuniyetsizliği riski yüksek.")
                        else:
                            st.caption("🙂 Nötr/ortalama seviyede memnuniyet.")


                    with st.expander("API Yanıtı (JSON)"):
                        st.json(result)

            elif response.status_code == 503:
                st.error("Model henüz hazır değil. Lütfen sunucunun eğitimi tamamladığından emin olun.")
            else:
                st.error(f"Sunucu Hatası: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("API'ye bağlanılamadı. `uvicorn app:app --reload` komutunun çalıştığından emin olun.")
        except Exception as e:
            st.error(f"Beklenmeyen Hata: {str(e)}")
