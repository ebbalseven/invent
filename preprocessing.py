import re
import pandas as pd
import numpy as np

"""

Metin İstatistikleri Entegre Edildi: Artık train_sota.py dosyasında word_count veya caps_ratio (bağırma oranı) gibi özellikleri de modele besleyebilirsiniz. Sinirli müşteriler genellikle BÜYÜK HARF kullanır, bu özellik puan tahmininde çok işe yarar.

Entegrasyon Eksikliği Giderildi: get_text_features fonksiyonunu yazmışsın ama preprocess_dataframe içinde çağırmamışsın. Bunu ana akışa dahil ettik.

Logaritmik Dönüşüm (Önemli!): Fiyat (price) genellikle "sağa çarpık" (right-skewed) dağılır. 10 dolarlık ürünle 1000 dolarlık ürün arasındaki farkı modelin daha iyi anlaması için Log-Price ekledik.

Döngüsel Zaman (Cyclical Time Features): Ay (1-12) verisi döngüseldir. 12. ay ile 1. ay birbirine çok yakındır ama sayısal olarak uzaktır. Bunu çözmek için Sinüs/Kosinüs dönüşümü ekledik. Bu, SOTA bir tekniktir.

Tutarlılık: Lambda fonksiyonları yerine kendi yazdığınız statik metodları çağırdık.

"""

class DataPreprocessor:
    """
    Veri temizleme, özellik üretimi ve doğrulama merkezi.
    Advanced Feature Engineering tekniklerini içerir.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """HTML, URL temizler, Transformer modelleri için noktalama işaretlerini korur."""
        if pd.isna(text) or text is None:
            return ""

        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)  # HTML
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # URL
        # Transformer'lar (!?.) gibi işaretlerden duygu anlar, o yüzden onları silmiyoruz.
        text = re.sub(r'[^a-z0-9\s.,!?\'"-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def process_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Tarih verisinden Döngüsel (Cyclical) özellikler üretir."""
        if 'review_date' in df.columns:
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')

            # 1. Temel Bileşenler
            df['review_year'] = df['review_date'].dt.year.fillna(2023).astype(int)
            month = df['review_date'].dt.month.fillna(1).astype(int)
            day_of_week = df['review_date'].dt.dayofweek.fillna(0).astype(int)

            df['review_month'] = month
            df['is_weekend'] = day_of_week.apply(lambda x: 1 if x >= 5 else 0)

            # 2. Döngüsel Zaman Özellikleri
            df['month_sin'] = np.sin(2 * np.pi * month / 12)
            df['month_cos'] = np.cos(2 * np.pi * month / 12)

        return df

    @staticmethod
    def process_category(category_str: str) -> str:
        """'Sports | Bags' -> 'Sports' dönüşümü."""
        if pd.isna(category_str) or category_str == "":
            return "Other"
        return category_str.split('|')[0].strip()

    @staticmethod
    def get_text_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Metin uzunluğu, kelime sayısı gibi meta-verileri ekler."""
        if 'review_text' in df.columns:
            # İşlem hızı için vektörize işlemler
            df['word_count'] = df['review_text'].astype(str).apply(lambda x: len(x.split()))
            df['char_count'] = df['review_text'].astype(str).apply(len)
            # Bağırma/Duygu oranı (Büyük harf oranı)
            df['caps_ratio'] = df['review_text'].astype(str).apply(
                lambda x: sum(1 for c in x if c.isupper()) / max(1, len(x))
            )
            # Ünlem sayısı (Duygu yoğunluğu)
            df['exclamation_count'] = df['review_text'].astype(str).apply(lambda x: x.count('!'))
        return df

    @staticmethod
    def process_numerics(df: pd.DataFrame) -> pd.DataFrame:
        """Sayısal verileri normalize eder ve Log dönüşümü yapar."""
        if 'price' in df.columns:
            # Negatif fiyatları düzelt
            df['price'] = df['price'].apply(
                lambda x: max(0.0, float(x)) if not pd.isna(x) else 0.0
            )

            # Log Dönüşümü
            df['log_price'] = np.log1p(df['price'])

        if 'discount_rate' in df.columns:
            df['discount_rate'] = df['discount_rate'].apply(
                lambda x: max(0.0, min(1.0, float(x))) if not pd.isna(x) else 0.0
            )

        return df

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        ANA PIPELINE: Tüm işlemleri sırasıyla uygular.
        Training ve inference tarafında BİREBİR aynı featurization'ın
        kullanılmasını garanti eder.
        """
        df = df.copy()

        # 1. Eksik Veri (FillNA)
        fill_vals = {
            'user_id': 'UNKNOWN',
            'product_id': 'UNKNOWN',
            'price': 0.0,
            'discount_rate': 0.0,
            'review_text': ''
        }
        df.fillna(fill_vals, inplace=True)

        # 2. Metin Temizliği
        if 'review_text' in df.columns:
            df['clean_review_text'] = df['review_text'].apply(
                DataPreprocessor.clean_text
            )

        # 2.1 Regex bazlı duygu bayrakları
        # (Train'de prepare_data'de yaptığın işi buraya taşıyoruz)
        if 'clean_review_text' in df.columns:
            df["has_negative_word"] = df["clean_review_text"].str.contains(
                r"\b(bad|terrible|awful|disappointed|useless|broke|refund)\b",
                case=False,
                regex=True,
            ).astype(int)

            df["has_positive_word"] = df["clean_review_text"].str.contains(
                r"\b(great|perfect|excellent|amazing|love|highly recommend)\b",
                case=False,
                regex=True,
            ).astype(int)
        else:
            df["has_negative_word"] = 0
            df["has_positive_word"] = 0

        # 3. Kategori İşleme
        if 'product_category' in df.columns:
            df['main_category'] = df['product_category'].apply(
                DataPreprocessor.process_category
            )

        # 4. Tarih İşlemleri (Cyclical features dahil)
        df = DataPreprocessor.process_dates(df)

        # 4.1 Eğer review_date yoksa bile, tarih feature'ları her zaman mevcut olsun
        for col in ["review_year", "month_sin", "month_cos", "is_weekend"]:
            if col not in df.columns:
                df[col] = 0

        # 5. Sayısal İşlemler (Log Price dahil)
        df = DataPreprocessor.process_numerics(df)

        # 6. Metin İstatistikleri (Feature Engineering)
        df = DataPreprocessor.get_text_stats(df)

        return df


