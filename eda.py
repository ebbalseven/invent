import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from preprocessing import DataPreprocessor

# --- Ayarlar ---
sns.set_theme(style="whitegrid", context="talk")
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_html_report(output_dir: str) -> None:
    print("HTML Raporu derleniyor...")

    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Invent.ai EDA Raporu</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f8f9fa; color: #333; }
            .container { max-width: 1000px; margin: 40px auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 10px; }
            p.subtitle { text-align: center; color: #7f8c8d; margin-bottom: 40px; }
            .chart-card { background: white; margin-bottom: 40px; border: 1px solid #e9ecef; border-radius: 8px; overflow: hidden; transition: transform 0.2s; }
            .chart-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.05); }
            .chart-header { background: #f1f3f5; padding: 15px 20px; font-weight: 600; color: #495057; border-bottom: 1px solid #e9ecef; }
            img { width: 100%; height: auto; display: block; }
            .footer { text-align: center; margin-top: 50px; font-size: 0.9em; color: #adb5bd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Invent.ai Keşifsel Veri Analizi</h1>
            <p class="subtitle">E-Ticaret Veri Seti İçgörü Raporu</p>
    """

    # Klasördeki tüm PNG dosyalarını bul ve rapora ekle
    images = sorted(glob.glob(os.path.join(output_dir, "*.png")))

    if not images:
        print("UYARI: Klasörde resim bulunamadı, rapor boş olabilir.")

    for img_path in images:
        filename = os.path.basename(img_path)
        title = (
            filename.replace(".png", "")
            .split("_", 1)[-1]
            .replace("_", " ")
            .title()
        )

        html_content += f"""
            <div class="chart-card">
                <div class="chart-header">{title}</div>
                <img src="{filename}" alt="{title}" loading="lazy">
            </div>
        """

    html_content += """
            <div class="footer">
                Generated automatically by Invent.ai Intelligence Engine
            </div>
        </div>
    </body>
    </html>
    """

    report_path = os.path.join(output_dir, "EDA_Report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Rapor Hazır: {report_path}")


def generate_eda_report() -> None:
    print("Keşifsel Veri Analizi (EDA) Başlatılıyor...")

    # 1. Veri Yükleme ve Temizleme
    try:
        raw_df = pd.read_csv("data/product_user_reviews.csv")
        df = DataPreprocessor.preprocess_dataframe(raw_df)
        print(f"✔ Veri yüklendi ve temizlendi: {len(df)} satır")
    except FileNotFoundError:
        print("HATA: 'data/product_user_reviews.csv' dosyası bulunamadı.")
        return

    # Rating'i numerik tipe zorla, hatalıları at
    if "rating" not in df.columns:
        print("HATA: 'rating' kolonu bulunamadı, EDA durduruluyor.")
        return

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    # --- Grafik 1: Puan Dağılımı (The J-Curve) ---
    print("   1. Puan Dağılımı analizi yapılıyor...")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x="rating", data=df, palette="viridis")
    plt.title("Puan Dağılımı (J-Curve Etkisi)", fontsize=16, fontweight="bold")
    plt.xlabel("Müşteri Puanı (1-10)")
    plt.ylabel("Yorum Sayısı")

    # Barların üzerine sayıları yaz
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_rating_distribution.png"))
    plt.close()

    # --- Grafik 2: En Popüler Kategoriler ---
    if "main_category" in df.columns:
        print("   2. Kategori analizi yapılıyor...")
        top_cats = df["main_category"].value_counts().head(10)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_cats.values, y=top_cats.index, palette="mako")
        plt.title("En Çok Yorum Alan 10 Kategori", fontsize=16, fontweight="bold")
        plt.xlabel("Yorum Sayısı")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2_top_categories.png"))
        plt.close()
    else:
        print("   2. main_category kolonu bulunamadı, kategori grafiği atlanıyor.")

    # --- Grafik 3: Fiyat vs Puan (Scatter Plot) ---
    print("   3. Fiyat-Puan ilişkisi inceleniyor...")
    price_col = None
    if "price" in df.columns:
        price_col = "price"
    elif "log_price" in df.columns:
        price_col = "log_price"

    if price_col is not None:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=price_col,
            y="rating",
            data=df,
            alpha=0.3,
            color="purple",
        )
        if price_col == "price":
            plt.title("Fiyat ve Müşteri Puanı İlişkisi", fontsize=16, fontweight="bold")
            plt.xlabel("Fiyat ($)")
            plt.xscale("log")
        else:
            plt.title(
                "Log(Fiyat) ve Müşteri Puanı İlişkisi",
                fontsize=16,
                fontweight="bold",
            )
            plt.xlabel("Log(Fiyat)")

        plt.ylabel("Puan")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "3_price_vs_rating.png"))
        plt.close()
    else:
        print("   3. price / log_price kolonları bulunamadı, Fiyat-Puan grafiği atlanıyor.")

    # --- Grafik 4: Yorum Uzunluğu Analizi ---
    if "word_count" in df.columns:
        print("   4. Yorum uzunluğu analizi yapılıyor...")
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            x="rating",
            y="word_count",
            data=df,
            showfliers=False,
            palette="coolwarm",
        )
        plt.title("Puanlara Göre Yorum Uzunluğu", fontsize=16, fontweight="bold")
        plt.xlabel("Puan")
        plt.ylabel("Kelime Sayısı")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "4_review_length.png"))
        plt.close()
    else:
        print("   4. word_count kolonu bulunamadı, yorum uzunluğu grafiği atlanıyor.")

    # --- Grafik 5: Korelasyon Matrisi (Isı Haritası) ---
    print("   5. Korelasyon matrisi çıkarılıyor...")
    numeric_cols = [
        "rating",
        "price",
        "log_price",
        "discount_rate",
        "word_count",
        "caps_ratio",
        "exclamation_count",
    ]
    valid_cols = [c for c in numeric_cols if c in df.columns]

    if valid_cols:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[valid_cols].dropna().corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
        )
        plt.title("Özellik Korelasyon Matrisi", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "5_correlation_matrix.png"))
        plt.close()
    else:
        print("   5. Geçerli numerik kolon bulunamadı, korelasyon matrisi atlanıyor.")

    # --- Grafik 6: Kelime Bulutları (Word Cloud) ---
    print("   6. Kelime bulutları oluşturuluyor...")

    if "clean_review_text" in df.columns:
        # Pozitif (9-10) ve Negatif (1-3) yorumları ayır
        pos_df = df[df["rating"] >= 9]
        neg_df = df[df["rating"] <= 3]

        # Büyük veri setlerinde RAM'i korumak için sample al
        if len(pos_df) > 0:
            pos_sample = pos_df.sample(
                min(len(pos_df), 5000),
                random_state=42,
            )
            pos_text = " ".join(pos_sample["clean_review_text"].astype(str))

            if len(pos_text) > 0:
                wc_pos = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    colormap="Greens",
                ).generate(pos_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc_pos, interpolation="bilinear")
                plt.axis("off")
                plt.title(
                    "Mutlu Müşterilerin Dili (Rating 9-10)",
                    fontsize=14,
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(OUTPUT_DIR, "6_wordcloud_positive.png")
                )
                plt.close()

        if len(neg_df) > 0:
            neg_sample = neg_df.sample(
                min(len(neg_df), 5000),
                random_state=42,
            )
            neg_text = " ".join(neg_sample["clean_review_text"].astype(str))

            if len(neg_text) > 0:
                wc_neg = WordCloud(
                    width=800,
                    height=400,
                    background_color="black",
                    colormap="Reds",
                ).generate(neg_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc_neg, interpolation="bilinear")
                plt.axis("off")
                plt.title(
                    "Şikayetçi Müşterilerin Dili (Rating 1-3)",
                    fontsize=14,
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(OUTPUT_DIR, "7_wordcloud_negative.png")
                )
                plt.close()
    else:
        print("   6. clean_review_text kolonu bulunamadı, kelime bulutları atlanıyor.")

    # --- Grafik 7: En Popüler Kategorilerde Puan Dağılımı ---
    if "main_category" in df.columns:
        print("   7. Kategoriye göre puan dağılımı inceleniyor...")
        top_cat_names = df["main_category"].value_counts().head(5).index
        top_cat_df = df[df["main_category"].isin(top_cat_names)]

        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=top_cat_df,
            x="main_category",
            y="rating",
            palette="Set2",
            showfliers=False,
        )
        plt.title(
            "En Popüler Kategorilerde Puan Dağılımı",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Kategori")
        plt.ylabel("Puan")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "8_rating_by_top_categories.png"))
        plt.close()

    print(f"\nAnalizler '{OUTPUT_DIR}' klasörüne kaydedildi.")

    # HTML raporu
    create_html_report(OUTPUT_DIR)


if __name__ == "__main__":
    generate_eda_report()
