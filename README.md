# 🎬 Film Verileri Üzerinde Veri Madenciliği ve Akıllı Film Analiz Sistemi

## 1. Projenin Amacı

Bu proje, **TMDB 5000 Movies** ve **TMDB 5000 Credits** veri setlerini birleştirerek film verileri üzerinde gerçek veri madenciliği ve makine öğrenmesi algoritmaları uygulayan, algoritma sonuçlarını doğruluk/kalite ölçütleriyle değerlendiren, kapsamlı görselleştirmeler üreten ve kullanıcılara içerik tabanlı film önerileri sunan interaktif bir **Streamlit** web uygulamasıdır.

Projede uygulanan dört temel veri madenciliği yöntemi şunlardır:

1. **Apriori Birliktelik Kuralı Madenciliği** (`mlxtend`)
2. **K-Means Kümeleme Algoritması** (`scikit-learn`)
3. **Random Forest Sınıflandırma Algoritması** (`scikit-learn`)
4. **TF-IDF ve Cosine Similarity ile İçerik Tabanlı Film Öneri Sistemi** (`scikit-learn`)

---

## 2. Kullanılan Veri Setleri

Projede iki temel CSV veri seti kullanılmıştır:

### Birinci Dosya: `tmdb_5000_movies.csv` (4.803 kayıt, 20 sütun)
- `id`: Film kimliği
- `title` / `original_title`: Film adı ve orijinal adı
- `budget`: Filmin bütçesi ($)
- `revenue`: Filmin gişe geliri ($)
- `genres`: JSON formatında film türleri
- `keywords`: JSON formatında anahtar kelimeler
- `overview`: Film açıklaması (özeti)
- `popularity`: Popülerlik skoru
- `production_companies` / `production_countries`: Yapım şirketleri ve ülkeleri
- `release_date`: Yayın tarihi
- `runtime`: Film süresi (dakika)
- `spoken_languages` / `original_language`: Konuşulan diller ve orijinal dil
- `vote_average` / `vote_count`: Ortalama kullanıcı puanı ve oy sayısı

### İkinci Dosya: `tmdb_5000_credits.csv` (4.803 kayıt, 4 sütun)
- `movie_id`: Film kimliği
- `title`: Film adı
- `cast`: JSON formatında oyuncu kadrosu bilgileri
- `crew`: JSON formatında film ekibi ve yönetmen bilgileri

---

## 3. Veri Birleştirme ve Ön İşleme (Data Preprocessing)

`src/data_preprocessing.py` modülü tarafından gerçekleştirilen işlemler ve gerçek veri seti bulguları:

- **CSV Birleştirme:** `pd.merge(movies, credits, left_on="id", right_on="movie_id", how="inner")` komutuyla iki tablo birleştirilmiştir.
  - Ortak film kimliği (`id == movie_id`) sayısı: **4.803**
  - Tekrar eden film kimliği sayısı: **0**
  - Eşleşmeyen film sayısı / Veri kaybı: **0**
  - İkinci dosyadan gelen mükerrer `title` sütunu kaldırılmıştır.
- **Eksik Verilerin Temizlenmesi:**
  - `overview` sütunundaki **3** eksik kayıt boş metin (`""`) olarak doldurulmuştur.
  - `runtime` sütunundaki **2** eksik kayıt medyan film süresi olan **103.0 dakika** ile doldurulmuştur.
  - `release_date` sütunundaki **1** eksik kayıt kontrol edilmiş, `release_year` türetilirken medyan yıl (**2005**) ile tamamlanmıştır.
  - `budget` değeri `0` olan **1.037** kayıt ve `revenue` değeri `0` olan **1.427** kayıt eksik/bilinmeyen finansal veri olarak işaretlenmiştir (bütçesi bilinen: **3.766**, geliri bilinen: **3.376** film).
- **JSON Sütunlarının Ayrıştırılması:**
  - `genres` sütunundan tür isimleri çıkarılarak `genre_list` oluşturulmuştur (boş tür sayısı: **28**).
  - `keywords` sütunundan anahtar kelimeler çıkarılarak `keyword_list` oluşturulmuştur (boş anahtar kelime sayısı: **412**).
  - `cast` sütunundan ilk 5 oyuncunun adı çıkarılarak `cast_list` oluşturulmuştur (boş oyuncu sayısı: **43**).
  - `crew` sütunundan `job == "Director"` olan yönetmen adı çıkarılarak `director` sütunu oluşturulmuştur (yönetmeni bilinmeyen: **30**).
- **Türetilen Yeni Özellikler:**
  - `release_year`: Yayın yılı
  - `profit`: Gelir - Bütçe (`revenue - budget`)
  - `profit_ratio`: Gelir / Bütçe oranı (yalnızca `budget > 0` olan filmler için hesaplanmış, bütçesi 0 olanlarda `NaN` bırakılmıştır)
  - Sağa çarpık dağılımlar için `budget_log`, `revenue_log`, `popularity_log` ve `vote_count_log` (`np.log1p`) dönüşümleri uygulanmıştır.
  - Temizlenmiş veri seti `data/cleaned_movies.csv` (ve `cleaned_movies.csv`) olarak kaydedilmiştir.

---

## 4. Kullanılan Veri Madenciliği Yöntemleri ve Çalışma Mantığı

### 4.1. Apriori Birliktelik Kuralı Madenciliği (`src/apriori_analysis.py`)
- Her film bir işlem (*transaction*) olarak kabul edilir; filmin türleri ve sık görülen anahtar kelimeleri öğe listesine dönüştürülür.
- `TransactionEncoder` ile One-Hot Encoding ikili (True/False) matrisine çevrilir.
- `mlxtend.frequent_patterns.apriori` ile sık öğe kümeleri (`min_support=0.02`), `association_rules` ile birliktelik kuralları (`min_confidence=0.30`, `min_lift=1.0`) üretilir.
- **Support**, **Confidence** ve **Lift** ölçütleri hesaplanır.

### 4.2. K-Means Kümeleme (`src/kmeans_clustering.py`)
- Filmler `budget`, `revenue`, `popularity`, `runtime`, `vote_average`, `vote_count` ve `genre_list` (`MultiLabelBinarizer`) özellikleri üzerinden kümelenir.
- Çarpık finansal değişkenlere `log1p` dönüşümü, sayısal değişkenlere `StandardScaler` ölçeklendirmesi uygulanır.
- $k \in [2, 8]$ aralığında **Elbow Method (WCSS / Inertia)** ve **Silhouette Score** hesaplanır.
- Kümelere sayısal etiketlerin yanı sıra küme ortalamalarına dayalı anlamlı Türkçe arketip isimleri atanır ve 2B PCA uzayında görselleştirilir.

### 4.3. Random Forest Sınıflandırma (`src/random_forest.py`)
- Geliri bilinen (**3.376**) filmlerin medyan geliri (**\$51.751.835**) hesaplanır; medyanın üzerindekilere **Yüksek Gelir**, medyan ve altındakilere **Düşük Gelir** etiketi verilir.
- Tahmin özellikleri olarak yalnızca `budget`, `popularity`, `runtime`, `vote_average`, `vote_count`, `release_year` ve `genres` kullanılır (`revenue` ve `profit` veri sızıntısını önlemek için kesinlikle dahil edilmez).
- Veri `%80 eğitim (2.700 film)` ve `%20 test (676 film)` olarak `stratify=y` ile bölünür. Ölçeklendirme ve tür kodlaması yalnızca eğitim verisi üzerinde `fit` edilir.
- `RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")` modeli eğitilir ve test verisi üzerinde değerlendirilir.

### 4.4. İçerik Tabanlı Film Öneri Sistemi (`src/movie_recommendation.py`)
- Her film için `overview`, `genres`, `keywords`, `cast_list` ve `director` bilgileri birleştirilerek tek bir metin oluşturulur.
- `TfidfVectorizer(stop_words="english", ngram_range=(1, 2))` ile metinler sayısal vektörlere çevrilir.
- `cosine_similarity` ile seçilen film ve diğer tüm filmler arasındaki benzerlik hesaplanır, seçilen film listeden çıkarılarak en benzer 10 film ortak içerik unsurlarıyla birlikte sıralanır.

---

## 5. Gerçek Test Sonuçları

### 5.1. Apriori Birliktelik Analizi Test Sonuçları (`min_support=0.02`, `min_confidence=0.30`)
- **Toplam İşlem (Film) Sayısı:** 4.777
- **Bulunan Sık Öğe Kümesi Sayısı:** 94
- **Üretilen Güçlü Kural Sayısı:** 74
- **En Sık Görülen 5 Tür:**
  1. `Drama` (Support: %48.09 — 2.297 film)
  2. `Comedy` (Support: %36.06 — 1.722 film)
  3. `Thriller` (Support: %26.67 — 1.274 film)
  4. `Action` (Support: %24.16 — 1.154 film)
  5. `Romance` (Support: %18.72 — 894 film)
- **En Güçlü 5 Birliktelik Kuralı (Lift Değerine Göre):**

| Birliktelik Kuralı | Support | Confidence | Lift |
| :--- | :---: | :---: | :---: |
| `Adventure, Family -> Animation` | 0.0205 | 0.4645 | **9.48** |
| `Animation -> Adventure, Family` | 0.0205 | 0.4188 | **9.48** |
| `Animation, Comedy -> Family` | 0.0232 | 0.8880 | **8.27** |
| `Adventure, Animation -> Family` | 0.0205 | 0.8596 | **8.00** |
| `Anahtar: dystopia -> Science Fiction` | 0.0255 | 0.8777 | **7.84** |

### 5.2. K-Means Kümeleme Test Sonuçları (Bilinen Finansal Verili 3.229 Film, $k=4$)
- **Değerlendirme Skorları:** $k=2$ için Silhouette = `0.2841`, $k=3$ için Silhouette = `0.2104`, $k=4$ için Silhouette = `0.2069` (Inertia = `10526.35`).
- **$k=4$ Küme Profilleri:**

| Küme Adı | Film Sayısı | Ort. Bütçe ($) | Ort. Gelir ($) | Ort. Popülerlik | Ort. Puan | Baskın Türler |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Küme 0: Düşük Puanlı & Düşük Etkileşimli Yapımlar** | 1.153 | $40,302,630 | $79,851,410 | 19.53 | 5.66 | Comedy, Thriller, Action |
| **Küme 1: Beğenilen Bağımsız & Orta Ölçekli Filmler** | 708 | $14,190,180 | $34,701,310 | 17.24 | 6.91 | Drama, Comedy, Thriller |
| **Küme 2: Gişe Rekortmeni Dev Yapımlar (Yüksek Bütçe & Gelir)** | 978 | $72,831,730 | $278,704,200 | 58.62 | 6.81 | Drama, Action, Adventure |
| **Küme 3: Düşük Bütçeli & Az Bilinen Niş Filmler** | 390 | $9,046,639 | $5,855,488 | 4.35 | 5.88 | Drama, Comedy, Romance |

### 5.3. Random Forest Sınıflandırma Test Sonuçları (Test Seti: 676 Film)
- **Medyan Gelir Eşiği:** \$51,751,835.00
- **Accuracy (Doğruluk):** `%83.14` (0.8314)
- **Precision (Kesinlik):** `%83.94` (0.8394)
- **Recall (Duyarlılık):** `%81.95` (0.8195)
- **F1-Score:** `%82.93` (0.8293)
- **ROC-AUC Skoru:** `%92.62` (0.9262)
- **Confusion Matrix (Test Verisi):**
  - Gerçek Düşük Gelir -> Tahmin Düşük Gelir (TN): **285** | Tahmin Yüksek Gelir (FP): **53**
  - Gerçek Yüksek Gelir -> Tahmin Düşük Gelir (FN): **61** | Tahmin Yüksek Gelir (TP): **277**
- **En Etkili Özellikler (Feature Importance):**
  1. `Oy Sayısı (vote_count)`: **%27.20**
  2. `Popülerlik (popularity)`: **%19.08**
  3. `Film Bütçesi (budget)`: **%19.06**
  4. `Yayın Yılı (release_year)`: **%7.15**
  5. `Film Süresi (runtime)`: **%6.98**
  6. `Ortalama Puan (vote_average)`: **%6.94**

### 5.4. İçerik Tabanlı Film Öneri Sistemi Test Sonuçları
- **Seçilen Film: `Avatar` (2009)** -> En Benzer İlk 3 Film:
  1. `Aliens` (%39.47 — Aynı Yönetmen: James Cameron, Ortak Oyuncu: Sigourney Weaver, Ortak Tema: alien, space travel, space colony)
  2. `Alien³` (%34.01 — Ortak Oyuncu: Sigourney Weaver, Ortak Tema: alien, space colony)
  3. `Silent Running` (%29.36 — Ortak Tema: space travel, space)
- **Seçilen Film: `The Dark Knight` (2008)** -> En Benzer İlk 3 Film:
  1. `Batman Begins` (%52.25 — Aynı Yönetmen: Christopher Nolan, Ortak Oyuncu: Christian Bale, Michael Caine)
  2. `The Dark Knight Rises` (%52.11 — Aynı Yönetmen: Christopher Nolan, Ortak Oyuncu: Christian Bale, Michael Caine)
  3. `Batman Returns` (%40.54 — Ortak Tema: dc comics, gotham city, super villain)

---

## 6. Proje Klasör Yapısı

```text
project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   └── cleaned_movies.csv
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── apriori_analysis.py
    ├── kmeans_clustering.py
    ├── random_forest.py
    ├── movie_recommendation.py
    └── visualizations.py
```

---

## 7. Kurulum ve Çalıştırma Adımları

1. Gerekli Python kütüphanelerini yükleyin:
```bash
pip install -r requirements.txt
```

2. (İsteğe bağlı) Veri ön işleme modülünü tek başına çalıştırarak `data/cleaned_movies.csv` dosyasını oluşturun:
```bash
python src/data_preprocessing.py
```

3. Streamlit web uygulamasını başlatın:
```bash
python -m streamlit run app.py
```
*(Alternatif olarak `streamlit` komutu PATH'e ekliyse `streamlit run app.py`, PowerShell'de `.\streamlit run app.py` veya proje klasöründeki `baslat.bat` dosyasına çift tıklayarak çalıştırabilirsiniz.)*
