# Film Veri Madenciliği ve Akıllı Analiz Sistemi

## 1. Amaç (Purpose)
TMDB 5000 Movies ve TMDB 5000 Credits veri setlerini birleştirip ön işleyerek; film verileri üzerinde Apriori birliktelik analizi, K-Means kümeleme, Random Forest gişe gelir sınıflandırması ve TF-IDF/Cosine Similarity ile içerik tabanlı film önerisi sunan interaktif bir Streamlit web uygulamasıdır.

## 2. Teknoloji Yığını (Stack)
- **Dil:** Python 3.14 / 3.13
- **Web Arayüzü:** Streamlit
- **Veri İşleme & Analiz:** Pandas, NumPy
- **Makine Öğrenmesi & Modelleme:** Scikit-Learn, MLxtend
- **Görselleştirme:** Plotly, Matplotlib, Seaborn

## 3. Mimari Kararlar (Architecture Decisions)
- **Modüler Yapı:** Her algoritma ve veri ön işleme adımı `src/` altında bağımsız modüller halinde yapılandırılmıştır.
- **Veri Bütünlüğü & Sızıntı Önleme:** Random Forest modelinde gelir (`revenue`) ve kâr (`profit`) girdiden çıkarılmış, `train_test_split` sonrası yalnızca eğitim setine ölçeklendirme uygulanmıştır.
- **Performans & Önbellek:** Veri okuma ve model eğitimleri Streamlit `@st.cache_data` ile önbelleğe alınmıştır.
- **Çoklu Kodlama & Metin Analizi:** Çoklu etiketler için `MultiLabelBinarizer`, metin benzerliği için 1-2 gram `TfidfVectorizer` kullanılmıştır.

## 4. Dizin Yapısı (Folder Structure)
```
film-veri-madenciligi/
├── app.py                      # Ana Streamlit arayüzü ve navigasyon
├── baslat.bat                  # Windows tek tıkla başlatma betiği
├── requirements.txt            # Python bağımlılıkları
├── docs/
│   └── PROJECT.md              # Proje özeti ve bağlamı
├── data/
│   ├── tmdb_5000_movies.csv    # Ham film verisi (4.803 satır)
│   ├── tmdb_5000_credits.csv   # Ham film kadro verisi (4.803 satır)
│   └── cleaned_movies.csv      # Ön işlenmiş ve birleştirilmiş veri seti
└── src/
    ├── __init__.py
    ├── data_preprocessing.py   # Veri birleştirme, temizleme ve özellik mühendisliği
    ├── apriori_analysis.py     # Sık öğe kümeleri ve birliktelik kuralları
    ├── kmeans_clustering.py    # K-Means kümeleme, Elbow ve Silhouette skorları
    ├── random_forest.py        # Gişe geliri sınıflandırma ve önem skorları
    ├── movie_recommendation.py # TF-IDF ve kosinüs benzerliği ile öneri motoru
    └── visualizations.py       # Plotly ve Matplotlib görselleştirme fonksiyonları
```

## 5. Kurallar ve Standartlar (Conventions)
- Tip ipuçları (type hints) ve docstring kullanımı zorunludur.
- Veri seti filtrelemelerinde orijinal veriyi bozmamak için defensive copy (`df.copy()`) kullanılır.
- UI dili Türkçe, kod ve değişken isimlendirmeleri İngilizce standartlarındadır.

## 6. Çalıştırma ve Test (How to Run / Test)
- **Bağımlılık Kurulumu:** `pip install -r requirements.txt`
- **Uygulamayı Başlatma:** `streamlit run app.py` veya `baslat.bat`
- **Doğrulama:** `python -m py_compile app.py src/*.py`

## 7. Mevcut Durum (Current Status)
- Veri ön işleme ve tüm 4 analitik modül kodlanmış durumdadır.
- Temizlenmiş veri seti (`data/cleaned_movies.csv`) mevcuttur.
- Bağımlılıkların sisteme kurulması ve uygulamanın test edilmesi aşamasına geçilmektedir.
