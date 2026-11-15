"""
Film Verileri Üzerinde Veri Madenciliği ve Akıllı Film Analiz Sistemi
Streamlit Web Uygulaması (app.py).

Menü Yapısı:
1. Ana Sayfa
2. Veri Seti İnceleme
3. Apriori Birliktelik Analizi
4. K-Means Kümeleme
5. Random Forest Tahmin
6. Film Öneri Sistemi
"""

import os
import sys
from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Proje kök dizinini, src dizinini ve varsa yerel .venv site-packages dizinini Python yoluna ekle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
VENV_SITE_PACKAGES = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages")
for extra_path in [VENV_SITE_PACKAGES, SRC_DIR, BASE_DIR]:
    if os.path.exists(extra_path) and extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from src.apriori_analysis import run_apriori_analysis
from src.data_preprocessing import get_or_create_cleaned_data
from src.kmeans_clustering import run_kmeans_analysis
from src.movie_recommendation import (
    build_tfidf_engine,
    get_movie_recommendations,
    search_movies_by_query,
)
from src.random_forest import (
    predict_single_movie_revenue_class,
    run_random_forest_classification,
)
from src.visualizations import (
    plot_apriori_frequent_items,
    plot_apriori_rules_scatter,
    plot_average_revenue_by_genre,
    plot_budget_vs_revenue,
    plot_confusion_matrix_heatmap,
    plot_elbow_and_silhouette,
    plot_feature_importance,
    plot_kmeans_cluster_comparison,
    plot_kmeans_clusters_pca,
    plot_movies_by_year,
    plot_popularity_vs_vote_average,
    plot_recommendations_similarity,
    plot_top_budget_movies,
    plot_top_directors,
    plot_top_genres_count,
    plot_top_popular_movies,
    plot_top_revenue_movies,
    plot_top_rules_bar,
    plot_vote_average_distribution,
)

# Sayfa yapılandırması
st.set_page_config(
    page_title="Film Veri Madenciliği ve Akıllı Analiz Sistemi",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# ÖNBELLEKLİ (CACHED) VERİ VE MODEL FONKSİYONLARI
# ==============================================================================

@st.cache_data(show_spinner="Veri setleri birleştiriliyor ve temizleniyor...")
def cached_load_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Veri setini bir kez yükler, temizler ve önbelleğe alır."""
    return get_or_create_cleaned_data(force_reprocess=False)


@st.cache_data(show_spinner="Apriori birliktelik kuralları hesaplanıyor...")
def cached_apriori(
    _df: pd.DataFrame,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    include_keywords: bool,
) -> Dict[str, Any]:
    """Apriori analiz sonuçlarını seçilen parametrelere göre önbelleğe alır."""
    return run_apriori_analysis(
        df=_df,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        include_keywords=include_keywords,
    )


@st.cache_data(show_spinner="K-Means kümeleme ve kalite metrikleri hesaplanıyor...")
def cached_kmeans(
    _df: pd.DataFrame,
    n_clusters: int,
    filter_zero_financials: bool,
    include_genres: bool,
) -> Dict[str, Any]:
    """K-Means kümeleme sonuçlarını önbelleğe alır."""
    return run_kmeans_analysis(
        df=_df,
        n_clusters=n_clusters,
        filter_zero_financials=filter_zero_financials,
        include_genres=include_genres,
    )


@st.cache_resource(show_spinner="Random Forest modeli eğitiliyor ve test ediliyor...")
def cached_random_forest(
    _df: pd.DataFrame,
    n_estimators: int,
    max_depth: Any,
    test_size: float,
    class_weight: Any,
) -> Dict[str, Any]:
    """Random Forest sınıflandırma modelini ve metriklerini önbelleğe alır."""
    return run_random_forest_classification(
        df=_df,
        n_estimators=n_estimators,
        max_depth=max_depth,
        test_size=test_size,
        class_weight=class_weight,
    )


@st.cache_resource(show_spinner="TF-IDF içerik vektörleri hazırlanıyor...")
def cached_recommendation_engine(_df: pd.DataFrame) -> Dict[str, Any]:
    """TF-IDF matrisini bir kez oluşturup bellekte tutar."""
    return build_tfidf_engine(_df)


def get_most_common_genre(df: pd.DataFrame) -> Tuple[str, int]:
    """En çok film bulunan türü ve film sayısını döndürür."""
    counter: Counter = Counter()
    for g_list in df["genre_list"]:
        if isinstance(g_list, list):
            for g in g_list:
                if g:
                    counter[g] += 1
    if not counter:
        return ("Bilinmiyor", 0)
    return counter.most_common(1)[0]


# ==============================================================================
# SAYFA 1: ANA SAYFA
# ==============================================================================

def render_home_page(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    st.title("🎬 Film Verileri Üzerinde Veri Madenciliği ve Akıllı Film Analiz Sistemi")
    st.markdown(
        """
        Bu platform, **TMDB 5000 Movies** ve **TMDB 5000 Credits** veri setlerini birleştirerek
        gerçek veri madenciliği algoritmaları (**Apriori Birliktelik Kuralı Madenciliği**, **K-Means Kümeleme**,
        **Random Forest Sınıflandırma** ve **TF-IDF & Cosine Similarity İçerik Tabanlı Öneri Sistemi**)
        uygulayan kapsamlı bir karar destek ve analiz uygulamasıdır.
        """
    )

    top_genre_name, top_genre_count = get_most_common_genre(df)
    total_movies = len(df)
    avg_vote = float(df["vote_average"].mean())
    avg_pop = float(df["popularity"].mean())

    # Üst Özet Metrik Kartları
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎞️ Toplam Film Sayısı", f"{total_movies:,}")
    col2.metric("⭐ Ortalama Film Puanı", f"{avg_vote:.2f} / 10")
    col3.metric("🔥 Ortalama Popülerlik", f"{avg_pop:.2f}")
    col4.metric("🎭 En Çok Film Bulunan Tür", f"{top_genre_name} ({top_genre_count:,})")

    st.divider()
    st.subheader("📊 Genel Veri Analizi ve Görselleştirmeler")
    st.caption(
        "Veri setindeki tür, yıl, bütçe, gelir, popülerlik, puan ve yönetmen dağılımlarını aşağıdaki sekmelerden inceleyebilirsiniz."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎭 Türler ve Yıllar",
            "💰 Gelir ve Bütçe Liderleri",
            "🌟 Popülerlik ve Puan Dağılımı",
            "📈 Korelasyon ve İlişki Analizleri",
            "🎬 Yönetmenler ve Tür Gelirleri",
        ]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_top_genres_count(df))
        with c2:
            st.pyplot(plot_movies_by_year(df))

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_top_revenue_movies(df, top_n=10))
        with c2:
            st.pyplot(plot_top_budget_movies(df, top_n=10))

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_top_popular_movies(df, top_n=10))
        with c2:
            st.pyplot(plot_vote_average_distribution(df))

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_budget_vs_revenue(df))
        with c2:
            st.pyplot(plot_popularity_vs_vote_average(df))

    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_top_directors(df, top_n=12))
        with c2:
            st.pyplot(plot_average_revenue_by_genre(df, top_n=15))


# ==============================================================================
# SAYFA 2: VERİ SETİ İNCELEME
# ==============================================================================

def render_dataset_page(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    st.title("🗂️ Veri Seti İnceleme ve Ön İşleme Raporu")
    st.markdown(
        """
        Bu sayfada `tmdb_5000_movies.csv` ve `tmdb_5000_credits.csv` dosyalarının birleştirilmesi,
        JSON sütunlarının ayrıştırılması, eksik verilerin temizlenmesi ve oluşturulan yeni özellikler
        detaylı olarak raporlanmaktadır.
        """
    )

    merge_rep = report.get("merge_report", {})

    st.subheader("1. CSV Birleştirme (Merge) ve Veri Kaybı Raporu")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Movies CSV Kayıt Sayısı", f"{merge_rep.get('movies_row_count', 4803):,}")
    m2.metric("Credits CSV Kayıt Sayısı", f"{merge_rep.get('credits_row_count', 4803):,}")
    m3.metric("Ortak Eşleşen Film (id = movie_id)", f"{merge_rep.get('common_id_count', 4803):,}")
    m4.metric("Birleştirme Veri Kaybı", f"{merge_rep.get('data_loss_count', 0)}")

    st.info(
        f"✅ **Birleştirme Kontrol Özeti:** `pd.merge(movies, credits, left_on='id', right_on='movie_id', how='inner')` "
        f"işlemi sonucunda **{merge_rep.get('merged_row_count', 4803)}** film kaydı eksiksiz birleştirilmiştir. "
        f"Tekrar eden film kimliği sayısı: **{merge_rep.get('duplicate_movie_ids_in_movies', 0)}**. "
        f"İkinci dosyadan gelen mükerrer `title` sütunu kaldırılmış ve veri kaybı yaşanmamıştır."
    )

    st.subheader("2. Veri Temizleme ve Ön İşleme Raporu")
    c1, c2 = st.columns(2)
    with c1:
        cleaning_summary = pd.DataFrame(
            [
                {
                    "Kontrol Edilen Alan": "overview (Film Özeti)",
                    "Tespit Edilen Durum": f"{report['overview_null_filled']} eksik (NaN) kayıt",
                    "Uygulanan Ön İşleme": "Boş metin ('') ile dolduruldu",
                },
                {
                    "Kontrol Edilen Alan": "runtime (Film Süresi)",
                    "Tespit Edilen Durum": f"{report['runtime_null_filled']} eksik (NaN) kayıt",
                    "Uygulanan Ön İşleme": f"Medyan süre ({report['runtime_median_used']:.0f} dk) ile dolduruldu",
                },
                {
                    "Kontrol Edilen Alan": "release_date (Yayın Tarihi)",
                    "Tespit Edilen Durum": f"{report['release_date_null_count']} eksik kayıt",
                    "Uygulanan Ön İşleme": f"release_year çıkarıldı, eksik yıl medyan ({report['release_year_median_used']}) ile tamamlandı",
                },
                {
                    "Kontrol Edilen Alan": "budget (Bütçe = 0)",
                    "Tespit Edilen Durum": f"{report['zero_budget_count']} kayıtta 0 (bilinmeyen), {report['known_budget_count']} bilinen",
                    "Uygulanan Ön İşleme": "Bilinmeyen bütçe olarak işaretlendi; profit_ratio hesabında 0 bütçeler NaN bırakıldı",
                },
                {
                    "Kontrol Edilen Alan": "revenue (Gelir = 0)",
                    "Tespit Edilen Durum": f"{report['zero_revenue_count']} kayıtta 0 (bilinmeyen), {report['known_revenue_count']} bilinen",
                    "Uygulanan Ön İşleme": "Bilinmeyen gelir olarak işaretlendi; Random Forest eğitiminde yalnızca geliri bilinenler kullanıldı",
                },
                {
                    "Kontrol Edilen Alan": "genres & keywords (JSON)",
                    "Tespit Edilen Durum": f"{report['empty_genres_count']} boş tür, {report['empty_keywords_count']} boş anahtar kelime",
                    "Uygulanan Ön İşleme": "JSON ayrıştırıldı; genre_list ve keyword_list oluşturuldu",
                },
                {
                    "Kontrol Edilen Alan": "cast & crew (JSON)",
                    "Tespit Edilen Durum": f"{report['empty_cast_count']} boş oyuncu, {report['empty_director_count']} yönetmensiz kayıt",
                    "Uygulanan Ön İşleme": "İlk 5 oyuncu (cast_list) ve Director (director) çıkarıldı",
                },
            ]
        )
        st.dataframe(cleaning_summary, width="stretch", hide_index=True)

    with c2:
        st.markdown("**Türetilen Yeni Özellikler (`cleaned_movies.csv`):**")
        new_cols_df = pd.DataFrame(
            [
                {"Yeni Sütun": "release_year", "Açıklama": "release_date sütunundan çıkarılan yayın yılı"},
                {"Yeni Sütun": "profit", "Açıklama": "Net kâr (revenue - budget)"},
                {"Yeni Sütun": "profit_ratio", "Açıklama": "Gelir / Bütçe oranı (bütçe > 0 olan filmler için)"},
                {"Yeni Sütun": "genre_list", "Açıklama": "JSON genres sütunundan ayrıştırılan tür listesi"},
                {"Yeni Sütun": "keyword_list", "Açıklama": "JSON keywords sütunundan ayrıştırılan anahtar kelimeler"},
                {"Yeni Sütun": "cast_list", "Açıklama": "JSON cast sütunundan çıkarılan ilk 5 oyuncu listesi"},
                {"Yeni Sütun": "director", "Açıklama": "JSON crew sütunundan çıkarılan yönetmen (job == Director)"},
                {"Yeni Sütun": "budget_log, revenue_log, popularity_log", "Açıklama": "Sağa çarpık dağılımlar için np.log1p dönüşümleri"},
            ]
        )
        st.dataframe(new_cols_df, width="stretch", hide_index=True)

    st.subheader("3. Sütun Bilgileri ve Eksik Veri Analizi")
    col_info_rows = []
    initial_nulls = report.get("initial_nulls", {})
    for col in df.columns:
        col_info_rows.append(
            {
                "Sütun Adı": col,
                "Veri Tipi": str(df[col].dtype),
                "Dolu Kayıt Sayısı": int(df[col].notnull().sum()),
                "Temizleme Öncesi Eksik": int(initial_nulls.get(col, 0)),
                "Temizleme Sonrası Eksik": int(df[col].isnull().sum()),
            }
        )
    st.dataframe(pd.DataFrame(col_info_rows), width="stretch", hide_index=True)

    st.subheader("4. Temizlenmiş Veri Tablosu (`cleaned_movies.csv`)")
    search_term = st.text_input("🔍 Film Adı veya Yönetmene Göre Tabloda Ara:", "")
    display_cols = [
        "id",
        "title",
        "release_year",
        "genres_str",
        "director",
        "cast_str",
        "budget",
        "revenue",
        "profit",
        "profit_ratio",
        "popularity",
        "runtime",
        "vote_average",
        "vote_count",
    ]
    view_df = df[display_cols].copy()
    if search_term.strip():
        q = search_term.strip().lower()
        view_df = view_df[
            view_df["title"].astype(str).str.lower().str.contains(q, regex=False)
            | view_df["director"].astype(str).str.lower().str.contains(q, regex=False)
        ]
    st.caption(f"Gösterilen kayıt sayısı: {len(view_df):,}")
    st.dataframe(view_df.head(500), width="stretch", hide_index=True)


# ==============================================================================
# SAYFA 3: APRIORI BİRLİKTELİK ANALİZİ
# ==============================================================================

def render_apriori_page(df: pd.DataFrame) -> None:
    st.title("🔗 Apriori Birliktelik Kuralı Madenciliği")
    st.markdown(
        """
        **Amaç:** Her filmi bir işlem (*transaction*) olarak kabul ederek hangi film türlerinin
        ve anahtar kelimelerin birlikte görülme eğiliminde olduğunu (`Support`, `Confidence`, `Lift`
        ölçütleriyle) keşfetmektir.
        """
    )

    with st.expander("⚙️ Algoritma Parametreleri (Minimum Support, Confidence ve Lift)", expanded=True):
        p1, p2, p3, p4 = st.columns(4)
        min_support = p1.slider(
            "Minimum Support (Destek)",
            min_value=0.005,
            max_value=0.20,
            value=0.02,
            step=0.005,
            help="Bir öğe kümesinin tüm filmler içinde görülme oranı eşiği (varsayılan: 0.02).",
        )
        min_confidence = p2.slider(
            "Minimum Confidence (Güven)",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help="Öncül görüldüğünde ardılın görülme olasılığı eşiği (varsayılan: 0.30).",
        )
        min_lift = p3.slider(
            "Minimum Lift Eşiği",
            min_value=1.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Lift > 1.0 pozitif birlikteliği ifade eder (varsayılan: 1.0).",
        )
        include_keywords = p4.checkbox(
            "Anahtar Kelimeleri Dahil Et",
            value=True,
            help="İşaretlenirse film türlerinin yanı sıra sık görülen anahtar kelimeler de analize dahil edilir.",
        )

    try:
        res = cached_apriori(
            _df=df,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            include_keywords=include_keywords,
        )
    except Exception as exc:
        st.error(f"Apriori analizi sırasında hata oluştu: {exc}")
        return

    if res["status"] != "success":
        st.warning(res["message"])
        return

    tx_info = res["transaction_info"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("İşlenen Film (Transaction)", f"{tx_info['valid_transactions']:,}")
    k2.metric("Sık Öğe Kümesi Sayısı", f"{len(res['frequent_itemsets']):,}")
    k3.metric("Üretilen Birliktelik Kuralı", f"{len(res['rules']):,}")
    k4.metric("En Yüksek Lift Değeri", f"{res['rules']['lift'].max():.2f}")

    st.subheader("1. En Sık Görülen Film Türleri ve Tür Kombinasyonları")
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(
            plot_apriori_frequent_items(
                res["single_items"], "En Sık Görülen Tekil Film Türleri / Öğeler", top_n=12
            )
        )
    with g2:
        st.pyplot(
            plot_apriori_frequent_items(
                res["combo_items"], "En Sık Görülen Tür / Öğe Kombinasyonları", top_n=12
            )
        )

    st.subheader("2. En Güçlü 10 Birliktelik Kuralı")
    top10_display = res["top_10_rules"][
        ["rule_str", "antecedents_str", "consequents_str", "support", "confidence", "lift"]
    ].rename(
        columns={
            "rule_str": "Birliktelik Kuralı",
            "antecedents_str": "Öncül (Antecedents)",
            "consequents_str": "Ardıl (Consequents)",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
        }
    )
    st.dataframe(
        top10_display.style.format(
            {"Support": "{:.4f}", "Confidence": "{:.4f}", "Lift": "{:.4f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("3. Birliktelik Kurallarının Grafiksel Gösterimi")
    v1, v2 = st.columns(2)
    with v1:
        st.pyplot(plot_top_rules_bar(res["top_10_rules"]))
    with v2:
        st.pyplot(plot_apriori_rules_scatter(res["rules"]))

    if include_keywords and not res["genre_keyword_rules"].empty:
        st.subheader("4. Anahtar Kelime ve Film Türü İlişki Kuralları")
        kw_display = res["genre_keyword_rules"][
            ["rule_str", "support", "confidence", "lift"]
        ].head(15).rename(
            columns={
                "rule_str": "Anahtar Kelime - Tür Kuralı",
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
            }
        )
        st.dataframe(
            kw_display.style.format(
                {"Support": "{:.4f}", "Confidence": "{:.4f}", "Lift": "{:.4f}"}
            ),
            width="stretch",
            hide_index=True,
        )


# ==============================================================================
# SAYFA 4: K-MEANS KÜMELEME
# ==============================================================================

def render_kmeans_page(df: pd.DataFrame) -> None:
    st.title("🎯 K-Means Kümeleme Analizi")
    st.markdown(
        """
        **Amaç:** Filmleri bütçe (`budget`), gelir (`revenue`), popülerlik (`popularity`),
        süre (`runtime`), ortalama puan (`vote_average`), oy sayısı (`vote_count`) ve film türlerine
        göre benzer gruplara (kümelere) ayırmak ve her kümenin profilini çıkarmaktır.
        """
    )

    with st.expander("⚙️ K-Means Kümeleme Parametreleri", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_clusters = c1.slider(
            "Küme Sayısı (k)",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
            help="Filmlerin kaç gruba ayrılacağını belirler (2-8 aralığı).",
        )
        filter_zero_fin = c2.checkbox(
            "Sadece Bütçe ve Geliri Bilinen Filmleri Kümele",
            value=True,
            help="İşaretlenirse bütçe ve geliri > 0 olan filmler kümelenir. Kaldırılırsa 0 olanlar medyan ile doldurulup tüm 4.803 film kümelenir.",
        )
        include_genres = c3.checkbox(
            "Film Türlerini (MultiLabelBinarizer) Dahil Et",
            value=True,
            help="Sayısal özelliklerin yanı sıra film türlerini çoklu ikili özellik olarak kümelemeye katar.",
        )

    try:
        res = cached_kmeans(
            _df=df,
            n_clusters=n_clusters,
            filter_zero_financials=filter_zero_fin,
            include_genres=include_genres,
        )
    except Exception as exc:
        st.error(f"K-Means analizi sırasında hata oluştu: {exc}")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kümelenen Film Sayısı", f"{res['prep_info']['clustered_movie_count']:,}")
    m2.metric("Seçilen Küme Sayısı (k)", f"{res['selected_k']}")
    m3.metric("Silhouette Score", f"{res['selected_silhouette']:.4f}")
    m4.metric("En Yüksek Silhouette Veren k", f"k = {res['optimal_k_by_silhouette']}")

    st.info(f"ℹ️ **Veri Hazırlama Bilgisi:** {res['prep_info']['financial_handling']}")

    st.subheader("1. Elbow Method (Dirsek Yöntemi) ve Silhouette Score Analizi (k = 2..8)")
    st.pyplot(plot_elbow_and_silhouette(res["evaluation_df"], res["selected_k"]))

    st.subheader("2. Kümelerin Açıklayıcı Profilleri ve Ortalama Değerleri")
    sum_display = res["summary_df"].rename(
        columns={
            "cluster_id": "Küme No",
            "cluster_name": "Açıklayıcı Küme Adı",
            "movie_count": "Film Sayısı",
            "percentage": "Oran (%)",
            "avg_budget": "Ortalama Bütçe ($)",
            "avg_revenue": "Ortalama Gelir ($)",
            "avg_profit": "Ortalama Kâr ($)",
            "avg_popularity": "Ortalama Popülerlik",
            "avg_vote_average": "Ortalama Puan",
            "avg_vote_count": "Ortalama Oy Sayısı",
            "avg_runtime": "Ortalama Süre (dk)",
            "top_genres": "Baskın Türler",
        }
    )
    st.dataframe(
        sum_display.style.format(
            {
                "Oran (%)": "%{:.1f}",
                "Ortalama Bütçe ($)": "${:,.0f}",
                "Ortalama Gelir ($)": "${:,.0f}",
                "Ortalama Kâr ($)": "${:,.0f}",
                "Ortalama Popülerlik": "{:.2f}",
                "Ortalama Puan": "{:.2f}",
                "Ortalama Oy Sayısı": "{:,.0f}",
                "Ortalama Süre (dk)": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("3. Kümelerin Grafiksel Karşılaştırması")
    st.pyplot(plot_kmeans_cluster_comparison(res["summary_df"]))
    st.pyplot(plot_kmeans_clusters_pca(res["clustered_df"]))

    st.subheader("4. Kümedeki Filmleri İncele")
    cluster_options = res["summary_df"]["cluster_name"].tolist()
    selected_cluster_label = st.selectbox(
        "Listelemek istediğiniz kümeyi seçin:", cluster_options
    )
    clustered_df = res["clustered_df"]
    cluster_movies = (
        clustered_df[clustered_df["cluster_name"] == selected_cluster_label]
        .sort_values(by="popularity", ascending=False)
        .copy()
    )
    st.caption(
        f"**{selected_cluster_label}** içerisinde toplam **{len(cluster_movies)}** film bulunmaktadır."
    )
    st.dataframe(
        cluster_movies[
            [
                "title",
                "release_year",
                "genres_str",
                "director",
                "budget",
                "revenue",
                "popularity",
                "vote_average",
                "vote_count",
            ]
        ].rename(
            columns={
                "title": "Film Adı",
                "release_year": "Yıl",
                "genres_str": "Türler",
                "director": "Yönetmen",
                "budget": "Bütçe ($)",
                "revenue": "Gelir ($)",
                "popularity": "Popülerlik",
                "vote_average": "Puan",
                "vote_count": "Oy Sayısı",
            }
        ),
        width="stretch",
        hide_index=True,
    )


# ==============================================================================
# SAYFA 5: RANDOM FOREST TAHMİN
# ==============================================================================

def render_random_forest_page(df: pd.DataFrame) -> None:
    st.title("🌲 Random Forest ile Film Gelir Sınıfı Tahmini")
    st.markdown(
        """
        **Amaç:** Bir filmin mevcut özelliklerine (`budget`, `popularity`, `runtime`, `vote_average`,
        `vote_count`, `release_year`, `genres`) göre **Yüksek Gelir** (medyan gelirin üzerinde) mi yoksa
        **Düşük Gelir** (medyan gelirin altında/eşit) mi elde edeceğini tahmin etmektir.
        *(Hedef değişken sızıntısını önlemek için `revenue` ve `profit` tahmin özelliklerine dahil edilmemiştir.)*
        """
    )

    with st.expander("⚙️ Random Forest Model Parametreleri", expanded=True):
        p1, p2, p3 = st.columns(3)
        n_estimators = p1.slider(
            "Ağaç Sayısı (n_estimators)",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Random Forest içindeki karar ağacı sayısı (varsayılan: 200).",
        )
        max_depth_opt = p2.selectbox(
            "Maksimum Derinlik (max_depth)",
            options=["Sınırsız (None)", 8, 12, 16, 20],
            index=0,
        )
        max_depth = None if max_depth_opt == "Sınırsız (None)" else int(max_depth_opt)
        test_size = p3.slider(
            "Test Verisi Oranı",
            min_value=0.15,
            max_value=0.30,
            value=0.20,
            step=0.05,
            help="Varsayılan: %80 Eğitim, %20 Test (stratify=y).",
        )

    try:
        res = cached_random_forest(
            _df=df,
            n_estimators=n_estimators,
            max_depth=max_depth,
            test_size=test_size,
            class_weight="balanced",
        )
    except Exception as exc:
        st.error(f"Random Forest modeli çalıştırılırken hata oluştu: {exc}")
        return

    metrics = res["metrics"]
    st.info(
        f"📌 **Veri Bölme ve Hedef Özeti:** Geliri bilinen **{res['total_known_movies']:,}** film kullanıldı "
        f"(geliri 0 olan **{res['removed_unknown_revenue']:,}** film çıkarıldı). "
        f"Medyan Gelir Eşiği: **${res['median_revenue']:,.0f}**. "
        f"Eğitim Seti: **{res['train_size']:,}** film | Test Seti: **{res['test_size']:,}** film."
    )

    st.subheader("1. Test Verisi Üzerinde Model Performans Ölçütleri")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🎯 Accuracy (Doğruluk)", f"%{metrics['accuracy'] * 100:.2f}")
    k2.metric("✔️ Precision (Kesinlik)", f"%{metrics['precision'] * 100:.2f}")
    k3.metric("📡 Recall (Duyarlılık)", f"%{metrics['recall'] * 100:.2f}")
    k4.metric("⚖️ F1-Score", f"%{metrics['f1_score'] * 100:.2f}")
    k5.metric("📈 ROC-AUC Skoru", f"%{metrics['roc_auc'] * 100:.2f}")

    st.subheader("2. Confusion Matrix ve Özellik Önem Düzeyleri (Feature Importance)")
    c1, c2 = st.columns([1, 1.35])
    with c1:
        st.pyplot(plot_confusion_matrix_heatmap(res["confusion_matrix_df"]))
        st.dataframe(
            res["class_report_df"].style.format(
                {
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-Score": "{:.4f}",
                    "Film Sayısı": "{:.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with c2:
        st.pyplot(plot_feature_importance(res["feature_importance_df"], top_n=12))

    st.subheader("3. Test Verisi Üzerindeki Gerçek ve Tahmin Edilen Sonuçlar")
    st.dataframe(
        res["test_predictions_df"].head(100),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("🧪 4. Yeni Film İçin Canlı Gelir Sınıfı Tahmin Simülatörü")
    st.caption(
        "Aşağıdaki özellikleri değiştirerek eğitilmiş Random Forest modelinin bir film için nasıl tahmin ürettiğini test edebilirsiniz."
    )
    all_genres = sorted(list(res["mlb"].classes_))
    s1, s2, s3 = st.columns(3)
    sim_budget = s1.number_input(
        "Bütçe ($)", min_value=0, max_value=400_000_000, value=45_000_000, step=5_000_000
    )
    sim_pop = s2.number_input(
        "Popülerlik Skoru", min_value=0.1, max_value=500.0, value=35.0, step=5.0
    )
    sim_runtime = s3.number_input(
        "Film Süresi (Dakika)", min_value=40, max_value=250, value=115, step=5
    )

    s4, s5, s6 = st.columns(3)
    sim_vote_avg = s4.slider("Ortalama Puan (vote_average)", 1.0, 10.0, 6.8, 0.1)
    sim_vote_cnt = s5.number_input(
        "Oy Sayısı (vote_count)", min_value=1, max_value=20000, value=1200, step=100
    )
    sim_year = s6.number_input(
        "Yayın Yılı", min_value=1930, max_value=2026, value=2015, step=1
    )

    sim_genres = st.multiselect(
        "Film Türleri",
        options=all_genres,
        default=[g for g in ["Action", "Adventure", "Science Fiction"] if g in all_genres],
    )

    if st.button("🚀 Gelir Sınıfını Tahmin Et", type="primary"):
        pred_out = predict_single_movie_revenue_class(
            rf_result=res,
            budget=float(sim_budget),
            popularity=float(sim_pop),
            runtime=float(sim_runtime),
            vote_average=float(sim_vote_avg),
            vote_count=int(sim_vote_cnt),
            release_year=int(sim_year),
            genres=sim_genres,
        )
        if pred_out["predicted_label"] == "Yüksek Gelir":
            st.success(
                f"🏆 **Tahmin Sonucu: YÜKSEK GELİR** "
                f"(Medyan eşik olan ${pred_out['median_threshold']:,.0f} üzerinde gelir beklentisi — "
                f"Olasılık: **%{pred_out['high_revenue_prob'] * 100:.1f}**)"
            )
        else:
            st.warning(
                f"📉 **Tahmin Sonucu: DÜŞÜK GELİR** "
                f"(Medyan eşik olan ${pred_out['median_threshold']:,.0f} altında gelir beklentisi — "
                f"Düşük Gelir Olasılığı: **%{pred_out['low_revenue_prob'] * 100:.1f}**)"
            )


# ==============================================================================
# SAYFA 6: FİLM ÖNERİ SİSTEMİ
# ==============================================================================

def render_recommendation_page(df: pd.DataFrame) -> None:
    st.title("🍿 TF-IDF ve Cosine Similarity ile İçerik Tabanlı Film Öneri Sistemi")
    st.markdown(
        """
        **Amaç:** Kullanıcının seçtiği filme yalnızca türüne göre değil; **film açıklaması (`overview`)**,
        **türleri (`genres`)**, **anahtar kelimeleri (`keywords`)**, **ilk 5 oyuncusu (`cast_list`)** ve
        **yönetmeni (`director`)** üzerinden hesaplanan **TF-IDF vektörleri** ve **Kosinüs Benzerliği (Cosine Similarity)**
        ile en benzer 10 filmi önermektir.
        """
    )

    engine = cached_recommendation_engine(df)

    c_search, c_select = st.columns([1, 1.5])
    with c_search:
        search_query = st.text_input(
            "🔎 Film Ara (İsteğe Bağlı):",
            placeholder="Örn: Avatar, Dark Knight, Inception, Matrix...",
        )

    all_titles = df["title"].dropna().astype(str).tolist()
    if search_query.strip():
        filtered_titles = search_movies_by_query(df, search_query, limit=50)
        if not filtered_titles:
            st.warning(f"'{search_query}' aramasına uygun film bulunamadı. Tüm liste gösteriliyor.")
            filtered_titles = all_titles
    else:
        filtered_titles = all_titles

    with c_select:
        default_idx = 0
        if not search_query.strip() and "Avatar" in filtered_titles:
            default_idx = filtered_titles.index("Avatar")
        selected_title = st.selectbox(
            "🎬 Öneri Almak İstediğiniz Filmi Seçin:",
            options=filtered_titles,
            index=default_idx,
        )

    try:
        rec_res = get_movie_recommendations(engine, selected_title, top_n=10)
    except Exception as exc:
        st.error(f"Film önerisi hesaplanırken hata oluştu: {exc}")
        return

    sel_movie = rec_res["selected_movie"]
    rec_df = rec_res["recommendations_df"]

    st.subheader(f"📌 Seçilen Film: {sel_movie['title']} ({sel_movie['release_year']})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⭐ Ortalama Puan", f"{sel_movie['vote_average']:.1f} / 10 ({sel_movie['vote_count']:,} oy)")
    m2.metric("🎬 Yönetmen", sel_movie["director"])
    m3.metric("💰 Bütçe / Gelir", f"${sel_movie['budget']/1e6:.0f}M / ${sel_movie['revenue']/1e6:.0f}M")
    m4.metric("⏱️ Süre", f"{sel_movie['runtime']:.0f} dk")

    st.markdown(f"**🎭 Türler:** {sel_movie['genres'] or 'Belirtilmemiş'}")
    st.markdown(f"**👥 İlk 5 Oyuncu:** {sel_movie['cast'] or 'Belirtilmemiş'}")
    st.markdown(f"**🏷️ Anahtar Kelimeler:** {sel_movie['keywords'] or 'Belirtilmemiş'}")
    st.markdown(f"**📝 Özet (Overview):** {sel_movie['overview'] or 'Özet bulunmuyor.'}")

    st.divider()
    st.subheader(f"✨ '{sel_movie['title']}' Filmine En Benzer 10 Film")

    display_rec = rec_df[
        [
            "Sıra",
            "Film Adı",
            "Benzerlik Yüzdesi (%)",
            "Türler",
            "Ortalama Puan",
            "Yayın Yılı",
            "Yönetmen",
            "Ortak Unsurlar",
        ]
    ]
    st.dataframe(
        display_rec.style.format(
            {
                "Benzerlik Yüzdesi (%)": "%{:.2f}",
                "Ortalama Puan": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.pyplot(plot_recommendations_similarity(rec_df, sel_movie["title"]))


# ==============================================================================
# ANA UYGULAMA AKIŞI VE SOL MENÜ
# ==============================================================================

def main() -> None:
    st.sidebar.title("🎬 Akıllı Film Analiz")
    st.sidebar.caption("TMDB 5000 Veri Madenciliği Sistemi")

    menu_options = [
        "1. Ana Sayfa",
        "2. Veri Seti İnceleme",
        "3. Apriori Birliktelik Analizi",
        "4. K-Means Kümeleme",
        "5. Random Forest Tahmin",
        "6. Film Öneri Sistemi",
    ]
    selected_page = st.sidebar.radio("📌 Menü", menu_options)

    try:
        df, report = cached_load_data()
    except FileNotFoundError as fnf_err:
        st.error(f"Veri dosyası hatası: {fnf_err}")
        st.stop()
    except Exception as err:
        st.error(f"Veri yükleme ve ön işleme sırasında beklenmeyen hata: {err}")
        st.stop()

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
        **📊 Veri Seti Özeti**
        - **Toplam Film:** `{len(df):,}`
        - **Geliri Bilinen:** `{report['known_revenue_count']:,}`
        - **Bütçesi Bilinen:** `{report['known_budget_count']:,}`
        - **Temiz Veri:** `data/cleaned_movies.csv`
        """
    )

    if selected_page == "1. Ana Sayfa":
        render_home_page(df, report)
    elif selected_page == "2. Veri Seti İnceleme":
        render_dataset_page(df, report)
    elif selected_page == "3. Apriori Birliktelik Analizi":
        render_apriori_page(df)
    elif selected_page == "4. K-Means Kümeleme":
        render_kmeans_page(df)
    elif selected_page == "5. Random Forest Tahmin":
        render_random_forest_page(df)
    elif selected_page == "6. Film Öneri Sistemi":
        render_recommendation_page(df)


if __name__ == "__main__":
    main()
