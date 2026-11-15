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

from collections import Counter
import html
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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
from src.poster_carousel import (
    get_recommended_movies_with_posters,
    get_top_rated_movies_with_posters,
    render_poster_carousel,
)
from src.random_forest import (
    predict_single_movie_revenue_class,
    run_random_forest_classification,
)
from src.ui_icons import (
    APRIORI_SVG,
    BRAND_LOGO_SVG,
    CARD_BUDGET_SVG,
    CARD_FILE_SVG,
    CARD_MOVIE_SVG,
    CARD_REV_SVG,
    DATASET_SVG,
    HOME_SVG,
    KMEANS_SVG,
    PIN_SVG,
    REC_SVG,
    RF_SVG,
    SUMMARY_CHART_SVG,
    get_brand_favicon_path,
    get_sidebar_custom_css,
    svg_to_data_uri,
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
    page_title="Film Analiz Platformu",
    page_icon=get_brand_favicon_path(),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# TASARIM SİSTEMİ VE YARDIMCI GÖRSEL BİLEŞENLER
# ==============================================================================

def inject_custom_styles() -> None:
    """
    Okunabilirliği artıran, gereksiz görsel karmaşadan uzak, tutarlı boşluk ve
    kart hiyerarşisi sağlayan tasarım sistemi stillerini uygular.
    """
    st.markdown(
        """
        <style>
        /* Ana kapsayıcı boşluk düzeni */
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1400px;
        }

        /* Metrik kartlarının hizalaması ve tipografisi */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetricLabel"] > div {
            font-size: 0.84rem !important;
            font-weight: 600 !important;
            color: #475569 !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
        }

        /* Sayfa üst başlık bileşeni */
        .page-header-box {
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 0.9rem;
            margin-bottom: 1.2rem;
        }
        .page-badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            color: #1D4ED8;
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 4px;
            padding: 0.15rem 0.55rem;
            margin-bottom: 0.4rem;
            letter-spacing: 0.02em;
        }
        .page-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0 0 0.35rem 0;
            line-height: 1.25;
        }
        .page-subtitle {
            font-size: 0.95rem;
            color: #475569;
            margin: 0;
            line-height: 1.5;
        }

        /* Teknik başlık altı açıklama kutusu (Nedir & Neyi Etkiler) */
        .section-heading-wrap {
            margin-top: 1.1rem;
            margin-bottom: 0.7rem;
        }
        .section-heading-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0 0 0.4rem 0;
            line-height: 1.35;
        }
        .section-heading-title-sm {
            font-size: 0.98rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0 0 0.35rem 0;
            line-height: 1.35;
        }
        .header-info-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 3.5px solid #2563EB;
            border-radius: 6px;
            padding: 0.6rem 0.9rem;
            margin-bottom: 0.75rem;
            font-size: 0.855rem;
            color: #334155;
            line-height: 1.48;
        }
        .header-info-row {
            margin: 0.15rem 0;
        }
        .header-info-tag {
            display: inline-block;
            font-size: 0.74rem;
            font-weight: 700;
            color: #1E40AF;
            background-color: #DBEAFE;
            border-radius: 4px;
            padding: 0.05rem 0.42rem;
            margin-right: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .header-info-tag-impact {
            color: #0F766E;
            background-color: #CCFBF1;
        }

        /* Bilgi ve bulgu kutusu */
        .insight-box {
            background-color: #F8FAFC;
            border-left: 4px solid #2563EB;
            border-top: 1px solid #E2E8F0;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
            border-radius: 0 6px 6px 0;
            padding: 0.8rem 1rem;
            margin: 0.65rem 0 1rem 0;
            font-size: 0.9rem;
            color: #1E293B;
            line-height: 1.5;
        }
        .insight-box strong {
            color: #0F172A;
        }

        /* Etiket (chip/tag) stili */
        .tag-chip {
            display: inline-block;
            background-color: #F1F5F9;
            color: #334155;
            border: 1px solid #CBD5E1;
            border-radius: 4px;
            padding: 0.12rem 0.5rem;
            font-size: 0.78rem;
            font-weight: 500;
            margin-right: 0.35rem;
            margin-bottom: 0.3rem;
        }
        .tag-chip-primary {
            background-color: #EFF6FF;
            color: #1E40AF;
            border-color: #BFDBFE;
        }
        .tag-chip-success {
            background-color: #F0FDF4;
            color: #166534;
            border-color: #BBF7D0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_info_explainer(what_it_is: str, what_it_affects: str) -> None:
    """
    Yalnızca teknik/algoritmik bölümlerin altında 'Nedir?' ve 'Neyi Etkiler?'
    sorularını yanıtlayan bilgi kutusu bileşenini oluşturur.
    """
    st.markdown(
        f"""
        <div class="header-info-box">
            <div class="header-info-row">
                <span class="header-info-tag">📌 Nedir?</span>
                <span>{html.escape(what_it_is)}</span>
            </div>
            <div class="header-info-row">
                <span class="header-info-tag header-info-tag-impact">⚡ Neyi Etkiler?</span>
                <span>{html.escape(what_it_affects)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(
    title: str,
    what_it_is: str,
    what_it_affects: str,
    small: bool = False,
) -> None:
    """Teknik bölüm alt başlığını ve hemen altındaki 'Nedir / Neyi Etkiler' bilgi kutusunu birlikte çizer."""
    cls = "section-heading-title-sm" if small else "section-heading-title"
    st.markdown(
        f"""
        <div class="section-heading-wrap">
            <div class="{cls}">{html.escape(title)}</div>
            <div class="header-info-box">
                <div class="header-info-row">
                    <span class="header-info-tag">📌 Nedir?</span>
                    <span>{html.escape(what_it_is)}</span>
                </div>
                <div class="header-info-row">
                    <span class="header-info-tag header-info-tag-impact">⚡ Neyi Etkiler?</span>
                    <span>{html.escape(what_it_affects)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    badge: str = "",
    icon_svg: Optional[str] = None,
) -> None:
    """Her panelin başında sol menüdeki özel SVG ikonu, kategori rozetini ve açıklamayı gösterir."""
    badge_html = f'<div class="page-badge">{html.escape(badge)}</div>' if badge else ""
    if icon_svg:
        icon_uri = svg_to_data_uri(icon_svg)
        title_html = (
            f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.35rem;">'
            f'<img src="{icon_uri}" style="width:40px; height:40px; flex-shrink:0;" alt="Sayfa İkonu" />'
            f'<h1 class="page-title" style="margin:0;">{html.escape(title)}</h1>'
            f"</div>"
        )
    else:
        title_html = f'<h1 class="page-title">{html.escape(title)}</h1>'

    st.markdown(
        f"""
        <div class="page-header-box">
            {badge_html}
            {title_html}
            <p class="page-subtitle">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_box(title: str, content: str) -> None:
    """Grafik veya tabloların altına anlaşılır özet yorum kutusu ekler."""
    st.markdown(
        f"""
        <div class="insight-box">
            <strong>💡 {html.escape(title)}:</strong> {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart(fig: Any, key: Optional[str] = None) -> None:
    """Plotly grafiklerini tutarlı genişlik ve temiz araç çubuğu ile ekrana çizer."""
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config={"displayModeBar": "hover", "displaylogo": False},
        key=key,
    )


def format_currency_short(val: float) -> str:
    """Dolar değerlerini okunabilir kısa formata ($395.1B, $72.8M vb.) çevirir."""
    if pd.isna(val) or val == 0:
        return "$0"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.1f}M"
    if abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.0f}K"
    return f"{sign}${abs_val:,.0f}"


def render_tags_html(items_str: str, primary: bool = False) -> str:
    """Virgülle ayrılmış metinleri şık rozet (chip) etiketlerine dönüştürür."""
    if not items_str or not str(items_str).strip():
        return '<span class="tag-chip">Belirtilmemiş</span>'
    parts = [p.strip() for p in str(items_str).split(",") if p.strip()]
    cls = "tag-chip tag-chip-primary" if primary else "tag-chip"
    return "".join(f'<span class="{cls}">{html.escape(p)}</span>' for p in parts)


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


def get_all_unique_genres(df: pd.DataFrame) -> List[str]:
    """Veri setindeki tüm tekil film türlerini alfabetik döndürür."""
    genres_set = set()
    for g_list in df["genre_list"]:
        if isinstance(g_list, list):
            for g in g_list:
                if g:
                    genres_set.add(g)
    return sorted(list(genres_set))


# ==============================================================================
# SAYFA 1: ANA SAYFA (GENEL BAKIŞ VE YÖNETİCİ PANELİ)
# ==============================================================================

def render_home_page(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    render_page_header(
        title="Ana Sayfa — Film Analiz Platformu",
        subtitle=(
            "TMDB 5000 veri seti üzerinde tür dağılımları, finansal performans, izleyici etkileşimi "
            "ve veri madenciliği modellerinin genel özeti."
        ),
        badge="GENEL BAKIŞ PANELİ",
        icon_svg=HOME_SVG,
    )

    # Hızlı Filtre Çubuğu
    all_genres = get_all_unique_genres(df)
    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())

    with st.container(border=True):
        f_col1, f_col2, f_col3 = st.columns([1.4, 1.8, 1.0])
        with f_col1:
            selected_years = st.slider(
                "📅 Yayın Yılı Aralığı",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                help="Grafikleri ve özet metrikleri belirli bir dönem için filtreler.",
            )
        with f_col2:
            selected_genres = st.multiselect(
                "🎭 Film Türü Filtresi (Boş = Tüm Türler)",
                options=all_genres,
                default=[],
                placeholder="Tür seçin (Örn: Action, Drama, Science Fiction)...",
            )
        with f_col3:
            only_known_fin = st.checkbox(
                "💰 Sadece Gişe Geliri Bilinenler",
                value=False,
                help="İşaretlenirse yalnızca gişe geliri > $0 olan filmler gösterilir.",
            )
            st.caption(f"Veri kaynağı: `cleaned_movies.csv` ({len(df):,} kayıt)")

    # Filtre uygula
    filtered_df = df[
        (df["release_year"] >= selected_years[0]) & (df["release_year"] <= selected_years[1])
    ].copy()
    if selected_genres:
        sel_set = set(selected_genres)
        filtered_df = filtered_df[
            filtered_df["genre_list"].apply(
                lambda gl: bool(sel_set.intersection(gl)) if isinstance(gl, list) else False
            )
        ]
    if only_known_fin:
        filtered_df = filtered_df[filtered_df["revenue"] > 0]

    if filtered_df.empty:
        st.warning("Seçilen filtre kriterlerine uygun film bulunamadı. Lütfen filtreleri genişletin.")
        return

    # Üst Özet Metrik Kartları (5 KPI)
    top_genre_name, top_genre_count = get_most_common_genre(filtered_df)
    total_movies = len(filtered_df)
    avg_vote = float(filtered_df["vote_average"].mean())
    median_vote = float(filtered_df["vote_average"].median())
    known_rev_df = filtered_df[filtered_df["revenue"] > 0]
    total_rev = float(known_rev_df["revenue"].sum()) if not known_rev_df.empty else 0.0
    avg_rev = float(known_rev_df["revenue"].mean()) if not known_rev_df.empty else 0.0
    known_bud_df = filtered_df[filtered_df["budget"] > 0]
    avg_bud = float(known_bud_df["budget"].mean()) if not known_bud_df.empty else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "🎞️ İncelenen Film",
        f"{total_movies:,}",
        delta=f"Toplamın %{total_movies / len(df) * 100:.0f}'i",
        delta_color="off",
    )
    k2.metric(
        "💵 Toplam Gişe Hacmi",
        format_currency_short(total_rev),
        delta=f"Ort: {format_currency_short(avg_rev)} / film",
        delta_color="off",
    )
    k3.metric(
        "🎬 Ortalama Yapım Bütçesi",
        format_currency_short(avg_bud),
        delta=f"{len(known_bud_df):,} bütçesi bilinen film",
        delta_color="off",
    )
    k4.metric(
        "⭐ Ortalama Puan",
        f"{avg_vote:.2f} / 10",
        delta=f"Medyan: {median_vote:.1f}",
        delta_color="off",
    )
    k5.metric(
        "🎭 En Yaygın Tür",
        top_genre_name,
        delta=f"{top_genre_count:,} film (%{top_genre_count / total_movies * 100:.0f})",
        delta_color="off",
    )

    top_poster_items = get_top_rated_movies_with_posters(filtered_df, top_n=15, min_votes=1500)
    if top_poster_items:
        vitrin_count = len(top_poster_items)
        if vitrin_count >= 15:
            header_suffix = "Top 15"
        elif vitrin_count > 1:
            header_suffix = f"Top {vitrin_count}"
        else:
            header_suffix = "Öne Çıkan Film"

        st.markdown(f"#### 🏆 En Yüksek Puanlı Klasik Filmler Vitrini (IMDb / TMDB {header_suffix})")
        if vitrin_count >= 6:
            st.caption(
                f"Filtrelenen yapımlar arasından en yüksek puanlı {vitrin_count} film — Fareyle afişlerin üzerine gelerek kaymayı durdurabilir, büyütebilir veya `<` `>` oklarıyla manuel kaydırabilirsiniz."
            )
        elif vitrin_count > 1:
            st.caption(
                f"Filtrelenen yapımlar arasından en yüksek puanlı {vitrin_count} film — Fareyle afişlerin üzerine gelerek detayları inceleyebilirsiniz."
            )
        else:
            st.caption(
                "Filtrelenen aralıktaki yapım — Fareyle afişin üzerine gelerek detayları inceleyebilirsiniz."
            )
        render_poster_carousel(top_poster_items, height=335, speed_seconds=70)

    st.markdown("#### 📊 Keşifsel Veri Analizi ve İnteraktif Grafikler")
    st.caption(
        "Grafiklerin üzerine gelerek (hover) her bir filmin bütçe, gelir, puan ve yönetmen detaylarını inceleyebilirsiniz."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎭 Türler ve Üretim Trendi",
            "💰 Gişe ve Bütçe Liderleri",
            "🌟 Popülerlik ve Puan Dağılımı",
            "📈 Bütçe - Gelir - Puan İlişkisi",
            "🎬 Yönetmenler ve Tür Kârlılığı",
        ]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            render_chart(plot_top_genres_count(filtered_df), key="home_genres")
        with c2:
            render_chart(plot_movies_by_year(filtered_df), key="home_years")
        render_insight_box(
            "Tür ve Üretim Özeti",
            f"İncelenen <b>{total_movies:,}</b> film içinde <b>{html.escape(top_genre_name)}</b> "
            f"(<b>{top_genre_count:,}</b> film) en çok üretilen türdür. Film üretimi özellikle 1995 sonrasında "
            "dijitalleşme ve uluslararası pazarların büyümesiyle ivme kazanmıştır.",
        )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            render_chart(plot_top_revenue_movies(filtered_df, top_n=10), key="home_rev")
        with c2:
            render_chart(plot_top_budget_movies(filtered_df, top_n=10), key="home_bud")
        if not known_rev_df.empty:
            best_row = known_rev_df.sort_values(by="revenue", ascending=False).iloc[0]
            render_insight_box(
                "Finansal Liderlik",
                f"Seçili kapsamda en yüksek gişe gelirine ulaşan yapım <b>{html.escape(str(best_row['title']))} "
                f"({int(best_row['release_year'])})</b> olup <b>{format_currency_short(float(best_row['revenue']))}</b> "
                f"gelir elde etmiştir (Bütçe: {format_currency_short(float(best_row['budget']))}).",
            )

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            render_chart(plot_top_popular_movies(filtered_df, top_n=10), key="home_pop")
        with c2:
            render_chart(plot_vote_average_distribution(filtered_df), key="home_vote")
        render_insight_box(
            "İzleyici Puan Dağılımı",
            f"Film puanları <b>{avg_vote:.2f}</b> ortalama ve <b>{median_vote:.2f}</b> medyan değer etrafında "
            "normal dağılıma yakın bir yapı sergilemektedir. Filmlerin büyük çoğunluğu 5.5 ile 7.2 puan bandında yoğunlaşır.",
        )

    with tab4:
        render_info_explainer(
            what_it_is="Yapım bütçesi ile gişe geliri arasındaki Pearson korelasyonunu (r) ve popülerlik-puan ilişkisini logaritmik ölçekte (log-scale) inceleyen saçılım analizidir.",
            what_it_affects="Doğrusal korelasyonun gücünü (r ≈ 0.71) ve sağa çarpık dağılımı göstererek K-Means ile Random Forest modellerinde neden np.log1p dönüşümü ve ölçeklendirme kullanıldığını belirler.",
        )
        c1, c2 = st.columns(2)
        with c1:
            render_chart(plot_budget_vs_revenue(filtered_df), key="home_b_vs_r")
        with c2:
            render_chart(plot_popularity_vs_vote_average(filtered_df), key="home_p_vs_v")

        both_valid = filtered_df[(filtered_df["budget"] > 0) & (filtered_df["revenue"] > 0)]
        if len(both_valid) > 1:
            profitable_pct = (both_valid["revenue"] > both_valid["budget"]).mean() * 100
            corr_br = float(both_valid["budget"].corr(both_valid["revenue"]))
            render_insight_box(
                "Korelasyon ve Başabaş Analizi",
                f"Bütçe ile gişe geliri arasında <b>r = {corr_br:.2f}</b> düzeyinde güçlü pozitif doğrusal ilişki vardır. "
                f"Finansal verisi bilinen <b>{len(both_valid):,}</b> filmin <b>%{profitable_pct:.1f}</b>'i "
                "kırmızı kesikli başabaş çizgisinin (Gelir > Bütçe) üzerinde kalarak yapım maliyetini aşmıştır.",
            )

    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            render_chart(plot_top_directors(filtered_df, top_n=12), key="home_dir")
        with c2:
            render_chart(plot_average_revenue_by_genre(filtered_df, top_n=15), key="home_g_rev")
        render_insight_box(
            "Adet vs. Kârlılık Farkı",
            "<b>Drama</b> ve <b>Comedy</b> sayıca en çok üretilen türler olmasına karşın, film başına ortalama gişe gelirinde "
            "<b>Animation</b>, <b>Adventure</b> ve <b>Fantasy</b> türleri yüksek prodüksiyon ölçekleriyle ilk sıralarda yer almaktadır.",
        )


# ==============================================================================
# SAYFA 2: VERİ SETİ İNCELEME VE KALİTE RAPORU
# ==============================================================================

def render_dataset_page(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    render_page_header(
        title="Veri Seti İnceleme, Ön İşleme ve Veri Gezgini",
        subtitle=(
            "tmdb_5000_movies.csv ve tmdb_5000_credits.csv tablolarının birleştirme kontrolü, "
            "eksik veri temizliği, türetilen yeni özellikler ve interaktif film arama tablosu."
        ),
        badge="VERİ MÜHENDİSLİĞİ & KALİTE RAPORU",
        icon_svg=DATASET_SVG,
    )

    merge_rep = report.get("merge_report", {})

    # 4 Üst Metrik Kartı
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "📄 Movies CSV Kayıt",
        f"{merge_rep.get('movies_row_count', 4803):,}",
        delta="20 ham sütun",
        delta_color="off",
    )
    m2.metric(
        "🎭 Credits CSV Kayıt",
        f"{merge_rep.get('credits_row_count', 4803):,}",
        delta="4 ham sütun (cast & crew)",
        delta_color="off",
    )
    m3.metric(
        "🔗 Eşleşen Ortak Film",
        f"{merge_rep.get('common_id_count', 4803):,}",
        delta="%100 Tam Eşleşme (id = movie_id)",
        delta_color="normal",
    )
    m4.metric(
        "🛡️ Birleştirme Veri Kaybı",
        f"{merge_rep.get('data_loss_count', 0)}",
        delta=f"Toplam {len(df.columns)} zenginleştirilmiş sütun",
        delta_color="off",
    )

    tab_explorer, tab_cleaning, tab_schema = st.tabs(
        [
            "🔍 Gelişmiş Veri Gezgini (cleaned_movies.csv)",
            "🧹 Veri Temizleme & Yeni Özellikler Raporu",
            "📐 Sütun Şeması & Doluluk Analizi",
        ]
    )

    with tab_explorer:
        with st.container(border=True):
            f1, f2, f3 = st.columns([1.6, 1.4, 1.2])
            with f1:
                search_term = st.text_input(
                    "🔍 Film Adı, Yönetmen veya Oyuncu Ara:",
                    value="",
                    placeholder="Örn: Christopher Nolan, Avatar, DiCaprio...",
                )
            with f2:
                all_genres = get_all_unique_genres(df)
                genre_filter = st.selectbox(
                    "🎭 Türe Göre Filtrele:",
                    options=["Tüm Türler"] + all_genres,
                    index=0,
                )
            with f3:
                sort_map = {
                    "Gişe Geliri (Yüksek -> Düşük)": ("revenue", False),
                    "Bütçe (Yüksek -> Düşük)": ("budget", False),
                    "Net Kâr (Yüksek -> Düşük)": ("profit", False),
                    "Ortalama Puan (Yüksek -> Düşük)": ("vote_average", False),
                    "Popülerlik (Yüksek -> Düşük)": ("popularity", False),
                    "Yayın Yılı (Yeni -> Eski)": ("release_year", False),
                }
                sort_choice = st.selectbox(
                    "🔃 Sıralama Ölçütü:",
                    options=list(sort_map.keys()),
                    index=0,
                )

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
            "vote_average",
            "vote_count",
            "popularity",
            "runtime",
        ]
        view_df = df[display_cols].copy()

        if search_term.strip():
            q = search_term.strip().lower()
            view_df = view_df[
                view_df["title"].astype(str).str.lower().str.contains(q, regex=False)
                | view_df["director"].astype(str).str.lower().str.contains(q, regex=False)
                | view_df["cast_str"].astype(str).str.lower().str.contains(q, regex=False)
            ]

        if genre_filter != "Tüm Türler":
            view_df = view_df[
                view_df["genres_str"].astype(str).str.contains(genre_filter, regex=False)
            ]

        sort_col, sort_asc = sort_map[sort_choice]
        view_df = view_df.sort_values(by=sort_col, ascending=sort_asc)

        st.caption(
            f"Filtreye uyan toplam **{len(view_df):,}** film bulundu (Tabloda ilk 500 kayıt listelenmektedir)."
        )

        st.dataframe(
            view_df.head(500),
            width="stretch",
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", format="%d", width="small"),
                "title": st.column_config.TextColumn("Film Adı", width="medium"),
                "release_year": st.column_config.NumberColumn("Yıl", format="%d", width="small"),
                "genres_str": st.column_config.TextColumn("Türler", width="medium"),
                "director": st.column_config.TextColumn("Yönetmen", width="medium"),
                "cast_str": st.column_config.TextColumn("Başrol Oyuncuları", width="medium"),
                "budget": st.column_config.NumberColumn("Bütçe ($)", format="$%d"),
                "revenue": st.column_config.NumberColumn("Gişe Geliri ($)", format="$%d"),
                "profit": st.column_config.NumberColumn("Net Kâr ($)", format="$%d"),
                "vote_average": st.column_config.ProgressColumn(
                    "Ortalama Puan",
                    min_value=0.0,
                    max_value=10.0,
                    format="⭐ %.1f",
                ),
                "vote_count": st.column_config.NumberColumn("Oy Sayısı", format="%d"),
                "popularity": st.column_config.NumberColumn("Popülerlik", format="%.1f"),
                "runtime": st.column_config.NumberColumn("Süre (dk)", format="%.0f dk"),
            },
        )

    with tab_cleaning:
        render_insight_box(
            "Birleştirme Doğrulaması",
            f"<code>pd.merge(movies, credits, left_on='id', right_on='movie_id', how='inner')</code> işlemiyle "
            f"<b>{merge_rep.get('merged_row_count', 4803):,}</b> kayıt sıfır veri kaybıyla birleştirilmiş, "
            "JSON sütunları ayrıştırılmış ve mükerrer sütunlar temizlenmiştir.",
        )
        c1, c2 = st.columns([1.25, 1.0])
        with c1:
            render_section_header(
                title="🛠️ Eksik Veri İmputasyonu ve Sıfır ($0) Değer Yönetimi",
                what_it_is="Ham verideki eksik (NaN) değerlerin medyan ile doldurulması (imputation) ve $0 olarak girilmiş finansal kayıtların ayrıştırılması sürecidir.",
                what_it_affects="Aykırı değerlere duyarsız olan medyan atamasıyla kayıt kaybını önler; $0 bütçe/gelir kayıtlarını filtreleyerek K-Means ve Random Forest modellerinde sıfıra bölme hatasını ve yanlılığı (bias) engeller.",
                small=True,
            )
            cleaning_summary = pd.DataFrame(
                [
                    {
                        "Kontrol Edilen Alan": "overview (Film Özeti)",
                        "Tespit Edilen Durum": f"{report['overview_null_filled']} eksik (NaN) kayıt",
                        "Uygulanan Çözüm": "Boş metin ('') ile dolduruldu",
                    },
                    {
                        "Kontrol Edilen Alan": "runtime (Film Süresi)",
                        "Tespit Edilen Durum": f"{report['runtime_null_filled']} eksik (NaN) kayıt",
                        "Uygulanan Çözüm": f"Medyan süre ({report['runtime_median_used']:.0f} dk) atandı",
                    },
                    {
                        "Kontrol Edilen Alan": "release_date (Yayın Tarihi)",
                        "Tespit Edilen Durum": f"{report['release_date_null_count']} eksik kayıt",
                        "Uygulanan Çözüm": f"Medyan yıl ({report['release_year_median_used']}) ile tamamlandı",
                    },
                    {
                        "Kontrol Edilen Alan": "budget (Bütçe = $0)",
                        "Tespit Edilen Durum": f"{report['zero_budget_count']:,} bilinmeyen, {report['known_budget_count']:,} bilinen",
                        "Uygulanan Çözüm": "Sıfır bütçelerde profit_ratio NaN bırakıldı (sıfıra bölme önlendi)",
                    },
                    {
                        "Kontrol Edilen Alan": "revenue (Gelir = $0)",
                        "Tespit Edilen Durum": f"{report['zero_revenue_count']:,} bilinmeyen, {report['known_revenue_count']:,} bilinen",
                        "Uygulanan Çözüm": "Sınıflandırma modelinde yalnızca geliri bilinenler kullanıldı",
                    },
                    {
                        "Kontrol Edilen Alan": "genres & keywords (JSON)",
                        "Tespit Edilen Durum": f"{report['empty_genres_count']} boş tür, {report['empty_keywords_count']} boş anahtar kelime",
                        "Uygulanan Çözüm": "JSON güvenli ayrıştırıldı; genre_list ve keyword_list üretildi",
                    },
                    {
                        "Kontrol Edilen Alan": "cast & crew (JSON)",
                        "Tespit Edilen Durum": f"{report['empty_cast_count']} boş kadro, {report['empty_director_count']} yönetmensiz",
                        "Uygulanan Çözüm": "İlk 5 oyuncu (cast_list) ve yönetmen (director) çıkarıldı",
                    },
                ]
            )
            st.dataframe(cleaning_summary, width="stretch", hide_index=True)

        with c2:
            render_section_header(
                title="🧬 Özellik Mühendisliği (Feature Engineering) ve Log1p Dönüşümü",
                what_it_is="Ham JSON metinlerinden yapısal listelerin çıkarılması ve sağa çarpık (right-skewed) finansal değişkenlere np.log1p(x) = ln(1+x) dönüşümü uygulanmasıdır.",
                what_it_affects="Aşırı yüksek bütçe/gelir uç değerlerinin Öklid uzaklığını (K-Means) domine etmesini önler; Apriori ve TF-IDF modelleri için temiz girdi vektörleri üretir.",
                small=True,
            )
            new_cols_df = pd.DataFrame(
                [
                    {"Yeni Sütun": "release_year", "Açıklama": "Yayın tarihinden çıkarılan tam yıl bilgisi"},
                    {"Yeni Sütun": "profit", "Açıklama": "Net gişe kârı (revenue - budget)"},
                    {"Yeni Sütun": "profit_ratio", "Açıklama": "Yatırım getirisi oranı (revenue / budget, bütçe > 0 için)"},
                    {"Yeni Sütun": "genre_list / genres_str", "Açıklama": "Ayrıştırılmış film türleri listesi ve metni"},
                    {"Yeni Sütun": "keyword_list / keywords_str", "Açıklama": "Ayrıştırılmış anahtar kelimeler listesi"},
                    {"Yeni Sütun": "cast_list / cast_str", "Açıklama": "Filmin ilk 5 başrol oyuncusu"},
                    {"Yeni Sütun": "director", "Açıklama": "Film ekibinden (crew) çekilen yönetmen adı"},
                    {"Yeni Sütun": "*_log (4 sütun)", "Açıklama": "Çarpık finansal değişkenler için np.log1p dönüşümü"},
                ]
            )
            st.dataframe(new_cols_df, width="stretch", hide_index=True)

    with tab_schema:
        col_info_rows = []
        initial_nulls = report.get("initial_nulls", {})
        total_len = max(len(df), 1)
        for col in df.columns:
            filled_cnt = int(df[col].notnull().sum())
            col_info_rows.append(
                {
                    "Sütun Adı": col,
                    "Veri Tipi": str(df[col].dtype),
                    "Dolu Kayıt": filled_cnt,
                    "Doluluk Oranı (%)": round(filled_cnt / total_len * 100, 1),
                    "Temizleme Öncesi Eksik": int(initial_nulls.get(col, 0)),
                    "Temizleme Sonrası Eksik": int(df[col].isnull().sum()),
                }
            )
        st.dataframe(
            pd.DataFrame(col_info_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Dolu Kayıt": st.column_config.NumberColumn("Dolu Kayıt", format="%d"),
                "Doluluk Oranı (%)": st.column_config.ProgressColumn(
                    "Doluluk Oranı",
                    min_value=0.0,
                    max_value=100.0,
                    format="%%%.1f",
                ),
                "Temizleme Öncesi Eksik": st.column_config.NumberColumn("Önceki Eksik", format="%d"),
                "Temizleme Sonrası Eksik": st.column_config.NumberColumn("Sonraki Eksik", format="%d"),
            },
        )


# ==============================================================================
# SAYFA 3: APRIORI BİRLİKTELİK ANALİZİ
# ==============================================================================

def render_apriori_page(df: pd.DataFrame) -> None:
    render_page_header(
        title="Apriori Birliktelik Kuralı Madenciliği",
        subtitle=(
            "Her filmi bir işlem (transaction) kabul ederek birlikte en sık kullanılan film türlerini ve "
            "tematik anahtar kelimeleri Support, Confidence ve Lift metrikleriyle keşfeder."
        ),
        badge="ALGORİTMA 1 • BİRLİKTELİK ANALİZİ",
        icon_svg=APRIORI_SVG,
    )

    # Metrik Rehberi
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        with st.container(border=True):
            st.markdown("**📌 Support (Destek Oranı)**")
            st.caption("İlgili tür veya anahtar kelime grubunun tüm filmler içinde birlikte görülme yüzdesidir: $P(X \\cap Y)$.")
    with g_col2:
        with st.container(border=True):
            st.markdown("**🎯 Confidence (Güven Olasılığı)**")
            st.caption("Öncül tür (Eğer X) bir filmde varsa, ardıl türün (İse Y) aynı filmde bulunma koşullu olasılığıdır: $P(Y|X)$.")
    with g_col3:
        with st.container(border=True):
            st.markdown("**🚀 Lift (Kaldıraç Katsayısı)**")
            st.caption("İki öğenin bağımsız/tesadüfi yan yana gelmesine kıyasla kaç kat daha güçlü birliktelik gösterdiğidir: $P(Y|X) / P(Y)$.")

    # Teknik Açıklama 1: Apriori Hiperparametreleri ve Budama (Pruning)
    render_section_header(
        title="⚙️ Apriori Algoritma Hiperparametreleri ve Eşik Budaması (Pruning)",
        what_it_is="TransactionEncoder ile One-Hot ikili matrise çevrilen film sepetlerinde sık öğe kümelerini (frequent itemsets) ve kuralları süzen eşik değerleridir.",
        what_it_affects="Apriori'nin anti-monotonluk ilkesine göre Minimum Support düşürüldükçe aday öğe uzayı genişler ve nadir kurallar yakalanır; Confidence ve Lift artırıldıkça zayıf/tesadüfi kurallar elenir.",
    )

    with st.container(border=True):
        p1, p2, p3, p4 = st.columns([1.2, 1.2, 1.2, 1.0])
        min_support = p1.slider(
            "Minimum Support (Destek)",
            min_value=0.005,
            max_value=0.20,
            value=0.02,
            step=0.005,
            help="Bir öğe kümesinin tüm filmler içinde görülme oranı eşiği (varsayılan: 0.02 = %2).",
        )
        min_confidence = p2.slider(
            "Minimum Confidence (Güven)",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help="Öncül görüldüğünde ardılın görülme olasılığı eşiği (varsayılan: 0.30 = %30).",
        )
        min_lift = p3.slider(
            "Minimum Lift Eşiği",
            min_value=1.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Lift > 1.0 pozitif birlikteliği ifade eder (varsayılan: 1.0).",
        )
        with p4:
            st.write("")
            include_keywords = st.checkbox(
                "Anahtar Kelimeleri Dahil Et",
                value=True,
                help="İşaretlenirse film türlerinin yanı sıra sık görülen anahtar kelimeler de analize katılır.",
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
    rules_df = res["rules"]
    best_rule = rules_df.iloc[0]

    # 4 KPI Kartı
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "🎬 İşlenen Film (Transaction)",
        f"{tx_info['valid_transactions']:,}",
        delta=f"{tx_info.get('total_movies', len(df)):,} toplam kayıttan",
        delta_color="off",
    )
    k2.metric(
        "📦 Sık Öğe Kümesi",
        f"{len(res['frequent_itemsets']):,}",
        delta=f"Min. Support >= %{min_support * 100:.1f}",
        delta_color="off",
    )
    k3.metric(
        "🔗 Üretilen Güçlü Kural",
        f"{len(rules_df):,}",
        delta=f"Min. Güven >= %{min_confidence * 100:.0f}",
        delta_color="off",
    )
    k4.metric(
        "🏆 En Yüksek Lift Değeri",
        f"{best_rule['lift']:.2f}x",
        delta=f"Güven: %{best_rule['confidence'] * 100:.1f}",
        delta_color="normal",
    )

    # Otomatik En Güçlü Kural Yorumlayıcısı
    render_insight_box(
        "En Güçlü Birliktelik Kuralı Yorumu",
        f"Bir filmde <b>[{html.escape(str(best_rule['antecedents_str']))}]</b> yer aldığında, "
        f"<b>%{best_rule['confidence'] * 100:.1f}</b> olasılıkla <b>[{html.escape(str(best_rule['consequents_str']))}]</b> "
        f"de yer almaktadır (Support: <b>%{best_rule['support'] * 100:.2f}</b>). "
        f"Bu iki grubun aynı filmde buluşma sıklığı, rastlantısal beklentinin <b>{best_rule['lift']:.2f} katıdır</b>.",
    )

    tab_rules, tab_freq, tab_kw = st.tabs(
        [
            "🔗 En Güçlü Birliktelik Kuralları & Harita",
            "📊 Sık Görülen Türler ve Kombinasyonlar",
            "🏷️ Anahtar Kelime - Tür İlişkileri",
        ]
    )

    with tab_rules:
        render_info_explainer(
            what_it_is="Üretilen kuralların 3 boyutlu metrik uzaydaki (X: Support, Y: Confidence, Renk/Boyut: Lift) dağılım haritası ve en yüksek Lift sıralamasıdır.",
            what_it_affects="Yüksek Support'lu genel kurallar (Drama/Comedy) ile düşük Support ama çok yüksek Lift ve Confidence üreten spesifik alt-tür kurallarını (Animation <-> Family) ayrıştırmanızı sağlar.",
        )
        v1, v2 = st.columns(2)
        with v1:
            render_chart(plot_top_rules_bar(res["top_10_rules"]), key="ap_top_bar")
        with v2:
            render_chart(plot_apriori_rules_scatter(rules_df), key="ap_scatter")

        st.markdown("##### 📋 Birliktelik Kuralları Tablosu")
        rules_display = rules_df[
            ["rule_str", "antecedents_str", "consequents_str", "support", "confidence", "lift"]
        ].copy()
        rules_display["support_pct"] = (rules_display["support"] * 100).round(2)
        rules_display["confidence_pct"] = (rules_display["confidence"] * 100).round(1)
        rules_display = rules_display[
            ["rule_str", "antecedents_str", "consequents_str", "support_pct", "confidence_pct", "lift"]
        ]

        st.dataframe(
            rules_display,
            width="stretch",
            hide_index=True,
            column_config={
                "rule_str": st.column_config.TextColumn("Birliktelik Kuralı (Eğer -> İse)", width="large"),
                "antecedents_str": st.column_config.TextColumn("Öncül (Eğer)", width="medium"),
                "consequents_str": st.column_config.TextColumn("Ardıl (İse)", width="medium"),
                "support_pct": st.column_config.NumberColumn("Support (%)", format="%%%.2f"),
                "confidence_pct": st.column_config.ProgressColumn(
                    "Confidence - Güven (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%%%.1f",
                ),
                "lift": st.column_config.NumberColumn("Lift Katsayısı", format="%.2fx"),
            },
        )

    with tab_freq:
        g1, g2 = st.columns(2)
        with g1:
            render_chart(
                plot_apriori_frequent_items(
                    res["single_items"], "En Sık Görülen Tekil Film Türleri / Öğeler", top_n=12
                ),
                key="ap_single",
            )
        with g2:
            render_chart(
                plot_apriori_frequent_items(
                    res["combo_items"], "En Sık Görülen Çoklu Tür Kombinasyonları", top_n=12
                ),
                key="ap_combo",
            )

    with tab_kw:
        if include_keywords and not res["genre_keyword_rules"].empty:
            st.caption(
                "Filmlerin olay örgüsü anahtar kelimeleri (`keywords`) ile film türleri (`genres`) arasındaki en güçlü kurallar:"
            )
            kw_df = res["genre_keyword_rules"][
                ["rule_str", "antecedents_str", "consequents_str", "support", "confidence", "lift"]
            ].copy()
            kw_df["support_pct"] = (kw_df["support"] * 100).round(2)
            kw_df["confidence_pct"] = (kw_df["confidence"] * 100).round(1)
            st.dataframe(
                kw_df[["rule_str", "antecedents_str", "consequents_str", "support_pct", "confidence_pct", "lift"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "rule_str": st.column_config.TextColumn("Anahtar Kelime - Tür Kuralı", width="large"),
                    "antecedents_str": st.column_config.TextColumn("Öncül", width="medium"),
                    "consequents_str": st.column_config.TextColumn("Ardıl", width="medium"),
                    "support_pct": st.column_config.NumberColumn("Support (%)", format="%%%.2f"),
                    "confidence_pct": st.column_config.ProgressColumn(
                        "Confidence (%)",
                        min_value=0.0,
                        max_value=100.0,
                        format="%%%.1f",
                    ),
                    "lift": st.column_config.NumberColumn("Lift", format="%.2fx"),
                },
            )
        else:
            st.info(
                "Anahtar kelime kurallarını görmek için üstteki parametrelerden 'Anahtar Kelimeleri Dahil Et' "
                "seçeneğini işaretleyin veya Support eşiğini düşürün."
            )


# ==============================================================================
# SAYFA 4: K-MEANS KÜMELEME
# ==============================================================================

def render_kmeans_page(df: pd.DataFrame) -> None:
    render_page_header(
        title="K-Means Kümeleme ve Film Arketip Analizi",
        subtitle=(
            "Filmleri bütçe, gişe geliri, popülerlik, süre, ortalama puan, oy sayısı ve türlerine göre "
            "homojen segmentlere ayırır ve her kümenin finansal/izleyici profilini çıkarır."
        ),
        badge="ALGORİTMA 2 • GÖZETİMSİZ ÖĞRENME (CLUSTERING)",
        icon_svg=KMEANS_SVG,
    )

    # Teknik Açıklama 1: K-Means Hiperparametreleri ve Özellik Ölçeklendirme
    render_section_header(
        title="⚙️ K-Means Hiperparametreleri, StandardScaler ve MultiLabelBinarizer",
        what_it_is="Sayısal özelliklerin log1p + StandardScaler (ortalama=0, std=1) ile normalize edildiği ve türlerin MultiLabelBinarizer ile ikili vektöre çevrildiği kümeleme yapılandırmasıdır.",
        what_it_affects="Küme sayısı (k) segment granülaritesini değiştirir; sıfır bütçe/gelir filtresi merkezlerin (centroid) gerçek finansal verilerle hesaplanmasını sağlar; tür ağırlığı ise tematik ayrışmayı kontrol eder.",
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
        n_clusters = c1.slider(
            "Küme Sayısı (k)",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
            help="Filmlerin kaç segmente ayrılacağını belirler (2-8 aralığı).",
        )
        with c2:
            st.write("")
            filter_zero_fin = st.checkbox(
                "Sadece Bütçe ve Geliri Bilinen Filmleri Kümele",
                value=True,
                help="İşaretlenirse bütçe ve geliri > $0 olan 3.229 film kümelenir. Kaldırılırsa medyan atamasıyla tüm 4.803 film kümelenir.",
            )
        with c3:
            st.write("")
            include_genres = st.checkbox(
                "Film Türlerini (MultiLabelBinarizer) Dahil Et",
                value=True,
                help="Sayısal özelliklerin yanı sıra film türlerini çoklu ikili vektör olarak kümelemeye katar.",
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

    # 4 Üst Metrik Kartı
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "🎞️ Kümelenen Film Sayısı",
        f"{res['prep_info']['clustered_movie_count']:,}",
        delta="Log1p + StandardScaler uygulandı",
        delta_color="off",
    )
    m2.metric(
        "🧩 Seçilen Küme Sayısı",
        f"k = {res['selected_k']}",
        delta=f"Inertia: {res['selected_inertia']:,.0f}",
        delta_color="off",
    )
    m3.metric(
        "📏 Silhouette Skoru",
        f"{res['selected_silhouette']:.4f}",
        delta=f"En yüksek skor: k = {res['optimal_k_by_silhouette']}",
        delta_color="off",
    )
    m4.metric(
        "🧭 2B PCA Varyans Oranı",
        f"%{res['pca_explained_variance'] * 100:.1f}",
        delta="İki temel bileşen açıklaması",
        delta_color="off",
    )

    # Küme Profil Kartları
    st.markdown("#### 🏷️ Oluşturulan Küme Profilleri (Segment Özeti)")
    summary_df = res["summary_df"]
    cols_per_row = min(len(summary_df), 4)
    card_cols = st.columns(cols_per_row)
    for idx, (_, row) in enumerate(summary_df.iterrows()):
        col = card_cols[idx % cols_per_row]
        with col:
            with st.container(border=True):
                st.markdown(f"**{row['cluster_name']}**")
                st.caption(
                    f"🎞️ **{int(row['movie_count']):,} film** (Verinin **%{row['percentage']:.1f}**'i)"
                )
                st.markdown(
                    f"- **Ort. Bütçe:** `{format_currency_short(float(row['avg_budget']))}`\n"
                    f"- **Ort. Gelir:** `{format_currency_short(float(row['avg_revenue']))}`\n"
                    f"- **Ort. Puan:** `⭐ {row['avg_vote_average']:.2f}` *(Pop: {row['avg_popularity']:.1f})*\n"
                    f"- **Baskın Türler:** {row['top_genres']}"
                )

    tab_comp, tab_pca, tab_eval, tab_movies = st.tabs(
        [
            "📊 Küme Karşılaştırma Grafikleri & Tablosu",
            "🧭 2B PCA Küme Dağılım Haritası",
            "📐 Optimal Küme Sayısı (Elbow & Silhouette)",
            "🎬 Kümedeki Filmleri Keşfet",
        ]
    )

    with tab_comp:
        render_chart(plot_kmeans_cluster_comparison(summary_df), key="km_comp")
        st.markdown("##### 📋 Küme Ortalamaları Detay Tablosu")
        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True,
            column_config={
                "cluster_id": st.column_config.NumberColumn("Küme No", format="%d", width="small"),
                "cluster_name": st.column_config.TextColumn("Açıklayıcı Küme Adı", width="large"),
                "movie_count": st.column_config.NumberColumn("Film Sayısı", format="%d"),
                "percentage": st.column_config.ProgressColumn(
                    "Kapsam Oranı (%)", min_value=0.0, max_value=100.0, format="%%%.1f"
                ),
                "avg_budget": st.column_config.NumberColumn("Ort. Bütçe ($)", format="$%d"),
                "avg_revenue": st.column_config.NumberColumn("Ort. Gelir ($)", format="$%d"),
                "avg_profit": st.column_config.NumberColumn("Ort. Net Kâr ($)", format="$%d"),
                "avg_popularity": st.column_config.NumberColumn("Ort. Popülerlik", format="%.1f"),
                "avg_vote_average": st.column_config.NumberColumn("Ort. Puan", format="⭐ %.2f"),
                "avg_vote_count": st.column_config.NumberColumn("Ort. Oy Sayısı", format="%d"),
                "avg_runtime": st.column_config.NumberColumn("Ort. Süre", format="%.0f dk"),
                "top_genres": st.column_config.TextColumn("Baskın 3 Tür", width="medium"),
            },
        )

    with tab_pca:
        render_info_explainer(
            what_it_is="Yüksek boyutlu özellik matrisinin (bütçe, gelir, popülerlik, puan, süre ve 20 tür) Temel Bileşenler Analizi (PCA) ile en yüksek varyanslı 2 dik bileşene (PCA 1 ve PCA 2) izdüşümüdür.",
            what_it_affects="Çok boyutlu küme merkezlerinin 2B düzlemde ne kadar ayrıştığını görselleştirir; PCA 1 ekseni finansal ölçek ve popülerliği, PCA 2 ekseni ise izleyici puanı ve tür ayrımını temsil eder.",
        )
        render_chart(plot_kmeans_clusters_pca(res["clustered_df"]), key="km_pca")

    with tab_eval:
        render_info_explainer(
            what_it_is="k = 2..8 aralığında hesaplanan Küme İçi Hata Kareler Toplamı (WCSS / Inertia - Dirsek Yöntemi) ve küme ayrışma katsayısı olan Silhouette Skoru (-1 ile +1 arası) analizidir.",
            what_it_affects="Inertia'nın düşüş hızının yavaşladığı kırılma noktasını (dirsek) ve kümeler arası ayrışmanın küme içi yoğunluğa oranını ölçerek optimal küme sayısı (k) seçimini doğrular.",
        )
        render_chart(
            plot_elbow_and_silhouette(res["evaluation_df"], res["selected_k"]),
            key="km_elbow",
        )
        render_insight_box(
            "Küme Sayısı Seçimi",
            f"Matematiksel olarak en yüksek ayrışma skoru <b>k = {res['optimal_k_by_silhouette']}</b> kümede elde edilse de "
            "(Yüksek Bütçeli vs Düşük Bütçeli ikili ayrımı), iş analitiği açısından filmleri <b>k = 4</b> segmente ayırmak "
            "<i>Dev Gişe Yapımları</i>, <i>Beğenilen Bağımsız Filmler</i>, <i>Orta Ölçekli Ticari Filmler</i> ve "
            "<i>Düşük Bütçeli Niş Yapımlar</i> profillerini çok daha net ortaya koymaktadır.",
        )

    with tab_movies:
        cluster_options = summary_df["cluster_name"].tolist()
        f_c1, f_c2 = st.columns([1.8, 1.2])
        with f_c1:
            selected_cluster_label = st.selectbox(
                "📂 İncelemek istediğiniz kümeyi seçin:", cluster_options
            )
        with f_c2:
            cluster_search = st.text_input(
                "🔍 Küme İçinde Film Ara:", value="", placeholder="Film adı veya yönetmen..."
            )

        clustered_df = res["clustered_df"]
        cluster_movies = (
            clustered_df[clustered_df["cluster_name"] == selected_cluster_label]
            .sort_values(by="revenue", ascending=False)
            .copy()
        )
        if cluster_search.strip():
            q = cluster_search.strip().lower()
            cluster_movies = cluster_movies[
                cluster_movies["title"].astype(str).str.lower().str.contains(q, regex=False)
                | cluster_movies["director"].astype(str).str.lower().str.contains(q, regex=False)
            ]

        st.caption(
            f"**{selected_cluster_label}** içerisinde gösterilen film sayısı: **{len(cluster_movies):,}**"
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
                    "profit",
                    "vote_average",
                    "vote_count",
                    "popularity",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "title": st.column_config.TextColumn("Film Adı", width="medium"),
                "release_year": st.column_config.NumberColumn("Yıl", format="%d", width="small"),
                "genres_str": st.column_config.TextColumn("Türler", width="medium"),
                "director": st.column_config.TextColumn("Yönetmen", width="medium"),
                "budget": st.column_config.NumberColumn("Bütçe ($)", format="$%d"),
                "revenue": st.column_config.NumberColumn("Gişe Geliri ($)", format="$%d"),
                "profit": st.column_config.NumberColumn("Net Kâr ($)", format="$%d"),
                "vote_average": st.column_config.ProgressColumn(
                    "Puan", min_value=0.0, max_value=10.0, format="⭐ %.1f"
                ),
                "vote_count": st.column_config.NumberColumn("Oy Sayısı", format="%d"),
                "popularity": st.column_config.NumberColumn("Popülerlik", format="%.1f"),
            },
        )


# ==============================================================================
# SAYFA 5: RANDOM FOREST TAHMİN
# ==============================================================================

def render_random_forest_page(df: pd.DataFrame) -> None:
    render_page_header(
        title="Random Forest ile Gişe Gelir Sınıfı Tahmini",
        subtitle=(
            "Bir filmin bütçe, popülerlik, süre, puan, oy sayısı, yayın yılı ve türlerine dayanarak "
            "Medyan Gişe Geliri ($51.75M) eşiğinin üzerine çıkıp çıkamayacağını tahmin eder."
        ),
        badge="ALGORİTMA 3 • GÖZETİMLİ SINIFLANDIRMA (CLASSIFICATION)",
        icon_svg=RF_SVG,
    )

    # Teknik Açıklama 1: Random Forest Hiperparametreleri ve Veri Sızıntısı Önleme
    render_section_header(
        title="⚙️ Random Forest Hiperparametreleri (Bagging, Max Depth ve Stratified Split)",
        what_it_is="Çoklu karar ağaçlarının Bootstrap örneklemesiyle eğitildiği (Bagging) ve hedef sınıf oranlarının korunarak (stratify=y) eğitim/test ayrımının yapıldığı model yapılandırmasıdır.",
        what_it_affects="Ağaç sayısı (n_estimators) tahmin varyansını düşürür; maksimum derinlik (max_depth) aşırı öğrenmeyi (overfitting) sınırlar; StandardScaler yalnızca eğitim verisinde fit edilerek veri sızıntısını (data leakage) önler.",
    )

    with st.container(border=True):
        p1, p2, p3 = st.columns(3)
        n_estimators = p1.slider(
            "Karar Ağacı Sayısı (n_estimators)",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Random Forest topluluğundaki ağaç sayısı (varsayılan: 200).",
        )
        max_depth_opt = p2.selectbox(
            "Maksimum Ağaç Derinliği (max_depth)",
            options=["Sınırsız (None)", 8, 12, 16, 20],
            index=0,
            help="Ağaçların aşırı öğrenmesini (overfitting) kontrol etmek için derinlik sınırı.",
        )
        max_depth = None if max_depth_opt == "Sınırsız (None)" else int(max_depth_opt)
        test_size = p3.slider(
            "Test Verisi Oranı",
            min_value=0.15,
            max_value=0.30,
            value=0.20,
            step=0.05,
            help="Modelin hiç görmediği doğrulama verisi oranı (varsayılan: %20 Test, %80 Eğitim).",
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

    # 5 Performans KPI Kartı
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "🎯 Accuracy (Doğruluk)",
        f"%{metrics['accuracy'] * 100:.2f}",
        delta=f"{res['test_size']} test filmi üzerinde",
        delta_color="off",
        help="Tüm tahminler içinde doğru sınıflandırılan filmlerin oranı.",
    )
    k2.metric(
        "✔️ Precision (Kesinlik)",
        f"%{metrics['precision'] * 100:.2f}",
        delta="Yüksek Gelir kesinliği",
        delta_color="off",
        help="Modelin 'Yüksek Gelir' dediği filmlerin gerçekten Yüksek Gelir olma oranı.",
    )
    k3.metric(
        "📡 Recall (Duyarlılık)",
        f"%{metrics['recall'] * 100:.2f}",
        delta="Yüksek Gelir yakalama",
        delta_color="off",
        help="Gerçekte Yüksek Gelir olan filmlerin ne kadarının doğru yakalandığı.",
    )
    k4.metric(
        "⚖️ F1-Score",
        f"%{metrics['f1_score'] * 100:.2f}",
        delta="Kesinlik & Duyarlılık dengesi",
        delta_color="off",
        help="Precision ve Recall değerlerinin harmonik ortalaması.",
    )
    k5.metric(
        "📈 ROC-AUC Skoru",
        f"%{metrics['roc_auc'] * 100:.2f}",
        delta=f"Eşik: {format_currency_short(res['median_revenue'])}",
        delta_color="normal",
        help="Modelin iki sınıfı birbirinden ayırma gücü (100'e yakın olması mükemmel ayrımı gösterir).",
    )

    tab_eval, tab_sim, tab_preds = st.tabs(
        [
            "📊 Özellik Önem Düzeyleri & Confusion Matrix",
            "🧪 Canlı Gişe Tahmin Simülatörü",
            "📋 Test Verisi Tahmin Sonuçları",
        ]
    )

    with tab_eval:
        render_info_explainer(
            what_it_is="Karar ağacı düğümlerinde Gini safsızlığını (impurity) en çok düşüren özelliklerin ağırlık sıralaması (Feature Importance) ve test seti Karmaşıklık Matrisidir (TN, FP, FN, TP).",
            what_it_affects="Gişe gelir sınıfını tahmin ederken hangi değişkenin (Oy Sayısı, Popülerlik, Bütçe vb.) karar mekanizmasını ne oranda domine ettiğini ve hata türlerini (Yanlış Pozitif / Yanlış Negatif) gösterir.",
        )
        c1, c2 = st.columns([1.3, 1.0])
        with c1:
            render_chart(
                plot_feature_importance(res["feature_importance_df"], top_n=12),
                key="rf_fi",
            )
        with c2:
            render_chart(
                plot_confusion_matrix_heatmap(res["confusion_matrix_df"]),
                key="rf_cm",
            )

        top_feat = res["feature_importance_df"].iloc[0]
        render_insight_box(
            "Model Metodolojisi ve En Belirleyici Faktörler",
            f"Veri sızıntısını (data leakage) önlemek için <code>revenue</code> ve <code>profit</code> değişkenleri modele "
            f"kesinlikle verilmemiştir. Geliri bilinen <b>{res['total_known_movies']:,}</b> film, medyan gelir olan "
            f"<b>${res['median_revenue']:,.0f}</b> sınırına göre iki eşit sınıfa ayrılmıştır. "
            f"Model kararlarında en etkili değişken <b>%{top_feat['importance_pct']:.1f}</b> ağırlıkla "
            f"<b>{html.escape(str(top_feat['feature_name']))}</b> olmuştur.",
        )

    with tab_sim:
        st.markdown("##### 🧪 Yeni Bir Film Senaryosu İçin Gişe Sınıfı Tahmini")
        st.caption(
            "Hazır film profillerinden birini seçebilir veya bütçe, süre, puan ve tür değerlerini değiştirerek modelin canlı kararını inceleyebilirsiniz."
        )

        presets = {
            "Özel Senaryo (Serbest Giriş)": {
                "budget": 45_000_000,
                "pop": 35.0,
                "runtime": 115,
                "vote_avg": 6.8,
                "vote_cnt": 1200,
                "year": 2015,
                "genres": ["Action", "Adventure", "Science Fiction"],
            },
            "Senaryo 1: Büyük Bütçeli Gişe Adayı (Blockbuster)": {
                "budget": 165_000_000,
                "pop": 95.0,
                "runtime": 142,
                "vote_avg": 7.6,
                "vote_cnt": 5500,
                "year": 2016,
                "genres": ["Action", "Adventure", "Science Fiction"],
            },
            "Senaryo 2: Düşük Bütçeli Bağımsız Festival Filmi": {
                "budget": 4_500_000,
                "pop": 9.5,
                "runtime": 98,
                "vote_avg": 7.4,
                "vote_cnt": 280,
                "year": 2014,
                "genres": ["Drama", "Romance"],
            },
            "Senaryo 3: Orta Ölçekli Aile & Animasyon Yapımı": {
                "budget": 60_000_000,
                "pop": 48.0,
                "runtime": 95,
                "vote_avg": 6.9,
                "vote_cnt": 1800,
                "year": 2013,
                "genres": ["Animation", "Family", "Comedy"],
            },
        }

        all_genres = sorted(list(res["mlb"].classes_))
        col_form, col_result = st.columns([1.35, 1.0])

        with col_form:
            with st.container(border=True):
                selected_preset_name = st.selectbox(
                    "🎬 Hızlı Senaryo Şablonu:",
                    options=list(presets.keys()),
                    index=0,
                )
                preset_vals = presets[selected_preset_name]

                s1, s2, s3 = st.columns(3)
                sim_budget = s1.number_input(
                    "Yapım Bütçesi ($)",
                    min_value=0,
                    max_value=400_000_000,
                    value=int(preset_vals["budget"]),
                    step=5_000_000,
                )
                sim_pop = s2.number_input(
                    "Popülerlik Skoru",
                    min_value=0.1,
                    max_value=500.0,
                    value=float(preset_vals["pop"]),
                    step=5.0,
                )
                sim_runtime = s3.number_input(
                    "Film Süresi (dk)",
                    min_value=40,
                    max_value=250,
                    value=int(preset_vals["runtime"]),
                    step=5,
                )

                s4, s5, s6 = st.columns(3)
                sim_vote_avg = s4.slider(
                    "Ortalama Puan", 1.0, 10.0, float(preset_vals["vote_avg"]), 0.1
                )
                sim_vote_cnt = s5.number_input(
                    "Oy Sayısı",
                    min_value=1,
                    max_value=25000,
                    value=int(preset_vals["vote_cnt"]),
                    step=100,
                )
                sim_year = s6.number_input(
                    "Yayın Yılı",
                    min_value=1930,
                    max_value=2026,
                    value=int(preset_vals["year"]),
                    step=1,
                )

                sim_genres = st.multiselect(
                    "Film Türleri",
                    options=all_genres,
                    default=[g for g in preset_vals["genres"] if g in all_genres],
                )

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

        with col_result:
            with st.container(border=True):
                st.markdown("##### 📌 Modelin Canlı Tahmin Kararı")
                high_p = pred_out["high_revenue_prob"] * 100
                low_p = pred_out["low_revenue_prob"] * 100
                med_thresh = pred_out["median_threshold"]

                if pred_out["predicted_label"] == "Yüksek Gelir":
                    st.success(
                        f"🏆 **TAHMİN: YÜKSEK GELİR SINIFI**\n\n"
                        f"Bu film profilinin medyan gişe eşiği olan **${med_thresh:,.0f}** değerini "
                        f"aşma olasılığı **%{high_p:.1f}** olarak hesaplanmıştır."
                    )
                else:
                    st.warning(
                        f"📉 **TAHMİN: DÜŞÜK / ORTA GELİR SINIFI**\n\n"
                        f"Bu film profilinin medyan gişe eşiği olan **${med_thresh:,.0f}** altında "
                        f"kalma olasılığı **%{low_p:.1f}** olarak hesaplanmıştır."
                    )

                st.progress(
                    min(max(pred_out["high_revenue_prob"], 0.0), 1.0),
                    text=f"Yüksek Gelir Olasılığı: %{high_p:.1f}  |  Düşük Gelir Olasılığı: %{low_p:.1f}",
                )
                st.markdown(
                    f"- **Girilen Bütçe:** `{format_currency_short(float(sim_budget))}`\n"
                    f"- **Karşılaştırılan Medyan Gişe Eşiği:** `${med_thresh:,.0f}`\n"
                    f"- **Seçilen Türler:** `{', '.join(sim_genres) if sim_genres else 'Tür seçilmedi'}`"
                )

    with tab_preds:
        preds_df = res["test_predictions_df"].copy()
        filter_col1, filter_col2 = st.columns([1.2, 2.0])
        with filter_col1:
            pred_filter = st.radio(
                "Tahmin Durumu Filtresi:",
                options=["Tümü", "Sadece Doğru Tahminler (Evet)", "Sadece Hatalı Tahminler (Hayır)"],
                horizontal=True,
            )
        if pred_filter == "Sadece Doğru Tahminler (Evet)":
            preds_df = preds_df[preds_df["Doğru Tahmin Mi?"] == "Evet"]
        elif pred_filter == "Sadece Hatalı Tahminler (Hayır)":
            preds_df = preds_df[preds_df["Doğru Tahmin Mi?"] == "Hayır"]

        st.caption(f"Gösterilen test verisi filmi sayısı: **{len(preds_df):,}**")
        st.dataframe(
            preds_df.head(200),
            width="stretch",
            hide_index=True,
            column_config={
                "Film Adı": st.column_config.TextColumn("Film Adı", width="medium"),
                "Bütçe ($)": st.column_config.NumberColumn("Bütçe ($)", format="$%d"),
                "Gerçek Gelir ($)": st.column_config.NumberColumn("Gerçek Gişe Geliri ($)", format="$%d"),
                "Gerçek Sınıf": st.column_config.TextColumn("Gerçek Sınıf"),
                "Tahmin Edilen Sınıf": st.column_config.TextColumn("Model Tahmini"),
                "Yüksek Gelir Olasılığı (%)": st.column_config.ProgressColumn(
                    "Yüksek Gelir Olasılığı",
                    min_value=0.0,
                    max_value=100.0,
                    format="%%%.1f",
                ),
                "Doğru Tahmin Mi?": st.column_config.TextColumn("İsabet Durumu"),
            },
        )


# ==============================================================================
# SAYFA 6: FİLM ÖNERİ SİSTEMİ
# ==============================================================================

def render_recommendation_page(df: pd.DataFrame) -> None:
    render_page_header(
        title="TF-IDF ve Kosinüs Benzerliği ile İçerik Tabanlı Film Öneri Sistemi",
        subtitle=(
            "Seçilen filmin özetini (overview), türlerini, anahtar kelimelerini, başrol oyuncularını ve "
            "yönetmenini TF-IDF vektörleriyle analiz ederek içerikçe en yakın filmleri önerir."
        ),
        badge="ALGORİTMA 4 • DOĞAL DİL İŞLEME & ÖNERİ MOTORU",
        icon_svg=REC_SVG,
    )

    engine = cached_recommendation_engine(df)

    # Teknik Açıklama: TF-IDF Vektör Uzayı ve Kosinüs Benzerliği
    render_section_header(
        title="🔤 TF-IDF Vektörleştirme (1-2 N-Gram) ve Kosinüs Benzerliği (Cosine Similarity)",
        what_it_is="Filmlerin özet, tür, anahtar kelime, oyuncu ve yönetmen metinlerini terim sıklığı-ters doküman sıklığı (TF-IDF) ile 15.000 boyutlu vektörlere dönüştürüp aralarındaki açının kosinüsünü hesaplayan NLP modelidir.",
        what_it_affects="Sık geçen sıradan kelimelerin ağırlığını düşürüp ayırt edici tematik kelimeleri, yönetmeni ve oyuncu kadrosunu öne çıkararak seçilen filme içerikçe en yakın yapımları sıralar.",
    )

    # Arama ve Seçim Kutusu
    with st.container(border=True):
        c_search, c_select, c_count = st.columns([1.2, 1.6, 0.8])
        with c_search:
            search_query = st.text_input(
                "🔎 Hızlı Film Ara (İsteğe Bağlı):",
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
                "🎬 Referans Filmi Seçin:",
                options=filtered_titles,
                index=default_idx,
            )

        with c_count:
            top_n_recs = st.slider(
                "Öneri Sayısı",
                min_value=5,
                max_value=20,
                value=10,
                step=5,
            )

    try:
        rec_res = get_movie_recommendations(engine, selected_title, top_n=top_n_recs)
    except Exception as exc:
        st.error(f"Film önerisi hesaplanırken hata oluştu: {exc}")
        return

    sel_movie = rec_res["selected_movie"]
    rec_df = rec_res["recommendations_df"]

    # Seçilen Film Kartı (Hero Card)
    with st.container(border=True):
        h_left, h_right = st.columns([1.8, 1.2])
        with h_left:
            st.markdown(f"### 🎬 {sel_movie['title']} ({sel_movie['release_year']})")
            st.markdown(
                f"**Yönetmen:** `{sel_movie['director']}` &nbsp;|&nbsp; "
                f"**Süre:** `{sel_movie['runtime']:.0f} dk` &nbsp;|&nbsp; "
                f"**Puan:** `⭐ {sel_movie['vote_average']:.1f} / 10 ({sel_movie['vote_count']:,} oy)`"
            )
            st.markdown(
                f"**🎭 Türler:** {render_tags_html(sel_movie['genres'], primary=True)}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**👥 Başrol Oyuncuları:** {render_tags_html(sel_movie['cast'], primary=False)}",
                unsafe_allow_html=True,
            )
            st.markdown(f"**📝 Konu Özeti:** {sel_movie['overview'] or 'Özet bulunmuyor.'}")

        with h_right:
            m1, m2 = st.columns(2)
            m1.metric("💰 Yapım Bütçesi", format_currency_short(sel_movie["budget"]))
            m2.metric("💵 Gişe Geliri", format_currency_short(sel_movie["revenue"]))
            m3, m4 = st.columns(2)
            m3.metric("🔥 Popülerlik Skoru", f"{sel_movie['popularity']:.1f}")
            m4.metric("🔤 TF-IDF Kelime Uzayı", f"{rec_res['vocab_size']:,} terim")
            if sel_movie["keywords"]:
                st.caption(f"**🏷️ Anahtar Kelimeler:** {sel_movie['keywords']}")

    # Otomatik Kayan Önerilen Filmler Afiş Vitrini
    st.markdown(f"#### 🎞️ '{sel_movie['title']}' İçin Önerilen Filmler Afiş Vitrini")
    st.caption(
        "Önerilen filmlerin afişleri (Sağ üstte IMDb/TMDB puan rozeti, sol üstte benzerlik yüzdesi rozeti yer alır; fareyle üzerine gelince duraklar veya `<` `>` oklarıyla kaydırılabilir)."
    )
    rec_poster_items = get_recommended_movies_with_posters(rec_df)
    render_poster_carousel(rec_poster_items, height=335, speed_seconds=80)

    # En Güçlü İlk 3 Öneri Kartı
    st.markdown(f"#### ✨ '{sel_movie['title']}' İçin En Güçlü İlk 3 Öneri")
    top3_cols = st.columns(3)
    for idx, (_, r_row) in enumerate(rec_df.head(3).iterrows()):
        with top3_cols[idx]:
            with st.container(border=True):
                st.markdown(
                    f"**#{int(r_row['Sıra'])} • {r_row['Film Adı']} ({int(r_row['Yayın Yılı'])})**"
                )
                st.markdown(
                    f"<span class='tag-chip tag-chip-success'>%{r_row['Benzerlik Yüzdesi (%)']:.1f} Benzerlik</span>"
                    f"<span class='tag-chip'>⭐ {r_row['Ortalama Puan']:.1f}</span>"
                    f"<span class='tag-chip'>🎬 {html.escape(str(r_row['Yönetmen']))}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"**🎭 Türler:** {r_row['Türler']}")
                st.caption(f"**🔗 Ortak Unsurlar:** {r_row['Ortak Unsurlar']}")

    # Tüm Öneriler Tablosu ve Grafik Sekmeleri
    tab_table, tab_chart = st.tabs(
        [
            f"📋 En Benzer {len(rec_df)} Film Karşılaştırma Tablosu",
            "📊 Kosinüs Benzerliği Grafiği",
        ]
    )

    with tab_table:
        display_rec = rec_df[
            [
                "Sıra",
                "Film Adı",
                "Yayın Yılı",
                "Benzerlik Yüzdesi (%)",
                "Ortalama Puan",
                "Türler",
                "Yönetmen",
                "Ortak Unsurlar",
            ]
        ]
        st.dataframe(
            display_rec,
            width="stretch",
            hide_index=True,
            column_config={
                "Sıra": st.column_config.NumberColumn("#", format="%d", width="small"),
                "Film Adı": st.column_config.TextColumn("Önerilen Film", width="medium"),
                "Yayın Yılı": st.column_config.NumberColumn("Yıl", format="%d", width="small"),
                "Benzerlik Yüzdesi (%)": st.column_config.ProgressColumn(
                    "İçerik Benzerliği (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%%%.2f",
                ),
                "Ortalama Puan": st.column_config.NumberColumn("Puan", format="⭐ %.1f"),
                "Türler": st.column_config.TextColumn("Türler", width="medium"),
                "Yönetmen": st.column_config.TextColumn("Yönetmen", width="medium"),
                "Ortak Unsurlar": st.column_config.TextColumn(
                    "Neden Önerildi? (Ortak Unsurlar)", width="large"
                ),
            },
        )

    with tab_chart:
        render_chart(
            plot_recommendations_similarity(rec_df, sel_movie["title"]),
            key="rec_sim_chart",
        )


# ==============================================================================
# ANA UYGULAMA AKIŞI VE SOL MENÜ
# ==============================================================================

def main() -> None:
    inject_custom_styles()
    st.markdown(get_sidebar_custom_css(), unsafe_allow_html=True)

    brand_uri = svg_to_data_uri(BRAND_LOGO_SVG)
    pin_uri = svg_to_data_uri(PIN_SVG)
    chart_uri = svg_to_data_uri(SUMMARY_CHART_SVG)
    movie_card_uri = svg_to_data_uri(CARD_MOVIE_SVG)
    rev_card_uri = svg_to_data_uri(CARD_REV_SVG)
    budget_card_uri = svg_to_data_uri(CARD_BUDGET_SVG)
    file_card_uri = svg_to_data_uri(CARD_FILE_SVG)

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand-box">
            <img src="{brand_uri}" alt="Film Analiz Platformu Logo" />
            <div class="sidebar-brand-title">Film Analiz Platformu</div>
        </div>
        <div class="sidebar-brand-sub">TMDB 5000 Veri Madenciliği Sistemi</div>
        <div class="sidebar-menu-header">
            <img src="{pin_uri}" alt="Menü" />
            <span>Menü</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu_options = [
        "1. Ana Sayfa",
        "2. Veri Seti İnceleme",
        "3. Apriori Birliktelik Analizi",
        "4. K-Means Kümeleme",
        "5. Random Forest Tahmin",
        "6. Film Öneri Sistemi",
    ]
    menu_labels = {
        "1. Ana Sayfa": "Ana Sayfa",
        "2. Veri Seti İnceleme": "Veri Seti İnceleme",
        "3. Apriori Birliktelik Analizi": "Apriori Birliktelik Analizi",
        "4. K-Means Kümeleme": "K-Means Kümeleme",
        "5. Random Forest Tahmin": "Random Forest Tahmin",
        "6. Film Öneri Sistemi": "Film Öneri Sistemi",
    }
    selected_page = st.sidebar.radio(
        "Menü",
        menu_options,
        format_func=lambda x: menu_labels.get(x, x),
        label_visibility="collapsed",
    )

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
        <div class="sidebar-summary-title">
            <img src="{chart_uri}" alt="Özet" />
            <span>Veri Seti Özeti</span>
        </div>
        <div class="sidebar-stat-card">
            <div class="sidebar-stat-icon-box" style="background:#fef2f2;">
                <img src="{movie_card_uri}" alt="Toplam Film" />
            </div>
            <div>
                <div class="sidebar-stat-label">Toplam Film</div>
                <div class="sidebar-stat-val">{len(df):,}</div>
            </div>
        </div>
        <div class="sidebar-stat-card">
            <div class="sidebar-stat-icon-box" style="background:#ecfdf5;">
                <img src="{rev_card_uri}" alt="Gelir Bilinen" />
            </div>
            <div>
                <div class="sidebar-stat-label">Gelir Bilinen</div>
                <div class="sidebar-stat-val">{report['known_revenue_count']:,}</div>
            </div>
        </div>
        <div class="sidebar-stat-card">
            <div class="sidebar-stat-icon-box" style="background:#eff6ff;">
                <img src="{budget_card_uri}" alt="Bütçe Bilinen" />
            </div>
            <div>
                <div class="sidebar-stat-label">Bütçe Bilinen</div>
                <div class="sidebar-stat-val">{report['known_budget_count']:,}</div>
            </div>
        </div>
        <div class="sidebar-clean-badge">
            <img src="{file_card_uri}" alt="Temiz Veri" />
            <div>
                <div style="font-size:12.5px; font-weight:700; color:#065f46;">Temiz Veri</div>
                <div style="font-size:11.5px; color:#047857;">data/cleaned_movies.csv</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
