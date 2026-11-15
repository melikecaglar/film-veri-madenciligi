"""
Dördüncü Yöntem: TF-IDF ve Cosine Similarity ile İçerik Tabanlı Film Öneri Sistemi Modülü.

Bu modül:
1. Her film için overview, genres (genre_list), keywords (keyword_list), cast_list ve director
   sütunlarını birleştirerek zengin bir içerik metni (combined_features) oluşturur.
2. TfidfVectorizer ile metinleri TF-IDF sayısal vektör matrisine dönüştürür.
3. cosine_similarity ile filmler arasındaki kosinüs benzerliğini hesaplar.
4. Kullanıcının seçtiği veya aradığı filme en benzer N filmi (varsayılan 10) benzerlik yüzdesi,
   türü, puanı, yayın yılı ve ortak içerik unsurlarıyla birlikte döndürür.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_combined_text(row: pd.Series) -> str:
    """
    Bir filmin başlık, özet (overview), türler (genre_list), anahtar kelimeler (keyword_list),
    ilk 5 oyuncu (cast_list) ve yönetmen (director) bilgilerini tek bir metinde birleştirir.
    Örnek:
    'Avatar Action Adventure Fantasy alien future space Sam Worthington Zoe Saldana James Cameron'
    """
    title_str = str(row.get("title", "")).strip()
    overview_str = str(row.get("overview", "")).strip()

    genres = row.get("genre_list", [])
    genres_str = " ".join(genres) if isinstance(genres, list) else str(genres)

    keywords = row.get("keyword_list", [])
    keywords_str = " ".join(keywords) if isinstance(keywords, list) else str(keywords)

    cast = row.get("cast_list", [])
    cast_str = " ".join(cast) if isinstance(cast, list) else str(cast)

    director_str = str(row.get("director", "")).strip()

    # Tür, anahtar kelime, oyuncu ve yönetmen bilgilerinin uzun overview metni içinde
    # kaybolmaması için dengeli bir şekilde metne dahil ediyoruz.
    parts = [
        title_str,
        genres_str,
        genres_str,
        keywords_str,
        keywords_str,
        cast_str,
        director_str,
        director_str,
        overview_str,
    ]
    return " ".join(p for p in parts if p)


def build_tfidf_engine(
    df: pd.DataFrame,
    max_features: int = 15000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Dict[str, Any]:
    """
    Veri seti üzerindeki tüm filmler için birleştirilmiş metinleri oluşturur,
    TF-IDF vektör matrisini hesaplar ve öneri motoru nesnesini döndürür.
    """
    work_df = df.copy().reset_index(drop=True)
    work_df["combined_features"] = work_df.apply(create_combined_text, axis=1)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
    )
    tfidf_matrix = vectorizer.fit_transform(work_df["combined_features"])

    # Başlıktan satır indeksine hızlı erişim haritası (küçük harf duyarsız arama desteğiyle)
    title_to_idx: Dict[str, int] = {}
    lower_to_idx: Dict[str, int] = {}
    for idx, title in enumerate(work_df["title"].astype(str)):
        if title not in title_to_idx:
            title_to_idx[title] = idx
        lower_t = title.lower().strip()
        if lower_t not in lower_to_idx:
            lower_to_idx[lower_t] = idx

    return {
        "df": work_df,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "title_to_idx": title_to_idx,
        "lower_to_idx": lower_to_idx,
        "vocab_size": len(vectorizer.vocabulary_),
    }


def search_movies_by_query(
    df: pd.DataFrame, query: str, limit: int = 20
) -> List[str]:
    """
    Kullanıcının arama sorgusuna göre film adlarını filtreler.
    """
    if not query or not query.strip():
        return df["title"].dropna().astype(str).head(limit).tolist()

    q = query.strip().lower()
    titles = df["title"].dropna().astype(str)
    starts_with = titles[titles.str.lower().str.startswith(q)].tolist()
    contains = titles[
        titles.str.lower().str.contains(q, regex=False)
        & (~titles.str.lower().str.startswith(q))
    ].tolist()
    return (starts_with + contains)[:limit]


def get_movie_recommendations(
    engine: Dict[str, Any],
    movie_title: str,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Seçilen filme TF-IDF + Cosine Similarity ile en benzer N filmi bulur.
    Seçilen filmi öneri listesinden çıkarır ve benzerlik yüzdelerini hesaplar.
    """
    work_df: pd.DataFrame = engine["df"]
    tfidf_matrix = engine["tfidf_matrix"]
    title_to_idx: Dict[str, int] = engine["title_to_idx"]
    lower_to_idx: Dict[str, int] = engine["lower_to_idx"]

    if not movie_title or not str(movie_title).strip():
        raise ValueError("Lütfen geçerli bir film adı seçin veya girin.")

    clean_title = str(movie_title).strip()
    if clean_title in title_to_idx:
        target_idx = title_to_idx[clean_title]
    elif clean_title.lower() in lower_to_idx:
        target_idx = lower_to_idx[clean_title.lower()]
    else:
        # Kısmi eşleşme kontrolü
        matches = search_movies_by_query(work_df, clean_title, limit=1)
        if not matches:
            raise ValueError(f"'{movie_title}' adlı film veri setinde bulunamadı.")
        target_idx = title_to_idx[matches[0]]

    target_row = work_df.iloc[target_idx]

    # Seçilen filmin TF-IDF vektörü ile tüm filmler arasındaki kosinüs benzerliğini hesapla
    target_vec = tfidf_matrix[target_idx]
    sim_scores = cosine_similarity(target_vec, tfidf_matrix).flatten()

    # Kendisini (ve aynı isimli olası tekrar kaydını) öneri listesinden çıkar
    sorted_indices = np.argsort(sim_scores)[::-1]
    rec_indices: List[int] = []
    for idx in sorted_indices:
        if int(idx) == int(target_idx):
            continue
        if str(work_df.iloc[idx]["title"]).strip().lower() == str(target_row["title"]).strip().lower():
            continue
        rec_indices.append(int(idx))
        if len(rec_indices) >= top_n:
            break

    target_genres = set(target_row["genre_list"]) if isinstance(target_row["genre_list"], list) else set()
    target_keywords = set(target_row["keyword_list"]) if isinstance(target_row["keyword_list"], list) else set()
    target_cast = set(target_row["cast_list"]) if isinstance(target_row["cast_list"], list) else set()
    target_director = str(target_row.get("director", "")).strip()

    rec_rows: List[Dict[str, Any]] = []
    for rank, idx in enumerate(rec_indices, start=1):
        row = work_df.iloc[idx]
        score = float(sim_scores[idx])

        # Ortak özellikleri belirle (açıklanabilir öneri için)
        row_genres = set(row["genre_list"]) if isinstance(row["genre_list"], list) else set()
        row_keywords = set(row["keyword_list"]) if isinstance(row["keyword_list"], list) else set()
        row_cast = set(row["cast_list"]) if isinstance(row["cast_list"], list) else set()
        row_director = str(row.get("director", "")).strip()

        shared_reasons: List[str] = []
        if target_director and row_director == target_director:
            shared_reasons.append(f"Aynı Yönetmen ({row_director})")
        common_cast = target_cast.intersection(row_cast)
        if common_cast:
            shared_reasons.append(f"Ortak Oyuncu: {', '.join(list(common_cast)[:2])}")
        common_kw = target_keywords.intersection(row_keywords)
        if common_kw:
            shared_reasons.append(f"Ortak Tema: {', '.join(list(common_kw)[:3])}")
        common_g = target_genres.intersection(row_genres)
        if common_g:
            shared_reasons.append(f"Ortak Tür: {', '.join( sorted(list(common_g)) )}")

        rec_rows.append(
            {
                "Sıra": rank,
                "id": int(row["id"]),
                "Film Adı": str(row["title"]),
                "Benzerlik Skoru": round(score, 4),
                "Benzerlik Yüzdesi (%)": round(score * 100, 2),
                "Türler": str(row.get("genres_str", ", ".join(row_genres))),
                "Ortalama Puan": float(row["vote_average"]),
                "Oy Sayısı": int(row["vote_count"]),
                "Yayın Yılı": int(row["release_year"]),
                "Yönetmen": row_director if row_director else "Bilinmiyor",
                "Oyuncular": str(row.get("cast_str", ", ".join(row_cast))),
                "Ortak Unsurlar": " | ".join(shared_reasons) if shared_reasons else "Metin / Konu Benzerliği",
                "Özet": str(row.get("overview", "")),
            }
        )

    recommendations_df = pd.DataFrame(rec_rows)

    selected_movie_info = {
        "id": int(target_row["id"]),
        "title": str(target_row["title"]),
        "release_year": int(target_row["release_year"]),
        "genres": str(target_row.get("genres_str", "")),
        "vote_average": float(target_row["vote_average"]),
        "vote_count": int(target_row["vote_count"]),
        "popularity": float(target_row["popularity"]),
        "budget": float(target_row["budget"]),
        "revenue": float(target_row["revenue"]),
        "runtime": float(target_row["runtime"]),
        "director": target_director if target_director else "Bilinmiyor",
        "cast": str(target_row.get("cast_str", "")),
        "keywords": str(target_row.get("keywords_str", "")),
        "overview": str(target_row.get("overview", "")),
        "combined_sample": str(target_row.get("combined_features", ""))[:300],
    }

    return {
        "selected_movie": selected_movie_info,
        "recommendations_df": recommendations_df,
        "vocab_size": engine["vocab_size"],
    }


if __name__ == "__main__":
    from data_preprocessing import get_or_create_cleaned_data

    df_clean, _ = get_or_create_cleaned_data()
    eng = build_tfidf_engine(df_clean)
    for sample_title in ["Avatar", "The Dark Knight", "Inception"]:
        res = get_movie_recommendations(eng, sample_title, top_n=10)
        print(f"\n=== '{res['selected_movie']['title']}' İÇİN EN BENZER 10 FİLM ===")
        print(
            res["recommendations_df"][
                [
                    "Sıra",
                    "Film Adı",
                    "Benzerlik Yüzdesi (%)",
                    "Türler",
                    "Ortalama Puan",
                    "Yayın Yılı",
                    "Ortak Unsurlar",
                ]
            ].to_string(index=False)
        )
