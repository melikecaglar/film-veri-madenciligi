"""
Veri Yükleme, Birleştirme ve Ön İşleme (Data Preprocessing) Modülü.

Bu modül:
1. tmdb_5000_movies.csv ve tmdb_5000_credits.csv dosyalarını okur.
2. id ve movie_id sütunları üzerinden inner join ile birleştirir.
3. Birleştirme ve veri kaybı kontrollerini yapar.
4. Eksik verileri temizler ve raporlar.
5. JSON formatındaki sütunları (genres, keywords, cast, crew) güvenli şekilde ayrıştırır.
6. Yeni özellikler (release_year, profit, profit_ratio, genre_list, keyword_list, cast_list, director, log1p dönüşümleri) üretir.
7. Temizlenmiş veri setini data/cleaned_movies.csv olarak kaydeder.
"""

import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def resolve_file_path(filename: str, base_dir: Optional[str] = None) -> str:
    """
    Verilen dosya adını önce data/ klasöründe, ardından proje kök dizininde arar.
    Bulunamazsa FileNotFoundError fırlatır.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    candidates = [
        os.path.join(base_dir, "data", filename),
        os.path.join(base_dir, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"'{filename}' dosyası bulunamadı. Aranan konumlar: {candidates}"
    )


def parse_json_names(json_str: Any, max_items: Optional[int] = None) -> List[str]:
    """
    JSON formatındaki metin sütunlarından (genres, keywords, cast) 'name' alanlarını çıkarır.
    Boş veya hatalı kayıtlar için güvenli hata kontrolü içerir.
    """
    if pd.isna(json_str) or not isinstance(json_str, str) or not json_str.strip():
        return []
    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            return []
        names = [
            str(item.get("name", "")).strip()
            for item in data
            if isinstance(item, dict) and item.get("name")
        ]
        if max_items is not None:
            return names[:max_items]
        return names
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return []


def extract_director(crew_str: Any) -> str:
    """
    JSON formatındaki 'crew' sütunundan job == 'Director' olan kişinin adını döndürür.
    Bulunamazsa veya hatalıysa boş metin ('') döndürür.
    """
    if pd.isna(crew_str) or not isinstance(crew_str, str) or not crew_str.strip():
        return ""
    try:
        crew_data = json.loads(crew_str)
        if not isinstance(crew_data, list):
            return ""
        for member in crew_data:
            if isinstance(member, dict) and member.get("job") == "Director":
                name = member.get("name", "")
                if name:
                    return str(name).strip()
        return ""
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return ""


def safe_parse_list_column(val: Any) -> List[str]:
    """
    CSV'den geri okunan liste sütunlarını (JSON veya Python list literal) tekrar listeye çevirir.
    """
    if isinstance(val, list):
        return val
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return []
    val_str = val.strip()
    try:
        parsed = json.loads(val_str)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


def load_and_merge_raw_datasets(
    movies_path: Optional[str] = None,
    credits_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    tmdb_5000_movies.csv ve tmdb_5000_credits.csv dosyalarını okur ve birleştirir.
    Birleştirme öncesi ve sonrası veri bütünlüğü raporunu döndürür.
    """
    if movies_path is None:
        movies_path = resolve_file_path("tmdb_5000_movies.csv")
    if credits_path is None:
        credits_path = resolve_file_path("tmdb_5000_credits.csv")

    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Movies CSV dosyası bulunamadı: {movies_path}")
    if not os.path.exists(credits_path):
        raise FileNotFoundError(f"Credits CSV dosyası bulunamadı: {credits_path}")

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Sütun kontrolü
    required_movies_cols = {"id", "title", "budget", "genres", "keywords", "overview", "popularity", "release_date", "revenue", "runtime", "vote_average", "vote_count"}
    required_credits_cols = {"movie_id", "title", "cast", "crew"}

    missing_m = required_movies_cols - set(movies.columns)
    missing_c = required_credits_cols - set(credits.columns)
    if missing_m:
        raise ValueError(f"movies.csv dosyasında eksik sütunlar var: {missing_m}")
    if missing_c:
        raise ValueError(f"credits.csv dosyasında eksik sütunlar var: {missing_c}")

    movies_count = len(movies)
    credits_count = len(credits)

    # Ortak ve eşleşmeyen kimliklerin kontrolü
    movies_ids = set(movies["id"].dropna().unique())
    credits_ids = set(credits["movie_id"].dropna().unique())
    common_ids = movies_ids.intersection(credits_ids)
    unmatched_in_movies = movies_ids - credits_ids
    unmatched_in_credits = credits_ids - movies_ids

    dup_movies_id = int(movies["id"].duplicated().sum())
    dup_credits_id = int(credits["movie_id"].duplicated().sum())

    # Birleştirme işlemi (talimata uygun olarak)
    merged = pd.merge(movies, credits, left_on="id", right_on="movie_id", how="inner")

    # İkinci dosyadaki tekrar eden title sütununun kaldırılması
    dropped_duplicate_title = False
    if "title_x" in merged.columns and "title_y" in merged.columns:
        merged = merged.rename(columns={"title_x": "title"})
        merged = merged.drop(columns=["title_y"])
        dropped_duplicate_title = True

    merged_count = len(merged)
    data_loss_count = movies_count - merged_count

    merge_report = {
        "movies_row_count": movies_count,
        "movies_col_count": len(movies.columns),
        "credits_row_count": credits_count,
        "credits_col_count": len(credits.columns),
        "common_id_count": len(common_ids),
        "unmatched_movies_count": len(unmatched_in_movies),
        "unmatched_credits_count": len(unmatched_in_credits),
        "duplicate_movie_ids_in_movies": dup_movies_id,
        "duplicate_movie_ids_in_credits": dup_credits_id,
        "merged_row_count": merged_count,
        "merged_col_count": len(merged.columns),
        "dropped_duplicate_title": dropped_duplicate_title,
        "data_loss_count": data_loss_count,
        "has_data_loss": data_loss_count > 0,
    }

    return merged, merge_report


def preprocess_movies_data(
    df_merged: pd.DataFrame,
    merge_report: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Birleştirilmiş film veri seti üzerinde tüm veri temizleme, JSON ayrıştırma
    ve yeni özellik türetme işlemlerini gerçekleştirir.
    """
    df = df_merged.copy()

    # İşlem öncesi eksik veri istatistikleri
    initial_nulls = df.isnull().sum().to_dict()
    zero_budget_count = int((df["budget"] == 0).sum())
    zero_revenue_count = int((df["revenue"] == 0).sum())
    both_zero_count = int(((df["budget"] == 0) & (df["revenue"] == 0)).sum())

    # 4.1. Eksik verilerin doldurulması ve ele alınması
    # overview boşsa boş metin olarak değerlendir
    overview_null_count = int(df["overview"].isnull().sum())
    df["overview"] = df["overview"].fillna("").astype(str)

    # runtime eksikse medyan ile doldur
    runtime_null_count = int(df["runtime"].isnull().sum())
    runtime_median = float(df["runtime"].median())
    df["runtime"] = df["runtime"].fillna(runtime_median)

    # release_date eksik olan kayıtları kontrol et
    release_date_null_count = int(df["release_date"].isnull().sum())
    parsed_dates = pd.to_datetime(df["release_date"], errors="coerce")

    # 4.2. JSON formatındaki sütunları düzenleme
    df["genre_list"] = df["genres"].apply(lambda x: parse_json_names(x))
    df["keyword_list"] = df["keywords"].apply(lambda x: parse_json_names(x))
    df["cast_list"] = df["cast"].apply(lambda x: parse_json_names(x, max_items=5))
    df["director"] = df["crew"].apply(extract_director)

    empty_genres_count = int(df["genre_list"].apply(lambda x: len(x) == 0).sum())
    empty_keywords_count = int(df["keyword_list"].apply(lambda x: len(x) == 0).sum())
    empty_cast_count = int(df["cast_list"].apply(lambda x: len(x) == 0).sum())
    empty_director_count = int((df["director"] == "").sum())

    # 4.3. Yeni özellikler oluşturma
    # release_year: Filmin yayın yılı (eksik olan 1 kayıt varsa medyan yıl ile doldurulabilir veya Int64/float tutulur)
    release_year_median = float(parsed_dates.dt.year.median())
    df["release_year"] = parsed_dates.dt.year.fillna(release_year_median).astype(int)

    # budget ve revenue değerleri 0 olan kayıtları eksik/bilinmeyen olarak işaretleme
    df["budget_is_known"] = df["budget"] > 0
    df["revenue_is_known"] = df["revenue"] > 0

    # profit: Gelir - bütçe
    df["profit"] = df["revenue"] - df["budget"]

    # profit_ratio: Gelir / bütçe (bütçe sıfır veya eksikse hesaplama yapma -> NaN)
    df["profit_ratio"] = np.where(
        df["budget"] > 0,
        df["revenue"] / df["budget"],
        np.nan,
    )

    # Aşırı sağa çarpık dağılımlar için log1p dönüşümü
    df["budget_log"] = np.log1p(df["budget"].clip(lower=0))
    df["revenue_log"] = np.log1p(df["revenue"].clip(lower=0))
    df["popularity_log"] = np.log1p(df["popularity"].clip(lower=0))
    df["vote_count_log"] = np.log1p(df["vote_count"].clip(lower=0))

    # Görselleştirme ve tablo gösterimi için okunabilir metin sütunları
    df["genres_str"] = df["genre_list"].apply(lambda lst: ", ".join(lst))
    df["keywords_str"] = df["keyword_list"].apply(lambda lst: ", ".join(lst))
    df["cast_str"] = df["cast_list"].apply(lambda lst: ", ".join(lst))

    # Kayıt işlemi
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if save_path is None:
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, "cleaned_movies.csv")

    # CSV'ye kaydederken liste sütunlarını JSON string olarak sakla ki geri okurken kayıpsız olsun
    df_to_save = df.copy()
    for list_col in ["genre_list", "keyword_list", "cast_list"]:
        df_to_save[list_col] = df_to_save[list_col].apply(
            lambda lst: json.dumps(lst, ensure_ascii=False)
        )

    try:
        if not os.path.exists(save_path):
            df_to_save.to_csv(save_path, index=False, encoding="utf-8")
    except OSError:
        pass

    # Kök dizine de kopyasını kaydet (hem data/cleaned_movies.csv hem cleaned_movies.csv erişilebilir olsun)
    root_clean_path = os.path.join(base_dir, "cleaned_movies.csv")
    if os.path.abspath(save_path) != os.path.abspath(root_clean_path):
        try:
            if not os.path.exists(root_clean_path):
                df_to_save.to_csv(root_clean_path, index=False, encoding="utf-8")
        except OSError:
            pass

    final_nulls = df.isnull().sum().to_dict()

    report = {
        "merge_report": merge_report or {},
        "initial_nulls": initial_nulls,
        "final_nulls": final_nulls,
        "overview_null_filled": overview_null_count,
        "runtime_null_filled": runtime_null_count,
        "runtime_median_used": runtime_median,
        "release_date_null_count": release_date_null_count,
        "release_year_median_used": int(release_year_median),
        "zero_budget_count": zero_budget_count,
        "known_budget_count": int(df["budget_is_known"].sum()),
        "zero_revenue_count": zero_revenue_count,
        "known_revenue_count": int(df["revenue_is_known"].sum()),
        "both_zero_budget_revenue_count": both_zero_count,
        "both_known_budget_revenue_count": int((df["budget_is_known"] & df["revenue_is_known"]).sum()),
        "empty_genres_count": empty_genres_count,
        "empty_keywords_count": empty_keywords_count,
        "empty_cast_count": empty_cast_count,
        "empty_director_count": empty_director_count,
        "saved_path": save_path,
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

    return df, report


def get_or_create_cleaned_data(force_reprocess: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Veri setini birleştirir, temizler ve raporla birlikte döndürür.
    Sonuçları tutarlı şekilde üretir ve cleaned_movies.csv dosyasını günceller.
    """
    merged_df, merge_report = load_and_merge_raw_datasets()
    cleaned_df, prep_report = preprocess_movies_data(merged_df, merge_report=merge_report)
    return cleaned_df, prep_report


if __name__ == "__main__":
    df_clean, report_info = get_or_create_cleaned_data(force_reprocess=True)
    print("=== VERİ BİRLEŞTİRME VE ÖN İŞLEME RAPORU ===")
    print(f"Toplam Film Sayısı: {report_info['total_rows']}")
    print(f"Toplam Sütun Sayısı: {report_info['total_columns']}")
    print(f"Birleştirme Veri Kaybı: {report_info['merge_report']['data_loss_count']}")
    print(f"Overview Eksik Doldurulan: {report_info['overview_null_filled']}")
    print(f"Runtime Eksik Doldurulan: {report_info['runtime_null_filled']} (Medyan: {report_info['runtime_median_used']})")
    print(f"Release Date Eksik Sayısı: {report_info['release_date_null_count']}")
    print(f"Bütçesi 0 (Bilinmeyen) Film Sayısı: {report_info['zero_budget_count']} | Bilinen: {report_info['known_budget_count']}")
    print(f"Geliri 0 (Bilinmeyen) Film Sayısı: {report_info['zero_revenue_count']} | Bilinen: {report_info['known_revenue_count']}")
    print(f"Boş Tür Sayısı: {report_info['empty_genres_count']} | Boş Anahtar Kelime Sayısı: {report_info['empty_keywords_count']}")
    print(f"Boş Oyuncu Sayısı: {report_info['empty_cast_count']} | Yönetmeni Bilinmeyen: {report_info['empty_director_count']}")
    print(f"Kaydedilen Dosya: {report_info['saved_path']}")
    print("\nİlk 3 Film Özeti:")
    print(df_clean[["title", "release_year", "genre_list", "director", "profit", "profit_ratio"]].head(3))
