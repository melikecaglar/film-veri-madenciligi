"""
İkinci Algoritma: K-Means Kümeleme Modülü.

Bu modül:
1. Kümeleme için kullanılacak sayısal (budget, revenue, popularity, runtime, vote_average, vote_count)
   ve kategorik (film türleri - MultiLabelBinarizer) özellikleri hazırlar.
2. Sıfır veya eksik bütçe/gelir değerlerini uygun şekilde ele alır (medyan atama veya filtreleme).
3. Gelir, bütçe, popülerlik ve oy sayısı değişkenlerine log1p dönüşümü uygular.
4. Sayısal özellikleri StandardScaler ile ölçeklendirir.
5. k = 2..8 aralığında K-Means algoritmasını çalıştırarak Elbow Method (WCSS / Inertia)
   ve Silhouette Score değerlerini hesaplar.
6. Seçilen küme sayısına (k) göre her filme bir küme numarası ve küme ortalamalarına dayalı
   açıklayıcı Türkçe küme ismi atar.
7. Görselleştirme için PCA 2B izdüşüm koordinatlarını hesaplar.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


NUMERIC_FEATURES = [
    "budget",
    "revenue",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
]


def prepare_clustering_features(
    df: pd.DataFrame,
    filter_zero_financials: bool = True,
    include_genres: bool = True,
    genre_weight: float = 0.25,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    K-Means kümeleme için veri setini temizler, log1p dönüşümü uygular,
    StandardScaler ile ölçeklendirir ve MultiLabelBinarizer ile türleri ekler.
    """
    work_df = df.copy()

    # Eksik sayısal değerlerin kontrolü
    for col in NUMERIC_FEATURES:
        if col not in work_df.columns:
            raise ValueError(f"Kümeleme için gerekli '{col}' sütunu bulunamadı.")

    initial_count = len(work_df)

    # Sıfır bütçe ve gelir değerlerinin ele alınması
    known_budget_median = float(work_df.loc[work_df["budget"] > 0, "budget"].median())
    known_revenue_median = float(work_df.loc[work_df["revenue"] > 0, "revenue"].median())

    if filter_zero_financials:
        # Bütçesi ve geliri 0'dan büyük (bilinen) filmler üzerinde kümeleme
        work_df = work_df[(work_df["budget"] > 0) & (work_df["revenue"] > 0)].copy()
        work_df = work_df.reset_index(drop=True)
        work_df["budget_imputed"] = work_df["budget"].astype(float)
        work_df["revenue_imputed"] = work_df["revenue"].astype(float)
        financial_handling = (
            f"Bütçe ve geliri 0 (bilinmeyen) olan kayıtlar çıkarıldı; "
            f"gerçek finansal verisi bilinen {len(work_df)} film kullanıldı."
        )
    else:
        # Tüm filmleri koruyarak 0 olan bütçe ve gelirleri bilinen medyan ile doldurma
        work_df = work_df.reset_index(drop=True)
        work_df["budget_imputed"] = np.where(
            work_df["budget"] > 0, work_df["budget"], known_budget_median
        )
        work_df["revenue_imputed"] = np.where(
            work_df["revenue"] > 0, work_df["revenue"], known_revenue_median
        )
        financial_handling = (
            f"Bütçe veya geliri 0 olan kayıtlar bilinen medyan değerlerle "
            f"(Bütçe Medyanı: ${known_budget_median:,.0f}, Gelir Medyanı: ${known_revenue_median:,.0f}) "
            f"doldurularak tüm {len(work_df)} film kümelemeye dahil edildi."
        )

    # Eksik runtime veya diğer sayısal alanları medyan ile tamamla
    for col in ["popularity", "runtime", "vote_average", "vote_count"]:
        if work_df[col].isnull().any():
            work_df[col] = work_df[col].fillna(work_df[col].median())

    # Sağa çarpık değişkenlere log1p dönüşümü
    num_matrix = pd.DataFrame(index=work_df.index)
    num_matrix["budget_log"] = np.log1p(work_df["budget_imputed"].clip(lower=0))
    num_matrix["revenue_log"] = np.log1p(work_df["revenue_imputed"].clip(lower=0))
    num_matrix["popularity_log"] = np.log1p(work_df["popularity"].clip(lower=0))
    num_matrix["runtime"] = work_df["runtime"].astype(float)
    num_matrix["vote_average"] = work_df["vote_average"].astype(float)
    num_matrix["vote_count_log"] = np.log1p(work_df["vote_count"].clip(lower=0))

    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(num_matrix)

    genre_classes: List[str] = []
    if include_genres and "genre_list" in work_df.columns:
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(work_df["genre_list"])
        genre_classes = list(mlb.classes_)
        # Çoklu ikili tür sütunlarının sayısal uzaklığı domine etmemesi için ağırlıklandırma
        X_features = np.hstack([scaled_num, genre_encoded * genre_weight])
    else:
        X_features = scaled_num

    prep_info = {
        "initial_movie_count": initial_count,
        "clustered_movie_count": len(work_df),
        "filter_zero_financials": filter_zero_financials,
        "financial_handling": financial_handling,
        "include_genres": include_genres,
        "genre_count": len(genre_classes),
        "feature_dimension": int(X_features.shape[1]),
    }

    return work_df, X_features, prep_info


def assign_descriptive_cluster_names(summary_df: pd.DataFrame) -> Dict[int, str]:
    """
    Her kümenin ortalama bütçe, gelir, popülerlik, puan ve oy sayısı değerlerini
    veri seti ortalamalarıyla karşılaştırarak benzersiz ve açıklayıcı Türkçe isimler üretir.
    """
    overall_budget = summary_df["avg_budget"].median()
    overall_revenue = summary_df["avg_revenue"].median()
    overall_pop = summary_df["avg_popularity"].median()
    overall_vote = summary_df["avg_vote_average"].median()

    names: Dict[int, str] = {}
    used_labels: set = set()

    # Kümeleri gelir + bütçe büyüklüğüne göre sıralayarak anlamlı arketipler ata
    sorted_clusters = summary_df.sort_values(
        by=["avg_revenue", "avg_budget"], ascending=[False, False]
    )

    for rank, (_, row) in enumerate(sorted_clusters.iterrows()):
        cid = int(row["cluster_id"])
        b = row["avg_budget"]
        r = row["avg_revenue"]
        p = row["avg_popularity"]
        v = row["avg_vote_average"]
        top_g = row.get("top_genres", "")

        if rank == 0 and r >= overall_revenue * 1.3:
            base_label = "Gişe Rekortmeni Dev Yapımlar (Yüksek Bütçe & Gelir)"
        elif v >= overall_vote + 0.35 and p >= overall_pop:
            base_label = "Yüksek Puanlı & Popüler Favori Filmler"
        elif b >= overall_budget and r < overall_revenue:
            base_label = "Yüksek Bütçeli / Beklentiyi Aşamayan Yapımlar"
        elif b < overall_budget and r >= overall_revenue:
            base_label = "Düşük Bütçeli Yüksek Kârlı Başarılar"
        elif v >= overall_vote + 0.2 and b < overall_budget:
            base_label = "Beğenilen Bağımsız & Orta Ölçekli Filmler"
        elif b < overall_budget * 0.6 and p < overall_pop * 0.8:
            base_label = "Düşük Bütçeli & Az Bilinen Niş Filmler"
        elif v < overall_vote - 0.3:
            base_label = "Düşük Puanlı & Düşük Etkileşimli Yapımlar"
        else:
            base_label = "Orta Ölçekli Ana Akım Sinema Filmleri"

        if base_label in used_labels:
            if top_g:
                first_genre = top_g.split(",")[0].strip()
                base_label = f"{base_label} ({first_genre} Ağırlıklı)"
            if base_label in used_labels:
                base_label = f"{base_label} - Küme {cid}"

        used_labels.add(base_label)
        names[cid] = f"Küme {cid}: {base_label}"

    return names


def run_kmeans_analysis(
    df: pd.DataFrame,
    n_clusters: int = 4,
    k_range: Tuple[int, int] = (2, 8),
    filter_zero_financials: bool = True,
    include_genres: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    K-Means kümeleme analizini çalıştırır:
    - k_range (varsayılan 2..8) için Elbow (Inertia) ve Silhouette Score hesaplar.
    - Seçilen n_clusters için K-Means modelini eğitir.
    - Küme özet istatistiklerini ve açıklayıcı küme adlarını oluşturur.
    """
    k_min, k_max = k_range
    if not (2 <= n_clusters <= 12):
        raise ValueError("Küme sayısı (n_clusters) 2 ile 12 arasında olmalıdır.")

    work_df, X_features, prep_info = prepare_clustering_features(
        df,
        filter_zero_financials=filter_zero_financials,
        include_genres=include_genres,
    )

    # 2-8 aralığında Elbow Method ve Silhouette Score değerlendirmesi
    eval_records: List[Dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        km_temp = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_temp = km_temp.fit_predict(X_features)
        inertia_val = float(km_temp.inertia_)
        sil_val = float(silhouette_score(X_features, labels_temp))
        eval_records.append(
            {
                "k": k,
                "inertia": inertia_val,
                "silhouette_score": sil_val,
            }
        )

    evaluation_df = pd.DataFrame(eval_records)
    best_k_row = evaluation_df.sort_values(
        by="silhouette_score", ascending=False
    ).iloc[0]
    optimal_k_by_silhouette = int(best_k_row["k"])

    # Seçilen n_clusters ile ana modeli çalıştır
    final_kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    )
    cluster_labels = final_kmeans.fit_predict(X_features)
    selected_silhouette = float(silhouette_score(X_features, cluster_labels))
    selected_inertia = float(final_kmeans.inertia_)

    work_df["cluster_id"] = cluster_labels

    # 2 Boyutlu PCA İzdüşümü (Görselleştirme için)
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_features)
    work_df["pca_1"] = pca_coords[:, 0]
    work_df["pca_2"] = pca_coords[:, 1]

    # Küme istatistiklerini hesapla
    summary_rows: List[Dict[str, Any]] = []
    for cid in sorted(work_df["cluster_id"].unique()):
        sub = work_df[work_df["cluster_id"] == cid]
        # Kümedeki en baskın 3 film türünü bul
        genre_counts: Dict[str, int] = {}
        for g_list in sub["genre_list"]:
            if isinstance(g_list, list):
                for g in g_list:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        top_3_genres = ", ".join(
            [
                g
                for g, _ in sorted(
                    genre_counts.items(), key=lambda item: item[1], reverse=True
                )[:3]
            ]
        )

        summary_rows.append(
            {
                "cluster_id": int(cid),
                "movie_count": len(sub),
                "percentage": round(len(sub) / len(work_df) * 100, 2),
                "avg_budget": float(sub["budget"].mean()),
                "avg_revenue": float(sub["revenue"].mean()),
                "avg_profit": float(sub["profit"].mean()),
                "avg_popularity": float(sub["popularity"].mean()),
                "avg_runtime": float(sub["runtime"].mean()),
                "avg_vote_average": float(sub["vote_average"].mean()),
                "avg_vote_count": float(sub["vote_count"].mean()),
                "top_genres": top_3_genres,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    cluster_name_map = assign_descriptive_cluster_names(summary_df)
    summary_df["cluster_name"] = summary_df["cluster_id"].map(cluster_name_map)
    work_df["cluster_name"] = work_df["cluster_id"].map(cluster_name_map)

    # Sütun sırasını düzenle
    summary_df = summary_df[
        [
            "cluster_id",
            "cluster_name",
            "movie_count",
            "percentage",
            "avg_budget",
            "avg_revenue",
            "avg_profit",
            "avg_popularity",
            "avg_vote_average",
            "avg_vote_count",
            "avg_runtime",
            "top_genres",
        ]
    ]

    return {
        "clustered_df": work_df,
        "summary_df": summary_df,
        "evaluation_df": evaluation_df,
        "selected_k": n_clusters,
        "selected_silhouette": selected_silhouette,
        "selected_inertia": selected_inertia,
        "optimal_k_by_silhouette": optimal_k_by_silhouette,
        "prep_info": prep_info,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


if __name__ == "__main__":
    from data_preprocessing import get_or_create_cleaned_data

    df_clean, _ = get_or_create_cleaned_data()
    res = run_kmeans_analysis(df_clean, n_clusters=4, filter_zero_financials=True)
    print("=== K-MEANS KÜMELEME ANALİZİ SONUÇLARI ===")
    print(res["prep_info"]["financial_handling"])
    print("\n2-8 Küme Aralığı Değerlendirme Tablosu (Elbow & Silhouette):")
    print(res["evaluation_df"].to_string(index=False))
    print(f"\nSeçilen k={res['selected_k']} Silhouette Score: {res['selected_silhouette']:.4f}")
    print("\nKüme Özetleri:")
    print(
        res["summary_df"][
            [
                "cluster_name",
                "movie_count",
                "avg_budget",
                "avg_revenue",
                "avg_popularity",
                "avg_vote_average",
                "top_genres",
            ]
        ].to_string(index=False)
    )
