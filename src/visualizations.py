"""
Görselleştirme ve Veri Analizi Modülü (visualizations.py).

Bu modül:
Genel veri analizi ve veri madenciliği algoritmalarının sonuçlarını görselleştirmek için
gerekli tüm grafikleri üretir:
1. En çok film üretilen türler
2. Yıllara göre film sayısı
3. En yüksek gelir elde eden filmler
4. En yüksek bütçeli filmler
5. En popüler 10 film
6. Film puanlarının dağılımı
7. Bütçe ve gelir arasındaki ilişki
8. Popülerlik ve oy ortalaması arasındaki ilişki
9. En çok film yöneten yönetmenler
10. Film türlerine göre ortalama gelir
Ayrıca Apriori, K-Means, Random Forest ve Film Öneri Sistemi sayfalarının özel grafiklerini oluşturur.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

# Matplotlib önbellek dizini uyarısını önle
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache")
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Modern grafik stili
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10


def _format_millions(x: float, pos: Any = None) -> str:
    """Dolar değerlerini Milyon ($M) veya Milyar ($B) olarak biçimlendirir."""
    if abs(x) >= 1e9:
        return f"${x / 1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.0f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


# ==============================================================================
# GENEL VERİ ANALİZİ GRAFİKLERİ (10 TEMEL GRAFİK)
# ==============================================================================

def plot_top_genres_count(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """1. En çok film üretilen türler grafiği."""
    genre_counts: Dict[str, int] = {}
    for g_list in df["genre_list"]:
        if isinstance(g_list, list):
            for g in g_list:
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

    g_df = (
        pd.DataFrame(list(genre_counts.items()), columns=["Tür", "Film Sayısı"])
        .sort_values(by="Film Sayısı", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=g_df,
        x="Film Sayısı",
        y="Tür",
        hue="Tür",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, padding=4, fontsize=9)
    ax.set_title("En Çok Film Üretilen Türler")
    ax.set_xlabel("Film Sayısı")
    ax.set_ylabel("Film Türü")
    ax.set_xlim(0, g_df["Film Sayısı"].max() * 1.12)
    fig.tight_layout()
    return fig


def plot_movies_by_year(df: pd.DataFrame) -> plt.Figure:
    """2. Yıllara göre film sayısı grafiği."""
    year_df = (
        df[df["release_year"] >= 1950]
        .groupby("release_year")
        .size()
        .reset_index(name="Film Sayısı")
        .sort_values("release_year")
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        year_df["release_year"],
        year_df["Film Sayısı"],
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
    )
    ax.fill_between(
        year_df["release_year"],
        year_df["Film Sayısı"],
        color="#1f77b4",
        alpha=0.18,
    )
    ax.set_title("Yıllara Göre Üretilen Film Sayısı (1950 - Günümüz)")
    ax.set_xlabel("Yayın Yılı")
    ax.set_ylabel("Film Sayısı")
    fig.tight_layout()
    return fig


def plot_top_revenue_movies(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """3. En yüksek gelir elde eden filmler grafiği."""
    top_rev = (
        df.sort_values(by="revenue", ascending=False)
        .head(top_n)
        .copy()
    )
    top_rev["revenue_m"] = top_rev["revenue"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=top_rev,
        x="revenue_m",
        y="title",
        hue="title",
        palette="magma",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="$%.0fM", padding=4, fontsize=8.5)
    ax.set_title(f"En Yüksek Gelir Elde Eden {top_n} Film")
    ax.set_xlabel("Gelir (Milyon $)")
    ax.set_ylabel("Film Adı")
    ax.set_xlim(0, top_rev["revenue_m"].max() * 1.15)
    fig.tight_layout()
    return fig


def plot_top_budget_movies(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """4. En yüksek bütçeli filmler grafiği."""
    top_bud = (
        df.sort_values(by="budget", ascending=False)
        .head(top_n)
        .copy()
    )
    top_bud["budget_m"] = top_bud["budget"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=top_bud,
        x="budget_m",
        y="title",
        hue="title",
        palette="rocket",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="$%.0fM", padding=4, fontsize=8.5)
    ax.set_title(f"En Yüksek Bütçeli {top_n} Film")
    ax.set_xlabel("Bütçe (Milyon $)")
    ax.set_ylabel("Film Adı")
    ax.set_xlim(0, top_bud["budget_m"].max() * 1.15)
    fig.tight_layout()
    return fig


def plot_top_popular_movies(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """5. En popüler 10 film grafiği."""
    top_pop = (
        df.sort_values(by="popularity", ascending=False)
        .head(top_n)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=top_pop,
        x="popularity",
        y="title",
        hue="title",
        palette="crest",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=4, fontsize=8.5)
    ax.set_title(f"En Popüler {top_n} Film")
    ax.set_xlabel("Popülerlik Skoru")
    ax.set_ylabel("Film Adı")
    ax.set_xlim(0, top_pop["popularity"].max() * 1.15)
    fig.tight_layout()
    return fig


def plot_vote_average_distribution(df: pd.DataFrame) -> plt.Figure:
    """6. Film puanlarının dağılımı grafiği."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(
        df["vote_average"],
        bins=30,
        kde=True,
        color="#2b5c8f",
        edgecolor="white",
        ax=ax,
    )
    mean_val = float(df["vote_average"].mean())
    median_val = float(df["vote_average"].median())
    ax.axvline(mean_val, color="#d95f02", linestyle="--", linewidth=2, label=f"Ortalama: {mean_val:.2f}")
    ax.axvline(median_val, color="#1b9e77", linestyle="-.", linewidth=2, label=f"Medyan: {median_val:.2f}")
    ax.set_title("Film Puanlarının (vote_average) Dağılımı")
    ax.set_xlabel("Ortalama Kullanıcı Puanı")
    ax.set_ylabel("Film Sayısı")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_budget_vs_revenue(df: pd.DataFrame) -> plt.Figure:
    """7. Bütçe ve gelir arasındaki ilişki grafiği."""
    valid = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    corr_val = float(valid["budget"].corr(valid["revenue"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        valid["budget"] / 1e6,
        valid["revenue"] / 1e6,
        c=valid["vote_average"],
        cmap="viridis",
        alpha=0.65,
        s=30,
        edgecolors="none",
    )
    # Başabaş (Gelir = Bütçe) doğrusu
    max_b = (valid["budget"] / 1e6).max()
    ax.plot([0, max_b], [0, max_b], color="red", linestyle="--", linewidth=1.5, label="Başabaş Çizgisi (Gelir = Bütçe)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Ortalama Puan")
    ax.set_title(f"Bütçe ve Gelir Arasındaki İlişki (Korelasyon r = {corr_val:.2f})")
    ax.set_xlabel("Bütçe (Milyon $)")
    ax.set_ylabel("Gelir (Milyon $)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_popularity_vs_vote_average(df: pd.DataFrame) -> plt.Figure:
    """8. Popülerlik ve oy ortalaması arasındaki ilişki grafiği."""
    valid = df[df["vote_count"] >= 10].copy()
    corr_val = float(valid["popularity"].corr(valid["vote_average"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        valid["vote_average"],
        valid["popularity"],
        c=np.log1p(valid["vote_count"]),
        cmap="plasma",
        alpha=0.6,
        s=28,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Log(Oy Sayısı)")
    ax.set_yscale("log")
    ax.set_title(f"Popülerlik ve Oy Ortalaması İlişkisi (r = {corr_val:.2f}, Log Ölçek)")
    ax.set_xlabel("Ortalama Kullanıcı Puanı (vote_average)")
    ax.set_ylabel("Popülerlik (Log Ölçek)")
    fig.tight_layout()
    return fig


def plot_top_directors(df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    """9. En çok film yöneten yönetmenler grafiği."""
    dir_df = (
        df[df["director"].astype(str).str.strip() != ""]
        .groupby("director")
        .size()
        .reset_index(name="Film Sayısı")
        .sort_values(by="Film Sayısı", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=dir_df,
        x="Film Sayısı",
        y="director",
        hue="director",
        palette="mako",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, padding=4, fontsize=9)
    ax.set_title(f"En Çok Film Yöneten {top_n} Yönetmen")
    ax.set_xlabel("Yönetilen Film Sayısı")
    ax.set_ylabel("Yönetmen")
    ax.set_xlim(0, dir_df["Film Sayısı"].max() * 1.12)
    fig.tight_layout()
    return fig


def plot_average_revenue_by_genre(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """10. Film türlerine göre ortalama gelir grafiği (Geliri bilinen filmler üzerinden)."""
    known = df[df["revenue"] > 0]
    records = []
    for _, row in known.iterrows():
        rev = float(row["revenue"])
        g_list = row["genre_list"]
        if isinstance(g_list, list):
            for g in g_list:
                if g:
                    records.append({"Tür": g, "Gelir": rev})

    g_rev_df = (
        pd.DataFrame(records)
        .groupby("Tür")
        .agg(Ortalama_Gelir=("Gelir", "mean"), Film_Sayisi=("Gelir", "count"))
        .reset_index()
    )
    g_rev_df = g_rev_df[g_rev_df["Film_Sayisi"] >= 10].sort_values(
        by="Ortalama_Gelir", ascending=False
    ).head(top_n)
    g_rev_df["Ortalama_Gelir_M"] = g_rev_df["Ortalama_Gelir"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=g_rev_df,
        x="Ortalama_Gelir_M",
        y="Tür",
        hue="Tür",
        palette="flare",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="$%.1fM", padding=4, fontsize=8.5)
    ax.set_title("Film Türlerine Göre Ortalama Gişe Geliri")
    ax.set_xlabel("Ortalama Gelir (Milyon $)")
    ax.set_ylabel("Film Türü")
    ax.set_xlim(0, g_rev_df["Ortalama_Gelir_M"].max() * 1.15)
    fig.tight_layout()
    return fig


# ==============================================================================
# APRIORI BİRLİKTELİK ANALİZİ GRAFİKLERİ
# ==============================================================================

def plot_apriori_frequent_items(items_df: pd.DataFrame, title: str, top_n: int = 12) -> plt.Figure:
    """Sık görülen tekil türler veya tür kombinasyonları bar grafiği."""
    sub = items_df.head(top_n).copy()
    sub["support_pct"] = sub["support"] * 100.0

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(
        data=sub,
        x="support_pct",
        y="items_str",
        hue="items_str",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%%%.1f", padding=4, fontsize=8.5)
    ax.set_title(title)
    ax.set_xlabel("Destek Oranı - Support (%)")
    ax.set_ylabel("Öğe Kümesi")
    ax.set_xlim(0, sub["support_pct"].max() * 1.15)
    fig.tight_layout()
    return fig


def plot_apriori_rules_scatter(rules_df: pd.DataFrame) -> plt.Figure:
    """Birliktelik kurallarının Support - Confidence - Lift dağılım grafiği."""
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        rules_df["support"],
        rules_df["confidence"],
        c=rules_df["lift"],
        s=np.clip(rules_df["lift"] * 45, 40, 300),
        cmap="turbo",
        alpha=0.78,
        edgecolors="black",
        linewidth=0.5,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Lift Değeri")

    # En yüksek lift'li ilk 6 kuralı grafik üzerinde etiketle
    top_annot = rules_df.sort_values(by="lift", ascending=False).head(6)
    for _, row in top_annot.iterrows():
        ax.annotate(
            row["rule_str"],
            (row["support"], row["confidence"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.35),
        )

    ax.set_title("Birliktelik Kuralları Dağılımı (Support vs Confidence vs Lift)")
    ax.set_xlabel("Support (Birlikte Görülme Oranı)")
    ax.set_ylabel("Confidence (Güven Olasılığı)")
    fig.tight_layout()
    return fig


def plot_top_rules_bar(top_rules_df: pd.DataFrame) -> plt.Figure:
    """En güçlü 10 birliktelik kuralının Lift ve Confidence karşılaştırması."""
    sub = top_rules_df.head(10).copy()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    sns.barplot(
        data=sub,
        x="lift",
        y="rule_str",
        hue="rule_str",
        palette="magma",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="Lift: %.2f", padding=4, fontsize=8.5)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="Rastlantısal Eşik (Lift = 1.0)")
    ax.set_title("En Güçlü 10 Birliktelik Kuralı (Lift Değerine Göre)")
    ax.set_xlabel("Lift Değeri")
    ax.set_ylabel("Birliktelik Kuralı (Öncül -> Ardıl)")
    ax.set_xlim(0, sub["lift"].max() * 1.18)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ==============================================================================
# K-MEANS KÜMELEME GRAFİKLERİ
# ==============================================================================

def plot_elbow_and_silhouette(eval_df: pd.DataFrame, selected_k: int) -> plt.Figure:
    """Elbow Method (WCSS / Inertia) ve Silhouette Score yan yana grafik."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Sol: Elbow Method
    ax1.plot(
        eval_df["k"],
        eval_df["inertia"],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
    )
    sel_row = eval_df[eval_df["k"] == selected_k]
    if not sel_row.empty:
        ax1.scatter(
            sel_row["k"],
            sel_row["inertia"],
            color="red",
            s=110,
            zorder=5,
            label=f"Seçilen k = {selected_k}",
        )
        ax1.legend()
    ax1.set_title("Elbow Method (Dirsek Yöntemi - WCSS)")
    ax1.set_xlabel("Küme Sayısı (k)")
    ax1.set_ylabel("Inertia (Küme İçi Kareler Toplamı)")
    ax1.set_xticks(eval_df["k"])

    # Sağ: Silhouette Score
    ax2.plot(
        eval_df["k"],
        eval_df["silhouette_score"],
        marker="s",
        linewidth=2.2,
        color="#2ca02c",
    )
    if not sel_row.empty:
        ax2.scatter(
            sel_row["k"],
            sel_row["silhouette_score"],
            color="red",
            s=110,
            zorder=5,
            label=f"Seçilen k = {selected_k} ({ float(sel_row['silhouette_score'].iloc[0]):.3f})",
        )
        ax2.legend()
    ax2.set_title("Küme Sayısına Göre Silhouette Score")
    ax2.set_xlabel("Küme Sayısı (k)")
    ax2.set_ylabel("Silhouette Skoru")
    ax2.set_xticks(eval_df["k"])

    fig.tight_layout()
    return fig


def plot_kmeans_clusters_pca(clustered_df: pd.DataFrame) -> plt.Figure:
    """PCA ile 2 boyuta indirgenmiş uzayda kümelerin dağılımı."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.scatterplot(
        data=clustered_df,
        x="pca_1",
        y="pca_2",
        hue="cluster_name",
        palette="tab10",
        alpha=0.72,
        s=40,
        ax=ax,
    )
    ax.set_title("K-Means Kümelerinin 2B PCA Uzayında Dağılımı")
    ax.set_xlabel("Birinci Temel Bileşen (PCA 1)")
    ax.set_ylabel("İkinci Temel Bileşen (PCA 2)")
    ax.legend(title="Kümeler", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_kmeans_cluster_comparison(summary_df: pd.DataFrame) -> plt.Figure:
    """Kümelerin ortalama bütçe, gelir, popülerlik ve puan karşılaştırması."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    short_labels = [f"Küme {cid}" for cid in summary_df["cluster_id"]]

    # 1. Küme Başına Film Sayısı
    sns.barplot(
        x=short_labels,
        y=summary_df["movie_count"],
        hue=short_labels,
        palette="Blues_d",
        legend=False,
        ax=axes[0, 0],
    )
    for c in axes[0, 0].containers:
        axes[0, 0].bar_label(c, padding=3, fontsize=8.5)
    axes[0, 0].set_title("Küme Başına Film Sayısı")
    axes[0, 0].set_ylabel("Film Sayısı")

    # 2. Ortalama Bütçe ve Gelir (Milyon $)
    x_idx = np.arange(len(short_labels))
    width = 0.36
    b_bars = axes[0, 1].bar(
        x_idx - width / 2,
        summary_df["avg_budget"] / 1e6,
        width,
        label="Ort. Bütçe ($M)",
        color="#e6550d",
    )
    r_bars = axes[0, 1].bar(
        x_idx + width / 2,
        summary_df["avg_revenue"] / 1e6,
        width,
        label="Ort. Gelir ($M)",
        color="#31a354",
    )
    axes[0, 1].set_xticks(x_idx)
    axes[0, 1].set_xticklabels(short_labels)
    axes[0, 1].set_title("Kümelere Göre Ortalama Bütçe ve Gelir (Milyon $)")
    axes[0, 1].set_ylabel("Milyon $")
    axes[0, 1].legend()

    # 3. Ortalama Popülerlik
    sns.barplot(
        x=short_labels,
        y=summary_df["avg_popularity"],
        hue=short_labels,
        palette="Purples_d",
        legend=False,
        ax=axes[1, 0],
    )
    for c in axes[1, 0].containers:
        axes[1, 0].bar_label(c, fmt="%.1f", padding=3, fontsize=8.5)
    axes[1, 0].set_title("Kümelere Göre Ortalama Popülerlik")
    axes[1, 0].set_ylabel("Ortalama Popülerlik")

    # 4. Ortalama Kullanıcı Puanı
    sns.barplot(
        x=short_labels,
        y=summary_df["avg_vote_average"],
        hue=short_labels,
        palette="Greens_d",
        legend=False,
        ax=axes[1, 1],
    )
    for c in axes[1, 1].containers:
        axes[1, 1].bar_label(c, fmt="%.2f", padding=3, fontsize=8.5)
    axes[1, 1].set_title("Kümelere Göre Ortalama Puan (vote_average)")
    axes[1, 1].set_ylabel("Ortalama Puan (0-10)")
    axes[1, 1].set_ylim(0, 10)

    fig.tight_layout()
    return fig


# ==============================================================================
# RANDOM FOREST SINIFLANDIRMA GRAFİKLERİ
# ==============================================================================

def plot_confusion_matrix_heatmap(cm_df: pd.DataFrame) -> plt.Figure:
    """Random Forest Confusion Matrix ısı haritası."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Karmaşıklık Matrisi (Confusion Matrix - Test Verisi)")
    ax.set_ylabel("Gerçek Gelir Sınıfı")
    ax.set_xlabel("Modelin Tahmin Ettiği Sınıf")
    fig.tight_layout()
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    """Random Forest özellik önem düzeyleri (Feature Importance) grafiği."""
    sub = fi_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=sub,
        x="importance_pct",
        y="feature_name",
        hue="feature_name",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%%%.2f", padding=4, fontsize=8.5)
    ax.set_title(f"Yüksek Gelir Tahmininde En Etkili {top_n} Özellik (Feature Importance)")
    ax.set_xlabel("Önem Düzeyi (%)")
    ax.set_ylabel("Özellik (Feature)")
    ax.set_xlim(0, sub["importance_pct"].max() * 1.15)
    fig.tight_layout()
    return fig


# ==============================================================================
# FİLM ÖNERİ SİSTEMİ GRAFİKLERİ
# ==============================================================================

def plot_recommendations_similarity(rec_df: pd.DataFrame, selected_title: str) -> plt.Figure:
    """Seçilen filme önerilen 10 filmin benzerlik yüzdesi bar grafiği."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(
        data=rec_df,
        x="Benzerlik Yüzdesi (%)",
        y="Film Adı",
        hue="Film Adı",
        palette="crest",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%%%.2f", padding=4, fontsize=8.5)
    ax.set_title(f"'{selected_title}' Filmine En Benzer 10 Film (TF-IDF & Cosine Similarity)")
    ax.set_xlabel("Kosinüs Benzerlik Yüzdesi (%)")
    ax.set_ylabel("Önerilen Filmler")
    ax.set_xlim(0, max(rec_df["Benzerlik Yüzdesi (%)"].max() * 1.18, 10))
    fig.tight_layout()
    return fig
