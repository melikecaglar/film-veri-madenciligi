"""
Görselleştirme ve Veri Analizi Modülü (visualizations.py).

Tutarlı tasarım sistemi (renk paleti, tipografi ve kenar boşlukları) kullanarak
interaktif Plotly grafikleri üretir:
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
import sys
from typing import Any, Dict, List, Optional

_VENV_SP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".venv",
    "Lib",
    "site-packages",
)
if os.path.exists(_VENV_SP) and _VENV_SP not in sys.path:
    sys.path.insert(0, _VENV_SP)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# TASARIM SİSTEMİ RENK VE STİL SABİTLERİ (DESIGN TOKENS)
# ==============================================================================

COLOR_PRIMARY = "#2563EB"       # Ana mavi
COLOR_PRIMARY_SOFT = "#DBEAFE"  # Açık mavi dolgu
COLOR_SECONDARY = "#0D9488"     # Deniz mavisi / Teal
COLOR_ACCENT = "#D97706"        # Kehribar / Vurgu
COLOR_SUCCESS = "#16A34A"       # Yeşil (Gelir / Başarı)
COLOR_DANGER = "#DC2626"        # Kırmızı (Eşik / Uyarı)
COLOR_PURPLE = "#7C3AED"        # Mor (Popülerlik / Küme)
COLOR_TEXT = "#0F172A"          # Koyu arduvaz metin
COLOR_MUTED = "#64748B"         # İkincil metin
COLOR_GRID = "#F1F5F9"          # Kılavuz çizgisi
COLOR_BORDER = "#E2E8F0"        # Eksen çizgisi

CLUSTER_COLORS = [
    "#2563EB",
    "#0D9488",
    "#D97706",
    "#7C3AED",
    "#DC2626",
    "#059669",
    "#DB2777",
    "#4F46E5",
]

SEQUENTIAL_BLUES = [
    [0.0, "#93C5FD"],
    [0.2, "#60A5FA"],
    [0.4, "#3B82F6"],
    [0.6, "#2563EB"],
    [0.8, "#1D4ED8"],
    [1.0, "#1E40AF"],
]
SEQUENTIAL_TEALS = [
    [0.0, "#99F6E4"],
    [0.2, "#5EEAD4"],
    [0.4, "#2DD4BF"],
    [0.6, "#14B8A6"],
    [0.8, "#0D9488"],
    [1.0, "#0F766E"],
]


def _shorten_label(text: Any, max_len: int = 30) -> str:
    """Uzun Y ekseni etiketlerinin grafik alanını daraltmaması için okunabilir şekilde kısaltır."""
    s = str(text or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _make_unique_labels(labels: List[str]) -> List[str]:
    """Kısaltılmış etiketlerde çakışma olursa görünmez boşluk ekleyerek Plotly'nin birleştirmesini önler."""
    seen: Dict[str, int] = {}
    result: List[str] = []
    for lbl in labels:
        if lbl not in seen:
            seen[lbl] = 1
            result.append(lbl)
        else:
            cnt = seen[lbl]
            seen[lbl] = cnt + 1
            result.append(lbl + ("\u200b" * cnt))
    return result


def _empty_figure(message: str = "Gösterilecek veri bulunamadı.", height: int = 400) -> go.Figure:
    """Filtreleme sonucunda boş veri kaldığında hatasız boş durum (empty state) grafiği döndürür."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLOR_MUTED),
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def _apply_clean_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 440,
    show_legend: bool = False,
    left_margin: int = 80,
    right_margin: int = 60,
    top_margin: int = 68,
    bottom_margin: int = 58,
) -> go.Figure:
    """Tüm grafiklere ortak tipografi, otomatik eksen boşluğu (automargin) ve etkileşim stili uygular."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=COLOR_TEXT, family="sans-serif"),
            x=0.01,
            xref="container",
            xanchor="left",
            y=0.95,
            yref="container",
            yanchor="top",
        ),
        template="plotly_white",
        height=height,
        font=dict(family="sans-serif", size=12, color=COLOR_TEXT),
        showlegend=show_legend,
        margin=dict(
            l=left_margin,
            r=right_margin,
            t=top_margin,
            b=bottom_margin,
            autoexpand=True,
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hoverlabel=dict(
            bgcolor=COLOR_TEXT,
            font_size=12,
            font_family="sans-serif",
            font_color="#FFFFFF",
        ),
    )
    fig.update_xaxes(
        title_text=xaxis_title,
        automargin=True,
        title_standoff=10,
        showgrid=True,
        gridcolor=COLOR_GRID,
        linecolor=COLOR_BORDER,
        title_font=dict(size=12, color=COLOR_MUTED),
        tickfont=dict(size=11, color=COLOR_TEXT),
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=yaxis_title,
        automargin=True,
        title_standoff=10,
        ticksuffix="  ",
        showgrid=False,
        linecolor=COLOR_BORDER,
        title_font=dict(size=12, color=COLOR_MUTED),
        tickfont=dict(size=11.5, color=COLOR_TEXT),
        zeroline=False,
    )
    return fig


# ==============================================================================
# GENEL VERİ ANALİZİ GRAFİKLERİ (10 TEMEL GRAFİK)
# ==============================================================================

def plot_top_genres_count(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """1. En çok film üretilen türler grafiği."""
    if df.empty:
        return _empty_figure("Seçilen filtrede film bulunamadı.")

    genre_counts: Dict[str, int] = {}
    for g_list in df["genre_list"]:
        if isinstance(g_list, list):
            for g in g_list:
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

    if not genre_counts:
        return _empty_figure("Tür verisi bulunamadı.")

    total_movies = max(len(df), 1)
    g_df = (
        pd.DataFrame(list(genre_counts.items()), columns=["Tür", "Film Sayısı"])
        .sort_values(by="Film Sayısı", ascending=False)
        .head(top_n)
    )
    g_df["Oran (%)"] = (g_df["Film Sayısı"] / total_movies * 100).round(1)
    g_df = g_df.sort_values(by="Film Sayısı", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=g_df["Film Sayısı"],
            y=g_df["Tür"],
            orientation="h",
            marker=dict(
                color=g_df["Film Sayısı"],
                colorscale=SEQUENTIAL_BLUES,
                line=dict(width=0),
            ),
            text=[f"{cnt:,} (%{pct:.1f})" for cnt, pct in zip(g_df["Film Sayısı"], g_df["Oran (%)"])],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Film Sayısı: %{x:,}<br>Veri Seti Oranı: %%{customdata:.1f}<extra></extra>",
            customdata=g_df["Oran (%)"],
        )
    )
    _apply_clean_layout(
        fig,
        title="En Çok Film Üretilen Türler",
        xaxis_title="Film Sayısı",
        yaxis_title="",
        height=450,
        left_margin=135,
        right_margin=75,
    )
    fig.update_xaxes(range=[0, float(g_df["Film Sayısı"].max()) * 1.32])
    return fig


def plot_movies_by_year(df: pd.DataFrame) -> go.Figure:
    """2. Yıllara göre film sayısı grafiği."""
    sub = df[df["release_year"] >= 1950] if not df.empty else df
    if sub.empty:
        return _empty_figure("Seçilen yıl aralığında film bulunamadı.")

    year_df = (
        sub.groupby("release_year")
        .agg(
            Film_Sayisi=("id", "count"),
            Ort_Puan=("vote_average", "mean"),
        )
        .reset_index()
        .sort_values("release_year")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=year_df["release_year"],
            y=year_df["Film_Sayisi"],
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=2.5),
            marker=dict(size=5, color=COLOR_PRIMARY),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.12)",
            customdata=year_df["Ort_Puan"].round(2),
            hovertemplate="<b>Yıl: %{x}</b><br>Üretilen Film: %{y:,}<br>Ortalama Puan: %{customdata:.2f} / 10<extra></extra>",
            name="Film Sayısı",
        )
    )
    _apply_clean_layout(
        fig,
        title="Yıllara Göre Üretilen Film Sayısı (1950+)",
        xaxis_title="Yayın Yılı",
        yaxis_title="Film Sayısı",
        height=450,
        left_margin=75,
        right_margin=40,
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)
    return fig


def plot_top_revenue_movies(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """3. En yüksek gelir elde eden filmler grafiği."""
    if df.empty:
        return _empty_figure("Veri bulunamadı.")

    top_rev = (
        df.sort_values(by="revenue", ascending=False)
        .head(top_n)
        .copy()
        .sort_values(by="revenue", ascending=True)
    )
    top_rev["revenue_m"] = top_rev["revenue"] / 1e6
    top_rev["budget_m"] = top_rev["budget"] / 1e6
    top_rev["full_label"] = (
        top_rev["title"].astype(str) + " (" + top_rev["release_year"].astype(str) + ")"
    )
    top_rev["label_title"] = _make_unique_labels(
        [
            f"{_shorten_label(t, 26)} ({y})"
            for t, y in zip(top_rev["title"], top_rev["release_year"])
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=top_rev["revenue_m"],
            y=top_rev["label_title"],
            orientation="h",
            marker=dict(color=COLOR_SUCCESS),
            text=[f"${val:,.0f}M" for val in top_rev["revenue_m"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [top_rev["budget_m"], top_rev["vote_average"], top_rev["director"], top_rev["full_label"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "Gişe Geliri: $%{x:,.1f}M<br>"
                "Bütçe: $%{customdata[0]:,.1f}M<br>"
                "Puan: ⭐ %{customdata[1]:.1f}<br>"
                "Yönetmen: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title=f"En Yüksek Gişe Geliri Elde Eden {top_n} Film",
        xaxis_title="Gişe Geliri (Milyon $)",
        yaxis_title="",
        height=450,
        left_margin=195,
        right_margin=75,
    )
    max_val = float(top_rev["revenue_m"].max()) if not top_rev.empty else 100.0
    fig.update_xaxes(range=[0, max(max_val * 1.28, 10.0)])
    return fig


def plot_top_budget_movies(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """4. En yüksek bütçeli filmler grafiği."""
    if df.empty:
        return _empty_figure("Veri bulunamadı.")

    top_bud = (
        df.sort_values(by="budget", ascending=False)
        .head(top_n)
        .copy()
        .sort_values(by="budget", ascending=True)
    )
    top_bud["budget_m"] = top_bud["budget"] / 1e6
    top_bud["revenue_m"] = top_bud["revenue"] / 1e6
    top_bud["full_label"] = (
        top_bud["title"].astype(str) + " (" + top_bud["release_year"].astype(str) + ")"
    )
    top_bud["label_title"] = _make_unique_labels(
        [
            f"{_shorten_label(t, 26)} ({y})"
            for t, y in zip(top_bud["title"], top_bud["release_year"])
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=top_bud["budget_m"],
            y=top_bud["label_title"],
            orientation="h",
            marker=dict(color=COLOR_ACCENT),
            text=[f"${val:,.0f}M" for val in top_bud["budget_m"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [top_bud["revenue_m"], top_bud["vote_average"], top_bud["director"], top_bud["full_label"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "Bütçe: $%{x:,.1f}M<br>"
                "Gişe Geliri: $%{customdata[0]:,.1f}M<br>"
                "Puan: ⭐ %{customdata[1]:.1f}<br>"
                "Yönetmen: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title=f"En Yüksek Bütçeli {top_n} Film",
        xaxis_title="Yapım Bütçesi (Milyon $)",
        yaxis_title="",
        height=450,
        left_margin=195,
        right_margin=75,
    )
    max_val = float(top_bud["budget_m"].max()) if not top_bud.empty else 100.0
    fig.update_xaxes(range=[0, max(max_val * 1.28, 10.0)])
    return fig


def plot_top_popular_movies(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """5. En popüler 10 film grafiği."""
    if df.empty:
        return _empty_figure("Veri bulunamadı.")

    top_pop = (
        df.sort_values(by="popularity", ascending=False)
        .head(top_n)
        .copy()
        .sort_values(by="popularity", ascending=True)
    )
    top_pop["full_label"] = (
        top_pop["title"].astype(str) + " (" + top_pop["release_year"].astype(str) + ")"
    )
    top_pop["label_title"] = _make_unique_labels(
        [
            f"{_shorten_label(t, 26)} ({y})"
            for t, y in zip(top_pop["title"], top_pop["release_year"])
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=top_pop["popularity"],
            y=top_pop["label_title"],
            orientation="h",
            marker=dict(color=COLOR_PURPLE),
            text=[f"{val:.1f}" for val in top_pop["popularity"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [top_pop["vote_average"], top_pop["vote_count"], top_pop["genres_str"], top_pop["full_label"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "Popülerlik Skoru: %{x:.1f}<br>"
                "Ortalama Puan: ⭐ %{customdata[0]:.1f} (%{customdata[1]:,} oy)<br>"
                "Türler: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title=f"En Popüler {top_n} Film",
        xaxis_title="Popülerlik Endeksi",
        yaxis_title="",
        height=450,
        left_margin=195,
        right_margin=75,
    )
    max_val = float(top_pop["popularity"].max()) if not top_pop.empty else 100.0
    fig.update_xaxes(range=[0, max(max_val * 1.28, 10.0)])
    return fig


def plot_vote_average_distribution(df: pd.DataFrame) -> go.Figure:
    """6. Film puanlarının dağılımı grafiği."""
    if df.empty:
        return _empty_figure("Veri bulunamadı.")

    mean_val = float(df["vote_average"].mean())
    median_val = float(df["vote_average"].median())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["vote_average"],
            nbinsx=30,
            marker=dict(
                color=COLOR_PRIMARY,
                line=dict(color="#FFFFFF", width=1),
            ),
            opacity=0.88,
            name="Film Sayısı",
            hovertemplate="Puan Aralığı: %{x}<br>Film Sayısı: %{y:,}<extra></extra>",
        )
    )
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=COLOR_DANGER,
        line_width=2,
        annotation_text=f"Ortalama: {mean_val:.2f}",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_DANGER, size=11),
    )
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=COLOR_SECONDARY,
        line_width=2,
        annotation_text=f"Medyan: {median_val:.2f}",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_SECONDARY, size=11),
    )
    _apply_clean_layout(
        fig,
        title="Kullanıcı Puanlarının (vote_average) Dağılımı",
        xaxis_title="Ortalama Puan (0 - 10)",
        yaxis_title="Film Sayısı",
        height=450,
        left_margin=75,
        right_margin=40,
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)
    return fig


def plot_budget_vs_revenue(df: pd.DataFrame) -> go.Figure:
    """7. Bütçe ve gelir arasındaki ilişki grafiği."""
    valid = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    if valid.empty:
        return _empty_figure("Bütçe ve geliri bilinen film bulunamadı.")

    corr_val = float(valid["budget"].corr(valid["revenue"])) if len(valid) > 1 else 0.0
    valid["budget_m"] = valid["budget"] / 1e6
    valid["revenue_m"] = valid["revenue"] / 1e6

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=valid["budget_m"],
            y=valid["revenue_m"],
            mode="markers",
            marker=dict(
                size=7,
                color=valid["vote_average"],
                colorscale="Viridis",
                opacity=0.72,
                colorbar=dict(title="Puan", thickness=12),
            ),
            text=valid["title"],
            customdata=np.stack(
                [valid["release_year"], valid["vote_average"], valid["profit"] / 1e6], axis=-1
            ),
            hovertemplate=(
                "<b>%{text} (%{customdata[0]})</b><br>"
                "Bütçe: $%{x:,.1f}M<br>"
                "Gelir: $%{y:,.1f}M<br>"
                "Net Kâr: $%{customdata[2]:,.1f}M<br>"
                "Puan: ⭐ %{customdata[1]:.1f}<extra></extra>"
            ),
            name="Filmler",
        )
    )
    max_b = float(valid["budget_m"].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_b],
            y=[0, max_b],
            mode="lines",
            line=dict(color=COLOR_DANGER, width=1.8, dash="dash"),
            name="Başabaş Çizgisi (Gelir = Bütçe)",
            hoverinfo="skip",
        )
    )
    _apply_clean_layout(
        fig,
        title=f"Bütçe ve Gelir İlişkisi (Korelasyon r = {corr_val:.2f})",
        xaxis_title="Bütçe (Milyon $)",
        yaxis_title="Gişe Geliri (Milyon $)",
        height=460,
        show_legend=True,
        left_margin=80,
        right_margin=70,
        top_margin=82,
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    return fig


def plot_popularity_vs_vote_average(df: pd.DataFrame) -> go.Figure:
    """8. Popülerlik ve oy ortalaması arasındaki ilişki grafiği."""
    valid = df[(df["vote_count"] >= 10) & (df["popularity"] > 0)].copy()
    if valid.empty:
        return _empty_figure("Yeterli oy sayısına sahip film bulunamadı.")

    corr_val = float(valid["popularity"].corr(valid["vote_average"])) if len(valid) > 1 else 0.0

    fig = go.Figure(
        go.Scatter(
            x=valid["vote_average"],
            y=valid["popularity"],
            mode="markers",
            marker=dict(
                size=7,
                color=np.log1p(valid["vote_count"]),
                colorscale="Tealgrn",
                opacity=0.68,
                colorbar=dict(title="Log(Oy)", thickness=12),
            ),
            text=valid["title"],
            customdata=np.stack(
                [valid["release_year"], valid["vote_count"], valid["genres_str"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{text} (%{customdata[0]})</b><br>"
                "Ortalama Puan: ⭐ %{x:.1f}<br>"
                "Popülerlik: %{y:.1f}<br>"
                "Oy Sayısı: %{customdata[1]:,}<br>"
                "Türler: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title=f"Popülerlik ve Puan İlişkisi (r = {corr_val:.2f}, Log Ölçek)",
        xaxis_title="Ortalama Kullanıcı Puanı (vote_average)",
        yaxis_title="Popülerlik (Log Ölçek)",
        height=460,
        left_margin=80,
        right_margin=70,
    )
    fig.update_yaxes(type="log", showgrid=True, gridcolor=COLOR_GRID)
    return fig


def plot_top_directors(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """9. En çok film yöneten yönetmenler grafiği."""
    sub = df[df["director"].astype(str).str.strip() != ""] if not df.empty else df
    if sub.empty:
        return _empty_figure("Yönetmen verisi bulunamadı.")

    dir_df = (
        sub.groupby("director")
        .agg(
            Film_Sayisi=("id", "count"),
            Ort_Puan=("vote_average", "mean"),
            Toplam_Gelir_M=("revenue", lambda s: s.sum() / 1e6),
        )
        .reset_index()
        .sort_values(by="Film_Sayisi", ascending=False)
        .head(top_n)
        .sort_values(by="Film_Sayisi", ascending=True)
    )

    fig = go.Figure(
        go.Bar(
            x=dir_df["Film_Sayisi"],
            y=dir_df["director"],
            orientation="h",
            marker=dict(color=COLOR_SECONDARY),
            text=[f"{cnt} film (⭐ {rt:.1f})" for cnt, rt in zip(dir_df["Film_Sayisi"], dir_df["Ort_Puan"])],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([dir_df["Ort_Puan"], dir_df["Toplam_Gelir_M"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Yönetilen Film Sayısı: %{x}<br>"
                "Ortalama Puan: ⭐ %{customdata[0]:.2f}<br>"
                "Toplam Gişe Geliri: $%{customdata[1]:,.0f}M<extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title=f"En Çok Film Yöneten {top_n} Yönetmen",
        xaxis_title="Film Sayısı",
        yaxis_title="",
        height=450,
        left_margin=165,
        right_margin=85,
    )
    fig.update_xaxes(range=[0, float(dir_df["Film_Sayisi"].max()) * 1.34])
    return fig


def plot_average_revenue_by_genre(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """10. Film türlerine göre ortalama gelir grafiği (Geliri bilinen filmler üzerinden)."""
    known = df[df["revenue"] > 0] if not df.empty else df
    if known.empty:
        return _empty_figure("Geliri bilinen film bulunamadı.")

    records = []
    for _, row in known.iterrows():
        rev = float(row["revenue"])
        bud = float(row["budget"])
        g_list = row["genre_list"]
        if isinstance(g_list, list):
            for g in g_list:
                if g:
                    records.append({"Tür": g, "Gelir": rev, "Bütçe": bud})

    if not records:
        return _empty_figure("Tür gelir verisi bulunamadı.")

    g_rev_df = (
        pd.DataFrame(records)
        .groupby("Tür")
        .agg(
            Ortalama_Gelir=("Gelir", "mean"),
            Ortalama_Butce=("Bütçe", "mean"),
            Film_Sayisi=("Gelir", "count"),
        )
        .reset_index()
    )
    filtered = g_rev_df[g_rev_df["Film_Sayisi"] >= 10]
    if filtered.empty:
        filtered = g_rev_df
    g_rev_df = (
        filtered.sort_values(by="Ortalama_Gelir", ascending=False)
        .head(top_n)
        .sort_values(by="Ortalama_Gelir", ascending=True)
    )
    g_rev_df["Ortalama_Gelir_M"] = g_rev_df["Ortalama_Gelir"] / 1e6
    g_rev_df["Ortalama_Butce_M"] = g_rev_df["Ortalama_Butce"] / 1e6

    fig = go.Figure(
        go.Bar(
            x=g_rev_df["Ortalama_Gelir_M"],
            y=g_rev_df["Tür"],
            orientation="h",
            marker=dict(
                color=g_rev_df["Ortalama_Gelir_M"],
                colorscale=SEQUENTIAL_TEALS,
            ),
            text=[f"${val:,.1f}M" for val in g_rev_df["Ortalama_Gelir_M"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [g_rev_df["Ortalama_Butce_M"], g_rev_df["Film_Sayisi"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Ortalama Gelir: $%{x:,.1f}M<br>"
                "Ortalama Bütçe: $%{customdata[0]:,.1f}M<br>"
                "Analiz Edilen Film: %{customdata[1]:,}<extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title="Film Türlerine Göre Ortalama Gişe Geliri",
        xaxis_title="Ortalama Gelir (Milyon $)",
        yaxis_title="",
        height=450,
        left_margin=135,
        right_margin=75,
    )
    fig.update_xaxes(range=[0, float(g_rev_df["Ortalama_Gelir_M"].max()) * 1.30])
    return fig


# ==============================================================================
# APRIORI BİRLİKTELİK ANALİZİ GRAFİKLERİ
# ==============================================================================

def plot_apriori_frequent_items(items_df: pd.DataFrame, title: str, top_n: int = 12) -> go.Figure:
    """Sık görülen tekil türler veya tür kombinasyonları bar grafiği."""
    if items_df.empty:
        return _empty_figure("Sık öğe kümesi bulunamadı.")

    sub = items_df.head(top_n).copy()
    sub["support_pct"] = sub["support"] * 100.0
    if "film_count" not in sub.columns:
        sub["film_count"] = 0
    sub = sub.sort_values(by="support_pct", ascending=True)
    sub["display_label"] = _make_unique_labels(
        [_shorten_label(s, 30) for s in sub["items_str"]]
    )

    fig = go.Figure(
        go.Bar(
            x=sub["support_pct"],
            y=sub["display_label"],
            orientation="h",
            marker=dict(color=COLOR_PRIMARY),
            text=[f"%{pct:.1f} ({cnt:,})" for pct, cnt in zip(sub["support_pct"], sub["film_count"])],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([sub["film_count"], sub["items_str"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Support (Görülme Oranı): %%{x:.2f}<br>"
                "Film Sayısı: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title=title,
        xaxis_title="Destek Oranı - Support (%)",
        yaxis_title="",
        height=440,
        left_margin=185,
        right_margin=85,
    )
    fig.update_xaxes(range=[0, float(sub["support_pct"].max()) * 1.35])
    return fig


def plot_apriori_rules_scatter(rules_df: pd.DataFrame) -> go.Figure:
    """Birliktelik kurallarının Support - Confidence - Lift dağılım grafiği."""
    if rules_df.empty:
        return _empty_figure("Birliktelik kuralı bulunamadı.")

    fig = go.Figure(
        go.Scatter(
            x=rules_df["support"] * 100,
            y=rules_df["confidence"] * 100,
            mode="markers",
            marker=dict(
                size=np.clip(rules_df["lift"] * 4.2, 9, 28),
                color=rules_df["lift"],
                colorscale="Viridis",
                opacity=0.82,
                line=dict(width=1, color="#FFFFFF"),
                colorbar=dict(title="Lift", thickness=12),
            ),
            text=rules_df["rule_str"],
            customdata=np.stack(
                [rules_df["antecedents_str"], rules_df["consequents_str"], rules_df["lift"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Öncül (Eğer): %{customdata[0]}<br>"
                "Ardıl (İse): %{customdata[1]}<br>"
                "Support: %%{x:.2f}<br>"
                "Confidence: %%{y:.1f}<br>"
                "Lift: <b>%{customdata[2]:.2f}x</b><extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title="Birliktelik Kuralları Haritası (Support vs Confidence vs Lift)",
        xaxis_title="Support - Birlikte Görülme Oranı (%)",
        yaxis_title="Confidence - Güven Olasılığı (%)",
        height=450,
        left_margin=75,
        right_margin=70,
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)
    return fig


def plot_top_rules_bar(top_rules_df: pd.DataFrame) -> go.Figure:
    """En güçlü 10 birliktelik kuralının Lift ve Confidence karşılaştırması."""
    if top_rules_df.empty:
        return _empty_figure("Gösterilecek kural bulunamadı.")

    sub = top_rules_df.head(10).copy().sort_values(by="lift", ascending=True)
    sub["conf_pct"] = sub["confidence"] * 100.0
    sub["supp_pct"] = sub["support"] * 100.0
    sub["display_rule"] = _make_unique_labels(
        [_shorten_label(r, 32) for r in sub["rule_str"]]
    )

    fig = go.Figure(
        go.Bar(
            x=sub["lift"],
            y=sub["display_rule"],
            orientation="h",
            marker=dict(color=COLOR_SECONDARY),
            text=[f"{lft:.2f}x (Güven: %{cnf:.0f})" for lft, cnf in zip(sub["lift"], sub["conf_pct"])],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([sub["conf_pct"], sub["supp_pct"], sub["rule_str"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "Lift Katsayısı: %{x:.2f}x<br>"
                "Confidence (Güven): %%{customdata[0]:.1f}<br>"
                "Support (Destek): %%{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=1.0,
        line_dash="dash",
        line_color=COLOR_DANGER,
        line_width=1.5,
    )
    _apply_clean_layout(
        fig,
        title="En Güçlü 10 Birliktelik Kuralı (Lift Katsayısına Göre)",
        xaxis_title="Lift Katsayısı (1.0 = Rastlantısal Eşik)",
        yaxis_title="",
        height=450,
        left_margin=215,
        right_margin=95,
    )
    fig.update_xaxes(range=[0, float(sub["lift"].max()) * 1.42])
    return fig


# ==============================================================================
# K-MEANS KÜMELEME GRAFİKLERİ
# ==============================================================================

def plot_elbow_and_silhouette(eval_df: pd.DataFrame, selected_k: int) -> go.Figure:
    """Elbow Method (WCSS / Inertia) ve Silhouette Score yan yana grafik."""
    if eval_df.empty:
        return _empty_figure("Değerlendirme verisi bulunamadı.")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Elbow Yöntemi (Küme İçi Hata Kareler Toplamı - WCSS)",
            "Silhouette Skoru (Küme Ayrışma Kalitesi)",
        ),
        horizontal_spacing=0.16,
    )

    sel_row = eval_df[eval_df["k"] == selected_k]

    # Sol: Elbow Method
    fig.add_trace(
        go.Scatter(
            x=eval_df["k"],
            y=eval_df["inertia"],
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=2.5),
            marker=dict(size=8, color=COLOR_PRIMARY),
            name="Inertia (WCSS)",
            hovertemplate="Küme Sayısı (k): %{x}<br>Inertia: %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if not sel_row.empty:
        fig.add_trace(
            go.Scatter(
                x=sel_row["k"],
                y=sel_row["inertia"],
                mode="markers",
                marker=dict(size=14, color=COLOR_DANGER, line=dict(width=2, color="#FFFFFF")),
                name=f"Seçilen k = {selected_k}",
                hovertemplate=f"<b>Seçilen k = {selected_k}</b><br>Inertia: %{{y:,.0f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Sağ: Silhouette Score
    fig.add_trace(
        go.Scatter(
            x=eval_df["k"],
            y=eval_df["silhouette_score"],
            mode="lines+markers",
            line=dict(color=COLOR_SECONDARY, width=2.5),
            marker=dict(size=8, symbol="square", color=COLOR_SECONDARY),
            name="Silhouette Skoru",
            hovertemplate="Küme Sayısı (k): %{x}<br>Silhouette: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    if not sel_row.empty:
        fig.add_trace(
            go.Scatter(
                x=sel_row["k"],
                y=sel_row["silhouette_score"],
                mode="markers",
                marker=dict(size=14, color=COLOR_DANGER, line=dict(width=2, color="#FFFFFF")),
                showlegend=False,
                hovertemplate=f"<b>Seçilen k = {selected_k}</b><br>Silhouette: %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    _apply_clean_layout(
        fig,
        title="Küme Sayısı (k = 2..8) Kalite Değerlendirmesi",
        height=420,
        show_legend=False,
        left_margin=80,
        right_margin=45,
        top_margin=82,
    )
    fig.update_xaxes(title_text="Küme Sayısı (k)", dtick=1, automargin=True, row=1, col=1)
    fig.update_xaxes(title_text="Küme Sayısı (k)", dtick=1, automargin=True, row=1, col=2)
    fig.update_yaxes(title_text="Inertia (WCSS)", automargin=True, showgrid=True, gridcolor=COLOR_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Skoru", automargin=True, showgrid=True, gridcolor=COLOR_GRID, row=1, col=2)
    return fig


def plot_kmeans_clusters_pca(clustered_df: pd.DataFrame) -> go.Figure:
    """PCA ile 2 boyuta indirgenmiş uzayda kümelerin interaktif dağılımı."""
    if clustered_df.empty:
        return _empty_figure("Kümeleme verisi bulunamadı.")

    fig = go.Figure()
    clusters = sorted(clustered_df["cluster_id"].unique())
    for idx, cid in enumerate(clusters):
        sub = clustered_df[clustered_df["cluster_id"] == cid]
        c_name = str(sub["cluster_name"].iloc[0]) if not sub.empty else f"Küme {cid}"
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=sub["pca_1"],
                y=sub["pca_2"],
                mode="markers",
                name=c_name,
                marker=dict(size=7.5, color=color, opacity=0.76),
                text=sub["title"],
                customdata=np.stack(
                    [
                        sub["release_year"],
                        sub["budget"] / 1e6,
                        sub["revenue"] / 1e6,
                        sub["vote_average"],
                        sub["genres_str"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{text} (%{customdata[0]})</b><br>"
                    f"Küme: {c_name}<br>"
                    "Bütçe: $%{customdata[1]:,.1f}M | Gelir: $%{customdata[2]:,.1f}M<br>"
                    "Puan: ⭐ %{customdata[3]:.1f}<br>"
                    "Türler: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    _apply_clean_layout(
        fig,
        title="K-Means Kümelerinin 2B PCA Uzayında Dağılımı (Film Detayları İçin Noktaların Üzerine Gelin)",
        xaxis_title="Birinci Temel Bileşen (PCA 1 - Finansal Ölçek & Popülerlik)",
        yaxis_title="İkinci Temel Bileşen (PCA 2 - Tür & Puan Etkisi)",
        height=500,
        show_legend=True,
        left_margin=80,
        right_margin=45,
        top_margin=95,
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    return fig


def plot_kmeans_cluster_comparison(summary_df: pd.DataFrame) -> go.Figure:
    """Kümelerin ortalama bütçe, gelir, popülerlik ve puan karşılaştırması."""
    if summary_df.empty:
        return _empty_figure("Küme özet verisi bulunamadı.")

    short_labels = [f"Küme {cid}" for cid in summary_df["cluster_id"]]
    full_names = summary_df["cluster_name"].tolist()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Küme Başına Film Sayısı",
            "Ortalama Bütçe ve Gişe Geliri (Milyon $)",
            "Ortalama Popülerlik Skoru",
            "Ortalama Kullanıcı Puanı (0-10)",
        ),
        vertical_spacing=0.22,
        horizontal_spacing=0.14,
    )

    # 1. Küme Başına Film Sayısı
    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=summary_df["movie_count"],
            marker_color=COLOR_PRIMARY,
            text=[f"{cnt:,} (%{pct:.1f})" for cnt, pct in zip(summary_df["movie_count"], summary_df["percentage"])],
            textposition="outside",
            cliponaxis=False,
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Film Sayısı: %{y:,}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Ortalama Bütçe ve Gelir
    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=summary_df["avg_budget"] / 1e6,
            name="Ort. Bütçe ($M)",
            marker_color=COLOR_ACCENT,
            text=[f"${v:.0f}M" for v in summary_df["avg_budget"] / 1e6],
            textposition="outside",
            cliponaxis=False,
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Ort. Bütçe: $%{y:,.1f}M<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=summary_df["avg_revenue"] / 1e6,
            name="Ort. Gelir ($M)",
            marker_color=COLOR_SUCCESS,
            text=[f"${v:.0f}M" for v in summary_df["avg_revenue"] / 1e6],
            textposition="outside",
            cliponaxis=False,
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Ort. Gelir: $%{y:,.1f}M<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3. Ortalama Popülerlik
    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=summary_df["avg_popularity"],
            marker_color=COLOR_PURPLE,
            text=[f"{v:.1f}" for v in summary_df["avg_popularity"]],
            textposition="outside",
            cliponaxis=False,
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Ort. Popülerlik: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. Ortalama Puan
    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=summary_df["avg_vote_average"],
            marker_color=COLOR_SECONDARY,
            text=[f"⭐ {v:.2f}" for v in summary_df["avg_vote_average"]],
            textposition="outside",
            cliponaxis=False,
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Ort. Puan: %{y:.2f} / 10<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    _apply_clean_layout(
        fig,
        title="Kümelerin Temel Metrikler Üzerinden Karşılaştırması",
        height=600,
        show_legend=True,
        left_margin=75,
        right_margin=45,
        top_margin=92,
    )
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    fig.update_yaxes(range=[0, float(summary_df["movie_count"].max()) * 1.30], automargin=True, showgrid=True, gridcolor=COLOR_GRID, row=1, col=1)
    max_fin = max(float((summary_df["avg_revenue"] / 1e6).max()), float((summary_df["avg_budget"] / 1e6).max()))
    fig.update_yaxes(range=[0, max_fin * 1.30], automargin=True, showgrid=True, gridcolor=COLOR_GRID, row=1, col=2)
    fig.update_yaxes(range=[0, float(summary_df["avg_popularity"].max()) * 1.30], automargin=True, showgrid=True, gridcolor=COLOR_GRID, row=2, col=1)
    fig.update_yaxes(range=[0, 10.8], automargin=True, showgrid=True, gridcolor=COLOR_GRID, row=2, col=2)
    return fig


# ==============================================================================
# RANDOM FOREST SINIFLANDIRMA GRAFİKLERİ
# ==============================================================================

def plot_confusion_matrix_heatmap(cm_df: pd.DataFrame) -> go.Figure:
    """Random Forest Confusion Matrix ısı haritası."""
    if cm_df.empty:
        return _empty_figure("Karmaşıklık matrisi verisi bulunamadı.")

    z_vals = cm_df.values
    total = max(int(z_vals.sum()), 1)
    text_labels = [
        [f"<b>{val}</b><br>(%{val / total * 100:.1f})" for val in row]
        for row in z_vals
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=list(cm_df.columns),
            y=list(cm_df.index),
            text=text_labels,
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorscale="Blues",
            showscale=False,
            hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Film Sayısı: %{z}<extra></extra>",
        )
    )
    _apply_clean_layout(
        fig,
        title="Karmaşıklık Matrisi (Confusion Matrix - Test Seti)",
        xaxis_title="Modelin Tahmin Ettiği Sınıf",
        yaxis_title="Gerçek Gelir Sınıfı",
        height=440,
        left_margin=160,
        right_margin=40,
        bottom_margin=65,
    )
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Random Forest özellik önem düzeyleri (Feature Importance) grafiği."""
    if fi_df.empty:
        return _empty_figure("Özellik önem verisi bulunamadı.")

    sub = fi_df.head(top_n).copy().sort_values(by="importance_pct", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=sub["importance_pct"],
            y=sub["feature_name"],
            orientation="h",
            marker=dict(
                color=sub["importance_pct"],
                colorscale=SEQUENTIAL_BLUES,
            ),
            text=[f"%{val:.2f}" for val in sub["importance_pct"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Model Karar Ağırlığı: %%{x:.2f}<extra></extra>",
        )
    )
    _apply_clean_layout(
        fig,
        title=f"Gişe Geliri Tahmininde En Etkili {top_n} Özellik",
        xaxis_title="Göreli Önem Düzeyi (%)",
        yaxis_title="",
        height=440,
        left_margin=195,
        right_margin=75,
    )
    fig.update_xaxes(range=[0, float(sub["importance_pct"].max()) * 1.32])
    return fig


# ==============================================================================
# FİLM ÖNERİ SİSTEMİ GRAFİKLERİ
# ==============================================================================

def plot_recommendations_similarity(rec_df: pd.DataFrame, selected_title: str) -> go.Figure:
    """Seçilen filme önerilen 10 filmin benzerlik yüzdesi bar grafiği."""
    if rec_df.empty:
        return _empty_figure("Öneri verisi bulunamadı.")

    sub = rec_df.copy().sort_values(by="Benzerlik Yüzdesi (%)", ascending=True)
    sub["full_label"] = (
        sub["Film Adı"].astype(str) + " (" + sub["Yayın Yılı"].astype(str) + ")"
    )
    sub["label_title"] = _make_unique_labels(
        [
            f"{_shorten_label(t, 28)} ({y})"
            for t, y in zip(sub["Film Adı"], sub["Yayın Yılı"])
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=sub["Benzerlik Yüzdesi (%)"],
            y=sub["label_title"],
            orientation="h",
            marker=dict(
                color=sub["Benzerlik Yüzdesi (%)"],
                colorscale=SEQUENTIAL_TEALS,
            ),
            text=[
                f"%{sim:.1f}  |  ⭐ {rt:.1f}"
                for sim, rt in zip(sub["Benzerlik Yüzdesi (%)"], sub["Ortalama Puan"])
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [sub["Türler"], sub["Yönetmen"], sub["Ortak Unsurlar"], sub["Ortalama Puan"], sub["full_label"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[4]}</b><br>"
                "Kosinüs Benzerliği: %%{x:.2f}<br>"
                "Puan: ⭐ %{customdata[3]:.1f}<br>"
                "Yönetmen: %{customdata[1]}<br>"
                "Türler: %{customdata[0]}<br>"
                "Ortak Unsurlar: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    _apply_clean_layout(
        fig,
        title=f"'{selected_title}' Filmine En Benzer 10 Film (TF-IDF & Kosinüs Benzerliği)",
        xaxis_title="İçerik Benzerlik Oranı (%)",
        yaxis_title="",
        height=450,
        left_margin=205,
        right_margin=95,
    )
    max_sim = float(sub["Benzerlik Yüzdesi (%)"].max()) if not sub.empty else 10.0
    fig.update_xaxes(range=[0, max(max_sim * 1.38, 10.0)])
    return fig
