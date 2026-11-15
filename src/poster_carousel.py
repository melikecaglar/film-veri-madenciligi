"""
Otomatik Kayan ve Oklarla Manuel Kaydırılabilen Film Afiş Vitrini (Poster Carousel) Modülü.

Bu modül:
1. Veri setindeki en yüksek puanlı klasik filmleri (vote_count >= 1500) seçer.
2. Filmler için yüksek çözünürlüklü gerçek afiş görsellerini (yerleşik TMDB/katalog önbelleği +
   iTunes Movie Search API + yerel JSON önbellek + çevrimdışı SVG sinema afişi yedeği) çözümler.
3. Hem otomatik kayan, hem fareyle üzerine gelindiğinde duraklayan (pause on hover),
   hem sağ/sol (< ve >) ok butonlarıyla manuel kaydırılabilen, kart büyüme efektli (scale on hover),
   altın sarısı puan rozetli (⭐ 8.5) ve benzerlik rozetli HTML/CSS/JS carousel üretir.
"""

import base64
import html
import json
import os
import ssl
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit.components.v1 as components


# En yüksek puanlı klasik filmler ve sık önerilen filmler için doğrulanmış resmi TMDB afiş adresleri
KNOWN_MOVIE_POSTERS: Dict[int, str] = {
    278: "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",     # The Shawshank Redemption
    238: "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",     # The Godfather
    550: "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",     # Fight Club
    680: "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",     # Pulp Fiction
    424: "https://image.tmdb.org/t/p/w500/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",     # Schindler's List
    244786: "https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg",  # Whiplash
    129: "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",     # Spirited Away
    240: "https://image.tmdb.org/t/p/w500/hek3koDUyRQk7FIhPXsa6mT2Zc3.jpg",     # The Godfather: Part II
    155: "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",     # The Dark Knight
    13: "https://image.tmdb.org/t/p/w500/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg",      # Forrest Gump
    1891: "https://image.tmdb.org/t/p/w500/nNAeTmF4CtdSgMDplXTDPOpYzsX.jpg",    # The Empire Strikes Back
    497: "https://image.tmdb.org/t/p/w500/8VG8fDNiy50H4FedGwdSVUPoaJe.jpg",     # The Green Mile
    769: "https://image.tmdb.org/t/p/w500/aKuFiU82s5ISJpGZp7YkIr3kCUd.jpg",     # GoodFellas
    73: "https://image.tmdb.org/t/p/w500/euypWkaYFOLW3e5rLIcTAjWnhhT.jpg",      # American History X
    510: "https://image.tmdb.org/t/p/w500/3jcbDmRFiQ83drXNOvRDeKHxS0C.jpg",     # One Flew Over the Cuckoo's Nest
    122: "https://image.tmdb.org/t/p/w500/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg",     # LOTR: The Return of the King
    27205: "https://image.tmdb.org/t/p/w500/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",   # Inception
    120: "https://image.tmdb.org/t/p/w500/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg",     # LOTR: The Fellowship of the Ring
    157336: "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",  # Interstellar
    807: "https://image.tmdb.org/t/p/w500/6yoghtyTpznpBik8EngEmJskVUO.jpg",     # Se7en
    121: "https://image.tmdb.org/t/p/w500/5VTN0pR8gcqV3EPUHHfMGnJYN9L.jpg",     # LOTR: The Two Towers
    603: "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",     # The Matrix
    11: "https://image.tmdb.org/t/p/w500/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",      # Star Wars
    274: "https://image.tmdb.org/t/p/w500/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",     # The Silence of the Lambs
    857: "https://image.tmdb.org/t/p/w500/uqx37cS8cpHg8U35f9U5IBlrCV3.jpg",     # Saving Private Ryan
    598: "https://image.tmdb.org/t/p/w500/k7eYdWvhYQyRQoU2TB2A2Xu2TfD.jpg",     # City of God
    637: "https://image.tmdb.org/t/p/w500/74hLDKjD5aGYOotO6esUVaeISa2.jpg",     # Life Is Beautiful
    1124: "https://image.tmdb.org/t/p/w500/tRNlZbgNCNOpLpbPEz5L8G8A0JN.jpg",    # The Prestige
    77: "https://image.tmdb.org/t/p/w500/yuNs09hvpHVU1cBTCAk9zxsL2oW.jpg",      # Memento
    98: "https://image.tmdb.org/t/p/w500/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg",      # Gladiator
    1422: "https://image.tmdb.org/t/p/w500/nT97ifVT2J1yMQmeq20Qblg61T.jpg",     # The Departed
    105: "https://image.tmdb.org/t/p/w500/fNOH9f1aA7XRTzl1sAOx9iF553Q.jpg",     # Back to the Future
    68718: "https://image.tmdb.org/t/p/w500/7oWY8VDWW7thTzWh3OKYRkWUlD5.jpg",   # Django Unchained
    8587: "https://image.tmdb.org/t/p/w500/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg",    # The Lion King
    10681: "https://image.tmdb.org/t/p/w500/hbhFnRzzg6ZDmm8YAmxBnQpQIPh.jpg",   # WALL-E
    348: "https://image.tmdb.org/t/p/w500/vfrQk5IPloGg1v9Rzbh2Eg3VGyM.jpg",     # Alien
    679: "https://image.tmdb.org/t/p/w500/r1x5JGpyqZU8PYhbs4UcrO1Xb6x.jpg",     # Aliens
    8077: "https://image.tmdb.org/t/p/w500/xh5wI0UoW7DfS1IyLy3d2CgrCEP.jpg",    # Alien 3
    19995: "https://image.tmdb.org/t/p/w500/kyeqWdyUXW608qlYkRqosgbbJyK.jpg",   # Avatar
    272: "https://image.tmdb.org/t/p/w500/4MpN4kIEqUjW8OPtOQJXlTdHiJV.jpg",     # Batman Begins
    49026: "https://image.tmdb.org/t/p/w500/hr0L2aueqlP2BYUblTTjmtn0hw4.jpg",   # The Dark Knight Rises
    597: "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",     # Titanic
    54138: "https://image.tmdb.org/t/p/w500/7XrRkhMa9lQ71RszzSyVrJVvhyS.jpg",   # Star Trek Into Darkness
    268: "https://image.tmdb.org/t/p/w500/cij4dd21v2Rk2YtUQbV5kW69WB2.jpg",     # Batman (1989)
    364: "https://image.tmdb.org/t/p/w500/jKBjeXM7iBBV9UkUcOXx3m7FSHY.jpg",     # Batman Returns
    414: "https://image.tmdb.org/t/p/w500/mzzNBVwTiiY94xAXDMWJpNPW2US.jpg",     # Batman Forever
    415: "https://image.tmdb.org/t/p/w500/bsg0mrxUKTsJo04I7X9c6I0tY7P.jpg",     # Batman & Robin
    209112: "https://image.tmdb.org/t/p/w500/5UsK3grJvtQrtzEgqNlDljJW96w.jpg",  # Batman v Superman
    1924: "https://image.tmdb.org/t/p/w500/d7px1FQxW4tngdACVRsCSaZq0Xl.jpg",    # Superman
    23483: "https://image.tmdb.org/t/p/w500/8citjdBmjddZjlPsAHEyCKwGVvD.jpg",   # Kick-Ass
}


def _get_cache_file_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "poster_cache_v3.json")


def _load_poster_cache() -> Dict[str, str]:
    cache_path = _get_cache_file_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def _save_poster_cache(cache_data: Dict[str, str]) -> None:
    cache_path = _get_cache_file_path()
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_svg_fallback_poster(title: str, year: int, genres: str = "") -> str:
    """
    İnternet bağlantısı olmadığında veya afiş bulunamadığında kullanılacak
    sinematik koyu temalı SVG poster veri URI'sini (data:image/svg+xml;base64,...) üretir.
    """
    safe_title = html.escape(str(title)[:28])
    safe_genre = html.escape(str(genres).split(",")[0][:22] if genres else "Sinema")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#0f172a"/>
          <stop offset="50%" stop-color="#1e293b"/>
          <stop offset="100%" stop-color="#334155"/>
        </linearGradient>
      </defs>
      <rect width="300" height="450" fill="url(#bg)"/>
      <rect x="14" y="14" width="272" height="422" rx="12" fill="none" stroke="#fbbf24" stroke-opacity="0.35" stroke-width="2"/>
      <text x="150" y="145" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="54" fill="#fbbf24">🎬</text>
      <text x="150" y="220" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="bold" fill="#f8fafc">{safe_title}</text>
      <text x="150" y="255" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#94a3b8">({year})</text>
      <rect x="85" y="280" width="130" height="28" rx="14" fill="#fbbf24" fill-opacity="0.15"/>
      <text x="150" y="299" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#fbbf24">{safe_genre}</text>
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _normalize_search_title(title: str) -> str:
    """Özel karakterli film adlarını (Alien³ -> Alien 3 vb.) arama motorları için temizler."""
    s = str(title or "").strip()
    replacements = {
        "³": " 3",
        "²": " 2",
        "¹": " 1",
        "·": "-",
        "–": "-",
        "—": "-",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return " ".join(s.split())


def _fetch_imdb_poster(title: str, year: Optional[int] = None) -> Optional[str]:
    """
    Ücretsiz ve API anahtarı gerektirmeyen resmi IMDb Suggestion CDN API üzerinden
    filmin yüksek çözünürlüklü orijinal sinema afişini sorgular.
    """
    clean_title = _normalize_search_title(title)
    if not clean_title:
        return None
    ctx = ssl._create_unverified_context()
    try:
        alnum_chars = [ch.lower() for ch in clean_title if ch.isalnum()]
        first_char = alnum_chars[0] if alnum_chars else "x"
        query_str = f"{clean_title} {year}" if year else clean_title
        encoded = urllib.parse.quote(query_str.lower())
        url = f"https://v3.sg.media-imdb.com/suggestion/{first_char}/{encoded}.json"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=3.0, context=ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            items = payload.get("d", [])
            if items:
                best_img = ""
                best_score = -999999
                t_lower = clean_title.lower()
                orig_lower = str(title or "").strip().lower()
                for it in items:
                    img_obj = it.get("i")
                    if not isinstance(img_obj, dict):
                        continue
                    img_url = str(img_obj.get("imageUrl", ""))
                    if not img_url:
                        continue
                    cand_title = str(it.get("l", "")).strip().lower()
                    cand_norm = _normalize_search_title(cand_title).lower()
                    cand_year = it.get("y")
                    qid = str(it.get("qid", "")).lower()
                    score = 0
                    if qid in ("movie", "feature", "tvmovie"):
                        score += 60
                    elif qid in ("videogame", "tvseries", "tvminiseries"):
                        score -= 50
                    if cand_title in (t_lower, orig_lower) or cand_norm == t_lower:
                        score += 110
                    elif t_lower in cand_title or cand_title in t_lower:
                        score += 45
                    if year and isinstance(cand_year, int):
                        diff = abs(int(year) - cand_year)
                        if diff == 0:
                            score += 150
                        elif diff == 1:
                            score += 95
                        else:
                            score -= diff * 10
                    if score > best_score:
                        best_score = score
                        best_img = img_url
                if best_img:
                    return best_img
    except Exception:
        pass
    return None


def _fetch_itunes_poster(title: str, year: Optional[int] = None) -> Optional[str]:
    """
    Ücretsiz ve API anahtarı gerektirmeyen iTunes Search API üzerinden
    filmin yüksek çözünürlüklü afiş görselini sorgular.
    """
    clean_title = _normalize_search_title(title)
    if not clean_title:
        return None
    ctx = ssl._create_unverified_context()
    try:
        encoded = urllib.parse.quote(clean_title)
        url = f"https://itunes.apple.com/search?term={encoded}&media=movie&entity=movie&limit=10"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=3.0, context=ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            results = payload.get("results", [])
            if results:
                best_art = ""
                best_score = -999999
                target_lower = clean_title.lower()
                for r in results:
                    art = str(r.get("artworkUrl100", ""))
                    if not art:
                        continue
                    track_name = _normalize_search_title(str(r.get("trackName", ""))).lower()
                    rel_date = str(r.get("releaseDate", ""))
                    rel_year = int(rel_date[:4]) if len(rel_date) >= 4 and rel_date[:4].isdigit() else 0
                    score = 0
                    if track_name == target_lower:
                        score += 100
                    elif target_lower in track_name or track_name in target_lower:
                        score += 40
                    if year and rel_year:
                        diff = abs(int(year) - rel_year)
                        if diff == 0:
                            score += 120
                        elif diff == 1:
                            score += 80
                        else:
                            score -= diff * 5
                    if score > best_score:
                        best_score = score
                        best_art = art
                if best_art:
                    return best_art.replace("100x100bb", "600x900bb")
    except Exception:
        pass
    return None


def _fetch_wikipedia_poster(title: str, year: Optional[int] = None) -> Optional[str]:
    """
    Wikipedia PageImages API (pilicense=any) üzerinden filmin resmi sinema afişini sorgular.
    """
    clean_title = _normalize_search_title(title)
    if not clean_title:
        return None
    ctx = ssl._create_unverified_context()
    try:
        q = f"{clean_title} {year} film" if year else f"{clean_title} film"
        encoded_q = urllib.parse.quote(q)
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&format=json"
            f"&generator=search&gsrsearch={encoded_q}&gsrlimit=5"
            "&prop=pageimages&pilicense=any&piprop=thumbnail|original&pithumbsize=500"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FilmAnalizPlatformu/1.0 (https://github.com/melikecaglar/film-veri-madenciligi)"},
        )
        with urllib.request.urlopen(req, timeout=3.0, context=ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            pages = payload.get("query", {}).get("pages", {})
            if isinstance(pages, dict):
                sorted_pages = sorted(pages.values(), key=lambda p: p.get("index", 999))
                for p in sorted_pages:
                    thumb = p.get("thumbnail", {}).get("source")
                    orig = p.get("original", {}).get("source")
                    candidate = str(thumb or orig or "")
                    cand_low = candidate.lower()
                    if (
                        candidate
                        and not cand_low.endswith(".svg")
                        and not cand_low.endswith(".svg.png")
                        and "logo" not in cand_low
                    ):
                        return candidate
    except Exception:
        pass
    return None


def resolve_posters_for_movies(movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Verilen film sözlükleri listesi için poster URL'lerini hızlıca çözümler.
    """
    cache = _load_poster_cache()
    cache_updated = False
    to_fetch: List[int] = []

    for idx, item in enumerate(movies):
        mid = int(item.get("id", 0))
        cache_key = str(mid) if mid else item.get("title", "")
        if mid in KNOWN_MOVIE_POSTERS:
            item["poster_url"] = KNOWN_MOVIE_POSTERS[mid]
        elif cache_key in cache and cache[cache_key] and not str(cache[cache_key]).startswith("data:image/svg"):
            item["poster_url"] = cache[cache_key]
        else:
            to_fetch.append(idx)

    if to_fetch:
        def _worker(i: int) -> Optional[str]:
            m = movies[i]
            t_str = str(m.get("title", ""))
            y_int = int(m.get("release_year", 0))
            url = _fetch_imdb_poster(t_str, y_int)
            if not url:
                url = _fetch_wikipedia_poster(t_str, y_int)
            if not url:
                url = _fetch_itunes_poster(t_str, y_int)
            return url

        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched_urls = list(pool.map(_worker, to_fetch))

        for i, url in zip(to_fetch, fetched_urls):
            m = movies[i]
            mid = int(m.get("id", 0))
            cache_key = str(mid) if mid else m.get("title", "")
            if url:
                m["poster_url"] = url
                cache[cache_key] = url
                cache_updated = True
            else:
                m["poster_url"] = build_svg_fallback_poster(
                    str(m.get("title", "")),
                    int(m.get("release_year", 2000)),
                    str(m.get("genres", "")),
                )

    if cache_updated:
        _save_poster_cache(cache)

    return movies


def get_top_rated_movies_with_posters(
    df: pd.DataFrame,
    top_n: int = 15,
    min_votes: int = 1500,
) -> List[Dict[str, Any]]:
    """
    Veri setinden en az min_votes (varsayılan 1500) oy almış, vote_average değeri
    en yüksek ilk top_n filmi seçer ve afiş bilgilerini hazırlar.
    """
    filtered = df[df["vote_count"] >= min_votes].copy()
    if filtered.empty:
        filtered = df.copy()

    top_df = (
        filtered.sort_values(by=["vote_average", "vote_count"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )

    items: List[Dict[str, Any]] = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        items.append(
            {
                "rank": rank,
                "id": int(row["id"]),
                "title": str(row["title"]),
                "release_year": int(row["release_year"]),
                "vote_average": float(row["vote_average"]),
                "vote_count": int(row["vote_count"]),
                "genres": str(row.get("genres_str", "")),
                "director": str(row.get("director", "")),
                "similarity_pct": None,
            }
        )

    return resolve_posters_for_movies(items)


def get_recommended_movies_with_posters(rec_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Film Öneri Sistemi'nin döndürdüğü en benzer 10 film tablosunu
    otomatik kayan afiş vitrini formatına dönüştürür.
    """
    items: List[Dict[str, Any]] = []
    for _, row in rec_df.iterrows():
        items.append(
            {
                "rank": int(row["Sıra"]),
                "id": int(row["id"]),
                "title": str(row["Film Adı"]),
                "release_year": int(row["Yayın Yılı"]),
                "vote_average": float(row["Ortalama Puan"]),
                "vote_count": int(row.get("Oy Sayısı", 0)),
                "genres": str(row.get("Türler", "")),
                "director": str(row.get("Yönetmen", "")),
                "similarity_pct": float(row["Benzerlik Yüzdesi (%)"]),
            }
        )

    return resolve_posters_for_movies(items)


def generate_carousel_html(
    items: List[Dict[str, Any]],
    speed_seconds: int = 38,
) -> str:
    """
    Hem otomatik kayan, hem fareyle üzerine gelince duraklayan (pause on hover),
    hem de sağ ve sol (< ve >) ok butonlarıyla manuel kaydırılabilen,
    kart büyüme efektli (scale(1.09)) ve puan rozetli (⭐ 8.5) HTML/CSS/JS kodunu üretir.
    """
    if not items:
        return ""

    # Sonsuz kesintisiz döngü için kart listesini 2 kez yan yana diziyoruz
    doubled_items = items + items

    cards_html_parts: List[str] = []
    for item in doubled_items:
        mid_val = int(item.get("id", 0))
        title_raw = str(item.get("title", ""))
        title_esc = html.escape(title_raw)
        year_val = int(item.get("release_year", 2000))
        vote_val = float(item.get("vote_average", 0.0))
        genres_raw = str(item.get("genres", ""))
        short_genres = html.escape(", ".join([g.strip() for g in genres_raw.split(",")[:2] if g.strip()]))
        director_esc = html.escape(str(item.get("director", "")))
        poster_url = str(item.get("poster_url", ""))
        fallback_svg = build_svg_fallback_poster(title_raw, year_val, genres_raw)
        needs_fetch = "1" if (not poster_url or poster_url.startswith("data:image/svg")) else "0"

        sim_pct = item.get("similarity_pct")
        sim_badge_html = ""
        if sim_pct is not None:
            sim_badge_html = f'<div class="sim-badge">%{float(sim_pct):.1f} Benzer</div>'

        director_line = f'<div class="card-director">🎬 {director_esc}</div>' if director_esc else ""

        card_html = f"""
        <div class="poster-card" title="{title_esc} ({year_val}) - ⭐ {vote_val:.1f}">
            <div class="poster-img-wrapper">
                <img class="poster-img"
                     src="{poster_url}"
                     alt="{title_esc}"
                     data-movie-id="{mid_val}"
                     data-title="{title_esc}"
                     data-year="{year_val}"
                     data-needs-fetch="{needs_fetch}"
                     data-fallback="{fallback_svg}"
                     onerror="if(!this.dataset.triedBrowser){{this.dataset.triedBrowser='1';window.__resolvePoster && window.__resolvePoster(this);}}else{{this.onerror=null;this.src=this.dataset.fallback;}}" />
                <div class="rating-badge">⭐ {vote_val:.1f}</div>
                {sim_badge_html}
                <div class="poster-overlay">
                    <div class="card-title">{title_esc}</div>
                    <div class="card-meta">{year_val} • {short_genres}</div>
                    {director_line}
                </div>
            </div>
        </div>
        """
        cards_html_parts.append(card_html)

    all_cards_html = "\n".join(cards_html_parts)
    # Hız: speed_seconds küçükse daha hızlı, büyükse daha yavaş otomatik kayma
    px_per_frame = round(max(0.45, 34.0 / max(15, speed_seconds)), 2)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background: transparent;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
            user-select: none;
        }}
        .carousel-viewport {{
            position: relative;
            width: 100%;
            overflow: hidden;
            padding: 22px 0;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 16px;
            border: 1px solid rgba(251, 191, 36, 0.25);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
        }}
        /* Kenarlarda yumuşak karartma geçişi */
        .carousel-viewport::before,
        .carousel-viewport::after {{
            content: "";
            position: absolute;
            top: 0;
            width: 64px;
            height: 100%;
            z-index: 15;
            pointer-events: none;
        }}
        .carousel-viewport::before {{
            left: 0;
            background: linear-gradient(to right, rgba(15, 23, 42, 0.95), transparent);
        }}
        .carousel-viewport::after {{
            right: 0;
            background: linear-gradient(to left, rgba(15, 23, 42, 0.95), transparent);
        }}
        .carousel-scroller {{
            width: 100%;
            overflow-x: hidden;
            padding: 10px 0;
        }}
        .carousel-track {{
            display: flex;
            gap: 20px;
            width: max-content;
            padding: 0 20px;
        }}
        /* Uyumluluk için hover duraklama kuralı */
        .carousel-viewport:hover .carousel-track {{
            animation-play-state: paused;
        }}
        /* Sol ve Sağ Ok (< ve >) Butonları */
        .nav-arrow {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            width: 46px;
            height: 46px;
            border-radius: 50%;
            background: rgba(15, 23, 42, 0.88);
            border: 2px solid #fbbf24;
            color: #fbbf24;
            font-size: 22px;
            font-weight: 900;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 30;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(6px);
            transition: transform 0.22s ease, background 0.22s ease, box-shadow 0.22s ease, color 0.22s ease;
        }}
        .nav-arrow:hover {{
            background: #fbbf24;
            color: #0f172a;
            transform: translateY(-50%) scale(1.14);
            box-shadow: 0 0 18px rgba(251, 191, 36, 0.65);
        }}
        .nav-arrow:active {{
            transform: translateY(-50%) scale(0.96);
        }}
        .nav-arrow.left {{
            left: 14px;
        }}
        .nav-arrow.right {{
            right: 14px;
        }}
        .poster-card {{
            position: relative;
            width: 185px;
            height: 278px;
            flex-shrink: 0;
            border-radius: 12px;
            overflow: hidden;
            background: #1e293b;
            border: 2px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.45);
            cursor: pointer;
            transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1),
                        box-shadow 0.32s ease,
                        border-color 0.32s ease;
        }}
        /* Fareyle kartın üzerine gelince büyüme (zoom) ve altın parlama efekti */
        .poster-card:hover {{
            transform: scale(1.09) translateY(-6px);
            z-index: 25;
            border-color: #fbbf24;
            box-shadow: 0 14px 28px rgba(251, 191, 36, 0.35), 0 0 14px rgba(251, 191, 36, 0.25);
        }}
        .poster-img-wrapper {{
            position: relative;
            width: 100%;
            height: 100%;
        }}
        .poster-img-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: filter 0.3s ease;
        }}
        .poster-card:hover .poster-img-wrapper img {{
            filter: brightness(1.06);
        }}
        /* Puan Rozeti (Sağ Üst) */
        .rating-badge {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(15, 23, 42, 0.90);
            color: #fbbf24;
            border: 1.5px solid #fbbf24;
            border-radius: 8px;
            padding: 4px 9px;
            font-size: 12.5px;
            font-weight: 800;
            letter-spacing: 0.3px;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(4px);
            z-index: 5;
        }}
        /* Benzerlik Rozeti (Sol Üst - Öneri Sayfası için) */
        .sim-badge {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(16, 185, 129, 0.92);
            color: #ffffff;
            border-radius: 8px;
            padding: 4px 8px;
            font-size: 11px;
            font-weight: 700;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.45);
            z-index: 5;
        }}
        /* Alt Bilgi Katmanı */
        .poster-overlay {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 32px 10px 10px 10px;
            background: linear-gradient(to top, rgba(15, 23, 42, 0.98) 0%, rgba(15, 23, 42, 0.82) 60%, transparent 100%);
            color: #f8fafc;
            transition: padding-bottom 0.25s ease;
        }}
        .poster-card:hover .poster-overlay {{
            padding-bottom: 14px;
        }}
        .card-title {{
            font-size: 13.5px;
            font-weight: 700;
            line-height: 1.25;
            color: #ffffff;
            margin-bottom: 3px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .card-meta {{
            font-size: 11px;
            color: #cbd5e1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .card-director {{
            font-size: 10.5px;
            color: #fbbf24;
            margin-top: 3px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
    </head>
    <body>
        <div class="carousel-viewport" id="viewport">
            <button class="nav-arrow left" id="btnPrev" aria-label="Sola Kaydır" title="Sola Kaydır">&lt;</button>
            <div class="carousel-scroller" id="scroller">
                <div class="carousel-track" id="track">
                    {all_cards_html}
                </div>
            </div>
            <button class="nav-arrow right" id="btnNext" aria-label="Sağa Kaydır" title="Sağa Kaydır">&gt;</button>
        </div>

        <script>
            (function() {{
                const viewport = document.getElementById('viewport');
                const scroller = document.getElementById('scroller');
                const track = document.getElementById('track');
                const btnPrev = document.getElementById('btnPrev');
                const btnNext = document.getElementById('btnNext');

                let isHovered = false;
                let manualTargetRemaining = 0;
                let currentPos = 0;
                const autoSpeed = {px_per_frame};
                const stepPixels = 410; // Her ok tıklamasında ~2 afiş boyu kaydır

                function getHalfWidth() {{
                    return track.scrollWidth / 2;
                }}

                function normalizePosition() {{
                    const half = getHalfWidth();
                    if (half <= 0) return;
                    if (currentPos >= half) {{
                        currentPos -= half;
                    }} else if (currentPos < 0) {{
                        currentPos += half;
                    }}
                }}

                function animateLoop() {{
                    if (Math.abs(manualTargetRemaining) > 0.5) {{
                        // Ok butonuna tıklandığında yumuşak geçişli manuel kaydırma
                        const delta = manualTargetRemaining * 0.16;
                        currentPos += delta;
                        manualTargetRemaining -= delta;
                        normalizePosition();
                        scroller.scrollLeft = currentPos;
                    }} else {{
                        manualTargetRemaining = 0;
                        if (!isHovered) {{
                            currentPos += autoSpeed;
                            normalizePosition();
                            scroller.scrollLeft = currentPos;
                        }}
                    }}
                    requestAnimationFrame(animateLoop);
                }}

                viewport.addEventListener('mouseenter', function() {{
                    isHovered = true;
                }});
                viewport.addEventListener('mouseleave', function() {{
                    isHovered = false;
                }});

                btnPrev.addEventListener('click', function(e) {{
                    e.stopPropagation();
                    manualTargetRemaining -= stepPixels;
                }});

                btnNext.addEventListener('click', function(e) {{
                    e.stopPropagation();
                    manualTargetRemaining += stepPixels;
                }});

                // Tarayıcı içi canlı afiş çözümleyici (IMDb CDN + Wikipedia PageImages pilicense=any + iTunes API)
                const pendingIds = new Set();

                function normalizeTitle(t) {{
                    return (t || '')
                        .replace(/³/g, ' 3')
                        .replace(/²/g, ' 2')
                        .replace(/¹/g, ' 1')
                        .replace(/·/g, '-')
                        .replace(/\\s+/g, ' ')
                        .trim();
                }}

                function applyPosterToCards(movieId, title, resolvedUrl) {{
                    if (!resolvedUrl) return;
                    try {{
                        localStorage.setItem('fap_poster_v3_' + movieId, resolvedUrl);
                    }} catch (e) {{}}
                    const allImgs = document.querySelectorAll('img.poster-img');
                    allImgs.forEach(function(el) {{
                        if (el.dataset.movieId === String(movieId) || el.dataset.title === title) {{
                            el.dataset.needsFetch = '0';
                            el.src = resolvedUrl;
                        }}
                    }});
                }}

                async function fetchPosterFromImdb(title, year) {{
                    try {{
                        const clean = normalizeTitle(title);
                        if (!clean) return null;
                        const alnum = clean.toLowerCase().replace(/[^a-z0-9]/g, '');
                        const firstChar = alnum ? alnum[0] : 'x';
                        const q = year ? (clean + ' ' + year) : clean;
                        const url = 'https://v3.sg.media-imdb.com/suggestion/' + firstChar + '/' + encodeURIComponent(q.toLowerCase()) + '.json';
                        const resp = await fetch(url);
                        if (!resp.ok) return null;
                        const data = await resp.json();
                        const list = data.d || [];
                        let bestImg = '';
                        let bestScore = -999999;
                        const tLower = clean.toLowerCase();
                        for (const it of list) {{
                            if (!it.i || !it.i.imageUrl) continue;
                            const cTitle = normalizeTitle(it.l || '').toLowerCase();
                            const cYear = it.y || 0;
                            const qid = (it.qid || '').toLowerCase();
                            let score = 0;
                            if (qid === 'movie' || qid === 'feature' || qid === 'tvmovie') score += 60;
                            else if (qid === 'videogame' || qid === 'tvseries') score -= 50;
                            if (cTitle === tLower) score += 110;
                            else if (cTitle.includes(tLower) || tLower.includes(cTitle)) score += 45;
                            if (year && cYear) {{
                                const diff = Math.abs(year - cYear);
                                if (diff === 0) score += 150;
                                else if (diff === 1) score += 95;
                                else score -= diff * 10;
                            }}
                            if (score > bestScore) {{
                                bestScore = score;
                                bestImg = it.i.imageUrl;
                            }}
                        }}
                        return bestImg || null;
                    }} catch (e) {{}}
                    return null;
                }}

                async function fetchPosterFromWikipedia(title, year) {{
                    try {{
                        const clean = normalizeTitle(title);
                        const q = year ? (clean + ' ' + year + ' film') : (clean + ' film');
                        const url = 'https://en.wikipedia.org/w/api.php?action=query&origin=*&format=json&generator=search&gsrsearch='
                                    + encodeURIComponent(q)
                                    + '&gsrlimit=5&prop=pageimages&pilicense=any&piprop=thumbnail|original&pithumbsize=500';
                        const resp = await fetch(url);
                        if (!resp.ok) return null;
                        const data = await resp.json();
                        const pages = (data.query && data.query.pages) ? Object.values(data.query.pages) : [];
                        pages.sort(function(a, b) {{ return (a.index || 999) - (b.index || 999); }});
                        for (const p of pages) {{
                            const cand = (p.thumbnail && p.thumbnail.source) || (p.original && p.original.source) || '';
                            const cLow = cand.toLowerCase();
                            if (cand && !cLow.endsWith('.svg') && !cLow.endsWith('.svg.png') && !cLow.includes('logo')) {{
                                return cand;
                            }}
                        }}
                    }} catch (e) {{}}
                    return null;
                }}

                async function fetchPosterFromItunes(title, year) {{
                    try {{
                        const clean = normalizeTitle(title);
                        const url = 'https://itunes.apple.com/search?term=' + encodeURIComponent(clean) + '&media=movie&entity=movie&limit=10';
                        const resp = await fetch(url);
                        if (!resp.ok) return null;
                        const data = await resp.json();
                        const results = data.results || [];
                        if (!results.length) return null;
                        let bestArt = '';
                        let bestScore = -999999;
                        const tLower = clean.toLowerCase();
                        for (const r of results) {{
                            const art = r.artworkUrl100 || '';
                            if (!art) continue;
                            const rName = normalizeTitle(r.trackName || '').toLowerCase();
                            const rYear = r.releaseDate ? parseInt(r.releaseDate.slice(0, 4), 10) : 0;
                            let score = 0;
                            if (rName === tLower) score += 100;
                            else if (rName.includes(tLower) || tLower.includes(rName)) score += 40;
                            if (year && rYear) {{
                                const diff = Math.abs(year - rYear);
                                if (diff === 0) score += 120;
                                else if (diff === 1) score += 80;
                                else score -= diff * 5;
                            }}
                            if (score > bestScore) {{
                                bestScore = score;
                                bestArt = art;
                            }}
                        }}
                        if (bestArt) {{
                            return bestArt.replace('100x100bb', '600x900bb');
                        }}
                    }} catch (e) {{}}
                    return null;
                }}

                window.__resolvePoster = async function(imgEl) {{
                    const movieId = imgEl.dataset.movieId || '';
                    const title = imgEl.dataset.title || '';
                    const year = parseInt(imgEl.dataset.year || '0', 10);
                    const key = movieId + '_' + title;
                    if (pendingIds.has(key)) return;
                    pendingIds.add(key);

                    try {{
                        const cached = localStorage.getItem('fap_poster_v3_' + movieId);
                        if (cached && !cached.startsWith('data:image/svg')) {{
                            applyPosterToCards(movieId, title, cached);
                            return;
                        }}
                    }} catch (e) {{}}

                    let found = await fetchPosterFromImdb(title, year);
                    if (!found) {{
                        found = await fetchPosterFromWikipedia(title, year);
                    }}
                    if (!found) {{
                        found = await fetchPosterFromItunes(title, year);
                    }}
                    if (found) {{
                        applyPosterToCards(movieId, title, found);
                    }} else if (imgEl.dataset.fallback) {{
                        imgEl.src = imgEl.dataset.fallback;
                    }}
                }};

                // Sayfa yüklendiğinde afişi eksik olan tüm kartları otomatik tamamla
                document.querySelectorAll('img.poster-img[data-needs-fetch="1"]').forEach(function(imgEl) {{
                    window.__resolvePoster(imgEl);
                }});

                requestAnimationFrame(animateLoop);
            }})();
        </script>
    </body>
    </html>
    """
    return full_html


def render_poster_carousel(
    items: List[Dict[str, Any]],
    height: int = 335,
    speed_seconds: int = 38,
) -> None:
    """
    Hazırlanan film afiş listesini Streamlit sayfasında otomatik kayan ve oklarla kontrol edilebilen
    bileşen olarak görüntüler.
    """
    html_code = generate_carousel_html(items, speed_seconds=speed_seconds)
    if html_code:
        components.html(html_code, height=height, scrolling=False)
