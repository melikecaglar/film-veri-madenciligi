"""
Film Analiz Platformu Özel Vektörel İkon (SVG) ve Sidebar Tasarım Modülü.
Kullanıcının paylaştığı referans görseldeki tüm ikonları birebir SVG olarak üretir.
"""

import base64
import os


def svg_to_data_uri(svg_str: str) -> str:
    """SVG metnini data:image/svg+xml;base64 URI formatına dönüştürür."""
    b64 = base64.b64encode(svg_str.strip().encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


# 1. Ana Marka Logosu: Renkli Play Butonlu Koyu Klapet / Kaset Logosu
BRAND_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <defs>
    <linearGradient id="rainbowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4ade80"/>
      <stop offset="30%" stop-color="#facc15"/>
      <stop offset="65%" stop-color="#fb7185"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
  </defs>
  <!-- Alt Gövde -->
  <rect x="6" y="22" width="52" height="36" rx="8" fill="#2d3748"/>
  <rect x="10" y="28" width="44" height="26" rx="5" fill="url(#rainbowGrad)"/>
  <!-- Play Üçgeni -->
  <polygon points="27,34 27,48 40,41" fill="#ffffff"/>
  <!-- Üst Klapet (Hafif Açık) -->
  <g transform="rotate(-11 8 22)">
    <rect x="6" y="10" width="52" height="12" rx="4" fill="#1e293b"/>
    <polygon points="14,10 22,10 17,22 9,22" fill="#f8fafc"/>
    <polygon points="28,10 36,10 31,22 23,22" fill="#f8fafc"/>
    <polygon points="42,10 50,10 45,22 37,22" fill="#f8fafc"/>
  </g>
</svg>
"""

# 2. Menü Başlığı İkonu: Renkli Raptiye (Pushpin)
PIN_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="36" height="36">
  <defs>
    <linearGradient id="pinGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#facc15"/>
      <stop offset="50%" stop-color="#4ade80"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
  </defs>
  <line x1="6" y1="30" x2="16" y2="20" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>
  <path d="M14,10 L26,22 L22,26 L10,14 Z" fill="#ec4899"/>
  <rect x="18" y="4" width="12" height="8" rx="3" transform="rotate(45 24 8)" fill="url(#pinGrad)"/>
</svg>
"""

# 3. Menü 1: Ana Sayfa (Mavi Ev İkonu)
HOME_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="36" height="36">
  <path d="M6,17 L18,6 L30,17 L27,17 L27,30 L9,30 L9,17 Z" fill="#3b82f6"/>
  <path d="M3,18 L18,4 L33,18 L30,20 L18,9 L6,20 Z" fill="#1d4ed8"/>
  <rect x="15" y="21" width="6" height="9" rx="1.5" fill="#1e3a8a"/>
</svg>
"""

# 4. Menü 2: Veri Seti İnceleme (Mavi Belge + Büyüteç)
DATASET_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="36" height="36">
  <rect x="6" y="4" width="20" height="26" rx="3" fill="#2563eb"/>
  <polygon points="20,4 26,10 20,10" fill="#93c5fd"/>
  <rect x="10" y="13" width="10" height="2.2" rx="1" fill="#e0f2fe"/>
  <rect x="10" y="18" width="8" height="2.2" rx="1" fill="#e0f2fe"/>
  <rect x="10" y="23" width="6" height="2.2" rx="1" fill="#e0f2fe"/>
  <circle cx="24" cy="24" r="5.5" fill="#1e40af" stroke="#ffffff" stroke-width="2"/>
  <line x1="28" y1="28" x2="32" y2="32" stroke="#4f46e5" stroke-width="3" stroke-linecap="round"/>
</svg>
"""

# 5. Menü 3: Apriori Birliktelik Analizi (5 Düğümlü Ağ Grafiği)
APRIORI_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="36" height="36">
  <line x1="18" y1="18" x2="8" y2="8" stroke="#64748b" stroke-width="2.2"/>
  <line x1="18" y1="18" x2="28" y2="8" stroke="#64748b" stroke-width="2.2"/>
  <line x1="18" y1="18" x2="8" y2="28" stroke="#64748b" stroke-width="2.2"/>
  <line x1="18" y1="18" x2="28" y2="28" stroke="#64748b" stroke-width="2.2"/>
  <circle cx="18" cy="18" r="5.5" fill="#f87171"/>
  <circle cx="8" cy="8" r="3.8" fill="#10b981"/>
  <circle cx="28" cy="8" r="3.8" fill="#0ea5e9"/>
  <circle cx="8" cy="28" r="3.8" fill="#2563eb"/>
  <circle cx="28" cy="28" r="3.8" fill="#10b981"/>
</svg>
"""

# 6. Menü 4: K-Means Kümeleme (Renkli Nokta Kümeleri)
KMEANS_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="36" height="36">
  <!-- Mercan Kümesi -->
  <circle cx="24" cy="10" r="3.2" fill="#f87171"/>
  <circle cx="29" cy="15" r="2.8" fill="#fb7185"/>
  <circle cx="22" cy="16" r="2.5" fill="#f43f5e"/>
  <!-- Mavi Kümesi -->
  <circle cx="10" cy="25" r="3.4" fill="#0284c7"/>
  <circle cx="16" cy="28" r="2.4" fill="#38bdf8"/>
  <circle cx="15" cy="21" r="2.2" fill="#0ea5e9"/>
  <!-- Mor Kümesi -->
  <circle cx="10" cy="11" r="2.2" fill="#6366f1"/>
  <circle cx="15" cy="8" r="1.8" fill="#8b5cf6"/>
  <circle cx="26" cy="25" r="2.2" fill="#f43f5e"/>
</svg>
"""

# 7. Menü 5: Random Forest Tahmin (Mercan Karar Ağacı Hiyerarşisi)
RF_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="36" height="36">
  <path d="M18,11 L18,18 M9,18 L27,18 M9,18 L9,24 M18,18 L18,24 M27,18 L27,24"
        stroke="#ef4444" stroke-width="2.3" stroke-linecap="round" fill="none"/>
  <circle cx="18" cy="9" r="4.2" fill="#f87171"/>
  <circle cx="9" cy="27" r="3.6" fill="#ef4444"/>
  <circle cx="18" cy="27" r="3.6" fill="#ef4444"/>
  <circle cx="27" cy="27" r="3.6" fill="#ef4444"/>
</svg>
"""

# 8. Menü 6: Film Öneri Sistemi (Mor Film Şeridi + Kırmızı Kalp)
REC_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="36" height="36">
  <rect x="5" y="7" width="20" height="23" rx="3" fill="#7c3aed"/>
  <rect x="7" y="10" width="2.5" height="2.5" fill="#ddd6fe"/>
  <rect x="7" y="16" width="2.5" height="2.5" fill="#ddd6fe"/>
  <rect x="7" y="22" width="2.5" height="2.5" fill="#ddd6fe"/>
  <rect x="20.5" y="16" width="2.5" height="2.5" fill="#ddd6fe"/>
  <rect x="20.5" y="22" width="2.5" height="2.5" fill="#ddd6fe"/>
  <!-- Kalp -->
  <path d="M24,19 C24,19 16,14 16,9.5 C16,7 18,5.5 20.2,5.5 C21.8,5.5 23.2,6.4 24,7.6 C24.8,6.4 26.2,5.5 27.8,5.5 C30,5.5 32,7 32,9.5 C32,14 24,19 24,19 Z"
        fill="#f43f5e" stroke="#ffffff" stroke-width="1.2"/>
</svg>
"""

# 9. Veri Seti Özeti Başlık İkonu (Renkli Çubuk Grafik)
SUMMARY_CHART_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="22" height="22">
  <rect x="4" y="14" width="5" height="14" rx="1.5" fill="#3b82f6"/>
  <rect x="11" y="18" width="5" height="10" rx="1.5" fill="#fb7185"/>
  <rect x="18" y="8" width="5" height="20" rx="1.5" fill="#10b981"/>
  <rect x="25" y="12" width="4" height="16" rx="1.5" fill="#facc15"/>
</svg>
"""

# 10. Toplam Film Kart İkonu (Mercan Klapet)
CARD_MOVIE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="24" height="24">
  <rect x="6" y="14" width="24" height="16" rx="3" fill="#f87171"/>
  <path d="M6,10 L30,7 L30,13 L6,14 Z" fill="#ef4444"/>
</svg>
"""

# 11. Gelir Bilinen Kart İkonu (Yeşil Dolar + Yükselen Ok)
CARD_REV_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="24" height="24">
  <text x="18" y="17" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#10b981">$</text>
  <polyline points="6,28 15,21 21,25 30,16" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline points="24,16 30,16 30,22" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

# 12. Bütçe Bilinen Kart İkonu (Mavi Cüzdan)
CARD_BUDGET_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="24" height="24">
  <rect x="5" y="9" width="26" height="19" rx="4" fill="#2563eb"/>
  <rect x="21" y="15" width="10" height="7" rx="2" fill="#60a5fa"/>
  <circle cx="24.5" cy="18.5" r="1.5" fill="#ffffff"/>
</svg>
"""

# 13. Temiz Veri Kart İkonu (Yeşil Dosya)
CARD_FILE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="24" height="24">
  <path d="M8,5 L21,5 L28,12 L28,31 L8,31 Z" fill="#0d9488"/>
  <polygon points="21,5 28,12 21,12" fill="#99f6e4"/>
</svg>
"""


def get_brand_favicon_path() -> str:
    """
    Tarayıcı sekmesi (page_icon) için renkli play butonlu klapet logosunu
    64x64 PNG dosyası olarak oluşturup yolunu döndürür.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    icon_path = os.path.join(data_dir, "brand_icon.png")
    if not os.path.exists(icon_path):
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Alt koyu gövde
            draw.rounded_rectangle([6, 20, 58, 58], radius=9, fill=(45, 55, 72, 255))

            # İç gökkuşağı gradyan alanı (yeşil -> sarı -> mercan -> mor)
            for x in range(11, 54):
                t = (x - 11) / 42.0
                if t < 0.35:
                    u = t / 0.35
                    r = int(74 + (250 - 74) * u)
                    g = int(222 + (204 - 222) * u)
                    b = int(128 + (21 - 128) * u)
                elif t < 0.7:
                    u = (t - 0.35) / 0.35
                    r = int(250 + (251 - 250) * u)
                    g = int(204 + (113 - 204) * u)
                    b = int(21 + (133 - 21) * u)
                else:
                    u = (t - 0.7) / 0.3
                    r = int(251 + (168 - 251) * u)
                    g = int(113 + (85 - 113) * u)
                    b = int(133 + (247 - 133) * u)
                draw.line([(x, 27), (x, 53)], fill=(r, g, b, 255))

            # Beyaz Play üçgeni
            draw.polygon([(27, 33), (27, 47), (40, 40)], fill=(255, 255, 255, 255))

            # Üst klapet şeridi ve beyaz çapraz şeritler
            draw.rounded_rectangle([6, 9, 58, 20], radius=4, fill=(30, 41, 59, 255))
            for sx in [14, 28, 42]:
                draw.polygon(
                    [(sx, 9), (sx + 7, 9), (sx + 3, 20), (sx - 4, 20)],
                    fill=(248, 250, 252, 255),
                )

            img.save(icon_path, "PNG")
        except Exception:
            return "🎬"
    return icon_path


def get_sidebar_custom_css() -> str:
    """
    Sol menüdeki (sidebar) radyo butonlarına kullanıcının paylaştığı görseldeki
    özel SVG logoları yerleştiren ve seçili öğeyi mercan-mavi geçişli vurgulayan CSS kodunu döndürür.
    """
    home_uri = svg_to_data_uri(HOME_SVG)
    dataset_uri = svg_to_data_uri(DATASET_SVG)
    apriori_uri = svg_to_data_uri(APRIORI_SVG)
    kmeans_uri = svg_to_data_uri(KMEANS_SVG)
    rf_uri = svg_to_data_uri(RF_SVG)
    rec_uri = svg_to_data_uri(REC_SVG)

    return f"""
    <style>
        /* Sidebar üst boşluğunu minimuma indir */
        [data-testid="stSidebarHeader"] {{
            height: 0.8rem !important;
            min-height: 0.8rem !important;
            padding: 0 !important;
        }}
        [data-testid="stSidebarUserContent"] {{
            padding-top: 0.1rem !important;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 0.1rem !important;
        }}

        /* Üst Marka Kutusu */
        .sidebar-brand-box {{
            display: flex;
            align-items: center;
            gap: 11px;
            margin-top: -12px;
            margin-bottom: 4px;
        }}
        .sidebar-brand-box img {{
            width: 42px;
            height: 42px;
            flex-shrink: 0;
            filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.15));
        }}
        .sidebar-brand-title {{
            font-size: 1.25rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.15;
            letter-spacing: -0.3px;
        }}
        .sidebar-brand-sub {{
            font-size: 0.80rem;
            color: #64748b;
            margin-top: 2px;
            margin-bottom: 14px;
        }}

        /* 📌 Menü Başlığı */
        .sidebar-menu-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.98rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 6px;
        }}
        .sidebar-menu-header img {{
            width: 20px;
            height: 20px;
        }}

        /* Streamlit Radio Varsayılan Dairelerini Gizle ve Özel İkonlu Menü Yap */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            gap: 4px !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
            display: flex !important;
            align-items: center !important;
            padding: 7px 10px !important;
            border-radius: 10px !important;
            margin: 0 !important;
            transition: background 0.2s ease, transform 0.15s ease !important;
            cursor: pointer !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label::before {{
            content: "";
            display: inline-block;
            width: 23px;
            height: 23px;
            margin-right: 10px;
            flex-shrink: 0;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1)::before {{
            background-image: url("{home_uri}");
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2)::before {{
            background-image: url("{dataset_uri}");
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3)::before {{
            background-image: url("{apriori_uri}");
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4)::before {{
            background-image: url("{kmeans_uri}");
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5)::before {{
            background-image: url("{rf_uri}");
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(6)::before {{
            background-image: url("{rec_uri}");
        }}

        /* Seçili Menü Öğesi Vurgusu (Görseldeki Mercan - Açık Mavi Geçişli Stil) */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
            background: linear-gradient(90deg, rgba(252, 165, 145, 0.55) 0%, rgba(186, 220, 254, 0.65) 100%) !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            background: rgba(148, 163, 184, 0.14);
        }}

        /* Veri Seti Özeti Kartları */
        .sidebar-summary-title {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
            font-size: 14px;
            color: #1e293b;
            margin-bottom: 10px;
        }}
        .sidebar-stat-card {{
            display: flex;
            align-items: center;
            gap: 12px;
            background: #ffffff;
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 14px;
            padding: 10px 14px;
            margin-bottom: 9px;
            box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
        }}
        .sidebar-stat-icon-box {{
            width: 38px;
            height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            flex-shrink: 0;
        }}
        .sidebar-stat-label {{
            font-size: 12px;
            color: #64748b;
            margin: 0;
        }}
        .sidebar-stat-val {{
            font-size: 16.5px;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }}
        .sidebar-clean-badge {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(209, 250, 229, 0.65);
            border: 1.5px solid rgba(52, 211, 153, 0.65);
            border-radius: 14px;
            padding: 10px 14px;
            margin-top: 12px;
        }}
    </style>
    """
