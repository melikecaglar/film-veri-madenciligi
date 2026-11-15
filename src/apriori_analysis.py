"""
Birinci Algoritma: Apriori Birliktelik Kuralı Madenciliği Modülü.

Bu modül:
1. Her filmi bir işlem (transaction) olarak kabul eder.
2. Her filmin türlerini (genre_list) ve uygun anahtar kelimelerini (keyword_list) bir listeye dönüştürür.
3. One-Hot Encoding (TransactionEncoder / ikili matris) ile boolean veri çerçevesine çevirir.
4. mlxtend.frequent_patterns.apriori ile sık görülen öğe kümelerini (frequent itemsets) bulur.
5. mlxtend.frequent_patterns.association_rules ile birliktelik kurallarını (support, confidence, lift) üretir.
6. En sık görülen tekli türleri, tür kombinasyonlarını ve en güçlü birliktelik kurallarını döndürür.
"""

from collections import Counter
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def build_transactions(
    df: pd.DataFrame,
    include_keywords: bool = True,
    keyword_prefix: str = "Anahtar: ",
    min_keyword_count: int = 15,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    """
    Film veri setinden Apriori algoritması için işlem (transaction) listeleri oluşturur.

    Parametreler:
    - df: Temizlenmiş film veri çerçevesi (genre_list ve keyword_list sütunlarını içeren).
    - include_keywords: Türlerin yanı sıra sık görülen anahtar kelimelerin de dahil edilip edilmeyeceği.
    - keyword_prefix: Anahtar kelimeleri türlerden ayırt etmek için önek (boş bırakılabilir).
    - min_keyword_count: Çok nadir anahtar kelimeleri ön-elemek için minimum görülme sayısı.
    """
    # Önce tüm anahtar kelimelerin frekansını hesapla
    keyword_counter: Counter = Counter()
    if include_keywords and "keyword_list" in df.columns:
        for kw_list in df["keyword_list"]:
            if isinstance(kw_list, list):
                for kw in set(kw_list):
                    if kw:
                        keyword_counter[kw] += 1

    valid_keywords = {
        kw for kw, cnt in keyword_counter.items() if cnt >= min_keyword_count
    }

    transactions: List[List[str]] = []
    empty_transaction_count = 0

    for _, row in df.iterrows():
        items: List[str] = []
        genres = row.get("genre_list", [])
        if isinstance(genres, list):
            for g in genres:
                if g and g not in items:
                    items.append(g)

        if include_keywords:
            keywords = row.get("keyword_list", [])
            if isinstance(keywords, list):
                for kw in keywords:
                    if kw in valid_keywords:
                        labeled_kw = f"{keyword_prefix}{kw}" if keyword_prefix else kw
                        if labeled_kw not in items:
                            items.append(labeled_kw)

        if items:
            transactions.append(items)
        else:
            empty_transaction_count += 1

    info = {
        "total_movies": len(df),
        "valid_transactions": len(transactions),
        "empty_transactions": empty_transaction_count,
        "include_keywords": include_keywords,
        "unique_valid_keywords": len(valid_keywords),
    }
    return transactions, info


def encode_transactions_one_hot(transactions: List[List[str]]) -> pd.DataFrame:
    """
    İşlem listelerini One-Hot Encoding ile boolean (True/False) matrise dönüştürür.
    """
    if not transactions:
        raise ValueError("İşlem (transaction) listesi boş! Apriori çalıştırılamaz.")

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot_df = pd.DataFrame(te_ary, columns=te.columns_)
    return onehot_df


def run_apriori_analysis(
    df: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.30,
    min_lift: float = 1.0,
    metric: str = "confidence",
    include_keywords: bool = True,
    keyword_prefix: str = "Anahtar: ",
) -> Dict[str, Any]:
    """
    Veri seti üzerinde Apriori birliktelik kuralı madenciliğini çalıştırır.

    Başlangıç parametreleri:
    - min_support = 0.02
    - min_confidence = 0.30
    - min_lift (min_threshold) = 1.0
    - metric = "confidence"
    """
    if not (0.001 <= min_support <= 1.0):
        raise ValueError("min_support değeri 0.001 ile 1.0 arasında olmalıdır.")
    if not (0.01 <= min_confidence <= 1.0):
        raise ValueError("min_confidence değeri 0.01 ile 1.0 arasında olmalıdır.")

    # Apriori anti-monotonluk özelliğine göre min_support altındaki tekil anahtar kelimeler
    # zaten sık öğe kümesi olamaz; bu eşik ile gürültüyü temizliyoruz.
    min_kw_cnt = max(5, int(len(df) * min_support * 0.8))

    transactions, tx_info = build_transactions(
        df,
        include_keywords=include_keywords,
        keyword_prefix=keyword_prefix,
        min_keyword_count=min_kw_cnt,
    )

    onehot_df = encode_transactions_one_hot(transactions)

    # Sık görülen öğe kümelerini bul
    frequent_itemsets = apriori(
        onehot_df,
        min_support=min_support,
        use_colnames=True,
    )

    if frequent_itemsets.empty:
        return {
            "status": "empty_itemsets",
            "message": "Seçilen min_support değeri için sık görülen öğe kümesi bulunamadı. Lütfen min_support değerini düşürün.",
            "transaction_info": tx_info,
            "frequent_itemsets": pd.DataFrame(),
            "single_items": pd.DataFrame(),
            "combo_items": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "top_10_rules": pd.DataFrame(),
            "genre_keyword_rules": pd.DataFrame(),
        }

    # Öğe sayısı ve okunabilir metin temsili
    frequent_itemsets["item_count"] = frequent_itemsets["itemsets"].apply(len)
    frequent_itemsets["items_str"] = frequent_itemsets["itemsets"].apply(
        lambda s: ", ".join(sorted(list(s)))
    )
    frequent_itemsets = frequent_itemsets.sort_values(
        by="support", ascending=False
    ).reset_index(drop=True)

    # Tekli sık görülen öğeler (türler / anahtar kelimeler)
    single_items = (
        frequent_itemsets[frequent_itemsets["item_count"] == 1]
        .copy()
        .reset_index(drop=True)
    )
    single_items["film_count"] = (
        single_items["support"] * tx_info["valid_transactions"]
    ).round().astype(int)

    # Çoklu tür/öğe kombinasyonları (2 veya daha fazla öğe içerenler)
    combo_items = (
        frequent_itemsets[frequent_itemsets["item_count"] >= 2]
        .copy()
        .reset_index(drop=True)
    )
    combo_items["film_count"] = (
        combo_items["support"] * tx_info["valid_transactions"]
    ).round().astype(int)

    # Birliktelik kurallarını oluştur
    threshold_val = min_confidence if metric == "confidence" else min_lift
    rules = association_rules(
        frequent_itemsets[["support", "itemsets"]],
        metric=metric,
        min_threshold=threshold_val,
    )

    if rules.empty:
        return {
            "status": "empty_rules",
            "message": "Sık öğe kümeleri bulundu ancak seçilen confidence/lift eşiklerini sağlayan kural üretilemedi.",
            "transaction_info": tx_info,
            "frequent_itemsets": frequent_itemsets,
            "single_items": single_items,
            "combo_items": combo_items,
            "rules": pd.DataFrame(),
            "top_10_rules": pd.DataFrame(),
            "genre_keyword_rules": pd.DataFrame(),
        }

    # Hem min_confidence hem min_lift filtrelerini uygula
    rules = rules[
        (rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift)
    ].copy()

    if rules.empty:
        return {
            "status": "empty_rules",
            "message": "Belirtilen min_confidence ve min_lift kriterlerini aynı anda sağlayan kural bulunamadı.",
            "transaction_info": tx_info,
            "frequent_itemsets": frequent_itemsets,
            "single_items": single_items,
            "combo_items": combo_items,
            "rules": pd.DataFrame(),
            "top_10_rules": pd.DataFrame(),
            "genre_keyword_rules": pd.DataFrame(),
        }

    # Okunabilir öncül (antecedents) ve ardıl (consequents) sütunları
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda s: ", ".join(sorted(list(s)))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda s: ", ".join(sorted(list(s)))
    )
    rules["rule_str"] = (
        rules["antecedents_str"] + " -> " + rules["consequents_str"]
    )

    # Anahtar kelime içeren kuralları işaretle
    if include_keywords and keyword_prefix:
        rules["has_keyword"] = rules["rule_str"].str.contains(keyword_prefix, regex=False)
    else:
        rules["has_keyword"] = False

    # Lift ve confidence değerine göre sırala
    rules = rules.sort_values(
        by=["lift", "confidence", "support"], ascending=[False, False, False]
    ).reset_index(drop=True)

    # En güçlü 10 birliktelik kuralı
    top_10_rules = rules.head(10).copy()

    # Anahtar kelime - film türü ilişkisini gösteren kurallar
    genre_keyword_rules = (
        rules[rules["has_keyword"]].reset_index(drop=True)
        if include_keywords
        else pd.DataFrame()
    )

    return {
        "status": "success",
        "message": f"Toplam {len(frequent_itemsets)} sık öğe kümesi ve {len(rules)} güçlü birliktelik kuralı bulundu.",
        "transaction_info": tx_info,
        "frequent_itemsets": frequent_itemsets,
        "single_items": single_items,
        "combo_items": combo_items,
        "rules": rules,
        "top_10_rules": top_10_rules,
        "genre_keyword_rules": genre_keyword_rules,
    }


if __name__ == "__main__":
    from data_preprocessing import get_or_create_cleaned_data

    df_clean, _ = get_or_create_cleaned_data()
    res = run_apriori_analysis(
        df_clean,
        min_support=0.02,
        min_confidence=0.30,
        min_lift=1.0,
        include_keywords=True,
    )
    print("=== APRIORI BİRLİKTELİK ANALİZİ SONUÇLARI ===")
    print(res["message"])
    print("\nEn Sık Görülen 5 Tekil Öğe:")
    print(res["single_items"][["items_str", "support", "film_count"]].head(5))
    print("\nEn Sık Görülen 5 Tür/Öğe Kombinasyonu:")
    print(res["combo_items"][["items_str", "support", "film_count"]].head(5))
    print("\nEn Güçlü 10 Birliktelik Kuralı:")
    print(
        res["top_10_rules"][
            ["rule_str", "support", "confidence", "lift"]
        ].to_string(index=False)
    )
