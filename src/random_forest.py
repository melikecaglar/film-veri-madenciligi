"""
Üçüncü Algoritma: Random Forest Sınıflandırma Modülü.

Bu modül:
1. Geliri bilinen (revenue > 0) filmleri filtreler ve medyan geliri hesaplar.
2. Medyanın üzerindeki filmlere 'Yüksek Gelir', medyan ve altındakilere 'Düşük Gelir' etiketi atar.
3. Tahmin özellikleri olarak SADECE budget, popularity, runtime, vote_average, vote_count,
   release_year ve genres kullanır (revenue veya profit kesinlikle kullanılmaz).
4. Veri sızıntısını (data leakage) önlemek için önce train_test_split (%80 eğitim, %20 test, stratify=y)
   uygular, ardından MultiLabelBinarizer ve StandardScaler dönüştürücülerini yalnızca eğitim
   verisi üzerinde fit eder.
5. RandomForestClassifier (n_estimators=200, random_state=42, class_weight='balanced') modelini eğitir.
6. Test verisi üzerinde Accuracy, Precision, Recall, F1-Score, Confusion Matrix ve
   Feature Importance sonuçlarını hesaplar.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


NUMERIC_PREDICTORS = [
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
    "release_year",
]

FEATURE_TURKISH_NAMES = {
    "budget": "Film Bütçesi (budget)",
    "popularity": "Popülerlik (popularity)",
    "runtime": "Film Süresi (runtime)",
    "vote_average": "Ortalama Puan (vote_average)",
    "vote_count": "Oy Sayısı (vote_count)",
    "release_year": "Yayın Yılı (release_year)",
}


def run_random_forest_classification(
    df: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
) -> Dict[str, Any]:
    """
    Random Forest sınıflandırma modelini eğitir ve test verisi üzerinde değerlendirir.
    """
    if not (10 <= n_estimators <= 1000):
        raise ValueError("n_estimators değeri 10 ile 1000 arasında olmalıdır.")
    if not (0.10 <= test_size <= 0.40):
        raise ValueError("test_size değeri 0.10 ile 0.40 arasında olmalıdır.")

    # 1. Geliri bilinen filmleri filtrele (revenue > 0)
    known_df = df[df["revenue"] > 0].copy().reset_index(drop=True)
    removed_unknown_revenue = len(df) - len(known_df)

    if len(known_df) < 50:
        raise ValueError("Sınıflandırma için yeterli sayıda geliri bilinen film yok.")

    # 2. Medyan geliri hesapla ve hedef değişkeni oluştur
    median_revenue = float(known_df["revenue"].median())
    known_df["target_label"] = np.where(
        known_df["revenue"] > median_revenue, "Yüksek Gelir", "Düşük Gelir"
    )
    known_df["target_binary"] = np.where(
        known_df["revenue"] > median_revenue, 1, 0
    )

    y = known_df["target_binary"].values
    X_raw = known_df[NUMERIC_PREDICTORS + ["genre_list", "title", "revenue"]].copy()

    # 3. Veriyi %80 eğitim, %20 test olarak böl (Strict Featurization Ordering)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 4. Eğitim verisi üzerinde eksik değer / ölçeklendirme ve tür kodlamasını fit et
    train_num = X_train_raw[NUMERIC_PREDICTORS].copy()
    test_num = X_test_raw[NUMERIC_PREDICTORS].copy()

    # Eğitim setindeki medyanları kullanarak eksik değerleri doldur (veri sızıntısını önler)
    train_medians = train_num.median()
    train_num = train_num.fillna(train_medians)
    test_num = test_num.fillna(train_medians)

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(train_num)
    X_test_num_scaled = scaler.transform(test_num)

    # Türleri MultiLabelBinarizer ile yalnızca eğitim verisi üzerinde fit et
    mlb = MultiLabelBinarizer()
    X_train_genres = mlb.fit_transform(X_train_raw["genre_list"])
    X_test_genres = mlb.transform(X_test_raw["genre_list"])

    genre_feature_names = [f"Tür: {g}" for g in mlb.classes_]
    all_feature_names = [
        FEATURE_TURKISH_NAMES.get(c, c) for c in NUMERIC_PREDICTORS
    ] + genre_feature_names
    raw_feature_keys = list(NUMERIC_PREDICTORS) + [f"genre_{g}" for g in mlb.classes_]

    X_train_final = np.hstack([X_train_num_scaled, X_train_genres])
    X_test_final = np.hstack([X_test_num_scaled, X_test_genres])

    # 5. RandomForestClassifier modelini eğit
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )
    rf_model.fit(X_train_final, y_train)

    # 6. Test ve eğitim verisi üzerinde tahmin yap
    y_pred_test = rf_model.predict(X_test_final)
    y_prob_test = rf_model.predict_proba(X_test_final)[:, 1]
    y_pred_train = rf_model.predict(X_train_final)

    # 7. Değerlendirme ölçütlerini hesapla (Test verisi üzerinde)
    test_accuracy = float(accuracy_score(y_test, y_pred_test))
    test_precision = float(precision_score(y_test, y_pred_test, zero_division=0))
    test_recall = float(recall_score(y_test, y_pred_test, zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred_test, zero_division=0))
    test_roc_auc = float(roc_auc_score(y_test, y_prob_test))
    train_accuracy = float(accuracy_score(y_train, y_pred_train))

    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(
        cm,
        index=["Gerçek: Düşük Gelir", "Gerçek: Yüksek Gelir"],
        columns=["Tahmin: Düşük Gelir", "Tahmin: Yüksek Gelir"],
    )

    # Sınıf bazlı detaylı rapor
    report_dict = classification_report(
        y_test,
        y_pred_test,
        target_names=["Düşük Gelir", "Yüksek Gelir"],
        output_dict=True,
        zero_division=0,
    )
    class_report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .loc[["Düşük Gelir", "Yüksek Gelir", "weighted avg"]]
        .reset_index()
        .rename(
            columns={
                "index": "Sınıf",
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1-Score",
                "support": "Film Sayısı",
            }
        )
    )

    # Feature Importance tablosu
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame(
        {
            "feature_key": raw_feature_keys,
            "feature_name": all_feature_names,
            "importance": importances,
            "importance_pct": importances * 100.0,
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Test seti örnek tahmin tablosu
    test_predictions_df = pd.DataFrame(
        {
            "Film Adı": X_test_raw["title"].values,
            "Bütçe ($)": X_test_raw["budget"].values,
            "Gerçek Gelir ($)": X_test_raw["revenue"].values,
            "Gerçek Sınıf": np.where(y_test == 1, "Yüksek Gelir", "Düşük Gelir"),
            "Tahmin Edilen Sınıf": np.where(
                y_pred_test == 1, "Yüksek Gelir", "Düşük Gelir"
            ),
            "Yüksek Gelir Olasılığı (%)": np.round(y_prob_test * 100, 1),
            "Doğru Tahmin Mi?": np.where(y_test == y_pred_test, "Evet", "Hayır"),
        }
    )

    return {
        "model": rf_model,
        "scaler": scaler,
        "mlb": mlb,
        "train_medians": train_medians,
        "median_revenue": median_revenue,
        "total_known_movies": len(known_df),
        "removed_unknown_revenue": removed_unknown_revenue,
        "train_size": len(X_train_final),
        "test_size": len(X_test_final),
        "high_revenue_count": int((y == 1).sum()),
        "low_revenue_count": int((y == 0).sum()),
        "metrics": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "roc_auc": test_roc_auc,
            "train_accuracy": train_accuracy,
        },
        "confusion_matrix": cm,
        "confusion_matrix_df": cm_df,
        "class_report_df": class_report_df,
        "feature_importance_df": fi_df,
        "test_predictions_df": test_predictions_df,
    }


def predict_single_movie_revenue_class(
    rf_result: Dict[str, Any],
    budget: float,
    popularity: float,
    runtime: float,
    vote_average: float,
    vote_count: int,
    release_year: int,
    genres: List[str],
) -> Dict[str, Any]:
    """
    Eğitilmiş Random Forest modelini kullanarak tek bir film profili için
    Yüksek Gelir / Düşük Gelir tahmini ve olasılığı üretir.
    """
    scaler: StandardScaler = rf_result["scaler"]
    mlb: MultiLabelBinarizer = rf_result["mlb"]
    model: RandomForestClassifier = rf_result["model"]

    num_df = pd.DataFrame(
        [
            {
                "budget": budget,
                "popularity": popularity,
                "runtime": runtime,
                "vote_average": vote_average,
                "vote_count": vote_count,
                "release_year": release_year,
            }
        ]
    )
    num_scaled = scaler.transform(num_df[NUMERIC_PREDICTORS])
    valid_genres = [g for g in genres if g in set(mlb.classes_)]
    genres_encoded = mlb.transform([valid_genres])

    X_input = np.hstack([num_scaled, genres_encoded])
    pred_class = int(model.predict(X_input)[0])
    probs = model.predict_proba(X_input)[0]

    return {
        "predicted_label": "Yüksek Gelir" if pred_class == 1 else "Düşük Gelir",
        "high_revenue_prob": float(probs[1]),
        "low_revenue_prob": float(probs[0]),
        "median_threshold": rf_result["median_revenue"],
    }


if __name__ == "__main__":
    from data_preprocessing import get_or_create_cleaned_data

    df_clean, _ = get_or_create_cleaned_data()
    res = run_random_forest_classification(
        df_clean, n_estimators=200, random_state=42, class_weight="balanced"
    )
    print("=== RANDOM FOREST SINIFLANDIRMA SONUÇLARI ===")
    print(f"Geliri Bilinen Film Sayısı: {res['total_known_movies']} (Çıkarılan: {res['removed_unknown_revenue']})")
    print(f"Medyan Gelir Eşiği: ${res['median_revenue']:,.2f}")
    print(f"Eğitim Seti: {res['train_size']} | Test Seti: {res['test_size']}")
    m = res["metrics"]
    print(
        f"Test Accuracy:  {m['accuracy']:.4f}\n"
        f"Test Precision: {m['precision']:.4f}\n"
        f"Test Recall:    {m['recall']:.4f}\n"
        f"Test F1-Score:  {m['f1_score']:.4f}\n"
        f"Test ROC-AUC:   {m['roc_auc']:.4f}"
    )
    print("\nConfusion Matrix:")
    print(res["confusion_matrix_df"])
    print("\nEn Önemli 10 Özellik (Feature Importance):")
    print(res["feature_importance_df"][["feature_name", "importance_pct"]].head(10).to_string(index=False))
