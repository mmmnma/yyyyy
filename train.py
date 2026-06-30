"""
感情分析モデル学習スクリプト

data/sentiment_dataset.csv を読み込み、
TF-IDF + ロジスティック回帰で感情分析モデルを学習し、
models/ 以下に .joblib ファイルとして保存する。

実行方法:
    python3 train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sentiment_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

tokenizer = Tokenizer()


def tokenize(text: str) -> str:
    return " ".join(tokenizer.tokenize(text, wakati=True))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"データセットが見つかりません: {DATA_PATH}\n"
            "先に `python3 create_data.py` を実行してください。"
        )

    df = pd.read_csv(DATA_PATH)
    print(f"データ件数: {len(df)}")
    print(df["label"].value_counts().rename({0: "ネガティブ", 1: "ポジティブ"}))

    print("\nテキストを分かち書きしています...")
    df["tokenized"] = df["text"].apply(tokenize)

    # 学習用・評価用に分割して、汎化性能（未知データへの強さ）を確認する
    X_train, X_test, y_train, y_test = train_test_split(
        df["tokenized"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    eval_model = LogisticRegression(max_iter=1000)
    eval_model.fit(X_train_vec, y_train)
    train_acc = eval_model.score(X_train_vec, y_train)
    test_acc = eval_model.score(X_test_vec, y_test)

    # 5-fold cross validationで安定性も確認(過学習や偶然のブレに強い指標)
    X_full_for_cv = vectorizer.fit_transform(df["tokenized"])
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000), X_full_for_cv, df["label"], cv=5
    )

    print(f"\n訓練データ精度: {train_acc:.3f}")
    print(f"テストデータ精度: {test_acc:.3f}")
    print(f"5-fold CV精度: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # 本番用モデルは全データを使って学習する
    print("\n全データで本番用モデルを学習しています...")
    final_vectorizer = TfidfVectorizer()
    X_all = final_vectorizer.fit_transform(df["tokenized"])

    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_all, df["label"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(final_vectorizer, VECTORIZER_PATH)
    print(f"\n保存しました:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {VECTORIZER_PATH}")

    # 動作確認: 代表的な文章でテスト
    print("\n--- 動作確認 ---")
    sample_texts = [
        "とても悲しい",
        "とても辛いです",
        "とても嬉しい",
        "とても良かったです",
    ]
    for text in sample_texts:
        tok = tokenize(text)
        vec = final_vectorizer.transform([tok])
        pred = final_model.predict(vec)[0]
        proba = final_model.predict_proba(vec)[0]
        sentiment = "ポジティブ" if pred == 1 else "ネガティブ"
        print(f"  {text!r} -> {sentiment} (pos={proba[1]:.3f})")


if __name__ == "__main__":
    main()
