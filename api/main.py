"""
感情分析推論API

models/ ディレクトリに保存された joblib モデル（ロジスティック回帰）と
TF-IDFベクトライザーを読み込み、テキストの感情（ポジティブ/ネガティブ）を
予測するエンドポイントを提供する。

起動方法:
    uvicorn api.main:app --reload

動作確認:
    http://127.0.0.1:8000/docs にアクセスすると Swagger UI が開く
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from janome.tokenizer import Tokenizer
from pydantic import BaseModel, Field

# プロジェクト直下からの相対パスでモデルを参照する
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"

# モデルとベクトライザー、トークナイザーを保持するための簡易コンテナ
ml_models = {}


def tokenize(text: str) -> str:
    """日本語テキストをJanomeで分かち書きする"""
    return " ".join(ml_models["tokenizer"].tokenize(text, wakati=True))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリ起動時にモデルを読み込み、終了時に解放するライフサイクル管理"""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise RuntimeError(
            f"モデルファイルが見つかりません: {MODEL_PATH} / {VECTORIZER_PATH}\n"
            "先に学習スクリプトを実行し、models/ にjoblibファイルを配置してください。"
        )

    ml_models["model"] = joblib.load(MODEL_PATH)
    ml_models["vectorizer"] = joblib.load(VECTORIZER_PATH)
    ml_models["tokenizer"] = Tokenizer()
    print("モデルとベクトライザーの読み込みが完了しました")

    yield  # ここでアプリが起動し、リクエストを受け付ける

    ml_models.clear()
    print("モデルを解放しました")


app = FastAPI(
    title="感情分析推論API",
    description="TF-IDF + ロジスティック回帰によるテキスト感情分析API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """推論リクエストのスキーマ"""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="感情分析の対象となる日本語テキスト",
        examples=["とても良かったです"],
    )


class PredictResponse(BaseModel):
    """推論レスポンスのスキーマ"""

    text: str = Field(..., description="入力されたテキスト")
    label: int = Field(..., description="予測ラベル (1: ポジティブ, 0: ネガティブ)")
    sentiment: str = Field(..., description="予測ラベルの日本語表記")
    positive_proba: float = Field(..., description="ポジティブである確率 (0.0〜1.0)")
    negative_proba: float = Field(..., description="ネガティブである確率 (0.0〜1.0)")


@app.get("/")
def read_root():
    """ヘルスチェック用のルートエンドポイント"""
    return {"status": "ok", "message": "感情分析APIは正常に稼働しています"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    テキストを受け取り、ポジティブ/ネガティブを予測して返す

    - **text**: 分析したい日本語テキスト
    """
    model = ml_models.get("model")
    vectorizer = ml_models.get("vectorizer")

    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="モデルが読み込まれていません")

    tokenized_text = tokenize(request.text)
    vectorized_text = vectorizer.transform([tokenized_text])

    predicted_label = int(model.predict(vectorized_text)[0])
    probabilities = model.predict_proba(vectorized_text)[0]

    return PredictResponse(
        text=request.text,
        label=predicted_label,
        sentiment="ポジティブ" if predicted_label == 1 else "ネガティブ",
        negative_proba=round(float(probabilities[0]), 4),
        positive_proba=round(float(probabilities[1]), 4),
    )
