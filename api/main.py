
"""
感情分析推論API
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from janome.tokenizer import Tokenizer
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
FRONTEND_DIR = BASE_DIR / "frontend"

ml_models = {}


def tokenize(text: str) -> str:
    return " ".join(ml_models["tokenizer"].tokenize(text, wakati=True))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise RuntimeError(
            f"モデルファイルが見つかりません: {MODEL_PATH} / {VECTORIZER_PATH}"
        )

    ml_models["model"] = joblib.load(MODEL_PATH)
    ml_models["vectorizer"] = joblib.load(VECTORIZER_PATH)
    ml_models["tokenizer"] = Tokenizer()
    print("モデルとベクトライザーの読み込みが完了しました")

    yield

    ml_models.clear()
    print("モデルを解放しました")


app = FastAPI(
    title="感情分析推論API",
    description="TF-IDF + ロジスティック回帰によるテキスト感情分析API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="感情分析の対象となる日本語テキスト",
        examples=["とても良かったです"],
    )


class PredictResponse(BaseModel):
    text: str = Field(..., description="入力されたテキスト")
    label: int = Field(..., description="予測ラベル (1: ポジティブ, 0: ネガティブ)")
    sentiment: str = Field(..., description="予測ラベルの日本語表記")
    positive_proba: float = Field(..., description="ポジティブである確率 (0.0〜1.0)")
    negative_proba: float = Field(..., description="ネガティブである確率 (0.0〜1.0)")


@app.get("/")
def read_root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
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
