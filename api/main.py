"""
PriceSense API

Run locally:
    uvicorn api.main:app --reload --port 8000

Example requests:
    curl http://127.0.0.1:8000/health
    curl http://127.0.0.1:8000/products
    curl "http://127.0.0.1:8000/forecast/Smartwatch"
    curl "http://127.0.0.1:8000/optimize/Smartwatch"
    curl -F "file=@data/synthetic_sales_data.csv" http://127.0.0.1:8000/upload
    curl "http://127.0.0.1:8000/products?dataset_id=<id from /upload response>"
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"message": "why did the smartwatch price change?"}'

Note: only one dataset is "active" at a time (the default, or the most recent
successful upload); pass ?dataset_id=... to any GET route to target a specific
past upload instead. This is a single-process, in-memory store sized for a
hackathon demo, not concurrent multi-user sessions.
"""

import io
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from models.chatbot import PricingChatbot, bound_to_forecaster
from models.forecast import DemandForecaster, get_forecaster, load_sales_data
from models.optimizer import DEFAULT_MIN_DAYS_OF_STOCK, optimize_price

app = FastAPI(title="PriceSense API")

REQUIRED_COLUMNS = {"date", "product", "units_sold", "price", "stock_level"}
DEFAULT_DATASET_ID = "default"
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "pricesense_uploads"
# Values must survive an internal float32 cast (scikit-learn casts features to float32) —
# a value that's finite in float64 can still overflow to inf at that point.
MAX_SAFE_MAGNITUDE = float(np.finfo(np.float32).max)

datasets = {}
active_dataset_id = DEFAULT_DATASET_ID


def register_default_dataset():
    forecaster = get_forecaster()
    default_df = load_sales_data()
    datasets[DEFAULT_DATASET_ID] = {
        "forecaster": forecaster,
        "filename": "synthetic_sales_data.csv",
        "row_count": len(default_df),
        "products": forecaster.list_products(),
        "date_range": {
            "start": default_df["date"].min().strftime("%Y-%m-%d"),
            "end": default_df["date"].max().strftime("%Y-%m-%d"),
        },
    }


register_default_dataset()


def resolve_dataset(dataset_id):
    key = dataset_id or active_dataset_id
    if key not in datasets:
        raise HTTPException(status_code=404, detail=f"Unknown dataset_id '{key}'")
    return key, datasets[key]


def validate_and_clean_upload(df):
    errors = []
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        return None, [f"Missing required columns: {sorted(missing_columns)}"]

    df = df[list(REQUIRED_COLUMNS)].copy()
    if len(df) == 0:
        return None, ["Uploaded file has no data rows"]

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        errors.append(f"{int(parsed_dates.isna().sum())} row(s) have an unparseable 'date' value")

    numeric_columns = {}
    for column in ["units_sold", "price", "stock_level"]:
        numeric = pd.to_numeric(df[column], errors="coerce")
        numeric_columns[column] = numeric
        if numeric.isna().any():
            errors.append(f"{int(numeric.isna().sum())} row(s) have a non-numeric '{column}' value")

    if errors:
        return None, errors

    df["date"] = parsed_dates
    for column, numeric in numeric_columns.items():
        df[column] = numeric

    for column in ["units_sold", "price", "stock_level"]:
        if not np.isfinite(df[column]).all():
            errors.append(f"'{column}' must be a finite number for every row (no infinity/NaN)")
        elif (df[column].abs() > MAX_SAFE_MAGNITUDE).any():
            errors.append(f"'{column}' has a value too large to model (must stay under {MAX_SAFE_MAGNITUDE:.2e})")

    if (df["price"] <= 0).any():
        errors.append("'price' must be greater than 0 for every row")
    if (df["units_sold"] < 0).any():
        errors.append("'units_sold' cannot be negative")
    if (df["stock_level"] < 0).any():
        errors.append("'stock_level' cannot be negative")

    if errors:
        return None, errors

    return df, []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file as CSV: {exc}")

    cleaned_df, errors = validate_and_clean_upload(df)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    dataset_id = uuid.uuid4().hex[:12]
    dataset_dir = UPLOAD_ROOT / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_path = dataset_dir / "data.csv"
    cleaned_df.to_csv(data_path, index=False)

    forecaster = DemandForecaster(data_path=data_path, artifacts_dir=dataset_dir / "artifacts")
    date_range = {
        "start": cleaned_df["date"].min().strftime("%Y-%m-%d"),
        "end": cleaned_df["date"].max().strftime("%Y-%m-%d"),
    }
    datasets[dataset_id] = {
        "forecaster": forecaster,
        "filename": file.filename,
        "row_count": len(cleaned_df),
        "products": forecaster.list_products(),
        "date_range": date_range,
    }

    global active_dataset_id
    active_dataset_id = dataset_id

    return {
        "dataset_id": dataset_id,
        "row_count": len(cleaned_df),
        "products": datasets[dataset_id]["products"],
        "date_range": date_range,
    }


@app.get("/products")
def list_products(dataset_id: Optional[str] = None):
    resolved_id, dataset = resolve_dataset(dataset_id)
    return {"dataset_id": resolved_id, "products": dataset["products"]}


@app.get("/forecast/{product}")
def get_forecast(product: str, dataset_id: Optional[str] = None, steps: int = 20, price: Optional[float] = None):
    if steps < 1:
        raise HTTPException(status_code=400, detail="steps must be at least 1")

    resolved_id, dataset = resolve_dataset(dataset_id)
    if product not in dataset["products"]:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product}' not found in dataset '{resolved_id}'",
        )

    forecaster = dataset["forecaster"]
    current_price, current_stock = forecaster.current_price_and_stock(product)
    price_range = (current_price * 0.7, current_price * 1.3)
    curve = forecaster.demand_curve(product, price_range, steps=steps)

    response = {
        "dataset_id": resolved_id,
        "product": product,
        "current_price": round(current_price, 2),
        "current_stock": current_stock,
        "curve": [{"price": p, "predicted_units": units} for p, units in curve],
    }

    if price is not None:
        if not np.isfinite(price) or price <= 0 or price > MAX_SAFE_MAGNITUDE:
            raise HTTPException(
                status_code=400,
                detail=f"price must be a finite number greater than 0 and under {MAX_SAFE_MAGNITUDE:.2e}",
            )
        response["what_if"] = {
            "price": round(price, 2),
            "predicted_units": round(forecaster.predict_demand(product, price), 2),
        }

    return response


@app.get("/optimize/{product}")
def get_optimization(product: str, dataset_id: Optional[str] = None):
    resolved_id, dataset = resolve_dataset(dataset_id)
    if product not in dataset["products"]:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product}' not found in dataset '{resolved_id}'",
        )

    with bound_to_forecaster(dataset["forecaster"]):
        result = optimize_price(product)

    return {"dataset_id": resolved_id, "min_days_of_stock": DEFAULT_MIN_DAYS_OF_STOCK, **result}


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest, dataset_id: Optional[str] = None):
    resolved_id, dataset = resolve_dataset(dataset_id)
    chatbot = PricingChatbot(dataset["forecaster"], dataset["products"])
    answer = chatbot.answer(request.message)
    return {"dataset_id": resolved_id, "answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
