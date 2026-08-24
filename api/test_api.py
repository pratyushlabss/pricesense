from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("[PASS] /health")


def test_products_defaults_to_synthetic_dataset():
    response = client.get("/products")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "default"
    assert "Smartwatch" in body["products"]
    print("[PASS] /products defaults to synthetic dataset")


def test_forecast_returns_demand_curve():
    response = client.get("/forecast/Smartwatch")
    assert response.status_code == 200
    body = response.json()
    assert body["product"] == "Smartwatch"
    assert len(body["curve"]) == 20
    assert all("price" in point and "predicted_units" in point for point in body["curve"])
    print("[PASS] /forecast/Smartwatch returns a 20-point demand curve")


def test_optimize_returns_suggestion():
    response = client.get("/optimize/Smartwatch")
    assert response.status_code == 200
    body = response.json()
    assert "suggested_price" in body
    assert "explanation" in body
    print("[PASS] /optimize/Smartwatch returns a suggestion with explanation")


def test_unknown_product_returns_404():
    response = client.get("/forecast/NotAProduct")
    assert response.status_code == 404
    print("[PASS] unknown product returns 404, not a stack trace")


def test_unknown_dataset_id_returns_404():
    response = client.get("/products?dataset_id=doesnotexist")
    assert response.status_code == 404
    print("[PASS] unknown dataset_id returns 404")


def test_chat_returns_answer():
    response = client.post("/chat", json={"message": "why did the smartwatch price change?"})
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "default"
    assert "revenue" in body["answer"].lower()
    print("[PASS] /chat returns an answer for a product-specific question")


def test_upload_valid_csv_switches_active_dataset():
    with open("data/uci_backtest_data.csv", "rb") as f:
        response = client.post("/upload", files={"file": ("uci_backtest_data.csv", f, "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] > 0
    assert "WHITE HANGING HEART T-LIGHT HOLDER" in body["products"]

    products_response = client.get("/products")
    assert products_response.json()["dataset_id"] == body["dataset_id"]
    print("[PASS] valid upload succeeds and becomes the active dataset")


def test_upload_missing_column_rejected():
    csv_bytes = b"date,product,units_sold,price\n2024-01-01,Widget,5,10\n"
    response = client.post("/upload", files={"file": ("bad.csv", csv_bytes, "text/csv")})
    assert response.status_code == 400
    assert "stock_level" in str(response.json())
    print("[PASS] upload missing a required column is rejected with a clear error")


def test_upload_negative_price_rejected():
    csv_bytes = (
        b"date,product,units_sold,price,stock_level\n"
        b"2024-01-01,Widget,5,-10,10\n"
        b"2024-01-02,Widget,3,7,10\n"
    )
    response = client.post("/upload", files={"file": ("bad.csv", csv_bytes, "text/csv")})
    assert response.status_code == 400
    assert "price" in str(response.json())
    print("[PASS] upload with a negative price is rejected with a clear error")


def test_upload_empty_data_rejected():
    csv_bytes = b"date,product,units_sold,price,stock_level\n"
    response = client.post("/upload", files={"file": ("empty.csv", csv_bytes, "text/csv")})
    assert response.status_code == 400
    print("[PASS] upload with no data rows is rejected")


def test_white_hanging_heart_reproduction_via_api():
    with open("data/uci_backtest_data.csv", "rb") as f:
        upload_response = client.post("/upload", files={"file": ("uci_backtest_data.csv", f, "text/csv")})
    dataset_id = upload_response.json()["dataset_id"]

    response = client.get(f"/optimize/WHITE HANGING HEART T-LIGHT HOLDER?dataset_id={dataset_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["current_price"] == 2.73
    assert body["suggested_price"] == 3.06
    print(f"[PASS] WHITE HANGING HEART T-LIGHT HOLDER reproduction via API: "
          f"{body['current_price']} -> {body['suggested_price']}, "
          f"revenue_change={body['projected_revenue_change_pct']}%")


if __name__ == "__main__":
    test_health()
    test_products_defaults_to_synthetic_dataset()
    test_forecast_returns_demand_curve()
    test_optimize_returns_suggestion()
    test_unknown_product_returns_404()
    test_unknown_dataset_id_returns_404()
    test_chat_returns_answer()
    test_upload_valid_csv_switches_active_dataset()
    test_upload_missing_column_rejected()
    test_upload_negative_price_rejected()
    test_upload_empty_data_rejected()
    test_white_hanging_heart_reproduction_via_api()
    print("\nAll API checks passed.")
