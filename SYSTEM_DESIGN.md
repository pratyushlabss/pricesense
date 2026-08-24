# PriceSense — System Design

PS-13 Dynamic E-commerce Pricing Simulator (AI/ML Hackathon 2026).

> **Note on this document's history:** this file did not exist prior to the final bug-sweep pass — it's being created now, from the system as actually built, rather than updated from a prior version. See `PROGRESS.md` for the chronological build log; this document is the reference snapshot of what exists today.

## 1. Overview

A self-contained dynamic pricing simulator: upload (or use demo) sales data → per-product demand forecasting → price optimization under business constraints → an explainability panel and chatbot that explain *why* → a what-if slider for manual exploration.

## 2. Constraints (NFR-1)

- $0 cost: no paid APIs, no paid compute, no cloud GPU. Runs entirely on a laptop.
- No LLM calls anywhere — the chatbot is retrieval-only (TF-IDF + logistic regression + templated answers over already-computed model output).
- Model + suggestion returns in under 5 seconds per product; chat responses under 2 seconds (retrieval over cached model output, not a fresh train).

## 3. Architecture

```
data/*.csv ──► models/forecast.py (DemandForecaster)
                       │
                       ▼
            models/optimizer.py (optimize_price)
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
models/explainability.py   models/chatbot.py (PricingChatbot)
        │                          │
        └──────────┬───────────────┘
                    ▼
              api/main.py (FastAPI)
                    │
                    ▼
            app/dashboard.py (Streamlit) ── pure API client, no model logic
```

Each layer only depends on the ones above it in this diagram; `api/main.py` is the sole integration point that wires `forecast.py`/`optimizer.py`/`chatbot.py` together per dataset.

## 4. Data schema

Both `data/synthetic_sales_data.csv` (generated) and `data/uci_backtest_data.csv` (ingested from UCI Online Retail II) and any user-uploaded CSV share one schema:

| column | type | notes |
|---|---|---|
| `date` | date | daily granularity |
| `product` | string | product identifier / display name |
| `units_sold` | int ≥ 0 | |
| `price` | float > 0 | |
| `stock_level` | int ≥ 0 | |

Upload validation (`api/main.py: validate_and_clean_upload`) enforces: required columns present, ≥1 data row, parseable dates, numeric coercion on the three numeric columns, **finite values within float32 range** (values that overflow when scikit-learn internally casts to float32 are rejected even if finite in float64 — e.g. `1e40`), `price > 0`, `units_sold ≥ 0`, `stock_level ≥ 0`.

## 5. Model layer

**`models/forecast.py` — `DemandForecaster`**: one `GradientBoostingRegressor` per product (features: `day_of_week`, `month`, `price`, `lag_7`, `lag_30`), lazily trained and joblib-cached per `(product, artifacts_dir)`. Products with under 30 valid history rows fall back to a flat historical-average estimate (`model=None` internally) rather than crashing.

**`models/optimizer.py` — `optimize_price`**: grid search over ±30% of current price, maximizing `price × min(predicted_units, stock)`, subject to a stock-runway floor (`DEFAULT_MIN_DAYS_OF_STOCK = 7` days) and a cost floor (`cost_price`, or `current_price × 0.6` by default). Detects flat/no-signal demand curves (`has_price_sensitivity`) and refuses to optimize against noise, returning `constraint_hit = "insufficient_price_sensitivity_data"` instead.

**`models/explainability.py`**: pure, framework-free functions shared by both the dashboard and the chatbot — `classify_panel_state` (buckets an `/optimize` result into one of `price_change_suggested` / `already_optimal` / `insufficient_data` / `stock_constrained` / `out_of_stock`), `build_headline`, `classify_price_sensitivity`, `CONSTRAINT_MESSAGES`.

**`models/chatbot.py` — `PricingChatbot`**: fixed 4-intent classifier (`explain_price_change`, `best_product`, `stock_risk`, `general_help`), TF-IDF + logistic regression trained on 80 hand-written examples, confidence-thresholded fallback to `general_help`. Product-name extraction: exact substring → significant-word token-overlap (with an explicit ambiguity guard — ties between products don't guess) → difflib fuzzy n-gram (same ambiguity guard). No network calls.

## 6. Dataset lifecycle (server-side)

`api/main.py` holds one in-memory `datasets` dict keyed by `dataset_id`, plus one `active_dataset_id` pointer. `"default"` is registered at startup over `synthetic_sales_data.csv`. A successful `/upload` creates a new id, a per-upload temp CSV + a **per-upload artifacts directory** (critical: sharing an artifacts dir across datasets would let a product name silently load a model cached from a different dataset), and becomes active. Uploaded files live under the OS temp dir, never inside the repo. Single-process/in-memory by design — not multi-user-safe, an explicit hackathon-scope tradeoff.

## 7. API contract

Base URL: `http://127.0.0.1:8000`. All GET routes accept an optional `?dataset_id=` to target a specific past upload instead of whichever is currently active.

### `GET /health`
→ `{"status": "ok"}`

### `POST /upload`
Multipart form field `file` (CSV). On success (200):
```json
{"dataset_id": "...", "row_count": 5840, "products": ["..."], "date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
```
On validation failure (400): `{"detail": {"errors": ["...", "..."]}}` — one or more human-readable messages, accumulated in a single response rather than one-at-a-time.

### `GET /products?dataset_id=`
→ `{"dataset_id": "...", "products": ["..."]}`. 404 (`{"detail": "Unknown dataset_id '...'"}`) if `dataset_id` doesn't exist.

### `GET /forecast/{product}?dataset_id=&steps=20&price=`
→
```json
{
  "dataset_id": "...", "product": "...",
  "current_price": 0.0, "current_stock": 0,
  "curve": [{"price": 0.0, "predicted_units": 0.0}, ...]
}
```
`curve` always spans ±30% of `current_price` in `steps` points (default 20) regardless of `price`. If the optional `price` query param is given, the response gains:
```json
"what_if": {"price": 0.0, "predicted_units": 0.0}
```
— a single-point prediction at that exact price, independent of `steps`/the curve's own range (used by the dashboard's what-if slider for prices outside ±30%).
Errors: 404 unknown product/dataset; 400 if `steps < 1`; 400 if `price` is present and is non-finite, ≤ 0, or exceeds float32 range (`~3.40e38`).

### `GET /optimize/{product}?dataset_id=`
→
```json
{
  "dataset_id": "...", "min_days_of_stock": 7,
  "product": "...", "current_price": 0.0, "current_stock": 0, "floor_price": 0.0,
  "current_predicted_units": 0.0, "suggested_price": 0.0, "suggested_predicted_units": 0.0,
  "projected_revenue_change_pct": 0.0, "days_of_stock_at_suggested_price": 0.0,
  "constraint_hit": null,
  "explanation": "..."
}
```
`constraint_hit` is one of `null` (unconstrained optimum found), `"no_feasible_price_found"` (stock-runway/cost-floor infeasible across the whole ±30% grid — covers both genuine stockouts and merely-tight stock; the dashboard/chatbot further split this into `out_of_stock` vs `stock_constrained` client-side using `current_stock == 0`), or `"insufficient_price_sensitivity_data"` (flat/no-signal demand curve). `days_of_stock_at_suggested_price` is `null` when `suggested_predicted_units` is 0. 404 for unknown product/dataset.

### `POST /chat?dataset_id=`
Body: `{"message": "..."}`. → `{"dataset_id": "...", "answer": "..."}` (a markdown-formatted string; never errors on well-formed input — falls back to a general-help message rather than a 4xx for anything it can't classify confidently. 404 for unknown `dataset_id`; 422 for a malformed request body, per FastAPI/Pydantic's standard validation error shape).

## 8. Dashboard (`app/dashboard.py`)

Pure API client — no model imports, no business logic beyond the pieces explicitly reused from `models/explainability.py`. Layout: CSV upload (optional) → active-dataset caption → product summary table (all products, one `/optimize` call each, cached) → per-product detail (selectbox, demand curve chart with current/suggested/what-if markers, price-sensitivity indicator, structured explanation panel, what-if slider) → chat.

Caching: `st.cache_data` keyed by `(product, dataset_id[, price])` — a new upload naturally invalidates via a new `dataset_id`, no manual cache-clearing needed. The what-if slider avoids API calls for prices within the already-fetched curve via client-side `np.interp`, only calling `/forecast?price=` for genuinely out-of-range values.

## 9. Known limitations (deliberate, not oversights)

- **Single-process, in-memory dataset store** — concurrent multi-user upload sessions would collide on `active_dataset_id`. Fine for a hackathon demo, not production-multi-tenant.
- **Tree-based demand model extrapolation** — `GradientBoostingRegressor` can flatten or behave non-monotonically at price extremes outside the training data's range.
- **Chatbot confidence is moderate** (0.44–0.71 on in-domain test questions), given only 80 training examples; every tested case classified correctly, but an adversarial phrasing could tip below the 0.4 threshold into `general_help` more easily than a larger classifier would. Single-product stock lookups ("is the mug in stock?") aren't a trained phrasing for `stock_risk` and fall to `general_help`.
- **`bound_to_forecaster` monkeypatch** (`models/chatbot.py`, used by both `/optimize` and the chatbot) swaps a module-level function for the duration of one call. Safe for the sequential request pattern this API is built for; not safe under true concurrent requests against *different* datasets in the same process.
- **One synthetic no-word-boundary product name** (a 150-character string with no spaces) won't match a short chat reference to it — no realistic retail product name looks like this, and the system already degrades safely (asks for clarification) rather than mismatching, so this was deliberately not engineered around further.
