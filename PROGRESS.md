# PS-13 Dynamic E-commerce Pricing Simulator — Progress

## Milestones

- [x] Data layer — synthetic generator + UCI ingestion
- [x] Demand forecasting model
- [x] Price optimizer — re-verified and hardened against current forecast.py
- [x] FastAPI backend
- [x] Streamlit dashboard
- [x] Explainability panel
- [x] NLP chatbot
- [x] What-if price slider
- [x] Polish & edge cases
- [x] Final bug sweep
- [x] Dashboard visual redesign (card-based, multi-page)
- [x] Premium SaaS UI pass (insights, comparison, design system)
- [x] Motion, command-center, and intelligence pass (opportunity map, insight feed, animations)
- [x] Hero positioning, Pricing Health, Guardrails, AI Insights page
- [ ] Demo prep

## Data layer (done)

- `data/generate_synthetic_data.py` → `data/synthetic_sales_data.csv`, 8 products, 730 days, built-in elasticity/seasonality/stock cycles
- `data/ingest_uci_retail.py` → `data/uci_backtest_data.csv`, 20 real products from UCI Online Retail II, same 5-column schema
- Raw UCI dump lives in `data/raw/` (gitignored, 96MB); only the cleaned 712K output is committed
- Both outputs share schema: `date, product, units_sold, price, stock_level`

## Demand forecasting model (done)

**Built:** `models/forecast.py`, class `DemandForecaster`

- One `GradientBoostingRegressor` per product (`n_estimators=150, max_depth=3, learning_rate=0.08`)
- Features: `day_of_week`, `month`, `price`, `lag_7`, `lag_30` (lags are shifted rolling means, no leakage)
- `fit(data_path=None)` — trains all per-product models upfront; bare filenames resolve against `data/`
- Lazy per-product load-or-train also still works via `predict_demand`/`_get_model` (joblib-cached to `models/artifacts/`), so callers that never call `fit()` explicitly (optimizer, CLI) still get fast cached predictions
- `predict_demand(product, price)` → expected units sold, holding latest known calendar/lag snapshot fixed
- `demand_curve(product, price_range, steps=20)` — accepts either a `(low, high)` tuple (linspace) or an explicit iterable of prices (backward-compat with existing callers)
- `current_price_and_stock(product)` → last known price/stock from data

**Design decisions:**
- Prediction varies price while holding day/month/lag features at the "next day" snapshot — true future lags for a hypothetical price are unknowable, so this is a deliberate simplification for the demand-curve use case
- Joblib caching keeps repeated calls fast (~0.005–0.02s warm vs ~0.15s cold train per product) — well under the 5s/product budget
- `train_all` kept as an alias of `fit` for backward compatibility

**Known limitations / edge cases handled:**
- Products with fewer than 30 valid feature rows (insufficient history) don't crash — they fall back to a flat historical-average estimate, price-blind, stored alongside real models in the same joblib cache format (`model: None`, `average: <float>`). Verified with a 10-row synthetic slice.
- Tree-based GBR models can flatten or behave non-monotonically at extrapolated price extremes (visible on some products, e.g. Air Fryer) — inherent to tree regressors, not a bug, already disclosed.

## Price optimizer (done, re-verified + hardened)

**Audited:** `models/optimizer.py` against the current `forecast.py` (grid search ±30%, stock-runway constraint, cost-floor constraint) — all still correct. Refactored the search loop to consume `forecaster.demand_curve(product, grid)` once instead of calling `predict_demand` per price in a loop — same results, fewer redundant calls.

**Bug found and fixed:** the optimizer had no defense against the forecast fallback case (flat, price-blind average for low-history products). Because revenue = price × units and units was constant regardless of price, the search would always pick the top of the ±30% grid as "optimal" — a nonsensical suggestion with zero real signal behind it. Added `has_price_sensitivity()`: computes the spread of predicted units across the full search grid, and if it's below both an absolute epsilon (1e-6) and 1% of the mean, treats the product as having no elasticity signal. New `constraint_hit = "insufficient_price_sensitivity_data"` short-circuits the search and holds current price with an explicit explanation instead of optimizing against noise.

**Verified via `models/test_optimizer.py`** (not eyeballed — asserted):
- Insufficient-history product (10-row Yoga Mat slice) → holds current price, `constraint_hit = "insufficient_price_sensitivity_data"` ✅
- Stockout (`current_stock=0`, Wireless Earbuds) → still returns `"no_feasible_price_found"` with the restock-soon explanation, not a misleading price change ✅
- Smartwatch (synthetic data) → suggested price stays within ±30% grid, respects floor price and min-days-of-stock ✅
- **WHITE HANGING HEART T-LIGHT HOLDER (UCI data) reproduction: current ₹2.73 → suggested ₹3.06, revenue change −29.02%** — matches the flagged demo case (price raised to protect stock runway despite the revenue cost). Minor revenue-% delta (−29.02% vs. the earlier −29.08%) is just from now reusing `demand_curve`'s 2-decimal-rounded values in the grid instead of raw floats — same story, same direction, same magnitude.
- Timing: all 8 synthetic products + all 20 UCI products stayed under 5s (0.06–0.21s range) ✅

## FastAPI backend (done)

**Built:** `api/main.py` — loads `forecast.py`/`optimizer.py` unmodified. Endpoints: `GET /health`, `POST /upload`, `GET /products`, `GET /forecast/{product}`, `GET /optimize/{product}`, `POST /chat` (placeholder, 501).

**Design decisions:**
- **Session/dataset handling:** single in-memory `datasets` dict keyed by `dataset_id`, plus one `active_dataset_id` pointer. `default` is registered at startup over `synthetic_sales_data.csv` (reuses `forecast.py`'s own singleton/cache, so no duplicate training). A successful `/upload` registers a new id, builds a fresh `DemandForecaster` pointed at a per-upload temp CSV + a **per-upload artifacts dir** (critical: reusing a shared artifacts dir across datasets would let a product name silently load a joblib model cached from a *different* dataset), and makes it active. Every GET route accepts an optional `?dataset_id=` to target a specific past upload instead of the active one. Explicitly scoped as single-process/in-memory, not multi-user-safe — correct call for a hackathon demo, called out in the module docstring.
- **Uploaded files never touch the repo:** saved under the OS temp dir (`tempfile.gettempdir()/pricesense_uploads/<id>/`), not inside the project, so no `.gitignore` changes were needed.
- **optimizer.py's hardcoded `get_forecaster()` dependency:** `optimize_price()` always pulls from `forecast.py`'s module-level singleton, so it can't be pointed at an uploaded dataset's forecaster directly without touching optimizer.py (out of scope). Worked around with `bound_to_forecaster()`, a context manager that swaps `optimizer_module.get_forecaster` for the duration of one request, then restores it — same technique already used in `models/test_optimizer.py`. Documented as a sequential-request simplification, not concurrency-safe.
- **Validation:** required-column check → date parseability → numeric coercion on `units_sold`/`price`/`stock_level` → positivity/non-negativity checks, short-circuiting with all accumulated errors returned as a list in one 400 response (not raised one at a time). Unknown product or dataset_id → explicit 404 with a message, never an unhandled 500/stack trace.

**Verified** (both manual curl and an automated suite, `api/test_api.py`, 12/12 passing):
- `/health`, `/products` (defaults to synthetic dataset)
- `/forecast/Smartwatch` → 20-point demand curve
- `/optimize/Smartwatch` → suggestion + explanation
- Unknown product → 404, unknown `dataset_id` → 404, `/chat` → 501
- `/upload` valid CSV (UCI backtest data) → switches active dataset, product list updates
- `/upload` rejected cases: missing column, non-numeric value, negative price, zero data rows — all 400 with clear messages
- **WHITE HANGING HEART T-LIGHT HOLDER reproduction via the live API: ₹2.73 → ₹3.06, revenue change −29.02%** — matches the optimizer-level result exactly
- Timing: `/health` 1ms, `/products` 1ms, `/forecast` 24ms, `/optimize` 66ms, `/upload` 33ms — all far under the 5s budget

## Streamlit dashboard (done)

**Built:** `app/dashboard.py` — pure API client, no model logic. Layout top to bottom: optional CSV upload → active-dataset caption → product summary table (renders on load, no clicks) → per-product detail (selectbox, Plotly demand curve with current/suggested price markers, metrics, explanation) → "Ask about your data" chatbot placeholder (`st.info("Coming soon")`).

**Design decisions:**
- **Caching:** `st.cache_data` on `fetch_products`/`fetch_forecast`/`fetch_optimize`, keyed by `(product, dataset_id)`. No manual cache-clearing needed — a new upload produces a new `dataset_id`, which is a different argument value and therefore a natural cache miss; the old dataset's cached entries just go unused rather than needing invalidation.
- **Default dataset flow:** dashboard never calls `/upload` for the default case — it just calls `/products`/`/forecast`/`/optimize` with `dataset_id=None` (omitted from the request), and the API's own "default" active dataset (synthetic data) answers. Once a response comes back, its resolved `dataset_id` is cached into `session_state` so all later calls this session are pinned explicitly rather than riding the API's mutable "active" pointer.
- **Active-dataset label is client-side only:** the API has no endpoint that returns a dataset's filename (only `dataset_id`), so "Demo dataset (synthetic_sales_data.csv)" vs. the uploaded filename is tracked purely in Streamlit `session_state`, set at upload time. **Flag for later:** would be cleaner if `/products` (or a new `/dataset` route) echoed back the filename so the label survives a page refresh — not built since it requires touching `api/main.py`, out of this task's scope.
- **Error handling:** every API call goes through `api_get`/`api_post_upload`, which catch `requests.exceptions.RequestException` (API not running) and non-2xx responses (unpacking the `{"errors": [...]}` shape from upload validation or a plain `detail` string) into a `(data, error_message)` tuple. Errors surface as `st.error`/`st.warning`, never a raw traceback; an unreachable API halts with `st.stop()` right after the first `/products` call so nothing downstream tries to render on missing data.
- Upload de-duplication via `uploaded_file.file_id` so the same script rerun (triggered by any later widget interaction) doesn't re-POST the same file to `/upload` repeatedly.

**Verified end-to-end** against the live API using Streamlit's `AppTest` headless framework (not just eyeballed):
- Default load: title, caption ("Demo dataset..."), 1 summary dataframe, 1 Plotly chart, 3 metrics, explanation text, and the chatbot placeholder all render with zero exceptions
- Selectbox interaction (switching products) re-renders the detail view correctly with no exceptions, values match `test_cli.py`'s Smartwatch numbers exactly (₹2934.50 → ₹2230.22, +26.4%)
- Simulated an upload (`dataset_id` set to a real `/upload` response) against the 20-product UCI dataset — caption updates to the filename, selectbox shows all 20 products
- **WHITE HANGING HEART T-LIGHT HOLDER reproduction via the dashboard's own rendered metrics/explanation: ₹2.73 → ₹3.06, −29.0%** — matches the API and optimizer results exactly
- API-down case: `st.error` with an actionable message ("Start it with: uvicorn api.main:app..."), zero exceptions, no crash

## Explainability panel (done)

**Built:** enhanced the per-product detail view in `app/dashboard.py` — no API or model changes, everything derived client-side from the already-returned `/forecast` and `/optimize` payloads.

- **Chart annotations:** the demand curve now marks the current price (gray diamond) and, when a change is actually suggested, the suggested price (green star) as scatter points with inline text labels showing both price and predicted units at each point — not just a vertical line implying the tradeoff.
- **Price-sensitivity indicator:** `classify_price_sensitivity()` computes an arc-elasticity-style index client-side from the `/forecast` curve — `(% spread in predicted units) / (% spread in price)` across the ±30% grid — bucketed into High/Medium/Low, shown next to the chart via `st.metric` with the raw index as a caption. Insufficient-data products get an explicit "No signal (insufficient data)" override rather than a possibly-misleading "Low" from a genuinely flat curve.
- **Structured explanation block**, replacing the old single `st.info(explanation)`: a bold headline (`₹2934.50 → ₹2230.22: +66% demand, +26.4% revenue`, or `Hold current price at ₹X` when nothing changes), a color-coded constraint line that's shown in **every** state (not just when a change is suggested), and the original rule-based sentence kept below as a supporting-detail caption.
- **Five distinct panel states**, classified by `classify_panel_state()` from `constraint_hit` + `current_stock` + price-equality: `price_change_suggested` (green ✅, two chart markers), `already_optimal` (green ✅, current price already wins the search — distinguished from the fallback states so "nothing to change" reads as good news, not missing data), `insufficient_data` (blue ℹ️), `stock_constrained` (yellow ⚠️, nonzero stock but too little runway), `out_of_stock` (red 🔴, `current_stock == 0` specifically — the API doesn't distinguish this from `stock_constrained` at the `constraint_hit` level, so the dashboard adds the split itself). Each fallback state shows only the Current Price metric (never a confusing "Suggested Price: same as current" with no stated reason).

**Bug caught during verification:** headline price formatting originally used `.0f` (zero decimals), which for low-price UCI products collapsed a real move like ₹2.73 → ₹3.06 down to a headline reading "₹3 → ₹3" — looked like no change had happened. Fixed to `.2f` throughout the headline and chart marker labels.

**Verified via Streamlit's `AppTest`** (all 5 panel states, not just the happy path):
- `price_change_suggested` (Smartwatch, synthetic): headline `₹3428 → ₹2606: +68% demand, +28.0% revenue`, sensitivity "Medium," two chart markers ✅
- **`already_optimal`** (synthetic optimize_data): headline `Hold current price at ₹100.00`, distinct green message — verified via direct unit test since no live product happened to land exactly at its unconstrained optimum
- `insufficient_data` (10-row Yoga Mat slice): "No signal (insufficient data)," blue ℹ️, single Current Price metric, no phantom suggestion ✅
- `stock_constrained` (ASSORTED COLOUR BIRD ORNAMENT, UCI, stock=19): yellow ⚠️, distinct wording from out-of-stock ✅
- `out_of_stock` (BLACK RECORD COVER FRAME, UCI, stock=0): red 🔴 "no price suggestion possible until restocked" ✅
- **WHITE HANGING HEART T-LIGHT HOLDER reproduction, headline: `₹2.73 → ₹3.06: -37% demand, -29.0% revenue`** — still matches the API/optimizer results exactly after the formatting fix

## NLP chatbot (done)

**Built:** `models/chatbot.py`, class `PricingChatbot` — retrieval only, zero network/LLM calls (verified by grep: no `requests`/`httpx`/`urlopen`/API-key patterns anywhere in the file). Fixed intent set: `explain_price_change`, `best_product`, `stock_risk`, `general_help` (fallback + low-confidence catch-all).

**Shared-logic factoring (step 4):** pulled the explainability panel's pure logic — `classify_panel_state`, `build_headline`, `CONSTRAINT_MESSAGES`, `classify_price_sensitivity` — out of `app/dashboard.py` into a new **`models/explainability.py`** (no Streamlit dependency). Both `dashboard.py` (chart/metrics rendering) and `chatbot.py` (`explain_price_change`) now import the same functions instead of two copies drifting apart. Also centralized the `bound_to_forecaster` monkeypatch context manager (needed because `optimizer.optimize_price()` hardcodes `forecast.py`'s singleton) in `chatbot.py` and had `api/main.py` import it from there instead of keeping its own duplicate — one implementation, two call sites (`/optimize` route and the chatbot).

**Data-access decision:** `PricingChatbot` takes a `DemandForecaster` instance + product list directly (same objects `api/main.py`'s dataset registry already holds) and calls `optimizer.optimize_price()` in-process via `bound_to_forecaster`, rather than having the chatbot make HTTP calls back to its own API. Avoids a self-referential HTTP round-trip from inside the API process, keeps dataset scoping trivial (just pass the right `forecaster`), and reuses the exact pattern `/optimize` already established.

**Intent classifier:** TF-IDF (1-2 grams) + logistic regression, trained on 20 hand-written examples per intent (80 total), varied phrasing not templated repeats. Confidence threshold 0.4 — below it, routes to `general_help` rather than guessing.

**Product extraction:** exact case-insensitive substring match against the product list first, then `difflib.get_close_matches` (cutoff 0.75) over 1/2/3-word n-grams of the message as a fuzzy fallback. Returns `None` (chatbot asks which product) rather than guessing when nothing matches — verified explicitly with a nonsense product name ("why did the flying spaghetti monster price change") — correctly asked for clarification instead of hallucinating a match.

**Verified via `models/chatbot.py`'s `__main__` block** (10 sample questions across all 4 intents, printed with intent + confidence + answer + timing) and additional edge-case probing — not just eyeballed:
- All 10 sample questions classified into the correct intent, product extraction correct in all `explain_price_change` cases
- "No data loaded" case (empty product list) → clear message, no crash
- Unknown/garbage product name → asks for clarification, doesn't hallucinate

**Honest accuracy note:** confidence scores are moderate (0.44–0.71 on in-domain questions) — expected given only 80 tiny training examples and real word overlap across intents (e.g. "price" appears in multiple categories). Classification was **correct on every test case run**, but confidence is not high-margin, so an adversarial or oddly-phrased question could tip below the 0.4 threshold into `general_help` more easily than a production-scale classifier would. One concrete gap found: single-product stock questions ("is the mug in stock?") fall to `general_help` rather than `stock_risk`, since `stock_risk` was trained on cross-product "what's low on stock" framing, not single-item lookups — a real but narrow blind spot, not a fabricated caveat.

**Performance note (also honest, not swept under the rug):** `best_product()`/`stock_risk()` loop over every product via `optimize_price`, parallelized with a `ThreadPoolExecutor` (up to 8 workers) after measuring the naive sequential loop was too slow on a fresh 20-product upload. Warm-cache timing is fast (8 products: ~0.48–0.51s; 20 products: ~1.2s), comfortably under the 2s budget. **Cold-start edge case:** hitting `/chat` directly via the API on a *just-uploaded, never-queried* 20-product dataset (before anything else has trained those models) took 3.89s sequential → 2.44s parallelized — still technically over 2s. This does **not** occur in the actual dashboard flow: the Product Summary table renders before the chat box is reachable and already triggers `/optimize` for every product on page load, so by the time a user can type a question the models are warm — verified end-to-end via `AppTest` (dashboard load → chat query on a fresh upload measured at 1.22s). Flagging the direct-API-cold-start case honestly rather than hiding it; fixing it fully would mean pre-training at `/upload` time, which is outside this task's file scope.

**Wired in:**
- `api/main.py`: real `POST /chat` (`{message}` in, `{answer}` out, optional `?dataset_id=`), replacing the `501` placeholder. `api/test_api.py`'s stale placeholder test updated to assert a real answer instead (kept the suite green — a broken assertion left behind would have been worse than the one-line fix).
- `app/dashboard.py`: "Coming soon" replaced with `st.chat_message`/`st.chat_input`, session-persisted history, calling the new `/chat` endpoint. Verified multi-turn history persists correctly via `AppTest`.

## What-if price slider (done)

**Built:** a manual price slider in the per-product detail view (`app/dashboard.py`), range 0.1×–2.0× current price (deliberately wider than the optimizer's ±30% search range, so it can exercise genuinely extreme prices) with 1%-of-current-price step size, defaulting to current price.

**API changes** (`api/main.py`, both backward-compatible additions, no existing behavior changed):
- `GET /forecast/{product}` gained an optional `price` query param — when present, adds a `what_if: {price, predicted_units}` field to the response (a single-point prediction, independent of the curve's own fixed ±30% range and `steps`). Rejects `price <= 0` with 400.
- `GET /optimize/{product}` now also returns `min_days_of_stock` (the optimizer's constant, `DEFAULT_MIN_DAYS_OF_STOCK`), so the dashboard can evaluate the stock-runway constraint against an arbitrary what-if price without importing optimizer internals into `dashboard.py` (which stays a pure API client, per its original design principle).

**Avoiding API hammering (the debounce requirement):** two-tier resolution, not one call per slider tick —
1. If the what-if price falls inside the already-fetched 20-point `/forecast` curve's range, predicted units are **interpolated client-side** via `np.interp` — zero network calls for the common case (dragging within a sane range).
2. Only when the price falls *outside* that curve (the deliberately-supported extreme-price case) does it fall back to one `/forecast?price=...` call, cached by `st.cache_data` keyed on `(product, dataset_id, price)` so repeated visits to the same value are also free.
3. `st.slider` itself only reruns the script on drag *release*/step-commit (Streamlit's built-in behavior), not per pixel — combined with (1) and (2), dragging through the normal range triggers no API traffic at all.

**Chart:** third marker (orange circle) added to `build_demand_curve_figure`, distinct from current (gray diamond) and suggested (green star).

**Constraint surfacing:** reuses `optimize_data["floor_price"]` (cost floor, same basis regardless of candidate price) and the new `min_days_of_stock` field to flag cost-floor and stock-runway violations at the what-if price, worded distinctly from each other, combinable when both apply.

**Bugs found and fixed during verification** (not just eyeballed — tested via `AppTest` at default/mid-range/both extremes, plus targeted constraint-boundary searches):
- **Precision mismatch at the default slider position:** interpolating the curve at exactly `current_price` gave a slightly different value than the direct model prediction already in `optimize_data` (the 20-point curve's midpoint isn't exactly `current_price` for even step counts, so linear interpolation between its two neighboring points ≠ the tree model's actual prediction there) — showed a confusing "-1% demand, -1.3% revenue" at zero actual change. Fixed by special-casing `abs(what_if_price - current_price) < 0.01` to reuse the already-known `current_predicted_units` directly.
- **Grammar bug in combined violation messages:** `"This price is " + " and ".join(violations)` read as "This price is would deplete stock..." for the stock-runway-only case. Fixed by making each violation phrase a complete clause and joining with `"; "`.

## Polish & edge cases (done)

Tested each of the 7 required scenarios concretely (not just by inspection) against a live API + `AppTest`-driven dashboard; fixed what was found, logged what was already correct.

**1. Empty / near-empty CSV upload** — 0 rows: already rejected with a clear 400 (`"Uploaded file has no data rows"`), verified again here. Near-empty (10 rows, below the 30-row forecast training threshold): falls back to the existing flat-average estimator; verified the *summary table* also renders sanely for it (nonzero forecasted demand, 0% revenue impact — self-explanatory, no fix needed), not just the detail panel (which was already covered in the explainability-panel task).

**2. All-zero-`units_sold` product ("dead product") — design decision, found a real mislabeling bug.** A product with 730 real rows of history but always-zero sales trains a real model (30-row threshold is met) that correctly learns to predict ~0 regardless of price — which the optimizer's flatness check then reports as `insufficient_price_sensitivity_data`, the same label used for genuinely sparse data, with a misleading explanation string ("not enough price-variation history") despite there being plenty of history. **Decision:** distinguish this client-side in `dashboard.py` (no optimizer.py changes, out of scope) — added `is_dead_product()`, which checks `panel_state == "insufficient_data"` *and* `current_predicted_units < 0.01`, and swaps in a distinct 💤 message ("likely delisted, out of season, or not actually being sold") instead of the ℹ️ insufficient-data one. Verified both states render distinctly on real test data (zero-sales Yoga Mat vs. a genuine 10-row sparse slice with nonzero average).

**3. Special characters / very long product names** — built a stress CSV: `Mug & Co. "Deluxe" 50% Off <Sale>`, `Product/Name\With/Slashes`, a 150-character unbroken string, and `O'Brien's Ultra-Premium Widget, Vol. 2 (Special Edition) — Limited Run 2026`. Chart labels, selectbox, what-if slider, and summary table all rendered correctly for all four — no fix needed there. **Chat product-matching had a real gap**, found by testing: short natural references ("the mug", "o'brien's widget") failed to match their long product names, since the existing algorithm only tried exact-substring or query-n-gram-vs-whole-name fuzzy matching — neither handles "a short keyword against one word buried in a long name." Fixed by adding a token-overlap tier to `extract_product()` in `models/chatbot.py`: extracts significant words (≥3 chars, stopword-filtered) from both the message and each product name, picks the product with the most overlapping words. Fixed 3 of 4 cases (mug, widget, slashes); the 150-character unbroken string still doesn't match a short reference to it, since it has no word boundaries to tokenize at all — a genuinely degenerate case with no real-world retail analogue, not chased further. Verified the fix doesn't regress the original `models/chatbot.py` sample questions or the "nonsense product name → ask for clarification" behavior.

**4. Extreme what-if price** — covered above under the what-if slider itself (near-zero → correct cost-floor warning with real numbers; 2× current → API-fallback prediction, no error).

**5. Rapid product switching** — verified via `AppTest` (5 rapid selectbox changes including a repeat): headline always matched the currently-selected product exactly, including returning to a previously-viewed product giving identical values. No fix needed — Streamlit's synchronous full-script-rerun model plus per-product cache keys (`(product, dataset_id)`) inherently prevent stale-data flashing; there's no async state to leak.

**6. API down / unreachable — verified two distinct scenarios, not just the cold-start one already covered earlier:**
   - Cold start (API down before the page ever loads): one clean `st.error` + `st.stop()` right after the first `/products` call, before summary table or chat render at all — a single actionable message, not a blank page or a stack trace.
   - **Mid-session drop** (API up, page loads fully, *then* dies): verified with a genuinely uncached call (dragging the what-if slider to a fresh out-of-range value after killing the API) — only that section shows `st.warning("Could not compute demand at ₹...: Cannot reach the PriceSense API...")`; the already-rendered summary table and chat input remain fully intact and usable. Confirms failures are section-scoped, not page-wide.

**7. Re-upload after already viewing one — found and fixed a real bug.** Product list and cached forecast/optimize data already reset correctly on a new `dataset_id` (verified, no fix needed — cache keys naturally miss). **Chat history did not reset** — old messages about the previous dataset's products would have stayed visible in the transcript after switching datasets, which is actively misleading (a stale answer about a product that may not even exist in the new data, with nothing marking it as outdated). Fixed: `st.session_state.chat_history = []` added to the upload-success branch. Verified via simulated re-upload: history correctly clears, new product list and caption take over immediately. Also proactively hardened the what-if slider's widget `key` to include `dataset_id` (not just product name) — confirmed via a targeted test (two datasets both containing a "Yoga Mat" at very different price scales) that switching datasets resets the slider to the *new* dataset's default rather than carrying over a stale value that could fall outside the new range.

**Known gap, deliberately not fixed (time tradeoff):** the 150-character no-word-boundary product name in edge case #3 still won't match a short chat reference. Fixing it generally (e.g., substring-anywhere fuzzy matching) risks false-positive matches on real, more reasonably-named products elsewhere, and no realistic retail product name looks like this — flagging it here rather than either hiding it or over-engineering around a synthetic worst case.

## Final bug sweep (done)

**Note on process:** this task referenced `CLAUDE.md`'s milestone list and asked to reconcile `SYSTEM_DESIGN.md`'s section 7 with reality. Neither file existed in the repository (confirmed via filesystem search and git history — only `README.md` and this `PROGRESS.md` existed before this pass). Proceeded pragmatically: used this file's own milestone checklist as the functional equivalent of "CLAUDE.md's list" (it already tracks exactly that), and **created `SYSTEM_DESIGN.md` fresh** rather than "updating" a prior version, with section 7 as an accurate API contract reflecting the system as it exists today. Flagging this plainly rather than silently treating a from-scratch document as a reconciliation.

**1. Closed the known gap** (`models/chatbot.py extract_product`): re-investigated before fixing — the specific case flagged in the polish pass (a 150-character no-word-boundary product name) turned out to **already** degrade safely (returns `None` → chatbot asks for clarification), confirmed by re-test. The real, previously-undetected risk was **silent tie-breaking**: two products sharing equally-strong word overlap (e.g. "Red Mountain Bike" / "Blue Mountain Bike", asked about as just "the mountain bike") — the old code picked whichever came first in list order with no signal it was guessing. Fixed both matching tiers (token-overlap and difflib fuzzy n-gram) to detect ties and return `None` (→ ask for clarification) instead of silently picking one. Verified: ambiguous query → `None`; disambiguated query ("the **red** mountain bike") → resolves correctly; all prior sample questions and special-character-name matches unaffected.

**2. Full-codebase pass — found and fixed 4 real bugs, all in `api/main.py`'s input validation** (everything else audited clean):
- **`price=nan` / `price=inf` on `/forecast`** crashed with a raw 500 (sklearn's own input validation rejected the value deep inside `predict_demand`, unhandled). Python's `float()` happily parses `"nan"`/`"inf"`, so FastAPI's automatic type coercion let them through as "valid" floats. Fixed: explicit `np.isfinite` + range check before use, clean 400 instead.
- **The same non-finite values were also silently accepted in CSV uploads** — a `price` column containing `inf` (a plausible real-world artifact, e.g. a spreadsheet `=revenue/units_sold` division by zero) passed validation, then crashed later — not at upload time, but the *next* time that product's model trained, making it hard to trace back to the bad upload. Fixed with the same finiteness check in `validate_and_clean_upload`.
- **Deeper variant of the same bug:** a value that's finite in float64 (e.g. `1e40`) can still overflow to `inf` when scikit-learn internally casts features to `float32` — so a naive `np.isfinite` check alone doesn't fully close the gap. Verified this crashes identically to literal `inf`, then fixed properly with a `MAX_SAFE_MAGNITUDE = np.finfo(np.float32).max` bound, applied consistently to both the `/forecast?price=` param and CSV upload validation.
- **`steps=0` / negative `steps` on `/forecast`** — `steps=0` silently returned an empty curve (`{"curve": []}`, 200 OK) rather than rejecting the request; negative `steps` crashed with a raw 500 (`numpy.linspace` rejects negative sample counts). Neither is reachable from the dashboard's own UI today, but both are reachable via direct API use. Fixed: explicit `steps < 1` → clean 400.
- **Minor defense-in-depth fix** (not a reachable bug today): `PricingChatbot.stock_risk()` lacked the same empty-products guard `best_product()` already had, so a direct call on a chatbot with zero products would hit `ThreadPoolExecutor(max_workers=0)` → `ValueError`. Not reachable via `answer()` (which already guards before dispatching), but inconsistent with the sibling method and a trap for any future caller. Made consistent.

**Everything else audited clean** — no other silent-`None`/empty-result paths without a caller check (every `return None` and every `(data, error)` tuple across `app/dashboard.py`, `models/forecast.py`, `models/chatbot.py`, `api/main.py` traced to a guarding caller); no other raw-exception-reaches-user paths found; no leftover TODO/FIXME/"coming soon"/placeholder text anywhere in `data/`, `models/`, `api/`, or `app/` (grepped clean); `data/generate_synthetic_data.py` and `data/ingest_uci_retail.py` re-run and still produce identical, deterministic output.

**3. Full regression suite re-run after all fixes** — all green: `api/test_api.py` (12/12, including the WHITE HANGING HEART reproduction), `models/test_optimizer.py` (all synthetic + UCI products), `models/chatbot.py`'s 10 sample questions (all correctly classified), and a fresh `AppTest` sweep re-covering the dead-product, sparse-data, and what-if-slider edge cases from the polish milestone.

**4. Known open issues remaining (genuinely can't be fully closed in scope, not overlooked):**
- The `bound_to_forecaster` monkeypatch's concurrency limitation (documented in `SYSTEM_DESIGN.md` §9) — a structural tradeoff from `optimizer.py`'s hardcoded singleton, not a bug fixable without touching `optimizer.py`'s own API.
- Chatbot confidence remains moderate (not high-margin) given the small hand-written training set — every test case still classifies correctly, but this is a soft spot under adversarial phrasing, already disclosed in the NLP chatbot milestone and unchanged by this pass.
- No other known open bugs.

## Dashboard visual redesign (done)

**Context:** a build prompt was handed in describing a card-based, sidebar-nav dashboard (Prisync/Tremor-style visual references, FR/NFR-numbered requirements) that didn't match the existing single-page Streamlit layout. Confirmed with the user before touching anything: **full redesign**, not incremental — the prior single-file `app/dashboard.py` is now fully replaced by this structure.

**Flagged discrepancies with the build prompt** (didn't silently paper over them):
- The prompt describes upload validation errors as HTTP `422`. Our API returns `400` (deliberate, tested, documented in `SYSTEM_DESIGN.md` §7) — kept `400`, since the field-level-error-list *behavior* the prompt actually cares about (NFR-4) is satisfied regardless of the specific status code, and changing a verified, tested API contract to match an external prompt's assumption would be the wrong tradeoff.
- The prompt references "reference screenshots attached" — none were actually present in the message. Built from the text description alone.
- The prompt's nav structure (Dashboard/Products/Chat/Settings) doesn't perfectly line up with its own "Screens to build" section (which only describes 2 screens, bundling upload+summary+detail all into "Dashboard"). Resolved by mapping every described piece of functionality to the nav item it most naturally belongs to: **Dashboard** = portfolio overview (KPIs, revenue-impact chart, summary table), **Products** = the per-product deep dive (selector, demand chart, explanation panel, what-if slider — previously the single-page "Product Detail" section), **Chat** = chat panel + suggested-question chips, **Settings** = dataset upload/management. FR-13's "zero setup" default path is preserved — Dashboard/Products/Chat all work immediately without ever visiting Settings.

**Built — new file structure** (`app/dashboard.py` is now the Dashboard page itself; other pages live in `app/pages/`, Streamlit's native multi-page convention, chosen over the newer `st.navigation` API specifically because `AppTest.switch_page` only supports file-based pages, and testability mattered more than API novelty here):
- `app/api_client.py` — every HTTP call, no rendering (satisfies the "presentation layer only talks to FastAPI" rule literally — this is the *only* file with `requests` calls).
- `app/components.py` — pure rendering functions (`render_kpi_cards`, `render_revenue_chart`, `render_demand_chart`, `render_explanation_panel`, `render_what_if_slider`/`render_what_if_result`, `render_chat_history`, `render_suggested_question_chips`, `render_upload_widget`/`render_upload_confirmation`/`render_upload_errors`, `render_health_banner`) — each takes already-fetched data and renders; no API calls inside any of them.
- `app/common.py` — the one place with genuine cross-page business logic that doesn't belong in "pure rendering": session-state init, `bootstrap()` (health check + dataset resolution, called by every page), `send_chat_message()`.
- `app/pages/1_Products.py`, `2_Chat.py`, `3_Settings.py` — thin page scripts that orchestrate `api_client` + `components`, no logic of their own beyond wiring.
- `.streamlit/config.toml` — indigo primary color (`#4F46E5`) applied via Streamlit's native theming rather than fighting it with CSS everywhere.

**Visual implementation notes:**
- Cards: `st.container(border=True)` + injected CSS (`box-shadow`, `border-radius: 12px`) on Streamlit's bordered-container DOM node — the closest native approximation to the requested card style without a custom component.
- Fallback-state badges styled as **neutral gray** (not error-red/warning-yellow) for `insufficient_data` and `out_of_stock`, per the explicit "not an error state" instruction — verified via rendered HTML, not just visually assumed. `stock_constrained` (a real, actionable constraint) intentionally kept amber/warning, distinct from the two pure-fallback states.
- KPI row + horizontal revenue-impact bar chart (red for negative, indigo for positive) added to the Dashboard page so it reads in ~30 seconds without a dense table (NFR-6 framing) — the detailed table is still present below, but as supporting detail, not the primary view.
- Suggested-question chips are real buttons wired into the same send-message path as `st.chat_input` (clicking one sends immediately, same as typing + Enter) and one chip dynamically references whichever product was last viewed on the Products page (cross-page `session_state.selected_product`), falling back to the first product if none has been viewed yet.
- Chat explicitly avoids anything resembling live-generation UX (no typing/streaming animation) — a plain `st.spinner("Thinking...")` for the round-trip, consistent with FR-12/Section 9's "retrieval-only, not a general chatbot" framing.
- Health banner (`GET /health` pinged on every page load) renders as a persistent `st.error` banner at the top of *every* page when the API is unreachable, verified across all 4 pages — additive to, not a replacement for, the existing per-section graceful-degradation handling underneath it.

**No new dependencies** — still just Streamlit + Plotly + pandas/numpy/requests, all already in `requirements.txt`.

**Verified, not just eyeballed** (`AppTest`, including real cross-page navigation via `switch_page`, plus one live `streamlit run` smoke test):
- All 4 pages load independently with zero exceptions, including with the API down (persistent banner, no blank page, no crash)
- Cross-page state sharing confirmed: selecting a product on Products page updates the corresponding suggested-question chip on the Chat page
- Chip-click → chat round-trip confirmed end-to-end (button click → user bubble → assistant bubble, correct answer)
- Fallback badges confirmed via rendered HTML: `insufficient_data` and `out_of_stock` → gray (`#F3F4F6`/`#374151`); `stock_constrained` → amber (`#FEF3C7`/`#92400E`)
- Malformed-upload field-level error list confirmed extracted and renderable (`get_error_fields`)
- **WHITE HANGING HEART T-LIGHT HOLDER reproduction reconfirmed through the redesigned Products page: ₹2.73 → ₹3.06, -37% demand, -29.0% revenue** — unchanged
- Full regression suite (`api/test_api.py` 12/12, `models/test_optimizer.py`, `models/chatbot.py` sample questions) still green — this was a presentation-layer-only change, no model/API logic touched

## Post-redesign hotfix: `ModuleNotFoundError: No module named 'app'` (done)

**What broke:** the user opened the real dashboard in a browser right after the redesign and hit `ModuleNotFoundError: No module named 'app'` on every page. Root cause: when Streamlit actually runs `app/dashboard.py` (or any `app/pages/*.py` file) as a script, Python adds only **that script's own directory** to `sys.path` — not the project root — so the new `from app.api_client import ...`-style absolute imports (added in the redesign, importing sibling modules within the same `app` package) couldn't resolve `app` as a package at all.

**Why my own verification missed it — logged honestly, not glossed over:** every check I ran before handing this off passed, but for the wrong reasons:
- My `AppTest`-based tests were invoked via `python3 -c "..."` **from the project root**, and Python's `-c` flag adds the current working directory to `sys.path` — which happened to include the project root, masking the bug. That's an artifact of how I ran the test, not something a real `streamlit run` process has.
- My "real streamlit run smoke test" only did `curl` against the app's root URL and checked for `HTTP 200` + grepped the log for tracebacks. `curl` only fetches Streamlit's static shell HTML — the actual Python script only executes once a browser's JS opens a WebSocket session, which `curl` never does. So that check could never have caught this class of bug, and I reported it as verified without noticing the gap in the check itself.

**Fix:** added an explicit, self-contained `sys.path` bootstrap (walks up from `__file__` until it finds `requirements.txt`, the project-root marker, then inserts that directory) at the very top of all 4 entry files — `app/dashboard.py` and all three `app/pages/*.py` files — before their `from app...` imports. Computed from `__file__` alone, so it's correct regardless of cwd or invocation method.

**Verified properly this time**, specifically designed to avoid repeating the same false-positive: ran each entry file directly via `python3 <path>` from an unrelated directory (`/tmp`), which authentically reproduces Streamlit's real sys.path condition (only the script's own directory present, no project-root leak) — all 4 clean, no `ModuleNotFoundError`, no traceback. Re-ran the `AppTest` suite the same way — invoked from `/tmp` instead of the project root, removing the `-c`-flag cwd-leak that hid the bug the first time — all 4 pages (`Dashboard`/`Products`/`Chat`/`Settings`) still zero exceptions under this honest condition. Restarted both servers live for the user afterward.

**Lesson applied going forward:** for Streamlit apps specifically, a clean `curl` response and an `AppTest` run from the project root are not sufficient proof that imports resolve correctly — the meaningful test has to either exercise the file via direct execution from a foreign working directory, or run `AppTest` from one, to match what the real `streamlit run` subprocess actually sees.

## Dashboard polish pass (done)

Quick improvement pass on the redesigned dashboard (`app/components.py`, `app/common.py`, `app/dashboard.py` only):
- **Fixed a real dead-CSS bug**: `.pricesense-chip button` styling was never actually applied — no element in the app carried that class. The only `st.button` usages in the app are the chat suggested-question chips, so broadened the selector to `[data-testid="stButton"] button` (pill-shaped, matches original intent) rather than leaving unreachable CSS in place.
- **Native red/green deltas on KPI cards** (Dashboard page), using `st.metric`'s built-in `delta`/`delta_color` rather than baking the sign into text: "Avg. revenue uplift" now shows a colored delta arrow; "Needs attention" uses `delta_color="inverse"` so a nonzero count reads red. Directly matches the original build prompt's "red/green only for negative/positive deltas" line, which the first redesign pass only partially delivered (colors existed on badges/bars, not on the KPI metrics themselves).
- **Added the "donut" half of the prompt's "bar/donut for portfolio-style summaries" line** — a portfolio-health donut chart (Constraint-label breakdown: None / Already optimal / Stock constrained / Insufficient data / Out of stock) next to the existing revenue-impact bar chart, reinforcing the "understand in 30 seconds" framing with a second, complementary view.
- **Sidebar branding** added once in `common.py`'s shared `init_page()` (title + subtitle + divider), so every page gets consistent branding without duplicating markup across the 4 page files.

**Verified using the corrected method from the import-path hotfix** (not the flawed curl-only check from before): direct `python3 <file>` execution from `/tmp` for all 4 entry files (clean), `AppTest` invoked from `/tmp` (not the project root) across all 4 pages including a dataset with real stock constraints (UCI data: KPI correctly showed "11 of 20" needs-attention with inverse red styling, donut correctly split 9 out-of-stock / 9 none / 2 stock-constrained), full API regression suite still green, live `streamlit run` restarted with a clean startup log.

## Premium SaaS UI pass (done)

A large intelligent-enhancement pass on the existing dashboard — explicitly scoped as improvement, not a rewrite. No backend/model/API files touched; `SYSTEM_DESIGN.md` §7's API contract is unchanged and still accurate.

### 1. What was improved
- **Removed every emoji from the UI** (hard requirement) — page titles, badges, banners, captions, sidebar branding. Verified by scripted Unicode-range scan across `app/`, not just eyeballed.
- **Fixed real bugs found while reading the existing code**, per the "understand before changing" instruction:
  - `PANEL_BADGE_STYLE` was defined but never actually used — the badge color/text logic was duplicated inline in the old explanation panel instead. Consolidated into one source of truth.
  - A backwards dict inversion in the new portfolio donut chart (`{v: k for k, v in {...}.items()}`) caused a `KeyError` — caught by the corrected direct-execution test method, not the flawed one, before it ever reached the user.
  - The new product-comparison table crashed Arrow serialization (mixing numeric and string values in the same transposed column). Fixed by formatting every value to a display string before transposing.
  - Cross-page navigation ("View product" buttons on Dashboard) initially wouldn't have worked — the Products page's selectbox used a different session-state key (`products_page_selector`) than the one being set (`selected_product`). Unified them onto the same key so navigation actually lands on the right product, verified via `AppTest`.
- **Semantic color correctness**: out-of-stock and insufficient-data now render as neutral gray, never red — matches "don't make unavailable data look like a system error." Red is reserved for genuine negative revenue impact. Stock-runway constraints (a real, actionable situation) get amber, not red.
- **Design system consolidation**: one `inject_global_css()` (cards, badges, typography, buttons), one `render_section_header()` for consistent heading hierarchy, one badge system reused everywhere instead of ad hoc styling per page.

### 2. New features (all built on existing data/API, nothing fabricated)
- **Executive KPIs reframed** around decisions, not raw counts: Products analyzed, Price opportunities, Avg. opportunity impact, Stock risk — each backed by real `/optimize` fields, computed in the new `app/insights.py`.
- **Opportunity Summary** — top revenue opportunity, highest demand sensitivity, highest stock risk, each with a "View product" button that navigates to Products with that product pre-selected (real navigation, verified, not a mockup).
- **Portfolio analytics**: added a stock-vs-opportunity scatter chart alongside the existing revenue bar and portfolio-health donut.
- **Priority Products table**: search box + status filter, native column-header sorting (Streamlit's built-in `st.dataframe` behavior — no custom sort code needed), a transparent Priority (High/Medium/Low) column with its logic documented in `app/insights.py:priority_level()`.
- **Product header** on the Products page: current price, recommended price + % change, demand change, revenue impact, stock + runway — all in one prominent row before any explanation.
- **"Why this price"** structured explainability panel: demand response, revenue impact, stock constraint, price sensitivity, and an honest confidence line ("Based on the available historical price and demand patterns" — no fabricated confidence score, since none exists).
- **What-if controls**: Reset-to-current and Reset-to-recommended buttons (via Streamlit `on_click` callbacks, the only safe way to mutate a slider's value before it re-renders), plus a cleaner 3-metric result row. The existing interpolate-first/API-fallback-second logic is untouched.
- **Product comparison**: multiselect two or more products, see them side by side.
- **Dataset insights on Settings**: a permanent "Current dataset" card (name, product count, dataset ID) that was previously only visible transiently right after an upload.
- **Better empty states**: a shared `render_empty_state()` used for "no products," "data unavailable" (API down/error), replacing bare `st.error`/`st.warning` one-liners with a title + explanation + what-to-do-next.
- Chat's "Thinking..." spinner renamed to "Retrieving answer..." — more honest about what's actually happening (a lookup, not generation).

### 3. Deliberately not built (honest, not silently dropped)
- **Demo Mode / guided tour** — the prompt itself said not to create a long tutorial obstructing normal use, and the existing suggested-question chips + opportunity summary already serve "helps a judge understand the system in the first minute" without adding a separate guided-mode surface.
- **Product search as a separate custom widget** — `st.selectbox` already supports type-to-filter search natively, which covers long product names; building a second, redundant search control would have duplicated existing functionality (explicitly against the rules given).
- **Formal saved-state beyond what already exists** — `selected_product` already persists across pages (and now also drives Dashboard-to-Products navigation); no further persistence was identified as adding real value without extra complexity.

### 4. Files changed
- **`app/insights.py`** (new) — pure ranking/scoring/summary logic (`build_summary_row`, `priority_level`, `portfolio_stats`, `top_revenue_opportunity`, `top_demand_sensitivity`, `top_stock_risk`). No Streamlit or HTTP calls, kept separate so `components.py` stays pure-rendering per the architecture's existing rule.
- **`app/components.py`** — design system + new components, emojis removed, dead code fixed, comparison/opportunity/priority-table/product-header/why-this-price/what-if-controls added.
- **`app/common.py`** — sidebar branding, empty-state wiring, emoji removed from `init_page`/spinner text.
- **`app/dashboard.py`** — restructured around KPIs → opportunities → portfolio analytics → priority table; added the `_go_to_product` cross-page navigation callback.
- **`app/pages/1_Products.py`** — restructured around product header → recommendation → why-this-price → demand curve → what-if → comparison.
- **`app/pages/2_Chat.py`**, **`app/pages/3_Settings.py`** — emoji removed, Settings gained the current-dataset card.
- **Not touched**: `api/main.py`, every file in `models/`, every file in `data/` — no backend/model logic changed, per the explicit constraint.

### 5. Architecture decisions
- Kept the existing 3-layer split (`api_client.py` = HTTP, `components.py` = rendering, page files = orchestration) and extended it with a 4th layer (`insights.py` = derived business logic) rather than letting ranking/scoring logic leak into either components or pages.
- Cross-page navigation deliberately unifies the Products page's selectbox key with the shared `selected_product` session-state key (rather than syncing two separate keys), since Streamlit widgets are simplest to control when they *are* the state, not synced to it.
- What-if reset buttons use `on_click` callbacks rather than post-hoc `st.session_state[key] = ...` assignment after slider creation, which Streamlit disallows (would raise an API exception) — this is the only reliable pattern for programmatically moving a slider.

### 6. Testing performed (not claimed without verification)
- Full Unicode emoji scan across `app/` — zero matches.
- Syntax check on all 8 touched/new files.
- **Corrected verification method** (learned from the earlier import-path incident): every entry file run via direct `python3 <path>` execution from `/tmp`, and every `AppTest` run invoked from `/tmp` rather than the project root — this is what actually caught the dict-inversion `KeyError` and the Arrow-serialization bug before they could reach a browser.
- Cross-page navigation verified end-to-end: click "View product" on Dashboard → lands on Products page → selectbox and header both show the correct product.
- What-if reset buttons verified: "Reset to recommended" and "Reset to current" both move the slider (and the result panel) to the correct price.
- Product comparison verified: two-product table renders cleanly with formatted values, no Arrow errors.
- Priority table search and filter verified against real data.
- All 5 panel states re-verified through the redesigned page (dead product, insufficient data, out-of-stock, stock-constrained, unconstrained-optimum) — badges and "why this price" constraint line correctly distinct for each, out-of-stock/insufficient-data confirmed neutral gray not red.
- Special/long/punctuated product names re-verified across Dashboard, Products, and Chat.
- API-down graceful degradation re-verified on all 4 pages, zero exceptions.
- Rapid product switching re-verified — no stale cross-product data.
- Re-upload chat-history reset re-verified.
- **WHITE HANGING HEART T-LIGHT HOLDER reproduction reconfirmed through the redesigned Products page header: ₹2.73 → ₹3.06 (+12.1%), −36.7% demand, −29.0% revenue, 7-day stock runway** — unchanged.
- Full backend regression suite (`api/test_api.py` 12/12, `models/test_optimizer.py`, `models/chatbot.py` sample questions) — unaffected, still green, as expected since no backend files were touched.
- Live `streamlit run` restarted, clean startup log, `curl` 200.

### Known limitations (honest)
- Demand sensitivity shown in the Opportunity Summary is a lightweight proxy (`|demand_change_pct / price_change_pct|` from the already-fetched `/optimize` response), not the same fuller-curve elasticity calculation the Products page's "Why this price" panel uses (`classify_price_sensitivity`, which needs the full demand curve). Computing the precise version for all products on the Dashboard would mean an extra `/forecast` call per product, doubling Dashboard load time — the proxy was chosen deliberately for performance, and is labeled distinctly enough not to be confused with the more precise per-product figure.
- The custom CSS card styling still depends on Streamlit's internal `data-testid` DOM attributes, which could shift in a future Streamlit version — same tradeoff already accepted in the prior redesign, not new to this pass.
- No automated visual/screenshot regression testing exists (no browser-automation tooling installed, and the task said not to add new dependencies) — verification is functional (AppTest + direct execution + live smoke test), not pixel-level.

## Motion, command-center, and intelligence pass (done)

An additive pass on top of the prior premium-SaaS redesign — deliberately scoped to avoid rebuilding anything already covered (opportunity summary, priority table, product header, why-this-price, comparison, dataset info card, empty states, no-emoji rule all already existed from the prior pass and were extended, not rebuilt).

### 1. Experience improvements
The Dashboard now opens with an optional "New here?" CTA pointing straight at the single biggest opportunity, then KPIs, an Opportunity Summary, an auto-generated Insight Feed, portfolio charts (now including a genuine Opportunity Map), and a priority-sorted product table — everything a judge needs to understand the platform's value is visible before any interaction. The Products page now shows a real three-way Current/Recommended/What-if comparison with an honest "outperforms/underperforms the recommendation by X%" readout, not just a single what-if number. Cards lift slightly on hover, buttons give press feedback, new content fades in on load — small, purposeful motion, not decoration.

### 2. Animations added (every one tied to a specific purpose)
- **Card entrance fade + slight upward motion** (`pricesenseFadeInUp` keyframe, 0.25s) on every bordered container and the priority table — helps the eye track what just loaded, especially after a product switch. Respects `prefers-reduced-motion`.
- **Card hover elevation** (shadow deepens on hover) — signals which card is interactive before the user clicks.
- **Button hover shadow + press scale(0.98)** — immediate physical feedback that a click registered.
- **KPI count-up** (0→value over 500ms, eased) on the 4 Dashboard KPIs via `st.components.v1.html` — draws attention to the headline numbers once, on load, not on every rerun of unrelated widgets (each KPI card is its own isolated iframe, so interacting with, say, the priority table's search box does not replay the other cards' count-up, only whichever ones are actually re-rendered).
- **Insight Feed entrance** — each insight row fades in with a slight stagger effect from Streamlit's natural top-to-bottom render order.

### 3. New features
- **AI Pricing Command Center framing**: "New here?" demo CTA at the top of Dashboard, jumping straight to the top opportunity's Products page.
- **Opportunity Map**: bubble chart, X = price sensitivity, Y = revenue impact, bubble size = forecasted demand (all three are real, already-available signals — nothing fabricated). Products without a defined sensitivity reading are explicitly excluded with a stated count, not plotted at a misleading position. Includes a "jump to a product" selector wired to the same cross-page navigation as the rest of the app.
- **Smart Insight Feed**: up to 5 rule-based observations generated from real summary-row data (`app/insights.py:generate_insights()`) — revenue opportunity, out-of-stock, stock-constrained, high/low sensitivity, already-optimal count — each with a "View" button that navigates to the relevant product where one exists.
- **Three-way pricing comparison**: Current / Recommended / What-if shown side by side with prices, predicted demand, and revenue impact vs. current, plus a direct verdict on whether the custom price beats the recommendation — verified with real math (a what-if price returned to current showed the recommendation correctly winning by 21.9%; resetting what-if to the recommended price correctly flipped the verdict to "about the same").
- **Richer, documented priority taxonomy**: `priority_level()` now returns High Priority / Medium Priority / Monitor / Already Optimized / Constrained (was a plain High/Medium/Low), each derived transparently from the optimizer's own panel_state plus revenue-impact magnitude — fully docstringed, and explicitly *not* labeled or treated as any kind of confidence score (there is no model-certainty metric in this system to conflate it with).
- **Priority table upgrades**: added a "Filter by priority" control alongside the existing search/constraint-filter, and the table now defaults to priority-sorted order (High Priority and Constrained products first) instead of arbitrary insertion order — the safe alternative to per-row cell coloring, which Streamlit's `st.dataframe` doesn't support reliably enough to depend on for a demo.

### 4. Files changed
- **`app/insights.py`** — added `generate_insights()`, `bubble_candidates()`, rewrote `priority_level()` with the 5-category taxonomy and a full docstring.
- **`app/components.py`** — added the motion CSS (fade-in keyframe, hover/press transitions, reduced-motion guard), `render_animated_metric()` (the count-up KPI), `render_opportunity_map()`, `render_insight_feed()`, `render_demo_cta()`; rewrote `render_what_if_result()` into the three-way comparison; extended `render_priority_table()` with the priority filter and default sort.
- **`app/dashboard.py`** — wired in the demo CTA, insight feed, and opportunity map alongside the existing KPI/opportunity-summary/portfolio/priority sections.
- **Not touched**: `app/pages/1_Products.py`'s call sites needed no signature changes (the what-if comparison upgrade is internal to the component); `api/main.py`; everything in `models/` and `data/`.

### 5. Architecture decisions
- **`st.components.v1.html` for the count-up, not raw `st.markdown` with a `<script>` tag** — Streamlit strips inline scripts from `st.markdown` for security; `components.v1.html` is the actual first-party supported extension point for embedding real JS, isolated in its own iframe so a bug there can't break the surrounding page. Verified the generated JS is actually correct by extracting it and running it in Node (not just checking Streamlit didn't raise a Python exception) — confirmed correct convergence for positive, negative, and integer-valued targets.
- **CSS-only for card/button motion, not JS** — hover and entrance effects don't need real interactivity, just a `transition`/`@keyframes`, which is both simpler and unaffected by Streamlit's rerun model (no state to lose).
- **Opportunity Map excludes rather than fakes** — products without a meaningful price-change (already-optimal, constrained-to-current-price) have no defined "price sensitivity" by the existing formula; plotting them at x=0 would misrepresent them as "zero sensitivity" when the truth is "not measured this way." Excluding them with a stated count was the honest choice, matching the prompt's own "do not fabricate unavailable metrics" instruction.
- **Priority sort-order over row-coloring** — `st.dataframe` cell/row-level conditional styling is not reliable enough across Streamlit versions to depend on for a live demo; sorting is a native, stable feature that achieves the same "important rows are easy to find" goal without the risk.

### 6. Testing performed
- Full Unicode emoji scan across `app/` — zero matches (including all newly added code).
- Syntax check on every touched file.
- Corrected verification method throughout (direct `python3 <path>` execution from `/tmp`, `AppTest` invoked from `/tmp`) — the same method that caught real bugs in the prior two passes.
- **The count-up JS was extracted and executed in a real Node.js engine** (not just "no Python exception") for positive, negative, and integer cases — confirmed correct final values and no infinite animation loop.
- Demo CTA, Insight Feed "View" buttons, and Opportunity Map's "jump to product" selector all verified to navigate correctly and land on the right product (`AppTest` + `switch_page`).
- Three-way comparison verified with real numbers at both extremes (what-if = current, what-if = recommended) — verdict math checked by hand against the displayed percentages.
- Priority filter and default priority-sort verified against the UCI dataset's real constrained/out-of-stock products (11 of 20 correctly returned when filtering by "Constrained").
- Insight Feed content verified against real data on both the default dataset (revenue + sensitivity insights) and the UCI dataset (out-of-stock, stock-constrained, and revenue insights all correctly generated and capped at 5).
- Re-ran all 4 pages under API-down — still zero exceptions, still degrades gracefully.
- Re-verified special/long/punctuated product names across Dashboard and Products.
- **WHITE HANGING HEART T-LIGHT HOLDER reproduction reconfirmed once more: ₹2.73 → ₹3.06 (+12.1%), −36.7% demand, −29.0% revenue** — unchanged through this pass.
- Full backend regression suite (`api/test_api.py` 12/12, `models/test_optimizer.py`) — unaffected, still green.
- Live `streamlit run` restarted, clean startup log, `curl` 200.

### 7. Known limitations (honest)
- The KPI count-up replays on every full page load/rerun of the Dashboard script (not continuously, but each time the script reruns top-to-bottom, e.g. on any widget interaction elsewhere on the page) — Streamlit remounts the iframe each rerun, and there's no reliable native way to detect "this is the same data as last render, skip the animation" without more state-tracking complexity than a 500ms cosmetic effect justifies. It never spams API calls or recomputes anything — it's a pure animation replay, not a functional issue — but it's not literally "runs once per session" either.
- No true page-transition animation between Dashboard/Products/Chat/Settings — Streamlit's multi-page navigation does a full page reload with no client-side transition hook available without custom JavaScript at the framework level, which was avoided per the explicit "avoid custom JavaScript unless absolutely necessary" and "stability over novelty" instructions.
- Skeleton-style loading placeholders were not added — every data fetch in this app is already fast (sub-100ms cached, low-100s ms cold) per the original 5-second budget, so a skeleton loader would flash and disappear rather than serve a real purpose; the existing `st.spinner` usage (upload, chat) already covers the only genuinely-perceptible waits.
- No pixel-level / visual regression testing — no browser automation tooling is installed and none was added, consistent with "do not introduce unnecessary dependencies." All verification is functional (AppTest, direct execution, a real Node.js check of the animation JS, and a live server smoke test), not screenshot-based.

## Hero positioning, Pricing Health, Guardrails, AI Insights (done)

**Context/stack mismatch, flagged rather than silently worked around:** this task's prompt was written for a React/Tailwind/shadcn/Framer Motion/Recharts SaaS stack. This project is Streamlit + Plotly, established across every prior pass. Rebuilding onto that stack would violate the prompt's own top rule ("do not rebuild from scratch," "continue using the existing chart library," "do not introduce unnecessary UI frameworks"). Every requested outcome was translated into what Streamlit + the existing design system actually supports, not skipped.

**Built this pass** (all additive to the prior three passes, nothing rebuilt):
- **Hero header** (`render_hero_header`): "PriceSense / Your pricing intelligence command center" positioning line, dataset label, and a real "● Pricing engine online/offline" status indicator wired to the actual `/health` check (not a static decoration).
- **Guided pipeline walkthrough** (`render_pipeline_walkthrough`): an optional expander narrating the 6 real pipeline stages (load → analyze → estimate sensitivity → simulate → identify opportunities → recommend) using numbers already computed this run. Deliberately **no artificial delays** — a fake staged-loading sequence would misrepresent computation that's already sub-100ms as taking longer than it does, which conflicts with this prompt's own "do not fake model computation" instruction and with the honesty principle established across every prior pass in this project.
- **Top Opportunity hero card** (`render_top_opportunity_hero`): a visually dominant block with animated current/recommended/demand/revenue metrics and a dynamically generated "why this matters" sentence (`insights.py:explain_opportunity()`) — built from the same real demand/price numbers shown elsewhere, not separately fabricated text. Replaced (not duplicated) the smaller "New here?" CTA from the prior pass, since this fully subsumes its purpose with real substance.
- **"Explain this recommendation"**: the existing "Why this price" panel is now a click-to-expand `st.expander` instead of always-open — same honest content, more interactive framing, zero new claims.
- **Pricing Health Score** (`insights.py:pricing_health_score()`): a transparent 0-100 composite — constraint health (0-40) + price-sensitivity signal quality (0-30) + opportunity magnitude (0-30) — fully docstringed, and explicitly labeled as *not* a confidence score, since no calibrated confidence metric exists in this system. Verified by hand: Running Shoes (32% impact, has signal, unconstrained) → 40+30+30=100; WHITE HANGING HEART (29% impact) → 40+30+29=99; an out-of-stock UCI product → 0+0+0=0. All match the displayed values exactly.
- **Pricing Guardrails** (`insights.py:guardrails_config()`): a **read-only** card showing the pricing engine's actual enforced constants, imported directly from `models/optimizer.py` (±30% search range, 7-day minimum stock runway, 40% margin floor) — not a re-typed copy that could drift, and explicitly not presented as user-editable, since making it genuinely editable would require new API surface this pass didn't add (per the prompt's own "don't pretend a guardrail is enforced when it isn't" instruction, the safest honest choice was read-only display of what's real, not a fake toggle).
- **New "AI Insights" page** (`app/pages/4_AI_Insights.py`): a structured "Business Brief" (biggest opportunity / inventory / pricing / risk / suggested action, all derived from real summary-row data) plus the full insight feed (10 items here vs. 5 on the Dashboard's teaser). Purely additive — Streamlit's `pages/` convention means adding a file doesn't touch any existing route.

**Deliberately not built, with reasoning** (consistent with the prompt's own P0/P1/P2 priority order — everything above is P0/P1-adjacent; these are P2 or structurally risky):
- **Full navigation restructure** into the suggested 7-page IA (Overview/Pricing Simulator/Analytics/Dataset as separate pages) — the existing Dashboard/Products/Chat/Settings/AI Insights structure is already working and tested; restructuring further would be "unnecessarily changing routing architecture," explicitly warned against, for uncertain benefit over the current IA.
- **Scroll-storytelling landing page** — explicitly P2 ("do NOT spend hours on P2 while P0 is unfinished"), and a true scroll-reveal experience isn't reliably buildable in Streamlit without custom JavaScript, which the prompt also says to avoid unless absolutely necessary. The hero positioning copy it asked for was captured on the Dashboard itself instead.
- **Editable guardrails wired to a new API parameter** — would mean adding new backend/API surface for a P2 polish feature; the read-only version delivers the honest core of the ask without that risk.
- **Model confidence display** — no such calibrated metric exists in this system (already disclosed since the chatbot's own confidence work); not fabricated here either, consistent with "do not invent model reasoning."

**Testing performed** (same corrected method established after the import-path incident — direct execution + `AppTest`, both from `/tmp`, never the project root):
- Emoji scan re-run and corrected — the plain arrow character (→) used in the new "Explore recommendation →" button was flagged by an overly-broad first pass of the scan pattern; verified by hand that arrows are standard typography already used throughout the pre-existing `models/explainability.py` headline text (and appear in this very prompt's own suggested CTA copy, "Try PriceSense →"), narrowed the scan to genuinely pictographic Unicode ranges, re-ran clean.
- All 5 pages (Dashboard, Products, Chat, Settings, AI Insights) verified with zero exceptions, both directly executed and via `AppTest`.
- Hero card's "Explore recommendation →" button verified to navigate correctly.
- Pricing Health Score verified by hand against three cases (unconstrained high-opportunity, unconstrained near-flagship-case, and out-of-stock) — all matched the documented formula exactly.
- Guardrails card verified to show the real optimizer constants (±30%, 7 days, 40%), not placeholder text.
- AI Insights page verified against both the default dataset and the UCI dataset (real stock risk: "11 of 20 products are stock-flagged").
- API-down graceful degradation re-verified across all 5 pages.
- Rapid product switching re-verified — no stale cross-product data.
- **WHITE HANGING HEART T-LIGHT HOLDER reproduction reconfirmed once more, unchanged: ₹2.73 → ₹3.06 (+12.1%), −36.7% demand, −29.0% revenue, 7-day runway.**
- Full backend regression suite (`api/test_api.py` 12/12, `models/test_optimizer.py`) — unaffected, still green.
- Live `streamlit run` restarted, clean startup log, `curl` 200.

## Unblocked next

- **Demo prep** — the only remaining milestone. Everything is built, verified end-to-end with a testing method that has now twice caught real bugs before they reached a browser, and the flagship WHITE HANGING HEART T-LIGHT HOLDER reproduction holds at every layer.
