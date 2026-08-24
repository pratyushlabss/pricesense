# Retrieval-only chatbot over already-computed forecast/optimizer output.
# No network calls, no LLM — TF-IDF + logistic regression intent classifier,
# difflib fuzzy product matching, templated answers.

import difflib
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import models.optimizer as optimizer_module
from models.explainability import CONSTRAINT_MESSAGES, build_headline, classify_panel_state

INTENT_EXPLAIN_PRICE_CHANGE = "explain_price_change"
INTENT_BEST_PRODUCT = "best_product"
INTENT_STOCK_RISK = "stock_risk"
INTENT_GENERAL_HELP = "general_help"
CONFIDENCE_THRESHOLD = 0.4

TRAINING_EXAMPLES = {
    INTENT_EXPLAIN_PRICE_CHANGE: [
        "why did the price change",
        "why did the smartwatch's price change",
        "why is the mug priced at 449",
        "explain the price change for the smartwatch",
        "why ₹449 for the mug",
        "what's the reasoning behind this price",
        "why did you suggest this price",
        "explain this price suggestion",
        "why is the price different from before",
        "reason for the new price on wireless earbuds",
        "why should I change the price of yoga mat",
        "what's driving the price change for air fryer",
        "explain why running shoes price moved",
        "why does office chair have a different suggested price",
        "can you explain the pricing logic for bluetooth speaker",
        "why is this price recommended",
        "what justifies this price change",
        "why not keep the same price",
        "explain the rationale behind the suggested price",
        "why did laptop backpack price go up",
    ],
    INTENT_BEST_PRODUCT: [
        "which product has the best revenue uplift",
        "what should I focus on",
        "which product should I prioritize",
        "what's the top opportunity right now",
        "which item has the highest revenue gain",
        "what product gives the biggest revenue boost",
        "show me the best pricing opportunity",
        "which product should I change the price of first",
        "what's the most impactful price change",
        "rank products by revenue impact",
        "which product will make the most money if repriced",
        "what's my best bet for a price change",
        "top product for revenue growth",
        "which item has the largest projected gain",
        "what should I prioritize for pricing",
        "which product offers the biggest opportunity",
        "best product to reprice today",
        "highest revenue impact product",
        "where's the biggest revenue win",
        "what product should get a new price first",
    ],
    INTENT_STOCK_RISK: [
        "which products are low on stock",
        "what's out of stock",
        "any stockout risk",
        "show me products running low on inventory",
        "which items need restocking",
        "what's at risk of selling out",
        "which products have a stock problem",
        "list out of stock items",
        "what needs to be restocked soon",
        "are any products low on inventory",
        "which products might run out",
        "stock levels that are concerning",
        "show inventory risk",
        "what's close to selling out",
        "which items are running low",
        "any products about to stock out",
        "check stock risk across products",
        "what's our stockout exposure",
        "low inventory alert",
        "which products need urgent restocking",
    ],
    INTENT_GENERAL_HELP: [
        "hello",
        "hi there",
        "what can you do",
        "help me",
        "what can I ask you",
        "how does this work",
        "what is this tool",
        "tell me about pricesense",
        "what kind of questions can I ask",
        "I need help",
        "how do I use this chatbot",
        "what commands do you support",
        "who are you",
        "good morning",
        "thanks",
        "can you help me with pricing",
        "what's this dashboard about",
        "explain what pricesense does",
        "give me an overview",
        "what topics can you cover",
    ],
}


def build_intent_classifier():
    texts, labels = [], []
    for intent, examples in TRAINING_EXAMPLES.items():
        texts.extend(examples)
        labels.extend([intent] * len(examples))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


_intent_classifier = None


def get_intent_classifier():
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = build_intent_classifier()
    return _intent_classifier


def classify_intent(message):
    classifier = get_intent_classifier()
    probabilities = classifier.predict_proba([message])[0]
    best_index = probabilities.argmax()
    best_intent = classifier.classes_[best_index]
    confidence = float(probabilities[best_index])
    if confidence < CONFIDENCE_THRESHOLD:
        return INTENT_GENERAL_HELP, confidence
    return best_intent, confidence


EXTRACTION_STOPWORDS = {
    "the", "and", "for", "why", "did", "does", "do", "this", "that", "what", "which",
    "should", "please", "tell", "your", "about", "with", "from", "into", "are", "was",
    "has", "not", "you", "can", "get", "its", "how", "who", "top", "new", "old", "any",
    "all", "see", "let", "out", "low",
    "price", "prices", "priced", "pricing", "change", "changed", "changing",
    "product", "products", "item", "items", "revenue", "stock", "demand", "cost",
    "focus", "best", "explain", "reasoning", "different", "reason", "justif",
}


def significant_words(text):
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) >= 3 and word not in EXTRACTION_STOPWORDS}


def extract_product(message, products):
    message_lower = message.lower()
    for product in products:
        if product.lower() in message_lower:
            return product

    message_words = significant_words(message)
    if message_words:
        overlap_scores = [(len(significant_words(product) & message_words), product) for product in products]
        best_overlap = max(score for score, _ in overlap_scores)
        if best_overlap > 0:
            top_matches = [product for score, product in overlap_scores if score == best_overlap]
            if len(top_matches) == 1:
                return top_matches[0]
            return None  # ambiguous — multiple products tie on overlap, don't guess

    words = message_lower.replace("?", "").replace(",", "").split()
    candidates = []
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            candidates.append(" ".join(words[i:i + n]))

    product_lookup = {product.lower(): product for product in products}
    for candidate in candidates:
        matches = difflib.get_close_matches(candidate, product_lookup.keys(), n=2, cutoff=0.75)
        if len(matches) == 1:
            return product_lookup[matches[0]]
        # 0 matches: this candidate n-gram found nothing, try the next one.
        # 2 matches: this candidate is ambiguous between two products; don't guess,
        # but a different (e.g. longer) candidate later in the loop might still resolve cleanly.

    return None


@contextmanager
def bound_to_forecaster(forecaster):
    original_get_forecaster = optimizer_module.get_forecaster
    optimizer_module.get_forecaster = lambda: forecaster
    try:
        yield
    finally:
        optimizer_module.get_forecaster = original_get_forecaster


class PricingChatbot:
    def __init__(self, forecaster, products=None):
        self.forecaster = forecaster
        self.products = products if products is not None else forecaster.list_products()

    def _optimize(self, product):
        with bound_to_forecaster(self.forecaster):
            return optimizer_module.optimize_price(product)

    def _optimize_all(self):
        with bound_to_forecaster(self.forecaster):
            worker_count = min(8, len(self.products))
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                results = list(pool.map(optimizer_module.optimize_price, self.products))
        return list(zip(self.products, results))

    def explain_price_change(self, product):
        if product is None:
            example = self.products[0] if self.products else "a product"
            return f"Which product did you mean? Try naming one, for example: \"why did {example}'s price change?\""
        if product not in self.products:
            sample = ", ".join(self.products[:5])
            return f"I don't have '{product}' in the current dataset. Try one of: {sample}"

        optimize_data = self._optimize(product)
        panel_state = classify_panel_state(optimize_data)
        headline = build_headline(optimize_data, panel_state)
        constraint_line = CONSTRAINT_MESSAGES[panel_state]
        return f"**{headline}**\n\n{constraint_line}\n\n_{optimize_data['explanation']}_"

    def best_product(self):
        if not self.products:
            return "No products are currently loaded."

        ranked = [(product, data["projected_revenue_change_pct"]) for product, data in self._optimize_all()]
        ranked.sort(key=lambda item: item[1], reverse=True)
        top_product, top_change_pct = ranked[0]
        return (
            f"**{top_product}** has the best projected revenue uplift: {top_change_pct:+.1f}%. "
            f"Ask \"why did {top_product}'s price change?\" for the full reasoning."
        )

    def stock_risk(self):
        if not self.products:
            return "No products are currently loaded."

        at_risk = []
        for product, data in self._optimize_all():
            panel_state = classify_panel_state(data)
            if panel_state == "out_of_stock":
                at_risk.append(f"{product} (out of stock)")
            elif panel_state == "stock_constrained":
                at_risk.append(f"{product} (low stock runway)")

        if not at_risk:
            return "No products are currently flagged for stock risk."
        return "Products at stock risk:\n" + "\n".join(f"- {entry}" for entry in at_risk)

    def general_help(self):
        return (
            "I can answer questions about your pricing data:\n"
            "- **Why a price changed** — \"why did the smartwatch price change?\"\n"
            "- **Best opportunity** — \"which product has the best revenue uplift?\"\n"
            "- **Stock risk** — \"what's low on stock?\"\n"
            "Try asking one of these."
        )

    def answer(self, message):
        if not self.products:
            return "No data is loaded yet — upload a CSV or wait for the demo dataset to load."

        intent, _confidence = classify_intent(message)
        if intent == INTENT_EXPLAIN_PRICE_CHANGE:
            return self.explain_price_change(extract_product(message, self.products))
        if intent == INTENT_BEST_PRODUCT:
            return self.best_product()
        if intent == INTENT_STOCK_RISK:
            return self.stock_risk()
        return self.general_help()


if __name__ == "__main__":
    import time
    from models.forecast import get_forecaster

    forecaster = get_forecaster()
    chatbot = PricingChatbot(forecaster)

    sample_questions = [
        "why did the smartwatch price change?",
        "why is wireless earbuds priced differently?",
        "explain the price change for yoga mat",
        "what should I focus on?",
        "which product has the best revenue uplift?",
        "top product for revenue growth",
        "what's out of stock?",
        "which products are low on stock?",
        "hello, what can you do?",
        "how does this tool work?",
    ]

    for question in sample_questions:
        start = time.time()
        intent, confidence = classify_intent(question)
        answer = chatbot.answer(question)
        elapsed = time.time() - start
        print(f"Q: {question}")
        print(f"   intent={intent} (confidence={confidence:.2f}), {elapsed:.3f}s")
        print(f"   A: {answer}")
        print()
