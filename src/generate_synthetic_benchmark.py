from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --- Knowledge Base ---
# A collection of documents that can act as candidates.

KB_DOCS = {
    # Billing & Accounts
    "billing-reset-pw": "To reset your password, navigate to Account > Security and click 'Reset Password'. You will receive an email with further instructions.",
    "billing-invoices": "You can find all past and current invoices under the Billing > Invoices section of your dashboard.",
    "billing-add-payment": "To add a new payment method, go to Billing > Payment Methods and select 'Add New Card'.",
    "billing-cancel-sub": "To cancel your subscription, please visit the Billing > Subscription page and follow the cancellation prompts. Note that cancellations are effective at the end of the current billing cycle.",

    # API & Integration
    "api-keys": "API keys can be generated, revoked, and managed in the Developer Settings > API Keys section of your account.",
    "api-rate-limits": "Our API has a rate limit of 1,000 requests per minute per user. Exceeding this limit will result in a 429 Too Many Requests error.",
    "api-python-sdk": "The official Python SDK is available on PyPI. Install it using 'pip install our-sdk' and refer to the official documentation for usage examples.",
    "api-webhooks": "To configure webhooks, go to Developer Settings > Webhooks and provide your endpoint URL. You can subscribe to events like 'payment.succeeded' or 'user.created'.",

    # General / Irrelevant
    "general-status-page": "You can check the current status of all our services on our public status page at status.example.com.",
    "general-careers": "We are always looking for talented individuals. Visit our careers page to see open positions.",
    "general-blog": "Our engineering team regularly posts updates and technical deep-dives on our official company blog.",
    "general-tos": "The Terms of Service can be reviewed at any time by visiting the 'Legal' link in the footer of our website.",
}

# --- Query Templates ---
# Templates to generate realistic user queries.

QUERY_TEMPLATES = [
    ("How do I reset my password?", "billing-reset-pw"),
    ("Where are my invoices located?", "billing-invoices"),
    ("How can I add a credit card?", "billing-add-payment"),
    ("I need to cancel my subscription.", "billing-cancel-sub"),
    ("Where do I get API keys?", "api-keys"),
    ("What are the API rate limits?", "api-rate-limits"),
    ("Is there a Python library for the API?", "api-python-sdk"),
    ("How to set up webhooks?", "api-webhooks"),
    ("Where can I check if the service is down?", "general-status-page"),
    ("Are you hiring?", "general-careers"),
]


def generate_synthetic_data(
    num_queries: int,
    candidates_per_query: int,
    output_path: Path,
) -> None:
    """Generates a synthetic JSONL dataset for reranker training."""
    all_doc_keys = list(KB_DOCS.keys())
    records = []

    for i in range(num_queries):
        query_text, golden_doc_key = random.choice(QUERY_TEMPLATES)

        # --- Assemble Candidates ---
        candidates = []
        # 1. Add the perfect match
        candidates.append(KB_DOCS[golden_doc_key])

        # 2. Add some partially relevant docs (from the same category)
        category = golden_doc_key.split("-")[0]
        partial_keys = [k for k in all_doc_keys if k.startswith(category) and k != golden_doc_key]
        num_partial = min(len(partial_keys), 3)  # Add up to 3 partial matches
        partial_to_add = random.sample(partial_keys, num_partial)
        for key in partial_to_add:
            candidates.append(KB_DOCS[key])

        # 3. Fill the rest with irrelevant docs
        irrelevant_keys = [k for k in all_doc_keys if not k.startswith(category)]
        num_needed = candidates_per_query - len(candidates)
        if num_needed > 0:
            irrelevant_to_add = random.sample(irrelevant_keys, min(num_needed, len(irrelevant_keys)))
            for key in irrelevant_to_add:
                candidates.append(KB_DOCS[key])

        # Shuffle candidates and create ground-truth ranking
        ground_truth_ranking = list(range(1 + num_partial))
        random.shuffle(candidates)

        # Find the new indices of the relevant docs
        oracle_ranking = []
        golden_doc_text = KB_DOCS[golden_doc_key]
        partial_doc_texts = [KB_DOCS[k] for k in partial_to_add]

        # Golden doc is always first
        oracle_ranking.append(candidates.index(golden_doc_text))

        # Partial docs follow
        partial_indices = [candidates.index(t) for t in partial_doc_texts]
        random.shuffle(partial_indices)
        oracle_ranking.extend(partial_indices)
        
        # Add remaining docs in random order
        other_indices = [i for i in range(len(candidates)) if i not in oracle_ranking]
        random.shuffle(other_indices)
        oracle_ranking.extend(other_indices)

        # --- Introduce Noise ---
        # Swap two elements to simulate an imperfect teacher
        if len(oracle_ranking) > 2:
            idx1, idx2 = random.sample(range(1, len(oracle_ranking)), 2)
            oracle_ranking[idx1], oracle_ranking[idx2] = oracle_ranking[idx2], oracle_ranking[idx1]

        records.append({
            "query": query_text,
            "candidates": candidates,
            "oracle_ranking": oracle_ranking,
        })

    # --- Write to File ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Generated {len(records)} synthetic records at {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic benchmark for reranker evaluation.")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--candidates-per-query", type=int, default=40)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    generate_synthetic_data(
        num_queries=args.num_queries,
        candidates_per_query=args.candidates_per_query,
        output_path=Path(args.output),
    )

if __name__ == "__main__":
    main()
