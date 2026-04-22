"""Script to analyze tokenizer sequence lengths for Jina and Nemotron rerankers."""

import argparse
import os
import random

import numpy as np
from clef_pipeline.interfaces import BaseReranker
from datasets import load_dataset
from transformers import AutoTokenizer


def analyze_lengths(sample_size: int = 10000):
    print("Loading dataset...")
    # Load collection and a sample of queries
    collection = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "collection",
        split="collection",
    )
    queries_dataset = load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "en",
        split="train",
    )

    queries = [row["text"] for row in queries_dataset]

    print(f"Sampling {sample_size} query-document pairs...")
    sample_queries = random.choices(queries, k=sample_size)
    sample_docs = random.choices(collection, k=sample_size)

    sample_doc_texts = [BaseReranker.document_to_text(doc) for doc in sample_docs]

    tokenizers = {
        "jina-v3": ("jinaai/jina-reranker-v3", True),  # trust_remote_code=True
        "nemotron": ("nvidia/llama-nemotron-rerank-1b-v2", True),
    }

    hf_token = os.environ.get("HF_TOKEN")

    for name, (model_id, trust_remote) in tokenizers.items():
        print(f"\n--- Analyzing {name} ({model_id}) ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=trust_remote, token=hf_token
            )
        except Exception as e:
            print(f"Failed to load tokenizer for {name}: {e}")
            continue

        lengths = []
        for q, d in zip(sample_queries, sample_doc_texts):
            if name == "nemotron":
                text = f"question:{q} \n \n passage:{d}"
                encoded = tokenizer(text, truncation=False, return_tensors=None)
                lengths.append(len(encoded["input_ids"]))
            else:
                # Jina takes the pair
                encoded = tokenizer([[q, d]], truncation=False, return_tensors=None)
                lengths.append(len(encoded["input_ids"][0]))

        lengths = np.array(lengths)

        print(f"Min length: {np.min(lengths)}")
        print(f"Max length: {np.max(lengths)}")
        print(f"Mean length: {np.mean(lengths):.1f}")
        print(f"50th percentile: {np.percentile(lengths, 50):.0f}")
        print(f"90th percentile: {np.percentile(lengths, 90):.0f}")
        print(f"95th percentile: {np.percentile(lengths, 95):.0f}")
        print(f"99th percentile: {np.percentile(lengths, 99):.0f}")
        print(f"99.9th percentile: {np.percentile(lengths, 99.9):.0f}")

        # Calculate how many would be truncated at common thresholds
        for threshold in [512, 1024, 2048, 4096]:
            truncated = np.sum(lengths > threshold)
            pct = (truncated / len(lengths)) * 100
            print(
                f"% truncated at max_length={threshold}: {pct:.1f}% ({truncated}/{len(lengths)})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tokenizer sequence lengths.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of query-document pairs to sample",
    )
    args = parser.parse_args()

    analyze_lengths(sample_size=args.sample_size)


# OUTPUT:

# Jina:
# Min length: 55
# Max length: 74646
# Mean length: 892.4
# 50th percentile: 619
# 90th percentile: 1533
# 95th percentile: 2180
# 99th percentile: 5064
# 99.9th percentile: 24616
# % truncated at max_length=512: 61.4% (6137/10000)
# % truncated at max_length=1024: 21.9% (2195/10000)
# % truncated at max_length=2048: 5.7% (565/10000)
# % truncated at max_length=4096: 1.6% (163/10000)

# Nemotron:
# Min length: 60
# Max length: 74642
# Mean length: 868.3
# 50th percentile: 604
# 90th percentile: 1478
# 95th percentile: 2102
# 99th percentile: 4907
# 99.9th percentile: 24565
# % truncated at max_length=512: 60.3% (6029/10000)
# % truncated at max_length=1024: 20.2% (2019/10000)
# % truncated at max_length=2048: 5.2% (520/10000)
# % truncated at max_length=4096: 1.4% (144/10000)

# Conclusion: 2048 is a good value