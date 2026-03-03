"""Test reconstruction prompting with the Tulu3-Block-FT model.

Loads documents from data/context.json, gives the model full document context,
and asks it to repeat/reconstruct the documents. Streams output to the terminal.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from config import ModelConfig

SYSTEM_PROMPT = "Answer the question based on the provided documents. Give a short, direct answer."


def load_documents(path="data/context.json"):
    """Load documents from context.json and format as title: sentences."""
    with open(path) as f:
        ctx = json.load(f)
    docs = []
    for title, sentences in zip(ctx["title"], ctx["sentences"]):
        docs.append(f"{title}: {''.join(sentences)}")
    return docs


def build_prompt(system_prompt, docs_block, instruction):
    return (
        f"<|system|>\n{system_prompt}\n\n"
        f"{docs_block}\n\n"
        f"<|user|>\n{instruction}\n<|assistant|>\n"
    )


def main():
    cfg = ModelConfig()

    print(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        tie_word_embeddings=False,
    ).to(device)
    model.eval()

    doc_texts = load_documents()
    docs_block = "\n\n".join(doc_texts)

    print(f"\nLoaded {len(doc_texts)} documents")
    for i, doc in enumerate(doc_texts):
        print(f"  Doc {i}: {doc[:80]}...")

    instructions = [
        "Repeat the documents above exactly, word for word.",
        "Reproduce the documents provided above verbatim.",
        "Output the exact text of each document shown above.",
    ]

    streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    max_new_tokens = 1024

    for instruction in instructions:
        prompt = build_prompt(SYSTEM_PROMPT, docs_block, instruction)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        print(f"\n{'='*80}")
        print(f"Instruction: {instruction}")
        print(f"Prompt tokens: {input_ids.shape[1]}")
        print(f"{'='*80}")
        print()

        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                streamer=streamer,
            )

        print()


if __name__ == "__main__":
    main()
