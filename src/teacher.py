from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
)

from .config import TeacherConfig
from .utils import console

RANK_RE = re.compile(r"\[(.*?)\]")


def parse_args() -> tuple[TeacherConfig, Path, Path, str, str]:
    parser = argparse.ArgumentParser(description="Generate oracle rankings using an open-source LLM")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="JSONL input with query and candidates")
    parser.add_argument("--output", type=str, required=True, help="Destination JSONL with oracle rankings")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--mode",
        type=str,
        default="causal-lm",
        choices=["causal-lm", "cross-encoder", "lexical"],
    )
    args = parser.parse_args()
    cfg = TeacherConfig(
        model_id=args.model_id,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
    return cfg, Path(args.input), Path(args.output), args.device, args.mode


def format_prompt(system_prompt: str, query: str, candidates: List[str]) -> str:
    bullet_points = "\n".join(f"{idx}. {text}" for idx, text in enumerate(candidates))
    return (
        f"{system_prompt}\n\n"
        f"Query: {query}\n"
        f"Candidates:\n{bullet_points}\n"
        "Return a JSON list of indices sorted from most to least relevant."
    )


def extract_ranking(output: str, num_candidates: int) -> List[int]:
    match = RANK_RE.search(output)
    if not match:
        return list(range(num_candidates))
    try:
        values = json.loads(f"[{match.group(1)}]")
        ranking = [int(v) for v in values if 0 <= int(v) < num_candidates]
    except json.JSONDecodeError:
        ranking = []
    if not ranking:
        ranking = list(range(num_candidates))
    return ranking


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # auto fallback
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    teacher_cfg, input_path, output_path, device_arg, mode = parse_args()
    if mode == "lexical":
        device = torch.device("cpu")
        console.print("Using lexical TF-IDF teacher (no model load)")
        tokenizer = None
        model = None
        gen_cfg = None
    else:
        device = resolve_device(device_arg)

        console.print(f"Loading teacher model {teacher_cfg.model_id} ({mode}) on {device}")
        torch_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(teacher_cfg.model_id)

        if mode == "causal-lm":
            model = AutoModelForCausalLM.from_pretrained(
                teacher_cfg.model_id,
                torch_dtype=torch_dtype,
                device_map={"": device.type if device.type in {"cuda", "mps"} else "cpu"},
            )
            gen_cfg = GenerationConfig(
                max_new_tokens=128,
                temperature=teacher_cfg.temperature,
                top_p=teacher_cfg.top_p,
                do_sample=True,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                teacher_cfg.model_id,
                torch_dtype=torch_dtype,
                device_map={"": device.type if device.type in {"cuda", "mps"} else "cpu"},
            )
            gen_cfg = None

    samples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for sample in samples:
            if mode == "lexical":
                vectorizer = TfidfVectorizer()
                docs = [sample["query"]] + sample["candidates"]
                tfidf = vectorizer.fit_transform(docs).toarray()
                query_vec = tfidf[0]
                cand_vecs = tfidf[1:]
                scores = cand_vecs @ query_vec
                ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            elif mode == "causal-lm":
                prompt = format_prompt(teacher_cfg.system_prompt, sample["query"], sample["candidates"])
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(**inputs, generation_config=gen_cfg)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                ranking = extract_ranking(decoded, len(sample["candidates"]))
            else:
                scores = []
                for passage in sample["candidates"]:
                    encoded = tokenizer(
                        sample["query"],
                        passage,
                        padding=True,
                        truncation=True,
                        max_length=teacher_cfg.max_length,
                        return_tensors="pt",
                    )
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    with torch.no_grad():
                        logits = model(**encoded).logits
                    score = logits.squeeze().float().item()
                    scores.append(score)
                ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            sample["oracle_ranking"] = ranking
            fout.write(json.dumps(sample) + "\n")
            console.log(f"Processed query: {sample['query'][:40]}...")

    console.print(f"Saved oracle rankings to {output_path}")


if __name__ == "__main__":
    main()
