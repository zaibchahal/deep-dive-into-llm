"""
evaluate.py
-----------
Compare the model's answers before and after continued pre-training.

Usage (standalone):

    # Before training
    python evaluate.py --model HuggingFaceTB/SmolLM2-360M --label "base"

    # After training
    python evaluate.py --model google-ads-smollm --label "fine-tuned"

Or call run_eval() directly from main.py.
"""

import argparse
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import EVAL_PROMPTS, MAX_NEW_TOKENS, MODEL_NAME


def run_eval(
    model,
    tokenizer,
    prompts: list[str] | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    label: str = "",
) -> dict[str, str]:
    """
    Run inference on each prompt and return a dict of {prompt: response}.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    prompts : list[str], optional
        Defaults to EVAL_PROMPTS.
    max_new_tokens : int
    label : str
        Printed as a header so you can tell before/after apart.

    Returns
    -------
    dict[str, str]
    """
    if prompts is None:
        prompts = EVAL_PROMPTS

    model.eval()
    results: dict[str, str] = {}

    separator = "─" * 60
    header = f"  [{label}]" if label else ""
    print(f"\n{separator}")
    print(f"EVALUATION{header}")
    print(separator)

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # greedy — deterministic for comparison
                temperature=1.0,
                repetition_penalty=1.1,
            )
            # Decode only the newly generated tokens (strip the prompt)
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            results[prompt] = response

            print(f"\nQ: {prompt}")
            print(f"A: {textwrap.fill(response, width=72, subsequent_indent='   ')}")

    print(f"\n{separator}\n")
    return results


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a SmolLM2 checkpoint")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="HF model ID or local path",
    )
    parser.add_argument("--label", default="", help="Label printed in the header")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    print(f"[evaluate] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    run_eval(mdl, tok, max_new_tokens=args.max_new_tokens, label=args.label)
