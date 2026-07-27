"""
convert.py
----------
Converts a Hugging Face model checkpoint to GGUF format and imports it
into Ollama so you can run it with `ollama run`.

Pipeline:
  1. Check that llama.cpp exists (clone it if not)
  2. Install llama.cpp Python requirements
  3. Convert HF model → GGUF (float16)
  4. (Optional) Quantize GGUF → smaller Q4_K_M file
  5. Write an Ollama Modelfile
  6. Import the model into Ollama via `ollama create`

Usage:
  python convert.py                                    # uses defaults
  python convert.py --model ../022-train-smollm/google-ads-smollm
  python convert.py --model ../022-train-smollm/google-ads-smollm --quantize
  python convert.py --model ../022-train-smollm/google-ads-smollm --name my-ads-model
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = "../022-train-smollm/google-ads-smollm"
DEFAULT_NAME      = "google-ads-smollm"
LLAMA_CPP_DIR     = Path("llama.cpp")
LLAMA_CPP_REPO    = "https://github.com/ggerganov/llama.cpp"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a shell command, stream output, and raise on failure."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\n[error] command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def step1_ensure_llama_cpp():
    """Clone llama.cpp if it is not already present."""
    if LLAMA_CPP_DIR.exists():
        print(f"[1/6] llama.cpp already present at '{LLAMA_CPP_DIR}'")
        return

    print("[1/6] cloning llama.cpp ...")
    run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(LLAMA_CPP_DIR)])


def step2_install_requirements():
    """Install Python packages required by llama.cpp conversion scripts."""
    req_file = LLAMA_CPP_DIR / "requirements.txt"
    if not req_file.exists():
        print("[2/6] no requirements.txt found in llama.cpp — skipping")
        return

    print("[2/6] installing llama.cpp Python requirements ...")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)])


def step3_convert_to_gguf(model_dir: Path, output_file: Path):
    """Convert a HF checkpoint to GGUF (float16)."""
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Older llama.cpp used a different name
        convert_script = LLAMA_CPP_DIR / "convert.py"
    if not convert_script.exists():
        print("[error] could not find convert_hf_to_gguf.py in llama.cpp")
        sys.exit(1)

    print(f"[3/6] converting {model_dir} → {output_file} (f16) ...")
    run([
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(output_file),
        "--outtype", "f16",
    ])
    print(f"[3/6] GGUF written: {output_file} ({output_file.stat().st_size / 1e9:.2f} GB)")


def step4_quantize(input_file: Path, output_file: Path, quant_type: str = "Q4_K_M"):
    """Quantize the GGUF to a smaller format using llama-quantize."""
    quantize_bin = LLAMA_CPP_DIR / "llama-quantize"
    if not quantize_bin.exists():
        # Try building it
        print("[4/6] llama-quantize not found — attempting to build llama.cpp ...")
        run(["cmake", "-B", "build", "-S", str(LLAMA_CPP_DIR)])
        run(["cmake", "--build", "build", "--config", "Release", "-j"])
        quantize_bin = Path("build") / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            print("[4/6] could not build llama-quantize — skipping quantization")
            return input_file

    print(f"[4/6] quantizing {input_file} → {output_file} ({quant_type}) ...")
    run([str(quantize_bin), str(input_file), str(output_file), quant_type])
    print(f"[4/6] quantized: {output_file} ({output_file.stat().st_size / 1e9:.2f} GB)")
    return output_file


def step5_write_modelfile(gguf_path: Path, model_name: str, system_prompt: str):
    """Write an Ollama Modelfile that points at the GGUF."""
    modelfile_path = Path(f"Modelfile.{model_name}")
    content = f"""\
FROM {gguf_path.resolve()}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
"""
    modelfile_path.write_text(content)
    print(f"[5/6] Modelfile written: {modelfile_path}")
    return modelfile_path


def step6_import_to_ollama(modelfile_path: Path, model_name: str):
    """Import the model into Ollama using `ollama create`."""
    print(f"[6/6] importing into Ollama as '{model_name}' ...")
    run(["ollama", "create", model_name, "-f", str(modelfile_path)])
    print(f"\n[done] run your model with:\n\n    ollama run {model_name}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Export HF model to Ollama")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_DIR,
        help="Path to the Hugging Face model directory (default: %(default)s)",
    )
    p.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="Ollama model name (default: %(default)s)",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize to Q4_K_M after conversion (smaller file, faster inference)",
    )
    p.add_argument(
        "--quant-type",
        default="Q4_K_M",
        help="Quantization type (default: %(default)s). Options: Q4_K_M, Q5_K_M, Q8_0",
    )
    p.add_argument(
        "--system-prompt",
        default=(
            "You are an expert Google Ads assistant. "
            "You help with GAQL queries, campaign setup, diagnostics, and the Google Ads API."
        ),
        help="System prompt embedded in the Modelfile",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        print(f"[error] model directory not found: {model_dir}")
        print("Train the model first with:\n  cd ../022-train-smollm && python main.py --skip-eval")
        sys.exit(1)

    gguf_f16  = Path(f"{args.name}-f16.gguf")
    gguf_final = Path(f"{args.name}-{args.quant_type}.gguf") if args.quantize else gguf_f16

    step1_ensure_llama_cpp()
    step2_install_requirements()
    step3_convert_to_gguf(model_dir, gguf_f16)

    if args.quantize:
        step4_quantize(gguf_f16, gguf_final, args.quant_type)
    else:
        print("[4/6] skipping quantization (pass --quantize to enable)")

    modelfile = step5_write_modelfile(gguf_final, args.name, args.system_prompt)
    step6_import_to_ollama(modelfile, args.name)


if __name__ == "__main__":
    main()
