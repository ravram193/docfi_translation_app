"""
eval.py — ASR Benchmark: ai4bharat IndicConformer vs. Whisper
=============================================================
Compares ASR engines on a Common Voice language test split.

Usage (CLI):
    python3 eval.py --lang hi
    python3 eval.py --lang bn
    python3 eval.py --lang hi --max_samples 25
    python3 eval.py --lang hi --engines ai4b
    python3 eval.py --lang hi --whisper_model medium

Expected directory structure:
    demo_translation_app/
      languages/
        cv-corpus-25.0-2026-03-09/   ← any corpus folder name works
          hi/
            clips/
            test.tsv
      results/
        hi_asr_results.csv           ← written automatically
        hi_asr_summary.json          ← written automatically

Requirements:
    pip install jiwer faster-whisper torchaudio transformers soundfile pandas
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from jiwer import cer, wer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Directory resolution
# ─────────────────────────────────────────────────────────────────────────────

# eval.py lives in demo_translation_app/ — everything is relative to that
BASE_DIR      = Path(__file__).parent
LANGUAGES_DIR = BASE_DIR / "languages"
RESULTS_DIR   = BASE_DIR / "results"


def find_lang_dir(lang: str) -> Path:
    """
    Search inside languages/ for a subfolder matching the language code.
    Works regardless of the corpus version folder name
    (e.g. cv-corpus-25.0-2026-03-09).
    """
    if not LANGUAGES_DIR.exists():
        raise FileNotFoundError(
            f"languages/ directory not found at {LANGUAGES_DIR}\n"
            f"Create it and put your Common Voice corpus folders inside."
        )

    # Direct match: languages/hi/
    direct = LANGUAGES_DIR / lang
    if direct.exists():
        return direct

    # Nested match: languages/<corpus-version>/hi/
    for corpus_dir in sorted(LANGUAGES_DIR.iterdir()):
        if corpus_dir.is_dir():
            candidate = corpus_dir / lang
            if candidate.is_dir():
                log.info("Found language folder: %s", candidate)
                return candidate

    raise FileNotFoundError(
        f"Could not find a '{lang}' folder inside {LANGUAGES_DIR}\n"
        f"Expected structure: languages/<corpus-version>/{lang}/clips/ + test.tsv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClipResult:
    clip_id:         str
    reference:       str
    ai4b_hyp:        str   = ""
    whisper_hyp:     str   = ""
    ai4b_wer:        float = None
    whisper_wer:     float = None
    ai4b_cer:        float = None
    whisper_cer:     float = None
    ai4b_latency:    float = None
    whisper_latency: float = None
    ai4b_error:      str   = ""
    whisper_error:   str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# Whisper (faster-whisper, lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None

def _load_whisper(model_size: str = "large-v3"):
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required. Run: pip install faster-whisper"
            ) from exc
        log.info("Loading Whisper '%s' …", model_size)
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        log.info("Whisper loaded.")
    return _whisper_model


def transcribe_with_whisper(audio_path: str, lang: str, model_size: str = "large-v3") -> str:
    model = _load_whisper(model_size)
    segments, _ = model.transcribe(audio_path, language=lang)
    return " ".join(seg.text.strip() for seg in segments).strip()


# ─────────────────────────────────────────────────────────────────────────────
# ai4bharat (re-uses pipeline.py in the same directory)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_with_ai4b(audio_path: str, lang: str) -> str:
    from app.pipeline import transcribe_with_ai4bharat
    return transcribe_with_ai4bharat(audio_path, source_lang=lang)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_wer(ref: str, hyp: str) -> float:
    return 1.0 if not hyp.strip() else wer(ref, hyp)


def safe_cer(ref: str, hyp: str) -> float:
    return 1.0 if not hyp.strip() else cer(ref, hyp)


def load_test_split(lang_dir: Path, max_samples: int = None) -> pd.DataFrame:
    tsv_path = lang_dir / "test.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"test.tsv not found in {lang_dir}\n"
            f"Make sure the folder contains test.tsv and clips/"
        )
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[["path", "sentence"]].dropna()
    df["full_path"] = df["path"].apply(lambda p: lang_dir / "clips" / p)
    df = df[df["full_path"].apply(lambda p: p.exists())].reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)
    log.info("Loaded %d clips from test split.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    lang_dir: Path,
    lang: str,
    engines: list,
    max_samples: int = None,
    whisper_model_size: str = "large-v3",
) -> pd.DataFrame:

    df = load_test_split(lang_dir, max_samples)
    results = []

    for i, row in df.iterrows():
        audio_path = str(row["full_path"])
        reference  = row["sentence"]
        clip_id    = row["path"]

        log.info("[%d/%d] %s", i + 1, len(df), clip_id)
        result = ClipResult(clip_id=clip_id, reference=reference)

        # ── ai4bharat ──────────────────────────────────────────────────────
        if "ai4b" in engines:
            t0 = time.perf_counter()
            try:
                result.ai4b_hyp     = transcribe_with_ai4b(audio_path, lang)
                result.ai4b_latency = time.perf_counter() - t0
                result.ai4b_wer     = safe_wer(reference, result.ai4b_hyp)
                result.ai4b_cer     = safe_cer(reference, result.ai4b_hyp)
            except Exception as exc:
                result.ai4b_error   = str(exc)
                result.ai4b_latency = time.perf_counter() - t0
                log.warning("ai4b error on %s: %s", clip_id, exc)

        # ── Whisper ────────────────────────────────────────────────────────
        if "whisper" in engines:
            t0 = time.perf_counter()
            try:
                result.whisper_hyp     = transcribe_with_whisper(audio_path, lang, whisper_model_size)
                result.whisper_latency = time.perf_counter() - t0
                result.whisper_wer     = safe_wer(reference, result.whisper_hyp)
                result.whisper_cer     = safe_cer(reference, result.whisper_hyp)
            except Exception as exc:
                result.whisper_error   = str(exc)
                result.whisper_latency = time.perf_counter() - t0
                log.warning("Whisper error on %s: %s", clip_id, exc)

        results.append(result)

    return pd.DataFrame([asdict(r) for r in results])


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, engines: list, lang: str):
    print("\n" + "=" * 60)
    print(f"  ASR BENCHMARK SUMMARY — Common Voice {lang.upper()} (test split)")
    print("=" * 60)
    print(f"  Total clips evaluated : {len(df)}")

    for engine in engines:
        label   = "ai4bharat IndicConformer" if engine == "ai4b" else "Whisper"
        err_col = f"{engine}_error"
        valid   = df[df[err_col] == ""]

        print(f"\n  {label}")
        print(f"    Clips with errors : {df[err_col].astype(bool).sum()}")
        if len(valid):
            print(f"    Mean WER          : {valid[f'{engine}_wer'].mean():.3f}  "
                  f"({valid[f'{engine}_wer'].mean()*100:.1f}%)")
            print(f"    Median WER        : {valid[f'{engine}_wer'].median():.3f}")
            print(f"    Mean CER          : {valid[f'{engine}_cer'].mean():.3f}")
            print(f"    Mean latency (s)  : {valid[f'{engine}_latency'].mean():.2f}")

    if "ai4b" in engines and "whisper" in engines:
        valid_both = df[(df["ai4b_error"] == "") & (df["whisper_error"] == "")]
        if len(valid_both):
            delta  = valid_both["ai4b_wer"].mean() - valid_both["whisper_wer"].mean()
            winner = "ai4bharat" if delta < 0 else "Whisper"
            print(f"\n  Winner (lower WER) : {winner} "
                  f"(Δ = {abs(delta)*100:.1f} pp on {len(valid_both)} shared clips)")

    print("=" * 60 + "\n")


def save_outputs(df: pd.DataFrame, engines: list, lang: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path  = RESULTS_DIR / f"{lang}_asr_results.csv"
    json_path = RESULTS_DIR / f"{lang}_asr_summary.json"

    df.to_csv(csv_path, index=False)
    log.info("Per-clip results → %s", csv_path)

    summary = {"lang": lang}
    for engine in engines:
        valid = df[df[f"{engine}_error"] == ""]
        summary[engine] = {
            "n_clips":      len(df),
            "n_errors":     int(df[f"{engine}_error"].astype(bool).sum()),
            "mean_wer":     round(float(valid[f"{engine}_wer"].mean()),     4) if len(valid) else None,
            "median_wer":   round(float(valid[f"{engine}_wer"].median()),   4) if len(valid) else None,
            "mean_cer":     round(float(valid[f"{engine}_cer"].mean()),     4) if len(valid) else None,
            "mean_latency": round(float(valid[f"{engine}_latency"].mean()), 3) if len(valid) else None,
        }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary          → %s", json_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark ai4bharat IndicConformer vs Whisper on Common Voice"
    )
    p.add_argument(
        "--lang", required=True,
        help="ISO language code to evaluate, e.g. hi, bn, ta, te. "
             "Must match a subfolder inside languages/<corpus>/<lang>/"
    )
    p.add_argument(
        "--engines", nargs="+", choices=["ai4b", "whisper"],
        default=["ai4b", "whisper"],
        help="Engines to benchmark (default: both)"
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap clips for a quick smoke-test, e.g. --max_samples 25"
    )
    p.add_argument(
        "--whisper_model", default="large-v3",
        help="Whisper model size: tiny / base / small / medium / large-v3 (default: large-v3)"
    )
    return p.parse_args()


def main():
    args    = parse_args()
    lang    = args.lang
    engines = args.engines

    lang_dir = find_lang_dir(lang)

    df = run_evaluation(
        lang_dir           = lang_dir,
        lang               = lang,
        engines            = engines,
        max_samples        = args.max_samples,
        whisper_model_size = args.whisper_model,
    )

    save_outputs(df, engines, lang)
    print_summary(df, engines, lang)


if __name__ == "__main__":
    main()