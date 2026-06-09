"""
pipeline.py  —  Transcript Processing Pipeline
===============================================
Handles:
  • transcribe_with_ai4bharat()  – ai4bharat IndicConformer ASR (South Asian languages)
  • clean_with_bedrock()         – AWS Bedrock (Claude Sonnet) transcript cleaning
  • translate_with_aws()         – AWS Translate / Bedrock Claude
  • clean_regex()                – Simple regex-based cleaning (no external calls)
  • get_available_engines()      – Returns which ASR engines are valid for a given lang code
"""

import json
import logging
import os
import re
import threading
from typing import Literal

import boto3
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

# ── AWS clients (lazy-initialised) ───────────────────────────────────────────
_bedrock_client   = None
_translate_client = None

def _bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    return _bedrock_client

def _translate():
    global _translate_client
    if _translate_client is None:
        _translate_client = boto3.client("translate", region_name="us-east-1")
    return _translate_client


# ─────────────────────────────────────────────────────────────────────────────
# Language definitions
# ─────────────────────────────────────────────────────────────────────────────

# Languages ONLY supported by ai4bharat (not in Whisper)
AI4B_ONLY_LANGS = {
    "as", "brx", "doi", "kok", "ks", "mai", "mni", "sat"
}

# Languages supported by BOTH Whisper and ai4bharat
DUAL_STACK_LANGS = {
    "hi", "bn", "ta", "te", "mr", "ur", "gu", "pa",
    "or", "ne", "si", "kn", "ml", "sd"
}

# All South Asian languages (union of both sets)
SOUTH_ASIAN_LANGUAGES: dict[str, str] = {
    "hi":  "Hindi",
    "bn":  "Bengali",
    "mr":  "Marathi",
    "ur":  "Urdu",
    "gu":  "Gujarati",
    "pa":  "Punjabi",
    "or":  "Odia",
    "as":  "Assamese",
    "mai": "Maithili",
    "kok": "Konkani",
    "sd":  "Sindhi",
    "ne":  "Nepali",
    "si":  "Sinhala",
    "ta":  "Tamil",
    "te":  "Telugu",
    "kn":  "Kannada",
    "ml":  "Malayalam",
    "mni": "Meitei (Manipuri)",
    "sat": "Santali",
    "brx": "Bodo",
    "doi": "Dogri",
    "ks":  "Kashmiri",
}

# All Whisper-supported languages (full official list)
WHISPER_ONLY_LANGUAGES: dict[str, str] = {
    "af":  "Afrikaans",
    "am":  "Amharic",
    "ar":  "Arabic",
    "az":  "Azerbaijani",
    "ba":  "Bashkir",
    "be":  "Belarusian",
    "bg":  "Bulgarian",
    "bo":  "Tibetan",
    "br":  "Breton",
    "bs":  "Bosnian",
    "ca":  "Catalan",
    "cs":  "Czech",
    "cy":  "Welsh",
    "da":  "Danish",
    "de":  "German",
    "el":  "Greek",
    "en":  "English",
    "es":  "Spanish",
    "et":  "Estonian",
    "eu":  "Basque",
    "fa":  "Persian",
    "fi":  "Finnish",
    "fo":  "Faroese",
    "fr":  "French",
    "gl":  "Galician",
    "ha":  "Hausa",
    "haw": "Hawaiian",
    "he":  "Hebrew",
    "hr":  "Croatian",
    "ht":  "Haitian Creole",
    "hu":  "Hungarian",
    "hy":  "Armenian",
    "id":  "Indonesian",
    "is":  "Icelandic",
    "it":  "Italian",
    "ja":  "Japanese",
    "jw":  "Javanese",
    "ka":  "Georgian",
    "kk":  "Kazakh",
    "km":  "Khmer",
    "ko":  "Korean",
    "la":  "Latin",
    "lb":  "Luxembourgish",
    "ln":  "Lingala",
    "lo":  "Lao",
    "lt":  "Lithuanian",
    "lv":  "Latvian",
    "mg":  "Malagasy",
    "mi":  "Maori",
    "mk":  "Macedonian",
    "mn":  "Mongolian",
    "ms":  "Malay",
    "mt":  "Maltese",
    "my":  "Myanmar",
    "nl":  "Dutch",
    "nn":  "Norwegian Nynorsk",
    "no":  "Norwegian",
    "oc":  "Occitan",
    "pl":  "Polish",
    "ps":  "Pashto",
    "pt":  "Portuguese",
    "ro":  "Romanian",
    "ru":  "Russian",
    "sa":  "Sanskrit",
    "sk":  "Slovak",
    "sl":  "Slovenian",
    "sn":  "Shona",
    "so":  "Somali",
    "sq":  "Albanian",
    "sr":  "Serbian",
    "su":  "Sundanese",
    "sv":  "Swedish",
    "sw":  "Swahili",
    "tg":  "Tajik",
    "th":  "Thai",
    "tk":  "Turkmen",
    "tl":  "Tagalog",
    "tr":  "Turkish",
    "tt":  "Tatar",
    "uk":  "Ukrainian",
    "uz":  "Uzbek",
    "vi":  "Vietnamese",
    "yi":  "Yiddish",
    "yo":  "Yoruba",
    "zh":  "Chinese",
    "yue": "Cantonese",
}

# Whisper also supports these South Asian langs (DUAL_STACK_LANGS that appear in Whisper's list)
_WHISPER_SOUTH_ASIAN = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",    # listed as "or" in some Whisper builds
    "pa": "Punjabi",
    "sd": "Sindhi",
    "si": "Sinhala",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

# Master list: everything, with a category tag for the UI grouping
# category: "whisper_only" | "dual" | "ai4b_only"
def _build_all_languages() -> dict[str, dict]:
    """
    Returns a dict keyed by ISO code:
      { "es": {"name": "Spanish", "category": "whisper_only"}, ... }
    """
    out = {}
    for code, name in WHISPER_ONLY_LANGUAGES.items():
        out[code] = {"name": name, "category": "whisper_only"}
    for code, name in _WHISPER_SOUTH_ASIAN.items():
        out[code] = {"name": name, "category": "dual"}
    for code, name in SOUTH_ASIAN_LANGUAGES.items():
        if code in AI4B_ONLY_LANGS:
            out[code] = {"name": name, "category": "ai4b_only"}
        elif code not in out:
            # Catch any South Asian lang not already in Whisper list
            out[code] = {"name": name, "category": "ai4b_only"}
    return out

ALL_LANGUAGES: dict[str, dict] = _build_all_languages()


def get_available_engines(lang_code):
    if lang_code in AI4B_ONLY_LANGS:
        return ["ai4b"]
    if lang_code in DUAL_STACK_LANGS:
        return ["whisper", "ai4b"]
    return ["whisper"]


# ─────────────────────────────────────────────────────────────────────────────
# Indian language AWS / ai4bharat codes
# ─────────────────────────────────────────────────────────────────────────────

INDIAN_LANGUAGE_CODES: dict[str, str] = {
    "hi":  "hin_Deva",
    "bn":  "ben_Beng",
    "ta":  "tam_Taml",
    "te":  "tel_Telu",
    "mr":  "mar_Deva",
    "ur":  "urd_Arab",
    "gu":  "guj_Gujr",
    "kn":  "kan_Knda",
    "ml":  "mal_Mlym",
    "pa":  "pan_Guru",
    "or":  "ory_Orya",
    "as":  "asm_Beng",
    "ne":  "npi_Deva",
    "si":  "sin_Sinh",
    "mai": "mai_Deva",
    "kok": "kok_Deva",
    "sd":  "snd_Arab",
    "mni": "mni_Mtei",
    "sat": "sat_Olck",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "ks":  "kas_Arab",
}

SUPPORTED_LANGUAGES = set(INDIAN_LANGUAGE_CODES.keys()) | set(WHISPER_ONLY_LANGUAGES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# ai4bharat IndicConformer ASR  —  local model inference
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_RATE     = 16_000
_AI4B_MODEL_ID   = "ai4bharat/indic-conformer-600m-multilingual"
_ai4b_model      = None
_ai4b_model_lock = threading.Lock()


def _load_ai4b_model():
    global _ai4b_model
    if _ai4b_model is not None:
        return _ai4b_model

    with _ai4b_model_lock:
        if _ai4b_model is not None:
            return _ai4b_model

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for ai4bharat ASR. "
                "Run: pip install torch torchaudio transformers"
            ) from exc

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        log.info("Loading IndicConformer model (first run downloads ~2.4 GB)…")

        try:
            model = AutoModel.from_pretrained(
                _AI4B_MODEL_ID,
                trust_remote_code=True,
                token=hf_token,
            )
        except OSError as exc:
            if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
                raise RuntimeError(
                    "IndicConformer model is gated. Please:\n"
                    "  1. Accept terms at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual\n"
                    "  2. Set HF_TOKEN=hf_... in your environment (or run huggingface-cli login)"
                ) from exc
            raise

        model.eval()
        _ai4b_model = model
        log.info("IndicConformer model loaded.")
        return _ai4b_model


def _read_audio_tensor(audio_path: str):
    import torch
    import torchaudio
    wav, sr = torchaudio.load(audio_path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != _SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, _SAMPLE_RATE)
    return wav


def transcribe_with_ai4bharat(audio_path: str, source_lang: str) -> str:
    all_south_asian = set(SOUTH_ASIAN_LANGUAGES.keys())
    if source_lang not in all_south_asian:
        raise ValueError(
            f"Language '{source_lang}' is not supported by ai4bharat ASR. "
            f"Supported: {sorted(all_south_asian)}"
        )

    model = _load_ai4b_model()
    wav   = _read_audio_tensor(audio_path)

    log.info("IndicConformer CTC  lang=%s  frames=%d", source_lang, wav.shape[-1])
    text = model(wav, source_lang, "ctc")

    if not text or not text.strip():
        raise RuntimeError(
            "IndicConformer returned an empty transcript. "
            "Check the audio contains speech in the selected language."
        )

    log.info("Transcript (%d chars): %s…", len(text), text[:80])
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# AWS Translate helpers
# ─────────────────────────────────────────────────────────────────────────────

AWS_LANG_MAP = {"auto": "auto", "zh": "zh", "he": "he", "ko": "ko"}

def _whisper_to_aws(lang: str) -> str:
    return AWS_LANG_MAP.get(lang, lang)


def _translate_aws(text: str, source_lang: str) -> str:
    src  = _whisper_to_aws(source_lang)
    resp = _translate().translate_text(
        Text=text,
        SourceLanguageCode=src,
        TargetLanguageCode="en",
        Settings={"Formality": "INFORMAL", "Profanity": "MASK"},
    )
    return resp["TranslatedText"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Bedrock translation  (Claude Sonnet)
# ─────────────────────────────────────────────────────────────────────────────

BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
TRANSLATE_SYSTEM = """You are a professional translator. Translate the provided transcript into natural, fluent English.
Rules:
- Preserve speaker turns (lines starting with dash -)
- Keep proper nouns and place names unless they have standard English equivalents
- Preserve the speaker's register and tone
- Output ONLY the English translation. No notes, no preamble, no commentary."""

def _translate_bedrock(text: str, source_lang: str) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens":        4096,
        "system":            TRANSLATE_SYSTEM,
        "messages": [{"role": "user", "content": f"Translate this {source_lang} transcript to English:\n\n{text}"}],
        "temperature": 0.1,
    })
    resp   = _bedrock().invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"].strip()


def translate_with_aws(text: str, source_lang: str, engine: str = "aws") -> str:
    log.info("Translating  engine=%s  lang=%s  chars=%d", engine, source_lang, len(text))
    if engine in ("ai4b", "bedrock"):
        return _translate_bedrock(text, source_lang)
    return _translate_aws(text, source_lang)


# ─────────────────────────────────────────────────────────────────────────────
# Bedrock cleaning  (Claude Sonnet)
# ─────────────────────────────────────────────────────────────────────────────

CLEAN_SYSTEM_TEMPLATE = """You are a linguistic expert in {language}. 
Clean this transcript by fixing spelling and grammar. 
Remove filler words (e.g., um, uh, okay, like) and ASR hallucinations.
Do NOT use speaker labels or dashes. Return a single, clean block of text.
Preserving the original words and vocabulary — do NOT paraphrase.
Return ONLY the cleaned text."""

DIARIZATION_SYSTEM_TEMPLATE = """You are a professional medical scribe. 
Convert the provided English transcript into a structured dialogue.

STRICT RULES:
1. If the input is not a dialogue, just clean up the text and return it as a single speaker. Diarize as "Speaker A" if there is no obvious name associated with the speaker.
2. If the input is apparently a clinical conversation, identify the 'Doctor:' and 'Patient:' based on medical context. If it is not clear who is who (or it is not a medical converation), simply identify speakers as "Speaker A", "Speaker B", etc.
3. Fix grammar and typos, but keep all clinical terminology. If a mispelled word is likely actually a clinical term, correct it.
4. DO NOT offer to help. DO NOT introduce yourself. DO NOT ask for more information. 
5. Return ONLY the dialogue with appropriate cleaning. DO NOT produce any other text aside from the polished translation with diarization.

Example Input: 

How are you feeling today. My back has hurting for tree days.

Example Output:

Doctor: How are you feeling today?
Patient: My back has been hurting for three days.
"""

def clean_with_bedrock(text: str, language: str = "Spanish", diarize: bool = False) -> str:
    template = DIARIZATION_SYSTEM_TEMPLATE if diarize else CLEAN_SYSTEM_TEMPLATE
    system = template.format(language=language)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.1,
    })

    log.info("Bedrock process (diarize=%s) model=%s chars=%d", diarize, BEDROCK_MODEL_ID, len(text))
    resp = _bedrock().invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Regex cleaning  (no external calls)
# ─────────────────────────────────────────────────────────────────────────────

_HALLUCINATION_PATTERNS = [
    r"\[Música\]", r"\[música\]", r"\[Music\]", r"\[music\]",
    r"\[Inaudible\]", r"\[inaudible\]", r"\[INAUDIBLE\]",
    r"\[Applause\]", r"\[applause\]",
    r"\[.*?\]",
    r"Subtítulos .*", r"Transcripción .*",
    r"(?i)subtitles by .*",
    r"(?i)www\.\S+\.(?:com|org|net)",
]

_FILLER_ES = r"\b(este|pues|este+|o sea|o sea que|verdad|¿verdad\?|eh+|ah+|um+|uh+)\b"
_FILLER_EN = r"\b(um+|uh+|er+|like|you know|I mean|sort of|kind of)\b"

def clean_regex(text: str) -> str:
    for pat in _HALLUCINATION_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(_FILLER_ES, "", text, flags=re.IGNORECASE)
    text = re.sub(_FILLER_EN, "", text, flags=re.IGNORECASE)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()