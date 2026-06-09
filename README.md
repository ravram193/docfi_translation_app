# Transcript Studio

Local-first audio transcription pipeline using **WhisperKit** (CoreML, macOS),
**AWS Bedrock Claude Sonnet** for cleaning, and **AWS Translate** (+ ai4bharat for
Indian languages) for English translation.

---

## Quick Overview

This repository provides an end-to-end pipeline to transcribe audio and translate it to English.

**Pipeline:**
```
Audio (mic / file)
    ↓
ASR  →  WhisperKit (CoreML, on-device)
        OR ai4bharat IndicConformer (Indian languages)
    ↓
Raw Transcript
    ↓
Clean  →  AWS Bedrock Claude Sonnet
    ↓
Cleaned Transcript
    ↓
Translate → English
    ↓  AWS Translate (fastest, most languages)
       OR ai4bharat IndicTrans2  (best for Indian languages)
       OR Bedrock Claude Sonnet  (highest quality, slowest)
    ↓
English Translation (using either Claude-polished or raw transcript)
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

> To deactivate later, run `deactivate`.

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

### 3. WhisperKit CLI (macOS only, Apple Silicon / macOS 13+)

```bash
brew install whisperkit-cli
```

Models are downloaded automatically on first use from `argmaxinc/whisperkit-coreml` on Hugging Face.

| Model | Size | Notes |
|---|---|---|
| `openai_whisper-large-v3-v20240930_turbo` | — | **Default** |
| `openai_whisper-large-v3-v20240930_626MB` | 626 MB | — |
| `openai_whisper-large-v3-v20240930_547MB` | 547 MB | — |
| `distil-whisper_distil-large-v3_594MB` | 594 MB | Fastest / smallest |
| `distil-whisper_distil-large-v3_turbo_600MB` | 600 MB | — |

### 4. Configure credentials

The app requires credentials for two services: **AWS** (Bedrock + Translate) and **Hugging Face** (ai4bharat IndicConformer).

#### Option A — `.env` file (recommended for local dev)

Create a `.env` file in the project root:

```bash
# AWS — required for Bedrock cleaning and AWS Translate
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# Hugging Face for ai4bharat IndicConformer ASR
# Accept model terms at: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
# Then generate a token at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here
```

Then install `python-dotenv` and load it at the top of `app.py` / `eval.py`:

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
load_dotenv()
```

> `.env` is already listed in `.gitignore` — never commit it.

#### Option B — `aws configure` (standard AWS CLI)

```bash
pip install awscli
aws configure
```

This writes credentials to `~/.aws/credentials` and is picked up automatically by `boto3`. Set `HF_TOKEN` separately as an environment variable:

```bash
export HF_TOKEN=hf_your_token_here   # macOS/Linux
# setx HF_TOKEN hf_your_token_here   # Windows
```

#### Option C — inline environment variables

```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1
export HF_TOKEN=hf_your_token_here
```

#### Getting your AWS credentials

**`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`**

1. Log in to the [AWS Console](https://console.aws.amazon.com)
2. Click your account name (top right) → **Security credentials**
3. Scroll to **Access keys** → **Create access key**
4. Choose **Local code** as the use case → proceed through the prompts
5. Copy both values — the secret is only shown once

> If you're on a company AWS account, ask your admin to create an IAM user with the permissions below and share the keys with you.

**`AWS_DEFAULT_REGION`**

Set to `us-east-1` (recommended — broadest Bedrock model availability). Visible in the top-right dropdown of the AWS Console.

**Enable Bedrock model access**

1. Go to **Bedrock** in the AWS Console → **Model access** (left sidebar)
2. Find **Claude Sonnet** under Anthropic and request access if not already enabled — approval is usually instant

**Required IAM permissions**

Attach these to the IAM user that owns your keys:

| Managed policy | Or custom permission |
|---|---|
| `AmazonBedrockFullAccess` | `bedrock:InvokeModel` |
| `TranslateFullAccess` | `translate:TranslateText` |
---

## Running Locally

```bash
python app.py
```

Then open **[http://localhost:8050](http://localhost:8050)** in your browser.

The app runs a local Dash server — no internet connection required for transcription (WhisperKit runs on-device). AWS calls (Bedrock, Translate) require connectivity and valid credentials.

---

## Language & ASR Engine Logic

The language dropdown drives ASR engine selection automatically:

| Language Category | Available Engines |
|---|---|
| Standard languages | WhisperKit only |
| Indian languages (dual-stack) | WhisperKit **or** ai4bharat (user selects) |
| Indian languages (ai4bharat only) | ai4bharat (only for Indic languages not available via WhisperKit) |

---

## Repository Layout

```text
├── app/
│   ├── app.py                         # Main Dash application entry point
│   └── pipeline.py                    # Core transcription & translation pipeline for app.py
│
├── assets/
│   └── style.css                      # App stylesheet
│
├── eval/
│   ├── languages/
│   │   └── cv-corpus-hi/
│   │       └── hi/                    # Hindi Common Voice corpus data
│   │           ├── clips/             # Audio clip files
│   │           ├── clip_durations.tsv
│   │           ├── dev.tsv
│   │           ├── invalidated.tsv
│   │           ├── other.tsv
│   │           ├── README.md
│   │           ├── reported.tsv
│   │           ├── test.tsv
│   │           ├── train.tsv
│   │           ├── unvalidated_sentences.tsv
│   │           ├── validated_sentences.tsv
│   │           └── validated.tsv
│   │
│   ├── results/
│   │   ├── hi_asr_results.csv         # ai4bharat vs WhisperKit ASR evaluation results for Hindi
│   │   └── hi_asr_summary.json        # Summary metrics from ai4bharat vs WhisperKit ASR evaluation
│   │
│   └── eval.py                        # ai4bharat vs WhisperKit ASR evaluation script
│
├── requirements.txt                   # Python dependencies
└── .gitignore
```

---

## Evaluation

While not related to running the translation app itself, the `eval/` folder is a new and evolving addition to this repository. This pipeline aims to provide a starting point for comparing ai4bharat and WhisperKit's ASR capabilities for Indic languages (although the pipeline could in theory be generalized for comparing any two models' performances for a given language). Currently, the only language available for testing in this repo is Hindi. The Hindi corpus from Common Voice contains short audio snippets of spoken Hindi, each of which is paired with a transcription. These transcriptions act as a ground truth against which to compare the outputs of ASR models.

The dataset for Hindi ASR evaluation was downloaded from Mozilla Data Collective. Browse other language options [here](https://mozilladatacollective.com/datasets?task=ASR). Once downloaded, add new language datasets to the `eval/languages/` folder. 

To run the evaluation pipeline for Hindi, simply write the following code in the terminal:

```bash
python3 eval.py --lang <hi> --max_samples 25
```

Feel free to change the number of samples. To change languages, type the abbreviation used to describe the language in the downloaded dataset.

Results are written to `eval_results/<abbreviation>_asr_results.csv` and `results/<abbreviation>_asr_summary.json`.