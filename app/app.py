"""
app.py  —  Whisper Transcription Pipeline
==========================================
Transcribe → Clean (AWS Claude Sonnet) → Translate (AWS Translate / ai4bharat)
→ Verify (Google TTS playback)

Single language dropdown drives dynamic ASR engine selection:
  - Most languages → WhisperKit only
  - DUAL_STACK_LANGS → WhisperKit or ai4bharat (user chooses)
  - AI4B_ONLY_LANGS  → ai4bharat only (no Whisper option shown)

Dependencies:
    pip install dash dash-bootstrap-components boto3 requests sounddevice soundfile numpy

WhisperKit CLI must be installed via Homebrew (Apple Silicon / macOS 13+ only):
    brew install whisperkit-cli

AWS credentials must be configured:
    aws configure  (needs Bedrock + Translate access)

Run:
    python app.py
"""

import base64
import json
import logging
import os
import subprocess
import tempfile
import threading
import time

import boto3
import dash
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from pipeline import (
    ALL_LANGUAGES,
    INDIAN_LANGUAGE_CODES,
    SOUTH_ASIAN_LANGUAGES,
    SUPPORTED_LANGUAGES,
    AI4B_ONLY_LANGS,
    DUAL_STACK_LANGS,
    clean_with_bedrock,
    translate_with_aws,
    transcribe_with_ai4bharat,
    get_available_engines,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Transcript Studio",
    suppress_callback_exceptions=True,
)

# ── WhisperKit models ─────────────────────────────────────────────────────────
WHISPERKIT_MODELS = [
    ("distil-large-v3",              "distil-whisper_distil-large-v3"),
    ("distil-large-v3  594MB",       "distil-whisper_distil-large-v3_594MB"),
    ("distil-large-v3  turbo",       "distil-whisper_distil-large-v3_turbo"),
    ("distil-large-v3  turbo 600MB", "distil-whisper_distil-large-v3_turbo_600MB"),
    ("large-v3",                     "openai_whisper-large-v3-v20240930"),
    ("large-v3  547MB",              "openai_whisper-large-v3-v20240930_547MB"),
    ("large-v3  626MB",              "openai_whisper-large-v3-v20240930_626MB"),
    ("large-v3  turbo",              "openai_whisper-large-v3-v20240930_turbo"),
]
DEFAULT_MODEL = "openai_whisper-large-v3-v20240930_turbo"

# ── Build grouped dropdown options ────────────────────────────────────────────
def _build_language_options():
    """
    Returns options list for dcc.Dropdown, grouped as:
      Auto-detect / Standard Languages / South Asian — Dual Engine / South Asian — ai4bharat Only
    """
    options = [{"label": "Auto-detect", "value": "auto", "title": "whisper_only"}]

    whisper_only = sorted(
        [(code, info["name"]) for code, info in ALL_LANGUAGES.items() if info["category"] == "whisper_only"],
        key=lambda x: x[1],
    )
    dual = sorted(
        [(code, info["name"]) for code, info in ALL_LANGUAGES.items() if info["category"] == "dual"],
        key=lambda x: x[1],
    )
    ai4b_only = sorted(
        [(code, info["name"]) for code, info in ALL_LANGUAGES.items() if info["category"] == "ai4b_only"],
        key=lambda x: x[1],
    )

    options.append({"label": "── Standard Languages (WhisperKit) ──", "value": "__sep1__", "disabled": True})
    for code, name in whisper_only:
        options.append({"label": name, "value": code, "title": "whisper_only"})

    options.append({"label": "── South Asian · Whisper + ai4bharat ──", "value": "__sep2__", "disabled": True})
    for code, name in dual:
        options.append({"label": name, "value": code, "title": "dual"})

    options.append({"label": "── South Asian · ai4bharat Only ──", "value": "__sep3__", "disabled": True})
    for code, name in ai4b_only:
        options.append({"label": name, "value": code, "title": "ai4b_only"})

    return options

LANGUAGE_OPTIONS = _build_language_options()

# ── Recording state ───────────────────────────────────────────────────────────
_recording   = False
_frames: list = []
_SAMPLE_RATE = 16_000
_rec_thread  = None

def _record_thread():
    global _recording, _frames
    _frames = []
    with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while _recording:
            data, _ = stream.read(1024)
            _frames.append(data)

# ── WhisperKit transcription ──────────────────────────────────────────────────
def run_whisperkit(audio_path: str, model_id: str, lang: str | None) -> str:
    cmd = [
        "whisperkit-cli", "transcribe",
        "--audio-path",   audio_path,
        "--model",        model_id,
        "--model-prefix", "",
        "--skip-special-tokens",
    ]
    if lang and lang != "auto":
        cmd += ["--language", lang]

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)

    stdout = result.stdout.strip()
    try:
        data = json.loads(stdout)
        return data.get("text", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return stdout

# ── Styles ────────────────────────────────────────────────────────────────────
FONT_DISPLAY = "'Cormorant Garamond', Georgia, serif"
FONT_MONO    = "'IBM Plex Mono', 'Courier New', monospace"

C_BG      = "#f5f2ed"
C_CARD    = "#ffffff"
C_BORDER  = "#ddd8d0"
C_TEXT    = "#1a1814"
C_DIM     = "#8a8278"
C_ACCENT  = "#9b6b2f"
C_ACCENT2 = "#3a6b8a"
C_ERR     = "#b93030"
C_GOOD    = "#2e7d52"

LABEL_STYLE = {
    "color": C_DIM, "fontSize": "10px", "letterSpacing": "3px",
    "textTransform": "uppercase", "display": "block", "marginBottom": "10px",
    "fontFamily": FONT_MONO,
}
SECTION    = {"marginBottom": "36px"}
OUTPUT_BOX = {
    "background": C_CARD, "border": f"1px solid {C_BORDER}",
    "borderRadius": "4px", "padding": "28px", "minHeight": "140px",
    "color": C_TEXT, "fontSize": "16px", "lineHeight": "1.85",
    "whiteSpace": "pre-wrap", "fontFamily": FONT_DISPLAY,
    "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
}
PLACEHOLDER_STYLE = {"color": C_DIM, "fontStyle": "italic"}

def pill_style(active=False, accent=C_ACCENT) -> dict:
    return {
        "background":    accent if active else "transparent",
        "color":         "#ffffff" if active else C_DIM,
        "border":        f"1px solid {accent if active else C_BORDER}",
        "padding":       "5px 14px",
        "fontSize":      "11px",
        "fontFamily":    FONT_MONO,
        "borderRadius":  "20px",
        "cursor":        "pointer",
        "letterSpacing": "0.5px",
        "whiteSpace":    "nowrap",
        "transition":    "all 0.15s ease",
    }

def pill(label, id_, active=False, accent=C_ACCENT):
    return html.Button(label, id=id_, n_clicks=0, style=pill_style(active, accent))

def action_btn(label, id_, icon="", disabled=False):
    return html.Button(
        [html.Span(icon, style={"marginRight": "8px"}), label] if icon else label,
        id=id_, n_clicks=0, disabled=disabled,
        style={
            "background":    C_ACCENT if not disabled else C_BORDER,
            "color":         "#ffffff" if not disabled else C_DIM,
            "border":        "none",
            "padding":       "10px 24px",
            "fontSize":      "12px",
            "fontFamily":    FONT_MONO,
            "borderRadius":  "3px",
            "cursor":        "pointer" if not disabled else "not-allowed",
            "letterSpacing": "1.5px",
            "textTransform": "uppercase",
        },
    )

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": C_BG,
        "fontFamily": FONT_MONO,
        "padding": "48px 20px 80px",
        "color": C_TEXT,
    },
    children=[
        html.Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=IBM+Plex+Mono:wght@300;400;500&display=swap",
        ),

        html.Div(style={"maxWidth": "780px", "margin": "0 auto"}, children=[

            # ── Header ────────────────────────────────────────────────────────
            html.Div(style={"marginBottom": "56px"}, children=[
                html.Div("Transcript Studio", style={
                    "fontFamily": FONT_DISPLAY,
                    "fontSize": "58px",
                    "fontWeight": "300",
                    "color": C_TEXT,
                    "lineHeight": "1",
                    "letterSpacing": "-1px",
                }),
                html.Div("whisperkit · ai4bharat · bedrock · aws translate", style={
                    "color": C_DIM,
                    "fontSize": "11px",
                    "letterSpacing": "4px",
                    "textTransform": "uppercase",
                    "marginTop": "8px",
                }),
                html.Div(style={
                    "width": "48px", "height": "2px",
                    "background": C_ACCENT, "marginTop": "20px",
                }),
            ]),

            # ── Step 1: Language selection (single dropdown) ──────────────────
            html.Div(style=SECTION, children=[
                html.Label("① Source Language", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="dd-lang",
                    options=LANGUAGE_OPTIONS,
                    value="es",
                    clearable=False,
                    placeholder="Select source language…",
                    style={
                        "width": "320px", "fontSize": "13px",
                        "fontFamily": FONT_MONO,
                        "background": C_CARD, "color": C_TEXT,
                    },
                ),
                # Stores
                dcc.Store(id="store-asr-engine",  data="whisper"),
                dcc.Store(id="store-active-lang", data="es"),
            ]),

            # ── Step 2: ASR Engine + Model (dynamic, driven by language) ──────
            html.Div(style=SECTION, children=[
                html.Label("② ASR Engine & Model", style=LABEL_STYLE),

                html.Div([

                    # ENGINE SELECTOR
                    html.Div([
                        html.Div("ASR Engine", style={**LABEL_STYLE, "marginBottom": "8px"}),

                        dcc.RadioItems(
                            id="radio-asr-engine",
                            options=[],
                            value=None,
                            inline=True,
                            inputStyle={"marginRight": "6px"},
                        ),
                    ], style={"marginBottom": "20px"}, id="engine-container"),

                    # MODEL SELECTOR
                    html.Div([
                        html.Div("WhisperKit Model", style={**LABEL_STYLE, "marginBottom": "8px"}),

                        dcc.RadioItems(
                            id="radio-wk-model",
                            options=[
                                {"label": label, "value": mid}
                                for label, mid in WHISPERKIT_MODELS
                            ],
                            value=DEFAULT_MODEL,
                            inline=True,
                            inputStyle={"marginRight": "6px"},
                        ),
                    ], id="model-container"),

                    # INFO BLOCK
                    html.Div(id="ai4b-info"),

                ]),

                dcc.Store(id="store-wk-model", data=DEFAULT_MODEL),
            ]),

            # ── Step 3: Audio Input ───────────────────────────────────────────
            html.Div(style=SECTION, children=[
                html.Label("③ Audio Input", style=LABEL_STYLE),
                html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "14px",
                                "alignItems": "center"}, children=[
                    html.Button(
                        [html.Span("●", style={"color": C_ERR, "marginRight": "8px"}), "Record"],
                        id="btn-record", n_clicks=0,
                        style={
                            "background": C_CARD, "color": C_TEXT,
                            "border": f"1px solid {C_BORDER}",
                            "padding": "10px 22px", "fontSize": "12px",
                            "fontFamily": FONT_MONO, "cursor": "pointer",
                            "borderRadius": "3px", "letterSpacing": "2px",
                        },
                    ),
                    html.Button(
                        [html.Span("■", style={"marginRight": "8px"}), "Stop"],
                        id="btn-stop", n_clicks=0, disabled=True,
                        style={
                            "background": C_CARD, "color": C_DIM,
                            "border": f"1px solid {C_BORDER}",
                            "padding": "10px 22px", "fontSize": "12px",
                            "fontFamily": FONT_MONO, "cursor": "not-allowed",
                            "borderRadius": "3px", "letterSpacing": "2px",
                        },
                    ),
                    html.Div(id="rec-indicator", style={
                        "width": "8px", "height": "8px",
                        "borderRadius": "50%", "background": C_BORDER,
                        "marginLeft": "4px",
                    }),
                    html.Div(id="rec-status", style={
                        "color": C_DIM, "fontSize": "11px", "letterSpacing": "1px",
                    }),
                ]),
                dcc.Upload(
                    id="upload-audio",
                    children=html.Div([
                        "Drop audio file here  —  or  ",
                        html.Span("browse", style={"color": C_ACCENT, "textDecoration": "underline"}),
                    ], style={"fontSize": "12px", "color": C_DIM}),
                    style={
                        "border": f"1px dashed {C_BORDER}", "borderRadius": "4px",
                        "padding": "18px", "textAlign": "center", "cursor": "pointer",
                    },
                    accept="audio/*",
                ),
            ]),

            # ── Active engine indicator ───────────────────────────────────────
            html.Div(id="engine-indicator", style={
                "marginBottom": "16px", "padding": "10px 16px",
                "borderRadius": "4px", "fontSize": "11px",
                "fontFamily": FONT_MONO, "letterSpacing": "1px",
            }),

            # ── Transcribe button ─────────────────────────────────────────────
            html.Div(style={"marginBottom": "36px"}, children=[
                action_btn("Transcribe", "btn-transcribe", icon="◎"),
            ]),

            # ── Step 4: Transcript ────────────────────────────────────────────
            html.Div(style=SECTION, children=[
                html.Label("④ Transcript (Source Language)", style=LABEL_STYLE),

                html.Div(style={"marginTop": "10px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"}, children=[
                        html.Span("RAW", style=LABEL_STYLE),
                        html.Div(id="raw-meta", style={"color": C_DIM, "fontSize": "10px", "fontFamily": FONT_MONO}),
                    ]),
                    dcc.Loading(type="dot", color=C_ACCENT, children=
                        html.Div(id="raw-output", style={**OUTPUT_BOX, "minHeight": "80px", "fontSize": "14px", "color": C_DIM}),
                    ),
                ]),

                html.Div(style={"marginTop": "20px"}, children=[
                    html.Span("POLISHED (CLAUDE)", style=LABEL_STYLE),
                    dcc.Loading(type="dot", color=C_ACCENT, children=
                        html.Div(id="clean-output", style=OUTPUT_BOX,
                            children=html.Span("Polished text will appear here...", style=PLACEHOLDER_STYLE)),
                    ),
                ]),
            ]),

            # ── Step 5: Translate ─────────────────────────────────────────────
            html.Div(style=SECTION, children=[
                html.Label("⑤ Translate → English", style=LABEL_STYLE),

                html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "14px",
                                "flexWrap": "wrap"}, children=[
                    pill("AWS Translate",                "tr-aws",     active=True),
                    pill("ai4bharat (Indian languages)", "tr-ai4b",    active=False),
                    pill("Bedrock Claude",               "tr-bedrock", active=False),
                ]),
                dcc.Store(id="store-tr-engine", data="aws"),

                html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "16px"}, children=[
                    pill("Cleaned", "tr-src-clean", active=True),
                    pill("Raw",     "tr-src-raw",   active=False),
                ]),
                dcc.Store(id="store-tr-src", data="clean"),

                action_btn("Translate", "btn-translate", icon="⟶"),

                html.Div(style={"marginTop": "16px"}, children=[
                    html.Label("RAW", style=LABEL_STYLE),
                    dcc.Loading(type="dot", color=C_ACCENT, children=
                        html.Div(id="tr-output-raw", style=OUTPUT_BOX,
                            children=html.Span("Raw translation will appear here…",
                                            style=PLACEHOLDER_STYLE)),
                    ),
                ]),

                html.Div(style={"marginTop": "16px"}, children=[
                    html.Label("POLISHED (CLAUDE)", style=LABEL_STYLE),
                    dcc.Loading(type="dot", color=C_ACCENT, children=
                        html.Div(id="tr-output-clean", style=OUTPUT_BOX,
                            children=html.Span("Polished translation will appear here…",
                                            style=PLACEHOLDER_STYLE)),
                    ),
                ]),
            ]),

            # ── Footer ────────────────────────────────────────────────────────
            html.Div(style={"marginTop": "60px", "borderTop": f"1px solid {C_BORDER}",
                            "paddingTop": "20px"}, children=[
                html.Div("local · aws bedrock · aws translate · ai4bharat",
                         style={"color": C_DIM, "fontSize": "10px",
                                "letterSpacing": "3px", "textTransform": "uppercase"}),
            ]),

            # ── Stores ────────────────────────────────────────────────────────
            dcc.Store(id="store-recording",    data=False),
            dcc.Store(id="store-tr-raw",       data=""),
            dcc.Store(id="store-raw-text",     data=""),
            dcc.Store(id="store-clean-text",   data=""),
            dcc.Store(id="store-tr-text",      data=""),
            # Permanent store — dynamic engine pills write here so callbacks
            # can safely reference a stable id that always exists in the layout.
            dcc.Store(id="store-engine-request", data=""),
            dcc.Interval(id="rec-pulse", interval=600, disabled=True),
        ]),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Unified lang + engine store callback ───────────────────────────────────
# engine-pill-whisper / engine-pill-ai4b are dynamic (only in DOM for dual langs)
# so they cannot be Inputs. Instead the pills write "whisper"/"ai4b" into the
# permanent store-engine-request, which this callback reads as an Input.

# ── 1c. Engine indicator banner (reads both stores, no circular dependency) ───
@callback(
    Output("engine-indicator", "children"),
    Output("engine-indicator", "style"),
    Input("store-asr-engine",  "data"),
    Input("store-active-lang", "data"),
    prevent_initial_call=False,
)
def update_engine_indicator(active_engine, lang_code):
    base_style = {
        "marginBottom": "16px", "padding": "10px 16px",
        "borderRadius": "4px", "fontSize": "11px",
        "fontFamily": FONT_MONO, "letterSpacing": "1px",
    }
    lang_code = lang_code or "es"
    lang_info = ALL_LANGUAGES.get(lang_code, {})
    lang_name = lang_info.get("name", lang_code) if lang_code != "auto" else "Auto-detect"

    if active_engine == "ai4b":
        msg   = f"◉  ai4bharat IndicConformer  ·  {lang_name}  ({lang_code})"
        style = {**base_style, "background": "#e8f0f5", "color": C_ACCENT2,
                 "border": f"1px solid {C_ACCENT2}"}
    else:
        msg   = f"◎  WhisperKit  ·  {lang_name}  ({lang_code})"
        style = {**base_style, "background": "#f5f0e8", "color": C_ACCENT,
                 "border": f"1px solid {C_ACCENT}"}
    return msg, style


# ── 2. Language dropdown → rebuild ASR engine/model panel ─────────────────────
# Uses dcc.RadioItems (stable Dash component with its own state) instead of
# html.Buttons, so clicks are handled natively without fragile dynamic IDs.


# ── 3. RadioItems → stores (model + engine) ───────────────────────────────────
# radio-wk-model and radio-asr-engine are rendered inside the dynamic panel but
# dcc.RadioItems with suppress_callback_exceptions=True works correctly because
# Dash treats RadioItems.value as a stable property even when conditionally shown.
@callback(
    Output("store-wk-model", "data"),
    Input("radio-wk-model", "value"),
    prevent_initial_call=True,
)
def pick_model(value):
    return value or DEFAULT_MODEL

@callback(
    Output("store-active-lang", "data"),
    Output("store-asr-engine", "data"),
    Output("radio-asr-engine", "options"),
    Output("radio-asr-engine", "value"),
    Output("engine-container", "style"),
    Output("model-container", "style"), # <--- This controls the Whisper models
    Output("ai4b-info", "children"),
    Input("dd-lang", "value"),
    Input("radio-asr-engine", "value"),
    State("store-active-lang", "data"),
    State("store-asr-engine", "data"),
)
def sync_asr_logic(dd_lang, radio_engine, current_lang, current_engine):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "dd-lang"

    # 1. Resolve Language
    active_lang = dd_lang if trigger_id == "dd-lang" else current_lang
    active_lang = active_lang or "es"

    # 2. Get Engines available for this specific language
    available_engines = get_available_engines(active_lang) if active_lang != "auto" else ["whisper"]
    category = ALL_LANGUAGES.get(active_lang, {}).get("category", "whisper_only")

    # 3. Determine which engine is ACTIVE
    if trigger_id == "dd-lang":
        # New language selected: reset to the first available engine
        active_engine = available_engines[0]
    else:
        # User manually clicked a radio button: use that if it's still valid
        active_engine = radio_engine if radio_engine in available_engines else available_engines[0]

    # 4. Filter the Radio Button options for the UI
    engine_options = []
    if "whisper" in available_engines:
        engine_options.append({"label": "WhisperKit", "value": "whisper"})
    if "ai4b" in available_engines:
        engine_options.append({"label": "ai4bharat", "value": "ai4b"})

    # 5. Visibility Logic (The Critical Part)
    # Only show engine choice if there is more than one option
    engine_selector_style = {"display": "block", "marginBottom": "20px"} if len(available_engines) > 1 else {"display": "none"}
    
    # STRICTOR CHECK: Only show Whisper models if:
    # A) Whisper is an available engine for this language AND
    # B) Whisper is currently selected as the active engine
    if "whisper" in available_engines and active_engine == "whisper":
        model_container_style = {"display": "block"}
    else:
        model_container_style = {"display": "none"}

    # 6. Optional Info Text
    info = None
    if category == "ai4b_only":
        info = html.Div(f"Note: This language requires the ai4bharat engine.", 
                        style={"color": C_ACCENT2, "fontSize": "11px", "marginTop": "10px"})

    return active_lang, active_engine, engine_options, active_engine, engine_selector_style, model_container_style, info

# ── 5. Translate engine ────────────────────────────────────────────────────────
@callback(
    Output("store-tr-engine", "data"),
    Output("tr-aws",     "style"),
    Output("tr-ai4b",    "style"),
    Output("tr-bedrock", "style"),
    Input("tr-aws",     "n_clicks"),
    Input("tr-ai4b",    "n_clicks"),
    Input("tr-bedrock", "n_clicks"),
    prevent_initial_call=False,
)
def set_tr_engine(*_):
    ctx  = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "tr-aws"
    eng  = {"tr-aws": "aws", "tr-ai4b": "ai4b", "tr-bedrock": "bedrock"}.get(trig, "aws")
    return (eng,
            pill_style(eng == "aws"),
            pill_style(eng == "ai4b"),
            pill_style(eng == "bedrock"))


# ── 6. Translate source ────────────────────────────────────────────────────────
@callback(
    Output("store-tr-src",    "data"),
    Output("tr-src-clean", "style"),
    Output("tr-src-raw",   "style"),
    Input("tr-src-clean", "n_clicks"),
    Input("tr-src-raw",   "n_clicks"),
    prevent_initial_call=False,
)
def set_tr_src(*_):
    ctx  = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "tr-src-clean"
    src  = "clean" if trig == "tr-src-clean" else "raw"
    return (src,
            pill_style(src == "clean"),
            pill_style(src == "raw"))


# ── 7. Recording ───────────────────────────────────────────────────────────────
@callback(
    Output("store-recording", "data"),
    Output("btn-record",      "disabled"),
    Output("btn-stop",        "disabled"),
    Output("btn-stop",        "style"),
    Output("rec-pulse",       "disabled"),
    Input("btn-record", "n_clicks"),
    Input("btn-stop",   "n_clicks"),
    prevent_initial_call=True,
)
def toggle_recording(*_):
    global _recording, _rec_thread
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    stop_on  = {
        "background": C_CARD, "color": C_TEXT, "border": f"1px solid {C_BORDER}",
        "padding": "10px 22px", "fontSize": "12px", "fontFamily": FONT_MONO,
        "cursor": "pointer", "borderRadius": "3px", "letterSpacing": "2px",
    }
    stop_off = {**stop_on, "color": C_DIM, "cursor": "not-allowed"}
    if trig == "btn-record":
        _recording  = True
        _rec_thread = threading.Thread(target=_record_thread, daemon=True)
        _rec_thread.start()
        return True, True, False, stop_on, False
    _recording = False
    if _rec_thread:
        _rec_thread.join(timeout=2)
    return False, False, True, stop_off, True


@callback(
    Output("rec-indicator", "style"),
    Output("rec-status",    "children"),
    Input("rec-pulse",       "n_intervals"),
    Input("store-recording", "data"),
)
def pulse(n, recording):
    base = {"width": "8px", "height": "8px", "borderRadius": "50%", "marginLeft": "4px"}
    if recording:
        c = C_ERR if (n or 0) % 2 == 0 else "#7a2020"
        return {**base, "background": c}, "recording…"
    return {**base, "background": C_BORDER}, ""


# ── 8. Transcribe ──────────────────────────────────────────────────────────────
@callback(
    Output("raw-output",       "children"),
    Output("clean-output",     "children"),
    Output("store-raw-text",   "data"),
    Output("store-clean-text", "data"),
    Output("raw-meta",         "children"),
    Input("btn-transcribe",    "n_clicks"),
    State("upload-audio",      "contents"),
    State("upload-audio",      "filename"),
    State("store-wk-model",    "data"),
    State("store-asr-engine",  "data"),
    State("store-active-lang", "data"),
    prevent_initial_call=True,
)
def do_transcribe_and_clean(_, upload_contents, upload_filename, wk_model, asr_engine, active_lang):
    if not _:
        raise PreventUpdate

    t0 = time.time()

    try:
        lang_arg = None if (not active_lang or active_lang == "auto") else active_lang

        if _frames:
            audio = np.concatenate(_frames, axis=0).flatten()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, _SAMPLE_RATE)
                tmp = f.name
            try:
                if asr_engine == "ai4b":
                    text = transcribe_with_ai4bharat(tmp, active_lang)
                else:
                    text = run_whisperkit(tmp, wk_model, lang_arg)
            finally:
                os.unlink(tmp)

        elif upload_contents:
            _, b64 = upload_contents.split(",")
            ext = os.path.splitext(upload_filename or "audio.wav")[-1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(base64.b64decode(b64))
                tmp = f.name
            try:
                if asr_engine == "ai4b":
                    text = transcribe_with_ai4bharat(tmp, active_lang)
                else:
                    text = run_whisperkit(tmp, wk_model, lang_arg)
            finally:
                os.unlink(tmp)

        else:
            placeholder = html.Span("No audio source found.", style=PLACEHOLDER_STYLE)
            return placeholder, placeholder, "", "", ""

        raw_text = text.strip()
        elapsed  = time.time() - t0
        meta     = f"{len(raw_text.split())} words · {elapsed:.1f}s"

        lang_info  = ALL_LANGUAGES.get(active_lang or "auto", {})
        lang_name  = lang_info.get("name", (active_lang or "").title()) or "Unknown"

        try:
            polished_text = clean_with_bedrock(raw_text, language=lang_name, diarize=False)
        except Exception as e:
            log.error("Auto-clean failed: %s", e)
            polished_text = raw_text

        return raw_text, polished_text, raw_text, polished_text, meta

    except Exception as e:
        log.exception("Pipeline failed")
        err_msg = html.Div(str(e), style={"color": C_ERR})
        return err_msg, err_msg, "", "", ""

# ── 9. Translate Callback ──────────────────────────────────────────────────────
# ── 9. Translate Callback ──────────────────────────────────────────────────────
@callback(
    Output("tr-output-raw", "children"),
    Output("tr-output-clean", "children"),
    Output("store-tr-text", "data"),
    Output("store-tr-raw", "data"),
    Input("btn-translate", "n_clicks"),         # Triggered by the Translate button
    Input("store-tr-engine", "data"),           # Re-run if user switches AWS/ai4b/Bedrock
    Input("store-tr-src", "data"),              # Re-run if user toggles Raw/Clean source
    State("store-raw-text", "data"),            # The text from the ASR step
    State("store-clean-text", "data"),          # The polished text from the ASR step
    State("store-active-lang", "data"),         # The language code (e.g., 'hi', 'bn', 'ta')
    prevent_initial_call=True,
)
def handle_translation(n_clicks, tr_engine, tr_src, raw, cleaned, active_lang):
    # Dash calls this on startup; prevent_initial_call stops it until user interaction
    if not n_clicks:
        raise PreventUpdate
    
    # ── Safety Check: Engine/Language Match ──────────────────────
    # ai4bharat (IndicTrans2) only supports these specific Indian languages
    indian_langs = ["hi", "bn", "pa", "gu", "kn", "ml", "mr", "ta", "te", "or", "as", "sa", "ur"]
    
    if tr_engine == "ai4b" and active_lang not in indian_langs:
        error_msg = html.Div([
            html.Strong("Engine Mismatch: "),
            f"ai4bharat only supports Indian languages. '{active_lang}' is not supported."
        ], style={"color": C_ERR, "fontSize": "12px"})
        
        # Return error to UI and clear background data stores to prevent "ghost" text
        return error_msg, error_msg, "", ""

    # Step 1: Determine which version of the transcription to translate
    # Users can toggle between the "Raw" ASR output or the "Cleaned" Claude version
    text_to_process = cleaned if (tr_src == "clean" and cleaned) else raw

    if not text_to_process or not str(text_to_process).strip():
        msg = html.Span("(No source text found. Transcribe something first!)", style=PLACEHOLDER_STYLE)
        return msg, msg, "", ""

    # Step 2: Skip if already English
    if active_lang == "en":
        msg = html.Span("Source is English — no translation needed.", style=PLACEHOLDER_STYLE)
        return msg, msg, "", ""

    try:
        log.info(f"Translating {active_lang} using {tr_engine}...")

        # 1. Primary Translation
        # This calls pipeline.translate_with_aws which handles AWS or ai4b logic
        translated_text = translate_with_aws(
            text=text_to_process,
            source_lang=active_lang,
            engine=tr_engine
        )
        translated_text = str(translated_text or "")

        # 2. English Polishing (Bedrock/Claude)
        # We check length to prevent sending empty snippets that cause Claude to "chat" back
        if len(translated_text.strip()) < 5:
            ui_raw = translated_text if translated_text.strip() else html.Span("(empty)", style=PLACEHOLDER_STYLE)
            ui_clean = html.Span("(Text too short for medical formatting)", style=PLACEHOLDER_STYLE)
            return ui_raw, ui_clean, "", translated_text

        try:
            # diarize=True triggers the "Medical Scribe" prompt in pipeline.py
            cleaned_en_text = clean_with_bedrock(translated_text, language="English", diarize=True)
        except Exception as e:
            log.warning(f"Bedrock polishing failed, falling back to regex: {e}")
            from pipeline import clean_regex
            cleaned_en_text = clean_regex(translated_text)
            
        cleaned_en_text = str(cleaned_en_text or "")

        # 3. Final UI Preparation
        ui_raw = translated_text if translated_text.strip() else html.Span("(empty)", style=PLACEHOLDER_STYLE)
        ui_clean = cleaned_en_text if cleaned_en_text.strip() else html.Span("(empty)", style=PLACEHOLDER_STYLE)

        return ui_raw, ui_clean, cleaned_en_text, translated_text

    except Exception as e:
        log.exception("Translation pipeline failed")
        err = html.Div(f"Translation error: {str(e)}", style={"color": C_ERR})
        # Wiping stores ensures no "ghost" translations from previous successful runs remain
        return err, err, "", ""

# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)