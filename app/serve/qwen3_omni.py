#!/usr/bin/env python3
"""
Qwen3-Omni vLLM Version - Flask API for audio transcription
Calls vLLM's OpenAI-compatible API instead of loading the model directly
"""

import base64
import difflib
import io
import os
import re
import subprocess
import sys
import threading
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_file
from openai import OpenAI
from opencc import OpenCC

# ==================== Flask App ====================

flask_app = Flask(__name__)
flask_app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# ==================== Global Config ====================

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(WORK_DIR, "inputs")
TEMP_DIR = os.path.join(WORK_DIR, "temp")
OUTPUTS_DIR = os.path.join(WORK_DIR, "outputs")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

processing_lock = threading.Lock()

opencc_converter = None


def _init_opencc_on_import():
    global opencc_converter
    try:
        opencc_converter = OpenCC('s2tw')
        print("OpenCC S2TW converter initialized")
    except Exception as e:
        print(f"Warning: OpenCC init failed: {e}")

_init_opencc_on_import()

app_config = {
    'vllm_base_url': os.environ.get('VLLM_BASE_URL', 'http://localhost:8901/v1'),
    'model': os.environ.get('VLLM_MODEL', 'cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit'),
    'segment_duration': int(os.environ.get('SEGMENT_DURATION', '60')),
    'max_tokens': int(os.environ.get('MAX_TOKENS', '8192')),
    'temperature': float(os.environ.get('TEMPERATURE', '0.1')),
    'concurrent_segments': int(os.environ.get('CONCURRENT_SEGMENTS', '4')),
    # Upper bound on simultaneous vLLM requests across all segments (each segment
    # issues several chunk-level requests).
    'max_inflight': int(os.environ.get('MAX_INFLIGHT', '32')),
    # Text-only LLM pass that applies number / Bible-reference / punctuation rules.
    'format_pass': os.environ.get('FORMAT_PASS', '1') == '1',
}

# Only used when a caller supplies its own prompt/system_prompt (single-pass mode).
# The default pipeline never puts formatting rules in front of the audio: rules
# like the Bible-reference format hijack the model on some clips (it emits a
# made-up citation or nothing instead of transcribing).
DEFAULT_SYSTEM_PROMPT = (
    "你是專業的聖經講道逐字稿轉錄員。\n"
    "音訊可能是韓語講座的口譯錄音：韓語原音與中文口譯在同一音軌上輪流或重疊出現。\n"
    "【任務】只轉錄實際用中文說出的話，輸出繁體中文。\n"
    "- 韓語一律略過：不轉錄、不翻譯、不輸出任何韓文字。\n"
    "- 中文部分一字不差：不增字、不減字、不改寫、不摘要、不加任何說明或標題。\n"
    "- 若沒有任何中文語音（只有韓語、音樂、掌聲或靜音），只輸出：（無中文）\n"
)

DEFAULT_USER_PROMPT = "請逐字轉錄這段音訊中所有的中文語音（略過韓語）。"

# Pass A: plain ASR per chunk. Keeping every utterance in its spoken language
# lets us identify Korean by script and drop it deterministically, instead of
# asking the model to skip it (which makes it translate or drop Chinese).
ASR_PROMPT = "Transcribe the audio into text in the original spoken language. Do not translate."
# Pass B: the clip is already Chinese-only. On long clips ASR_PROMPT sometimes
# yields just "...", so use bare ASR prompts, the second as a retry.
RUN_ASR_PROMPTS = ("Transcribe the audio into text.", "请将这段中文语音转换为纯文本。")

# Pass C: chunks whose plain ASR came back Korean may still carry an interpreter
# talking over the Korean speaker; pull out only the Chinese that is spoken.
# Asked directly, the model translates short Korean phrases ("예수님은" → 耶穌說)
# about as often as it finds real overlapping Chinese, so a yes/no check gates it.
HAS_CHINESE_PROMPT = ("Is anyone speaking Mandarin Chinese in this audio "
                      "(even quietly or overlapping with Korean)? Answer only yes or no.")
NO_CHINESE_MARK = "（無中文）"
EXTRACT_SYSTEM_PROMPT = (
    "這段音訊來自韓語講座的同步/交替口譯錄音，韓語原音與中文口譯可能先後出現，也可能互相重疊。\n"
    "任務：只把音訊中「實際有人用中文說出來」的話逐字寫下。\n"
    "- 韓語一律略過，絕對不要翻譯韓語。\n"
    "- 不增字、不減字、不改寫。\n"
    f"- 如果沒有任何人說中文，只輸出：{NO_CHINESE_MARK}"
)
EXTRACT_USER_PROMPT = "請逐字寫下音訊中用中文說出的話。"

# Pass D: text-only formatting of the finished transcript.
FORMAT_SYSTEM_PROMPT = (
    "你是逐字稿排版員。使用者會給你一段由語音辨識產生的中文逐字稿（聖經講座的口譯內容）。"
    "你只能調整格式，輸出排版後的全文，不加任何說明。\n"
    "【絕對禁止】\n"
    "- 不得增加、刪除、替換或重新排列任何字詞；口語贅詞（呃、嗯、啊、哎、那）與重複的詞都要原樣保留。\n"
    "- 不得改正你認為聽錯的字，不得潤飾、摘要或翻譯。\n"
    "- 保持原文的字體（繁體或簡體）不變。\n"
    "【要做的事】\n"
    "1. 補上或修正標點：句末用句號（。）或問號（？），句中用逗號（，）、頓號（、）、冒號（：）。\n"
    "2. 表示年份、日期、數量、時間、年紀、章節的中文數字改為阿拉伯數字，例如：二零二五年五月十一號→2025年5月11號、"
    "一九九零年→1990年、六百萬年前→600萬年前、三十六年→36年。成語與固定用語保持原樣，例如：一樣、一些、統一、一五一十、第一次、一個。\n"
    "3. 只有當文中真的說出聖經書卷與章節時，才改寫為《書卷全名章:節》，例如：約翰福音第三章十六節→《約翰福音3:16》、"
    "馬太福音二十四章二十九到三十一節→《馬太福音24:29-31》、啟示錄十九章→《啟示錄19》。書卷名用全稱。"
    "文中沒有說出的經文絕不可自行加入。\n"
)

HANGUL_RE = re.compile(r'[가-힣ᄀ-ᇿ㄰-㆏]')
HAN_RE = re.compile(r'[一-鿿]')
# A Korean stretch: starts and ends with Hangul, may contain spaces, digits,
# Latin and ASCII punctuation in between; trailing punctuation goes with it.
KOREAN_SPAN_RE = re.compile(
    r'[가-힣ᄀ-ᇿ㄰-㆏]'
    r'(?:[가-힣ᄀ-ᇿ㄰-㆏\sA-Za-z0-9.,!?~\'"\-]*[가-힣ᄀ-ᇿ㄰-㆏])?'
    r'[\s.,!?~。，！？、]*'
)
# A phrase of 4-50 chars repeated 5+ times back to back is a decoder loop; real
# speech repeats (哈利路亞 x3) stay.
LOOP_RE = re.compile(r'(.{4,50}?)\1{4,}', re.S)
CHINESE_NUMERAL_CHARS = set('零〇一二兩三四五六七八九十百千萬億0123456789')

# OpenCC s2tw phrase-level slips seen in real transcripts.
OPENCC_FIXES = [
    (re.compile(r'(闡|說|證|表|聲|發|查|指|標|註|講)明瞭'), r'\1明了'),   # 闡明了 → 闡明瞭
    (re.compile(r'瞭解決'), '了解決'),                                # 為了解決 → 為瞭解決
    (re.compile(r'(?<=[是不也或還就都有])隻(?=[有要是能會在做給為好剩用治擅挑])'), '只'),  # 不是只有 → 不是隻有
    (re.compile(r'阿門'), '阿們'),  # 阿门 → 阿門; Taiwanese church transcripts write 阿們
]

# ==================== Helpers ====================

_client = None
_client_lock = threading.Lock()
_request_pool = None


def get_openai_client():
    global _client
    with _client_lock:
        if _client is None or str(_client.base_url).rstrip('/') != app_config['vllm_base_url'].rstrip('/'):
            _client = OpenAI(base_url=app_config['vllm_base_url'], api_key="unused")
        return _client


def get_request_pool():
    """Shared pool for individual vLLM calls. Segment workers submit here and wait,
    so its tasks never block on each other."""
    global _request_pool
    with _client_lock:
        if _request_pool is None:
            _request_pool = ThreadPoolExecutor(max_workers=app_config['max_inflight'])
        return _request_pool


def init_opencc():
    global opencc_converter
    if opencc_converter is None:
        try:
            opencc_converter = OpenCC('s2tw')
            print("OpenCC S2TW converter initialized")
        except Exception as e:
            print(f"Warning: OpenCC init failed: {e}")


def sanitize_filename(filename):
    filename = os.path.basename(filename)
    filename = re.sub(r'[/\\:*?"<>|]', '', filename)
    filename = re.sub(r'\s+', ' ', filename)
    return filename[:255]


def audio_to_base64(audio_path):
    """Read audio file and return base64 data URL"""
    with open(audio_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    return f"data:audio/wav;base64,{b64}"


# ==================== Audio Processing ====================

def convert_to_wav(input_path, request_id):
    """Convert any audio/video to 16kHz mono WAV using FFmpeg"""
    output_path = os.path.join(TEMP_DIR, f"{request_id}_converted.wav")
    print(f"[{request_id}] Converting to WAV: {input_path} -> {output_path}")

    cmd = [
        'ffmpeg', '-i', input_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        '-y', output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("FFmpeg produced empty output")

    print(f"[{request_id}] Conversion done: {os.path.getsize(output_path)} bytes")
    return output_path


def normalize_audio(input_path, request_id):
    """Normalize audio loudness with FFmpeg loudnorm (EBU R128), output 16kHz mono WAV"""
    output_path = os.path.join(TEMP_DIR, f"{request_id}_normalized.wav")
    print(f"[{request_id}] Normalizing audio loudness...")
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vn',
        '-af', 'loudnorm=I=-16:LRA=11:TP=-1.5',
        '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        '-y', output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        stderr = result.stderr.decode('utf-8', errors='ignore')
        print(f"[{request_id}] Warning: Normalization failed, using original audio. {stderr[-300:]}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None
    print(f"[{request_id}] Normalization done: {os.path.getsize(output_path)} bytes")
    return output_path


def transcribe_audio_file(audio_path, request_id, user_prompt=None, system_prompt=None, enable_s2t=True):
    """
    Send a single audio file to vLLM and return transcription text.
    Audio is base64-encoded and sent via the OpenAI-compatible API.
    """
    client = get_openai_client()
    prompt = user_prompt or DEFAULT_USER_PROMPT
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    print(f"[{request_id}] Encoding audio...")
    audio_data_url = audio_to_base64(audio_path)

    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_data_url}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    print(f"[{request_id}] Calling vLLM API...")
    response = client.chat.completions.create(
        model=app_config['model'],
        messages=messages,
        max_tokens=app_config['max_tokens'],
        temperature=app_config['temperature'],
    )

    text = (response.choices[0].message.content or '').strip()
    if text == NO_CHINESE_MARK:
        text = ''

    if enable_s2t:
        text = to_traditional(text, request_id)

    print(f"[{request_id}] Transcription: {len(text)} chars")
    return text


def to_traditional(text, request_id=''):
    if opencc_converter is None or not text:
        return text
    try:
        text = opencc_converter.convert(text)
    except Exception as e:
        print(f"[{request_id}] OpenCC warning: {e}")
        return text
    for pattern, repl in OPENCC_FIXES:
        text = pattern.sub(repl, text)
    return text


# ==================== Chunked bilingual pipeline ====================
#
# Recordings of a Korean lecture with a Chinese interpreter on the same track
# (consecutive or simultaneous) break single-pass "transcribe to Chinese"
# prompts: the model transcribes or translates the Korean, loops on Korean-only
# stretches, and drops or invents content. Per segment we instead:
#   A. split on pauses and run plain ASR on each chunk; the script of the result
#      (Hangul vs Han) tells us the chunk's language,
#   B. re-transcribe each run of consecutive Chinese chunks as one Chinese-only
#      clip (more context, no Korean to confuse the model), cross-checked
#      against A,
#   C. ask Korean chunks only for Chinese spoken over them (simultaneous
#      interpretation),
#   D. apply formatting rules in a text-only pass, guarded against rewrites.

def _wav_data_url(samples, sr):
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format='WAV', subtype='PCM_16')
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')


def _max_tokens_for(seconds):
    # ~12 tokens/s is well above real speech rates; the cap stops decoder loops
    # from running to MAX_TOKENS.
    return min(app_config['max_tokens'], 64 + int(seconds * 12))


def _chat(messages, max_tokens):
    response = get_openai_client().chat.completions.create(
        model=app_config['model'],
        messages=messages,
        max_tokens=max_tokens,
        temperature=app_config['temperature'],
    )
    choice = response.choices[0]
    return (choice.message.content or '').strip(), choice.finish_reason


def _audio_chat(samples, sr, user_text, system_text=None, max_tokens=None):
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": _wav_data_url(samples, sr)}},
        {"type": "text", "text": user_text},
    ]})
    return _chat(messages, max_tokens or _max_tokens_for(len(samples) / sr))


def _speech_chunks(samples, sr, max_chunk_sec=12.0):
    """Split a segment into utterance-sized (start, end) sample ranges at pauses."""
    intervals = librosa.effects.split(samples, top_db=35, frame_length=1024, hop_length=256)
    merged = []
    for s, e in intervals:
        if merged and s - merged[-1][1] < 0.25 * sr:
            merged[-1][1] = e
        else:
            merged.append([int(s), int(e)])
    chunks = []
    for s, e in merged:
        if chunks and e - chunks[-1][0] <= max_chunk_sec * sr and s - chunks[-1][1] < 0.6 * sr:
            chunks[-1][1] = e
        else:
            chunks.append([s, e])
    pad = int(0.1 * sr)
    return [(max(0, s - pad), min(len(samples), e + pad)) for s, e in chunks if e - s > 0.3 * sr]


def _script_lang(text):
    hangul = len(HANGUL_RE.findall(text))
    han = len(HAN_RE.findall(text))
    if hangul > han:
        return 'ko'
    return 'zh' if han else None


def _clean(text):
    """Drop Korean stretches and collapse decoder loops."""
    text = KOREAN_SPAN_RE.sub('', text)
    text = LOOP_RE.sub(r'\1', text)
    text = re.sub(r'[ \t]*\n+[ \t]*', '', text)
    text = re.sub(r'([，。、！？：；,.!?])[，。、！？：；,.!?\s]+', r'\1', text)
    return text.strip(' ，、,')


def _han_only(text):
    return ''.join(HAN_RE.findall(text))


def _coverage(a, b):
    """Fraction of a's and of b's characters that the other text also has, in order."""
    if not a or not b:
        return (0.0 if a else 1.0), (0.0 if b else 1.0)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(a), matched / len(b)


def _content_key(text):
    """Han characters with numerals and chapter/verse words removed, so the
    formatting pass's legitimate edits don't count as content changes."""
    return ''.join(ch for ch in _han_only(text) if ch not in CHINESE_NUMERAL_CHARS and ch not in '第章節节篇')


def _format_text(text, request_id):
    """Pass D. Returns the formatted text, or the input if the model changed content."""
    try:
        out, finish = _chat([
            {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ], max_tokens=min(app_config['max_tokens'], 64 + 2 * len(text)))
    except Exception as e:
        print(f"[{request_id}] Format pass failed, keeping raw text: {e}")
        return text
    cov_in, cov_out = _coverage(_content_key(text), _content_key(out))
    # Every book name the pass puts in 《》 must already be in the text.
    invented = [name for name in re.findall(r'《([^》\d:：\-]+)', out) if name not in text]
    if finish != 'stop' or cov_in < 0.95 or cov_out < 0.95 or invented:
        print(f"[{request_id}] Format pass rejected (coverage {cov_in:.2f}/{cov_out:.2f}, invented={invented})")
        return text
    return out


def transcribe_segment(samples, sr, request_id, enable_s2t=True):
    """Chinese-only transcript of one segment of a possibly Korean/Chinese recording."""
    pool = get_request_pool()
    chunks = _speech_chunks(samples, sr)
    if not chunks:
        return ''

    # Pass A: plain ASR per chunk -> language by script.
    asr = list(pool.map(lambda c: _audio_chat(samples[c[0]:c[1]], sr, ASR_PROMPT), chunks))
    langs = [_script_lang(text) for text, _ in asr]

    # Group into ordered items: runs of consecutive Chinese chunks, single Korean
    # chunks. Chunks with neither script (music, noise) are skipped.
    items = []
    for idx, lang in enumerate(langs):
        if lang == 'zh':
            if items and items[-1][0] == 'zh':
                items[-1][1].append(idx)
            else:
                items.append(('zh', [idx]))
        elif lang == 'ko':
            items.append(('ko', [idx]))

    gap = np.zeros(int(0.4 * sr), dtype=samples.dtype)

    def _resolve(item):
        kind, idxs = item
        if kind == 'ko':
            # Pass C
            c = chunks[idxs[0]]
            try:
                answer, _ = _audio_chat(samples[c[0]:c[1]], sr, HAS_CHINESE_PROMPT, max_tokens=4)
                if not answer.lower().startswith('yes'):
                    return '', 'ko-none'
                text, finish = _audio_chat(samples[c[0]:c[1]], sr, EXTRACT_USER_PROMPT, EXTRACT_SYSTEM_PROMPT)
            except Exception as e:
                print(f"[{request_id}] Chinese extraction from Korean chunk failed: {e}")
                return '', 'ko-none'
            if finish != 'stop' or NO_CHINESE_MARK in text:
                return '', 'ko-none'
            text = _clean(text)
            return (text, 'ko-zh') if HAN_RE.search(text) else ('', 'ko-none')
        joined = ''.join(_clean(asr[i][0]) for i in idxs)
        if len(idxs) == 1:
            return joined, 'zh-a'
        # Pass B
        clip = np.concatenate([np.concatenate([samples[chunks[i][0]:chunks[i][1]], gap]) for i in idxs])
        for prompt in RUN_ASR_PROMPTS:
            try:
                text, finish = _audio_chat(clip, sr, prompt)
            except Exception as e:
                print(f"[{request_id}] Run re-transcription failed: {e}")
                continue
            text = _clean(text)
            cov_a, cov_b = _coverage(_han_only(joined), _han_only(text))
            # B has context A lacks but occasionally skips a sentence or returns
            # nothing; A occasionally loops on a long chunk. Take B when the two
            # agree or when B contains everything A has.
            if finish == 'stop' and ((cov_a >= 0.85 and cov_b >= 0.85) or cov_a >= 0.9):
                return text, 'zh-b'
            print(f"[{request_id}] Run of {len(idxs)} chunks: re-transcription disagrees "
                  f"(coverage {cov_a:.2f}/{cov_b:.2f}, finish={finish}): {text[:40]!r}")
        return joined, 'zh-a'

    resolved = list(pool.map(_resolve, items))
    text = ''.join(t for t, _ in resolved)

    stats = {}
    for _, how in resolved:
        stats[how] = stats.get(how, 0) + 1
    zh_sec = sum(chunks[i][1] - chunks[i][0] for i, l in enumerate(langs) if l == 'zh') / sr
    ko_sec = sum(chunks[i][1] - chunks[i][0] for i, l in enumerate(langs) if l == 'ko') / sr
    print(f"[{request_id}] {len(chunks)} chunks (zh {zh_sec:.0f}s, ko {ko_sec:.0f}s) -> {stats}")

    if not HAN_RE.search(text):
        return ''
    if enable_s2t:
        text = to_traditional(text, request_id)
    if app_config['format_pass']:
        text = _format_text(text, request_id)
        if enable_s2t:
            text = to_traditional(text, request_id)
    return text


def transcribe_wav(audio_path, request_id, user_prompt=None, system_prompt=None, enable_s2t=True):
    """Custom prompts keep the single-pass behaviour; otherwise use the chunked pipeline."""
    if user_prompt or system_prompt:
        return transcribe_audio_file(audio_path, request_id, user_prompt=user_prompt,
                                     system_prompt=system_prompt, enable_s2t=enable_s2t)
    samples, sr = librosa.load(audio_path, sr=16000, mono=True)
    text = transcribe_segment(samples, sr, request_id, enable_s2t=enable_s2t)
    print(f"[{request_id}] Transcription: {len(text)} chars")
    return text


def get_silence_based_split_points(audio_array, sr, segment_duration, top_db=30, search_window_sec=60):
    """
    Find split points near segment_duration boundaries that fall within silence gaps.
    Returns a list of sample indices where audio should be split.
    """
    total_samples = len(audio_array)
    target_samples = int(segment_duration * sr)
    search_window = int(search_window_sec * sr)

    # Find non-silent intervals; each row is [start_sample, end_sample]
    non_silent = librosa.effects.split(audio_array, top_db=top_db)

    # Collect midpoints of silence gaps between consecutive non-silent intervals
    silence_mids = []
    for i in range(len(non_silent) - 1):
        gap_start = int(non_silent[i][1])
        gap_end = int(non_silent[i + 1][0])
        if gap_end > gap_start:
            silence_mids.append((gap_start + gap_end) // 2)

    split_points = []
    target = target_samples
    while target < total_samples:
        best = None
        best_dist = float('inf')
        for sm in silence_mids:
            dist = abs(sm - target)
            if dist < search_window and dist < best_dist:
                best = sm
                best_dist = dist
        split_point = best if best is not None else target
        split_points.append(split_point)
        target = split_point + target_samples

    return split_points


def process_audio_segments(audio_path, request_id, segment_duration=None, segment_start=0, output_path=None, **kwargs):
    """
    Split long audio into segments and transcribe each via vLLM.
    Results are appended to a text file as each segment completes.
    """
    if segment_duration is None:
        segment_duration = app_config['segment_duration']
    print(f"[{request_id}] Loading audio: {audio_path}")

    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_path = os.path.join(OUTPUTS_DIR, f"{request_id}_{timestamp}.txt")

    # Decode to raw 16kHz mono WITHOUT loudnorm. Loudness normalization is done
    # per-segment in parallel below (multi-core) rather than once over the whole
    # file, which would be a single-threaded serial prefix.
    converted_path = None
    try:
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        converted_path = convert_to_wav(audio_path, request_id)
        audio_array, sr = librosa.load(converted_path, sr=16000, mono=True)

    duration = len(audio_array) / sr
    print(f"[{request_id}] Duration: {duration:.1f}s ({duration/60:.1f} mins)")

    # Short audio: normalize the whole (short) file once and process directly
    if duration <= segment_duration:
        print(f"[{request_id}] Short audio, processing directly")
        normalized_path = normalize_audio(audio_path, request_id)
        short_audio = normalized_path or converted_path or audio_path
        try:
            return transcribe_wav(short_audio, request_id, **kwargs)
        finally:
            for p in [normalized_path, converted_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    # Split at silence boundaries near segment_duration intervals
    print(f"[{request_id}] Detecting silence boundaries...")
    split_points = get_silence_based_split_points(audio_array, sr, segment_duration)

    min_samples = int(5 * sr)
    segments = []
    prev = 0
    for sp in split_points:
        chunk = audio_array[prev:sp]
        if len(chunk) >= min_samples:
            segments.append(chunk)
        prev = sp
    # Last (tail) chunk
    tail = audio_array[prev:]
    if len(tail) >= min_samples:
        segments.append(tail)
    elif segments:
        # Merge short tail into last segment
        segments[-1] = np.concatenate([segments[-1], tail])

    print(f"[{request_id}] Split into {len(segments)} segments (silence-aware, target {segment_duration}s each)")

    if segment_start >= len(segments):
        return f"Error: segment_start ({segment_start}) >= total segments ({len(segments)})"

    segment_indices = list(range(segment_start, len(segments)))
    num_workers = min(len(segment_indices), app_config['concurrent_segments'])
    results_map = {}

    def _process_segment(idx):
        # Write the raw slice, then loudnorm it. Running inside the
        # ThreadPoolExecutor means up to `concurrent_segments` ffmpeg loudnorm
        # processes run at once -> spreads normalization across CPU cores.
        seg_path = os.path.join(TEMP_DIR, f"{request_id}_seg{idx}_raw.wav")
        sf.write(seg_path, segments[idx], sr)
        norm_path = None
        try:
            start_mins = (idx * segment_duration) / 60
            print(f"[{request_id}] Segment {idx+1}/{len(segments)} started (at {start_mins:.1f} min)...")
            norm_path = normalize_audio(seg_path, f"{request_id}_seg{idx}")
            seg_audio = norm_path or seg_path
            text = transcribe_wav(seg_audio, f"{request_id}_seg{idx}", **kwargs)
            print(f"[{request_id}] Segment {idx+1}/{len(segments)} done ({len(text)} chars)")
            return idx, text, None
        except Exception as e:
            return idx, None, e
        finally:
            for p in [seg_path, norm_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    try:
        print(f"[{request_id}] Processing {len(segment_indices)} segments with {num_workers} concurrent workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_segment, idx): idx for idx in segment_indices}
            for future in as_completed(futures):
                idx, text, err = future.result()
                if err:
                    results_map[idx] = f"[Error in segment {idx}: {err}]"
                    print(f"[{request_id}] Segment {idx+1} error: {err}")
                else:
                    results_map[idx] = text

        results = [results_map[idx] for idx in segment_indices]
        merged = "\n\n".join(r for r in results if r)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(merged)
        print(f"[{request_id}] Done. Total: {len(merged)} chars, written to {output_path}")
        return merged

    finally:
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except Exception:
                pass


# ==================== Flask Routes ====================

@flask_app.route('/health', methods=['GET'])
def health_check():
    """Check health of this service and vLLM backend"""
    vllm_ok = False
    vllm_model = None
    try:
        client = get_openai_client()
        models = client.models.list()
        vllm_ok = True
        vllm_model = models.data[0].id if models.data else None
    except Exception as e:
        print(f"vLLM health check failed: {e}")

    return jsonify({
        "status": "ok" if vllm_ok else "degraded",
        "vllm_connected": vllm_ok,
        "vllm_model": vllm_model,
        "vllm_url": app_config['vllm_base_url'],
        "timestamp": datetime.now().isoformat()
    })


@flask_app.route('/transcribe', methods=['POST'])
def transcribe():
    return _transcribe_impl(return_format='file')


@flask_app.route('/transcribe/json', methods=['POST'])
def transcribe_json():
    return _transcribe_impl(return_format='json')


def _transcribe_impl(return_format='file'):
    if 'file' not in request.files and not request.data:
        return jsonify({"error": "No audio file provided"}), 400

    if 'file' in request.files:
        file = request.files['file']
        filename = sanitize_filename(file.filename or 'audio.wav')
        audio_data = file.read()
    else:
        audio_data = request.data
        filename = "audio.wav"

    if not audio_data:
        return jsonify({"error": "Empty audio file"}), 400

    request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    input_path = os.path.join(INPUTS_DIR, f"{request_id}_{filename}")

    with open(input_path, 'wb') as f:
        f.write(audio_data)

    print(f"[{request_id}] Received: {filename} ({len(audio_data)} bytes)")

    try:
        with processing_lock:
            segment_duration = int(request.form.get('segment_duration', app_config['segment_duration']))
            segment_start = int(request.args.get('segment_start', request.form.get('segment_start', 0)))
            enable_s2t = request.form.get('enable_s2t', 'true').lower() == 'true'
            custom_prompt = request.form.get('prompt', None)
            custom_system_prompt = request.form.get('system_prompt', None)

            transcription = process_audio_segments(
                input_path,
                request_id,
                segment_duration=segment_duration,
                segment_start=segment_start,
                enable_s2t=enable_s2t,
                user_prompt=custom_prompt,
                system_prompt=custom_system_prompt,
            )

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_filename = f"{timestamp}.txt"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)

        if return_format == 'json':
            return jsonify({
                "status": "success",
                "transcription": transcription,
                "output_file": output_filename,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return send_file(
                output_path,
                as_attachment=True,
                download_name='transcription.txt',
                mimetype='application/octet-stream'
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass


# ==================== Entry Point ====================

def _get_args():
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--vllm-url', type=str, default=None,
                        help='vLLM base URL (default: VLLM_BASE_URL env or http://localhost:8901/v1)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (default: VLLM_MODEL env)')
    parser.add_argument('--segment-duration', type=int, default=60,
                        help='Audio segment duration in seconds (default: 300)')
    parser.add_argument('--max-tokens', type=int, default=int(os.environ.get('MAX_TOKENS', '8192')))
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('TEMPERATURE', '0.1')))
    parser.add_argument('--file', type=str, default=None,
                        help='Process a single file locally and exit (skips Flask server)')
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    if args.vllm_url:
        app_config['vllm_base_url'] = args.vllm_url
    if args.model:
        app_config['model'] = args.model
    app_config['segment_duration'] = args.segment_duration
    app_config['max_tokens'] = args.max_tokens
    app_config['temperature'] = args.temperature

    # --file mode: process a single file locally and exit
    if args.file:
        stem = sanitize_filename(Path(args.file).name)
        stem_no_ext = Path(stem).stem
        request_id = re.sub(r'[^\w\-]', '_', stem_no_ext)[:60]
        out_path = os.path.join(OUTPUTS_DIR, f"{stem_no_ext}.txt")
        print(f"[INFO] vLLM URL: {app_config['vllm_base_url']}")
        print(f"[INFO] File:     {args.file}")
        print(f"[INFO] Output:   {out_path}")
        result = process_audio_segments(
            args.file,
            request_id,
            segment_duration=args.segment_duration,
            output_path=out_path,
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"[DONE] Saved: {out_path}")
        sys.exit(0)

    print(f"[INFO] vLLM URL:        {app_config['vllm_base_url']}")
    print(f"[INFO] Model:           {app_config['model']}")
    print(f"[INFO] Segment:         {app_config['segment_duration']}s")
    print(f"[INFO] Flask:           {args.host}:{args.port}")
    print("[INFO] Endpoints:")
    print("  GET  /health")
    print("  POST /transcribe         (returns .txt file)")
    print("  POST /transcribe/json    (returns JSON)")
    print("[INFO] Query params:")
    print("  segment_start=N  - resume from segment N (0-based)")

    flask_app.run(host=args.host, port=args.port, debug=False, threaded=True)
