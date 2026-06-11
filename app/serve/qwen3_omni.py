#!/usr/bin/env python3
"""
Qwen3-Omni vLLM Version - Flask API for audio transcription
Calls vLLM's OpenAI-compatible API instead of loading the model directly
"""

import base64
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
}

DEFAULT_SYSTEM_PROMPT = (
    "你是專業的聖經講道逐字稿轉錄員。任務：將音訊內容一字不差地轉錄為繁體中文文字。\n"
    "【核心原則】\n"
    "- 只輸出轉錄文字，絕對不加入任何解釋、摘要、標題或元資料。\n"
    "- 忠實還原說話者的每一句話，不得增字、減字、改寫或意譯。\n"
    "- 音訊為片段切割，直接從聽到的內容開始轉錄，不假設前後文。\n"
    "【標點符號】\n"
    "- 句末：句號(。) 問號(？) \n"
    "- 句中：逗號(，) 頓號(、) 冒號(：)\n"
    "【數字】\n"
    "- 所有中文數字一律轉為阿拉伯數字：一→1、十二→12。\n"
    "【聖經章節格式（嚴格遵守）】\n"
    "- 正確格式：《書卷名章:節》，章節數字全部置於《》內。\n"
    "  ✓ 《約翰福音3:16》《啟示錄19:11》《馬太福音24:29-31》\n"
    "  ✓ 只引書名不引節次：《創世記1》《啟示錄》\n"
    "  ✓ 多書並列：《馬太福音24》《啟示錄6》\n"
    "- 書卷名用全稱，不縮寫：約翰福音（不寫「約」）、馬太福音（不寫「太」）。\n"
    "- 說話者若以中文說出章節，轉為標準格式：「約翰福音第三章十六節」→《約翰福音3:16》。\n"
)

DEFAULT_USER_PROMPT = "請逐字轉錄以下音訊，嚴格遵守系統提示的所有格式規則。"

# ==================== Helpers ====================

def get_openai_client():
    return OpenAI(base_url=app_config['vllm_base_url'], api_key="unused")


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

    text = response.choices[0].message.content.strip()

    if enable_s2t and opencc_converter is not None:
        try:
            text = opencc_converter.convert(text)
        except Exception as e:
            print(f"[{request_id}] OpenCC warning: {e}")

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
            return transcribe_audio_file(short_audio, request_id, **kwargs)
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
            text = transcribe_audio_file(seg_audio, f"{request_id}_seg{idx}", **kwargs)
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
        merged = "\n\n".join(results)

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
