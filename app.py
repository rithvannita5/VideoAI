# ============================================================
# AI Video Studio - Desktop App (PyWebView + Flask)
# ============================================================
import os
import sys
import json
import time
import math
import re
import asyncio
import shutil
import tempfile
import subprocess
import threading
import concurrent.futures

os.environ["PYTHONIOENCODING"] = "utf-8"

# ============================================================
# ពិនិត្យបណ្ណាល័យ
# ============================================================
try:
    import webview
    import edge_tts
    import requests
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    from groq import Groq
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ ខ្វះបណ្ណាល័យ: {e}")
    print("សូមដំឡើង: pip install flask flask-cors pywebview groq requests edge-tts python-dotenv")
    sys.exit(1)

# ============================================================
# ផ្លូវឯកសារ
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

for d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# ផ្ទុក .env
# ============================================================
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "").strip()

if not GROQ_API_KEY:
    print("⚠️  សូមកំណត់ GROQ_API_KEY ក្នុង .env")
if not AGNES_API_KEY:
    print("⚠️  សូមកំណត់ AGNES_API_KEY ក្នុង .env")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

AGNES_BASE_URL = "https://apihub.agnes-ai.com"
AGNES_VIDEO_MODEL = "agnes-video-v2.0"

# ============================================================
# បញ្ជីភាសា
# ============================================================
LANGUAGES = [
    {"name": "ភាសាខ្មែរ (Khmer)", "code": "km", "male": "km-KH-PisethNeural", "female": "km-KH-SreymomNeural"},
    {"name": "English", "code": "en", "male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
    {"name": "中文 (Chinese)", "code": "zh", "male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
    {"name": "ไทย (Thai)", "code": "th", "male": "th-TH-NiwatNeural", "female": "th-TH-PremwadeeNeural"},
    {"name": "Tiếng Việt", "code": "vi", "male": "vi-VN-NamMinhNeural", "female": "vi-VN-HoaiMyNeural"},
    {"name": "日本語 (Japanese)", "code": "ja", "male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
    {"name": "한국어 (Korean)", "code": "ko", "male": "ko-KR-InJoonNeural", "female": "ko-KR-SunHiNeural"},
    {"name": "Français (French)", "code": "fr", "male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    {"name": "Español (Spanish)", "code": "es", "male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
    {"name": "Deutsch (German)", "code": "de", "male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
    {"name": "हिन्दी (Hindi)", "code": "hi", "male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    {"name": "អារ៉ាប់ (Arabic)", "code": "ar", "male": "ar-SA-HamedNeural", "female": "ar-SA-ZariyahNeural"},
]


def get_lang_info(lang_code):
    for lang in LANGUAGES:
        if lang["code"] == lang_code:
            return lang
    return LANGUAGES[0]


# ============================================================
# TTS
# ============================================================
async def _tts_async(text, voice, output_path):
    tts = edge_tts.Communicate(text, voice)
    await tts.save(output_path)


def generate_speech(text, voice, output_path):
    try:
        asyncio.run(_tts_async(text, voice, output_path))
        return os.path.exists(output_path) and os.path.getsize(output_path) > 500
    except Exception as e:
        print(f"TTS error: {e}")
        return False


# ============================================================
# Groq Helper
# ============================================================
def get_active_chat_model():
    if not groq_client:
        return "llama-3.3-70b-versatile"
    try:
        models = groq_client.models.list().data
        chat_models = [
            m.id for m in models
            if "whisper" not in m.id.lower()
            and "tts" not in m.id.lower()
            and "audio" not in m.id.lower()
        ]
        for pref in [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
        ]:
            if pref in chat_models:
                return pref
        if chat_models:
            return chat_models[0]
    except Exception:
        pass
    return "llama-3.3-70b-versatile"


# ============================================================
# Video Utils
# ============================================================
def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def extract_audio(video_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-ab", "64k",
        output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def merge_video_audio(video_path, audio_path, output_path):
    cmd = [
        "ffmpeg", "-fflags", "+igndts", "-y",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def concat_audio_safe(segment_files, output_path, temp_dir):
    """ផ្គុំឯកសារសំឡេងច្រើន (ជាមួយ fallback)"""
    if not segment_files:
        return False
    if len(segment_files) == 1:
        shutil.copyfile(segment_files[0], output_path)
        return True

    inputs = []
    for f in segment_files:
        inputs.extend(["-i", f])

    filter_str = "".join([f"[{i}:a]" for i in range(len(segment_files))])
    filter_str += f"concat=n={len(segment_files)}:v=0:a=1[out]"

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str, "-map", "[out]",
        "-c:a", "libmp3lame", "-ar", "24000", "-ab", "128k",
        output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        concat_list = os.path.join(temp_dir, "concat_audio.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")
        cmd2 = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:a", "libmp3lame", "-ar", "24000", "-ab", "128k",
            output_path
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        return r2.returncode == 0
    return True


def concat_videos(video_files, output_path):
    if not video_files:
        return False
    if len(video_files) == 1:
        shutil.copyfile(video_files[0], output_path)
        return True

    concat_file = os.path.join(TEMP_DIR, "concat_list.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for v in video_files:
            f.write(f"file '{v}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file, "-c", "copy", output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-pix_fmt", "yuv420p", output_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)

    return r.returncode == 0


# ============================================================
# Script Analysis (សម្រាប់បកប្រែ)
# ============================================================
def validate_transcription(text):
    if not text or len(text.strip()) < 3:
        return False, "អត្ថបទខ្លីពេក ឬទទេ"
    cleaned = text.strip()
    digits = len(re.findall(r'[0-9]', cleaned))
    total = len(cleaned.replace(' ', ''))
    if total == 0:
        return False, "អត្ថបទទទេ"
    if (digits / total) > 0.8:
        return False, "អត្ថបទជាលេខសុទ្ធ"
    return True, "OK"


def split_into_sentences(text):
    sentence_endings = r'([.!?।。！？។]+)'
    parts = re.split(sentence_endings, text)
    sentences = []
    current = ""
    for part in parts:
        current += part
        if re.match(sentence_endings, part):
            stripped = current.strip()
            if stripped and len(stripped) > 3:
                sentences.append(stripped)
            current = ""
    if current.strip() and len(current.strip()) > 3:
        sentences.append(current.strip())
    if len(sentences) <= 1:
        sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 3]
        if not sentences:
            sentences = [text.strip()]
    return sentences


def analyze_speakers_v4(transcription_text, active_model):
    sentences = split_into_sentences(transcription_text)
    if len(sentences) == 1:
        text = sentences[0]
        parts = re.split(r'[,;，、]', text)
        if len(parts) >= 2:
            mid = len(parts) // 2
            sentences = [
                ",".join(parts[:mid]).strip(),
                ",".join(parts[mid:]).strip()
            ]
        else:
            words = text.split()
            if len(words) >= 4:
                mid = len(words) // 2
                sentences = [
                    " ".join(words[:mid]),
                    " ".join(words[mid:])
                ]

    segments = []
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        segments.append({
            "speaker": "male" if i % 2 == 0 else "female",
            "text": sentence.strip()
        })
    if not segments:
        segments = [{"speaker": "male", "text": transcription_text}]
    return segments


def translate_segment_single(text, target_lang_name, active_model):
    prompt = (
        f"Translate the following text into {target_lang_name}. "
        f"Output ONLY the translation. No explanations, no notes:\n\n{text}"
    )
    for attempt in range(3):
        try:
            res = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=active_model,
                temperature=0.3
            )
            result = res.choices[0].message.content.strip()
            if result:
                return result
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(1)
    return text


def translate_segments_parallel(segments, target_lang_name, active_model):
    results = [None] * len(segments)

    def translate_one(idx_seg):
        idx, seg = idx_seg
        try:
            translated = translate_segment_single(seg["text"], target_lang_name, active_model)
            return idx, translated
        except Exception:
            return idx, seg["text"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(translate_one, (i, seg)) for i, seg in enumerate(segments)]
        for future in concurrent.futures.as_completed(futures):
            idx, translated = future.result()
            results[idx] = translated

    translated_segments = []
    for i, seg in enumerate(segments):
        translated_segments.append({
            "speaker": seg["speaker"],
            "original": seg["text"],
            "translated": results[i] or seg["text"]
        })
    return translated_segments


def get_voice(speaker, male_voice, female_voice):
    return male_voice if speaker == "male" else female_voice


# ============================================================
# បកប្រែវីដេអូ (ដំណើរការសំខាន់)
# ============================================================
def process_single_video_dub(video_path, target_lang, target_lang_name,
                              male_voice, female_voice, progress_cb):
    """
    បកប្រែវីដេអូមួយ (រក្សាទុកក្នុង OUTPUT_DIR)
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = int(time.time())

    audio_extracted = os.path.join(TEMP_DIR, f"{base_name}_audio_{timestamp}.mp3")
    final_audio = os.path.join(TEMP_DIR, f"{base_name}_final_{timestamp}.mp3")
    output_video = os.path.join(OUTPUT_DIR, f"{base_name}_dubbed_{timestamp}.mp4")

    segment_files = []

    try:
        # 1. ទាញសំឡេង
        progress_cb(5, "កំពុងទាញសំឡេង...")
        if not extract_audio(video_path, audio_extracted):
            raise Exception("ការទាញសំឡេងបរាជ័យ")

        if not os.path.exists(audio_extracted) or os.path.getsize(audio_extracted) < 1000:
            raise Exception("ឯកសារសំឡេងតូចពេក ឬទទេ")

        # 2. Whisper Transcription
        progress_cb(15, "កំពុងបម្លែងសំឡេងទៅជាអក្សរ...")
        with open(audio_extracted, "rb") as a_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.mp3", a_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        transcription_text = str(transcription).strip()

        is_valid, msg = validate_transcription(transcription_text)
        if not is_valid:
            raise Exception(f"បញ្ហាអត្ថបទ: {msg}")

        active_model = get_active_chat_model()

        # 3. វិភាគតួអង្គ
        progress_cb(35, "កំពុងវិភាគតួអង្គ...")
        segments = analyze_speakers_v4(transcription_text, active_model)

        # 4. បកប្រែ
        progress_cb(55, f"កំពុងបកប្រែ {len(segments)} កំណាត់...")
        translated_segments = translate_segments_parallel(segments, target_lang_name, active_model)

        # 5. បង្កើតសំឡេង
        progress_cb(75, "កំពុងបង្កើតសំឡេង AI...")
        successful_segments = []
        male_count = 0
        female_count = 0

        for i, seg in enumerate(translated_segments):
            try:
                voice = get_voice(seg["speaker"], male_voice, female_voice)
                seg_file = os.path.join(TEMP_DIR, f"{base_name}_seg_{timestamp}_{i}.mp3")

                if generate_speech(seg["translated"], voice, seg_file):
                    if os.path.exists(seg_file) and os.path.getsize(seg_file) > 500:
                        segment_files.append(seg_file)
                        successful_segments.append(seg)
                        if seg["speaker"] == "male":
                            male_count += 1
                        else:
                            female_count += 1
            except Exception as e:
                print(f"Segment {i} failed: {e}")

        if not segment_files:
            raise Exception("គ្មានកំណាត់ណាមួយបង្កើតសំឡេងបានសម្រេច!")

        # 6. ផ្គុំសំឡេង
        progress_cb(90, f"កំពុងផ្គុំសំឡេង {len(segment_files)} កំណាត់...")
        if not concat_audio_safe(segment_files, final_audio, TEMP_DIR):
            raise Exception("ការផ្គុំសំឡេងបរាជ័យ")

        if not os.path.exists(final_audio) or os.path.getsize(final_audio) < 1000:
            raise Exception("ឯកសារសំឡេងចុងក្រោយទទេ")

        # 7. ផ្គុំវីដេអូ + សំឡេង
        progress_cb(95, "កំពុងផ្គុំវីដេអូចុងក្រោយ...")
        if not merge_video_audio(video_path, final_audio, output_video):
            raise Exception("ការផ្គុំវីដេអូបរាជ័យ")

        # 8. សង្ខេប
        segments_info = "\n".join([
            f"• [{seg['speaker'].upper()}] {seg['translated'][:100]}"
            for seg in successful_segments[:20]
        ])

        summary = (
            f"អត្ថបទដើម: {transcription_text[:200]}\n\n"
            f"ចំនួនកំណាត់សរុប: {len(translated_segments)}\n"
            f"ចំនួនកំណាត់ជោគជ័យ: {len(successful_segments)}\n"
            f"សំឡេងប្រុស: {male_count} | សំឡេងស្រី: {female_count}\n\n"
            f"{segments_info}"
        )

        return output_video, summary

    finally:
        for f in [audio_extracted, final_audio] + segment_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def dub_video_full(video_path, target_lang, progress_cb=None):
    """
    បកប្រែវីដេអូពេញលេញ
    """
    if progress_cb is None:
        progress_cb = lambda pct, msg="": print(f"[{pct}%] {msg}")

    if not video_path or not os.path.exists(video_path):
        return {"success": False, "error": "វីដេអូមិនត្រឹមត្រូវ"}

    lang_info = get_lang_info(target_lang)
    target_lang_name = lang_info["name"]
    male_voice = lang_info["male"]
    female_voice = lang_info["female"]

    try:
        progress_cb(2, "កំពុងចាប់ផ្តើម...")
        out, summary = process_single_video_dub(
            video_path, target_lang, target_lang_name,
            male_voice, female_voice, progress_cb
        )
        return {
            "success": True,
            "video": out,
            "status": f"✅ ជោគជ័យ!\n\n{summary}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# Agnes AI (បង្កើតវីដេអូ) - រក្សាដូចដើម
# ============================================================
AGNES_CLIP_FRAMES = 121
AGNES_FPS = 24
AGNES_CLIP_SECONDS = AGNES_CLIP_FRAMES / AGNES_FPS

KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
LOW_MEM_ENCODE = ["-c:v", "libx264", "-preset", "ultrafast",
                  "-pix_fmt", "yuv420p"]


def agnes_create_video(prompt, num_frames=121, frame_rate=24,
                       width=1152, height=768, seed=None):
    headers = {
        "Authorization": f"Bearer {AGNES_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": AGNES_VIDEO_MODEL,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": frame_rate
    }
    if seed is not None:
        payload["seed"] = seed

    r = requests.post(f"{AGNES_BASE_URL}/v1/videos",
                      headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def agnes_poll_video(video_id, max_wait=600):
    headers = {"Authorization": f"Bearer {AGNES_API_KEY}"}
    start = time.time()
    while time.time() - start < max_wait:
        r = requests.get(f"{AGNES_BASE_URL}/agnesapi",
                         params={"video_id": video_id},
                         headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = str(data.get("status", "")).lower()
        if status in {"succeeded", "success", "completed", "done"}:
            return data
        if status in {"failed", "error", "cancelled"}:
            raise Exception(f"Agnes បរាជ័យ: {data}")
        print(f"  Agnes {video_id}: {status} ({data.get('progress', 0)}%)")
        time.sleep(5)
    raise TimeoutError("Agnes ហួសពេល")


def agnes_download(video_url, output_path):
    r = requests.get(video_url, stream=True, timeout=120)
    r.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 1000


def parse_scene_script(script_text):
    text = script_text.strip().translate(KHMER_DIGITS)
    blocks = re.split(r'(?=\[\s*ឈុតទី)', text)
    scenes = []
    for block in blocks:
        block = block.strip()
        if not block.startswith("["):
            continue
        m = re.match(
            r'\[\s*ឈុតទី\s*(\d+)\s*[៖:]?\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)[^\]]*\]\s*[-–—]?\s*([^\n]*)\n?(.*)',
            block, re.DOTALL
        )
        if not m:
            continue
        start = int(m.group(2)) * 60 + int(m.group(3))
        end = int(m.group(4)) * 60 + int(m.group(5))
        title = m.group(6).strip()
        body = m.group(7)

        v = re.search(r'វីដេអូ\s*[៖:]\s*(.*?)(?=សំឡេងសម្រាយ|$)', body, re.DOTALL)
        a = re.search(r'សំឡេងសម្រាយ[^:៖\n]*[:៖]\s*(.*)', body, re.DOTALL)

        visual = v.group(1).strip() if v else ""
        voice = a.group(1).strip().strip('"“”\' \n') if a else ""

        if not visual and not voice:
            continue

        scenes.append({
            "speaker": "narration",
            "name": title or f"ឈុតទី {m.group(1)}",
            "khmer_text": voice,
            "visual": visual or title,
            "duration": max(end - start, 0),
        })
    return scenes


def parse_legacy_script(script_text):
    raw_blocks = re.split(r'(?=\[[^\]]+\])', script_text.strip())
    scenes = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        named = re.match(r'^\[([^\]|]+)\|(ប្រុស|ស្រី|male|female)\]\s*[:：]?\s*(.+)$', block, re.DOTALL | re.IGNORECASE)
        gender = re.match(r'^\[(ប្រុស|ស្រី|male|female)\]\s*[:：]?\s*(.+)$', block, re.DOTALL | re.IGNORECASE)
        narr = re.match(r'^\[(និទាន|narration|ទេសភាព|scene)\]\s*[:：]?\s*(.+)$', block, re.DOTALL | re.IGNORECASE)
        if named:
            g = "male" if named.group(2).lower() in ["ប្រុស", "male"] else "female"
            scenes.append({"speaker": g, "name": named.group(1).strip(),
                           "khmer_text": named.group(3).strip(), "visual": "", "duration": 0})
        elif gender:
            g = "male" if gender.group(1).lower() in ["ប្រុស", "male"] else "female"
            scenes.append({"speaker": g, "name": "",
                           "khmer_text": gender.group(2).strip(), "visual": "", "duration": 0})
        elif narr:
            scenes.append({"speaker": "narration", "name": "និទាន",
                           "khmer_text": narr.group(2).strip(), "visual": narr.group(2).strip(), "duration": 0})
        else:
            scenes.append({"speaker": "narration", "name": "និទាន",
                           "khmer_text": block, "visual": block, "duration": 0})
    return scenes


def agnes_generate_shot_prompts(scene, n_shots, active_model):
    visual = scene.get("visual") or scene["khmer_text"]
    style = ", cinematic, photorealistic 4k, natural lighting, realistic skin, smooth motion"

    system_instruction = (
        "You are a film director writing prompts for a text-to-video model. "
        f"Split the Khmer scene description into exactly {n_shots} consecutive shots. "
        "Return ONLY a JSON object, no markdown: "
        '{"subject": "...", "shots": ["...", "..."]}. '
        "\"subject\" = one fixed English description of all characters (age, hair, clothes), "
        "vehicles and location, reused in every shot for consistency. "
        f"\"shots\" = exactly {n_shots} strings in chronological order. Each shot must show a "
        "DIFFERENT moment with a DIFFERENT visible action, camera angle and framing. "
        "Each shot under 35 words, describing only what happens in that moment."
    )

    shots, subject = [], ""
    try:
        res = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Scene (Khmer): {visual}"}
            ],
            model=active_model,
            temperature=0.5
        )
        raw = res.choices[0].message.content.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        subject = str(data.get("subject", "")).strip()
        shots = [str(s).strip() for s in data.get("shots", []) if str(s).strip()]
    except Exception as e:
        print(f"Shot prompt failed: {e}")

    if len(shots) < n_shots:
        stages = ["Opening wide establishing shot:", "Medium shot:", "Close-up:", "Final moment:"]
        shots = [f"{stages[min(i, 3)]} {visual}" for i in range(n_shots)]
        subject = ""

    shots = shots[:n_shots]
    return [f"{s}. {subject}{style}" if subject else f"{s}{style}" for s in shots]


def agnes_download_clip(prompt, width, height, seed, temp_dir, tag):
    last_err = None
    for attempt in range(2):
        try:
            result = agnes_create_video(
                prompt=prompt, num_frames=AGNES_CLIP_FRAMES, frame_rate=AGNES_FPS,
                width=width, height=height, seed=seed + attempt * 7919
            )
            video_id = result.get("video_id") or result.get("id")
            if not video_id:
                raise Exception(f"គ្មាន video_id: {result}")

            final = agnes_poll_video(video_id)
            video_url = final.get("video_url") or final.get("url") or final.get("remixed_from_video_id")
            if not video_url:
                raise Exception(f"គ្មាន video URL")

            path = os.path.join(temp_dir, f"agnes_clip_{tag}.mp4")
            agnes_download(video_url, path)
            if os.path.getsize(path) < 1000:
                raise Exception("clip តូចពេក")
            return path
        except Exception as e:
            last_err = e
            print(f"Clip {tag} attempt {attempt + 1} failed: {e}")
    raise last_err


def agnes_generate_video_for_scene(scene, active_model, session_id, temp_dir,
                                    width, height, narration_voice):
    temp_files = []
    errors = []
    try:
        # 1. សំឡេង
        if scene["speaker"] == "male":
            voice = "km-KH-PisethNeural"
        elif scene["speaker"] == "female":
            voice = "km-KH-SreymomNeural"
        else:
            voice = narration_voice

        audio_file = None
        audio_dur = 0.0
        if scene["khmer_text"].strip():
            audio_file = os.path.join(temp_dir, f"khmer_audio_{session_id}.mp3")
            if generate_speech(scene["khmer_text"], voice, audio_file):
                temp_files.append(audio_file)
                if os.path.getsize(audio_file) > 500:
                    audio_dur = get_video_duration(audio_file)
                else:
                    audio_file = None
            else:
                audio_file = None

        # 2. គោលដៅរយៈពេល
        target = max(scene.get("duration", 0), audio_dur + 0.3, 3.0)
        n_shots = max(1, math.ceil(target / AGNES_CLIP_SECONDS))

        # 3. Prompts
        prompts = agnes_generate_shot_prompts(scene, n_shots, active_model)
        print(f"🎬 {scene['name']}: target {target:.1f}s, {n_shots} shots")

        # 4. បង្កើត clips
        clips = []
        total = 0.0
        MAX_CLIPS = n_shots + 4
        idx = 0
        while total < target - 0.3 and idx < MAX_CLIPS:
            p = prompts[idx] if idx < len(prompts) else (
                prompts[-1] + ", scene continues, new camera angle"
            )
            try:
                path = agnes_download_clip(
                    p, width, height,
                    seed=(session_id * 10 + idx * 1013) % 2147483647,
                    temp_dir=temp_dir, tag=f"{session_id}_{idx}"
                )
                d = get_video_duration(path)
                if d > 0.5:
                    clips.append(path)
                    temp_files.append(path)
                    total += d
                    print(f"   clip {idx+1}: {d:.1f}s (សរុប {total:.1f}/{target:.1f}s)")
            except Exception as e:
                errors.append(f"clip {idx+1}: {str(e)[:150]}")
                print(f"   clip {idx+1} failed: {e}")
                time.sleep(3)
            idx += 1

        scene["clips_ok"] = f"{len(clips)} clips = {total:.1f}s / គោលដៅ {target:.1f}s"
        if not clips:
            return None

        # 5. ផ្គុំ clips
        concat_txt = os.path.join(temp_dir, f"concat_shots_{session_id}.txt")
        temp_files.append(concat_txt)
        with open(concat_txt, "w", encoding="utf-8") as f:
            for c in clips:
                f.write(f"file '{c}'\n")

        shots_file = os.path.join(temp_dir, f"shots_{session_id}.mp4")
        temp_files.append(shots_file)
        r = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
            "-c", "copy", "-an", shots_file
        ], capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
            ] + LOW_MEM_ENCODE + ["-an", shots_file], capture_output=True, text=True)
        if r.returncode != 0:
            return None

        # 6. កែរយៈពេល
        shots_dur = get_video_duration(shots_file)
        vf_parts = []
        if shots_dur > 0 and shots_dur < target:
            factor = min(target / shots_dur, 1.6)
            vf_parts.append(f"setpts={factor:.4f}*PTS")
            shots_dur *= factor
        pad = max(target - shots_dur, 0)
        if pad > 0.05:
            vf_parts.append(f"tpad=stop_mode=clone:stop_duration={pad:.2f}")

        output_file = os.path.join(temp_dir, f"agnes_scene_{session_id}.mp4")
        cmd = ["ffmpeg", "-y", "-i", shots_file]
        if audio_file:
            cmd += ["-i", audio_file, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-map", "0:v:0", "-an"]
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]
        if vf_parts:
            cmd += LOW_MEM_ENCODE
        else:
            cmd += ["-c:v", "copy"]
        cmd += ["-t", f"{target:.2f}", output_file]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(output_file):
            return output_file
        return None

    except Exception as e:
        print(f"Agnes scene failed: {e}")
        return None
    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def create_video_with_agnes(script_text, narration_gender, resolution):
    if not script_text or len(script_text.strip()) < 10:
        return {"success": False, "error": "សូមសរសេរ Script ជាមុនសិន!"}

    session_id = int(time.time())
    width, height = resolution
    narration_voice = "km-KH-PisethNeural" if narration_gender == "male" else "km-KH-SreymomNeural"

    scene_files = []
    try:
        active_model = get_active_chat_model()
        scenes = parse_scene_script(script_text)
        if not scenes:
            scenes = parse_legacy_script(script_text)
        if not scenes:
            return {"success": False, "error": "មិនអាចវិភាគ Script បានទេ!"}

        num_scenes = len(scenes)
        failed = []

        for i, scene in enumerate(scenes):
            print(f"\n=== ឈុត {i+1}/{num_scenes}: {scene['name']} ===")
            f = agnes_generate_video_for_scene(
                scene, active_model, session_id * 100 + i, TEMP_DIR,
                width, height, narration_voice
            )
            if f:
                scene_files.append(f)
            else:
                failed.append(i + 1)

        if not scene_files:
            return {"success": False, "error": "មិនអាចបង្កើតវីដេអូបានទេ!"}

        # ផ្គុំ
        output_video = os.path.join(OUTPUT_DIR, f"agnes_final_{session_id}.mp4")
        if not concat_videos(scene_files, output_video):
            return {"success": False, "error": "ការផ្គុំវីដេអូបរាជ័យ"}

        total = get_video_duration(output_video)
        status = (
            f"✅ ជោគជ័យ!\n"
            f"ចំនួនឈុត: {num_scenes} | ជោគជ័យ: {len(scene_files)}"
            + (f" | បរាជ័យ: ឈុតទី {', '.join(map(str, failed))}" if failed else "")
            + f"\nរយៈពេលសរុប: {total:.1f} វិនាទី | ទំហំ: {width}x{height}\n\n"
            + "\n".join(f"• {s['name']} ({s.get('duration', 0)}s): {s.get('clips_ok', '-')}"
                        for s in scenes[:15])
        )
        return {"success": True, "video": output_video, "status": status}

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        for f in scene_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


# ============================================================
# Flask API
# ============================================================
app = Flask(__name__)
CORS(app)


@app.route('/api/health')
def api_health():
    return jsonify({"status": "ok"})


@app.route('/api/languages')
def api_languages():
    return jsonify(LANGUAGES)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """ទទួលវីដេអូ upload"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "គ្មានឯកសារ"}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({"success": False, "error": "ឈ្មោះទទេ"}), 400

    path = os.path.join(INPUT_DIR, file.filename)
    file.save(path)
    return jsonify({"success": True, "path": path, "name": file.filename})


@app.route('/api/dub_video', methods=['POST'])
def api_dub_video():
    """
    បកប្រែវីដេអូ → រក្សាទុកក្នុង OUTPUT_DIR
    """
    data = request.json
    video_path = data.get('video_path', '').strip()
    target_lang = data.get('target_lang', 'km')

    if not video_path or not os.path.exists(video_path):
        return jsonify({"success": False, "error": "វីដេអូមិនត្រឹមត្រូវ"}), 400

    result = dub_video_full(video_path, target_lang)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 500


@app.route('/api/generate_video', methods=['POST'])
def api_generate_video():
    """បង្កើតវីដេអូដោយ Agnes AI"""
    data = request.json
    script_text = data.get('script_text', '').strip()
    narration_gender = data.get('narration_gender', 'female')
    resolution_str = data.get('resolution', '1152x768')

    if len(script_text) < 10:
        return jsonify({"success": False, "error": "Script ខ្លីពេក"}), 400

    if not AGNES_API_KEY:
        return jsonify({"success": False, "error": "សូមកំណត់ AGNES_API_KEY ក្នុង .env"}), 400

    try:
        w, h = resolution_str.split("x")
        resolution = (int(w), int(h))
    except Exception:
        resolution = (1152, 768)

    result = create_video_with_agnes(script_text, narration_gender, resolution)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 500


@app.route('/api/video/<path:filename>')
def api_get_video(filename):
    """បម្រើឯកសារវីដេអូ"""
    return send_file(filename)


# ============================================================
# PyWebView API
# ============================================================
class DesktopApi:
    def select_video(self):
        """បើកប្រអប់ជ្រើសរើសវីដេអូ"""
        w = webview.windows[0]
        r = w.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=('Video files (*.mp4;*.mov;*.avi;*.mkv)', 'All files (*.*)')
        )
        if r:
            return {"success": True, "path": r[0], "name": os.path.basename(r[0])}
        return {"success": False}

    def open_output_folder(self):
        """បើកថត output"""
        try:
            if os.name == 'nt':
                os.startfile(OUTPUT_DIR)
            elif sys.platform == 'darwin':
                subprocess.run(["open", OUTPUT_DIR])
            else:
                subprocess.run(["xdg-open", OUTPUT_DIR])
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================
# Server + Main
# ============================================================
def start_server():
    app.run(host='127.0.0.1', port=5000, debug=False,
            threaded=True, use_reloader=False)


if __name__ == '__main__':
    print("=" * 60)
    print("  🎬 AI Video Studio")
    print("=" * 60)

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)

    ui_path = os.path.join(BASE_DIR, "ui.html")
    if not os.path.exists(ui_path):
        print(f"❌ រកមិនឃើញ: {ui_path}")
        sys.exit(1)

    api = DesktopApi()
    window = webview.create_window(
        'AI Video Studio',
        ui_path,
        js_api=api,
        width=1400,
        height=900,
        min_size=(1000, 700)
    )

    webview.start()
