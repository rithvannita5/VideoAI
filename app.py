import os
import sys
import shutil
import json
import math
import re
import time
import base64
import concurrent.futures

os.environ["PYTHONIOENCODING"] = "utf-8"

import subprocess
import asyncio
import tempfile
import requests
import gradio as gr
from groq import Groq
import edge_tts

# ============================================================
# API Keys
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY", "")

if not GROQ_API_KEY:
    raise ValueError("សូមកំណត់ GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY.strip())

POLLINATIONS_BASE = "https://gen.pollinations.ai"
POLLINATIONS_VIDEO_MODEL = "veo-3-fast"


# ============================================================
# បញ្ជីភាសា
# ============================================================
LANGUAGES = [
    ("ភាសាខ្មែរ (Khmer)", "km", "km-KH-PisethNeural", "km-KH-SreymomNeural"),
    ("English", "en", "en-US-GuyNeural", "en-US-JennyNeural"),
    ("中文 (Chinese)", "zh", "zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"),
    ("ไทย (Thai)", "th", "th-TH-NiwatNeural", "th-TH-PremwadeeNeural"),
    ("Tiếng Việt (Vietnamese)", "vi", "vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"),
    ("日本語 (Japanese)", "ja", "ja-JP-KeitaNeural", "ja-JP-NanamiNeural"),
    ("한국어 (Korean)", "ko", "ko-KR-InJoonNeural", "ko-KR-SunHiNeural"),
    ("Français (French)", "fr", "fr-FR-HenriNeural", "fr-FR-DeniseNeural"),
    ("Español (Spanish)", "es", "es-ES-AlvaroNeural", "es-ES-ElviraNeural"),
    ("Deutsch (German)", "de", "de-DE-ConradNeural", "de-DE-KatjaNeural"),
    ("हिन्दी (Hindi)", "hi", "hi-IN-MadhurNeural", "hi-IN-SwaraNeural"),
    ("អារ៉ាប់ (Arabic)", "ar", "ar-SA-HamedNeural", "ar-SA-ZariyahNeural"),
]


def get_lang_info(lang_code):
    for name, code, male_voice, female_voice in LANGUAGES:
        if code == lang_code:
            return name, male_voice, female_voice
    return "English", "en-US-GuyNeural", "en-US-JennyNeural"


async def generate_speech_segment(text, voice, output_path):
    tts = edge_tts.Communicate(text, voice)
    await tts.save(output_path)


def get_active_chat_model():
    try:
        models = groq_client.models.list().data
        chat_models = [
            m.id for m in models
            if "whisper" not in m.id.lower()
            and "orpheus" not in m.id.lower()
            and "tts" not in m.id.lower()
            and "audio" not in m.id.lower()
        ]
        for preferred in [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "gemma2-9b-it",
        ]:
            if preferred in chat_models:
                return preferred
        if chat_models:
            return chat_models[0]
    except Exception:
        pass
    return "llama-3.3-70b-versatile"


def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def split_video(video_path, segment_minutes, temp_dir):
    duration = get_video_duration(video_path)
    if duration <= 0:
        return [], 0

    segment_seconds = segment_minutes * 60
    num_segments = math.ceil(duration / segment_seconds)

    segments = []
    for i in range(num_segments):
        start_time = i * segment_seconds
        seg_path = os.path.join(temp_dir, f"segment_{i}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(segment_seconds),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            seg_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(seg_path):
            segments.append(seg_path)

    return segments, duration


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


def safe_concat_audio(segment_files, output_path, temp_dir):
    if not segment_files:
        raise Exception("គ្មានឯកសារសំឡេងសម្រាប់ផ្គុំ")

    if len(segment_files) == 1:
        shutil.copyfile(segment_files[0], output_path)
        return

    inputs = []
    for f in segment_files:
        inputs.extend(["-i", f])

    filter_parts = []
    for i in range(len(segment_files)):
        filter_parts.append(f"[{i}:a]")
    filter_str = "".join(filter_parts) + f"concat=n={len(segment_files)}:v=0:a=1[out]"

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:a", "libmp3lame", "-ar", "24000", "-ab", "128k",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        concat_list = os.path.join(temp_dir, "concat_fallback.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        cmd2 = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:a", "libmp3lame", "-ar", "24000", "-ab", "128k",
            output_path
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            raise Exception(f"ការផ្គុំសំឡេងបរាជ័យ: {result2.stderr[:300]}")


def process_single_video(video_path, target_lang, target_lang_name,
                          male_voice, female_voice, progress, progress_start, progress_end):
    temp_dir = tempfile.gettempdir()
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    audio_extracted = os.path.join(temp_dir, f"{base_name}_audio.mp3")
    final_audio = os.path.join(temp_dir, f"{base_name}_final.mp3")
    output_video = os.path.join(temp_dir, f"{base_name}_dubbed.mp4")

    segment_files = []

    try:
        progress(progress_start + 0.05 * (progress_end - progress_start),
                 desc="កំពុងទាញសំឡេង...")

        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-ab", "64k",
            audio_extracted
        ]
        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"ការទាញសំឡេងបរាជ័យ: {result.stderr[:200]}")

        if not os.path.exists(audio_extracted) or os.path.getsize(audio_extracted) < 1000:
            raise Exception("ឯកសារសំឡេងតូចពេក ឬទទេ")

        progress(progress_start + 0.15 * (progress_end - progress_start),
                 desc="កំពុងបម្លែងសំឡេងទៅជាអក្សរ...")
        with open(audio_extracted, "rb") as a_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.mp3", a_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )

        transcription_text = str(transcription).strip()

        is_valid, msg = validate_transcription(transcription_text)
        if not is_valid:
            raise Exception(f"បញ្ហាអត្ថបទ: {msg}\nអត្ថបទ: {transcription_text[:300]}")

        active_model = get_active_chat_model()

        progress(progress_start + 0.35 * (progress_end - progress_start),
                 desc="កំពុងវិភាគតួអង្គ...")
        segments = analyze_speakers_v4(transcription_text, active_model)

        progress(progress_start + 0.55 * (progress_end - progress_start),
                 desc=f"កំពុងបកប្រែ {len(segments)} កំណាត់...")
        translated_segments = translate_segments_parallel(segments, target_lang_name, active_model)

        progress(progress_start + 0.75 * (progress_end - progress_start),
                 desc="កំពុងបង្កើតសំឡេង AI...")

        successful_segments = []
        male_count = 0
        female_count = 0

        for i, seg in enumerate(translated_segments):
            try:
                voice = get_voice(seg["speaker"], male_voice, female_voice)
                seg_file = os.path.join(temp_dir, f"{base_name}_seg_{i}.mp3")

                asyncio.run(generate_speech_segment(seg["translated"], voice, seg_file))

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

        progress(progress_start + 0.9 * (progress_end - progress_start),
                 desc=f"កំពុងផ្គុំសំឡេង {len(segment_files)} កំណាត់...")

        safe_concat_audio(segment_files, final_audio, temp_dir)

        if not os.path.exists(final_audio) or os.path.getsize(final_audio) < 1000:
            raise Exception("ឯកសារសំឡេងចុងក្រោយទទេ")

        progress(progress_start + 0.95 * (progress_end - progress_start),
                 desc="កំពុងផ្គុំវីដេអូ...")
        merge_cmd = [
            "ffmpeg", "-fflags", "+igndts", "-y",
            "-i", video_path,
            "-i", final_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_video
        ]
        result = subprocess.run(merge_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"ការផ្គុំវីដេអូបរាជ័យ: {result.stderr[:200]}")

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


def dub_video(video_path, target_lang, segment_minutes, progress=gr.Progress()):
    if not video_path:
        return None, None, "សូម Upload វីដេអូជាមុនសិន!"

    target_lang_name, male_voice, female_voice = get_lang_info(target_lang)

    temp_dir = tempfile.gettempdir()

    try:
        if segment_minutes and segment_minutes > 0:
            progress(0.02, desc="កំពុងពិនិត្យរយៈពេលវីដេអូ...")
            segments, duration = split_video(video_path, segment_minutes, temp_dir)

            if not segments:
                return None, None, "ការបំបែកវីដេអូបរាជ័យ!"

            num_segments = len(segments)
            outputs = []
            summaries = []

            for i, seg_path in enumerate(segments):
                p_start = i / num_segments
                p_end = (i + 1) / num_segments

                progress(p_start, desc=f"កំពុងដំណើរការផ្នែកទី {i+1}/{num_segments}...")

                try:
                    out, summary = process_single_video(
                        seg_path, target_lang, target_lang_name,
                        male_voice, female_voice,
                        progress, p_start, p_end
                    )
                    outputs.append(out)
                    summaries.append(f"=== ផ្នែកទី {i+1} ===\n{summary}")
                except Exception as e:
                    summaries.append(f"=== ផ្នែកទី {i+1} បរាជ័យ ===\n{str(e)}")

            for seg in segments:
                if os.path.exists(seg):
                    try:
                        os.remove(seg)
                    except Exception:
                        pass

            if not outputs:
                return None, None, "គ្មានផ្នែកណាមួយដំណើរការបានសម្រេច!\n\n" + "\n\n".join(summaries)

            status = (
                f" ជោគជ័យ! បានបំបែកវីដេអូជា {len(outputs)} ផ្នែក\n"
                f" រយៈពេលសរុប: {duration/60:.1f} នាទី\n"
                f" រយៈពេលកំណត់: {segment_minutes} នាទី/ផ្នែក\n\n"
                + "\n\n".join(summaries)
            )

            return None, outputs, status

        else:
            progress(0.05, desc="កំពុងដំណើរការវីដេអូពេញលេញ...")

            out, summary = process_single_video(
                video_path, target_lang, target_lang_name,
                male_voice, female_voice,
                progress, 0.0, 1.0
            )

            status = f" ជោគជ័យ!\n\n{summary}"
            return out, None, status

    except Exception as e:
        return None, None, f"មានបញ្ហា៖ {str(e)}"


# ============================================================
# Pollinations Video Generation
# ============================================================
def pollinations_generate_video(prompt, duration=5, aspect_ratio="16:9", seed=None):
    """
    បង្កើតវីដេអូជាមួយ Pollinations API
    """
    if not POLLINATIONS_API_KEY:
        print("Pollinations API Key មិនបានកំណត់")
        return None

    # កំណត់ទំហំតាម aspect ratio
    if aspect_ratio == "16:9":
        width, height = 1280, 720
    elif aspect_ratio == "9:16":
        width, height = 720, 1280
    else:
        width, height = 1024, 1024

    headers = {
        "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "model": POLLINATIONS_VIDEO_MODEL,
        "duration": duration,
        "aspectRatio": aspect_ratio,
        "width": width,
        "height": height
    }

    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post(
            f"{POLLINATIONS_BASE}/video/generations",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            print(f"Pollinations error: {response.status_code} - {response.text[:200]}")
            return None

        data = response.json()
        return data
    except Exception as e:
        print(f"Pollinations request failed: {e}")
        return None


def pollinations_poll_video(video_id, max_wait=900):
    """រង់ចាំវីដេអូបង្កើតរួច"""
    headers = {"Authorization": f"Bearer {POLLINATIONS_API_KEY}"}

    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"{POLLINATIONS_BASE}/video/generations/{video_id}",
                headers=headers,
                timeout=60
            )

            if response.status_code != 200:
                time.sleep(10)
                continue

            data = response.json()
            status = str(data.get("status", "")).lower()

            if status in {"succeeded", "success", "completed", "done"}:
                return data
            if status in {"failed", "error", "cancelled"}:
                raise Exception(f"Pollinations បរាជ័យ: {data}")

            elapsed = int(time.time() - start_time)
            print(f"Pollinations {video_id}: {status} - {elapsed}s")
        except Exception as e:
            print(f"Poll error: {e}")

        time.sleep(10)

    raise TimeoutError(f"Pollinations ហួសពេល: {video_id}")


def pollinations_generate_video_for_scene(scene, active_model, session_id, temp_dir):
    """បង្កើតវីដេអូសម្រាប់ scene មួយ"""
    khmer_text = scene["khmer_text"]

    # បកប្រែទៅអង់គ្លេស
    english_prompt = translate_segment_single(
        khmer_text,
        "English (cinematic scene description with characters, setting, action, mood)",
        active_model
    )

    # បន្ថែម style
    full_prompt = (
        f"{english_prompt}. "
        f"Cinematic, photorealistic, 8K, detailed, professional cinematography, "
        f"warm natural lighting, Cambodian countryside setting, characters with natural movement."
    )

    # កំណត់រយៈពេល
    duration = min(max(scene.get("duration", 5), 4), 8)

    try:
        result = pollinations_generate_video(
            prompt=full_prompt,
            duration=int(duration),
            aspect_ratio="16:9",
            seed=session_id + hash(khmer_text) % 10000
        )

        if not result:
            return None

        video_id = result.get("id") or result.get("video_id")
        if not video_id:
            print(f"គ្មាន video_id: {result}")
            return None

        final = pollinations_poll_video(video_id)

        video_url = final.get("url") or final.get("video_url") or final.get("output", {}).get("url")
        if not video_url:
            print(f"គ្មាន video URL: {final}")
            return None

        video_file = os.path.join(temp_dir, f"poll_{session_id}_{int(time.time())}.mp4")
        r = requests.get(video_url, stream=True, timeout=180)
        r.raise_for_status()
        with open(video_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        if os.path.exists(video_file) and os.path.getsize(video_file) > 1000:
            return video_file
    except Exception as e:
        print(f"Scene video failed: {e}")

    return None


def create_video_with_pollinations(script_text, narration_gender, resolution,
                                    progress=gr.Progress()):
    """បង្កើតវីដេអូពី Script ដោយប្រើ Pollinations Video API"""
    if not script_text or len(script_text.strip()) < 10:
        return None, "សូមសរសេរ Script ជាភាសាខ្មែរជាមុនសិន!"

    if not POLLINATIONS_API_KEY:
        return None, "សូមកំណត់ POLLINATIONS_API_KEY ជា environment variable"

    temp_dir = tempfile.gettempdir()
    session_id = int(time.time())

    try:
        # ជំហានទី 1: វិភាគ Script
        progress(0.05, desc="កំពុងវិភាគ Script...")
        active_model = get_active_chat_model()

        lines = script_text.strip().split('\n')
        scenes = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            named_match = re.match(r'^\[([^\]|]+)\|(ប្រុស|ស្រី|male|female|Male|Female|MALE|FEMALE)\]\s*[:：]?\s*(.+)$', line)
            gender_match = re.match(r'^\[(ប្រុស|ស្រី|male|female|Male|Female|MALE|FEMALE)\]\s*[:：]?\s*(.+)$', line)
            narration_match = re.match(r'^\[(និទាន|narration|Narration|NARRATION|ទេសភាព|scene|Scene)\]\s*[:：]?\s*(.+)$', line)

            if named_match:
                name = named_match.group(1).strip()
                gender_raw = named_match.group(2).strip().lower()
                gender = "male" if gender_raw in ["ប្រុស", "male"] else "female"
                text = named_match.group(3).strip()
                scenes.append({"speaker": gender, "name": name, "khmer_text": text})
            elif gender_match:
                gender_raw = gender_match.group(1).strip().lower()
                gender = "male" if gender_raw in ["ប្រុស", "male"] else "female"
                text = gender_match.group(2).strip()
                scenes.append({"speaker": gender, "name": "", "khmer_text": text})
            elif narration_match:
                text = narration_match.group(2).strip()
                scenes.append({"speaker": "narration", "name": "និទាន", "khmer_text": text})
            else:
                scenes.append({"speaker": "narration", "name": "និទាន", "khmer_text": line})

        if not scenes:
            return None, "មិនអាចវិភាគ Script បានទេ!"

        num_scenes = len(scenes)

        # ជំហានទី 2: បង្កើតសំឡេង
        progress(0.1, desc=f"កំពុងបង្កើតសំឡេង {num_scenes} scenes...")

        male_voice = "km-KH-PisethNeural"
        female_voice = "km-KH-SreymomNeural"
        narration_voice = male_voice if narration_gender == "male" else female_voice

        audio_files = []
        for i, scene in enumerate(scenes):
            khmer_text = scene["khmer_text"]
            if not khmer_text:
                continue

            speaker = scene["speaker"]
            if speaker == "male":
                voice = male_voice
            elif speaker == "female":
                voice = female_voice
            else:
                voice = narration_voice

            scene["voice_used"] = voice

            audio_file = os.path.join(temp_dir, f"scene_{session_id}_{i}.mp3")
            try:
                asyncio.run(generate_speech_segment(khmer_text, voice, audio_file))
                if os.path.exists(audio_file) and os.path.getsize(audio_file) > 500:
                    audio_files.append(audio_file)
                    scene["audio_file"] = audio_file
                    scene["duration"] = get_video_duration(audio_file)
            except Exception as e:
                print(f"TTS failed for scene {i}: {e}")

            progress(
                0.1 + 0.1 * (i / num_scenes),
                desc=f"កំពុងបង្កើតសំឡេង {i+1}/{num_scenes}..."
            )

        if not audio_files:
            return None, "មិនអាចបង្កើតសំឡេងបានទេ!"

        # ជំហានទី 3: បង្កើតវីដេអូជាមួយ Pollinations
        progress(0.25, desc=f"កំពុងបង្កើតវីដេអូជាមួយ Pollinations {num_scenes} scenes...")

        video_files = []

        for i, scene in enumerate(scenes):
            if "audio_file" not in scene:
                continue

            progress(
                0.25 + 0.55 * (i / num_scenes),
                desc=f"កំពុងបង្កើតវីដេអូ {i+1}/{num_scenes} (អាចយឺត ១-២ នាទី)..."
            )

            video_file = pollinations_generate_video_for_scene(
                scene, active_model, session_id + i, temp_dir
            )

            if video_file:
                scene["video_file"] = video_file
                video_files.append(video_file)
                print(f"Scene {i+1} video OK: {video_file}")

        if not video_files:
            return None, "មិនអាចបង្កើតវីដេអូជាមួយ Pollinations បានទេ!"

        # ជំហានទី 4: ផ្គុំវីដេអូ
        progress(0.85, desc="កំពុងផ្គុំវីដេអូជាមួយសំឡេង...")

        output_video = os.path.join(temp_dir, f"poll_final_{session_id}.mp4")

        clip_files_with_audio = []
        for i, scene in enumerate(scenes):
            if "video_file" not in scene or "audio_file" not in scene:
                continue

            clip_with_audio = os.path.join(temp_dir, f"clip_audio_{session_id}_{i}.mp4")

            cmd = [
                "ffmpeg", "-y",
                "-i", scene["video_file"],
                "-i", scene["audio_file"],
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                "-pix_fmt", "yuv420p",
                clip_with_audio
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(clip_with_audio):
                clip_files_with_audio.append(clip_with_audio)

        if not clip_files_with_audio:
            return None, "មិនអាចផ្គុំវីដេអូជាមួយសំឡេងបានទេ!"

        concat_file = os.path.join(temp_dir, f"concat_poll_{session_id}.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in clip_files_with_audio:
                f.write(f"file '{clip}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_video
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                output_video
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 or not os.path.exists(output_video):
            return None, f"ការផ្គុំវីដេអូបរាជ័យ: {result.stderr[:300]}"

        # សម្អាត
        for f in video_files + audio_files + clip_files_with_audio:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        male_count = sum(1 for s in scenes if s.get("speaker") == "male")
        female_count = sum(1 for s in scenes if s.get("speaker") == "female")
        narration_count = sum(1 for s in scenes if s.get("speaker") == "narration")

        status = (
            f" ជោគជ័យ! បង្កើតវីដេអូពិតដោយ Pollinations\n\n"
            f" ចំនួន scenes: {num_scenes}\n"
            f" វីដេអូជោគជ័យ: {len(video_files)}/{num_scenes}\n"
            f" តួអង្គប្រុស: {male_count}\n"
            f" តួអង្គស្រី: {female_count}\n"
            f" ការនិទាន: {narration_count}\n\n"
            f" បញ្ជី scenes:\n"
            + "\n".join([
                f"• [{s.get('name', s['speaker'])}] {s['khmer_text'][:60]}..."
                for s in scenes[:15]
            ])
        )

        return output_video, status

    except Exception as e:
        return None, f"មានបញ្ហា៖ {str(e)}"


# ============================================================
# CSS
# ============================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Khmer:wght@400;500;600;700&family=Kantumruy+Pro:wght@400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%) !important;
}

.main-title {
    text-align: center;
    padding: 20px 0 10px 0;
}

.main-title h1 {
    font-size: 2.2em;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    line-height: 1.8;
}

.main-title p {
    color: #5a6478;
    font-size: 1.05em;
    line-height: 1.8;
}

button.primary-btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-family: 'Kantumruy Pro', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05em !important;
    padding: 12px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

button.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .gr-form label {
    font-family: 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
    font-weight: 500 !important;
    color: #2d3748 !important;
    font-size: 1em !important;
    line-height: 1.8 !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    font-family: 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
    font-size: 1em !important;
    line-height: 1.8 !important;
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

.section-header {
    font-family: 'Kantumruy Pro', sans-serif;
    font-weight: 600;
    color: #2d3748;
    font-size: 1.1em;
    padding: 10px 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 12px;
}

footer {
    display: none !important;
}
"""


# ============================================================
# Interface
# ============================================================
with gr.Blocks(title="AI Video Studio", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
        <div class="main-title">
            <h1>🎬 AI Video Studio</h1>
            <p>បកប្រែវីដេអូ និងបង្កើតវីដេអូ AI ពី Script ខ្មែរ</p>
        </div>
    """)

    with gr.Tabs():

        with gr.TabItem("🎥 បកប្រែវីដេអូ"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-header">⚙️ ការកំណត់</div>')

                    video_input = gr.Video(label="📤 Upload វីដេអូ")

                    target_lang = gr.Dropdown(
                        choices=[(name, code) for name, code, _, _ in LANGUAGES],
                        value="km",
                        label="🌐 ភាសាគោលដៅ"
                    )

                    segment_minutes = gr.Number(
                        value=0,
                        label="⏱️ បំបែកវីដេអូជាផ្នែក (នាទី)",
                        info="ដាក់ 0 សម្រាប់ការបកប្រែពេញលេញ",
                        minimum=0,
                        precision=0
                    )

                    submit_btn = gr.Button(
                        "🚀 ចាប់ផ្ដើមបកប្រែ",
                        variant="primary",
                        elem_classes="primary-btn"
                    )

                with gr.Column(scale=2):
                    gr.HTML('<div class="section-header">📺 លទ្ធផល</div>')

                    video_output = gr.Video(label="🎥 វីដេអូដែលបានបកប្រែ")

                    status_output = gr.Textbox(
                        label="📋 ស្ថានភាព",
                        lines=10
                    )

            gr.HTML('<div class="section-header" style="margin-top:24px;">🎞️ ផ្នែកវីដេអូទាំងអស់</div>')

            segments_gallery = gr.Gallery(
                label="",
                columns=3,
                rows=2,
                height="auto",
                object_fit="contain",
                show_label=False
            )

            submit_btn.click(
                fn=dub_video,
                inputs=[video_input, target_lang, segment_minutes],
                outputs=[video_output, segments_gallery, status_output]
            )

        with gr.TabItem("✨ បង្កើតវីដេអូ AI (Pollinations)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-header">⚙️ ការកំណត់</div>')

                    poll_script_input = gr.Textbox(
                        label="📝 សរសេរ Script ជាភាសាខ្មែរ",
                        placeholder=(
                            "ឧទាហរណ៍:\n\n"
                            "[ទេសភាព]: ព្រឹកព្រលឹមនៅវាលស្រែខ្មែរ ពន្លឺព្រះអាទិត្យរះបំភ្លឺពណ៌មាស។\n\n"
                            "[លីដា|female]: នីតា! តើអូនទៅណាដែរ?\n"
                            "[នីតា|female]: ខ្ញុំទៅស្រែ ចុះបងលីដា ទៅណាដែរ?\n"
                            "[លីដា|female]: បងដើរមើលទេសភាពពេលព្រឹកព្រលឹម។"
                        ),
                        lines=14
                    )

                    gr.HTML("""
                        <div style="background:#d1fae5; padding:10px; border-radius:8px; margin-top:8px; font-size:0.9em; line-height:1.8;">
                            <b>✅ ចំណាំ:</b><br>
                            • ការបង្កើតវីដេអូពិតដោយ Pollinations ត្រូវការពេល <b>១-២ នាទី</b> ក្នុងមួយ scene<br>
                            • វីដេអូនីមួយៗមានរយៈពេល <b>៤-៨ វិនាទី</b><br>
                            • មានចលនាពិតប្រាកដ
                        </div>
                    """)

                    poll_voice = gr.Radio(
                        choices=[("សំឡេងប្រុស", "male"), ("សំឡេងស្រី", "female")],
                        value="female",
                        label="🎤 សំឡេងសម្រាប់ការនិទាន"
                    )

                    poll_resolution = gr.Dropdown(
                        choices=[
                            ("16:9 (HD)", "16:9"),
                            ("9:16 (បញ្ឈរ)", "9:16"),
                            ("1:1 (ការេ)", "1:1"),
                        ],
                        value="16:9",
                        label="📐 ទំហំវីដេអូ"
                    )

                    poll_btn = gr.Button(
                        "✨ បង្កើតវីដេអូ AI",
                        variant="primary",
                        elem_classes="primary-btn"
                    )

                with gr.Column(scale=2):
                    gr.HTML('<div class="section-header">📺 លទ្ធផល</div>')

                    poll_video_output = gr.Video(label="🎥 វីដេអូ AI")

                    poll_status = gr.Textbox(
                        label="📋 ស្ថានភាព",
                        lines=14
                    )

            poll_btn.click(
                fn=create_video_with_pollinations,
                inputs=[poll_script_input, poll_voice, poll_resolution],
                outputs=[poll_video_output, poll_status]
            )

    gr.HTML("""
        <div style="text-align:center; padding:20px; color:#8a94a6; font-size:0.9em;">
            💡 ប្រព័ន្ធនឹងបង្កើតវីដេអូពិតដោយ Pollinations AI ជាមួយតួអង្គមានចលនា
        </div>
    """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
