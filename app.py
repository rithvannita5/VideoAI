import os
import sys
import shutil
import json
import math
import re
import time
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
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")

if not GROQ_API_KEY:
    raise ValueError("សូមកំណត់ GROQ_API_KEY ជា environment variable")
if not AGNES_API_KEY:
    raise ValueError("សូមកំណត់ AGNES_API_KEY ជា environment variable")

groq_client = Groq(api_key=GROQ_API_KEY.strip())

AGNES_BASE_URL = "https://apihub.agnes-ai.com"
AGNES_VIDEO_MODEL = "agnes-video-v2.0"

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


# ============================================================
# បកប្រែវីដេអូ
# ============================================================
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
# Agnes AI Video Generation
# ============================================================
AGNES_CLIP_FRAMES = 121      # ~5 វិនាទី ក្នុងមួយ clip
AGNES_FPS = 24
AGNES_CLIP_SECONDS = AGNES_CLIP_FRAMES / AGNES_FPS

KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")

# ffmpeg ប្រើ RAM តិច (សម្រាប់ Render Free ដែលមាន RAM ~512MB)
LOW_MEM_ENCODE = ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                  "-threads", "1", "-pix_fmt", "yuv420p"]


def agnes_create_video(prompt, image_url=None, num_frames=121, frame_rate=24,
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

    if image_url:
        payload["image"] = image_url
        payload["mode"] = "ti2vid"

    if seed is not None:
        payload["seed"] = seed

    response = requests.post(
        f"{AGNES_BASE_URL}/v1/videos",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def agnes_poll_video(video_id, max_wait=600):
    headers = {
        "Authorization": f"Bearer {AGNES_API_KEY}"
    }

    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{AGNES_BASE_URL}/agnesapi",
            params={"video_id": video_id},
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        status = str(data.get("status", "")).lower()
        if status in {"succeeded", "success", "completed", "done"}:
            return data
        if status in {"failed", "error", "cancelled"}:
            raise Exception(f"Agnes បង្កើតវីដេអូបរាជ័យ: {data}")

        progress_pct = data.get("progress", 0)
        print(f"Agnes video {video_id}: {status} ({progress_pct}%)")
        time.sleep(5)

    raise TimeoutError(f"Agnes វីដេអូហួសពេល: {video_id}")


def parse_scene_script(script_text):
    """អាន script ទម្រង់ [ឈុតទី១៖ ០:០០ - ០:១៥ នាទី] - ចំណងជើង / វីដេអូ៖ / សំឡេងសម្រាយ:"""
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
    """ទម្រង់ចាស់ [ប្រុស]: ... / [ស្រី]: ... / [និទាន]: ..."""
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
    """បំប្លែងការពិពណ៌នា 'វីដេអូ' ជា English prompt ចំនួន n shots ដែលមានសកម្មភាពខុសៗគ្នា"""
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
        "DIFFERENT moment with a DIFFERENT visible action, camera angle and framing "
        "(e.g. wide establishing -> medium -> close-up). Shot 1 is the beginning of the "
        "described story, the last shot is the end. Never repeat the same action twice. "
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
        print(f"Shot prompt generation failed: {e}")

    if len(shots) < n_shots:
        stages = [
            "Opening wide establishing shot:",
            "Medium shot, the action develops:",
            "Close-up on faces, emotional reaction:",
            "Dramatic final moment:",
        ]
        shots = [f"{stages[min(i, len(stages) - 1) if i < n_shots - 1 else len(stages) - 1]} {visual}"
                 for i in range(n_shots)]
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
                raise Exception(f"គ្មាន video URL: {final}")

            path = os.path.join(temp_dir, f"agnes_clip_{tag}.mp4")
            r = requests.get(video_url, stream=True, timeout=120)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            if os.path.getsize(path) < 1000:
                raise Exception("ឯកសារ clip តូចពេក")
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
        # 1) សំឡេងខ្មែរ
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
            asyncio.run(generate_speech_segment(scene["khmer_text"], voice, audio_file))
            temp_files.append(audio_file)
            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 500:
                audio_dur = get_video_duration(audio_file)
            else:
                audio_file = None

        # 2) រយៈពេលគោលដៅ (យកតាមសំឡេង ឬ timestamp ណាវែងជាង)
        target = max(scene.get("duration", 0), audio_dur + 0.3, 3.0)
        n_shots = max(1, math.ceil(target / AGNES_CLIP_SECONDS))

        # 3) prompt ក្នុងមួយ shot
        prompts = agnes_generate_shot_prompts(scene, n_shots, active_model)
        print(f"🎬 {scene['name']}: target {target:.1f}s, planned {n_shots} shots")

        # 4) បង្កើត clip ម្ដងមួយៗ រហូតដល់ប្រវែងវីដេអូពិត >= គោលដៅ
        clips = []
        total = 0.0
        MAX_CLIPS = n_shots + 4          # អនុញ្ញាត clip បន្ថែមបើខ្លះបរាជ័យ/ខ្លីជាងរំពឹង
        idx = 0
        while total < target - 0.3 and idx < MAX_CLIPS:
            if idx < len(prompts):
                p = prompts[idx]
            else:
                p = (prompts[-1] + ", the scene continues, different camera angle, new small action")
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
                else:
                    errors.append(f"clip {idx+1}: រយៈពេល {d:.1f}s")
            except Exception as e:
                errors.append(f"clip {idx+1}: {str(e)[:150]}")
                print(f"   clip {idx+1} failed: {e}")
                time.sleep(3)
            idx += 1

        scene["clips_ok"] = f"{len(clips)} clips = {total:.1f}s / គោលដៅ {target:.1f}s"
        scene["debug"] = (
            "\n".join(f"   shot {i+1}: {p[:140]}" for i, p in enumerate(prompts))
            + ("\n   ⚠️ " + "\n   ⚠️ ".join(errors) if errors else "")
        )
        if not clips:
            return None

        # 5) ផ្គុំ clip
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
            print(f"Shots concat failed: {r.stderr[:300]}")
            return None

        # 6) បើនៅខ្លីជាងសំឡេង៖ បន្ថយល្បឿនបន្តិច (អតិបរមា 1.6x) ហើយ freeze ចុងក្រោយសម្រាប់ចំណែកដែលនៅសល់
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
        print(f"Final merge failed: {r.stderr[:300]}")
        return None

    except Exception as e:
        print(f"Agnes scene failed: {e}")
        scene["debug"] = f"   ⚠️ {e}"
        return None
    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def create_video_with_agnes(script_text, narration_gender, resolution,
                            progress=gr.Progress()):
    if not script_text or len(script_text.strip()) < 10:
        return None, "សូមសរសេរ Script ជាមុនសិន!"

    temp_dir = tempfile.gettempdir()
    session_id = int(time.time())
    width, height = resolution
    narration_voice = "km-KH-PisethNeural" if narration_gender == "male" else "km-KH-SreymomNeural"

    scene_files = []
    try:
        progress(0.05, desc="កំពុងវិភាគ Script...")
        active_model = get_active_chat_model()

        scenes = parse_scene_script(script_text)
        if not scenes:
            scenes = parse_legacy_script(script_text)
        if not scenes:
            return None, "មិនអាចវិភាគ Script បានទេ!"

        num_scenes = len(scenes)
        failed = []

        for i, scene in enumerate(scenes):
            progress(0.1 + 0.75 * (i / num_scenes),
                     desc=f"កំពុងបង្កើតឈុត {i+1}/{num_scenes} (អាចយឺតច្រើននាទី)...")
            f = agnes_generate_video_for_scene(
                scene, active_model, session_id * 100 + i, temp_dir,
                width, height, narration_voice
            )
            if f:
                scene_files.append(f)
            else:
                failed.append(i + 1)

        if not scene_files:
            return None, "មិនអាចបង្កើតវីដេអូជាមួយ Agnes បានទេ!"

        progress(0.9, desc="កំពុងផ្គុំវីដេអូទាំងអស់...")
        output_video = os.path.join(temp_dir, f"agnes_final_{session_id}.mp4")
        concat_file = os.path.join(temp_dir, f"concat_agnes_{session_id}.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for vf in scene_files:
                f.write(f"file '{vf}'\n")

        result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", output_video
        ], capture_output=True, text=True)
        if result.returncode != 0:
            result = subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            ] + LOW_MEM_ENCODE + ["-c:a", "aac", output_video], capture_output=True, text=True)

        if result.returncode != 0 or not os.path.exists(output_video):
            return None, f"ការផ្គុំវីដេអូបរាជ័យ: {result.stderr[:300]}"

        total = get_video_duration(output_video)
        status = (
            f"✅ ជោគជ័យ!\n"
            f"ចំនួនឈុត: {num_scenes} | ជោគជ័យ: {len(scene_files)}"
            + (f" | បរាជ័យ: ឈុតទី {', '.join(map(str, failed))}" if failed else "")
            + f"\nរយៈពេលសរុប: {total:.1f} វិនាទី | ទំហំ: {width}x{height}\n\n"
            + "\n".join(f"• {s['name']} ({s.get('duration', 0)}s) clips: {s.get('clips_ok', '-')}\n{s.get('debug', '')}"
                        for s in scenes[:15])
        )
        return output_video, status

    except Exception as e:
        return None, f"មានបញ្ហា៖ {str(e)}"
    finally:
        for f in scene_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


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
with gr.Blocks(title="AI Video Studio") as demo:

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

        with gr.TabItem("✨ បង្កើតវីដេអូ AI ពិត (Agnes)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-header">⚙️ ការកំណត់</div>')

                    agnes_script_input = gr.Textbox(
                        label="📝 សរសេរ Script ជាភាសាខ្មែរ",
                        placeholder=(
                            "[ឈុតទី១៖ ០:០០ - ០:១៥ នាទី] - ការបើកឆាកទាក់ទាញចិត្ត (The Hook)\n\n"
                            "វីដេអូ៖ បង្ហាញឈុតតួស្រីត្រូវគេមើលងាយក្នុងពិធីមង្គលការ ទឹកមុខស្រងូតស្រងាត់ "
                            "តែភ្លាមនោះមានរថយន្តទំនើបបើកមកកាក់មុខ រួចតួប្រុសចុះមកយ៉ាងសង្ហា។\n\n"
                            "សំឡេងសម្រាយ (Voiceover): \"ពេលខ្លះ មនុស្សដែលអ្នកធ្លាប់ជាន់ឈ្លី និងមើលងាយ... "
                            "ថ្ងៃស្អែកអាចជាម្ចាស់វាសនាដែលអ្នកគ្មានថ្ងៃស្រមើស្រមៃដល់!\"\n\n"
                            "(ឬប្រើទម្រង់ចាស់: [ប្រុស]: ... / [ស្រី]: ... / [និទាន]: ...)"
                        ),
                        lines=14
                    )

                    gr.HTML("""
                        <div style="background:#fff3cd; padding:10px; border-radius:8px; margin-top:8px; font-size:0.9em; line-height:1.8;">
                            <b>⚡ ចំណាំសំខាន់:</b><br>
                            • ការបង្កើតវីដេអូដោយ Agnes AI ត្រូវការពេល <b>១-៣ នាទី</b> ក្នុងមួយ clip (≈៥ វិនាទី)<br>
                            • ឈុត ១៥ វិនាទី = ៣ clips ផ្គុំចូលគ្នា ដោយអាន <b>"វីដេអូ៖"</b> ជាការពិពណ៌នារូបភាព<br>
                            • <b>"សំឡេងសម្រាយ"</b> ត្រូវបានបម្លែងជាសំឡេងខ្មែរដោយ Edge TTS<br>
                            • ទម្រង់ឈុត: <code>[ឈុតទី១៖ ០:០០ - ០:១៥ នាទី] - ចំណងជើង</code>
                        </div>
                    """)

                    agnes_voice = gr.Radio(
                        choices=[("សំឡេងប្រុស", "male"), ("សំឡេងស្រី", "female")],
                        value="male",
                        label="🎤 សំឡេងសម្រាប់ការនិទាន / សំឡេងសម្រាយ"
                    )

                    agnes_resolution = gr.Dropdown(
                        choices=[
                            ("768x768 (ការេ)", "768x768"),
                            ("1152x768 (HD)", "1152x768"),
                            ("768x1152 (បញ្ឈរ)", "768x1152"),
                        ],
                        value="1152x768",
                        label="📐 ទំហំវីដេអូ"
                    )

                    agnes_btn = gr.Button(
                        "✨ បង្កើតវីដេអូ AI ពិត",
                        variant="primary",
                        elem_classes="primary-btn"
                    )

                with gr.Column(scale=2):
                    gr.HTML('<div class="section-header">📺 លទ្ធផល</div>')

                    agnes_video_output = gr.Video(label="🎥 វីដេអូ AI ពិត")

                    agnes_status = gr.Textbox(
                        label="📋 ស្ថានភាព",
                        lines=14
                    )

            def agnes_wrapper(script_text, voice_gender, resolution_str, progress=gr.Progress()):
                try:
                    w, h = resolution_str.split("x")
                    resolution = (int(w), int(h))
                except Exception:
                    resolution = (1152, 768)

                return create_video_with_agnes(
                    script_text, voice_gender, resolution, progress
                )

            agnes_btn.click(
                fn=agnes_wrapper,
                inputs=[agnes_script_input, agnes_voice, agnes_resolution],
                outputs=[agnes_video_output, agnes_status]
            )

    gr.HTML("""
        <div style="text-align:center; padding:20px; color:#8a94a6; font-size:0.9em;">
            💡 ប្រព័ន្ធនឹងបង្កើតវីដេអូពិតដោយ Agnes AI ជាមួយសំឡេងខ្មែរ តាម Script ដែលអ្នកសរសេរ
        </div>
    """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft()
    )
