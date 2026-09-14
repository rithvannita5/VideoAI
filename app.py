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
import gradio as gr
from groq import Groq
import edge_tts

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("សូមកំណត់ GROQ_API_KEY ជា environment variable")

groq_client = Groq(api_key=GROQ_API_KEY.strip())


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
    """បកប្រែច្រើនកំណាត់ក្នុងពេលតែមួយ"""
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
# មុខងារបង្កើតវីដេអូពី Script - កំណែលឿន
# ============================================================

def generate_image_fast(prompt, width=1280, height=720, seed=0):
    """បង្កើតរូបភាពដោយប្រើ Pollinations API - កំណែលឿន"""
    import urllib.parse
    import urllib.request

    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&seed={seed}&model=flux"

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"img_{seed}_{int(time.time()*1000)}.jpg")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return output_path
    except Exception as e:
        print(f"Image generation failed: {e}")
        return None


def generate_images_parallel(prompts_with_meta, width, height, session_id):
    """
    បង្កើតរូបភាពច្រើនក្នុងពេលតែមួយ (Parallel)
    """
    results = {}

    def gen_one(item):
        idx, prompt = item
        try:
            img = generate_image_fast(prompt, width, height, session_id + idx)
            return idx, img
        except Exception as e:
            print(f"Image {idx} failed: {e}")
            return idx, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(gen_one, (idx, prompt))
                   for idx, prompt in prompts_with_meta]
        for future in concurrent.futures.as_completed(futures):
            idx, img = future.result()
            results[idx] = img

    return results


def parse_script_and_prompts(script_text, character_images, active_model):
    """
    វិភាគ Script និងបង្កើត prompts ក្នុងពេលតែមួយ
    """
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
            scenes.append({
                "speaker": gender,
                "name": name,
                "khmer_text": text,
                "uploaded_image": character_images.get(name, None)
            })
        elif gender_match:
            gender_raw = gender_match.group(1).strip().lower()
            gender = "male" if gender_raw in ["ប្រុស", "male"] else "female"
            text = gender_match.group(2).strip()
            scenes.append({
                "speaker": gender,
                "name": "",
                "khmer_text": text,
                "uploaded_image": None
            })
        elif narration_match:
            text = narration_match.group(2).strip()
            scenes.append({
                "speaker": "narration",
                "name": "និទាន",
                "khmer_text": text,
                "uploaded_image": None
            })
        else:
            scenes.append({
                "speaker": "narration",
                "name": "និទាន",
                "khmer_text": line,
                "uploaded_image": None
            })

    # បង្កើត prompts សម្រាប់ scenes ដែលគ្មានរូបភាព
    scenes_need_image = [s for s in scenes if not s.get("uploaded_image")]

    if scenes_need_image:
        # បង្កើត prompt ទាំងអស់ក្នុងពេលតែមួយ
        text_list = "\n".join([
            f"{i+1}. [{s.get('name', s['speaker'])}|{s['speaker'].upper()}] {s['khmer_text']}"
            for i, s in enumerate(scenes_need_image)
        ])

        prompt = f"""Generate image prompts for these Khmer scenes. Return ONLY a JSON array of English prompts.

Scenes:
{text_list}

Rules:
- If speaker is MALE: main character is a man/boy
- If speaker is FEMALE: main character is a woman/girl  
- If NARRATION: scenic/atmospheric image
- Cambodian setting: rice fields, palm trees, traditional clothing
- Include: character (gender, age, appearance), action, setting, mood, lighting
- Style: cinematic, photorealistic, 8K, golden hour

Return JSON array only:
["prompt 1", "prompt 2", ...]"""

        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=active_model,
                temperature=0.4
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            prompts = json.loads(raw)
        except Exception as e:
            print(f"Prompt generation failed: {e}")
            prompts = []

        for i, scene in enumerate(scenes_need_image):
            if i < len(prompts):
                scene["image_prompt"] = prompts[i]
            else:
                if scene["speaker"] == "male":
                    char = f"A young Cambodian man named {scene.get('name', '')}"
                elif scene["speaker"] == "female":
                    char = f"A young Cambodian girl named {scene.get('name', '')}"
                else:
                    char = "Cambodian countryside"
                scene["image_prompt"] = f"{char}, {scene['khmer_text'][:80]}, cinematic, photorealistic, 8K"

    return scenes


def create_clip_fast(image_file, audio_file, duration, clip_file, width, height, motion):
    """បង្កើត clip លឿនបំផុត"""
    fps = 25
    frames = int(duration * fps)

    if motion == "zoom_in":
        vf = f"scale={width*2}:{height*2},zoompan=z='min(zoom+0.001,1.12)':d={frames}:s={width}x{height}:fps={fps}"
    elif motion == "zoom_out":
        vf = f"scale={width*2}:{height*2},zoompan=z='if(lte(zoom,1.0),1.12,max(1.0,zoom-0.001))':d={frames}:s={width}x{height}:fps={fps}"
    elif motion == "pan_left":
        vf = f"scale={width*2}:{height*2},zoompan=z='1.12':x='if(lte(on,1),iw/2,on*3)':d={frames}:s={width}x{height}:fps={fps}"
    else:
        vf = f"scale={width*2}:{height*2},zoompan=z='1.12':x='if(lte(on,1),0,iw/2-on*3)':d={frames}:s={width}x{height}:fps={fps}"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_file,
        "-i", audio_file,
        "-c:v", "libx264",
        "-preset", "ultrafast",  # លឿនបំផុត
        "-tune", "stillimage",
        "-crf", "25",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        "-vf", vf,
        "-r", str(fps),
        "-shortest",
        clip_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(clip_file)


def create_video_from_script_fast(script_text, character_files, narration_gender,
                                    resolution, progress=gr.Progress()):
    """
    បង្កើតវីដេអូពី Script - កំណែលឿនបំផុត
    ប្រើ Parallel Processing
    """
    if not script_text or len(script_text.strip()) < 10:
        return None, "សូមសរសេរ Script ជាមុនសិន!"

    temp_dir = tempfile.gettempdir()
    session_id = int(time.time())

    try:
        # ជំហាន 1: អានរូបភាពតួអង្គ
        progress(0.03, desc="កំពុងអានរូបភាពតួអង្គ...")
        character_images = {}
        if character_files:
            for file_obj in character_files:
                try:
                    if hasattr(file_obj, 'name'):
                        file_path = file_obj.name
                    else:
                        file_path = str(file_obj)
                    file_name = os.path.basename(file_path)
                    name = os.path.splitext(file_name)[0]
                    character_images[name] = file_path
                except Exception as e:
                    print(f"Failed: {e}")

        # ជំហាន 2: វិភាគ Script + បង្កើត prompts
        progress(0.08, desc="កំពុងវិភាគ Script...")
        active_model = get_active_chat_model()
        scenes = parse_script_and_prompts(script_text, character_images, active_model)

        if not scenes:
            return None, "មិនអាចវិភាគ Script បានទេ!"

        num_scenes = len(scenes)

        # ជំហាន 3: បង្កើតសំឡេងព្រមគ្នា (Parallel TTS)
        progress(0.15, desc=f"កំពុងបង្កើតសំឡេង {num_scenes} scenes...")

        male_voice = "km-KH-PisethNeural"
        female_voice = "km-KH-SreymomNeural"
        narration_voice = male_voice if narration_gender == "male" else female_voice

        def gen_tts(idx_scene):
            idx, scene = idx_scene
            khmer_text = scene.get("khmer_text", "").strip()
            if not khmer_text:
                return idx, None

            speaker = scene.get("speaker", "narration")
            if speaker == "male":
                voice = male_voice
            elif speaker == "female":
                voice = female_voice
            else:
                voice = narration_voice

            audio_file = os.path.join(temp_dir, f"tts_{session_id}_{idx}.mp3")
            try:
                asyncio.run(generate_speech_segment(khmer_text, voice, audio_file))
                if os.path.exists(audio_file) and os.path.getsize(audio_file) > 500:
                    return idx, (audio_file, voice, get_video_duration(audio_file))
            except Exception as e:
                print(f"TTS {idx} failed: {e}")
            return idx, None

        tts_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(gen_tts, (i, s)) for i, s in enumerate(scenes)]
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                if result:
                    tts_results[idx] = result

        if not tts_results:
            return None, "មិនអាចបង្កើតសំឡេងបានទេ!"

        for i, scene in enumerate(scenes):
            if i in tts_results:
                audio_file, voice, duration = tts_results[i]
                scene["audio_file"] = audio_file
                scene["voice_used"] = voice
                scene["duration"] = duration

        progress(0.35, desc=f"បានបង្កើតសំឡេង {len(tts_results)}/{num_scenes}")

        # ជំហាន 4: បង្កើតរូបភាពព្រមគ្នា (Parallel Image Gen)
        progress(0.4, desc="កំពុងបង្កើតរូបភាពព្រមគ្នា...")

        width, height = resolution
        scenes_need_ai = [(i, s) for i, s in enumerate(scenes)
                          if "audio_file" in s and not s.get("uploaded_image")]

        prompts_with_meta = []
        for idx, scene in scenes_need_ai:
            prompt = scene.get("image_prompt", "Cinematic scene")
            enhanced = f"{prompt}, cinematic, photorealistic, 8K, detailed, dramatic lighting"
            prompts_with_meta.append((idx, enhanced))

        ai_images = {}
        if prompts_with_meta:
            # បង្កើតរូបភាព 4 ក្នុងពេលតែមួយ
            def gen_img(item):
                idx, prompt = item
                try:
                    img = generate_image_fast(prompt, width, height, session_id + idx)
                    return idx, img
                except Exception as e:
                    print(f"Img {idx} failed: {e}")
                    return idx, None

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(gen_img, item) for item in prompts_with_meta]
                for future in concurrent.futures.as_completed(futures):
                    idx, img = future.result()
                    if img:
                        ai_images[idx] = img

        progress(0.7, desc=f"បានបង្កើតរូបភាព {len(ai_images)}/{len(prompts_with_meta)}")

        # ភ្ជាប់រូបភាពទៅ scenes
        for i, scene in enumerate(scenes):
            if "audio_file" not in scene:
                continue
            if scene.get("uploaded_image") and os.path.exists(scene["uploaded_image"]):
                scene["image_file"] = scene["uploaded_image"]
                scene["image_source"] = "uploaded"
            elif i in ai_images:
                scene["image_file"] = ai_images[i]
                scene["image_source"] = "ai"
            else:
                fallback = os.path.join(temp_dir, f"fb_{session_id}_{i}.jpg")
                cmd = ["ffmpeg", "-y", "-f", "lavfi",
                       "-i", f"color=c=0x1a1a2e:s={width}x{height}:d=1",
                       "-frames:v", "1", fallback]
                subprocess.run(cmd, capture_output=True)
                if os.path.exists(fallback):
                    scene["image_file"] = fallback
                    scene["image_source"] = "fallback"

        # ជំហាន 5: បង្កើត clips ព្រមគ្នា (Parallel Clip Gen)
        progress(0.75, desc="កំពុងបង្កើត clips...")

        motion_types = ["zoom_in", "zoom_out", "pan_left", "pan_right"]

        def gen_clip(idx_scene):
            idx, scene = idx_scene
            if "audio_file" not in scene or "image_file" not in scene:
                return idx, None
            duration = max(scene.get("duration", 3.0), 3.0)
            clip_file = os.path.join(temp_dir, f"clip_{session_id}_{idx}.mp4")
            motion = motion_types[idx % len(motion_types)]
            success = create_clip_fast(
                scene["image_file"], scene["audio_file"],
                duration, clip_file, width, height, motion
            )
            return idx, clip_file if success else None

        clips_with_meta = [(i, s) for i, s in enumerate(scenes) if "audio_file" in s]

        clip_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(gen_clip, item) for item in clips_with_meta]
            for future in concurrent.futures.as_completed(futures):
                idx, clip = future.result()
                if clip:
                    clip_results[idx] = clip

        clip_files = [clip_results[i] for i in sorted(clip_results.keys())]

        if not clip_files:
            return None, "មិនអាចបង្កើត clips បានទេ!"

        # ជំហាន 6: ផ្គុំ clips
        progress(0.92, desc="កំពុងផ្គុំវីដេអូ...")

        output_video = os.path.join(temp_dir, f"final_{session_id}.mp4")

        concat_file = os.path.join(temp_dir, f"cc_{session_id}.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in clip_files:
                f.write(f"file '{clip}'\n")

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
        for f in clip_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        for scene in scenes:
            if scene.get("image_source") == "ai" and scene.get("image_file"):
                if os.path.exists(scene["image_file"]):
                    try:
                        os.remove(scene["image_file"])
                    except Exception:
                        pass
            if "audio_file" in scene and os.path.exists(scene["audio_file"]):
                try:
                    os.remove(scene["audio_file"])
                except Exception:
                    pass

        uploaded_count = sum(1 for s in scenes if s.get("image_source") == "uploaded")
        ai_count = sum(1 for s in scenes if s.get("image_source") == "ai")

        status = (
            f" ជោគជ័យ! (កំណែលឿន)\n\n"
            f" ចំនួន scenes: {num_scenes}\n"
            f" រូបភាព Upload: {uploaded_count}\n"
            f" រូបភាព AI: {ai_count}\n"
            f" ទំហំ: {width}x{height}\n"
            f" ចលនា: Zoom & Pan\n\n"
            f" បញ្ជី scenes:\n"
            + "\n".join([
                f"• [{s.get('name', s['speaker'])}] ({s.get('image_source', '?')}) {s['khmer_text'][:50]}..."
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
            <p>បកប្រែវីដេអូ និងបង្កើតវីដេអូ AI ពី Script ខ្មែរ (កំណែលឿន ⚡)</p>
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

        with gr.TabItem("✍️ បង្កើតវីដេអូពី Script"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-header">⚙️ ការកំណត់</div>')

                    script_input = gr.Textbox(
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
                        <div style="background:#f0f4ff; padding:10px; border-radius:8px; margin-top:8px; font-size:0.9em; line-height:1.8;">
                            <b>💡 របៀបសរសេរ Script:</b><br>
                            <code>[ឈ្មោះ|ភេទ]: អត្ថបទ</code> — តួអង្គ<br>
                            <code>[និទាន]: អត្ថបទ</code> — ការនិទាន<br>
                            <code>[ទេសភាព]: អត្ថបទ</code> — ទេសភាព<br>
                            <b>ភេទ:</b> <code>male</code> ឬ <code>female</code>
                        </div>
                    """)

                    character_upload = gr.File(
                        label="🖼️ Upload រូបភាពតួអង្គ (ស្រេចចិត្ត)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath"
                    )

                    gr.HTML("""
                        <div style="background:#fff8e1; padding:10px; border-radius:8px; margin-top:8px; font-size:0.85em; line-height:1.7;">
                            <b>📌 របៀបដាក់ឈ្មោះឯកសារ:</b><br>
                            ដាក់ឈ្មោះឯកសារជា<b>ឈ្មោះតួអង្គ</b><br>
                            ឧទាហរណ៍: <code>លីដា.jpg</code>, <code>នីតា.jpg</code>
                        </div>
                    """)

                    script_voice = gr.Radio(
                        choices=[("សំឡេងប្រុស", "male"), ("សំឡេងស្រី", "female")],
                        value="male",
                        label="🎤 សំឡេងសម្រាប់ការនិទាន"
                    )

                    script_resolution = gr.Dropdown(
                        choices=[
                            ("1024x1024 (ការេ)", "1024x1024"),
                            ("1280x720 (HD)", "1280x720"),
                            ("720x1280 (បញ្ឈរ)", "720x1280"),
                        ],
                        value="1280x720",
                        label="📐 ទំហំវីដេអូ"
                    )

                    script_btn = gr.Button(
                        "⚡ បង្កើតវីដេអូ AI (លឿន)",
                        variant="primary",
                        elem_classes="primary-btn"
                    )

                with gr.Column(scale=2):
                    gr.HTML('<div class="section-header">📺 លទ្ធផល</div>')

                    script_video_output = gr.Video(label="🎥 វីដេអូ AI")

                    script_status = gr.Textbox(
                        label="📋 ស្ថានភាព",
                        lines=14
                    )

            def create_video_wrapper(script_text, character_files, voice_gender,
                                      resolution_str, progress=gr.Progress()):
                try:
                    w, h = resolution_str.split("x")
                    resolution = (int(w), int(h))
                except Exception:
                    resolution = (1280, 720)

                return create_video_from_script_fast(
                    script_text, character_files, voice_gender, resolution, progress
                )

            script_btn.click(
                fn=create_video_wrapper,
                inputs=[script_input, character_upload, script_voice, script_resolution],
                outputs=[script_video_output, script_status]
            )

    gr.HTML("""
        <div style="text-align:center; padding:20px; color:#8a94a6; font-size:0.9em;">
            ⚡ កំណែលឿន - ប្រើ Parallel Processing សម្រាប់ល្បឿន ៣-៥ ដង
        </div>
    """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
