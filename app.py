import os
import sys
import shutil
import json
import math
import re
import time

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
            time.sleep(2)

    return text


def translate_segments(segments, target_lang_name, active_model, progress=None):
    translated_segments = []
    total = len(segments)

    for i, seg in enumerate(segments):
        try:
            if progress:
                progress(
                    0.55 + 0.15 * (i / total),
                    desc=f"កំពុងបកប្រែ {i+1}/{total}..."
                )

            translated_text = translate_segment_single(
                seg["text"], target_lang_name, active_model
            )
            translated_segments.append({
                "speaker": seg["speaker"],
                "original": seg["text"],
                "translated": translated_text
            })
        except Exception as e:
            print(f"Translation failed for segment {i}: {e}")
            translated_segments.append({
                "speaker": seg["speaker"],
                "original": seg["text"],
                "translated": seg["text"]
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
        translated_segments = translate_segments(
            segments, target_lang_name, active_model,
            progress=lambda p, desc: progress(
                progress_start + p * (progress_end - progress_start),
                desc=desc
            )
        )

        progress(progress_start + 0.75 * (progress_end - progress_start),
                 desc="កំពុងបង្កើតសំឡេង AI...")

        successful_segments = []
        male_count = 0
        female_count = 0

        for i, seg in enumerate(translated_segments):
            try:
                voice = get_voice(seg["speaker"], male_voice, female_voice)
                seg_file = os.path.join(temp_dir, f"{base_name}_seg_{i}.mp3")

                progress(
                    progress_start + (0.75 + 0.15 * (i / len(translated_segments))) * (progress_end - progress_start),
                    desc=f"កំពុងបង្កើតសំឡេង {i+1}/{len(translated_segments)} ({seg['speaker']})..."
                )

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
# មុខងារបង្កើតវីដេអូពី Script ជាមួយតួអង្គច្រើន
# ============================================================

def generate_image_from_prompt(prompt, width=1024, height=1024, seed=0):
    """បង្កើតរូបភាព AI ដោយប្រើ Pollinations API"""
    import urllib.parse
    import urllib.request

    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&seed={seed}"

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"ai_image_{seed}_{int(time.time())}.jpg")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return output_path
    except Exception as e:
        print(f"Image generation failed: {e}")
        return None


def generate_image_prompts_batch(scenes, active_model):
    """
    បង្កើត image prompts ដែលបញ្ជាក់តួអង្គច្បាស់
    """
    text_list = "\n".join([
        f"{i+1}. [{s['speaker'].upper()}] {s['khmer_text']}"
        for i, s in enumerate(scenes)
    ])

    prompt = f"""You are a cinematic image prompt generator for AI image generation.

Below are Khmer text scenes. For each scene, create a DETAILED ENGLISH image prompt.

Khmer scenes:
{text_list}

CRITICAL RULES:
1. If the speaker is "MALE", the image MUST show a man (or boy) as the main character.
2. If the speaker is "FEMALE", the image MUST show a woman (or girl) as the main character.
3. If the speaker is "NARRATION", include characters in the scene if the text describes people.
4. Each prompt MUST include:
   - Character: gender, age (young/middle-aged/elderly), appearance, clothing
   - Action: what the character is doing
   - Setting: where the scene takes place
   - Mood: emotions and atmosphere
   - Style: "cinematic, photorealistic, detailed, 8K"
   - Lighting: describe light source and mood
5. Khmer/Cambodian context when appropriate (traditional clothing, Cambodian landscapes)

EXAMPLE GOOD PROMPT:
"A young Cambodian woman in her 20s with long black hair, wearing a red traditional Khmer dress, standing in a rice field at golden hour, looking hopeful towards the horizon, warm sunlight, cinematic composition, photorealistic, 8K, detailed"

Return ONLY a valid JSON array with one prompt per scene:
[
  "prompt for scene 1",
  "prompt for scene 2",
  "prompt for scene 3"
]

JSON:"""

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=active_model,
            temperature=0.5
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        prompts = json.loads(raw)
        if isinstance(prompts, list) and len(prompts) >= len(scenes):
            return prompts[:len(scenes)]
    except Exception as e:
        print(f"Batch prompt generation failed: {e}")

    # Fallback: បង្កើត prompt ដោយខ្លួនឯង
    prompts = []
    for s in scenes:
        speaker = s.get("speaker", "narration")
        if speaker == "male":
            character = "A young Cambodian man in his 20s, short black hair, wearing casual modern clothes"
        elif speaker == "female":
            character = "A young Cambodian woman in her 20s, long black hair, wearing a red traditional Khmer dress"
        else:
            character = "Cambodian people in a traditional setting"

        prompts.append(
            f"{character}, {s['khmer_text'][:80]}, cinematic composition, "
            f"photorealistic, 8K, warm golden lighting, detailed, emotional atmosphere"
        )

    return prompts


def parse_script_with_characters(script_text, active_model):
    """
    វិភាគ Script ដែលមានតួអង្គច្រើន
    """
    lines = script_text.strip().split('\n')
    scenes = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        male_match = re.match(r'^\[(ប្រុស|male|Male|MALE)\]\s*[:：]?\s*(.+)$', line)
        female_match = re.match(r'^\[(ស្រី|female|Female|FEMALE)\]\s*[:：]?\s*(.+)$', line)
        narration_match = re.match(r'^\[(និទាន|narration|Narration|NARRATION)\]\s*[:：]?\s*(.+)$', line)

        if male_match:
            scenes.append({
                "speaker": "male",
                "khmer_text": male_match.group(2).strip()
            })
        elif female_match:
            scenes.append({
                "speaker": "female",
                "khmer_text": female_match.group(2).strip()
            })
        elif narration_match:
            scenes.append({
                "speaker": "narration",
                "khmer_text": narration_match.group(2).strip()
            })
        else:
            scenes.append({
                "speaker": "narration",
                "khmer_text": line
            })

    if all(s["speaker"] == "narration" for s in scenes):
        sentences = split_into_sentences(script_text)
        scenes = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                scenes.append({
                    "speaker": "male" if i % 2 == 0 else "female",
                    "khmer_text": sentence.strip()
                })

    # បង្កើត image prompts ជាប់គ្នា
    image_prompts = generate_image_prompts_batch(scenes, active_model)
    for i, scene in enumerate(scenes):
        if i < len(image_prompts):
            scene["image_prompt"] = image_prompts[i]
        else:
            scene["image_prompt"] = f"Cinematic scene: {scene['khmer_text'][:80]}"

    return scenes


def create_video_clip_with_motion(image_file, audio_file, duration,
                                    clip_file, width, height):
    """
    បង្កើតវីដេអូ clip ជាមួយចលនា zoom
    """
    fps = 25
    total_frames = int(duration * fps)

    # Zoom in បន្តិចម្តងៗ
    zoompan_filter = (
        f"scale={width*2}:{height*2},"
        f"zoompan=z='min(zoom+0.0008,1.15)':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:"
        f"s={width}x{height}:"
        f"fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_file,
        "-i", audio_file,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        "-vf", zoompan_filter,
        "-r", str(fps),
        "-shortest",
        clip_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(clip_file)


def create_video_from_script_multi(script_text, narration_gender, resolution,
                                     progress=gr.Progress()):
    """
    បង្កើតវីដេអូ AI ពី Script ជាមួយតួអង្គច្រើន និងចលនា
    """
    if not script_text or len(script_text.strip()) < 10:
        return None, "សូមសរសេរ Script ជាភាសាខ្មែរជាមុនសិន!"

    temp_dir = tempfile.gettempdir()
    session_id = int(time.time())

    try:
        # ជំហានទី 1: វិភាគ Script
        progress(0.05, desc="កំពុងវិភាគ Script និងតួអង្គ...")
        active_model = get_active_chat_model()
        scenes = parse_script_with_characters(script_text, active_model)

        if not scenes:
            return None, "មិនអាចវិភាគ Script បានទេ!"

        num_scenes = len(scenes)

        # ជំហានទី 2: បង្កើតសំឡេង
        progress(0.15, desc=f"កំពុងបង្កើតសំឡេង {num_scenes} scenes...")

        male_voice = "km-KH-PisethNeural"
        female_voice = "km-KH-SreymomNeural"
        narration_voice = male_voice if narration_gender == "male" else female_voice

        audio_files = []
        for i, scene in enumerate(scenes):
            khmer_text = scene.get("khmer_text", "").strip()
            if not khmer_text:
                continue

            speaker = scene.get("speaker", "narration")
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
                0.15 + 0.15 * (i / num_scenes),
                desc=f"កំពុងបង្កើតសំឡេង {i+1}/{num_scenes} ({speaker})..."
            )

        if not audio_files:
            return None, "មិនអាចបង្កើតសំឡេងបានទេ!"

        # ជំហានទី 3: បង្កើតរូបភាព AI
        progress(0.3, desc=f"កំពុងបង្កើតរូបភាព AI {num_scenes} scenes...")

        width, height = resolution
        image_files = []

        for i, scene in enumerate(scenes):
            if "audio_file" not in scene:
                continue

            image_prompt = scene.get("image_prompt", "Cinematic scene with a character")
            enhanced_prompt = (
                f"{image_prompt}, cinematic, photorealistic, 8K, "
                f"detailed, professional photography, dramatic lighting"
            )

            image_file = generate_image_from_prompt(
                enhanced_prompt, width=width, height=height, seed=session_id + i
            )

            if image_file and os.path.exists(image_file):
                image_files.append(image_file)
                scene["image_file"] = image_file
            else:
                fallback = os.path.join(temp_dir, f"fallback_{session_id}_{i}.jpg")
                cmd = [
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", f"color=c=0x1a1a2e:s={width}x{height}:d=1",
                    "-frames:v", "1", fallback
                ]
                subprocess.run(cmd, capture_output=True)
                if os.path.exists(fallback):
                    image_files.append(fallback)
                    scene["image_file"] = fallback

            progress(
                0.3 + 0.35 * (i / num_scenes),
                desc=f"កំពុងបង្កើតរូបភាព {i+1}/{num_scenes}..."
            )

        # ជំហានទី 4: បង្កើតវីដេអូ clips ជាមួយចលនា
        progress(0.65, desc="កំពុងបង្កើតវីដេអូ clips ជាមួយចលនា...")

        clip_files = []
        for i, scene in enumerate(scenes):
            if "audio_file" not in scene or "image_file" not in scene:
                continue

            duration = max(scene.get("duration", 3.0), 3.0)
            clip_file = os.path.join(temp_dir, f"clip_{session_id}_{i}.mp4")

            success = create_video_clip_with_motion(
                scene["image_file"], scene["audio_file"],
                duration, clip_file, width, height
            )

            if success:
                clip_files.append(clip_file)

            progress(
                0.65 + 0.2 * (i / num_scenes),
                desc=f"កំពុងបង្កើត clip {i+1}/{num_scenes}..."
            )

        if not clip_files:
            return None, "មិនអាចបង្កើតវីដេអូ clips បានទេ!"

        # ជំហានទី 5: ផ្គុំ clips
        progress(0.88, desc="កំពុងផ្គុំវីដេអូ...")

        output_video = os.path.join(temp_dir, f"ai_video_{session_id}.mp4")

        concat_file = os.path.join(temp_dir, f"concat_{session_id}.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in clip_files:
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
                "-c:v", "libx264", "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                output_video
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 or not os.path.exists(output_video):
            return None, f"ការផ្គុំវីដេអូបរាជ័យ: {result.stderr[:300]}"

        # សម្អាត
        for f in clip_files + audio_files + image_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        male_count = sum(1 for s in scenes if s.get("speaker") == "male")
        female_count = sum(1 for s in scenes if s.get("speaker") == "female")
        narration_count = sum(1 for s in scenes if s.get("speaker") == "narration")

        status = (
            f" ជោគជ័យ! បានបង្កើតវីដេអូ AI\n\n"
            f" ចំនួន scenes សរុប: {num_scenes}\n"
            f" តួអង្គប្រុស: {male_count}\n"
            f" តួអង្គស្រី: {female_count}\n"
            f" ការនិទាន: {narration_count}\n"
            f" ទំហំ: {width}x{height}\n"
            f" ចលនា: Zoom in\n\n"
            f" បញ្ជី scenes:\n"
            + "\n".join([
                f"• [{s['speaker'].upper()}] {s['khmer_text'][:60]}..."
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

        with gr.TabItem("✍️ បង្កើតវីដេអូពី Script"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-header">⚙️ ការកំណត់</div>')

                    script_input = gr.Textbox(
                        label="📝 សរសេរ Script ជាភាសាខ្មែរ",
                        placeholder=(
                            "ឧទាហរណ៍ Script ជាមួយតួអង្គច្រើន:\n\n"
                            "[និទាន]: កាលពីអតីតកាល មានក្រុងមួយដ៏ស្រស់ស្អាត។\n"
                            "[ប្រុស]: សួស្តី! អ្នកសុខសប្បាយទេ?\n"
                            "[ស្រី]: ខ្ញុំសុខសប្បាយ អរគុណ!\n"
                            "[ប្រុស]: តោះយើងទៅផ្សារជាមួយគ្នា។"
                        ),
                        lines=12
                    )

                    gr.HTML("""
                        <div style="background:#f0f4ff; padding:10px; border-radius:8px; margin-top:8px; font-size:0.9em; line-height:1.8;">
                            <b>💡 របៀបសរសេរ Script ជាមួយតួអង្គ:</b><br>
                            <code>[ប្រុស]:</code> អត្ថបទសម្រាប់តួអង្គប្រុស<br>
                            <code>[ស្រី]:</code> អត្ថបទសម្រាប់តួអង្គស្រី<br>
                            <code>[និទាន]:</code> អត្ថបទសម្រាប់ការនិទាន<br>
                            បើគ្មាន tag ទេ ប្រព័ន្ធនឹងឆ្លាស់សំឡេងស្វ័យប្រវត្តិ
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
                        "🎬 បង្កើតវីដេអូ AI",
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

            def create_video_wrapper(script_text, voice_gender, resolution_str, progress=gr.Progress()):
                try:
                    w, h = resolution_str.split("x")
                    resolution = (int(w), int(h))
                except Exception:
                    resolution = (1280, 720)

                return create_video_from_script_multi(
                    script_text, voice_gender, resolution, progress
                )

            script_btn.click(
                fn=create_video_wrapper,
                inputs=[script_input, script_voice, script_resolution],
                outputs=[script_video_output, script_status]
            )

    gr.HTML("""
        <div style="text-align:center; padding:20px; color:#8a94a6; font-size:0.9em;">
            💡 ប្រព័ន្ធនឹងបង្កើតរូបភាព AI និងសំឡេងពី Script ខ្មែររបស់អ្នក
        </div>
    """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
