import os
import sys
import shutil
import json
import math
import re

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
    """បំបែកអត្ថបទជាប្រយោគពេញលេញ"""
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


def chunk_text(text, max_chars=3000):
    """
    បំបែកអត្ថបទធំជាកំណាត់តូចៗ ដើម្បីជៀសវាង Error 413
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def analyze_speakers_v3(transcription_text, active_model):
    """
    វិភាគតួអង្គដោយបំបែកអត្ថបទធំជាកំណាត់តូចៗ
    """
    # បំបែកអត្ថបទជាកំណាត់តូចៗ
    chunks = chunk_text(transcription_text, max_chars=3000)

    all_segments = []

    for chunk in chunks:
        sentences = split_into_sentences(chunk)

        if len(sentences) <= 1:
            # ប្រើ Groq ដើម្បីបំបែកប្រយោគវែង
            prompt = f"""Split the following text into 2-3 segments at natural break points.

Text:
{chunk}

RULES:
- Each segment must be a COMPLETE clause with full meaning.
- Do NOT cut in the middle of a phrase.
- Do NOT change any words.
- First = "male", second = "female", third = "male".

Return ONLY JSON:
[{{"speaker": "male", "text": "..."}}, {{"speaker": "female", "text": "..."}}]

JSON:"""

            try:
                response = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=active_model,
                    temperature=0.2
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

                segments = json.loads(raw)
                if isinstance(segments, list) and len(segments) > 0:
                    for i, seg in enumerate(segments):
                        seg["speaker"] = "male" if i % 2 == 0 else "female"
                    all_segments.extend(segments)
                    continue
            except Exception:
                pass

            all_segments.append({"speaker": "male", "text": chunk})
        else:
            # បំបែកតាមប្រយោគ និងឆ្លាស់សំឡេង
            for i, sentence in enumerate(sentences):
                all_segments.append({
                    "speaker": "male" if i % 2 == 0 else "female",
                    "text": sentence
                })

    # ឆ្លាស់សំឡេងឡើងវិញសម្រាប់ segments ទាំងអស់
    for i, seg in enumerate(all_segments):
        seg["speaker"] = "male" if i % 2 == 0 else "female"

    return all_segments if all_segments else [{"speaker": "male", "text": transcription_text}]


def translate_segment_single(text, target_lang_name, active_model):
    """បកប្រែកំណាត់តែមួយ"""
    prompt = (
        f"Translate the following text into {target_lang_name}. "
        f"Output ONLY the translation. No explanations:\n\n{text}"
    )
    res = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=active_model,
        temperature=0.3
    )
    return res.choices[0].message.content.strip()


def translate_segments(segments, target_lang_name, active_model):
    translated_segments = []
    total = len(segments)

    for i, seg in enumerate(segments):
        try:
            translated_text = translate_segment_single(
                seg["text"], target_lang_name, active_model
            )
            translated_segments.append({
                "speaker": seg["speaker"],
                "original": seg["text"],
                "translated": translated_text
            })
        except Exception as e:
            # ប្រសិនបើបកប្រែបរាជ័យ សូមប្រើអត្ថបទដើម
            translated_segments.append({
                "speaker": seg["speaker"],
                "original": seg["text"],
                "translated": seg["text"]
            })

    return translated_segments


def get_voice(speaker, male_voice, female_voice):
    return male_voice if speaker == "male" else female_voice


def safe_concat_audio(segment_files, output_path, temp_dir):
    """
    ផ្គុំសំឡេងដោយ Re-encode ដើម្បីធានាថាមិនបាត់សំឡេង
    """
    # ប្រើ filter_complex សម្រាប់ការផ្គុំដែលអាចទុកចិត្តបាន
    inputs = []
    for f in segment_files:
        inputs.extend(["-i", f])

    # បង្កើត filter សម្រាប់ concat
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
        # Fallback: ប្រើ concat demuxer
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
        segments = analyze_speakers_v3(transcription_text, active_model)

        progress(progress_start + 0.55 * (progress_end - progress_start),
                 desc=f"កំពុងបកប្រែ {len(segments)} កំណាត់...")
        translated_segments = translate_segments(
            segments, target_lang_name, active_model
        )

        progress(progress_start + 0.7 * (progress_end - progress_start),
                 desc="កំពុងបង្កើតសំឡេង AI...")

        successful_segments = []
        for i, seg in enumerate(translated_segments):
            try:
                voice = get_voice(seg["speaker"], male_voice, female_voice)
                seg_file = os.path.join(temp_dir, f"{base_name}_seg_{i}.mp3")
                asyncio.run(generate_speech_segment(seg["translated"], voice, seg_file))

                # ត្រួតពិនិត្យថាឯកសារត្រូវបានបង្កើត
                if os.path.exists(seg_file) and os.path.getsize(seg_file) > 500:
                    segment_files.append(seg_file)
                    successful_segments.append(seg)
            except Exception as e:
                print(f"Segment {i} failed: {e}")

        if not segment_files:
            raise Exception("គ្មានកំណាត់ណាមួយបង្កើតសំឡេងបានសម្រេច!")

        progress(progress_start + 0.85 * (progress_end - progress_start),
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
            f"ចំនួនកំណាត់ជោគជ័យ: {len(successful_segments)}\n\n"
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


with gr.Blocks(title="AI Video Dubbing Studio", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
        <div class="main-title">
            <h1>🎬 AI Video Dubbing Studio</h1>
            <p>បកប្រែ និងបញ្ចូលសំឡេង AI ទៅក្នុងវីដេអូរបស់អ្នកដោយស្វ័យប្រវត្តិ</p>
        </div>
    """)

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
                info="ឧទាហរណ៍៖ បើវីដេអូ 60 នាទី ហើយកំណត់ 6 នាទី → បំបែកជា 10 ផ្នែក ។ បើដាក់ 0 នោះដំណើរការពេញលេញ។",
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
                label="📋 ស្ថានភាព និងព័ត៌មានលម្អិត",
                lines=10
            )

    gr.HTML('<div class="section-header" style="margin-top:24px;">🎞️ ផ្នែកវីដេអូទាំងអស់ (ចុចដើម្បីចាក់ ឬទាញយក)</div>')

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

    gr.HTML("""
        <div style="text-align:center; padding:20px; color:#8a94a6; font-size:0.9em;">
            💡 ប្រព័ន្ធនឹងវិភាគតួអង្គដោយស្វ័យប្រវត្តិ និងប្រើសំឡេងប្រុស/ស្រីតាមតួអង្គ
        </div>
    """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
