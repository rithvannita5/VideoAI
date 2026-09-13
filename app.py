import os
import sys
import shutil
import json
import math

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
# បញ្ជីភាសាដែលគាំទ្រ
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
    ("ហិណ្ឌូ (Hindi)", "hi", "hi-IN-MadhurNeural", "hi-IN-SwaraNeural"),
    ("អារ៉ាប់ (Arabic)", "ar", "ar-SA-HamedNeural", "ar-SA-ZariyahNeural"),
]


def get_lang_info(lang_code):
    """ទាញយកព័ត៌មានភាសា"""
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
    """ទាញយករយៈពេលវីដេអូគិតជាវិនាទី"""
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
    """បំបែកវីដេអូជាផ្នែកៗ តាមរយៈពេលដែលបានកំណត់"""
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


def analyze_speakers(transcription_text, active_model):
    """វិភាគតួអង្គ និងភេទសំឡេង"""
    prompt = f"""You are analyzing a transcript to identify different speakers.

Transcript:
{transcription_text}

Task:
1. Split the transcript into segments based on who is speaking.
2. For each segment, determine if the speaker is likely "male" or "female".
3. If you cannot determine the gender, alternate between male and female.

IMPORTANT: Return ONLY a valid JSON array. Do NOT add explanation or markdown.
Format:
[
  {{"speaker": "male", "text": "first sentence"}},
  {{"speaker": "female", "text": "second sentence"}}
]

Return the JSON array now:"""

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

    try:
        segments = json.loads(raw)
        if not isinstance(segments, list) or len(segments) == 0:
            raise ValueError("Invalid segments")
        return segments
    except Exception:
        return [{"speaker": "male", "text": transcription_text}]


def translate_segments(segments, target_lang, target_lang_name, active_model):
    """បកប្រែអត្ថបទនីមួយៗ"""
    translated_segments = []
    for seg in segments:
        prompt = (
            f"You are a professional translator. Translate the following text into {target_lang_name}. "
            f"Output ONLY the translation in {target_lang_name}. "
            f"Do NOT add explanations or notes:\n\n{seg['text']}"
        )
        res = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=active_model,
            temperature=0.3
        )
        translated_text = res.choices[0].message.content.strip()
        translated_segments.append({
            "speaker": seg["speaker"],
            "original": seg["text"],
            "translated": translated_text
        })
    return translated_segments


def get_voice(speaker, male_voice, female_voice):
    return male_voice if speaker == "male" else female_voice


def process_single_video(video_path, target_lang, target_lang_name,
                          male_voice, female_voice, progress, progress_start, progress_end):
    """ដំណើរការវីដេអូតែមួយ ហើយត្រឡប់ជា output path"""
    temp_dir = tempfile.gettempdir()
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    audio_extracted = os.path.join(temp_dir, f"{base_name}_audio.mp3")
    final_audio = os.path.join(temp_dir, f"{base_name}_final.mp3")
    output_video = os.path.join(temp_dir, f"{base_name}_dubbed.mp4")

    segment_files = []

    try:
        # ជំហានទី 1: ទាញសំឡេង
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

        # ជំហានទី 2: Whisper
        progress(progress_start + 0.15 * (progress_end - progress_start),
                 desc="កំពុងបម្លែងសំឡេងទៅជាអក្សរ...")
        with open(audio_extracted, "rb") as a_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.mp3", a_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )

        transcription_text = str(transcription).strip()
        if not transcription_text:
            raise Exception("រកមិនឃើញសំឡេងមនុស្សនិយាយ")

        active_model = get_active_chat_model()

        # ជំហានទី 3: វិភាគតួអង្គ
        progress(progress_start + 0.35 * (progress_end - progress_start),
                 desc="កំពុងវិភាគតួអង្គ...")
        segments = analyze_speakers(transcription_text, active_model)

        # ជំហានទី 4: បកប្រែ
        progress(progress_start + 0.55 * (progress_end - progress_start),
                 desc="កំពុងបកប្រែ...")
        translated_segments = translate_segments(
            segments, target_lang, target_lang_name, active_model
        )

        # ជំហានទី 5: បង្កើតសំឡេង
        progress(progress_start + 0.7 * (progress_end - progress_start),
                 desc="កំពុងបង្កើតសំឡេង AI...")
        for i, seg in enumerate(translated_segments):
            voice = get_voice(seg["speaker"], male_voice, female_voice)
            seg_file = os.path.join(temp_dir, f"{base_name}_seg_{i}.mp3")
            segment_files.append(seg_file)
            asyncio.run(generate_speech_segment(seg["translated"], voice, seg_file))

        # ជំហានទី 6: ផ្គុំសំឡេង
        progress(progress_start + 0.85 * (progress_end - progress_start),
                 desc="កំពុងផ្គុំសំឡេង...")
        concat_list = os.path.join(temp_dir, f"{base_name}_concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:a", "libmp3lame", "-ar", "24000",
            final_audio
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"ការផ្គុំសំឡេងបរាជ័យ: {result.stderr[:200]}")

        # ជំហានទី 7: ផ្គុំជាមួយវីដេអូ
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

        # បង្កើតសារសង្ខេប
        segments_info = "\n".join([
            f"• [{seg['speaker'].upper()}] {seg['translated'][:100]}"
            for seg in translated_segments
        ])
        summary = f"ចំនួនកំណាត់: {len(translated_segments)}\n{segments_info}"

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
        return None, [], "សូម Upload វីដេអូជាមុនសិន!"

    target_lang_name, male_voice, female_voice = get_lang_info(target_lang)

    temp_dir = tempfile.gettempdir()

    try:
        # ពិនិត្យថាតើត្រូវបំបែកវីដេអូឬអត់
        if segment_minutes and segment_minutes > 0:
            progress(0.02, desc="កំពុងពិនិត្យរយៈពេលវីដេអូ...")
            segments, duration = split_video(video_path, segment_minutes, temp_dir)

            if not segments:
                return None, [], "ការបំបែកវីដេអូបរាជ័យ!"

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

            # សម្អាតឯកសារបំបែក
            for seg in segments:
                if os.path.exists(seg):
                    try:
                        os.remove(seg)
                    except Exception:
                        pass

            if not outputs:
                return None, [], "គ្មានផ្នែកណាមួយដំណើរការបានសម្រេច!"

            status = (
                f" ជោគជ័យ! បានបំបែកវីដេអូជា {len(outputs)} ផ្នែក\n"
                f" រយៈពេលសរុប: {duration/60:.1f} នាទី\n"
                f" រយៈពេលកំណត់: {segment_minutes} នាទី/ផ្នែក\n\n"
                + "\n\n".join(summaries)
            )

            return outputs[0], outputs, status

        else:
            # ដំណើរការវីដេអូពេញលេញធម្មតា
            progress(0.05, desc="កំពុងដំណើរការវីដេអូពេញលេញ...")

            out, summary = process_single_video(
                video_path, target_lang, target_lang_name,
                male_voice, female_voice,
                progress, 0.0, 1.0
            )

            status = f" ជោគជ័យ!\n\n{summary}"
            return out, [out], status

    except Exception as e:
        return None, [], f"មានបញ្ហា៖ {str(e)}"


# ============================================================
# CSS សម្រាប់ Dashboard ស្អាត និងអក្សរខ្មែរ
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
    font-family: 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif;
}

.main-title h1 {
    font-size: 2.2em;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    line-height: 1.6;
}

.main-title p {
    color: #5a6478;
    font-size: 1.05em;
    line-height: 1.8;
}

.card-box {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-bottom: 16px;
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
    transition: border 0.2s !important;
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

.segment-gallery {
    margin-top: 16px;
}

footer {
    display: none !important;
}
"""


# ============================================================
# បង្កើត Interface
# ============================================================
with gr.Blocks(title="AI Video Dubbing Studio", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
        <div class="main-title">
            <h1>🎬 AI Video Dubbing Studio</h1>
            <p>បកប្រែ និងបញ្ចូលសំឡេង AI ទៅក្នុងវីដេអូរបស់អ្នកដោយស្វ័យប្រវត្តិ</p>
        </div>
    """)

    with gr.Row():
        # ============ ផ្នែកខាងឆ្វេង: ការកំណត់ ============
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

        # ============ ផ្នែកខាងស្ដាំ: លទ្ធផល ============
        with gr.Column(scale=2):
            gr.HTML('<div class="section-header">📺 លទ្ធផល</div>')

            video_output = gr.Video(label="🎥 វីដេអូដែលបានបកប្រែ (ផ្នែកទី ១)")

            status_output = gr.Textbox(
                label="📋 ស្ថានភាព និងព័ត៌មានលម្អិត",
                lines=12,
                show_copy_button=True
            )

    # ============ ផ្នែកខាងក្រោម: បញ្ជីផ្នែកទាំងអស់ ============
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
