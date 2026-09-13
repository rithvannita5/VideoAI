import os
import sys
import shutil
import json

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


async def generate_speech_segment(text, voice, output_path):
    """បង្កើតសំឡេងសម្រាប់អត្ថបទមួយកំណាត់"""
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


def analyze_speakers(transcription_text, target_lang, active_model):
    """
    ប្រើ Groq ដើម្បីវិភាគអត្ថបទ និងបែងចែកជាកំណាត់តាមតួអង្គ
    ត្រឡប់មកវិញជា list of dict: [{"speaker": "male"/"female", "text": "..."}]
    """
    prompt = f"""You are analyzing a transcript to identify different speakers.

Transcript:
{transcription_text}

Task:
1. Split the transcript into segments based on who is speaking.
2. For each segment, determine if the speaker is likely "male" or "female" based on context, content, or any available cues.
3. If you cannot determine the gender, alternate between male and female for dialogue.

IMPORTANT: Return ONLY a valid JSON array. Do NOT add any explanation, markdown formatting, or code blocks.
Format:
[
  {{"speaker": "male", "text": "first sentence in original language"}},
  {{"speaker": "female", "text": "second sentence in original language"}}
]

Return the JSON array now:"""

    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=active_model,
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()

    # សម្អាត markdown code block បើមាន
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
        # Fallback: ប្រើអត្ថបទទាំងមូលជាសំឡេងប្រុស
        return [{"speaker": "male", "text": transcription_text}]


def translate_segments(segments, target_lang, active_model):
    """បកប្រែអត្ថបទនីមួយៗទៅជាភាសាគោលដៅ"""
    if target_lang == "km":
        target_name = "Khmer (ភាសាខ្មែរ)"
        script_note = "You MUST output ONLY Khmer script (អក្សរខ្មែរ)."
    else:
        target_name = "English"
        script_note = "Output ONLY English."

    translated_segments = []
    for seg in segments:
        prompt = (
            f"You are a professional translator. Translate the following text into {target_name}. "
            f"{script_note} Do NOT add explanations or notes. "
            f"Output ONLY the translation:\n\n{seg['text']}"
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


def get_voice(speaker, target_lang):
    """ជ្រើសរើសសំឡេងតាមភេទ និងភាសា"""
    if target_lang == "km":
        return "km-KH-PisethNeural" if speaker == "male" else "km-KH-SreymomNeural"
    else:
        return "en-US-GuyNeural" if speaker == "male" else "en-US-JennyNeural"


def dub_video(video_path, target_lang, progress=gr.Progress()):
    if not video_path:
        return None, "សូម Upload វីដេអូជាមុនសិន!"

    temp_dir = tempfile.gettempdir()
    safe_input_video = os.path.join(temp_dir, "input_temp_video.mp4")
    audio_extracted = os.path.join(temp_dir, "temp_audio.mp3")
    final_audio = os.path.join(temp_dir, "final_audio.mp3")
    output_video = os.path.join(temp_dir, "output_dubbed.mp4")

    for f in [safe_input_video, audio_extracted, final_audio, output_video]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    segment_files = []

    try:
        # ជំហានទី 1: ទាញសំឡេង
        progress(0.05, desc="កំពុងទាញសំឡេងចេញពីវីដេអូ...")
        shutil.copyfile(str(video_path), safe_input_video)

        extract_cmd = [
            "ffmpeg", "-y", "-i", safe_input_video,
            "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-ab", "64k",
            audio_extracted
        ]
        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None, f"ការទាញសំឡេងបរាជ័យ!\n{result.stderr}"

        if not os.path.exists(audio_extracted) or os.path.getsize(audio_extracted) == 0:
            return None, "វីដេអូនេះគ្មានសំឡេងសម្រាប់បកប្រែទេ!"

        # ជំហានទី 2: បម្លែងសំឡេងទៅជាអក្សរ
        progress(0.15, desc="កំពុងបម្លែងសំឡេងទៅជាអក្សរ (Whisper)...")
        with open(audio_extracted, "rb") as a_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.mp3", a_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )

        transcription_text = str(transcription).strip()
        if not transcription_text:
            return None, "រកមិនឃើញសំឡេងមនុស្សនិយាយនៅក្នុងវីដេអូនេះទេ!"

        active_model = get_active_chat_model()

        # ជំហានទី 3: វិភាគតួអង្គ
        progress(0.35, desc="កំពុងវិភាគតួអង្គ និងភេទសំឡេង...")
        segments = analyze_speakers(transcription_text, target_lang, active_model)

        # ជំហានទី 4: បកប្រែកំណាត់នីមួយៗ
        progress(0.55, desc=f"កំពុងបកប្រែ {len(segments)} កំណាត់...")
        translated_segments = translate_segments(segments, target_lang, active_model)

        # ជំហានទី 5: បង្កើតសំឡេងសម្រាប់កំណាត់នីមួយៗ
        progress(0.7, desc="កំពុងបង្កើតសំឡេង AI សម្រាប់កំណាត់នីមួយៗ...")
        for i, seg in enumerate(translated_segments):
            voice = get_voice(seg["speaker"], target_lang)
            seg_file = os.path.join(temp_dir, f"seg_{i}.mp3")
            segment_files.append(seg_file)

            asyncio.run(generate_speech_segment(seg["translated"], voice, seg_file))

            if not os.path.exists(seg_file) or os.path.getsize(seg_file) == 0:
                return None, f"ការបង្កើតសំឡេងសម្រាប់កំណាត់ទី {i+1} បរាជ័យ!"

        # ជំហានទី 6: ផ្គុំសំឡេងកំណាត់ទាំងអស់ចូលគ្នា
        progress(0.85, desc="កំពុងផ្គុំសំឡេងកំណាត់ទាំងអស់...")
        concat_list = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            final_audio
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # បើ copy មិនដំណើរការ សាកល្បង re-encode
            concat_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c:a", "libmp3lame", "-ar", "24000",
                final_audio
            ]
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None, f"ការផ្គុំសំឡេងបរាជ័យ!\n{result.stderr}"

        if not os.path.exists(final_audio) or os.path.getsize(final_audio) == 0:
            return None, "ឯកសារសំឡេងចុងក្រោយមិនមាន!"

        # ជំហានទី 7: ផ្គុំសំឡេងចូលវីដេអូ
        progress(0.95, desc="កំពុងផ្គុំសំឡេងចូលវីដេអូ...")
        merge_cmd = [
            "ffmpeg", "-fflags", "+igndts", "-y",
            "-i", safe_input_video,
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
            return None, f"ការផ្គុំសំឡេងចូលវីដេអូបរាជ័យ!\n{result.stderr}"

        # បង្កើតសារស្ថានភាព
        segments_info = "\n".join([
            f"[{seg['speaker'].upper()}] {seg['translated'][:80]}..."
            for seg in translated_segments
        ])

        status_msg = (
            f" ជោគជ័យ!\n\n"
            f" ម៉ូដែល: {active_model}\n"
            f" ចំនួនកំណាត់: {len(translated_segments)}\n\n"
            f" កំណាត់ដែលបានបកប្រែ:\n{segments_info}"
        )
        return output_video, status_msg

    except Exception as e:
        return None, f"មានបញ្ហា៖ {str(e)}"
    finally:
        # សម្អាតឯកសារបណ្ដោះអាសន្ន
        for f in [safe_input_video, audio_extracted, final_audio]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        for f in segment_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


# បង្កើត Web Interface
with gr.Blocks(title="AI Video Dubbing Tool") as demo:
    gr.Markdown("# 🎬 AI Video Translator & Dubbing Tool")
    gr.Markdown(
        "Upload វីដេអូដើម្បីបកប្រែ ។ "
        "ប្រព័ន្ធនឹងវិភាគតួអង្គដោយស្វ័យប្រវត្តិ និងប្រើសំឡេងប្រុស/ស្រីតាមតួអង្គ ។"
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            target_lang = gr.Dropdown(
                choices=[("ភាសាខ្មែរ (Khmer)", "km"), ("English", "en")],
                value="km",
                label="ជ្រើសរើសភាសាគោលដៅ"
            )
            submit_btn = gr.Button("ដំណើរការបកប្រែ", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="វីដេអូទទួលបាន")
            status_output = gr.Textbox(label="ស្ថានភាព និងកំណាត់ដែលបានបកប្រែ", lines=10)

    submit_btn.click(
        fn=dub_video,
        inputs=[video_input, target_lang],
        outputs=[video_output, status_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
