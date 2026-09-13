import os
import sys
import shutil

# កំណត់ Encoding ជា UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

import subprocess
import asyncio
import tempfile
import gradio as gr
from groq import Groq
import edge_tts

# អាន Groq API Key ពី Environment Variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("សូមកំណត់ GROQ_API_KEY ជា environment variable")

groq_client = Groq(api_key=GROQ_API_KEY.strip())


async def generate_speech(text, voice, output_path):
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


def dub_video(video_path, target_lang, voice_gender, progress=gr.Progress()):
    if not video_path:
        return None, "សូម Upload វីដេអូជាមុនសិន!"

    temp_dir = tempfile.gettempdir()
    safe_input_video = os.path.join(temp_dir, "input_temp_video.mp4")
    audio_extracted = os.path.join(temp_dir, "temp_audio.mp3")
    dubbed_audio = os.path.join(temp_dir, "temp_dubbed.mp3")
    output_video = os.path.join(temp_dir, "output_dubbed.mp4")

    for f in [safe_input_video, audio_extracted, dubbed_audio, output_video]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    try:
        # ជំហានទី 1: ទាញសំឡេង
        progress(0.1, desc="កំពុងទាញសំឡេងចេញពីវីដេអូ...")
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
        progress(0.3, desc="កំពុងបម្លែងសំឡេងទៅជាអក្សរ (Whisper)...")
        with open(audio_extracted, "rb") as a_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.mp3", a_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )

        transcription_text = str(transcription).strip()
        if not transcription_text:
            return None, "រកមិនឃើញសំឡេងមនុស្សនិយាយនៅក្នុងវីដេអូនេះទេ!"

        # ជំហានទី 3: បកប្រែអត្ថបទ
        progress(0.6, desc="កំពុងបកប្រែអត្ថបទ...")
        active_model = get_active_chat_model()

        if target_lang == "km":
            prompt = (
                "You are a professional translator. Translate the following text into Khmer (ភាសាខ្មែរ). "
                "You MUST output ONLY the Khmer translation using Khmer script (អក្សរខ្មែរ). "
                "Do NOT output English, do NOT add explanations, do NOT add notes. "
                "Output ONLY the Khmer translation:\n\n"
                f"{transcription_text}"
            )
        else:
            prompt = (
                "You are a professional translator. Translate the following text into English. "
                "Output ONLY the English translation, without any introduction or notes:\n\n"
                f"{transcription_text}"
            )

        translation_res = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=active_model,
            temperature=0.3
        )
        translated_text = translation_res.choices[0].message.content.strip()
        if not translated_text:
            return None, "ការបកប្រែបរាជ័យ! សូមព្យាយាមម្ដងទៀត។"

        # ជំហានទី 4: បង្កើតសំឡេង AI តាមភេទដែលបានជ្រើស
        progress(0.8, desc="កំពុងបង្កើតសំឡេង AI ថ្មី...")
        
        # ជ្រើសរើសសំឡេងតាមភាសា និងភេទ
        if target_lang == "km":
            # សំឡេងខ្មែរ
            voice = "km-KH-PisethNeural" if voice_gender == "male" else "km-KH-SreymomNeural"
        else:
            # សំឡេងអង់គ្លេស
            voice = "en-US-GuyNeural" if voice_gender == "male" else "en-US-JennyNeural"
        
        asyncio.run(generate_speech(translated_text, voice, dubbed_audio))

        if not os.path.exists(dubbed_audio) or os.path.getsize(dubbed_audio) == 0:
            return None, "ការបង្កើតសំឡេង AI បរាជ័យ!"

        # ជំហានទី 5: ផ្គុំសំឡេងចូលវីដេអូ
        progress(0.9, desc="កំពុងផ្គុំសំឡេងចូលវីដេអូ...")
        merge_cmd = [
            "ffmpeg", "-fflags", "+igndts", "-y",
            "-i", safe_input_video,
            "-i", dubbed_audio,
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

        status_msg = (
            f" ជោគជ័យ!\n\n"
            f" ម៉ូដែលបកប្រែ: {active_model}\n"
            f" សំឡេងដែលប្រើ: {voice}\n\n"
            f" អត្ថបទដើម:\n{transcription_text}\n\n"
            f" អត្ថបទបកប្រែ:\n{translated_text}"
        )
        return output_video, status_msg

    except Exception as e:
        return None, f"មានបញ្ហា៖ {str(e)}"
    finally:
        for f in [safe_input_video, audio_extracted, dubbed_audio]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


# បង្កើត Web Interface
with gr.Blocks(title="AI Video Dubbing Tool") as demo:
    gr.Markdown("# 🎬 AI Video Translator & Dubbing Tool")
    gr.Markdown("Upload វីដេអូដើម្បីបកប្រែ និងបញ្ចូលសំឡេង AI ថ្មីដោយស្វ័យប្រវត្តិ។")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            target_lang = gr.Dropdown(
                choices=[("ភាសាខ្មែរ (Khmer)", "km"), ("English", "en")],
                value="km",
                label="ជ្រើសរើសភាសាគោលដៅ"
            )
            # បន្ថែមជម្រើសភេទសំឡេង
            voice_gender = gr.Radio(
                choices=[("សំឡេងប្រុស (Male)", "male"), ("សំឡេងស្រី (Female)", "female")],
                value="male",
                label="ជ្រើសរើសភេទសំឡេង"
            )
            submit_btn = gr.Button("ដំណើរការបកប្រែ", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="វីដេអូទទួលបាន")
            status_output = gr.Textbox(label="ស្ថានភាព និងអត្ថបទបកប្រែ", lines=10)

    submit_btn.click(
        fn=dub_video,
        inputs=[video_input, target_lang, voice_gender],  # បន្ថែម voice_gender
        outputs=[video_output, status_output]
    )

# ចំណុចសំខាន់: ការភ្ជាប់ Port សម្រាប់ Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
