# ==========================================
# AI Video Translator - SRT Mode
# Layout Manager + Font Fix
# ==========================================

import os
import sys
import re
import glob
import math
import shutil
import subprocess
import time
import asyncio
import requests
import webbrowser
import threading
import concurrent.futures
from datetime import timedelta

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def _quiet_qt_message_handler(mode, context, message):
    if 'OpenType support missing' in message:
        return
    try:
        sys.__stderr__.write(message + '\n')
    except:
        pass


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QProgressBar, QComboBox, QCheckBox,
    QLineEdit, QFrame, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

QtCore.qInstallMessageHandler(_quiet_qt_message_handler)


# ==========================================
# Folder Setup
# ==========================================
def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = get_base_dir()
FFMPEG_EXE = os.path.join(BASE_DIR, 'ffmpeg', 'ffmpeg.exe')
FFPROBE_EXE = os.path.join(BASE_DIR, 'ffmpeg', 'ffprobe.exe')

CREATION_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0


def out_path(*parts):
    return os.path.join(BASE_DIR, *parts)


# ==========================================
# Language Config
# ==========================================
TARGET_LANG_CONFIG = {
    'Khmer': {'code': 'km', 'female': 'km-KH-SreymomNeural', 'male': 'km-KH-PisethNeural'},
    'English': {'code': 'en', 'female': 'en-US-AriaNeural', 'male': 'en-US-GuyNeural'},
    'Thai': {'code': 'th', 'female': 'th-TH-PremwadeeNeural', 'male': 'th-TH-NiwatNeural'},
    'Chinese': {'code': 'zh-CN', 'female': 'zh-CN-XiaoxiaoNeural', 'male': 'zh-CN-YunxiNeural'},
    'Korean': {'code': 'ko', 'female': 'ko-KR-SunHiNeural', 'male': 'ko-KR-InJoonNeural'},
    'Vietnamese': {'code': 'vi', 'female': 'vi-VN-HoaiMyNeural', 'male': 'vi-VN-NamMinhNeural'},
    'Lao': {'code': 'lo', 'female': 'lo-LA-KeomanyNeural', 'male': 'lo-LA-ChanthavongNeural'}
}
DEFAULT_TARGET_LANG = 'Khmer'


# ==========================================
# Helper Functions
# ==========================================
def run_command(command):
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True,
            encoding='utf-8', errors='ignore', creationflags=CREATION_FLAGS
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def get_media_duration(media_path):
    command = [
        FFPROBE_EXE, '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(media_path)
    ]
    success, output = run_command(command)
    if success:
        try:
            return float(output.strip())
        except:
            return 0.0
    return 0.0


def collapse_repeats(text):
    if not text:
        return text
    text = text.strip()
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'(.+?)\1{2,}', r'\1', text)
    return text.strip()


def clean_text(text):
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([។៕៖ៗ])', r'\1', text)
    return text.strip()


def detect_text_language(text):
    if not text:
        return 'en'
    counts = {'km': 0, 'th': 0, 'zh': 0, 'ko': 0, 'ja': 0, 'en': 0}
    for ch in text:
        o = ord(ch)
        if 6016 <= o <= 6143:
            counts['km'] += 1
        elif 3584 <= o <= 3711:
            counts['th'] += 1
        elif 44032 <= o <= 55215 or 4352 <= o <= 4607:
            counts['ko'] += 1
        elif 12352 <= o <= 12543:
            counts['ja'] += 1
        elif 19968 <= o <= 40959:
            counts['zh'] += 1
        elif 'a' <= ch.lower() <= 'z':
            counts['en'] += 1
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else 'en'


# ==========================================
# Translation
# ==========================================
def translate_batch_fast(texts, source_lang="auto", target_lang="km"):
    if not texts:
        return texts

    if len(texts) == 1:
        return [translate_single(texts[0], source_lang, target_lang)]

    separator = "\n@@@\n"
    combined = separator.join(texts)

    try:
        from deep_translator import GoogleTranslator
        src = 'auto' if source_lang == 'auto' else source_lang
        translator = GoogleTranslator(source=src, target=target_lang)
        result = translator.translate(combined)

        if result:
            if "@@@" in result:
                parts = [p.strip() for p in result.split("@@@")]
            elif "\n\n" in result:
                parts = [p.strip() for p in result.split("\n\n")]
            else:
                parts = [p.strip() for p in result.split("\n") if p.strip()]

            if len(parts) == len(texts):
                return [clean_text(p) for p in parts]
    except Exception as e:
        print(f"Batch error: {e}")

    return None


def translate_single(text, source_lang="auto", target_lang="km"):
    if not text or not text.strip():
        return text

    try:
        from deep_translator import GoogleTranslator
        src = 'auto' if source_lang == 'auto' else source_lang
        translator = GoogleTranslator(source=src, target=target_lang)
        result = translator.translate(text)
        if result and result.strip():
            return clean_text(result)
    except Exception:
        pass

    try:
        from translate import Translator
        src2 = 'en' if source_lang == 'auto' else source_lang
        translator = Translator(from_lang=src2, to_lang=target_lang)
        result = translator.translate(text)
        if result and result.strip():
            return clean_text(result)
    except Exception:
        pass

    return text


# ==========================================
# Speaker Detection
# ==========================================
def detect_gender_simple(text):
    if not text:
        return None

    female_words = ['នាង', 'នារី', 'ស្រី', 'ម្តាយ', 'ម៉ែ', 'យាយ', 'អ្នកស្រី',
                    'បងស្រី', 'ប្អូនស្រី', 'ភរិយា', 'ប្រពន្ធ', 'ក្រមុំ']
    male_words = ['បុរស', 'ប្រុស', 'ឪពុក', 'ពុក', 'តា', 'អ្នកប្រុស',
                  'បងប្រុស', 'ប្អូនប្រុស', 'ស្វាមី', 'ប្តី', 'កំលោះ']

    for w in female_words:
        if w in text:
            return 'female'
    for w in male_words:
        if w in text:
            return 'male'

    return None


def estimate_pitch_yin(audio, sr):
    try:
        import numpy as np

        W = 1024
        hop = 512
        fmin = 70
        fmax = 400
        tau_min = int(sr / fmax)
        tau_max = int(sr / fmin)

        if len(audio) < W + tau_max:
            return None

        f0s = []

        for start in range(0, len(audio) - W - tau_max, hop):
            frame = audio[start:start + W + tau_max]
            base = frame[:W]

            d = np.zeros(tau_max + 1)
            for tau in range(tau_max + 1):
                d[tau] = np.sum((base - frame[tau:tau + W]) ** 2)

            cmnd = np.ones(tau_max + 1)
            running = 0
            for tau in range(1, tau_max + 1):
                running += d[tau]
                cmnd[tau] = d[tau] * tau / running if running > 0 else 1

            threshold = 0.15
            tau_est = -1
            for t in range(tau_min, tau_max):
                if cmnd[t] < threshold:
                    while t + 1 < tau_max and cmnd[t + 1] < cmnd[t]:
                        t += 1
                    tau_est = t
                    break

            if tau_est == -1:
                tau_est = tau_min + np.argmin(cmnd[tau_min:tau_max])
                if cmnd[tau_est] > 0.5:
                    continue

            if 1 < tau_est < tau_max:
                a, b, c = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
                denom = a + c - 2 * b
                if denom != 0:
                    tau_est = tau_est + 0.5 * (a - c) / denom

            if tau_est > 0:
                f0 = sr / tau_est
                if fmin <= f0 <= fmax:
                    f0s.append(f0)

        if len(f0s) < 3:
            return None

        return float(np.median(f0s))
    except Exception as e:
        print(f"YIN error: {e}")
        return None


def detect_gender_from_audio(audio_path, start_time, end_time):
    try:
        import numpy as np

        duration = end_time - start_time
        if duration <= 0.1:
            return None

        cmd = [
            FFMPEG_EXE, '-y', '-i', audio_path,
            '-ss', str(start_time), '-t', str(duration),
            '-ar', '16000', '-ac', '1',
            '-f', 's16le', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, creationflags=CREATION_FLAGS)
        audio = np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0

        if audio.size < 1600:
            return None

        f0 = estimate_pitch_yin(audio, 16000)
        if f0 is None:
            return None

        if f0 < 165:
            return 'male'
        elif f0 > 165:
            return 'female'

        return None
    except Exception as e:
        print(f"Gender audio error: {e}")
        return None


# ==========================================
# Dubbing Worker Thread
# ==========================================
class DubbingWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, srt_path, voice_var='auto',
                 target_lang='Khmer', source_lang='auto'):
        super().__init__()
        self.video_path = video_path
        self.srt_path = srt_path
        self.voice_var = voice_var
        self.target_lang = target_lang if target_lang in TARGET_LANG_CONFIG else DEFAULT_TARGET_LANG
        self.target_cfg = TARGET_LANG_CONFIG[self.target_lang]
        self.source_lang = source_lang

    def log(self, text):
        self.log_signal.emit(text)

    def update_progress(self, percent, text_status):
        self.progress_signal.emit(int(percent), text_status)

    def run(self):
        try:
            import srt
            import edge_tts

            self.log(f'🚀 ចាប់ផ្តើម — ភាសាគោលដៅ: {self.target_lang}')
            self.update_progress(2, 'កំពុងផ្ទុក... | ⏱️ នៅសល់: --')

            if not os.path.exists(FFMPEG_EXE):
                raise Exception(f'រកមិនឃើញ FFmpeg!\n{FFMPEG_EXE}')

            srt_dir = out_path('2_srt_output')
            temp_dir = out_path('0_temp_output')
            seg_dir = out_path('3_audio_segments')

            for d in [srt_dir, temp_dir]:
                os.makedirs(d, exist_ok=True)
            if os.path.exists(seg_dir):
                shutil.rmtree(seg_dir)
            os.makedirs(seg_dir)

            filename = os.path.basename(self.video_path)
            base = os.path.splitext(filename)[0]
            final_out = os.path.join(temp_dir, f'{self.target_lang}_{filename}')

            # ========== STEP 1: អាន SRT ==========
            self.update_progress(5, 'Step 1: កំពុងអាន SRT... | ⏱️ --')
            self.log(f'\n[STEP 1] កំពុងអាន SRT: {os.path.basename(self.srt_path)}')

            try:
                with open(self.srt_path, 'r', encoding='utf-8-sig') as f:
                    raw = f.read()
            except UnicodeDecodeError:
                with open(self.srt_path, 'r', encoding='cp1252') as f:
                    raw = f.read()

            subs = list(srt.parse(raw))
            for i, s in enumerate(subs, start=1):
                s.index = i

            total_subs = len(subs)
            self.log(f'   ✅ អានបាន {total_subs} បន្ទាត់')

            sample = ' '.join(s.content for s in subs[:20])
            detected_lang = detect_text_language(sample)
            self.log(f'   ភាសា SRT: {detected_lang}')

            # ========== STEP 2: បកប្រែលឿន ==========
            self.log(f'\n[STEP 2] កំពុងបកប្រែទៅ {self.target_lang}...')
            target_code = self.target_cfg['code']

            all_texts = []
            text_indices = []
            for i, sub in enumerate(subs):
                txt = sub.content.strip()
                if txt:
                    all_texts.append(txt)
                    text_indices.append(i)

            if all_texts:
                batch_size = 50
                batches = []
                for i in range(0, len(all_texts), batch_size):
                    batches.append(all_texts[i:i + batch_size])

                num_batches = len(batches)
                self.log(f'   📦 ចំនួន Batch: {num_batches} (Batch Size: {batch_size})')
                self.log(f'   ⚡ ប្រើ 5 Threads ក្នុងពេលតែមួយ...')

                step2_start = time.time()
                all_translated = []
                completed_batches = [0]
                lock = threading.Lock()

                def process_batch(b_idx, batch):
                    try:
                        result = translate_batch_fast(batch, detected_lang, target_code)
                        if not result or len(result) != len(batch):
                            result = [translate_single(t, detected_lang, target_code) for t in batch]
                    except Exception as e:
                        self.log(f'   ⚠️ Batch {b_idx+1} error: {e}')
                        result = [translate_single(t, detected_lang, target_code) for t in batch]

                    with lock:
                        completed_batches[0] += 1
                        completed = completed_batches[0]

                    elapsed = time.time() - step2_start
                    progress_ratio = completed / num_batches

                    if progress_ratio > 0.05 and elapsed > 2:
                        total_est = elapsed / progress_ratio
                        remaining = total_est - elapsed
                        if remaining > 60:
                            mins = int(remaining / 60)
                            secs = int(remaining % 60)
                            time_text = f"⏱️ នៅសល់: {mins} នាទី {secs} វិនាទី"
                        elif remaining > 0:
                            time_text = f"⏱️ នៅសល់: {int(remaining)} វិនាទី"
                        else:
                            time_text = "⏱️ ជិតរួចហើយ!"
                    else:
                        time_text = "⏱️ កំពុងគណនា..."

                    percent = 15 + int(progress_ratio * 40)
                    self.log(f'   [{completed}/{num_batches}] ✅ Batch {b_idx+1}')
                    self.update_progress(percent,
                        f'Step 2: បកប្រែ {completed}/{num_batches} Batch | {time_text}')

                    return b_idx, result

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(process_batch, i, b) for i, b in enumerate(batches)]
                    results_dict = {}
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            b_idx, result = f.result()
                            results_dict[b_idx] = result
                        except Exception as e:
                            self.log(f'   ⚠️ Thread error: {e}')

                for i in range(num_batches):
                    if i in results_dict:
                        all_translated.extend(results_dict[i])
                    else:
                        all_translated.extend(batches[i])

                total_time = time.time() - step2_start
                self.log(f'   ✅ បកប្រែរួចរាល់ក្នុង {total_time:.1f} វិនាទី')

                for idx, trans in zip(text_indices, all_translated):
                    subs[idx].content = collapse_repeats(trans)

            srt_km_path = os.path.join(srt_dir, f'{base}_{target_code}.srt')
            with open(srt_km_path, 'w', encoding='utf-8') as f:
                f.write(srt.compose(subs))

            self.log(f'   ✅ បកប្រែរួចរាល់: {srt_km_path}')

            # ========== STEP 3a: រកយេនឌ័រ ==========
            genders = {}
            if self.voice_var == 'auto':
                self.update_progress(55, 'Step 3a: កំពុងរកតួអង្គ... | ⏱️ --')
                self.log(f'\n[STEP 3a] កំពុងរកយេនឌ័រតួអង្គ...')

                temp_audio = out_path('_full_audio.wav')
                subprocess.run([
                    FFMPEG_EXE, '-y', '-i', self.video_path,
                    '-vn', '-ar', '16000', '-ac', '1',
                    temp_audio
                ], capture_output=True, creationflags=CREATION_FLAGS)

                for i, sub in enumerate(subs, 1):
                    g = detect_gender_simple(sub.content)
                    if not g:
                        start = sub.start.total_seconds()
                        end = sub.end.total_seconds()
                        g = detect_gender_from_audio(temp_audio, start, end)
                    genders[sub.index] = g or 'female'

                if os.path.exists(temp_audio):
                    os.remove(temp_audio)

                n_male = sum(1 for g in genders.values() if g == 'male')
                n_female = len(genders) - n_male
                self.log(f'   📊 ប្រុស: {n_male}, ស្រី: {n_female}')

            # ========== STEP 3b: បង្កើតសំលេង ==========
            self.update_progress(65, 'Step 3b: កំពុងបង្កើតសំលេង... | ⏱️ --')
            self.log(f'\n[STEP 3b] កំពុងបង្កើតសំលេង AI...')

            female_voice = self.target_cfg['female']
            male_voice = self.target_cfg['male']

            async def make_voice_all():
                sem = asyncio.Semaphore(15)

                async def make_voice(sub):
                    txt = sub.content.strip()
                    if not txt:
                        return
                    ms = int(sub.start.total_seconds() * 1000)
                    out_mp3 = os.path.join(seg_dir, f'{sub.index:04d}_{ms}.mp3')

                    if self.voice_var == 'female':
                        voice_id = female_voice
                    elif self.voice_var == 'male':
                        voice_id = male_voice
                    else:
                        g = genders.get(sub.index, 'female')
                        voice_id = male_voice if g == 'male' else female_voice

                    async with sem:
                        for attempt in range(3):
                            try:
                                tts = edge_tts.Communicate(txt, voice_id, rate='+10%')
                                await tts.save(out_mp3)
                                if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 100:
                                    return
                            except Exception:
                                await asyncio.sleep(0.5)

                await asyncio.gather(*[make_voice(s) for s in subs])

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(make_voice_all())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

            self.log(f'   ✅ បង្កើតសំលេងរួចរាល់')

            # ========== STEP 4: Merge ==========
            self.update_progress(85, 'Step 4: កំពុង Merge... | ⏱️ នៅសល់: 1-2 នាទី')
            self.log(f'\n[STEP 4] កំពុង Merge វីដេអូ + សំលេង...')

            raw_segments = sorted(glob.glob(os.path.join(seg_dir, '[0-9]*.mp3')))
            segments = [seg for seg in raw_segments if os.path.getsize(seg) >= 200]

            if not segments:
                raise RuntimeError('គ្មានឯកសារសំលេង!')

            video_dur = get_media_duration(self.video_path)
            if video_dur <= 0:
                video_dur = 3600

            silent_wav = out_path('_silent_base.wav')
            subprocess.run([
                FFMPEG_EXE, '-y',
                '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                '-t', str(video_dur), '-ar', '44100',
                silent_wav
            ], capture_output=True, creationflags=CREATION_FLAGS)

            current_mix = silent_wav
            batch_size = 20
            batch_idx = 0

            for start_idx in range(0, len(segments), batch_size):
                batch = segments[start_idx:start_idx + batch_size]
                n = len(batch)
                batch_out = out_path(f'_mix_batch_{batch_idx}.wav')

                filter_parts = []
                for i, seg in enumerate(batch):
                    m = re.search(r'_(\d+)\.mp3$', seg)
                    ms_v = m.group(1) if m else '0'
                    filter_parts.append(f'[{i + 1}:a]adelay={ms_v}|{ms_v}[a{i}]')

                mix_labels = ''.join([f'[a{i}]' for i in range(n)])
                fc = ';\n'.join(filter_parts)
                fc += f';\n{mix_labels}amix=inputs={n}:duration=longest:normalize=0[seg];'
                fc += '[0:a][seg]amix=inputs=2:duration=longest:normalize=0[out]'

                cmd = [FFMPEG_EXE, '-y', '-i', current_mix]
                for seg in batch:
                    cmd.extend(['-i', seg])
                cmd.extend(['-filter_complex', fc, '-map', '[out]', '-ar', '44100', batch_out])

                subprocess.run(cmd, capture_output=True, creationflags=CREATION_FLAGS)

                if current_mix != silent_wav and os.path.exists(current_mix):
                    os.remove(current_mix)
                current_mix = batch_out
                batch_idx += 1

            dubbed_audio = current_mix

            cmd_final = [
                FFMPEG_EXE, '-y',
                '-i', self.video_path,
                '-i', dubbed_audio,
                '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                '-shortest', final_out
            ]
            subprocess.run(cmd_final, capture_output=True, creationflags=CREATION_FLAGS)

            for tmp in glob.glob(out_path('_mix_batch_*.wav')):
                try:
                    os.remove(tmp)
                except:
                    pass
            if os.path.exists(silent_wav):
                os.remove(silent_wav)
            if os.path.exists(seg_dir):
                shutil.rmtree(seg_dir)

            self.update_progress(100, 'រួចរាល់! 100% | ✅ បានបញ្ចប់')
            self.log(f'\n✅ ជោគជ័យ! វីដេអូ: {final_out}')
            self.finished_signal.emit(os.path.abspath(final_out))

        except Exception as e:
            import traceback
            self.error_signal.emit(f'{str(e)}\n\n{traceback.format_exc()}')


# ==========================================
# Style Sheet
# ==========================================
STYLE_SHEET = """
QDialog {
    background-color: #F8FAFC;
}

* {
    font-family: 'Segoe UI', 'Khmer OS', 'Arial';
}

QLabel {
    color: #1E293B;
    font-size: 12px;
}

QLabel#titleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #1E293B;
    padding: 6px;
}

QLabel#sectionLabel {
    font-size: 14px;
    font-weight: bold;
    color: #2563EB;
    padding: 4px;
}

QLabel#sectionLabel2 {
    font-size: 14px;
    font-weight: bold;
    color: #F97316;
    padding: 4px;
}

QPushButton {
    font-size: 11px;
    font-weight: bold;
    border-radius: 6px;
    padding: 4px 8px;
    border: none;
    min-height: 28px;
    color: white;
}

QPushButton#btnBlue {
    background-color: #2563EB;
}
QPushButton#btnBlue:hover { background-color: #1D4ED8; }

QPushButton#btnOrange {
    background-color: #F97316;
}
QPushButton#btnOrange:hover { background-color: #EA580C; }

QPushButton#btnGreen {
    background-color: #16A34A;
    font-size: 14px;
    min-height: 40px;
}
QPushButton#btnGreen:hover { background-color: #15803D; }

QPushButton#btnPurple {
    background-color: #8B5CF6;
}
QPushButton#btnPurple:hover { background-color: #7C3AED; }

QPushButton#btnRed {
    background-color: #E11D48;
}
QPushButton#btnRed:hover { background-color: #BE123C; }

QPushButton#btnGray {
    background-color: #475569;
}
QPushButton#btnGray:hover { background-color: #334155; }

QPushButton#btnVoiceActive {
    background-color: #2563EB;
    border: 2px solid #1D4ED8;
    font-size: 11px;
    min-height: 30px;
}

QPushButton#btnVoiceInactive {
    background-color: #F1F5F9;
    color: #334155;
    border: 2px solid #CBD5E1;
    font-size: 11px;
    min-height: 30px;
}
QPushButton#btnVoiceInactive:hover { background-color: #E2E8F0; }

QTextEdit {
    border: 2px solid #CBD5E1;
    border-radius: 8px;
    background-color: white;
    color: #1E293B;
    font-size: 11px;
    padding: 6px;
}
QTextEdit#splitLog {
    border: 2px solid #F97316;
}

QProgressBar {
    border: 2px solid #16A34A;
    border-radius: 8px;
    background-color: #F0FDF4;
    text-align: center;
    color: #1E293B;
    font-weight: bold;
    font-size: 12px;
    min-height: 24px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #22C55E, stop:0.5 #16A34A, stop:1 #15803D);
    border-radius: 6px;
}

QComboBox {
    border: 2px solid #2563EB;
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 12px;
    font-weight: bold;
    background-color: white;
    min-height: 30px;
}

QCheckBox {
    font-size: 12px;
    font-weight: bold;
    color: #334155;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px; height: 18px;
    border: 2px solid #CBD5E1;
    border-radius: 4px;
    background-color: white;
}
QCheckBox::indicator:checked {
    background-color: #F97316;
    border: 2px solid #EA580C;
}

QLineEdit {
    border: 2px solid #F97316;
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 13px;
    font-weight: bold;
    background-color: white;
    color: #1E293B;
    min-height: 28px;
}
QLineEdit:focus { border: 2px solid #EA580C; }

QLabel#videoPreview {
    border: 2px dashed #CBD5E1;
    border-radius: 8px;
    background-color: #F1F5F9;
    color: #64748B;
    font-size: 12px;
    font-weight: bold;
    padding: 8px;
}
QLabel#videoPreviewActive {
    border: 2px solid #16A34A;
    border-radius: 8px;
    background-color: #F0FDF4;
    color: #166534;
    font-size: 12px;
    font-weight: bold;
    padding: 8px;
}
QLabel#progressLabel {
    font-size: 12px;
    font-weight: bold;
    color: #1E293B;
    background-color: #EFF6FF;
    border: 2px solid #BFDBFE;
    border-radius: 6px;
    padding: 4px;
}
QLabel#timeLabel {
    font-size: 11px;
    font-weight: bold;
    color: #92400E;
    background-color: #FEF3C7;
    border: 2px solid #FCD34D;
    border-radius: 6px;
    padding: 4px;
}
QLabel#statusLabel {
    font-size: 11px;
    font-weight: bold;
    color: #2563EB;
    padding: 4px;
}
QLabel#linkLabel {
    font-size: 11px;
    color: #2563EB;
    font-weight: bold;
    background-color: #EFF6FF;
    border: 2px solid #BFDBFE;
    border-radius: 6px;
    padding: 6px;
}
QLabel#linkLabel:hover {
    background-color: #DBEAFE;
}
QLabel#splitInfo {
    font-size: 10px;
    color: #64748B;
    font-style: italic;
}
QFrame#divider {
    background-color: #F97316;
    max-height: 3px;
    min-height: 3px;
}
"""


# ==========================================
# UI - Layout Manager Version
# ==========================================
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName('Dialog')
        Dialog.setWindowTitle('AI Video Translator - SRT Mode')

        Dialog.setMinimumSize(1200, 750)
        Dialog.setStyleSheet(STYLE_SHEET)

        main_layout = QVBoxLayout(Dialog)
        main_layout.setContentsMargins(15, 10, 15, 10)
        main_layout.setSpacing(8)

        # ========== Title ==========
        self.titleLabel = QLabel('🎬 AI Video Translator - SRT Mode')
        self.titleLabel.setObjectName('titleLabel')
        self.titleLabel.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.titleLabel)

        # ========== Row 1: Section 1 ==========
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(10)

        # ---- Left: Log + Progress ----
        left1_layout = QVBoxLayout()
        left1_layout.setSpacing(6)

        self.labelSection1 = QLabel('📝 ផ្នែកទី ១: បកប្រែវីដេអូ')
        self.labelSection1.setObjectName('sectionLabel')
        self.labelSection1.setAlignment(Qt.AlignCenter)
        left1_layout.addWidget(self.labelSection1)

        self.SystemLog = QTextEdit()
        self.SystemLog.setReadOnly(True)
        self.SystemLog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left1_layout.addWidget(self.SystemLog, stretch=1)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(6)

        self.labelProgress = QLabel('📊 ភាគរយ: 0%')
        self.labelProgress.setObjectName('progressLabel')
        self.labelProgress.setAlignment(Qt.AlignCenter)
        progress_row.addWidget(self.labelProgress, stretch=1)

        self.labelTimeLeft = QLabel('⏱️ នៅសល់: --')
        self.labelTimeLeft.setObjectName('timeLabel')
        self.labelTimeLeft.setAlignment(Qt.AlignCenter)
        progress_row.addWidget(self.labelTimeLeft, stretch=1)

        left1_layout.addLayout(progress_row)

        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.progressBar.setFormat("0%")
        left1_layout.addWidget(self.progressBar)

        self.label = QLabel('រួចរាល់ដើម្បីចាប់ផ្តើម')
        self.label.setObjectName('statusLabel')
        self.label.setAlignment(Qt.AlignCenter)
        left1_layout.addWidget(self.label)

        row1_layout.addLayout(left1_layout, stretch=1)

        # ---- Right: Controls ----
        right1_layout = QVBoxLayout()
        right1_layout.setSpacing(6)

        browse_row = QHBoxLayout()
        browse_row.setSpacing(6)

        self.pushButton = QPushButton('📹 ជ្រើសរើសវីដេអូ')
        self.pushButton.setObjectName('btnBlue')
        self.pushButton.setMinimumHeight(36)
        browse_row.addWidget(self.pushButton)

        self.pushButton_srt = QPushButton('📄 ជ្រើសរើស SRT')
        self.pushButton_srt.setObjectName('btnOrange')
        self.pushButton_srt.setMinimumHeight(36)
        browse_row.addWidget(self.pushButton_srt)

        right1_layout.addLayout(browse_row)

        voice_row = QHBoxLayout()
        voice_row.setSpacing(6)

        self.pushButton_2 = QPushButton('🎙️ ស្រី')
        self.pushButton_2.setObjectName('btnVoiceInactive')
        self.pushButton_2.setMinimumHeight(32)
        voice_row.addWidget(self.pushButton_2)

        self.pushButton_3 = QPushButton('🎙️ ប្រុស')
        self.pushButton_3.setObjectName('btnVoiceInactive')
        self.pushButton_3.setMinimumHeight(32)
        voice_row.addWidget(self.pushButton_3)

        self.pushButton_auto = QPushButton('🎙️ ស្វ័យប្រវត្តិ')
        self.pushButton_auto.setObjectName('btnVoiceInactive')
        self.pushButton_auto.setMinimumHeight(32)
        voice_row.addWidget(self.pushButton_auto)

        right1_layout.addLayout(voice_row)

        self.labelLang = QLabel('ភាសាគោលដៅ:')
        self.labelLang.setStyleSheet('font-size: 11px; font-weight: bold; color: #334155;')
        right1_layout.addWidget(self.labelLang)

        self.comboLang = QComboBox()
        self.comboLang.addItems(['Khmer', 'English', 'Thai', 'Chinese', 'Korean', 'Vietnamese', 'Lao'])
        self.comboLang.setMinimumHeight(32)
        right1_layout.addWidget(self.comboLang)

        self.pushButton_6 = QPushButton('🚀 ចាប់ផ្តើម')
        self.pushButton_6.setObjectName('btnGreen')
        self.pushButton_6.setMinimumHeight(45)
        right1_layout.addWidget(self.pushButton_6)

        self.pushButton_new = QPushButton('🆕 គម្រោងថ្មី')
        self.pushButton_new.setObjectName('btnRed')
        self.pushButton_new.setMinimumHeight(32)
        right1_layout.addWidget(self.pushButton_new)

        self.linkLabel = QLabel('🔗 ដើម្បីទាញ SRT សូមចុចតំណរនេះ (voicecheap.ai)')
        self.linkLabel.setObjectName('linkLabel')
        self.linkLabel.setAlignment(Qt.AlignCenter)
        self.linkLabel.setCursor(Qt.PointingHandCursor)
        self.linkLabel.setMinimumHeight(34)
        right1_layout.addWidget(self.linkLabel)

        play_row = QHBoxLayout()
        play_row.setSpacing(6)

        self.btnPlayVideo = QPushButton('▶️ Play វីដេអូ')
        self.btnPlayVideo.setObjectName('btnPurple')
        self.btnPlayVideo.setMinimumHeight(36)
        self.btnPlayVideo.setEnabled(False)
        play_row.addWidget(self.btnPlayVideo)

        self.btnSaveVideo = QPushButton('💾 រក្សាទុក')
        self.btnSaveVideo.setObjectName('btnGreen')
        self.btnSaveVideo.setMinimumHeight(36)
        self.btnSaveVideo.setEnabled(False)
        play_row.addWidget(self.btnSaveVideo)

        right1_layout.addLayout(play_row)

        right1_layout.addStretch()

        row1_layout.addLayout(right1_layout, stretch=1)

        main_layout.addLayout(row1_layout, stretch=3)

        # ========== Divider ==========
        self.dividerLine = QFrame()
        self.dividerLine.setObjectName('divider')
        self.dividerLine.setFrameShape(QFrame.HLine)
        main_layout.addWidget(self.dividerLine)

        # ========== Row 2: Section 2 ==========
        self.labelSection2 = QLabel('✂️ ផ្នែកទី ២: បំបែកវីដេអូ')
        self.labelSection2.setObjectName('sectionLabel2')
        self.labelSection2.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.labelSection2)

        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(10)

        left2_layout = QVBoxLayout()
        left2_layout.setSpacing(6)

        self.videoPreviewLabel = QLabel("🎬 មិនទាន់មានវីដេអូ\n\n(រង់ចាំបកប្រែ ឬ Upload)")
        self.videoPreviewLabel.setObjectName('videoPreview')
        self.videoPreviewLabel.setAlignment(Qt.AlignCenter)
        self.videoPreviewLabel.setWordWrap(True)
        self.videoPreviewLabel.setMinimumHeight(250)
        self.videoPreviewLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left2_layout.addWidget(self.videoPreviewLabel, stretch=1)

        self.btnUploadSplitVideo = QPushButton('📤 Upload វីដេអូផ្ទាល់ខ្លួន')
        self.btnUploadSplitVideo.setObjectName('btnOrange')
        self.btnUploadSplitVideo.setMinimumHeight(40)
        left2_layout.addWidget(self.btnUploadSplitVideo)

        row2_layout.addLayout(left2_layout, stretch=1)

        right2_layout = QVBoxLayout()
        right2_layout.setSpacing(6)

        self.SplitLog = QTextEdit()
        self.SplitLog.setObjectName('splitLog')
        self.SplitLog.setReadOnly(True)
        self.SplitLog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right2_layout.addWidget(self.SplitLog, stretch=1)

        self.checkSplitAfter = QCheckBox('✂️ បំបែកវីដេអូវែង')
        self.checkSplitAfter.setChecked(True)
        right2_layout.addWidget(self.checkSplitAfter)

        self.checkMergeLast = QCheckBox('🔗 ចុងក្រោយបញ្ចូលជាមួយផ្នែកចុងក្រោយ')
        self.checkMergeLast.setChecked(True)
        right2_layout.addWidget(self.checkMergeLast)

        minutes_row = QHBoxLayout()
        minutes_row.setSpacing(6)

        self.labelSplitEvery2 = QLabel('រៀងរាល់')
        self.labelSplitEvery2.setStyleSheet('font-size: 11px; color: #334155; font-weight: bold;')
        minutes_row.addWidget(self.labelSplitEvery2)

        self.inputSplitMinutes2 = QLineEdit('6')
        self.inputSplitMinutes2.setAlignment(Qt.AlignCenter)
        self.inputSplitMinutes2.setMaximumWidth(70)
        self.inputSplitMinutes2.setMinimumHeight(30)
        minutes_row.addWidget(self.inputSplitMinutes2)

        self.labelSplitMinutes2 = QLabel('នាទី')
        self.labelSplitMinutes2.setStyleSheet('font-size: 11px; color: #334155; font-weight: bold;')
        minutes_row.addWidget(self.labelSplitMinutes2)

        self.labelSplitInfo2 = QLabel('')
        self.labelSplitInfo2.setObjectName('splitInfo')
        minutes_row.addWidget(self.labelSplitInfo2, stretch=1)

        right2_layout.addLayout(minutes_row)

        self.btnSplitNow = QPushButton('✂️ បំបែកវីដេអូឥឡូវ')
        self.btnSplitNow.setObjectName('btnOrange')
        self.btnSplitNow.setMinimumHeight(40)
        right2_layout.addWidget(self.btnSplitNow)

        row2_layout.addLayout(right2_layout, stretch=1)

        main_layout.addLayout(row2_layout, stretch=3)


# ==========================================
# Main App
# ==========================================
class MainApp(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.selected_video = ''
        self.selected_srt = ''
        self.voice_var = 'auto'
        self.temp_output_video = ''
        self.split_video_path = ''
        self.lang_keys = ['Khmer', 'English', 'Thai', 'Chinese', 'Korean', 'Vietnamese', 'Lao']

        self.pushButton.clicked.connect(self.select_video)
        self.pushButton_srt.clicked.connect(self.select_srt)
        self.pushButton_2.clicked.connect(lambda: self.set_voice('female'))
        self.pushButton_3.clicked.connect(lambda: self.set_voice('male'))
        self.pushButton_auto.clicked.connect(lambda: self.set_voice('auto'))
        self.pushButton_6.clicked.connect(self.start_thread)
        self.pushButton_new.clicked.connect(self.new_project)

        self.btnPlayVideo.clicked.connect(self.play_video)
        self.btnSaveVideo.clicked.connect(self.save_video)
        self.btnSplitNow.clicked.connect(self.do_split_video)
        self.btnUploadSplitVideo.clicked.connect(self.upload_split_video)
        self.inputSplitMinutes2.textChanged.connect(self.update_split_info2)

        self.linkLabel.mousePressEvent = self.open_srt_link

        self.update_voice_buttons()
        self.set_voice('auto')
        self.update_split_info2()

    def set_voice(self, voice_type):
        self.voice_var = voice_type
        labels = {'female': 'ស្រី', 'male': 'ប្រុស', 'auto': 'ស្វ័យប្រវត្តិ'}
        self.log(f'🎙️ សំលេង: {labels.get(voice_type, voice_type)}')
        self.update_voice_buttons()

    def update_voice_buttons(self):
        buttons = {
            'female': self.pushButton_2,
            'male': self.pushButton_3,
            'auto': self.pushButton_auto
        }
        for key, btn in buttons.items():
            if self.voice_var == key:
                btn.setObjectName('btnVoiceActive')
            else:
                btn.setObjectName('btnVoiceInactive')
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def log(self, text):
        self.SystemLog.append(text)
        sb = self.SystemLog.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_progress(self, percent, text_status):
        self.progressBar.setValue(percent)
        self.progressBar.setFormat(f"{percent}%")
        self.labelProgress.setText(f'📊 ភាគរយ: {percent}%')

        if '|' in text_status:
            parts = text_status.split('|')
            self.label.setText(parts[0].strip())
            if len(parts) > 1:
                self.labelTimeLeft.setText(parts[1].strip())
        else:
            self.label.setText(text_status)
            self.labelTimeLeft.setText('⏱️ នៅសល់: --')

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, 'ជ្រើសរើសវីដេអូ', '',
            'Video Files (*.mp4 *.mkv *.avi *.mov *.m4v)'
        )
        if file:
            self.selected_video = file
            self.log(f'📹 វីដេអូ: {os.path.basename(file)}')

    def select_srt(self):
        file, _ = QFileDialog.getOpenFileName(
            self, 'ជ្រើសរើស SRT', '',
            'Subtitle Files (*.srt)'
        )
        if file:
            self.selected_srt = file
            self.log(f'📄 SRT: {os.path.basename(file)}')

    def open_srt_link(self, event):
        webbrowser.open('https://www.voicecheap.ai/tools/video-to-srt')
        self.log('🔗 បើកតំណរទាញ SRT: voicecheap.ai')

    def start_thread(self):
        if not self.selected_video:
            QMessageBox.critical(self, 'កំហុស', 'សូមជ្រើសរើសវីដេអូជាមុនសិន!')
            return

        if not self.selected_srt:
            QMessageBox.critical(self, 'កំហុស', 'សូមជ្រើសរើសឯកសារ SRT!')
            return

        self.pushButton_6.setEnabled(False)
        self.pushButton_6.setText('កំពុងដំណើរការ...')
        self.SystemLog.clear()
        self.progressBar.setValue(0)
        self.progressBar.setFormat("0%")
        self.labelProgress.setText('📊 ភាគរយ: 0%')
        self.labelTimeLeft.setText('⏱️ នៅសល់: --')

        target_idx = max(0, self.comboLang.currentIndex())
        target_lang = self.lang_keys[target_idx] if target_idx < len(self.lang_keys) else DEFAULT_TARGET_LANG

        self.log(f'📹 វីដេអូ: {os.path.basename(self.selected_video)}')
        self.log(f'📄 SRT: {os.path.basename(self.selected_srt)}')
        self.log(f'🎙️ សំលេង: {self.voice_var}')
        self.log(f'🌐 ភាសាគោលដៅ: {target_lang}')
        self.log('')

        self.worker = DubbingWorker(
            video_path=self.selected_video,
            srt_path=self.selected_srt,
            voice_var=self.voice_var,
            target_lang=target_lang,
            source_lang='auto'
        )
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self, path):
        self.pushButton_6.setEnabled(True)
        self.pushButton_6.setText('🚀 ចាប់ផ្តើម')
        self.temp_output_video = path

        self.show_video_in_preview(path)

        self.btnPlayVideo.setEnabled(True)
        self.btnSaveVideo.setEnabled(True)

        QMessageBox.information(self, 'ជោគជ័យ',
            f'ដំណើរការបានជោគជ័យ!\n\n'
            f'ចុច "▶️ Play" ដើម្បីមើល\n'
            f'ចុច "💾 រក្សាទុក" ដើម្បីរក្សាទុក')

    def show_video_in_preview(self, video_path):
        if not video_path or not os.path.exists(video_path):
            return

        try:
            thumb_path = out_path('_preview_thumb.jpg')

            cmd = [
                FFMPEG_EXE, '-y',
                '-i', video_path,
                '-ss', '00:00:01',
                '-vframes', '1',
                thumb_path
            ]

            result = subprocess.run(cmd, capture_output=True, creationflags=CREATION_FLAGS)

            if os.path.exists(thumb_path):
                pixmap = QtGui.QPixmap(thumb_path)
                if not pixmap.isNull():
                    label_size = self.videoPreviewLabel.size()
                    if label_size.width() > 50 and label_size.height() > 50:
                        scaled = pixmap.scaled(
                            label_size,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        self.videoPreviewLabel.setPixmap(scaled)
                    else:
                        self.videoPreviewLabel.setPixmap(pixmap)
                    self.videoPreviewLabel.setAlignment(Qt.AlignCenter)
                    self.videoPreviewLabel.setObjectName('videoPreviewActive')
                    self.videoPreviewLabel.style().unpolish(self.videoPreviewLabel)
                    self.videoPreviewLabel.style().polish(self.videoPreviewLabel)

                    duration = get_media_duration(video_path)
                    file_size = os.path.getsize(video_path) / (1024 * 1024)
                    self.videoPreviewLabel.setToolTip(
                        f"📹 {os.path.basename(video_path)}\n"
                        f"⏱️ {duration/60:.2f} នាទី\n"
                        f"📊 {file_size:.1f} MB"
                    )

                    try:
                        os.remove(thumb_path)
                    except:
                        pass
                    return
        except Exception as e:
            print(f"Thumbnail error: {e}")

        duration = get_media_duration(video_path)
        file_size = os.path.getsize(video_path) / (1024 * 1024)

        self.videoPreviewLabel.setText(
            f"🎬 វីដេអូរួចរាល់!\n\n"
            f"📹 {os.path.basename(video_path)}\n"
            f"⏱️ {duration/60:.2f} នាទី | 📊 {file_size:.1f} MB"
        )
        self.videoPreviewLabel.setObjectName('videoPreviewActive')
        self.videoPreviewLabel.style().unpolish(self.videoPreviewLabel)
        self.videoPreviewLabel.style().polish(self.videoPreviewLabel)

    def on_error(self, error_msg):
        self.pushButton_6.setEnabled(True)
        self.pushButton_6.setText('🚀 ចាប់ផ្តើម')
        self.progressBar.setValue(0)
        self.log(f'❌ ERROR: {error_msg}')
        QMessageBox.critical(self, 'កំហុស', error_msg[:500])

    def new_project(self):
        self.selected_video = ''
        self.selected_srt = ''
        self.temp_output_video = ''
        self.split_video_path = ''
        self.SystemLog.clear()
        self.SplitLog.clear()
        self.progressBar.setValue(0)
        self.progressBar.setFormat("0%")
        self.labelProgress.setText('📊 ភាគរយ: 0%')
        self.labelTimeLeft.setText('⏱️ នៅសល់: --')

        self.videoPreviewLabel.clear()
        self.videoPreviewLabel.setText("🎬 មិនទាន់មានវីដេអូ\n\n(រង់ចាំបកប្រែ ឬ Upload)")
        self.videoPreviewLabel.setObjectName('videoPreview')
        self.videoPreviewLabel.style().unpolish(self.videoPreviewLabel)
        self.videoPreviewLabel.style().polish(self.videoPreviewLabel)

        self.btnPlayVideo.setEnabled(False)
        self.btnSaveVideo.setEnabled(False)
        self.log('✅ រួចរាល់សម្រាប់គម្រោងថ្មី។')

    def play_video(self):
        if not self.temp_output_video or not os.path.exists(self.temp_output_video):
            QMessageBox.warning(self, 'កំហុស', 'រកមិនឃើញវីដេអូ!')
            return

        try:
            if sys.platform == 'win32':
                os.startfile(self.temp_output_video)
            elif sys.platform == 'darwin':
                subprocess.run(['open', self.temp_output_video])
            else:
                subprocess.run(['xdg-open', self.temp_output_video])
            self.log(f'▶️ បើកវីដេអូ: {os.path.basename(self.temp_output_video)}')
        except Exception as e:
            QMessageBox.critical(self, 'កំហុស', f'មិនអាចបើកវីដេអូ:\n{e}')

    def save_video(self):
        if not self.temp_output_video or not os.path.exists(self.temp_output_video):
            QMessageBox.warning(self, 'កំហុស', 'រកមិនឃើញវីដេអូ!')
            return

        save_dir = out_path('4_final_output')
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.basename(self.temp_output_video)
        dest_path = os.path.join(save_dir, filename)

        try:
            shutil.copy2(self.temp_output_video, dest_path)
            self.log(f'💾 រក្សាទុករួចរាល់: {dest_path}')
            QMessageBox.information(self, 'ជោគជ័យ', f'រក្សាទុករួចរាល់!\n\n📁 {dest_path}')
            if sys.platform == 'win32':
                os.startfile(save_dir)
        except Exception as e:
            QMessageBox.critical(self, 'កំហុស', f'រក្សាទុកបរាជ័យ:\n{e}')

    def upload_split_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, 'ជ្រើសរើសវីដេអូដើម្បីបំបែក', '',
            'Video Files (*.mp4 *.mkv *.avi *.mov *.m4v)'
        )
        if file:
            self.split_video_path = file
            self.show_video_in_preview(file)
            self.SplitLog.append(f'📹 បានជ្រើសវីដេអូ: {os.path.basename(file)}')
            self.update_split_info2()

    def update_split_info2(self):
        try:
            minutes = int(self.inputSplitMinutes2.text())
            if minutes <= 0:
                self.labelSplitInfo2.setText("⚠️ សូមបញ្ចូលលេខត្រឹមត្រូវ")
                return

            video_to_check = None
            if self.split_video_path and os.path.exists(self.split_video_path):
                video_to_check = self.split_video_path
            elif self.temp_output_video and os.path.exists(self.temp_output_video):
                video_to_check = self.temp_output_video

            if video_to_check:
                duration = get_media_duration(video_to_check)
                if duration > 0:
                    total_min = duration / 60
                    num_parts = int(math.ceil(total_min / minutes))
                    merge_last = self.checkMergeLast.isChecked()

                    if merge_last and num_parts >= 2:
                        last_dur = total_min - (num_parts - 1) * minutes
                        if last_dur < minutes:
                            num_parts -= 1

                    self.labelSplitInfo2.setText(f"💡 {total_min:.1f} នាទី → {num_parts} ផ្នែក")
                    return

            self.labelSplitInfo2.setText(f"💡 ឧទាហរណ៍: 13 នាទី → {int(math.ceil(13/minutes))} ផ្នែក")
        except ValueError:
            self.labelSplitInfo2.setText("⚠️ សូមបញ្ចូលលេខត្រឹមត្រូវ")

    def do_split_video(self):
        if self.split_video_path and os.path.exists(self.split_video_path):
            video_to_split = self.split_video_path
        elif self.temp_output_video and os.path.exists(self.temp_output_video):
            video_to_split = self.temp_output_video
        else:
            QMessageBox.critical(self, 'កំហុស',
                'សូមជ្រើសវីដេអូជាមុនសិន!\n\n'
                'អ្នកអាច:\n'
                '1. Upload វីដេអូផ្ទាល់ខ្លួន\n'
                '2. ឬបកប្រែវីដេអូជាមុន')
            return

        try:
            minutes = int(self.inputSplitMinutes2.text())
            if minutes <= 0:
                minutes = 6
        except ValueError:
            minutes = 6

        merge_last = self.checkMergeLast.isChecked()

        self.SplitLog.clear()
        self.SplitLog.append('=' * 55)
        self.SplitLog.append('✂️ ចាប់ផ្តើមបំបែកវីដេអូ')
        self.SplitLog.append('=' * 55)
        self.SplitLog.append(f'📹 វីដេអូ: {os.path.basename(video_to_split)}')
        self.SplitLog.append(f'⏱️ រៀងរាល់: {minutes} នាទី')
        self.SplitLog.append(f'🔗 បញ្ចូលចុងក្រោយ: {"បាទ" if merge_last else "ទេ"}')
        self.SplitLog.append('')

        video_name = os.path.splitext(os.path.basename(video_to_split))[0]
        split_dir = out_path('5_split_output')
        os.makedirs(split_dir, exist_ok=True)

        video_duration = get_media_duration(video_to_split)
        if video_duration <= 0:
            self.SplitLog.append('❌ មិនអាចអានរយៈពេលវីដេអូ!')
            return

        segment_seconds = int(minutes * 60)
        total_seconds = int(video_duration)

        num_parts = int(math.ceil(total_seconds / segment_seconds))

        if merge_last and num_parts >= 2:
            last_part_duration = total_seconds - (num_parts - 1) * segment_seconds
            if last_part_duration < segment_seconds:
                num_parts -= 1
                self.SplitLog.append(f'📊 ផ្នែកចុងក្រោយតូច → បញ្ចូលជាមួយផ្នែកមុន')

        self.SplitLog.append(f'📊 ចំនួនផ្នែក: {num_parts}')
        self.SplitLog.append('')

        for f in glob.glob(os.path.join(split_dir, f'{video_name}_part*.mp4')):
            try:
                os.remove(f)
            except:
                pass

        output_files = []
        for i in range(num_parts):
            start_time = i * segment_seconds

            if i == num_parts - 1:
                duration = total_seconds - start_time
            else:
                duration = segment_seconds

            output_file = os.path.join(split_dir, f'{video_name}_part{i+1:03d}.mp4')

            cmd = [
                FFMPEG_EXE, '-y',
                '-ss', str(start_time),
                '-i', video_to_split,
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                output_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=CREATION_FLAGS)

            if result.returncode == 0 and os.path.exists(output_file):
                output_files.append(output_file)
                dur_min = duration / 60
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                self.SplitLog.append(f'✅ ផ្នែកទី {i+1}: {dur_min:.1f} នាទី ({size_mb:.1f} MB)')
            else:
                self.SplitLog.append(f'❌ បំបែកផ្នែកទី {i+1} បរាជ័យ')

        self.SplitLog.append('')
        self.SplitLog.append('=' * 55)
        self.SplitLog.append(f'🎉 បញ្ចប់! បានបង្កើត {len(output_files)} ផ្នែក')
        self.SplitLog.append(f'📁 ទីតាំង: {split_dir}')
        self.SplitLog.append('=' * 55)

        if output_files and sys.platform == 'win32':
            os.startfile(split_dir)

        QMessageBox.information(self, 'ជោគជ័យ', f'បំបែករួចរាល់!\n\nចំនួនផ្នែក: {len(output_files)}\n\n📁 {split_dir}')


# ==========================================
# Run App
# ==========================================
if __name__ == '__main__':
    try:
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass

        app = QApplication(sys.argv)
        window = MainApp()
        window.show()
        sys.exit(app.exec_())
    except SystemExit:
        raise
    except Exception:
        import traceback as _tb
        _err = _tb.format_exc()
        print('\n[main.py] STARTUP FAILED:\n' + _err)
        try:
            with open(os.path.join(BASE_DIR, 'startup_error.txt'), 'w', encoding='utf-8') as f:
                f.write(_err)
        except:
            pass
        sys.exit(1)
