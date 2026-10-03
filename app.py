import os
import sys
import re
import math
import glob
import time
import shutil
import asyncio
import subprocess
import tempfile
import threading
import concurrent.futures

os.environ["PYTHONIOENCODING"] = "utf-8"

import requests
import gradio as gr
import edge_tts

try:
    from groq import Groq
except Exception:
    Groq = None

# ============================================================
# API Keys
# - ផ្នែក "បកប្រែវីដេអូ (SRT)" មិនត្រូវការ API Key ទេ
# - ផ្នែក Agnes ត្រូវការ AGNES_API_KEY (និង GROQ_API_KEY សម្រាប់បង្កើត prompt)
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "").strip()

groq_client = Groq(api_key=GROQ_API_KEY) if (Groq and GROQ_API_KEY) else None

AGNES_BASE_URL = "https://apihub.agnes-ai.com"
AGNES_VIDEO_MODEL = "agnes-video-v2.0"

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# ============================================================
# Config
# ============================================================
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


# ============================================================
# Helper Functions
# ============================================================
def get_media_duration(media_path):
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
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


def _src_code(source_lang):
    if source_lang == 'auto':
        return 'auto'
    if source_lang == 'zh':
        return 'zh-CN'
    return source_lang


# ============================================================
# Translation (ដូច Desktop App)
# ============================================================
def translate_single(text, source_lang="auto", target_lang="km"):
    if not text or not text.strip():
        return text

    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=_src_code(source_lang), target=target_lang)
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


def translate_batch_fast(texts, source_lang="auto", target_lang="km"):
    if not texts:
        return texts
    if len(texts) == 1:
        return [translate_single(texts[0], source_lang, target_lang)]

    separator = "\n@@@\n"
    combined = separator.join(texts)

    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=_src_code(source_lang), target=target_lang)
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


# ============================================================
# Speaker / Gender Detection (ដូច Desktop App)
# ============================================================
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


def load_full_audio_array(video_path):
    """អានសំឡេងទាំងមូលម្តងគត់ (16kHz mono) ដើម្បីកាត់ប្រើសម្រាប់រកយេនឌ័រ"""
    try:
        import numpy as np
        cmd = [
            FFMPEG, "-y", "-i", video_path,
            "-vn", "-ar", "16000", "-ac", "1",
            "-f", "s16le", "-"
        ]
        result = subprocess.run(cmd, capture_output=True)
        return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        print(f"Load audio error: {e}")
        return None


def detect_gender_from_audio(full_audio, start_time, end_time, sr=16000):
    try:
        if full_audio is None:
            return None
        if end_time - start_time <= 0.1:
            return None

        a = int(start_time * sr)
        b = int(end_time * sr)
        audio = full_audio[a:b]
        if audio.size < 1600:
            return None

        f0 = estimate_pitch_yin(audio, sr)
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


# ============================================================
# TAB 1: បកប្រែវីដេអូ ជាមួយ SRT (បេះបិទ Desktop App)
# ============================================================
def dub_with_srt(video_path, srt_path, voice_var, target_lang_name):
    """
    Generator ដែល yield ជាបន្តបន្ទាប់ដើម្បីបង្ហាញ Log / Progress ផ្ទាល់
    Outputs:
      log, label_progress, label_time, progress_bar, status_label,
      video_output, download_files, last_video_state
    """
    st = {
        "log": [], "pct": 0, "status": "រួចរាល់ដើម្បីចាប់ផ្តើម",
        "time": "⏱️ នៅសល់: --", "video": None, "files": None, "last": None,
    }

    def log(t):
        st["log"].append(t)

    def setp(p, s, t="⏱️ នៅសល់: --"):
        st["pct"] = int(p)
        st["status"] = s
        st["time"] = t

    def ui():
        return (
            "\n".join(st["log"]),
            f"📊 ភាគរយ: {st['pct']}%",
            st["time"],
            st["pct"],
            st["status"],
            st["video"],
            st["files"],
            st["last"],
        )

    # ---------- Validate ----------
    if not video_path:
        log("❌ សូមជ្រើសរើសវីដេអូជាមុនសិន!")
        yield ui()
        return
    if not srt_path:
        log("❌ សូមជ្រើសរើសឯកសារ SRT!")
        yield ui()
        return

    if target_lang_name not in TARGET_LANG_CONFIG:
        target_lang_name = DEFAULT_TARGET_LANG
    cfg = TARGET_LANG_CONFIG[target_lang_name]
    target_code = cfg['code']

    work = tempfile.mkdtemp(prefix="dub_")
    seg_dir = os.path.join(work, "3_audio_segments")
    os.makedirs(seg_dir, exist_ok=True)

    try:
        import srt

        log(f"📹 វីដេអូ: {os.path.basename(video_path)}")
        log(f"📄 SRT: {os.path.basename(srt_path)}")
        log(f"🎙️ សំលេង: {voice_var}")
        log(f"🌐 ភាសាគោលដៅ: {target_lang_name}")
        log("")
        log(f"🚀 ចាប់ផ្តើម — ភាសាគោលដៅ: {target_lang_name}")
        setp(2, "កំពុងផ្ទុក...")
        yield ui()

        filename = os.path.basename(video_path)
        base = os.path.splitext(filename)[0]
        final_out = os.path.join(work, f"{target_lang_name}_{filename}")
        if not final_out.lower().endswith(".mp4"):
            final_out = os.path.splitext(final_out)[0] + ".mp4"

        # ========== STEP 1: អាន SRT ==========
        setp(5, "Step 1: កំពុងអាន SRT...")
        log(f"\n[STEP 1] កំពុងអាន SRT: {os.path.basename(srt_path)}")
        yield ui()

        try:
            with open(srt_path, 'r', encoding='utf-8-sig') as f:
                raw = f.read()
        except UnicodeDecodeError:
            with open(srt_path, 'r', encoding='cp1252') as f:
                raw = f.read()

        subs = list(srt.parse(raw))
        for i, s in enumerate(subs, start=1):
            s.index = i

        total_subs = len(subs)
        if total_subs == 0:
            raise RuntimeError("ឯកសារ SRT ទទេ ឬមិនត្រឹមត្រូវ!")
        log(f"   ✅ អានបាន {total_subs} បន្ទាត់")

        sample = ' '.join(s.content for s in subs[:20])
        detected_lang = detect_text_language(sample)
        log(f"   ភាសា SRT: {detected_lang}")
        yield ui()

        # ========== STEP 2: បកប្រែលឿន ==========
        log(f"\n[STEP 2] កំពុងបកប្រែទៅ {target_lang_name}...")
        yield ui()

        all_texts, text_indices = [], []
        for i, sub in enumerate(subs):
            txt = sub.content.strip()
            if txt:
                all_texts.append(txt)
                text_indices.append(i)

        if all_texts:
            batch_size = 50
            batches = [all_texts[i:i + batch_size] for i in range(0, len(all_texts), batch_size)]
            num_batches = len(batches)
            log(f"   📦 ចំនួន Batch: {num_batches} (Batch Size: {batch_size})")
            log("   ⚡ ប្រើ 5 Threads ក្នុងពេលតែមួយ...")
            yield ui()

            step2_start = time.time()

            def process_batch(b_idx, batch):
                try:
                    result = translate_batch_fast(batch, detected_lang, target_code)
                    if not result or len(result) != len(batch):
                        result = [translate_single(t, detected_lang, target_code) for t in batch]
                except Exception:
                    result = [translate_single(t, detected_lang, target_code) for t in batch]
                return b_idx, result

            results_dict = {}
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_batch, i, b) for i, b in enumerate(batches)]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        b_idx, result = fut.result()
                        results_dict[b_idx] = result
                    except Exception as e:
                        log(f"   ⚠️ Thread error: {e}")
                        b_idx = None
                    completed += 1

                    elapsed = time.time() - step2_start
                    ratio = completed / num_batches
                    if ratio > 0.05 and elapsed > 2:
                        remaining = elapsed / ratio - elapsed
                        if remaining > 60:
                            time_text = f"⏱️ នៅសល់: {int(remaining // 60)} នាទី {int(remaining % 60)} វិនាទី"
                        elif remaining > 0:
                            time_text = f"⏱️ នៅសល់: {int(remaining)} វិនាទី"
                        else:
                            time_text = "⏱️ ជិតរួចហើយ!"
                    else:
                        time_text = "⏱️ កំពុងគណនា..."

                    log(f"   [{completed}/{num_batches}] ✅ Batch")
                    setp(15 + int(ratio * 40),
                         f"Step 2: បកប្រែ {completed}/{num_batches} Batch", time_text)
                    yield ui()

            all_translated = []
            for i in range(num_batches):
                if i in results_dict:
                    all_translated.extend(results_dict[i])
                else:
                    all_translated.extend(batches[i])

            log(f"   ✅ បកប្រែរួចរាល់ក្នុង {time.time() - step2_start:.1f} វិនាទី")

            for idx, trans in zip(text_indices, all_translated):
                subs[idx].content = collapse_repeats(trans)

        srt_out_path = os.path.join(work, f"{base}_{target_code}.srt")
        with open(srt_out_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subs))
        log(f"   ✅ បកប្រែរួចរាល់: {os.path.basename(srt_out_path)}")
        st["files"] = [srt_out_path]
        yield ui()

        # ========== STEP 3a: រកយេនឌ័រ ==========
        genders = {}
        if voice_var == 'auto':
            setp(55, "Step 3a: កំពុងរកតួអង្គ...")
            log("\n[STEP 3a] កំពុងរកយេនឌ័រតួអង្គ...")
            yield ui()

            full_audio = load_full_audio_array(video_path)

            for sub in subs:
                g = detect_gender_simple(sub.content)
                if not g:
                    g = detect_gender_from_audio(
                        full_audio,
                        sub.start.total_seconds(),
                        sub.end.total_seconds()
                    )
                genders[sub.index] = g or 'female'

            n_male = sum(1 for g in genders.values() if g == 'male')
            n_female = len(genders) - n_male
            log(f"   📊 ប្រុស: {n_male}, ស្រី: {n_female}")
            yield ui()

        # ========== STEP 3b: បង្កើតសំលេង ==========
        setp(65, "Step 3b: កំពុងបង្កើតសំលេង...")
        log("\n[STEP 3b] កំពុងបង្កើតសំលេង AI...")
        yield ui()

        female_voice = cfg['female']
        male_voice = cfg['male']

        async def make_voice_all():
            sem = asyncio.Semaphore(15)

            async def make_voice(sub):
                txt = sub.content.strip()
                if not txt:
                    return
                ms = int(sub.start.total_seconds() * 1000)
                out_mp3 = os.path.join(seg_dir, f'{sub.index:04d}_{ms}.mp3')

                if voice_var == 'female':
                    voice_id = female_voice
                elif voice_var == 'male':
                    voice_id = male_voice
                else:
                    g = genders.get(sub.index, 'female')
                    voice_id = male_voice if g == 'male' else female_voice

                async with sem:
                    for _ in range(3):
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

        log("   ✅ បង្កើតសំលេងរួចរាល់")
        yield ui()

        # ========== STEP 4: Merge ==========
        setp(85, "Step 4: កំពុង Merge...", "⏱️ នៅសល់: 1-2 នាទី")
        log("\n[STEP 4] កំពុង Merge វីដេអូ + សំលេង...")
        yield ui()

        raw_segments = sorted(glob.glob(os.path.join(seg_dir, '[0-9]*.mp3')))
        segments = [s for s in raw_segments if os.path.getsize(s) >= 200]
        if not segments:
            raise RuntimeError("គ្មានឯកសារសំលេង!")

        video_dur = get_media_duration(video_path)
        if video_dur <= 0:
            video_dur = 3600

        silent_wav = os.path.join(work, "_silent_base.wav")
        subprocess.run([
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(video_dur), "-ar", "44100",
            silent_wav
        ], capture_output=True)

        current_mix = silent_wav
        mix_batch = 20
        batch_idx = 0
        total_mix = math.ceil(len(segments) / mix_batch)

        for start_idx in range(0, len(segments), mix_batch):
            batch = segments[start_idx:start_idx + mix_batch]
            n = len(batch)
            batch_out = os.path.join(work, f"_mix_batch_{batch_idx}.wav")

            filter_parts = []
            for i, seg in enumerate(batch):
                m = re.search(r'_(\d+)\.mp3$', seg)
                ms_v = m.group(1) if m else '0'
                filter_parts.append(f'[{i + 1}:a]adelay={ms_v}|{ms_v}[a{i}]')

            mix_labels = ''.join([f'[a{i}]' for i in range(n)])
            fc = ';\n'.join(filter_parts)
            fc += f';\n{mix_labels}amix=inputs={n}:duration=longest:normalize=0[seg];'
            fc += '[0:a][seg]amix=inputs=2:duration=longest:normalize=0[out]'

            cmd = [FFMPEG, "-y", "-i", current_mix]
            for seg in batch:
                cmd.extend(["-i", seg])
            cmd.extend(["-filter_complex", fc, "-map", "[out]", "-ar", "44100", batch_out])
            subprocess.run(cmd, capture_output=True)

            if current_mix != silent_wav and os.path.exists(current_mix):
                try:
                    os.remove(current_mix)
                except Exception:
                    pass
            if os.path.exists(batch_out):
                current_mix = batch_out
            batch_idx += 1

            setp(85 + int(10 * batch_idx / total_mix),
                 f"Step 4: Mix សំឡេង {batch_idx}/{total_mix}", "⏱️ នៅសល់: --")
            yield ui()

        dubbed_audio = current_mix

        cmd_final = [
            FFMPEG, "-y",
            "-i", video_path,
            "-i", dubbed_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", final_out
        ]
        res = subprocess.run(cmd_final, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(final_out):
            raise RuntimeError(f"ការផ្គុំវីដេអូបរាជ័យ: {res.stderr[-300:]}")

        # សម្អាត temp
        for tmp in glob.glob(os.path.join(work, "_mix_batch_*.wav")):
            try:
                os.remove(tmp)
            except Exception:
                pass
        if os.path.exists(silent_wav):
            os.remove(silent_wav)
        shutil.rmtree(seg_dir, ignore_errors=True)

        setp(100, "រួចរាល់! 100% | ✅ បានបញ្ចប់", "✅ បានបញ្ចប់")
        log(f"\n✅ ជោគជ័យ! វីដេអូ: {os.path.basename(final_out)}")
        log("ចុច Download ដើម្បីរក្សាទុក ឬបំបែកវីដេអូនៅផ្នែកទី ២")
        st["video"] = final_out
        st["files"] = [final_out, srt_out_path]
        st["last"] = final_out
        yield ui()

    except Exception as e:
        import traceback
        log(f"❌ ERROR: {e}")
        log(traceback.format_exc())
        st["status"] = "❌ មានបញ្ហា"
        yield ui()


# ============================================================
# ផ្នែកទី ២: បំបែកវីដេអូ (បេះបិទ Desktop App)
# ============================================================
def _pick_split_source(uploaded_video, last_output):
    if uploaded_video and os.path.exists(uploaded_video):
        return uploaded_video
    if last_output and os.path.exists(last_output):
        return last_output
    return None


def update_split_info(uploaded_video, last_output, minutes, merge_last):
    try:
        minutes = int(minutes)
        if minutes <= 0:
            return "⚠️ សូមបញ្ចូលលេខត្រឹមត្រូវ"
    except Exception:
        return "⚠️ សូមបញ្ចូលលេខត្រឹមត្រូវ"

    src = _pick_split_source(uploaded_video, last_output)
    if src:
        duration = get_media_duration(src)
        if duration > 0:
            total_min = duration / 60
            num_parts = int(math.ceil(total_min / minutes))
            if merge_last and num_parts >= 2:
                last_dur = total_min - (num_parts - 1) * minutes
                if last_dur < minutes:
                    num_parts -= 1
            return f"💡 {total_min:.1f} នាទី → {num_parts} ផ្នែក"

    return f"💡 ឧទាហរណ៍: 13 នាទី → {int(math.ceil(13 / minutes))} ផ្នែក"


def do_split_video(uploaded_video, last_output, minutes, merge_last):
    lines = []

    def L(t):
        lines.append(t)

    video_to_split = _pick_split_source(uploaded_video, last_output)
    if not video_to_split:
        L("❌ សូមជ្រើសវីដេអូជាមុនសិន!")
        L("អ្នកអាច:")
        L("1. Upload វីដេអូផ្ទាល់ខ្លួន")
        L("2. ឬបកប្រែវីដេអូជាមុន")
        return "\n".join(lines), None

    try:
        minutes = int(minutes)
        if minutes <= 0:
            minutes = 6
    except Exception:
        minutes = 6

    L("=" * 55)
    L("✂️ ចាប់ផ្តើមបំបែកវីដេអូ")
    L("=" * 55)
    L(f"📹 វីដេអូ: {os.path.basename(video_to_split)}")
    L(f"⏱️ រៀងរាល់: {minutes} នាទី")
    L(f"🔗 បញ្ចូលចុងក្រោយ: {'បាទ' if merge_last else 'ទេ'}")
    L("")

    video_name = os.path.splitext(os.path.basename(video_to_split))[0]
    split_dir = tempfile.mkdtemp(prefix="split_")

    video_duration = get_media_duration(video_to_split)
    if video_duration <= 0:
        L("❌ មិនអាចអានរយៈពេលវីដេអូ!")
        return "\n".join(lines), None

    segment_seconds = int(minutes * 60)
    total_seconds = int(video_duration)
    num_parts = int(math.ceil(total_seconds / segment_seconds))

    if merge_last and num_parts >= 2:
        last_part_duration = total_seconds - (num_parts - 1) * segment_seconds
        if last_part_duration < segment_seconds:
            num_parts -= 1
            L("📊 ផ្នែកចុងក្រោយតូច → បញ្ចូលជាមួយផ្នែកមុន")

    L(f"📊 ចំនួនផ្នែក: {num_parts}")
    L("")

    output_files = []
    for i in range(num_parts):
        start_time = i * segment_seconds
        if i == num_parts - 1:
            duration = total_seconds - start_time
        else:
            duration = segment_seconds

        output_file = os.path.join(split_dir, f"{video_name}_part{i + 1:03d}.mp4")
        cmd = [
            FFMPEG, "-y",
            "-ss", str(start_time),
            "-i", video_to_split,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_file):
            output_files.append(output_file)
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            L(f"✅ ផ្នែកទី {i + 1}: {duration / 60:.1f} នាទី ({size_mb:.1f} MB)")
        else:
            L(f"❌ បំបែកផ្នែកទី {i + 1} បរាជ័យ")

    L("")
    L("=" * 55)
    L(f"🎉 បញ្ចប់! បានបង្កើត {len(output_files)} ផ្នែក")
    L("=" * 55)

    return "\n".join(lines), (output_files or None)


# ============================================================
# Agnes AI Video Generation (មិនប្តូរ)
# ============================================================
def get_active_chat_model():
    if groq_client is None:
        return "llama-3.3-70b-versatile"
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


async def generate_speech_segment(text, voice, output_path):
    tts = edge_tts.Communicate(text, voice)
    await tts.save(output_path)


def agnes_generate_visual_prompt(scene, active_model):
    khmer_text = scene["khmer_text"]
    speaker = scene["speaker"]
    name = scene.get("name", "")

    system_instruction = (
        "You are an expert cinematic director and AI prompt engineer for text-to-video models. "
        "Convert the dialogue or narration into a detailed visual scene description. "
        "Focus heavily on realistic motions: character gestures, natural mouth moving as speaking, "
        "expressive eyes, subtle breathing, dynamic camera motions (tracking shot, pan, zoom), "
        "and lively background elements. Keep it under 45 words. Output ONLY the English prompt."
    )
    user_content = (
        f"Speaker: {name or speaker} ({speaker})\n"
        f"Line context: \"{khmer_text}\"\n"
        f"Setting: Cambodian countryside or everyday cultural setting."
    )
    try:
        if groq_client is None:
            raise RuntimeError("no groq")
        res = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            model=active_model,
            temperature=0.4
        )
        visual_desc = res.choices[0].message.content.strip()
    except Exception:
        visual_desc = f"A realistic {speaker} person speaking naturally with expressive gestures in the Cambodian countryside"

    full_prompt = (
        f"{visual_desc}, cinematic natural motion, dynamic movement, 4k photorealistic, "
        f"natural lighting, highly detailed face, realistic skin texture, 24fps smooth movement"
    )
    return full_prompt


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
    headers = {"Authorization": f"Bearer {AGNES_API_KEY}"}
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
        print(f"Agnes video {video_id}: {status} ({data.get('progress', 0)}%)")
        time.sleep(5)
    raise TimeoutError(f"Agnes វីដេអូហួសពេល: {video_id}")


def agnes_generate_video_for_scene(scene, active_model, session_id, temp_dir):
    full_prompt = agnes_generate_visual_prompt(scene, active_model)
    print(f"🎬 Prompt: {full_prompt}")
    num_frames = 121
    frame_rate = 24

    try:
        result = agnes_create_video(
            prompt=full_prompt,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=session_id + hash(scene["khmer_text"]) % 10000
        )
        video_id = result.get("video_id") or result.get("id")
        if not video_id:
            return None

        final = agnes_poll_video(video_id)
        video_url = final.get("video_url") or final.get("url") or final.get("remixed_from_video_id")
        if not video_url:
            return None

        video_file = os.path.join(temp_dir, f"agnes_raw_{session_id}_{int(time.time())}.mp4")
        r = requests.get(video_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(video_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        if not os.path.exists(video_file) or os.path.getsize(video_file) < 1000:
            return None

        khmer_text = scene["khmer_text"]
        speaker = scene["speaker"]
        if speaker == "male":
            voice = "km-KH-PisethNeural"
        elif speaker == "female":
            voice = "km-KH-SreymomNeural"
        else:
            voice = "km-KH-PisethNeural"

        audio_file = os.path.join(temp_dir, f"khmer_audio_{session_id}_{int(time.time())}.mp3")
        asyncio.run(generate_speech_segment(khmer_text, voice, audio_file))

        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 500:
            return video_file

        audio_duration = get_media_duration(audio_file)
        output_file = os.path.join(temp_dir, f"agnes_khmer_{session_id}_{int(time.time())}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", video_file,
            "-i", audio_file,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(max(audio_duration, 1.0)),
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_file):
            for f in [video_file, audio_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            return output_file
        else:
            return video_file
    except Exception as e:
        print(f"Agnes video generation failed: {e}")
    return None


def create_video_with_agnes(script_text, narration_gender, resolution,
                            progress=gr.Progress()):
    if not script_text or len(script_text.strip()) < 10:
        return None, "សូមសរសេរ Script ជាភាសាខ្មែរជាមុនសិន!"
    if not AGNES_API_KEY:
        return None, "សូមកំណត់ AGNES_API_KEY ជា environment variable"

    temp_dir = tempfile.gettempdir()
    session_id = int(time.time())

    try:
        progress(0.05, desc="កំពុងវិភាគ Script...")
        active_model = get_active_chat_model()

        raw_blocks = re.split(r'(?=\[[^\]]+\])', script_text.strip())
        scenes = []
        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue
            named_match = re.match(r'^\[([^\]|]+)\|(ប្រុស|ស្រី|male|female)\]\s*[:：]?\s*(.+)$', block, re.DOTALL | re.IGNORECASE)
            gender_match = re.match(r'^\[(ប្រុស|ស្រី|male|female)\]\s*[:：]?\s*(.+)$', block, re.DOTALL | re.IGNORECASE)
            narration_match = re.match(r'^\[(និទាន|narration|ទេសភាព|scene)\]\s*[:：]?\s*(.+)$', block, re.DOTALL | re.IGNORECASE)

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
                scenes.append({"speaker": "narration", "name": "និទាន", "khmer_text": block})

        if not scenes:
            return None, "មិនអាចវិភាគ Script បានទេ!"

        num_scenes = len(scenes)
        progress(0.1, desc=f"កំពុងបង្កើតវីដេអូ {num_scenes} scenes...")

        width, height = resolution
        video_files = []
        for i, scene in enumerate(scenes):
            progress(
                0.1 + 0.7 * ((i + 1) / num_scenes),
                desc=f"កំពុងបង្កើតវីដេអូ {i+1}/{num_scenes} (អាចយឺត ១-៣ នាទី)..."
            )
            video_file = agnes_generate_video_for_scene(
                scene, active_model, session_id + i, temp_dir
            )
            if video_file:
                scene["video_file"] = video_file
                video_files.append(video_file)

        if not video_files:
            return None, "មិនអាចបង្កើតវីដេអូជាមួយ Agnes បានទេ!"

        progress(0.85, desc="កំពុងផ្គុំវីដេអូ...")
        output_video = os.path.join(temp_dir, f"agnes_final_{session_id}.mp4")
        concat_file = os.path.join(temp_dir, f"concat_agnes_{session_id}.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for vf in video_files:
                f.write(f"file '{vf}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", concat_file, "-c", "copy", output_video]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                   "-i", concat_file, "-c:v", "libx264", "-preset", "ultrafast",
                   "-c:a", "aac", "-pix_fmt", "yuv420p", output_video]
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 or not os.path.exists(output_video):
            return None, f"ការផ្គុំវីដេអូបរាជ័យ: {result.stderr[:300]}"

        for f in video_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        male_count = sum(1 for s in scenes if s.get("speaker") == "male")
        female_count = sum(1 for s in scenes if s.get("speaker") == "female")
        narration_count = sum(1 for s in scenes if s.get("speaker") == "narration")

        status = (
            f" ជោគជ័យ! បង្កើតវីដេអូពិតដោយ Agnes AI\n\n"
            f" ចំនួន scenes: {num_scenes}\n"
            f" វីដេអូជោគជ័យ: {len(video_files)}\n"
            f" តួអង្គប្រុស: {male_count}\n"
            f" តួអង្គស្រី: {female_count}\n"
            f" ការនិទាន: {narration_count}\n"
            f" ទំហំ: {width}x{height}\n\n"
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

html, body {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
}

gradio-app, #root {
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gradio-container {
    font-family: 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
    background: #F8FAFC !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    border-radius: 0 !important;
}

.gradio-container > .main,
.gradio-container .main,
.gradio-container .contain {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gradio-container .tabs,
.gradio-container .tab-nav,
.gradio-container .tabitem {
    width: 100% !important;
    max-width: 100% !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .row,
.gradio-container .column {
    max-width: 100% !important;
    width: 100% !important;
}

.gradio-container .gr-group,
.gradio-container .gap,
.gradio-container .wrap {
    padding-left: 8px !important;
    padding-right: 8px !important;
}

footer { display: none !important; }

.main-title { text-align: center; padding: 15px 0 10px 0; width: 100% !important; }
.main-title h1 { font-size: 1.8em; font-weight: 700; color: #1E293B; margin-bottom: 5px; }

.section-header-blue {
    font-family: 'Kantumruy Pro', sans-serif; font-weight: 700; color: #2563EB;
    font-size: 1.05em; text-align: center; padding: 8px;
    border-bottom: 2px solid #2563EB; margin-bottom: 10px;
}
.section-header-orange {
    font-family: 'Kantumruy Pro', sans-serif; font-weight: 700; color: #F97316;
    font-size: 1.05em; text-align: center; padding: 8px;
    border-bottom: 2px solid #F97316; margin-bottom: 10px;
}

button.primary-btn {
    background: linear-gradient(90deg, #16A34A 0%, #15803D 100%) !important;
    border: none !important; color: white !important;
    font-family: 'Kantumruy Pro', sans-serif !important;
    font-weight: 700 !important; font-size: 1.05em !important;
    padding: 12px !important; border-radius: 8px !important;
}
button.primary-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(22,163,74,0.4) !important; }

button.blue-btn { background: #2563EB !important; color: white !important; font-family: 'Kantumruy Pro', sans-serif !important; font-weight: 600 !important; border-radius: 6px !important; border: 2px solid #1D4ED8 !important; }
button.voice-off { background: #F1F5F9 !important; color: #334155 !important; font-family: 'Kantumruy Pro', sans-serif !important; font-weight: 600 !important; border-radius: 6px !important; border: 2px solid #CBD5E1 !important; }
button.orange-btn { background: #F97316 !important; color: white !important; font-family: 'Kantumruy Pro', sans-serif !important; font-weight: 600 !important; border-radius: 6px !important; }
button.red-btn { background: #E11D48 !important; color: white !important; font-family: 'Kantumruy Pro', sans-serif !important; font-weight: 600 !important; border-radius: 6px !important; }
button.purple-btn { background: #8B5CF6 !important; color: white !important; font-family: 'Kantumruy Pro', sans-serif !important; font-weight: 600 !important; border-radius: 6px !important; }

.gradio-container label,
.gradio-container .label-wrap {
    font-family: 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
    font-weight: 600 !important; color: #334155 !important; font-size: 0.95em !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    font-family: 'Kantumruy Pro', 'Noto Sans Khmer', sans-serif !important;
    font-size: 0.95em !important; border-radius: 8px !important;
    border: 2px solid #CBD5E1 !important;
}
.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
"""

VOICE_LABELS = {'female': 'ស្រី', 'male': 'ប្រុស', 'auto': 'ស្វ័យប្រវត្តិ'}


def voice_button_updates(active):
    out = []
    for key in ('female', 'male', 'auto'):
        cls = ["blue-btn"] if key == active else ["voice-off"]
        out.append(gr.update(elem_classes=cls))
    return out


# ============================================================
# Interface
# ============================================================
with gr.Blocks(title="AI Video Translator - SRT Mode", css=CUSTOM_CSS, fill_width=True) as demo:

    gr.HTML("""
        <div class="main-title">
            <h1>🎬 AI Video Translator - SRT Mode</h1>
        </div>
    """)

    with gr.Tabs():

        # ============================================================
        # TAB 1: បកប្រែវីដេអូ (SRT)
        # ============================================================
        with gr.TabItem("🎥 បកប្រែវីដេអូ"):

            last_video_state = gr.State(None)
            voice_state = gr.State("auto")

            # ========== ផ្នែកទី ១ ==========
            gr.HTML('<div class="section-header-blue">📝 ផ្នែកទី ១: បកប្រែវីដេអូ</div>')

            with gr.Row():
                # ---- Left: Log + Progress ----
                with gr.Column(scale=1):
                    system_log = gr.Textbox(
                        label="📋 System Log",
                        value="🎙️ សំលេង: ស្វ័យប្រវត្តិ",
                        lines=14,
                        max_lines=14,
                        interactive=False
                    )

                    with gr.Row():
                        label_progress = gr.Textbox(
                            value="📊 ភាគរយ: 0%", label="", show_label=False,
                            interactive=False, max_lines=1
                        )
                        label_time = gr.Textbox(
                            value="⏱️ នៅសល់: --", label="", show_label=False,
                            interactive=False, max_lines=1
                        )

                    progress_bar = gr.Slider(
                        minimum=0, maximum=100, value=0,
                        label="វឌ្ឍនភាព", interactive=False
                    )

                    status_label = gr.Textbox(
                        value="រួចរាល់ដើម្បីចាប់ផ្តើម", label="", show_label=False,
                        interactive=False, max_lines=1
                    )

                # ---- Right: Controls ----
                with gr.Column(scale=1):
                    video_input = gr.Video(label="📹 ជ្រើសរើសវីដេអូ")

                    srt_input = gr.File(
                        label="📄 ជ្រើសរើស SRT",
                        file_types=[".srt"],
                        type="filepath"
                    )

                    with gr.Row():
                        btn_voice_female = gr.Button("🎙️ ស្រី", elem_classes="voice-off")
                        btn_voice_male = gr.Button("🎙️ ប្រុស", elem_classes="voice-off")
                        btn_voice_auto = gr.Button("🎙️ ស្វ័យប្រវត្តិ", elem_classes="blue-btn")

                    target_lang = gr.Dropdown(
                        choices=list(TARGET_LANG_CONFIG.keys()),
                        value=DEFAULT_TARGET_LANG,
                        label="🌐 ភាសាគោលដៅ"
                    )

                    submit_btn = gr.Button(
                        "🚀 ចាប់ផ្តើម",
                        variant="primary",
                        elem_classes="primary-btn",
                        size="lg"
                    )

                    new_project_btn = gr.Button("🆕 គម្រោងថ្មី", elem_classes="red-btn")

                    gr.HTML("""
                        <a href="https://www.voicecheap.ai/tools/video-to-srt" target="_blank"
                           style="text-decoration:none;">
                        <div style="text-align:center; padding:8px; background:#EFF6FF;
                                    border:2px solid #BFDBFE; border-radius:6px;
                                    color:#2563EB; font-weight:600; font-size:0.9em;">
                            🔗 ដើម្បីទាញ SRT សូមចុចតំណរនេះ (voicecheap.ai)
                        </div></a>
                    """)

            # ---- Output ----
            with gr.Row():
                video_output = gr.Video(label="🎥 វីដេអូដែលបានបកប្រែ", height=320)
                download_files = gr.File(
                    label="💾 ទាញយក (វីដេអូ + SRT ដែលបកប្រែ)",
                    file_count="multiple",
                    interactive=False
                )

            # ========== ផ្នែកទី ២: បំបែកវីដេអូ ==========
            gr.HTML('<div class="section-header-orange" style="margin-top:20px;">✂️ ផ្នែកទី ២: បំបែកវីដេអូ</div>')

            with gr.Row():
                with gr.Column(scale=1):
                    split_video_input = gr.Video(label="📤 Upload វីដេអូផ្ទាល់ខ្លួន (ទុកទទេ = ប្រើវីដេអូដែលទើបបកប្រែ)")

                with gr.Column(scale=1):
                    split_log = gr.Textbox(label="📋 Split Log", lines=10, interactive=False)

                    check_merge_last = gr.Checkbox(
                        label="🔗 ចុងក្រោយបញ្ចូលជាមួយផ្នែកចុងក្រោយ",
                        value=True
                    )

                    with gr.Row():
                        split_minutes = gr.Number(label="រៀងរាល់ (នាទី)", value=6, precision=0, minimum=1)
                        split_info = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1,
                                                value="💡 ឧទាហរណ៍: 13 នាទី → 3 ផ្នែក")

                    btn_split = gr.Button("✂️ បំបែកវីដេអូឥឡូវ", elem_classes="orange-btn", size="lg")

                    split_outputs = gr.File(
                        label="📁 ផ្នែកវីដេអូដែលបានបំបែក",
                        file_count="multiple",
                        interactive=False
                    )

            # ---- Events: ជ្រើសសំឡេង ----
            def set_voice(kind):
                def _fn():
                    return [kind, f"🎙️ សំលេង: {VOICE_LABELS[kind]}"] + voice_button_updates(kind)
                return _fn

            voice_outputs = [voice_state, system_log, btn_voice_female, btn_voice_male, btn_voice_auto]
            btn_voice_female.click(fn=set_voice('female'), outputs=voice_outputs)
            btn_voice_male.click(fn=set_voice('male'), outputs=voice_outputs)
            btn_voice_auto.click(fn=set_voice('auto'), outputs=voice_outputs)

            # ---- Events: ចាប់ផ្តើមបកប្រែ ----
            submit_btn.click(
                fn=dub_with_srt,
                inputs=[video_input, srt_input, voice_state, target_lang],
                outputs=[system_log, label_progress, label_time, progress_bar,
                         status_label, video_output, download_files, last_video_state]
            )

            # ---- Events: គម្រោងថ្មី ----
            def new_project():
                return [
                    None, None,                       # video_input, srt_input
                    "✅ រួចរាល់សម្រាប់គម្រោងថ្មី។",   # system_log
                    "📊 ភាគរយ: 0%", "⏱️ នៅសល់: --", 0,
                    "រួចរាល់ដើម្បីចាប់ផ្តើម",
                    None, None, None,                 # video_output, download_files, last_video_state
                    None, "", None,                   # split_video_input, split_log, split_outputs
                    "auto",
                ] + voice_button_updates('auto')

            new_project_btn.click(
                fn=new_project,
                outputs=[video_input, srt_input, system_log,
                         label_progress, label_time, progress_bar, status_label,
                         video_output, download_files, last_video_state,
                         split_video_input, split_log, split_outputs,
                         voice_state,
                         btn_voice_female, btn_voice_male, btn_voice_auto]
            )

            # ---- Events: បំបែកវីដេអូ ----
            split_info_inputs = [split_video_input, last_video_state, split_minutes, check_merge_last]
            split_minutes.change(fn=update_split_info, inputs=split_info_inputs, outputs=split_info)
            check_merge_last.change(fn=update_split_info, inputs=split_info_inputs, outputs=split_info)
            split_video_input.change(fn=update_split_info, inputs=split_info_inputs, outputs=split_info)
            last_video_state.change(fn=update_split_info, inputs=split_info_inputs, outputs=split_info)

            btn_split.click(
                fn=do_split_video,
                inputs=split_info_inputs,
                outputs=[split_log, split_outputs]
            )

        # ============================================================
        # TAB 2: បង្កើតវីដេអូ AI (Agnes)
        # ============================================================
        with gr.TabItem("✨ បង្កើតវីដេអូ AI ពិត (Agnes)"):

            gr.HTML('<div class="section-header-orange">✨ បង្កើតវីដេអូ AI ពិតដោយ Agnes</div>')

            with gr.Row():
                with gr.Column(scale=1):
                    agnes_script_input = gr.Textbox(
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
                        <div style="background:#fff3cd; padding:10px; border-radius:8px;
                                    margin-top:8px; font-size:0.9em; line-height:1.8;">
                            <b>⚡ ចំណាំសំខាន់:</b><br>
                            • ការបង្កើតវីដេអូពិតដោយ Agnes AI ត្រូវការពេល <b>១-៣ នាទី</b> ក្នុងមួយ scene<br>
                            • ប្រព័ន្ធគាំទ្រការបំបែក Scene ដោយស្វ័យប្រវត្តិតាមស្លាក []<br>
                            • សំឡេងនឹងត្រូវបានបន្ថែមជា<b>ភាសាខ្មែរ</b>ដោយ Edge TTS
                        </div>
                    """)

                    agnes_voice = gr.Radio(
                        choices=[("សំឡេងប្រុស", "male"), ("សំឡេងស្រី", "female")],
                        value="female",
                        label="🎤 សំឡេងសម្រាប់ការនិទាន"
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
                        elem_classes="primary-btn",
                        size="lg"
                    )

                with gr.Column(scale=2):
                    agnes_video_output = gr.Video(label="🎥 វីដេអូ AI ពិត", height=400)
                    agnes_status = gr.Textbox(label="📋 ស្ថានភាព", lines=14)

            def agnes_wrapper(script_text, voice_gender, resolution_str, progress=gr.Progress()):
                try:
                    w, h = resolution_str.split("x")
                    resolution = (int(w), int(h))
                except Exception:
                    resolution = (1152, 768)
                return create_video_with_agnes(script_text, voice_gender, resolution, progress)

            agnes_btn.click(
                fn=agnes_wrapper,
                inputs=[agnes_script_input, agnes_voice, agnes_resolution],
                outputs=[agnes_video_output, agnes_status]
            )

    gr.HTML("""
        <div style="text-align:center; padding:20px; color:#8a94a6; font-size:0.9em;">
            💡 បកប្រែវីដេអូដោយ SRT ឬបង្កើតវីដេអូពិតដោយ Agnes AI ជាមួយតួអង្គមានចលនា និងសំឡេងខ្មែរ
        </div>
    """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
        fill_width=True
    )
