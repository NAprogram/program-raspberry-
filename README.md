sudo apt update

sudo apt install -y python3-pip python3-opencv python3-numpy python3-pil python3-soundfile \
                    ffmpeg portaudio19-dev libatlas-base-dev libsndfile1 \
                    python3-dotenv

sudo pip3 install --upgrade pip wheel setuptools

sudo pip3 install vosk openwakeword pydub google-generativeai


—------------------------------------------------------------------------------------------------------------------------
cd ~
ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ]; then
  URL="https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_1.2.0_linux_aarch64.tar.gz"
else
  URL="https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_1.2.0_linux_armv7l.tar.gz"
fi
curl -L -o piper.tgz "$URL"
sudo mkdir -p /usr/local/piper
sudo tar -xzf piper.tgz -C /usr/local/piper --strip-components=1
sudo ln -sf /usr/local/piper/piper /usr/local/bin/piper
piper --help

—------------------------------------------------------------------------------------------------------------------------
mkdir -p ~/ai-assistant && cd ~/ai-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
# Try OpenCV regular first; if it fails, the fallback line will install headless
pip install google-generativeai pillow opencv-python numpy sounddevice soundfile \
            vosk openwakeword pydub python-dotenv || \
pip install google-generativeai pillow opencv-python-headless numpy sounddevice soundfile \
            vosk openwakeword pydub python-dotenv

—------------------------------------------------------------------------------------------------------------------------
mkdir -p models && cd models
curl -L -o vosk-en.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-en.zip && rm vosk-en.zip
mv vosk-model-small-en-us-0.15 vosk-en
cd ..

—------------------------------------------------------------------------------------------------------------------------
mkdir -p voices && cd voices
curl -L -O https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-ryan-high.onnx
curl -L -O https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-ryan-high.onnx.json
cd ..

—------------------------------------------------------------------------------------------------------

GEMINI_API_KEY="AIzaSyBbzTu6ml5ozonWJR86w_inh9PLj-hc5a0"
PIPER_VOICE=en_US-ryan-high
—------------------------------------------------------------------------------------------------------
import os, io, time, queue, threading, wave, subprocess, json
from pathlib import Path
from datetime import datetime

# ---- Load env
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_API_KEY, "Please put GEMINI_API_KEY in .env"

# Optional voice override via .env (defaults to male 'ryan')
PIPER_VOICE_NAME = os.getenv("PIPER_VOICE", "en_US-ryan-high")

# ---- Audio
import sounddevice as sd
import soundfile as sf
import numpy as np

# ---- Wake Word (local)
from openwakeword import Model as WakeModel

# ---- STT (offline)
from vosk import Model as VoskModel, KaldiRecognizer

# ---- Vision
# If picamera2 is unavailable, you can fall back to OpenCV VideoCapture(0)
from picamera2 import Picamera2
import cv2
from PIL import Image

# ---- Gemini
import google.generativeai as genai

# ---- Paths
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE/"models"/"vosk-en"
VOICE_DIR = BASE/"voices"
TMP_DIR = BASE/"tmp"
TMP_DIR.mkdir(exist_ok=True)

# ---- Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_TEXT = "gemini-1.5-flash"
GEMINI_MODEL_VISION = "gemini-1.5-flash"

# ---- Piper TTS helper (Male voice by default)
def speak_tts(text: str):
    """
    Use Piper CLI to synthesize speech and play it.
    Voice is controlled by PIPER_VOICE in .env (default: en_US-ryan-high).
    """
    if not text:
        return
    wav_path = TMP_DIR / f"tts_{int(time.time()*1000)}.wav"
    voice = VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx"
    voice_cfg = VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx.json"

    if not voice.exists():
        raise FileNotFoundError(f"Piper voice model not found: {voice}")
    if not voice_cfg.exists():
        raise FileNotFoundError(f"Piper voice config not found: {voice_cfg}")

    cmd = [
        "piper",
        "--model", str(voice),
        "--config", str(voice_cfg),
        "--output_file", str(wav_path)
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(input=text.encode("utf-8"))
    p.wait()

    subprocess.run(["aplay", "-q", str(wav_path)], check=False)
    try:
        wav_path.unlink()
    except:
        pass

# ---- Microphone stream config
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

# ---- Wake Word model
wake = WakeModel()  # default bundle includes "hey jarvis"
WAKEWORD = "hey jarvis"
WAKE_THRESHOLD = 0.45  # lower = more sensitive

# ---- Vosk STT
if not MODEL_DIR.exists():
    raise RuntimeError("Vosk model not found. Put it at models/vosk-en")
vosk_model = VoskModel(str(MODEL_DIR))
recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
recognizer.SetWords(True)

# ---- Camera (Picamera2)
# If this fails on your OS, comment these 3 lines and use OpenCV fallback below.
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))
picam2.start()

# # ---- Camera Fallback (OpenCV)
# cap = cv2.VideoCapture(0)  # uncomment if Picamera2 not available

# ---- Audio queues
audio_q = queue.Queue()

def mic_callback(indata, frames, time_info, status):
    if status:
        pass
    audio_q.put(indata.copy())
    return None

def start_mic_stream():
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=mic_callback,
        device=None
    )

def detect_wake_word():
    """
    Continuously read mic data and run wakeword detection.
    Returns when wakeword fires.
    """
    while True:
        block = audio_q.get()
        scores = wake.predict(block)
        if not scores:
            continue
        name, score = max(scores.items(), key=lambda x: x[1])
        if score >= WAKE_THRESHOLD:
            return

def record_until_silence(max_seconds=8, silence_ms=900, thresh=0.01):
    """
    Record user speech after wake word until silence.
    Returns path to WAV file recorded at 16kHz mono.
    """
    frames = []
    start_time = time.time()
    silent_for = 0.0
    ms_per_block = 1000.0 * (BLOCK_SIZE / SAMPLE_RATE)

    while True:
        try:
            block = audio_q.get(timeout=2.0)
        except queue.Empty:
            break
        frames.append(block)
        energy = np.mean(np.abs(block))
        if energy < thresh:
            silent_for += ms_per_block
        else:
            silent_for = 0.0

        if silent_for >= silence_ms or (time.time() - start_time) > max_seconds:
            break

    if not frames:
        return None

    data = np.concatenate(frames, axis=0)
    wav_path = TMP_DIR / f"rec_{int(time.time()*1000)}.wav"
    sf.write(str(wav_path), data, SAMPLE_RATE, subtype="PCM_16")
    return wav_path

def transcribe_vosk(wav_path: Path) -> str:
    with wave.open(str(wav_path), "rb") as wf:
        recognizer.Reset()
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)
        try:
            res = json.loads(recognizer.FinalResult())
            return (res.get("text") or "").strip()
        except:
            return ""

def capture_image() -> Image.Image:
    # Picamera2 capture
    frame = picam2.capture_array()  # RGB numpy array
    return Image.fromarray(frame)

    # # OpenCV fallback:
    # ok, frame = cap.read()
    # if not ok:
    #     raise RuntimeError("Camera capture failed.")
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # return Image.fromarray(frame)

def ask_gemini_text(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL_TEXT)
    resp = model.generate_content(prompt, safety_settings=None)
    return (resp.text or "").strip()

def describe_image_with_gemini(img: Image.Image, user_prompt: str = "Describe what you see in detail."):
    model = genai.GenerativeModel(GEMINI_MODEL_VISION)
    resp = model.generate_content([user_prompt, img], safety_settings=None)
    return (resp.text or "").strip()

def route_command(cmd_text: str) -> str:
    """
    Decide what to do based on user speech text.
    Expand here for smart devices (GPIO/Home Assistant).
    """
    txt = cmd_text.lower()
    if any(k in txt for k in [
        "what do you see", "what do you see?",
        "what's in front", "what do i look like",
        "describe the room"
    ]):
        img = capture_image()
        return describe_image_with_gemini(
            img,
            "Describe the scene concisely like a smart home assistant. Mention objects and their positions."
        )
    # Default: general question/answer
    return ask_gemini_text(f"You are a friendly voice assistant for a smart room. User said: {cmd_text}")

def main():
    print("Starting mic… (Ctrl+C to stop)")
    with start_mic_stream():
        speak_tts("Assistant ready. Say 'Hey Jarvis' to start.")
        while True:
            try:
                # 1) Wait for wakeword
                detect_wake_word()
                speak_tts("Yes?")
                # 2) Record command
                wav_path = record_until_silence()
                if not wav_path:
                    speak_tts("Sorry, I didn't catch that.")
                    continue
                # 3) STT
                text = transcribe_vosk(wav_path)
                try:
                    wav_path.unlink()
                except:
                    pass
                if not text:
                    speak_tts("Sorry, I didn't hear anything.")
                    continue
                print(f"[You]: {text}")
                # 4) Route → Gemini
                reply = route_command(text)
                print(f"[Assistant]: {reply}")
                # 5) TTS back
                speak_tts(reply)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print("Error:", e)
                speak_tts("An error occurred.")
                time.sleep(0.5)

if __name__ == "__main__":
    main()

—------------------------------------------------------------------------------------------------------
cd ~/ai-assistant
source .venv/bin/activate
python assistant.py

—------------------------------------------------------------------------------------------------------

pip install python-dotenv

pip install sounddevice

pip install soundfile

pip install numpy

pip install openwakeword

pip install vosk

pip install picamera2

pip install pillow

pip install google-generativeai
---------------------------------------------------------------
# assistant.py
# Full, robust AI assistant for Raspberry Pi 4B
# - Wake word via openwakeword ("hey jarvis")
# - Offline STT via Vosk
# - Vision Q&A via Gemini (uses your GEMINI_API_KEY / GOOGLE_API_KEY)
# - TTS via Piper (default male voice: en_US-ryan-high)
# - Camera: Picamera2 if available, else OpenCV fallback
# - Plays audio via aplay (ALSA)

import os, io, time, queue, threading, wave, subprocess, json, sys
from pathlib import Path
from datetime import datetime

# ---------- Paths & .env (force-load from script directory) ----------
BASE = Path(__file__).resolve().parent
from dotenv import load_dotenv
load_dotenv(BASE / ".env")

# Accept either name; some docs say GOOGLE_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "API key not found.\nCreate ~/ai-assistant/.env with:\n"
        "GEMINI_API_KEY=YOUR_REAL_KEY_HERE\n"
        "PIPER_VOICE=en_US-ryan-high"
    )

# Optional voice override via .env (defaults to male 'ryan')
PIPER_VOICE_NAME = os.getenv("PIPER_VOICE", "en_US-ryan-high")

# ---------- Audio ----------
import sounddevice as sd
import soundfile as sf
import numpy as np

# ---------- Wake Word (local) ----------
from openwakeword import Model as WakeModel

# ---------- STT (offline) ----------
from vosk import Model as VoskModel, KaldiRecognizer

# ---------- Vision (Picamera2 if available, else OpenCV) ----------
import cv2
from PIL import Image

USE_PICAMERA2 = False
try:
    from picamera2 import Picamera2  # will not exist on Ubuntu unless installed
    USE_PICAMERA2 = True
except Exception:
    USE_PICAMERA2 = False

# ---------- Gemini ----------
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_TEXT = "gemini-1.5-flash"
GEMINI_MODEL_VISION = "gemini-1.5-flash"

# ---------- Folders ----------
MODEL_DIR = BASE / "models" / "vosk-en"
VOICE_DIR = BASE / "voices"
TMP_DIR = BASE / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# ---------- TTS: Piper helper ----------
def speak_tts(text: str):
    """
    Use Piper CLI to synthesize speech and play it.
    Voice is controlled by PIPER_VOICE in .env (default: en_US-ryan-high).
    """
    if not text:
        return
    voice = VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx"
    voice_cfg = VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx.json"
    if not voice.exists() or not voice_cfg.exists():
        raise FileNotFoundError(
            f"Piper voice missing.\nExpected:\n  {voice}\n  {voice_cfg}\n"
            "Download Ryan voice into ~/ai-assistant/voices/"
        )
    wav_path = TMP_DIR / f"tts_{int(time.time()*1000)}.wav"

    cmd = [
        "piper",
        "--model", str(voice),
        "--config", str(voice_cfg),
        "--output_file", str(wav_path)
    ]
    # Piper reads text from stdin
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.communicate(input=text.encode("utf-8"))
    p.wait()

    # Play with ALSA
    subprocess.run(["aplay", "-q", str(wav_path)], check=False)
    try:
        wav_path.unlink()
    except Exception:
        pass

# ---------- Mic stream config ----------
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

# ---------- Wake word ----------
wake = WakeModel()  # default bundle includes "hey jarvis"
WAKEWORD = "hey jarvis"
WAKE_THRESHOLD = 0.45  # lower = more sensitive

# ---------- Vosk STT ----------
if not MODEL_DIR.exists():
    raise RuntimeError(
        f"Vosk model not found at {MODEL_DIR}\n"
        "Download small EN model and unpack to models/vosk-en/"
    )
vosk_model = VoskModel(str(MODEL_DIR))
recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
recognizer.SetWords(True)

# ---------- Camera setup ----------
picam2 = None
cap = None

if USE_PICAMERA2:
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_still_configuration(main={"size": (1280, 720)}))
        picam2.start()
        print("[Camera] Using Picamera2")
    except Exception as e:
        print("[Camera] Picamera2 failed, falling back to OpenCV:", e)
        USE_PICAMERA2 = False

if not USE_PICAMERA2:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] OpenCV could not open /dev/video0")
        print("         If you have a Raspberry Pi Camera, install picamera2 via apt and reboot.")
        # We won't crash here; just handle gracefully when called.
    else:
        print("[Camera] Using OpenCV VideoCapture(0)")

def capture_image() -> Image.Image:
    """
    Returns a PIL.Image from the active camera.
    """
    if USE_PICAMERA2 and picam2 is not None:
        frame = picam2.capture_array()  # RGB ndarray
        return Image.fromarray(frame)
    if cap is not None and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera capture failed (OpenCV).")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    raise RuntimeError("No camera available. (Neither Picamera2 nor OpenCV capture is working.)")

# ---------- Audio queues ----------
audio_q = queue.Queue()

def mic_callback(indata, frames, time_info, status):
    if status:
        # print(status)
        pass
    audio_q.put(indata.copy())
    return None

def start_mic_stream():
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=mic_callback,
        device=None  # default device
    )

def detect_wake_word():
    """
    Continuously read mic data and run wakeword detection.
    Returns when wake word fires.
    """
    while True:
        block = audio_q.get()
        scores = wake.predict(block)
        if not scores:
            continue
        # Top keyword score
        _, score = max(scores.items(), key=lambda x: x[1])
        if score >= WAKE_THRESHOLD:
            return

def record_until_silence(max_seconds=8, silence_ms=900, thresh=0.01):
    """
    Record user speech after wake word until silence.
    Returns path to WAV (16kHz mono).
    """
    frames = []
    start_time = time.time()
    silent_for = 0.0
    ms_per_block = 1000.0 * (BLOCK_SIZE / SAMPLE_RATE)

    while True:
        try:
            block = audio_q.get(timeout=2.0)
        except queue.Empty:
            break
        frames.append(block)
        energy = np.mean(np.abs(block))
        if energy < thresh:
            silent_for += ms_per_block
        else:
            silent_for = 0.0

        if silent_for >= silence_ms or (time.time() - start_time) > max_seconds:
            break

    if not frames:
        return None

    data = np.concatenate(frames, axis=0)
    wav_path = TMP_DIR / f"rec_{int(time.time()*1000)}.wav"
    sf.write(str(wav_path), data, SAMPLE_RATE, subtype="PCM_16")
    return wav_path

def transcribe_vosk(wav_path: Path) -> str:
    with wave.open(str(wav_path), "rb") as wf:
        recognizer.Reset()
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)
        try:
            res = json.loads(recognizer.FinalResult())
            return (res.get("text") or "").strip()
        except Exception:
            return ""

def ask_gemini_text(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL_TEXT)
    resp = model.generate_content(prompt, safety_settings=None)
    return (getattr(resp, "text", None) or "").strip()

def describe_image_with_gemini(img: Image.Image, user_prompt: str = "Describe what you see in detail."):
    model = genai.GenerativeModel(GEMINI_MODEL_VISION)
    resp = model.generate_content([user_prompt, img], safety_settings=None)
    return (getattr(resp, "text", None) or "").strip()

def route_command(cmd_text: str) -> str:
    """
    Expand here for smart devices (GPIO/Home Assistant, plugs, etc.).
    """
    txt = cmd_text.lower().strip()
    if any(k in txt for k in [
        "what do you see", "what do you see?",
        "what's in front", "what do i look like",
        "describe the room"
    ]):
        try:
            img = capture_image()
            return describe_image_with_gemini(
                img,
                "Describe the scene concisely like a smart home assistant. Mention objects and their positions."
            )
        except Exception as e:
            return f"I couldn't access the camera: {e}"

    # Default: send to Gemini as a voice assistant
    return ask_gemini_text(
        f"You are a friendly, concise smart-room voice assistant. Respond for speech.\nUser: {cmd_text}"
    )

def main():
    print("Starting mic… (Ctrl+C to stop)")
    # Warm greet after audio stream starts
    with start_mic_stream():
        try:
            speak_tts("Assistant ready. Say 'Hey Jarvis' to start.")
        except Exception as e:
            print("[TTS] Could not speak:", e)

        while True:
            try:
                # 1) Wait for wake word
                detect_wake_word()
                try:
                    speak_tts("Yes?")
                except Exception as e:
                    print("[TTS] Could not speak:", e)

                # 2) Record command
                wav_path = record_until_silence()
                if not wav_path:
                    speak_tts("Sorry, I didn't catch that.")
                    continue

                # 3) STT
                text = transcribe_vosk(wav_path)
                try:
                    wav_path.unlink()
                except Exception:
                    pass
                if not text:
                    speak_tts("Sorry, I didn't hear anything.")
                    continue

                print(f"[You]: {text}")

                # 4) Route → Gemini / Vision
                reply = route_command(text)
                print(f"[Assistant]: {reply}")

                # 5) TTS back
                speak_tts(reply)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print("[Main Loop Error]:", e)
                try:
                    speak_tts("An error occurred.")
                except Exception:
                    pass
                time.sleep(0.5)

    # Cleanup
    try:
        if USE_PICAMERA2 and picam2 is not None:
            picam2.stop()
        if not USE_PICAMERA2 and cap is not None:
            cap.release()
    except Exception:
        pass

if __name__ == "__main__":
    main()


