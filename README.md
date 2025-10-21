sudo dpkg --configure -a
sudo apt --fix-broken install -y
sudo apt update
python -c "import os, pathlib; from dotenv import load_dotenv; load_dotenv(pathlib.Path('~','ai-assistant','.env').expanduser()); print('Key length:', len(os.getenv('GEMINI_API_KEY','') or os.getenv('GOOGLE_API_KEY','')))"
sudo apt install -y python3-venv python3-pip portaudio19-dev ffmpeg git curl
# (Nice-to-have BLAS, skip if not found)
sudo apt install -y libatlas-base-dev || sudo apt install -y libopenblas-dev || true

# Camera stack (works on Raspberry Pi OS Bookworm; skip if not found)
sudo apt install -y python3-picamera2 libcamera-apps || true

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

