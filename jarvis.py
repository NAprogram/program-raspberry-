from __future__ import annotations
import json, os, time, sys, traceback, tempfile, asyncio
import numpy as np
import sounddevice as sd
import soundfile as sf
# import pyttsx3  # no longer used with Fix B
import playsound
import edge_tts
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from lights import make_light_controller
from vision import Vision

# ---------------- Config & Env ----------------
CONFIG_PATH = Path("config.json")
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

load_dotenv()  # for GEMINI_API_KEY

ASSISTANT_NAME = cfg.get("assistant_name", "Jarvis")
STT_MODEL_SIZE = cfg.get("stt_model_size", "small")
AI_BACKEND = cfg.get("ai_backend", "gemini").lower()
GEMINI_MODEL = cfg.get("gemini_model", "gemini-1.5-flash")
MAX_REPLY_CHARS = int(cfg.get("max_reply_chars", 800))
vision_cfg = cfg.get("vision", {"enabled": True, "camera_index": 0, "top_k": 3, "confidence": 0.35})

# ---------------- TTS (Fix B: Edge-TTS) ----------------
# Choose a neural voice you like; a few examples:
#   en-US-AriaNeural, en-US-GuyNeural, en-GB-RyanNeural, en-GB-LibbyNeural
EDGE_TTS_VOICE = "en-US-GuyNeural"

async def _say_edge_async(text: str):
    text = (text or "").strip()
    if not text:
        return
    communicate = edge_tts.Communicate(text, voice=EDGE_TTS_VOICE)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        out = f.name
    try:
        await communicate.save(out)
        playsound.playsound(out)  # blocks until audio finishes
    finally:
        try:
            os.remove(out)
        except Exception:
            pass

def say(text: str):
    # Edge-TTS wrapper (sync call that runs the async function)
    out = (text or "").strip()
    if not out:
        out = "I don't have anything to say yet."
    out = out[:MAX_REPLY_CHARS]
    print(f"{ASSISTANT_NAME}: {out}")
    try:
        asyncio.run(_say_edge_async(out))
    except RuntimeError:
        # If already in an event loop (rare on Windows console), fallback to a new loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_say_edge_async(out))
        finally:
            loop.close()



# ---------------- STT (Whisper) ----------------
from faster_whisper import WhisperModel
# int8 CPU is fine; if you have GPU, you can change device="cuda", compute_type="int8_float16"
stt_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")

# Tunables: make recording a bit more tolerant so it doesn't cut off
REC_SAMPLE_RATE = 16000
REC_MAX_SEC = 10            # allow up to 10s command
REC_SILENCE_SEC = 1.0       # stop after ~1s of near-silence
REC_SILENCE_THRESH = 0.006  # lower = more sensitive (was 0.008)

def record_until_silence(sr=REC_SAMPLE_RATE, max_sec=REC_MAX_SEC, silence_sec=REC_SILENCE_SEC, thresh=REC_SILENCE_THRESH):
    print("[Mic] Recording… speak now (stops after short silence).")
    block = 1024
    silence_blocks_needed = int(silence_sec * sr / block)
    silent_blocks = 0
    captured = []

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block):
            start = time.time()
            while True:
                data, _ = sd.rec(block, samplerate=sr, channels=1, dtype="float32"), None
                sd.wait()
                frame = data.reshape(-1, 1)
                captured.append(frame.copy())
                rms = float(np.sqrt(np.mean(np.square(frame))))
                if rms < thresh:
                    silent_blocks += 1
                else:
                    silent_blocks = 0
                if silent_blocks >= silence_blocks_needed:
                    break
                if time.time() - start > max_sec:
                    print("[Mic] Reached max seconds; stopping.")
                    break
    except Exception as e:
        print("[Mic] Error opening/reading microphone:", e)
        return np.zeros((0,1), dtype=np.float32), sr

    audio = np.concatenate(captured, axis=0) if captured else np.zeros((0,1), dtype=np.float32)
    print(f"[Mic] Captured {len(audio)/sr:.2f}s of audio.")
    return audio.squeeze(), sr

def transcribe_numpy(audio: np.ndarray, sr: int) -> str:
    if audio is None or audio.size == 0:
        print("[STT] No audio captured.")
        return ""
    tmp = Path("last_command.wav")
    sf.write(tmp.as_posix(), audio, sr)
    try:
        segments, info = stt_model.transcribe(tmp.as_posix(), beam_size=1)
        text = "".join(seg.text for seg in segments).strip()
        print(f"[STT] Transcript: {text!r}")
        return text
    except Exception as e:
        print("[STT] Transcription error:", e)
        return ""

# ---------------- Devices ----------------
light = make_light_controller(cfg.get("light_controller", "dummy"), cfg)
vision = Vision(
    camera_index=vision_cfg.get("camera_index", 0),
    top_k=vision_cfg.get("top_k", 3),
    conf=vision_cfg.get("confidence", 0.35),
) if vision_cfg.get("enabled", True) else None

# ---------------- Gemini ----------------
def ask_gemini(prompt: str) -> str:
    try:
        import google.generativeai as genai
    except Exception:
        return "Gemini SDK not installed. Run: pip install google-generativeai"

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "Gemini API key missing. Put GEMINI_API_KEY=... in your .env."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
        text = (text or "").strip()
        if not text:
            return "I couldn't find an answer."
        return text
    except Exception as e:
        print("[Gemini] Error:", e)
        traceback.print_exc()
        return f"Gemini error: {e}"

# ---------------- Intent Routing ----------------
def handle_command(cmd: str) -> str:
    c = (cmd or "").lower().strip()
    if not c:
        return "I didn't hear anything. Try again."

    # quit
    if c in {"quit", "exit", "goodbye", "bye"}:
        return "__QUIT__"

    # lights
    if ("turn on" in c and "light" in c) or ("lights on" in c):
        try:
            light.turn_on()
            return "Lights on."
        except Exception as e:
            return f"I couldn't turn on the lights: {e}"

    if ("turn off" in c and "light" in c) or ("lights off" in c):
        try:
            light.turn_off()
            return "Lights off."
        except Exception as e:
            return f"I couldn't turn off the lights: {e}"

    # vision
    if "what do you see" in c or "what can you see" in c or "what do you see now" in c or "see me" in c:
        if vision is None:
            return "Vision is disabled."
        return vision.describe_scene()

    # utilities
    if "time" in c:
        return "It's " + datetime.now().strftime("%H:%M")
    if ("date" in c) or ("day" in c and "today" in c):
        return "Today is " + datetime.now().strftime("%A, %B %d")
    if "joke" in c:
        return "Why did the computer go to therapy? It had too many unresolved issues."
    if "help" in c:
        return ("Say: 'turn on the light', 'turn off the light', 'what do you see', "
                "'what time is it', or just ask any question.")

    # fallback → Gemini (if enabled)
    if AI_BACKEND == "gemini":
        ans = ask_gemini(cmd)
        return ans or "I couldn't find an answer."

    # no backend
    return "I heard: " + cmd

# ---------------- Main Loop ----------------
def main_loop():
    say(f"Hello your honorable Mr Proffessor and my creator. How can I help you in this wonderful day?")
    while True:
        raw = input("> Press Enter to speak (or type q + Enter to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            say("Goodbye.")
            break

        audio, sr = record_until_silence()
        cmd = transcribe_numpy(audio, sr)
        print(f"[You]: {cmd!r}")

        reply = handle_command(cmd)
        print(f"[Reply raw]: {reply!r}")

        if reply == "__QUIT__":
            say("Goodbye.")
            break

        if not reply or not reply.strip():
            reply = "I processed your request but produced an empty response. Please try again."

        say(reply)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[Exit] Goodbye!")


