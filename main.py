# ===============================================================
# PEST BOT — FastAPI Backend (Production Safe)
# ===============================================================
# ✅ Features:
# - Chat (Groq + Llama 3.1)
# - Image-based pest analysis
# - Voice input support
# - RAG from local CSV/TXT files
# ✅ Fully compatible with Render & Flutter
# ===============================================================

import os
import io
import base64
import csv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from PIL import Image
import speech_recognition as sr

# ============================================================
# 1. LOAD GROQ API KEY (ENVIRONMENT + LOCAL FALLBACK)
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        with open("config/groq_key.txt", "r") as f:
            GROQ_API_KEY = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("❌ GROQ_API_KEY not found — please add it in Render environment variables.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

# ============================================================
# 2. SYSTEM PROMPT
# ============================================================
PESTBOT_SYSTEM_PROMPT = """
You are Pest Bot, an advanced agricultural AI assistant.
You can identify pests and diseases from text, voice, or images,
and provide chemical and organic treatment and prevention guidance.
Always answer clearly, practically, and based on scientific context.
"""

# ============================================================
# 3. LOAD LOCAL RAG DATA (OPTIONAL)
# ============================================================
RAG_KB = []
DATA_FOLDER = "data"

if os.path.exists(DATA_FOLDER):
    for file in os.listdir(DATA_FOLDER):
        try:
            if file.endswith(".csv"):
                with open(os.path.join(DATA_FOLDER, file), encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        RAG_KB.append(" ".join(row))
            elif file.endswith(".txt"):
                with open(os.path.join(DATA_FOLDER, file), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            RAG_KB.append(line.strip())
        except Exception as e:
            print(f"[RAG LOAD ERROR] {file}: {e}")
else:
    print("⚠️ Warning: 'data' folder not found — continuing without RAG context.")


def retrieve_relevant_chunks(query, limit=5):
    """Simple keyword-based retrieval from local RAG KB."""
    q = query.lower()
    results = []
    for line in RAG_KB:
        if any(word in line.lower() for word in q.split()):
            results.append(line)
        if len(results) >= limit:
            break
    return "\n".join(results) if results else ""


# ============================================================
# 4. GROQ API HANDLER
# ============================================================
def ask_groq(system_prompt, user_input, rag_context=""):
    """Send user input to Groq API and get AI response."""
    final_prompt = (
        f"User Question:\n{user_input}\n\n"
        f"Dataset Context:\n{rag_context}\n\n"
        "Answer as Pest Bot:"
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq API Error] {e}")
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")


# ============================================================
# 5. FASTAPI APP CONFIG
# ============================================================
app = FastAPI(title="Pest Bot API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 6. CHAT ENDPOINT (UNIVERSAL FIX FOR 422 ERROR)
# ============================================================
@app.post("/chat")
async def chat(request: Request):
    """
    Accepts message as:
    - application/x-www-form-urlencoded
    - multipart/form-data
    - raw JSON { "message": "..." }
    ✅ Prevents 422 error
    """
    try:
        content_type = request.headers.get("Content-Type", "")
        message = None

        if "application/json" in content_type:
            data = await request.json()
            message = data.get("message") or data.get("prompt")
        elif "form" in content_type:
            form = await request.form()
            message = form.get("message") or form.get("prompt")

        if not message:
            raise HTTPException(status_code=400, detail="No message provided.")

        rag = retrieve_relevant_chunks(message)
        answer = ask_groq(PESTBOT_SYSTEM_PROMPT, message, rag_context=rag)
        return {"reply": answer}

    except Exception as e:
        print(f"[Chat Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 7. IMAGE ANALYSIS
# ============================================================
@app.post("/image")
async def analyze_image(file: UploadFile = File(...)):
    """Accepts an image and analyzes pest or disease."""
    try:
        img = Image.open(file.file)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    prompt = f"""
Analyze this crop image (base64 encoded). Provide:
1. Pest or disease name
2. Severity
3. Likely cause
4. Chemical and organic treatment
5. Prevention advice

IMAGE_DATA (truncated): {encoded[:300]}
"""
    answer = ask_groq(PESTBOT_SYSTEM_PROMPT, prompt)
    return {"response": answer}


# ============================================================
# 8. VOICE RECOGNITION
# ============================================================
@app.post("/voice")
async def voice_chat(file: UploadFile = File(...)):
    """Converts audio to text and responds intelligently."""
    try:
        audio_bytes = await file.read()
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            data = recognizer.record(source)
        text = recognizer.recognize_google(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Voice recognition failed: {e}")

    rag = retrieve_relevant_chunks(text)
    answer = ask_groq(PESTBOT_SYSTEM_PROMPT, text, rag_context=rag)
    return {"transcript": text, "response": answer}


# ============================================================
# 9. ROOT ROUTE (OPTIONAL)
# ============================================================
@app.get("/")
async def root():
    return {"status": "✅ Pest Bot API is running successfully."}


# ============================================================
# 10. ENTRY POINT (Render Auto-Detect Safe)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
