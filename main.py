# ===============================================================
#       FASTAPI BACKEND FOR ANDROID APP — PEST BOT AI
#   Features:
#     ✔ Chat (LLaMA 3.1)
#     ✔ Image Analysis
#     ✔ Voice Input
#     ✔ RAG Dataset Support (CSV / TXT)
# ===============================================================

import base64
import io
import os
import csv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from PIL import Image
import speech_recognition as sr

# ============================================================
#                    LOAD GROQ API KEY
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        with open("config/groq_key.txt", "r") as f:
            GROQ_API_KEY = f.read().strip()
    except FileNotFoundError:
        raise ValueError(
            "Groq API key not found. Please set GROQ_API_KEY in Render environment or add config/groq_key.txt locally."
        )

GROQ_MODEL = "llama-3.1-8b-instant"

# ============================================================
#                  PEST BOT SYSTEM PROMPT
# ============================================================
PESTBOT_SYSTEM_PROMPT = """
You are Pest Bot, an advanced AI expert specializing in identifying 
agricultural pests, diagnosing plant diseases, and providing accurate 
treatment and prevention advice.

Your responsibilities:
- Detect pests and diseases from user queries or images
- Recommend pesticides, organic solutions, and prevention steps
- Provide accurate agricultural guidance for farmers and students
- Use dataset knowledge whenever relevant
- Always answer clearly, professionally, and practically
"""

# ============================================================
#           LOAD ALL DATASETS FROM backend/data/
# ============================================================
DATA_FOLDER = "data"
RAG_KB = []

if os.path.exists(DATA_FOLDER):
    for file in os.listdir(DATA_FOLDER):
        try:
            path = os.path.join(DATA_FOLDER, file)
            if file.endswith(".csv"):
                with open(path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        RAG_KB.append(" ".join(row))
            elif file.endswith(".txt"):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            RAG_KB.append(line.strip())
        except Exception as e:
            print(f"Error reading dataset file {file}: {e}")
else:
    print("Warning: 'data' folder not found — RAG dataset is empty.")


def retrieve_relevant_chunks(query, limit=5):
    """Simple keyword-based RAG search"""
    q = query.lower()
    results = []
    for line in RAG_KB:
        if any(word in line.lower() for word in q.split()):
            results.append(line)
        if len(results) >= limit:
            break
    return "\n".join(results) if results else ""


# ============================================================
#                    GROQ CLIENT
# ============================================================
client = Groq(api_key=GROQ_API_KEY)


def ask_groq(system_prompt, user_input, rag_context=""):
    """Send the constructed prompt to Groq API and return the answer."""
    final_prompt = (
        f"User Question:\n{user_input}\n\n"
        f"Dataset Information:\n{rag_context}\n\n"
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
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Groq API Error] {e}")
        raise HTTPException(status_code=500, detail="Groq API error occurred.")


# ============================================================
#                  FASTAPI SERVER SETUP
# ============================================================
app = FastAPI(title="Pest Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#                    CHAT ENDPOINT
# ============================================================
@app.post("/chat")
async def chat(prompt: str = Form(...)):
    try:
        rag = retrieve_relevant_chunks(prompt)
        answer = ask_groq(PESTBOT_SYSTEM_PROMPT, prompt, rag_context=rag)
        return {"reply": answer}
    except Exception as e:
        print("[Chat Error]", e)
        return {"reply": f"Server error: {str(e)}"}


# ============================================================
#                    IMAGE ANALYSIS
# ============================================================
@app.post("/image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    prompt = f"""
Analyze this crop image (base64 encoded). Provide:
1. Pest/disease identification
2. Severity level
3. Reason for attack
4. Chemical and organic treatments
5. Prevention advice

IMAGE_DATA: {encoded}
"""
    answer = ask_groq(PESTBOT_SYSTEM_PROMPT, prompt)
    return {"response": answer}


# ============================================================
#                    VOICE ANALYSIS
# ============================================================
@app.post("/voice")
async def voice_chat(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            data = recognizer.record(source)
        text = recognizer.recognize_google(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Voice processing failed: {e}")

    rag = retrieve_relevant_chunks(text)
    answer = ask_groq(PESTBOT_SYSTEM_PROMPT, text, rag_context=rag)
    return {"transcript": text, "response": answer}


# ============================================================
#                    ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
