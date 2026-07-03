import os
import json
import base64
import requests
import cloudinary.uploader
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEYS = [os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 6) if os.environ.get(f"GEMINI_API_KEY_{i}")]

def get_best_model_name(key: str) -> str:
    # Use cached or default to avoid metadata calls on every single page
    return "models/gemini-1.5-flash"

async def process_page(file: UploadFile, key: str):
    file_bytes = await file.read()
    base64_image = base64.b64encode(file_bytes).decode('utf-8')
    
    prompt = (
        "You are an expert exam OCR system. Analyze this exam image. "
        "Extract ALL questions present. Return JSON ONLY in this format: "
        "{\"year\": \"extract the year if present, otherwise 'Unknown'\", "
        "\"questions\": [{\"has_diagram\": bool, \"diagram_description\": \"str\", "
        "\"question\": \"str\", \"options\": [\"A\", \"B\", \"C\", \"D\"], "
        "\"correct_answer\": \"str\", \"explanation\": \"str\"}]}"
        "CRITICAL: If has_diagram is true, describe it in diagram_description. No LaTeX/HTML."
    )
    
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": file.content_type, "data": base64_image}}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    url = f"https://generativelanguage.googleapis.com/v1beta/{get_best_model_name(key)}:generateContent?key={key}"
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            text = response.json()['candidates'][0]['content']['parts'][0]['text']
            return json.loads(text.replace("```json", "").replace("```", "").strip()), file_bytes
        return {"year": "Unknown", "questions": []}, file_bytes
    except Exception:
        return {"year": "Unknown", "questions": []}, file_bytes

@app.post("/extract-batch")
async def extract_batch(files: List[UploadFile] = File(...)):
    # Create tasks for all pages
    tasks = [process_page(file, API_KEYS[i % len(API_KEYS)]) for i, file in enumerate(files)]
    results = await asyncio.gather(*tasks)
    
    final_output = {"year": "Unknown", "questions": []}
    
    for (data, file_bytes) in results:
        # Aggregate year
        if data.get("year") != "Unknown": final_output["year"] = data.get("year")
        
        # Process diagrams for each question in this page
        for q in data.get("questions", []):
            if q.get("has_diagram"):
                # Handle Cloudinary
                temp_path = f"temp_diagram.jpg"
                with open(temp_path, "wb") as f: f.write(file_bytes)
                res = cloudinary.uploader.upload(temp_path)
                q["diagram_url"] = res.get("secure_url")
                os.remove(temp_path)
            
            final_output["questions"].append(q)
            
    return final_output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    
