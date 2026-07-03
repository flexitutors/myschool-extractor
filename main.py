import os
import json
import base64
import requests
import cloudinary.uploader
import asyncio
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load keys directly from environment
API_KEYS = [os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 6) if os.environ.get(f"GEMINI_API_KEY_{i}")]

def get_best_model_name(key: str) -> str:
    # Keeps return layout uniform as "models/gemini-1.5-flash"
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
    
    #  CORRECTED: Fully qualified Gemini endpoint URL layout path string
    url = f"https://googleapis.com{get_best_model_name(key)}:generateContent?key={key}"
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            try:
                # Correct response dictionary nested key traversal paths
                text = res_json['candidates'][0]['content']['parts'][0]['text']
                cleaned_text = text.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned_text), file_bytes
            except Exception as json_err:
                print(f"❌ JSON Parsing/Extraction Failed. Raw API Response: {res_json}. Error: {json_err}")
                return {"year": "Unknown", "questions": []}, file_bytes
        else:
            print(f"❌ Gemini API returned non-200 status: {response.status_code}. Response: {response.text}")
            return {"year": "Unknown", "questions": []}, file_bytes
    except Exception as e:
        print(f"❌ System Exception in process_page: {str(e)}")
        return {"year": "Unknown", "questions": []}, file_bytes

@app.post("/extract")
async def extract_data(
    file: Optional[UploadFile] = File(None), 
    files: Optional[List[UploadFile]] = File(None)
):
    if not API_KEYS:
        print("❌ Error: No API keys found in environment variables!")
        raise HTTPException(status_code=500, detail="Server Configuration Error: API_KEYS array is empty.")

    all_uploads = []
    if file is not None:
        all_uploads.append(file)
    if files is not None:
        all_uploads.extend(files)
        
    if not all_uploads:
        raise HTTPException(status_code=400, detail="Payload error: You must provide data under 'file' or 'files'.")
    
    tasks = [process_page(f, API_KEYS[i % len(API_KEYS)]) for i, f in enumerate(all_uploads)]
    results = await asyncio.gather(*tasks)
    
    final_output = {"year": "Unknown", "questions": []}
    
    for (data, file_bytes) in results:
        if data.get("year") != "Unknown": 
            final_output["year"] = data.get("year")
        
        for q in data.get("questions", []):
            if q.get("has_diagram"):
                unique_id = uuid.uuid4().hex
                temp_path = f"temp_{unique_id}.jpg"
                
                with open(temp_path, "wb") as f: 
                    f.write(file_bytes)
                    
                res = cloudinary.uploader.upload(temp_path)
                q["diagram_url"] = res.get("secure_url")
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                q["diagram_url"] = None
            
            final_output["questions"].append(q)
            
    return final_output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    
