import os
import json
import base64
import requests
import cloudinary.uploader
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load keys directly from environment
API_KEYS = [os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 6) if os.environ.get(f"GEMINI_API_KEY_{i}")]

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    # 1. Read file as bytes and convert to Base64
    file_bytes = await file.read()
    base64_image = base64.b64encode(file_bytes).decode('utf-8')
    
    # 2. Prepare Payload (Strictly following the REST API structure)
    prompt = (
        "Analyze this exam question. Return JSON only with fields: "
        "has_diagram, year, question, options (list), correct_answer, explanation. "
        "Use plain text for math. No LaTeX/HTML."
    )
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": file.content_type, "data": base64_image}}
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    # 3. Rotate through keys
    last_error = "No keys available"
    for key in API_KEYS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
            response = requests.post(url, json=payload, timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                text_content = data['candidates'][0]['content']['parts'][0]['text']
                result = json.loads(text_content)
                
                # Handle Diagram via Cloudinary if needed
                if result.get("has_diagram"):
                    # Temporarily save to upload to cloudinary
                    temp_path = f"temp_{file.filename}"
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)
                    upload_result = cloudinary.uploader.upload(temp_path)
                    result["diagram_url"] = upload_result.get("secure_url")
                    os.remove(temp_path)
                else:
                    result["diagram_url"] = None
                    
                return result
            else:
                last_error = f"API Error {response.status_code}: {response.text}"
                print(f"Key failed: {last_error}")
                continue
                
        except Exception as e:
            last_error = str(e)
            print(f"Exception: {last_error}")
            continue
            
    raise HTTPException(status_code=500, detail=f"All keys failed. Last error: {last_error}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
  
