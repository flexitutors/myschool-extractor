import os
import json
import base64
import requests
import cloudinary.uploader
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load keys from environment
API_KEYS = [os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 6) if os.environ.get(f"GEMINI_API_KEY_{i}")]

def get_best_model_name(key: str) -> str:
    """Queries the Gemini metadata endpoint and selects a supported model."""
    try:
        # Corrected Metadata Endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        res = requests.get(url, timeout=10)
        
        if res.status_code == 200:
            models_data = res.json().get("models", [])
            # Find models that support generateContent
            supported = [m["name"] for m in models_data if "generateContent" in m.get("supportedGenerationMethods", [])]
            
            # Preference hierarchy
            for pref in ["1.5-pro", "1.5-flash", "2.0-pro"]:
                for model in supported:
                    if pref in model.lower():
                        return model # returns 'models/gemini-1.5-flash' format
            return supported[0] if supported else "models/gemini-1.5-flash"
    except Exception as e:
        print(f"Metadata check failed: {e}")
    return "models/gemini-1.5-flash"

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    file_bytes = await file.read()
    base64_image = base64.b64encode(file_bytes).decode('utf-8')
    
    prompt = (
        "Analyze this exam question. Return JSON only with fields: "
        "has_diagram, year, question, options (list), correct_answer, explanation. "
        "Use plain text for math. No LaTeX/HTML."
    )
    
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": file.content_type, "data": base64_image}}
        ]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    last_error = "No keys available"
    for key in API_KEYS:
        try:
            model_name = get_best_model_name(key)
            # URL now correctly includes the 'models/' prefix returned by the metadata API
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
            
            response = requests.post(url, json=payload, timeout=45)
            
            if response.status_code == 200:
                text_content = response.json()['candidates'][0]['content']['parts'][0]['text']
                result = json.loads(text_content)
                
                if result.get("has_diagram"):
                    temp_path = f"temp_{file.filename}"
                    with open(temp_path, "wb") as f: f.write(file_bytes)
                    result["diagram_url"] = cloudinary.uploader.upload(temp_path).get("secure_url")
                    os.remove(temp_path)
                else:
                    result["diagram_url"] = None
                return result
            else:
                last_error = f"Status {response.status_code}: {response.text}"
                continue
        except Exception as e:
            last_error = str(e)
            continue
            
    raise HTTPException(status_code=500, detail=f"All keys failed. Last error: {last_error}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    
