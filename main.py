import os
import random
import json
import cloudinary.uploader
from fastapi import FastAPI, UploadFile, File, HTTPException
import google.generativeai as genai

# Configuration
cloudinary.config(
  cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"),
  api_key = os.environ.get("CLOUDINARY_API_KEY"),
  api_secret = os.environ.get("CLOUDINARY_API_SECRET")
)

app = FastAPI()

# API Key Rotation Logic
KEYS_STRING = os.environ.get("GEMINI_API_KEYS")
API_KEYS = KEYS_STRING.split(",") if KEYS_STRING else []

def get_model():
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="No API keys configured.")
    api_key = random.choice(API_KEYS).strip()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        myfile = genai.upload_file(temp_filename)
        model = get_model()
        
        prompt = (
            "Analyze this exam question. Return JSON only: "
            "{\"has_diagram\": bool, \"year\": \"str\", \"question\": \"str\", "
            "\"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct_answer\": \"str\", "
            "\"explanation\": \"str\"}"
        )
        
        response = model.generate_content([prompt, myfile])
        # Clean response and parse
        json_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_text)
        
        # 3. Handle Diagram Upload
        if data.get("has_diagram"):
            upload_result = cloudinary.uploader.upload(temp_filename)
            data["diagram_url"] = upload_result.get("secure_url")
        else:
            data["diagram_url"] = None
            
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    # DYNAMIC PORT: This tells Render exactly which port to use
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
  
