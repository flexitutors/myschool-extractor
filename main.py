import os
import random
import json
import cloudinary.uploader
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# Cloudinary Setup
cloudinary.config(
  cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"),
  api_key = os.environ.get("CLOUDINARY_API_KEY"),
  api_secret = os.environ.get("CLOUDINARY_API_SECRET")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HARDCODED KEYS LIST
# Replace the placeholder text with your actual 8 API keys
API_KEYS = [
    "GEMINI_API_KEY_1",
    "GEMINI_API_KEY_2",
    "GEMINI_API_KEY_3",
    "GEMINI_API_KEY_4",
    "GEMINI_API_KEY_5",
    "GEMINI_API_KEY_6",
    "GEMINI_API_KEY_7",
    "GEMINI_API_KEY_8"
]

def get_model():
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="No API keys defined.")
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
            "\"explanation\": \"str\"}. "
            "IMPORTANT: Use Unicode/plain text for all math (e.g., 'x²', 'π', '√') "
            "and NO LaTeX or HTML math tags."
        )
        
        response = model.generate_content([prompt, myfile])
        json_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_text)
        
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
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
  
