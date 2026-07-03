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

# Allow connections from your web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load keys from GEMINI_API_KEY_1 to GEMINI_API_KEY_8
API_KEYS = []
for i in range(1, 9):
    key = os.environ.get(f"GEMINI_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

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
        # Save file locally
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # Upload to Gemini
        myfile = genai.upload_file(temp_filename)
        model = get_model()
        
        # Prompt with Unicode/Plaintext Math instructions
        prompt = (
            "Analyze this exam question. Return JSON only: "
            "{\"has_diagram\": bool, \"year\": \"str\", \"question\": \"str\", "
            "\"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct_answer\": \"str\", "
            "\"explanation\": \"str\"}. "
            "IMPORTANT: Use Unicode/plain text for all math (e.g., use 'x²', 'π', '√') "
            "and absolutely NO LaTeX or HTML math tags."
        )
        
        response = model.generate_content([prompt, myfile])
        json_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_text)
        
        # Upload diagram to Cloudinary if detected
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
    # Use dynamic port from Render
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
          
