import os
import random
import json
import cloudinary.uploader
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# Cloudinary Configuration
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

# API Key Loading - Optimized for 5 keys
API_KEYS = []
print("--- LOADING API KEYS ---")
for i in range(1, 6):  # Now only looks for 1 through 5
    key_name = f"GEMINI_API_KEY_{i}"
    val = os.environ.get(key_name)
    if val:
        API_KEYS.append(val.strip())
        print(f"Successfully loaded: {key_name}")
    else:
        print(f"Variable not found: {key_name}")

print(f"Total keys ready: {len(API_KEYS)}")

def get_model():
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="No Gemini API keys found.")
    
    api_key = random.choice(API_KEYS).strip()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        myfile = genai.upload_file(temp_filename)
        model = get_model()
        
        # Enhanced Prompt for Unicode/Plaintext Math
        prompt = (
            "Analyze this exam question. Return JSON only: "
            "{\"has_diagram\": bool, \"year\": \"str\", \"question\": \"str\", "
            "\"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct_answer\": \"str\", "
            "\"explanation\": \"str\"}. "
            "IMPORTANT: Use Unicode/plain text for all math symbols (e.g., 'x²', 'π', '√'). "
            "ABSOLUTELY NO LaTeX, TeX, or HTML math tags."
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
          
