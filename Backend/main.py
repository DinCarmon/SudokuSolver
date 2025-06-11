from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from PIL import Image
import shutil
from starlette.status import HTTP_400_BAD_REQUEST
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

"""
## CORS Issues and Fix
When running the frontend and backend on different ports (like React on `localhost:5173` and FastAPI on `localhost:8000`), the browser blocks requests by default due to CORS (Cross-Origin Resource Sharing) rules.
To fix this in FastAPI, add the CORS middleware to allow requests from your frontend:
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = './user_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Invalid image type")

    file_location = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # open with PIL to check it's valid
    try:
        img = Image.open(file_location)
        img.verify()
    except Exception:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Corrupted image")

    return JSONResponse(content={"message": "Image uploaded successfully", "filename": image.filename})