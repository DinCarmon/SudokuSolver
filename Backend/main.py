from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
import numpy as np
import base64
import sys
import os
from PIL import Image
import shutil
from starlette.status import HTTP_400_BAD_REQUEST
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import cv2
import json

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Loader.SudokuImageExtractor.sudoku_extractor import extract_soduko_from_image
from Loader.SudokuImageExtractor.sudoku_extractor import get_image_wrap

from Solver.solver import solve_sudoku_automatic
from Solver.solver import Board
from Solver.solver import update_cell_notation
from Solver.solver import cli_print_board
from Solver.solver import safe_replace

app = FastAPI()

"""
## CORS Issues and Fix
When running the frontend and backend on different ports (like React on `localhost:5173` and FastAPI on `localhost:8000`), the browser blocks requests by default due to CORS (Cross-Origin Resource Sharing) rules.
To fix this in FastAPI, add the CORS middleware to allow requests from your frontend:
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = './user_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Add this line before any routes that use session
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

@app.post("/upload-image")
async def upload_image(request: Request, image: UploadFile = File(...)):
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

    img = cv2.imread(file_location)

    img_wrap = get_image_wrap(img)
    wrapped_filename = os.path.splitext(file_location)[0] + '_wrapped' + os.path.splitext(file_location)[1]
    cv2.imwrite(wrapped_filename, img_wrap)

    board = extract_soduko_from_image(img)

    board_inst = Board(board)
    print(board_inst)
    request.session['board_inst'] = json.dumps(board_inst.to_dict())  

    # Read and encode the original image to base64
    with open(file_location, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
    with open(wrapped_filename, "rb") as img_file:
        encoded_wrapped_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Determine the correct MIME type
    ext = image.filename.lower().split('.')[-1]
    if ext == "png":
        mime_type = "image/png"
    elif ext in ("jpg", "jpeg"):
        mime_type = "image/jpeg"
    else:
        mime_type = "application/octet-stream"  # fallback
    data_uri_original_image = f"data:{mime_type};base64,{encoded_image}"
    data_uri_wrapped_image = f"data:{mime_type};base64,{encoded_wrapped_image}"


    return JSONResponse(content={
        "message": "Image uploaded successfully",
        "filename": image.filename,
        "board": board.tolist(),
        "original_image": data_uri_original_image,
        "wrapped_image": data_uri_wrapped_image,
        "cell_notation": board_inst.cell_notation
    })

@app.options("/solve-sudoku")
async def cors_preflight():
    return {"preflight": "ok"}

@app.post("/solve-sudoku")
async def solve_sudoku_route(
    request: Request,
    payload: dict = Body(...)
):
    board = payload.get("board")
    image = payload.get("image")
    image_wrap = payload.get("image_wrap")

    if board is None or image is None or image_wrap is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Missing data")

    original_board = np.array(board)
    solved_board = original_board.copy()
    success = solve_sudoku_automatic(solved_board)

    if success:
        original_board = solved_board
        print(solved_board)
        print("success")
    else:
        print("Unsolvable")

    if not success:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Board is unsolvable")

    return {
        "solvable": success,
        "board": original_board.tolist(),
        "solved_board": solved_board.tolist(),
        "original_image": image,
        "wrapped_image": image_wrap,
    }

@app.post("/update-cell-notation")
async def update_board(
    request: Request,
    payload: dict = Body(...)
):
    board_inst = Board.from_dict(json.loads(request.session['board_inst']))
    row = int(payload.get("row"))
    col = int(payload.get("col"))
    digit = int(payload.get("digit"))
    safe_replace(board_inst, digit, row, col)
    request.session['board_inst'] = json.dumps(board_inst.to_dict())
    #cli_print_board(board_inst, print_cell_notation=True)
    return {
        "cell_notation": board_inst.cell_notation
    }

@app.post("/get-cell-notation")
async def get_cell_notation(
    request: Request,
):
    board_inst = Board.from_dict(json.loads(request.session['board_inst']))
    return {
        "cell_notation": board_inst.cell_notation
    }

