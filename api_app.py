from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from predictVer2 import get_number
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello"}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pred = get_number(contents)
        return JSONResponse(content={"prediction": int(pred[0])})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
