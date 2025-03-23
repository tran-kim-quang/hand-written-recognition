from predictVer2 import get_number
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
# Định nghĩa model cho dữ liệu đầu vào
class ChatInput(BaseModel):
    message: str
@app.get("/")
async def root():
    return {"message":"Hello"}
@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pred = get_number(contents)
        return JSONResponse(content={"prediction": int(pred[0])})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})