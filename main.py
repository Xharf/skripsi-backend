from typing import Union
from fastapi import FastAPI, File, UploadFile, Response
from pydantic import BaseModel
from model import Model
import cv2
from keras.utils import img_to_array
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as Image
import base64
import io
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Base64img(BaseModel):
    imgbase64: str


model = Model()


@app.post("/predict", responses={200: {"content": {"image/jpeg": {}}}}, response_class=Response)
def getPredictionImage(item: UploadFile):
    try:
        if item.filename:
            img = item.file.read()
            # print(img)
            npimg = np.fromstring(img, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            result = model.predict(img)
            return Response(content=result, media_type="image/jpeg")
        else:
            return {
                "code": 400,
                "message": "Gambar tidak boleh kosong",
            }
    except Exception as e:
        return {
            "code": 500,
            "message": str(e),
        }


# @app.post("/predict64", responses={200: {"content": {"image/jpeg": {}}}}, response_class=Response)
@app.post("/predict64")
def getPredictionImage2(item: Base64img):
    try:
        if item.imgbase64:
            image_b64 = item.imgbase64.split(",")[1]
            binary = base64.b64decode(image_b64)
            npimg = np.fromstring(binary, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            # print(img)
            result = model.predict(img)
            # return Response(content=result, media_type="image/jpeg")
            return {
                "code": 200,
                "message": "Success",
                "data": result
            }
        else:
            return {
                "code": 400,
                "message": "Gambar tidak boleh kosong",
            }
    except Exception as e:
        return {
            "code": 500,
            "message": str(e),
        }


@app.post("/*")
def notFound():
    return {
        "code": 404,
        "message": "Endpoint tidak ditemukan",
    }
