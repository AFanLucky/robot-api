from typing import Union

from fastapi import FastAPI, Body, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import module.match as match

app = FastAPI()

class ImageData(BaseModel):
    bg_img: str
    slide_img: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/robot-analysis")
async def handle_analysis(
    bg_img: str = Form(...),
    slide_img: str = Form(...)
):
    try:
        # 这里处理两个字符串
        print(f"接收到bg_img: {bg_img}")
        print(f"接收到slide_img: {slide_img}")

        best_x=match.handle_calculate(bg_img, slide_img)

        return {
            "code": 0,
            "data": {
                "best_x": best_x,
            },
            "message": "处理成功"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"错误: {str(e)}"}
        )