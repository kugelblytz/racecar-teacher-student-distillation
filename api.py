import datetime
import time

import uvicorn
from fastapi import Body, FastAPI

from controller import FullThrottleLaneController, PredictState
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI()
start_time = time.time()
controller = FullThrottleLaneController()


@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    actions = controller.predict(PredictState(**payload))
    return RaceCarPredictResponseDto(actions=actions)


@app.get("/api")
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def index():
    return {
        "status": "ok",
        "message": "Race-car controller is running",
        "predict": "/predict",
        "docs": "/docs",
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
