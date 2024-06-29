import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
from functions import train_vgg16_model, predict_image

app = FastAPI(
    title="Melanoma Detection API",
    description="API to train a model to detect melanoma and predict images using the trained model. It also includes a chatbot using OpenAI's GPT-3.5",
    version="1.0.0",
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

openai.api_key = "sk-proj-aF0mmTn9ZTjRbfDTe5SoT3BlbkFJ7uGdIu3eFhJ09vRkXTng"

class TrainingData(BaseModel):
    filenames: list
    labels: list

@app.post("/training", tags=["Model Training"])
async def train_model(training_data: TrainingData):
    if len(training_data.filenames) != len(training_data.labels):
        raise HTTPException(status_code=400, detail="Filenames and labels must have the same length")

    path = 'training/'
    missing_files = [f for f in training_data.filenames if not os.path.exists(os.path.join(path, f))]
    if missing_files:
        raise HTTPException(status_code=400, detail=f"Some images are missing in the specified directory: {missing_files}")

    train_vgg16_model(path, training_data.filenames, training_data.labels)

    return JSONResponse(content={"message": "Model trained successfully"})

@app.post("/predict", tags=["Model Prediction"])
async def predict(file: UploadFile = File(...)):
    model_path = "model/model.h5"

    image_bytes = await file.read()
    result = predict_image(model_path, image_bytes)

    return JSONResponse(content={"prediction": result})

@app.post("/model", tags=["Model Info"])
async def get_model_response(chat_request: ChatRequest):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        )
        return JSONResponse(content={"response": response.choices[0].message['content']})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from OpenAI: {e}")

@app.exception_handler(Exception)
async def validation_exception_handler(request, err):
    return JSONResponse(
        status_code=400,
        content={"message": f"An error occurred: {err}"}
    )
    