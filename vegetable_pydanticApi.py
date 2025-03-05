from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
from PIL import Image
import numpy as np
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, Field

model = joblib.load("veggies_predictor.h5")

# Define class labels
class_map = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli',
             5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber',
             10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}

# Initialize FastAPI app
app = FastAPI()

# image upload with validator pydantic class
class ImageUploadRequest(BaseModel):
  filename: str = Field(...)

  # @field_validator('filename')
  # def image_extension_validator(cls, value):
  #   allowed_extensions = (".jpg", ".png")
  #   if not value.lower().endswith(allowed_extensions):
  #     raise ValueError('Only JPG and png images are allowed')
  #   return value

# prediction response pydantic class
class ResponsePrediction(BaseModel):
  prediction: str

# post request for prediction
@app.post('/predict', response_model=ResponsePrediction)
async def predict(file: UploadFile):
  # try:
    request_data = ImageUploadRequest(filename=file.filename)

    image = Image.open(file.file)
    image = image.resize((224,224))

    image = np.array(image)

    if image is None:
      return JSONResponse(content={'error':'Invalid Image'}, status_code=400)

    image = image.reshape(1,224,224,3)

    prediction = np.argmax(model.predict(image))
    predicted_class = class_map[prediction]
    print(f'Prediction is {predicted_class}')

    return ResponsePrediction(prediction=predicted_class)
  
  # except ValueError as ve:
  #   raise HTTPException(status_code=400, detail=str(ve))

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, port=5000)

