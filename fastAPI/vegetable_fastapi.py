# from fastapi import FastAPI, File, UploadFile
# import cv2
# import numpy as np
# import joblib
# from fastapi.responses import JSONResponse

# # Load the trained model correctly
# model = joblib.load("veggies_predictor.h5")

# # Define class labels
# class_map = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli',
#              5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber',
#              10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}

# # Initialize FastAPI app
# app = FastAPI()

# # Define a route for prediction
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     print("Predict called")
    
#     # Read image bytes
#     contents = await file.read()

#     # Convert image bytes to NumPy array
#     image = np.frombuffer(contents, np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#     if image is None:
#         return JSONResponse(content={"error": "Invalid image"}, status_code=400)

#     # Preprocess image
#     image = cv2.resize(image, (224, 224))
#     image = image.reshape(1, 224, 224, 3)

#     # Make prediction
#     prediction = np.argmax(model.predict(image))
#     predicted_class = class_map[prediction]
#     print("Predicted class:", predicted_class)

#     return {"prediction": predicted_class}

# # Run the FastAPI app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app)

from fastapi import FastAPI, File, UploadFile
import numpy as np
import joblib
from fastapi.responses import JSONResponse
from PIL import Image

# Load the trained model correctly
model = joblib.load("veggies_predictor.h5")

# Define class labels
class_map = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli',
             5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber',
             10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}

# Initialize FastAPI app
app = FastAPI()

# Define a route for prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("Predict called")
    
    # Open image directly using PIL
    image = Image.open(file.file)  # No need to convert bytes manually!

    # Convert image to NumPy array
    image = image.resize((224, 224))  # Resize directly using PIL
    image = np.array(image)  # Convert to NumPy array

    if image is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    # Reshape for model input
    image = image.reshape(1, 224, 224, 3)

    # Make prediction
    prediction = np.argmax(model.predict(image))
    predicted_class = class_map[prediction]
    print("Predicted class:", predicted_class)

    return {"prediction": predicted_class}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

